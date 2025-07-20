"""
Backtesting engine for testing trading strategies on historical data.
Provides realistic simulation of strategy execution with position tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from strategies.base import BaseStrategy, StrategySignal, SignalType
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Represents a completed trade in backtesting."""

    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    strategy_name: str
    entry_signal: StrategySignal
    exit_signal: StrategySignal
    profit_loss: float
    profit_loss_pct: float
    holding_period_days: int


@dataclass
class BacktestResult:
    """Contains the complete results of a backtest run."""

    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    trades: List[BacktestTrade]
    daily_portfolio_values: pd.Series
    performance_metrics: Dict[str, Any]
    signals_generated: List[StrategySignal]


class BacktestingEngine:
    """
    Backtesting engine that replays historical market data and executes strategies.

    Features:
    - Realistic trade execution with slippage and fees
    - Position tracking and portfolio management
    - Performance metrics calculation
    - Support for multiple strategies
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        commission_per_trade: float = 0.0,  # Alpaca is commission-free
        slippage_pct: float = 0.001,  # 0.1% slippage
        min_cash_reserve: float = 100,  # Keep minimum cash
    ):
        """
        Initialize the backtesting engine.

        Args:
            initial_capital: Starting capital for backtesting
            commission_per_trade: Commission cost per trade
            slippage_pct: Slippage percentage (0.001 = 0.1%)
            min_cash_reserve: Minimum cash to keep available
        """
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct
        self.min_cash_reserve = min_cash_reserve

        # Backtesting state
        self.current_capital = initial_capital
        self.current_cash = initial_capital
        self.current_positions = {}  # symbol -> {"quantity": int, "avg_price": float}
        self.portfolio_values = []
        self.completed_trades = []
        self.all_signals = []

        logger.info(
            f"Initialized backtesting engine with ${initial_capital:,.2f} capital"
        )

    def run_backtest(
        self,
        strategy: BaseStrategy,
        market_data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """
        Run a complete backtest for a strategy.

        Args:
            strategy: The trading strategy to test
            market_data: Dictionary of symbol -> OHLCV DataFrame
            start_date: Start date for backtesting (optional)
            end_date: End date for backtesting (optional)

        Returns:
            BacktestResult with complete backtest information
        """
        logger.info(f"Starting backtest for {strategy.name}")

        # Reset strategy and engine state
        self._reset_state()
        strategy.reset_positions()

        # Determine date range
        all_dates = set()
        for df in market_data.values():
            if not df.empty:
                all_dates.update(df.index)

        if not all_dates:
            raise ValueError("No market data provided for backtesting")

        sorted_dates = sorted(all_dates)
        backtest_start = start_date if start_date else sorted_dates[0]
        backtest_end = end_date if end_date else sorted_dates[-1]

        # Filter market data to backtest period
        filtered_data = self._filter_data_by_date_range(
            market_data, backtest_start, backtest_end
        )

        # Get all trading dates
        trading_dates = sorted(
            set().union(*[df.index for df in filtered_data.values()])
        )
        logger.info(
            f"Backtesting from {backtest_start} to {backtest_end} ({len(trading_dates)} trading days)"
        )

        # Run day-by-day simulation
        for date in trading_dates:
            try:
                self._simulate_trading_day(strategy, filtered_data, date)
            except Exception as e:
                logger.error(f"Error simulating trading day {date}: {str(e)}")
                continue

        # Calculate final results
        final_capital = self._calculate_portfolio_value(
            filtered_data, trading_dates[-1]
        )
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # Generate performance metrics
        daily_values = pd.Series(
            [pv["total_value"] for pv in self.portfolio_values],
            index=[pv["date"] for pv in self.portfolio_values],
        )

        metrics_calculator = PerformanceMetrics()
        performance_metrics = metrics_calculator.calculate_all_metrics(
            daily_values, self.completed_trades
        )

        result = BacktestResult(
            strategy_name=strategy.name,
            start_date=backtest_start,
            end_date=backtest_end,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            trades=self.completed_trades.copy(),
            daily_portfolio_values=daily_values,
            performance_metrics=performance_metrics,
            signals_generated=self.all_signals.copy(),
        )

        logger.info(
            f"Backtest completed: {total_return_pct:.2f}% return, {len(self.completed_trades)} trades"
        )
        return result

    def _reset_state(self):
        """Reset the backtesting engine state."""
        self.current_capital = self.initial_capital
        self.current_cash = self.initial_capital
        self.current_positions = {}
        self.portfolio_values = []
        self.completed_trades = []
        self.all_signals = []

    def _filter_data_by_date_range(
        self,
        market_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Filter market data to the specified date range."""
        filtered_data = {}
        for symbol, df in market_data.items():
            if df.empty:
                continue

            # Convert datetime if needed
            if isinstance(df.index[0], str):
                df.index = pd.to_datetime(df.index)

            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_df = df.loc[mask]

            if not filtered_df.empty:
                filtered_data[symbol] = filtered_df

        return filtered_data

    def _simulate_trading_day(
        self,
        strategy: BaseStrategy,
        market_data: Dict[str, pd.DataFrame],
        date: datetime,
    ):
        """Simulate trading for a single day."""
        # Get market data up to this date for strategy analysis
        historical_data = {}
        for symbol, df in market_data.items():
            historical_data[symbol] = df[df.index <= date]

        # Generate signals
        try:
            signals = strategy.generate_signals(historical_data)
            self.all_signals.extend(signals)

            # Execute signals
            for signal in signals:
                self._execute_signal(signal, market_data, date)

        except Exception as e:
            logger.error(f"Error generating/executing signals on {date}: {str(e)}")

        # Update portfolio value
        portfolio_value = self._calculate_portfolio_value(market_data, date)
        self.portfolio_values.append(
            {
                "date": date,
                "total_value": portfolio_value,
                "cash": self.current_cash,
                "positions_value": portfolio_value - self.current_cash,
            }
        )

    def _execute_signal(
        self,
        signal: StrategySignal,
        market_data: Dict[str, pd.DataFrame],
        date: datetime,
    ):
        """Execute a trading signal with realistic simulation."""
        symbol = signal.symbol

        # Get current price (use close price for simplicity)
        if symbol not in market_data or date not in market_data[symbol].index:
            logger.warning(f"No price data for {symbol} on {date}")
            return

        current_price = market_data[symbol].loc[date, "Close"]

        # Apply slippage
        if signal.signal_type == SignalType.BUY:
            execution_price = current_price * (1 + self.slippage_pct)
        else:
            execution_price = current_price * (1 - self.slippage_pct)

        if signal.signal_type == SignalType.BUY:
            self._execute_buy_signal(signal, execution_price, date)
        elif signal.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]:
            self._execute_sell_signal(signal, execution_price, date)

    def _execute_buy_signal(self, signal: StrategySignal, price: float, date: datetime):
        """Execute a buy signal."""
        symbol = signal.symbol

        # Calculate position size based on available cash and risk management
        available_cash = self.current_cash - self.min_cash_reserve
        if available_cash <= 0:
            logger.warning(f"Insufficient cash to buy {symbol}")
            return

        # Use signal quantity if provided, otherwise calculate based on portfolio percentage
        if signal.quantity:
            quantity = signal.quantity
        else:
            # Default to using available cash with maximum position size constraint
            max_position_value = self.current_capital * 0.3  # 30% max position
            position_value = min(available_cash, max_position_value)
            quantity = int(position_value / price)

        if quantity <= 0:
            logger.warning(f"Cannot buy {symbol}: quantity={quantity}")
            return

        trade_value = quantity * price + self.commission_per_trade

        if trade_value > available_cash:
            # Reduce quantity to fit available cash
            quantity = int((available_cash - self.commission_per_trade) / price)
            if quantity <= 0:
                logger.warning(f"Insufficient cash to buy even 1 share of {symbol}")
                return
            trade_value = quantity * price + self.commission_per_trade

        # Execute the trade
        self.current_cash -= trade_value

        if symbol in self.current_positions:
            # Add to existing position
            existing_quantity = self.current_positions[symbol]["quantity"]
            existing_avg_price = self.current_positions[symbol]["avg_price"]

            new_quantity = existing_quantity + quantity
            new_avg_price = (
                (existing_quantity * existing_avg_price) + (quantity * price)
            ) / new_quantity

            self.current_positions[symbol] = {
                "quantity": new_quantity,
                "avg_price": new_avg_price,
                "entry_date": self.current_positions[symbol]["entry_date"],
                "entry_signal": self.current_positions[symbol]["entry_signal"],
            }
        else:
            # New position
            self.current_positions[symbol] = {
                "quantity": quantity,
                "avg_price": price,
                "entry_date": date,
                "entry_signal": signal,
            }

        logger.debug(f"BUY: {quantity} shares of {symbol} at ${price:.2f} on {date}")

    def _execute_sell_signal(
        self, signal: StrategySignal, price: float, date: datetime
    ):
        """Execute a sell signal."""
        symbol = signal.symbol

        if symbol not in self.current_positions:
            logger.warning(f"Cannot sell {symbol}: no position exists")
            return

        position = self.current_positions[symbol]
        quantity_to_sell = signal.quantity if signal.quantity else position["quantity"]
        quantity_to_sell = min(quantity_to_sell, position["quantity"])

        if quantity_to_sell <= 0:
            return

        # Execute the trade
        trade_value = quantity_to_sell * price - self.commission_per_trade
        self.current_cash += trade_value

        # Calculate P&L
        profit_loss = (
            price - position["avg_price"]
        ) * quantity_to_sell - self.commission_per_trade
        profit_loss_pct = (
            profit_loss / (position["avg_price"] * quantity_to_sell)
        ) * 100
        holding_period = (date - position["entry_date"]).days

        # Record completed trade
        trade = BacktestTrade(
            symbol=symbol,
            entry_date=position["entry_date"],
            exit_date=date,
            entry_price=position["avg_price"],
            exit_price=price,
            quantity=quantity_to_sell,
            strategy_name=signal.strategy_name,
            entry_signal=position["entry_signal"],
            exit_signal=signal,
            profit_loss=profit_loss,
            profit_loss_pct=profit_loss_pct,
            holding_period_days=holding_period,
        )
        self.completed_trades.append(trade)

        # Update position
        remaining_quantity = position["quantity"] - quantity_to_sell
        if remaining_quantity <= 0:
            del self.current_positions[symbol]
        else:
            self.current_positions[symbol]["quantity"] = remaining_quantity

        logger.debug(
            f"SELL: {quantity_to_sell} shares of {symbol} at ${price:.2f} on {date}, P&L: ${profit_loss:.2f}"
        )

    def _calculate_portfolio_value(
        self, market_data: Dict[str, pd.DataFrame], date: datetime
    ) -> float:
        """Calculate total portfolio value on a given date."""
        total_value = self.current_cash

        for symbol, position in self.current_positions.items():
            if symbol in market_data and date in market_data[symbol].index:
                current_price = market_data[symbol].loc[date, "Close"]
                position_value = position["quantity"] * current_price
                total_value += position_value

        return total_value

    def get_trade_summary(self) -> Dict[str, Any]:
        """Get a summary of all completed trades."""
        if not self.completed_trades:
            return {"total_trades": 0}

        profits = [t.profit_loss for t in self.completed_trades]
        profit_pcts = [t.profit_loss_pct for t in self.completed_trades]
        holding_periods = [t.holding_period_days for t in self.completed_trades]

        winning_trades = [t for t in self.completed_trades if t.profit_loss > 0]
        losing_trades = [t for t in self.completed_trades if t.profit_loss <= 0]

        return {
            "total_trades": len(self.completed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (
                len(winning_trades) / len(self.completed_trades)
                if self.completed_trades
                else 0
            ),
            "total_profit_loss": sum(profits),
            "avg_profit_loss": np.mean(profits),
            "avg_profit_loss_pct": np.mean(profit_pcts),
            "avg_holding_period": np.mean(holding_periods),
            "max_profit": max(profits) if profits else 0,
            "max_loss": min(profits) if profits else 0,
        }
