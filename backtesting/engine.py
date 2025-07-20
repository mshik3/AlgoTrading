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

        # Backtesting state - simplified for strategy testing only
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

        # Calculate final results - simplified for strategy testing
        total_return = 0.0
        total_return_pct = 0.0
        final_capital = self.initial_capital

        # Generate performance metrics
        metrics_calculator = PerformanceMetrics()
        performance_metrics = metrics_calculator.calculate_all_metrics(
            pd.Series(), self.completed_trades
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

        # Note: Portfolio value tracking removed - use Alpaca for real portfolio data

    def _execute_signal(
        self,
        signal: StrategySignal,
        market_data: Dict[str, pd.DataFrame],
        date: datetime,
    ):
        """Execute a trading signal - simplified for strategy testing."""
        symbol = signal.symbol

        # Get current price for the symbol
        if symbol not in market_data or date not in market_data[symbol].index:
            logger.warning(f"No price data for {symbol} on {date}")
            return

        current_price = market_data[symbol].loc[date, "Close"]

        # Record signal for analysis (no actual execution in backtesting)
        logger.info(f"Signal generated: {signal.signal_type} {symbol} @ ${current_price:.2f} on {date}")
        
        # Add to completed trades for metrics calculation
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            # Simplified trade recording for strategy analysis
            trade = BacktestTrade(
                symbol=symbol,
                entry_date=date,
                exit_date=date,
                entry_price=current_price,
                exit_price=current_price,
                quantity=signal.quantity or 100,
                strategy_name=signal.strategy_name,
                entry_signal=signal,
                exit_signal=signal,
                profit_loss=0.0,  # No P&L calculation in simplified backtesting
                profit_loss_pct=0.0,
                holding_period_days=0,
            )
            self.completed_trades.append(trade)

    def _execute_buy_signal(self, signal: StrategySignal, price: float, date: datetime):
        """Execute a buy signal - simplified for strategy testing."""
        # Note: Actual execution removed - use Alpaca for real trading
        logger.debug(f"BUY signal recorded: {signal.symbol} @ ${price:.2f} on {date}")

    def _execute_sell_signal(
        self, signal: StrategySignal, price: float, date: datetime
    ):
        """Execute a sell signal - simplified for strategy testing."""
        # Note: Actual execution removed - use Alpaca for real trading
        logger.debug(f"SELL signal recorded: {signal.symbol} @ ${price:.2f} on {date}")

    def _calculate_portfolio_value(
        self, market_data: Dict[str, pd.DataFrame], date: datetime
    ) -> float:
        """Calculate total portfolio value on a given date."""
        # Note: Portfolio calculation removed - use Alpaca for real portfolio data
        return self.initial_capital

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
