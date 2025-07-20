"""
Paper trading simulation for testing strategies without real money.
Provides realistic simulation of trade execution and position tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from strategies.base import StrategySignal, SignalType
from data import get_engine, get_session, MarketData

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Represents a simulated trade."""

    trade_id: str
    symbol: str
    signal_type: SignalType
    quantity: int
    price: float
    timestamp: datetime
    strategy_name: str
    signal: StrategySignal
    fill_price: Optional[float] = None
    commission: float = 0.0
    status: str = "pending"  # pending, filled, cancelled


@dataclass
class PaperPosition:
    """Represents a simulated position."""

    symbol: str
    quantity: int
    avg_price: float
    entry_date: datetime
    strategy_name: str
    unrealized_pnl: float = 0.0
    current_price: float = 0.0


class PaperTradingSimulator:
    """
    Paper trading simulator for testing strategies.

    Features:
    - Realistic order execution simulation
    - Position tracking and P&L calculation
    - Portfolio management and risk controls
    - Performance tracking
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        commission_per_trade: float = 0.0,  # Alpaca is commission-free
        slippage_pct: float = 0.001,  # 0.1% slippage
        fill_delay_minutes: int = 1,  # Simulated execution delay
    ):
        """
        Initialize paper trading simulator.

        Args:
            initial_capital: Starting cash amount
            commission_per_trade: Commission per trade
            slippage_pct: Slippage percentage for market orders
            fill_delay_minutes: Minutes delay for order fills
        """
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct
        self.fill_delay_minutes = fill_delay_minutes

        # Trading state
        self.current_cash = initial_capital
        self.positions = {}  # symbol -> PaperPosition
        self.pending_orders = {}  # order_id -> PaperTrade
        self.completed_trades = []
        self.trade_counter = 0

        # Performance tracking
        self.daily_portfolio_values = []
        self.start_date = datetime.now()

        logger.info(f"Paper trading simulator initialized with ${initial_capital:,.2f}")

    def execute_signal(self, signal: StrategySignal) -> bool:
        """
        Execute a trading signal in the paper trading environment.

        Args:
            signal: The trading signal to execute

        Returns:
            True if order was placed successfully, False otherwise
        """
        try:
            # Get current market price
            current_price = self._get_current_price(signal.symbol)
            if current_price is None:
                logger.warning(f"No current price available for {signal.symbol}")
                return False

            # Calculate order details
            if signal.signal_type == SignalType.BUY:
                return self._execute_buy_order(signal, current_price)
            elif signal.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]:
                return self._execute_sell_order(signal, current_price)
            else:
                logger.warning(f"Unsupported signal type: {signal.signal_type}")
                return False

        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {str(e)}")
            return False

    def _execute_buy_order(self, signal: StrategySignal, current_price: float) -> bool:
        """Execute a buy order."""
        symbol = signal.symbol

        # Calculate position size
        if signal.quantity:
            quantity = signal.quantity
        else:
            # Default position sizing (30% of available cash for ETFs)
            available_cash = self.current_cash * 0.3
            quantity = int(available_cash / current_price)

        if quantity <= 0:
            logger.warning(f"Cannot buy {symbol}: calculated quantity is {quantity}")
            return False

        # Check if we have enough cash
        estimated_cost = quantity * current_price + self.commission_per_trade
        if estimated_cost > self.current_cash:
            # Reduce quantity to fit available cash
            quantity = int(
                (self.current_cash - self.commission_per_trade) / current_price
            )
            if quantity <= 0:
                logger.warning(f"Insufficient cash to buy {symbol}")
                return False

        # Apply slippage for market orders
        fill_price = current_price * (1 + self.slippage_pct)
        actual_cost = quantity * fill_price + self.commission_per_trade

        # Create trade record
        trade = PaperTrade(
            trade_id=f"BUY_{symbol}_{self.trade_counter}",
            symbol=symbol,
            signal_type=SignalType.BUY,
            quantity=quantity,
            price=current_price,
            timestamp=datetime.now(),
            strategy_name=signal.strategy_name,
            signal=signal,
            fill_price=fill_price,
            commission=self.commission_per_trade,
            status="filled",
        )

        # Update cash
        self.current_cash -= actual_cost

        # Update position
        if symbol in self.positions:
            # Add to existing position
            existing_pos = self.positions[symbol]
            total_quantity = existing_pos.quantity + quantity
            new_avg_price = (
                (existing_pos.quantity * existing_pos.avg_price)
                + (quantity * fill_price)
            ) / total_quantity

            existing_pos.quantity = total_quantity
            existing_pos.avg_price = new_avg_price
        else:
            # Create new position
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                quantity=quantity,
                avg_price=fill_price,
                entry_date=datetime.now(),
                strategy_name=signal.strategy_name,
            )

        # Record trade
        self.completed_trades.append(trade)
        self.trade_counter += 1

        logger.info(
            f"PAPER BUY: {quantity} {symbol} @ ${fill_price:.2f} "
            f"(cost: ${actual_cost:.2f})"
        )
        return True

    def _execute_sell_order(self, signal: StrategySignal, current_price: float) -> bool:
        """Execute a sell order."""
        symbol = signal.symbol

        if symbol not in self.positions:
            logger.warning(f"Cannot sell {symbol}: no position exists")
            return False

        position = self.positions[symbol]

        # Determine quantity to sell
        if signal.quantity:
            quantity_to_sell = min(signal.quantity, position.quantity)
        else:
            quantity_to_sell = position.quantity  # Close entire position

        if quantity_to_sell <= 0:
            logger.warning(f"Cannot sell {symbol}: no shares to sell")
            return False

        # Apply slippage for market orders
        fill_price = current_price * (1 - self.slippage_pct)

        # Calculate proceeds
        gross_proceeds = quantity_to_sell * fill_price
        net_proceeds = gross_proceeds - self.commission_per_trade

        # Calculate P&L
        cost_basis = quantity_to_sell * position.avg_price
        realized_pnl = gross_proceeds - cost_basis - self.commission_per_trade

        # Create trade record
        trade = PaperTrade(
            trade_id=f"SELL_{symbol}_{self.trade_counter}",
            symbol=symbol,
            signal_type=SignalType.SELL,
            quantity=quantity_to_sell,
            price=current_price,
            timestamp=datetime.now(),
            strategy_name=signal.strategy_name,
            signal=signal,
            fill_price=fill_price,
            commission=self.commission_per_trade,
            status="filled",
        )

        # Update cash
        self.current_cash += net_proceeds

        # Update position
        position.quantity -= quantity_to_sell
        if position.quantity <= 0:
            del self.positions[symbol]

        # Record trade
        self.completed_trades.append(trade)
        self.trade_counter += 1

        logger.info(
            f"PAPER SELL: {quantity_to_sell} {symbol} @ ${fill_price:.2f} "
            f"(proceeds: ${net_proceeds:.2f}, P&L: ${realized_pnl:.2f})"
        )
        return True

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        try:
            engine = get_engine()
            session = get_session(engine)

            # Get most recent price
            latest_record = (
                session.query(MarketData)
                .filter(MarketData.symbol == symbol)
                .order_by(MarketData.date.desc())
                .first()
            )

            session.close()

            if latest_record:
                return float(latest_record.close)
            else:
                logger.warning(f"No price data found for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {str(e)}")
            return None

    def update_positions(self):
        """Update position values and unrealized P&L."""
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            if current_price:
                position.current_price = current_price
                position.unrealized_pnl = (
                    current_price - position.avg_price
                ) * position.quantity

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        self.update_positions()

        portfolio_value = self.current_cash
        for position in self.positions.values():
            portfolio_value += position.quantity * position.current_price

        return portfolio_value

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        self.update_positions()

        total_value = self.get_portfolio_value()
        positions_value = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        total_pnl = total_value - self.initial_capital

        return {
            "total_value": total_value,
            "cash": self.current_cash,
            "positions_value": positions_value,
            "initial_capital": self.initial_capital,
            "total_pnl": total_pnl,
            "total_return_pct": (total_pnl / self.initial_capital) * 100,
            "num_positions": len(self.positions),
            "num_trades": len(self.completed_trades),
        }

    def get_positions_summary(self) -> List[Dict[str, Any]]:
        """Get detailed positions summary."""
        self.update_positions()

        positions = []
        for symbol, position in self.positions.items():
            positions.append(
                {
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "avg_price": position.avg_price,
                    "current_price": position.current_price,
                    "market_value": position.quantity * position.current_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "unrealized_pnl_pct": (
                        position.unrealized_pnl
                        / (position.avg_price * position.quantity)
                    )
                    * 100,
                    "entry_date": position.entry_date,
                    "strategy": position.strategy_name,
                }
            )

        return positions

    def get_trades_summary(self) -> List[Dict[str, Any]]:
        """Get trades summary."""
        trades = []
        for trade in self.completed_trades:
            trades.append(
                {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "signal_type": trade.signal_type.value,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "fill_price": trade.fill_price,
                    "timestamp": trade.timestamp,
                    "strategy": trade.strategy_name,
                    "commission": trade.commission,
                    "status": trade.status,
                }
            )

        return trades

    def generate_performance_report(self) -> str:
        """Generate a formatted performance report."""
        summary = self.get_portfolio_summary()
        positions = self.get_positions_summary()

        report = []
        report.append("=" * 50)
        report.append("PAPER TRADING PERFORMANCE REPORT")
        report.append("=" * 50)

        report.append(f"\nPortfolio Summary:")
        report.append(f"  Initial Capital: ${summary['initial_capital']:,.2f}")
        report.append(f"  Current Value: ${summary['total_value']:,.2f}")
        report.append(f"  Cash: ${summary['cash']:,.2f}")
        report.append(f"  Positions Value: ${summary['positions_value']:,.2f}")
        report.append(
            f"  Total P&L: ${summary['total_pnl']:,.2f} ({summary['total_return_pct']:+.2f}%)"
        )

        report.append(f"\nTrading Activity:")
        report.append(f"  Total Trades: {summary['num_trades']}")
        report.append(f"  Open Positions: {summary['num_positions']}")

        if positions:
            report.append(f"\nCurrent Positions:")
            for pos in positions:
                report.append(
                    f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']:.2f} "
                    f"(current: ${pos['current_price']:.2f}, "
                    f"P&L: ${pos['unrealized_pnl']:+.2f})"
                )

        return "\n".join(report)

    def reset(self):
        """Reset the paper trading simulator."""
        self.current_cash = self.initial_capital
        self.positions = {}
        self.pending_orders = {}
        self.completed_trades = []
        self.trade_counter = 0
        self.daily_portfolio_values = []
        self.start_date = datetime.now()

        logger.info("Paper trading simulator reset")
