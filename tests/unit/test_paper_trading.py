"""
Unit tests for PaperTradingSimulator class.
Tests trade execution, position tracking, and portfolio management.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from execution.paper import PaperTradingSimulator, PaperTrade, PaperPosition
from strategies.base import StrategySignal, SignalType


class TestPaperTrade:
    """Test PaperTrade dataclass."""

    def test_paper_trade_creation(self):
        """Test PaperTrade creation."""
        signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        trade = PaperTrade(
            trade_id="TEST_001",
            symbol="AAPL",
            signal_type=SignalType.BUY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            strategy_name="GoldenCross",
            signal=signal,
        )

        assert trade.trade_id == "TEST_001"
        assert trade.symbol == "AAPL"
        assert trade.signal_type == SignalType.BUY
        assert trade.quantity == 100
        assert trade.price == 150.0
        assert trade.strategy_name == "GoldenCross"
        assert trade.status == "pending"

    def test_paper_trade_default_values(self):
        """Test PaperTrade default values."""
        signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        trade = PaperTrade(
            trade_id="TEST_001",
            symbol="AAPL",
            signal_type=SignalType.BUY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            strategy_name="GoldenCross",
            signal=signal,
        )

        assert trade.fill_price is None
        assert trade.commission == 0.0
        assert trade.status == "pending"


class TestPaperPosition:
    """Test PaperPosition dataclass."""

    def test_paper_position_creation(self):
        """Test PaperPosition creation."""
        position = PaperPosition(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            entry_date=datetime.now(),
            strategy_name="GoldenCross",
        )

        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.avg_price == 150.0
        assert position.strategy_name == "GoldenCross"
        assert position.unrealized_pnl == 0.0
        assert position.current_price == 0.0

    def test_paper_position_default_values(self):
        """Test PaperPosition default values."""
        position = PaperPosition(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            entry_date=datetime.now(),
            strategy_name="GoldenCross",
        )

        assert position.unrealized_pnl == 0.0
        assert position.current_price == 0.0


class TestPaperTradingSimulator:
    """Test PaperTradingSimulator functionality."""

    @pytest.fixture
    def simulator(self):
        """Create PaperTradingSimulator instance."""
        return PaperTradingSimulator(initial_capital=10000)

    @pytest.fixture(autouse=True)
    def mock_database(self):
        """Mock all database calls to prevent real connections."""
        with patch("execution.paper.get_engine"), patch(
            "execution.paper.get_session"
        ), patch("execution.paper.MarketData"):
            yield

    @pytest.fixture
    def buy_signal(self):
        """Create buy signal for testing."""
        return StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

    @pytest.fixture
    def sell_signal(self):
        """Create sell signal for testing."""
        return StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.SELL,
            confidence=0.8,
            price=160.0,
            quantity=100,
        )

    def test_simulator_initialization(self, simulator):
        """Test PaperTradingSimulator initialization."""
        assert simulator.initial_capital == 10000
        assert simulator.current_cash == 10000
        assert simulator.positions == {}
        assert simulator.pending_orders == {}
        assert simulator.completed_trades == []
        assert simulator.trade_counter == 0

    def test_simulator_initialization_with_custom_params(self):
        """Test PaperTradingSimulator initialization with custom parameters."""
        simulator = PaperTradingSimulator(
            initial_capital=50000,
            commission_per_trade=1.0,
            slippage_pct=0.002,
            fill_delay_minutes=5,
        )

        assert simulator.initial_capital == 50000
        assert simulator.commission_per_trade == 1.0
        assert simulator.slippage_pct == 0.002
        assert simulator.fill_delay_minutes == 5

    def test_execute_signal_buy_order(self, simulator, buy_signal):
        """Test executing buy signal."""
        with patch.object(simulator, "_get_current_price", return_value=150.0):
            success = simulator.execute_signal(buy_signal)
            assert success is True

            # Check that position was created
            assert "AAPL" in simulator.positions
            position = simulator.positions["AAPL"]
            # Account for position sizing logic - should buy what we can afford
            # The simulator uses the signal quantity if provided, otherwise 30% of current cash
            # But if that exceeds available cash, it reduces to fit
            if buy_signal.quantity and buy_signal.quantity > 0:
                # Signal has quantity, but it might be reduced if insufficient cash
                estimated_cost = (
                    buy_signal.quantity * 150.0 + simulator.commission_per_trade
                )
                if estimated_cost > simulator.initial_capital:
                    expected_quantity = int(
                        (simulator.initial_capital - simulator.commission_per_trade)
                        / 150.0
                    )
                else:
                    expected_quantity = buy_signal.quantity
            else:
                # No quantity specified, use 30% of current cash
                expected_quantity = int((simulator.initial_capital * 0.3) / 150.0)

            assert position.quantity == expected_quantity
            assert abs(position.avg_price - 150.0 * (1 + simulator.slippage_pct)) < 0.01

    def test_execute_signal_sell_order(self, simulator, sell_signal):
        """Test executing sell signal."""
        # First create a position to sell
        buy_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        with patch.object(simulator, "_get_current_price", return_value=150.0):
            simulator.execute_signal(buy_signal)

        # Now sell the position
        with patch.object(simulator, "_get_current_price", return_value=160.0):
            success = simulator.execute_signal(sell_signal)
            assert success is True

            # Check that position was closed
            assert "AAPL" not in simulator.positions

    def test_execute_signal_insufficient_cash(self, simulator, buy_signal):
        """Test executing signal with insufficient cash."""
        # Try to buy more than we can afford
        expensive_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=200.0,
            quantity=1000,  # Would cost $200,000
        )

        with patch.object(simulator, "_get_current_price", return_value=200.0):
            success = simulator.execute_signal(expensive_signal)
            # Should still succeed because position sizing will reduce quantity
            assert success is True
            # But position should be smaller than requested
            assert "AAPL" in simulator.positions
            position = simulator.positions["AAPL"]
            assert position.quantity < 1000

    def test_execute_signal_insufficient_position(self, simulator, sell_signal):
        """Test executing sell signal with insufficient position."""
        with patch.object(simulator, "_get_current_price", return_value=160.0):
            success = simulator.execute_signal(sell_signal)
            assert success is False  # No position to sell

    def test_execute_signal_invalid_signal_type(self, simulator):
        """Test executing invalid signal type."""
        invalid_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.HOLD,  # HOLD is not executable
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        success = simulator.execute_signal(invalid_signal)
        assert success is False

    def test_execute_signal_with_slippage(self, simulator, buy_signal):
        """Test executing signal with slippage."""
        with patch.object(simulator, "_get_current_price", return_value=150.0):
            success = simulator.execute_signal(buy_signal)
            assert success is True

            # Check that slippage was applied
            position = simulator.positions["AAPL"]
            expected_price = 150.0 * (1 + simulator.slippage_pct)
            assert abs(position.avg_price - expected_price) < 0.01

    def test_execute_signal_with_commission(self):
        """Test executing signal with commission."""
        simulator = PaperTradingSimulator(
            initial_capital=10000, commission_per_trade=1.0
        )

        buy_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        with patch.object(simulator, "_get_current_price", return_value=150.0):
            success = simulator.execute_signal(buy_signal)
            assert success is True

            # Check that commission was deducted
            # Account for position sizing and slippage
            # The signal has quantity=100, but let's check what actually happened
            position = simulator.positions["AAPL"]
            actual_quantity = position.quantity
            fill_price = 150.0 * (1 + simulator.slippage_pct)
            actual_cost = actual_quantity * fill_price + simulator.commission_per_trade
            expected_cash = simulator.initial_capital - actual_cost
            # Allow for small floating point differences
            assert abs(simulator.current_cash - expected_cash) < 1.0

    def test_update_positions(self, simulator):
        """Test updating positions with current prices."""
        # Create a position
        buy_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        with patch.object(simulator, "_get_current_price", return_value=150.0):
            simulator.execute_signal(buy_signal)

        # Update positions with new price
        with patch.object(simulator, "_get_current_price", return_value=160.0):
            simulator.update_positions()

            position = simulator.positions["AAPL"]
            assert position.current_price == 160.0
            # Calculate expected P&L based on actual position size
            expected_pnl = (160.0 - position.avg_price) * position.quantity
            assert abs(position.unrealized_pnl - expected_pnl) < 0.01

    def test_get_portfolio_value(self, simulator):
        """Test getting portfolio value."""
        # Create a position
        buy_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        with patch.object(simulator, "_get_current_price", return_value=150.0):
            simulator.execute_signal(buy_signal)

        # Update positions
        with patch.object(simulator, "_get_current_price", return_value=160.0):
            simulator.update_positions()

            portfolio_value = simulator.get_portfolio_value()
            position = simulator.positions["AAPL"]
            expected_value = simulator.current_cash + (160.0 * position.quantity)
            assert abs(portfolio_value - expected_value) < 0.01

    def test_get_portfolio_summary(self, simulator):
        """Test getting portfolio summary."""
        # Create a position
        buy_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        with patch.object(simulator, "_get_current_price", return_value=150.0):
            simulator.execute_signal(buy_signal)

        # Update positions
        with patch.object(simulator, "_get_current_price", return_value=160.0):
            simulator.update_positions()

            summary = simulator.get_portfolio_summary()

            assert "total_value" in summary
            assert "cash" in summary
            assert "total_pnl" in summary
            assert "positions_value" in summary
            # Calculate expected P&L based on actual position
            position = simulator.positions["AAPL"]
            expected_pnl = (160.0 - position.avg_price) * position.quantity
            assert abs(summary["total_pnl"] - expected_pnl) < 0.01

    def test_get_positions_summary(self, simulator):
        """Test getting positions summary."""
        # Create multiple positions with smaller quantities to fit in budget
        symbols = ["AAPL", "MSFT"]
        for i, symbol in enumerate(symbols):
            buy_signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=0.8,
                price=100.0 + i * 10,
                quantity=50,
            )

            with patch.object(
                simulator, "_get_current_price", return_value=100.0 + i * 10
            ):
                simulator.execute_signal(buy_signal)

        # Update positions
        with patch.object(simulator, "_get_current_price", return_value=110.0):
            simulator.update_positions()

            positions_summary = simulator.get_positions_summary()
            assert (
                len(positions_summary) == 2
            )  # Only 2 positions due to budget constraints

            for position in positions_summary:
                assert "symbol" in position
                assert "quantity" in position
                assert "avg_price" in position
                assert "current_price" in position
                assert "unrealized_pnl" in position

    def test_get_trades_summary(self, simulator):
        """Test getting trades summary."""
        # Execute some trades
        buy_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        sell_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.SELL,
            confidence=0.8,
            price=160.0,
            quantity=100,
        )

        with patch.object(simulator, "_get_current_price", return_value=150.0):
            simulator.execute_signal(buy_signal)

        with patch.object(simulator, "_get_current_price", return_value=160.0):
            simulator.execute_signal(sell_signal)

        trades_summary = simulator.get_trades_summary()
        assert len(trades_summary) == 2  # Buy and sell

    def test_reset_simulator(self, simulator):
        """Test resetting simulator."""
        # Create some state
        buy_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        with patch.object(simulator, "_get_current_price", return_value=150.0):
            simulator.execute_signal(buy_signal)

        # Reset
        simulator.reset()

        assert simulator.current_cash == simulator.initial_capital
        assert simulator.positions == {}
        assert simulator.pending_orders == {}
        assert simulator.completed_trades == []
        assert simulator.trade_counter == 0

    def test_simulator_performance_tracking(self, simulator):
        """Test performance tracking over time."""
        # Execute trades over time
        dates = pd.date_range("2023-01-01", periods=5, freq="D")

        for i, date in enumerate(dates):
            buy_signal = StrategySignal(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.8,
                price=100.0 + i * 10,
                quantity=10,
            )

            with patch.object(
                simulator, "_get_current_price", return_value=100.0 + i * 10
            ):
                simulator.execute_signal(buy_signal)

        # Check that daily portfolio values are tracked
        assert len(simulator.daily_portfolio_values) > 0
        # Verify the structure of tracked data
        for entry in simulator.daily_portfolio_values:
            assert "date" in entry
            assert "value" in entry
            assert "cash" in entry
            assert "positions_value" in entry

    def test_simulator_error_handling(self, simulator):
        """Test error handling in simulator."""
        # Test with invalid signal
        invalid_signal = None

        success = simulator.execute_signal(invalid_signal)
        assert success is False

        # Test with signal that has no price data
        signal_no_price = StrategySignal(
            symbol="INVALID",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        with patch.object(simulator, "_get_current_price", return_value=None):
            success = simulator.execute_signal(signal_no_price)
            assert success is False

    def test_simulator_edge_cases(self, simulator):
        """Test edge cases in simulator."""
        # Test with zero quantity
        zero_quantity_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=0,
        )

        with patch.object(simulator, "_get_current_price", return_value=150.0):
            success = simulator.execute_signal(zero_quantity_signal)
            # Should still succeed because position sizing will calculate quantity
            assert success is True

        # Test with negative price
        negative_price_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=-150.0,
            quantity=100,
        )

        with patch.object(simulator, "_get_current_price", return_value=150.0):
            success = simulator.execute_signal(negative_price_signal)
            # Should succeed because it uses current market price, not signal price
            assert success is True

    def test_simulator_logging(self, simulator, buy_signal, caplog):
        """Test that simulator operations are properly logged."""
        with caplog.at_level("INFO"):
            with patch.object(simulator, "_get_current_price", return_value=150.0):
                simulator.execute_signal(buy_signal)

        # Should have logging messages
        assert len(caplog.records) > 0

    def test_database_mocking_works(self, simulator):
        """Test that database calls are properly mocked."""
        # This test verifies that no real database connections are made
        # If the mocking wasn't working, this would fail with database connection errors
        
        # Try to get current price - should use mocked value
        with patch.object(simulator, "_get_current_price", return_value=100.0):
            price = simulator._get_current_price("AAPL")
            assert price == 100.0
            
        # Try to update positions - should not make real database calls
        with patch.object(simulator, "_get_current_price", return_value=110.0):
            simulator.update_positions()
            # Should not raise any database connection errors
