"""
Unit tests for strategy implementations.
Focused tests for core strategy functionality using modern strategies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from strategies.base import BaseStrategy, StrategySignal, SignalType
from strategies.modern_strategies import ModernGoldenCrossStrategy


class TestBaseStrategy:
    """Test BaseStrategy functionality."""

    def test_base_strategy_initialization(self):
        """Test BaseStrategy initialization."""
        # Test abstract class can't be instantiated
        with pytest.raises(TypeError):
            BaseStrategy("test", ["AAPL"])

    def test_strategy_signal_creation(self):
        """Test StrategySignal creation."""
        signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            quantity=100,
        )

        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.8
        assert signal.price == 150.0
        assert signal.quantity == 100

    def test_strategy_signal_defaults(self):
        """Test StrategySignal default values."""
        signal = StrategySignal(
            symbol="AAPL", signal_type=SignalType.BUY, confidence=0.8
        )

        assert signal.timestamp is not None
        assert signal.metadata == {}


class TestModernGoldenCrossStrategy:
    """Test Modern Golden Cross Strategy functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range(
            "2023-01-01", periods=250, freq="D"
        )  # 250 days for MA calculations
        data = {
            "open": [100.0] * 250,
            "high": [102.0] * 250,
            "low": [99.0] * 250,
            "close": [101.0] * 250,
            "volume": [1000000] * 250,
        }
        return pd.DataFrame(data, index=dates)

    def test_golden_cross_initialization(self):
        """Test Modern Golden Cross Strategy initialization."""
        strategy = ModernGoldenCrossStrategy(symbols=["AAPL", "MSFT"])
        assert strategy.name == "ModernGoldenCross"
        assert strategy.symbols == ["AAPL", "MSFT"]
        assert strategy.config["strategy_type"] == "modern_golden_cross"

    def test_generate_signals_basic(self, sample_data):
        """Test basic signal generation."""
        strategy = ModernGoldenCrossStrategy(symbols=["AAPL"])
        market_data = {"AAPL": sample_data}

        signals = strategy.generate_signals(market_data)
        assert isinstance(signals, list)
        # With constant data, no signals should be generated (no crossovers)
        # The test verifies the method returns a list, not that signals are generated

    def test_generate_signals_empty_data(self):
        """Test signal generation with empty data."""
        strategy = ModernGoldenCrossStrategy(symbols=["AAPL"])
        empty_data = pd.DataFrame()
        market_data = {"AAPL": empty_data}

        signals = strategy.generate_signals(market_data)
        assert isinstance(signals, list)

    def test_generate_signals_missing_symbol(self, sample_data):
        """Test signal generation with missing symbol."""
        strategy = ModernGoldenCrossStrategy(symbols=["AAPL", "MSFT"])
        market_data = {"AAPL": sample_data}  # Missing MSFT

        signals = strategy.generate_signals(market_data)
        assert isinstance(signals, list)

    def test_validate_signal(self):
        """Test signal validation."""
        strategy = ModernGoldenCrossStrategy(symbols=["AAPL"])

        # Valid signal
        valid_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
        )
        assert strategy.validate_signal(valid_signal) is True

        # Invalid signal (low confidence)
        invalid_signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.1,  # Below min_confidence threshold
            price=150.0,
        )
        assert strategy.validate_signal(invalid_signal) is False

        # Test with maximum positions reached
        strategy.positions = {
            "MSFT": {"quantity": 100},
            "GOOGL": {"quantity": 100},
            "TSLA": {"quantity": 100},
            "NVDA": {"quantity": 100},
            "META": {"quantity": 100},
        }  # Max 5 positions

        max_positions_signal = StrategySignal(
            symbol="NFLX",  # New symbol not in positions
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
        )
        assert strategy.validate_signal(max_positions_signal) is False

    def test_calculate_position_size(self):
        """Test position size calculation."""
        strategy = ModernGoldenCrossStrategy(symbols=["AAPL"])
        signal = StrategySignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
        )

        position_size = strategy.calculate_position_size(signal, 10000, 150.0)
        assert isinstance(position_size, int)
        assert position_size > 0

    def test_strategy_config(self):
        """Test strategy configuration."""
        strategy = ModernGoldenCrossStrategy(symbols=["AAPL"])

        assert "fast_ma_period" in strategy.config
        assert "slow_ma_period" in strategy.config
        assert strategy.config["strategy_type"] == "modern_golden_cross"

    def test_strategy_performance_tracking(self, sample_data):
        """Test strategy performance tracking."""
        strategy = ModernGoldenCrossStrategy(symbols=["AAPL"])
        market_data = {"AAPL": sample_data}

        # Generate some signals
        signals = strategy.generate_signals(market_data)

        # Add signals to history
        for signal in signals:
            strategy.add_signal_to_history(signal)

        # Check performance summary
        summary = strategy.get_performance_summary()
        assert isinstance(summary, dict)
        assert "total_signals" in summary

    def test_strategy_reset(self, sample_data):
        """Test strategy reset functionality."""
        strategy = ModernGoldenCrossStrategy(symbols=["AAPL"])
        market_data = {"AAPL": sample_data}

        # Generate signals and add to history
        signals = strategy.generate_signals(market_data)
        for signal in signals:
            strategy.add_signal_to_history(signal)

        # Reset strategy
        strategy.reset_positions()

        # Check that positions are reset
        assert len(strategy.positions) == 0

    def test_strategy_activation(self):
        """Test strategy activation/deactivation."""
        strategy = ModernGoldenCrossStrategy(symbols=["AAPL"])

        # Initially active
        assert strategy.is_active is True

        # Deactivate
        strategy.set_active(False)
        assert strategy.is_active is False

        # Reactivate
        strategy.set_active(True)
        assert strategy.is_active is True
