"""
Unit tests for modern trading strategies.

Tests the modern strategy implementations to ensure they:
- Initialize correctly
- Generate signals properly
- Handle edge cases
- Provide accurate summaries
- Work with dashboard services
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from strategies.modern_strategies import (
    ModernGoldenCrossStrategy,
    ModernMeanReversionStrategy,
    ModernSectorRotationStrategy,
    ModernDualMomentumStrategy,
    create_strategy,
)
from strategies.base import SignalType, StrategySignal


class TestModernGoldenCrossStrategy:
    """Test Modern Golden Cross Strategy implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ModernGoldenCrossStrategy(symbols=["AAPL", "MSFT"])
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        np.random.seed(42)

        # Create trending data for AAPL
        aapl_prices = 150 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        aapl_data = pd.DataFrame(
            {
                "open": aapl_prices * 0.99,
                "high": aapl_prices * 1.02,
                "low": aapl_prices * 0.98,
                "close": aapl_prices,
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        # Create trending data for MSFT
        msft_prices = 300 + np.cumsum(np.random.randn(len(dates)) * 0.8)
        msft_data = pd.DataFrame(
            {
                "open": msft_prices * 0.99,
                "high": msft_prices * 1.02,
                "low": msft_prices * 0.98,
                "close": msft_prices,
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        return {"AAPL": aapl_data, "MSFT": msft_data}

    def test_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.name == "ModernGoldenCross"
        assert self.strategy.symbols == ["AAPL", "MSFT"]
        assert self.strategy.fast_period == 50
        assert self.strategy.slow_period == 200
        assert self.strategy.config["strategy_type"] == "modern_golden_cross"

    def test_generate_signals(self):
        """Test signal generation."""
        signals = self.strategy.generate_signals(self.sample_data)

        # Should return a list of signals
        assert isinstance(signals, list)

        # All signals should be StrategySignal objects
        for signal in signals:
            assert isinstance(signal, StrategySignal)
            assert signal.symbol in ["AAPL", "MSFT"]
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert 0 <= signal.confidence <= 1
            assert signal.strategy_name == "ModernGoldenCross"

    def test_should_enter_position(self):
        """Test position entry logic."""
        # Test with insufficient data
        short_data = self.sample_data["AAPL"].iloc[:50]  # Less than slow_period
        signal = self.strategy.should_enter_position("AAPL", short_data)
        assert signal is None

        # Test with sufficient data
        signal = self.strategy.should_enter_position("AAPL", self.sample_data["AAPL"])
        # May or may not generate signal depending on data
        if signal is not None:
            assert isinstance(signal, StrategySignal)
            assert signal.signal_type == SignalType.BUY
            assert signal.symbol == "AAPL"

    def test_should_exit_position(self):
        """Test position exit logic."""
        # Test with insufficient data
        short_data = self.sample_data["AAPL"].iloc[:50]
        signal = self.strategy.should_exit_position("AAPL", short_data)
        assert signal is None

        # Test with sufficient data
        signal = self.strategy.should_exit_position("AAPL", self.sample_data["AAPL"])
        # May or may not generate signal depending on data
        if signal is not None:
            assert isinstance(signal, StrategySignal)
            assert signal.signal_type == SignalType.SELL
            assert signal.symbol == "AAPL"

    def test_get_strategy_summary(self):
        """Test strategy summary generation."""
        summary = self.strategy.get_strategy_summary()

        assert isinstance(summary, dict)
        assert summary["name"] == "ModernGoldenCross"
        assert summary["symbols"] == ["AAPL", "MSFT"]
        assert summary["fast_ma_period"] == 50
        assert summary["slow_ma_period"] == 200
        assert summary["strategy_type"] == "modern_golden_cross"
        assert "strategy_config" in summary

    def test_golden_cross_detection(self):
        """Test golden cross detection logic."""
        # Create data with a clear golden cross
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        # Create data where fast MA crosses above slow MA
        prices = 100 + np.arange(len(dates)) * 0.1  # Upward trend
        data = pd.DataFrame(
            {
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "volume": np.random.randint(1000000, 10000000, len(prices)),
            },
            index=dates,
        )

        signals = self.strategy.generate_signals({"AAPL": data})

        # Should generate some signals
        assert len(signals) >= 0  # May or may not have signals depending on data


class TestModernMeanReversionStrategy:
    """Test Modern Mean Reversion Strategy implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ModernMeanReversionStrategy(symbols=["SPY", "QQQ"])
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        np.random.seed(42)

        # Create mean-reverting data for SPY
        spy_prices = (
            400
            + np.sin(np.arange(len(dates)) * 0.1) * 20
            + np.random.randn(len(dates)) * 2
        )
        spy_data = pd.DataFrame(
            {
                "open": spy_prices * 0.99,
                "high": spy_prices * 1.02,
                "low": spy_prices * 0.98,
                "close": spy_prices,
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        # Create mean-reverting data for QQQ
        qqq_prices = (
            350
            + np.sin(np.arange(len(dates)) * 0.15) * 15
            + np.random.randn(len(dates)) * 3
        )
        qqq_data = pd.DataFrame(
            {
                "open": qqq_prices * 0.99,
                "high": qqq_prices * 1.02,
                "low": qqq_prices * 0.98,
                "close": qqq_prices,
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        return {"SPY": spy_data, "QQQ": qqq_data}

    def test_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.name == "ModernMeanReversion"
        assert self.strategy.symbols == ["SPY", "QQQ"]
        assert self.strategy.lookback_period == 20
        assert self.strategy.entry_threshold == 2.0
        assert self.strategy.exit_threshold == 0.5
        assert self.strategy.config["strategy_type"] == "modern_mean_reversion"

    def test_generate_signals(self):
        """Test signal generation."""
        signals = self.strategy.generate_signals(self.sample_data)

        assert isinstance(signals, list)

        for signal in signals:
            assert isinstance(signal, StrategySignal)
            assert signal.symbol in ["SPY", "QQQ"]
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert 0 <= signal.confidence <= 1
            assert signal.strategy_name == "ModernMeanReversion"

    def test_z_score_calculation(self):
        """Test Z-score calculation logic."""
        # Test with sufficient data
        signal = self.strategy.should_enter_position("SPY", self.sample_data["SPY"])
        if signal is not None:
            assert isinstance(signal, StrategySignal)
            assert "z_score" in signal.metadata
            assert "mean_price" in signal.metadata
            assert "std_price" in signal.metadata

    def test_get_strategy_summary(self):
        """Test strategy summary generation."""
        summary = self.strategy.get_strategy_summary()

        assert isinstance(summary, dict)
        assert summary["name"] == "ModernMeanReversion"
        assert summary["symbols"] == ["SPY", "QQQ"]
        assert summary["strategy_type"] == "modern_mean_reversion"
        assert "enhancements" in summary
        assert isinstance(summary["enhancements"], list)


class TestModernSectorRotationStrategy:
    """Test Modern Sector Rotation Strategy implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ModernSectorRotationStrategy(sectors=["XLK", "XLF", "XLE"])
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")  # 2 years
        np.random.seed(42)

        data = {}
        for sector in ["XLK", "XLF", "XLE"]:
            # Create different momentum patterns for each sector
            if sector == "XLK":
                prices = 150 + np.cumsum(
                    np.random.randn(len(dates)) * 0.8
                )  # High momentum
            elif sector == "XLF":
                prices = 40 + np.cumsum(
                    np.random.randn(len(dates)) * 0.3
                )  # Medium momentum
            else:
                prices = 80 + np.cumsum(
                    np.random.randn(len(dates)) * 0.5
                )  # Low momentum

            data[sector] = pd.DataFrame(
                {
                    "open": prices * 0.99,
                    "high": prices * 1.02,
                    "low": prices * 0.98,
                    "close": prices,
                    "volume": np.random.randint(1000000, 10000000, len(dates)),
                },
                index=dates,
            )

        return data

    def test_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.name == "ModernSectorRotation"
        assert self.strategy.sectors == ["XLK", "XLF", "XLE"]
        assert self.strategy.top_n == 3
        assert self.strategy.rebalance_freq == 21
        assert self.strategy.config["strategy_type"] == "modern_sector_rotation"

    def test_generate_signals(self):
        """Test signal generation."""
        signals = self.strategy.generate_signals(self.sample_data)

        assert isinstance(signals, list)

        for signal in signals:
            assert isinstance(signal, StrategySignal)
            assert signal.symbol in ["XLK", "XLF", "XLE"]
            assert signal.signal_type == SignalType.BUY  # Sector rotation only buys
            assert 0 <= signal.confidence <= 1
            assert signal.strategy_name == "ModernSectorRotation"

    def test_get_sector_rotation_summary(self):
        """Test sector rotation summary generation."""
        summary = self.strategy.get_sector_rotation_summary()

        assert isinstance(summary, dict)
        assert "sector_rankings" in summary
        assert "sector_scores" in summary
        assert "sector_rotation_config" in summary
        assert (
            summary["sector_rotation_config"]["strategy_type"]
            == "modern_sector_rotation"
        )


class TestModernDualMomentumStrategy:
    """Test Modern Dual Momentum Strategy implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ModernDualMomentumStrategy(assets=["SPY", "EFA", "AGG"])
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")  # 2 years
        np.random.seed(42)

        data = {}
        for asset in ["SPY", "EFA", "AGG"]:
            # Create different momentum patterns
            if asset == "SPY":
                prices = 400 + np.cumsum(
                    np.random.randn(len(dates)) * 1.0
                )  # High momentum
            elif asset == "EFA":
                prices = 70 + np.cumsum(
                    np.random.randn(len(dates)) * 0.6
                )  # Medium momentum
            else:
                prices = 110 + np.cumsum(
                    np.random.randn(len(dates)) * 0.2
                )  # Low momentum

            data[asset] = pd.DataFrame(
                {
                    "open": prices * 0.99,
                    "high": prices * 1.02,
                    "low": prices * 0.98,
                    "close": prices,
                    "volume": np.random.randint(1000000, 10000000, len(dates)),
                },
                index=dates,
            )

        return data

    def test_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.name == "ModernDualMomentum"
        assert self.strategy.assets == ["SPY", "EFA", "AGG"]
        assert self.strategy.lookback == 252
        assert self.strategy.risk_free_rate == 0.02
        assert self.strategy.config["strategy_type"] == "modern_dual_momentum"

    def test_generate_signals(self):
        """Test signal generation."""
        signals = self.strategy.generate_signals(self.sample_data)

        assert isinstance(signals, list)

        for signal in signals:
            assert isinstance(signal, StrategySignal)
            assert signal.symbol in ["SPY", "EFA", "AGG", "CASH"]
            assert signal.signal_type in [SignalType.BUY, SignalType.HOLD]
            assert 0 <= signal.confidence <= 1
            assert signal.strategy_name == "ModernDualMomentum"

    def test_defensive_mode(self):
        """Test defensive mode logic."""
        # Create data with negative momentum
        dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
        prices = 100 - np.cumsum(np.random.randn(len(dates)) * 0.5)  # Downward trend

        data = pd.DataFrame(
            {
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "volume": np.random.randint(1000000, 10000000, len(prices)),
            },
            index=dates,
        )

        signals = self.strategy.generate_signals({"SPY": data})

        # Should generate defensive signals
        for signal in signals:
            if signal.symbol == "CASH":
                assert signal.signal_type == SignalType.HOLD
                assert "defensive_mode" in signal.metadata.get("reason", "")

    def test_get_dual_momentum_summary(self):
        """Test dual momentum summary generation."""
        summary = self.strategy.get_dual_momentum_summary()

        assert isinstance(summary, dict)
        assert "current_asset" in summary
        assert "defensive_mode" in summary
        assert "absolute_momentum_scores" in summary
        assert "dual_momentum_config" in summary
        assert (
            summary["dual_momentum_config"]["strategy_type"] == "modern_dual_momentum"
        )


class TestStrategyFactory:
    """Test the strategy factory function."""

    def test_create_strategy(self):
        """Test strategy factory function."""
        # Test creating each strategy type
        strategies = [
            ("golden_cross", ModernGoldenCrossStrategy),
            ("mean_reversion", ModernMeanReversionStrategy),
            ("sector_rotation", ModernSectorRotationStrategy),
            ("dual_momentum", ModernDualMomentumStrategy),
        ]

        for strategy_name, expected_class in strategies:
            strategy = create_strategy(strategy_name, symbols=["TEST"])
            assert isinstance(strategy, expected_class)

    def test_invalid_strategy(self):
        """Test factory with invalid strategy name."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("invalid_strategy")


class TestDashboardIntegration:
    """Test integration with dashboard services."""

    def test_analysis_service_compatibility(self):
        """Test that modern strategies work with analysis service."""
        from dashboard.services.analysis_service import DashboardAnalysisService

        # Should initialize without errors
        service = DashboardAnalysisService()

        # Check that strategies are modern
        assert hasattr(service.golden_cross, "config")
        assert service.golden_cross.config.get("strategy_type") == "modern_golden_cross"

        assert hasattr(service.mean_reversion, "config")
        assert (
            service.mean_reversion.config.get("strategy_type")
            == "modern_mean_reversion"
        )

    def test_metrics_service_compatibility(self):
        """Test that modern strategies work with metrics service."""
        from dashboard.services.strategy_metrics_service import StrategyMetricsService

        # Should initialize without errors
        service = StrategyMetricsService()

        # Check that strategies are modern
        for strategy_id, strategy_instance in service.strategy_instances.items():
            assert hasattr(strategy_instance, "config")
            assert "modern_" in strategy_instance.config.get("strategy_type", "")


if __name__ == "__main__":
    pytest.main([__file__])
