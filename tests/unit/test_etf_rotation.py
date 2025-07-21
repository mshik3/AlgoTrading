"""
Unit tests for ETF rotation strategies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from strategies.etf.rotation_base import BaseETFRotationStrategy
from strategies.etf.dual_momentum import DualMomentumStrategy
from strategies.etf.sector_rotation import SectorRotationStrategy
from strategies.base import StrategySignal, SignalType
from utils.asset_categorization import get_etf_universe_for_strategy


class TestBaseETFRotationStrategy:
    """Test BaseETFRotationStrategy functionality."""

    def test_base_rotation_strategy_initialization(self):
        """Test BaseETFRotationStrategy initialization."""
        # Test abstract class can't be instantiated
        with pytest.raises(TypeError):
            BaseETFRotationStrategy("test", {"category": ["SPY"]})

    def test_momentum_calculation(self):
        """Test momentum calculation methods."""

        # Create a concrete subclass for testing
        class TestRotationStrategy(BaseETFRotationStrategy):
            def generate_signals(self, market_data):
                return []

            def should_enter_position(self, symbol, data):
                return None

            def should_exit_position(self, symbol, data):
                return None

        strategy = TestRotationStrategy("test", {"category": ["SPY"]})

        # Create sample data with known returns
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        data = pd.DataFrame(
            {
                "Open": [100.0] * 300,
                "High": [102.0] * 300,
                "Low": [99.0] * 300,
                "Close": [100.0 + i * 0.1 for i in range(300)],  # Upward trend
                "Volume": [1000000] * 300,
            },
            index=dates,
        )

        # Test returns method
        momentum = strategy.calculate_momentum(data, lookback=252, method="returns")
        assert not np.isnan(momentum)
        assert momentum > 0  # Should be positive for upward trend

    def test_relative_strength_calculation(self):
        """Test relative strength calculation."""

        class TestRotationStrategy(BaseETFRotationStrategy):
            def generate_signals(self, market_data):
                return []

            def should_enter_position(self, symbol, data):
                return None

            def should_exit_position(self, symbol, data):
                return None

        strategy = TestRotationStrategy("test", {"category": ["SPY"]})

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        symbol_data = pd.DataFrame(
            {
                "Close": [100.0 + i * 0.2 for i in range(300)],  # Stronger trend
            },
            index=dates,
        )

        benchmark_data = pd.DataFrame(
            {
                "Close": [100.0 + i * 0.1 for i in range(300)],  # Weaker trend
            },
            index=dates,
        )

        rs = strategy.calculate_relative_strength(
            symbol_data, benchmark_data, lookback=252
        )
        assert not np.isnan(rs)
        assert rs > 0  # Symbol should outperform benchmark

    def test_etf_ranking(self):
        """Test ETF ranking functionality."""

        class TestRotationStrategy(BaseETFRotationStrategy):
            def generate_signals(self, market_data):
                return []

            def should_enter_position(self, symbol, data):
                return None

            def should_exit_position(self, symbol, data):
                return None

        strategy = TestRotationStrategy("test", {"category": ["SPY", "QQQ"]})

        # Create market data
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        market_data = {
            "SPY": pd.DataFrame(
                {
                    "Close": [100.0 + i * 0.1 for i in range(300)],
                },
                index=dates,
            ),
            "QQQ": pd.DataFrame(
                {
                    "Close": [
                        100.0 + i * 0.15 for i in range(300)
                    ],  # Better performance
                },
                index=dates,
            ),
        }

        rankings = strategy.rank_etfs_by_momentum(market_data)
        assert len(rankings) == 2
        assert rankings[0][0] == "QQQ"  # QQQ should rank higher
        assert rankings[0][1] > rankings[1][1]  # Higher momentum score

    def test_rebalancing_logic(self):
        """Test rebalancing frequency logic."""

        class TestRotationStrategy(BaseETFRotationStrategy):
            def generate_signals(self, market_data):
                return []
            
            def should_enter_position(self, symbol, data):
                return None
            
            def should_exit_position(self, symbol, data):
                return None

        strategy = TestRotationStrategy("test", {"category": ["SPY"]})

        # Test initial rebalancing
        assert strategy.should_rebalance(datetime.now()) is True

        # Test after setting rebalance date
        strategy.last_rebalance_date = datetime.now()
        assert strategy.should_rebalance(datetime.now()) is False

        # Test after rebalance frequency period
        future_date = datetime.now() + timedelta(days=25)
        assert strategy.should_rebalance(future_date) is True


class TestDualMomentumStrategy:
    """Test DualMomentumStrategy functionality."""

    def test_dual_momentum_initialization(self):
        """Test DualMomentumStrategy initialization."""
        strategy = DualMomentumStrategy()
        assert strategy.name == "Dual Momentum ETF Rotation"
        assert "US_Equities" in strategy.etf_universe
        assert "Bonds" in strategy.etf_universe

    def test_absolute_momentum_calculation(self):
        """Test absolute momentum calculation."""
        strategy = DualMomentumStrategy()

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        data = pd.DataFrame(
            {
                "Close": [100.0 + i * 0.1 for i in range(300)],
            },
            index=dates,
        )

        absolute_momentum = strategy.calculate_absolute_momentum(data)
        assert not np.isnan(absolute_momentum)
        # Should be positive for upward trend minus risk-free rate

    def test_qualified_assets_selection(self):
        """Test qualified assets selection."""
        strategy = DualMomentumStrategy()

        # Create market data with mixed performance
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        market_data = {
            "SPY": pd.DataFrame(
                {
                    "Close": [100.0 + i * 0.1 for i in range(300)],  # Positive trend
                },
                index=dates,
            ),
            "TLT": pd.DataFrame(
                {
                    "Close": [100.0 - i * 0.05 for i in range(300)],  # Negative trend
                },
                index=dates,
            ),
        }

        qualified_assets = strategy.get_qualified_assets(market_data)
        assert "SPY" in qualified_assets  # Should be qualified
        assert "TLT" not in qualified_assets  # Should not be qualified

    def test_relative_momentum_calculation(self):
        """Test relative momentum calculation."""
        strategy = DualMomentumStrategy()

        # Create market data
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        market_data = {
            "SPY": pd.DataFrame(
                {
                    "Close": [100.0 + i * 0.1 for i in range(300)],
                },
                index=dates,
            ),
            "QQQ": pd.DataFrame(
                {
                    "Close": [
                        100.0 + i * 0.15 for i in range(300)
                    ],  # Better performance
                },
                index=dates,
            ),
        }

        qualified_assets = ["SPY", "QQQ"]
        relative_momentums = strategy.calculate_relative_momentum(
            market_data, qualified_assets
        )

        assert len(relative_momentums) == 2
        assert relative_momentums[0][0] == "QQQ"  # QQQ should rank higher
        assert relative_momentums[0][1] > relative_momentums[1][1]

    def test_signal_generation(self):
        """Test signal generation."""
        strategy = DualMomentumStrategy()

        # Create market data
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        market_data = {
            "SPY": pd.DataFrame(
                {
                    "Close": [100.0 + i * 0.1 for i in range(300)],
                },
                index=dates,
            ),
            "TLT": pd.DataFrame(
                {
                    "Close": [100.0 - i * 0.05 for i in range(300)],
                },
                index=dates,
            ),
        }

        signals = strategy.generate_signals(market_data)
        assert isinstance(signals, list)
        # Should generate signals based on momentum rankings


class TestSectorRotationStrategy:
    """Test SectorRotationStrategy functionality."""

    def test_sector_rotation_initialization(self):
        """Test SectorRotationStrategy initialization."""
        strategy = SectorRotationStrategy()
        assert strategy.name == "Sector ETF Rotation"
        assert "Technology" in strategy.etf_universe
        assert "Financials" in strategy.etf_universe

    def test_sector_score_calculation(self):
        """Test sector score calculation."""
        strategy = SectorRotationStrategy()

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        sector_data = pd.DataFrame(
            {
                "Close": [100.0 + i * 0.1 for i in range(300)],
            },
            index=dates,
        )

        benchmark_data = pd.DataFrame(
            {
                "Close": [100.0 + i * 0.05 for i in range(300)],
            },
            index=dates,
        )

        scores = strategy.calculate_sector_score(sector_data, benchmark_data, "XLK")
        assert "momentum" in scores
        assert "relative_strength" in scores
        assert "combined_score" in scores
        assert scores["momentum"] > 0
        assert scores["relative_strength"] > 0

    def test_sector_ranking(self):
        """Test sector ranking functionality."""
        strategy = SectorRotationStrategy()

        # Create market data
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        market_data = {
            "SPY": pd.DataFrame(
                {  # Benchmark
                    "Close": [100.0 + i * 0.05 for i in range(300)],
                },
                index=dates,
            ),
            "XLK": pd.DataFrame(
                {  # Technology
                    "Close": [100.0 + i * 0.15 for i in range(300)],
                },
                index=dates,
            ),
            "XLF": pd.DataFrame(
                {  # Financials
                    "Close": [100.0 + i * 0.08 for i in range(300)],
                },
                index=dates,
            ),
        }

        rankings = strategy.rank_sectors(market_data)
        assert len(rankings) > 0
        # XLK should rank higher than XLF due to better performance

    def test_sector_rotation_signals(self):
        """Test sector rotation signal generation."""
        strategy = SectorRotationStrategy()

        # Create market data
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        market_data = {
            "SPY": pd.DataFrame(
                {
                    "Close": [100.0 + i * 0.05 for i in range(300)],
                },
                index=dates,
            ),
            "XLK": pd.DataFrame(
                {
                    "Close": [100.0 + i * 0.15 for i in range(300)],
                },
                index=dates,
            ),
            "XLF": pd.DataFrame(
                {
                    "Close": [100.0 + i * 0.08 for i in range(300)],
                },
                index=dates,
            ),
        }

        signals = strategy.generate_signals(market_data)
        assert isinstance(signals, list)
        # Should generate signals based on sector rankings


class TestETFUniverseConfiguration:
    """Test ETF universe configuration utilities."""

    def test_etf_universe_retrieval(self):
        """Test ETF universe retrieval."""
        dual_momentum_universe = get_etf_universe_for_strategy("dual_momentum")
        assert "US_Equities" in dual_momentum_universe
        assert "Bonds" in dual_momentum_universe
        assert "SPY" in dual_momentum_universe["US_Equities"]

        sector_universe = get_etf_universe_for_strategy("sector_rotation")
        assert "Technology" in sector_universe
        assert "Financials" in sector_universe
        assert "XLK" in sector_universe["Technology"]

    def test_invalid_strategy_type(self):
        """Test handling of invalid strategy type."""
        universe = get_etf_universe_for_strategy("invalid_strategy")
        assert universe == {}


class TestMomentumRankingUtilities:
    """Test momentum ranking utilities."""

    def test_momentum_score_calculation(self):
        """Test momentum score calculation."""
        from utils.momentum_ranking import calculate_momentum_score

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        data = pd.DataFrame(
            {
                "Close": [100.0 + i * 0.1 for i in range(300)],
            },
            index=dates,
        )

        momentum = calculate_momentum_score(data, lookback=252, method="returns")
        assert not np.isnan(momentum)
        assert momentum > 0

    def test_relative_strength_calculation(self):
        """Test relative strength calculation."""
        from utils.momentum_ranking import calculate_relative_strength

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        symbol_data = pd.DataFrame(
            {
                "Close": [100.0 + i * 0.15 for i in range(300)],
            },
            index=dates,
        )

        benchmark_data = pd.DataFrame(
            {
                "Close": [100.0 + i * 0.1 for i in range(300)],
            },
            index=dates,
        )

        rs = calculate_relative_strength(symbol_data, benchmark_data, lookback=252)
        assert not np.isnan(rs)
        assert rs > 0

    def test_asset_ranking(self):
        """Test asset ranking functionality."""
        from utils.momentum_ranking import rank_assets_by_momentum

        # Create market data
        dates = pd.date_range("2023-01-01", periods=300, freq="D")
        market_data = {
            "SPY": pd.DataFrame(
                {
                    "Close": [100.0 + i * 0.1 for i in range(300)],
                },
                index=dates,
            ),
            "QQQ": pd.DataFrame(
                {
                    "Close": [100.0 + i * 0.15 for i in range(300)],
                },
                index=dates,
            ),
        }

        rankings = rank_assets_by_momentum(market_data, ["SPY", "QQQ"])
        assert len(rankings) == 2
        assert rankings[0][0] == "QQQ"  # QQQ should rank higher
