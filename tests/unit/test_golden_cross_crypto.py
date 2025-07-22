"""
Unit tests for Golden Cross strategy with crypto assets.
Tests strategy initialization, signal generation, and crypto asset handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestGoldenCrossCryptoStrategy:
    """Test Golden Cross strategy with crypto assets."""

    @pytest.fixture
    def sample_crypto_data(self):
        """Generate sample crypto data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300, freq="D")

        # Generate realistic crypto price data (more volatile than stocks)
        base_price = 50000  # BTC-like starting price
        returns = np.random.normal(0.02, 0.05, len(dates))  # 2% mean, 5% std
        prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.01, len(dates))),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.02, len(dates)))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.02, len(dates)))),
                "Close": prices,
                "Volume": np.random.randint(1000, 10000, len(dates)),
                "Adj Close": prices,
            },
            index=dates,
        )

        # Ensure logical price relationships
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        return data

    @pytest.fixture
    def sample_stock_data(self):
        """Generate sample stock data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300, freq="D")

        # Generate realistic stock price data
        base_price = 100  # SPY-like starting price
        returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% mean, 2% std
        prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, len(dates)),
                "Adj Close": prices,
            },
            index=dates,
        )

        # Ensure logical price relationships
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        return data

    def test_strategy_initialization_with_crypto(self):
        """Test that Golden Cross strategy initializes with crypto symbols."""
        from strategies.equity.golden_cross_crypto import GoldenCrossCryptoStrategy

        strategy = GoldenCrossCryptoStrategy()

        # Verify crypto symbols are included
        crypto_symbols = [
            "BTCUSD",
            "ETHUSD",
            "SOLUSD",
            "DOTUSD",
            "LINKUSD",
            "LTCUSD",
            "BCHUSD",
            "XRPUSD",
            "SOLUSD",
            "MATICUSD",
        ]

        for symbol in crypto_symbols:
            assert (
                symbol in strategy.symbols
            ), f"Crypto symbol {symbol} should be in strategy"

        # Verify total symbol count
        assert (
            len(strategy.symbols) >= 10
        ), f"Strategy should have at least 10 symbols, got {len(strategy.symbols)}"

        # Verify all symbols are crypto
        for symbol in strategy.symbols:
            assert symbol.endswith(
                "USD"
            ), f"Symbol {symbol} should be a crypto symbol ending in USD"

    def test_strategy_configuration_with_crypto(self):
        """Test that strategy configuration works with crypto assets."""
        from strategies.equity.golden_cross import GoldenCrossStrategy

        # Test with custom configuration
        custom_config = {
            "fast_ma_period": 20,  # Shorter for crypto (more volatile)
            "slow_ma_period": 100,  # Shorter for crypto
            "min_trend_strength": 0.05,  # Higher for crypto volatility
            "max_position_size": 0.20,  # Smaller for crypto risk
        }

        strategy = GoldenCrossStrategy(**custom_config)

        # Verify configuration was applied
        assert strategy.fast_ma_period == 20
        assert strategy.slow_ma_period == 100
        assert strategy.min_trend_strength == 0.05
        assert strategy.max_position_size == 0.20

        # Verify crypto symbols are still included
        assert "BTCUSD" in strategy.symbols
        assert "ETHUSD" in strategy.symbols

    def test_crypto_golden_cross_detection(self, sample_crypto_data):
        """Test Golden Cross detection with crypto data."""
        from strategies.equity.golden_cross import GoldenCrossStrategy
        from indicators.technical import TechnicalIndicators

        strategy = GoldenCrossStrategy()

        # Add technical indicators to crypto data
        indicators = TechnicalIndicators(sample_crypto_data)
        indicators.add_sma(strategy.fast_ma_period, "Close")
        indicators.add_sma(strategy.slow_ma_period, "Close")
        enhanced_data = indicators.get_data()

        # Test Golden Cross detection
        signal = strategy.should_enter_position("BTCUSD", enhanced_data)

        # Signal should be None if no Golden Cross, or a valid signal if detected
        if signal is not None:
            assert signal.symbol == "BTCUSD"
            assert signal.signal_type.value == "buy"
            assert signal.confidence > 0
            assert signal.price > 0

    def test_crypto_death_cross_detection(self, sample_crypto_data):
        """Test Death Cross detection with crypto data."""
        from strategies.equity.golden_cross import GoldenCrossStrategy
        from indicators.technical import TechnicalIndicators

        strategy = GoldenCrossStrategy()

        # Add technical indicators to crypto data
        indicators = TechnicalIndicators(sample_crypto_data)
        indicators.add_sma(strategy.fast_ma_period, "Close")
        indicators.add_sma(strategy.slow_ma_period, "Close")
        enhanced_data = indicators.get_data()

        # Test Death Cross detection
        signal = strategy.should_exit_position("BTCUSD", enhanced_data)

        # Signal should be None if no Death Cross, or a valid signal if detected
        if signal is not None:
            assert signal.symbol == "BTCUSD"
            assert signal.signal_type.value == "sell"
            assert signal.confidence > 0
            assert signal.price > 0

    def test_mixed_asset_signal_generation(self, sample_crypto_data, sample_stock_data):
        """Test signal generation with mixed crypto and stock data."""
        from strategies.equity.golden_cross import GoldenCrossStrategy
        from indicators.technical import TechnicalIndicators

        strategy = GoldenCrossStrategy()

        # Create market data with both crypto and stocks
        market_data = {
            "BTCUSD": sample_crypto_data,
            "SPY": sample_stock_data,
        }

        # Generate signals for all symbols
        signals = strategy.generate_signals(market_data)

        # Verify signals structure
        assert isinstance(signals, list)

        # Check if signals were generated for any symbols
        if signals:
            # Verify signal properties
            for signal in signals:
                assert signal.symbol in ["BTCUSD", "SPY"]
                assert signal.signal_type.value in ["buy", "sell"]
                assert signal.confidence > 0
                assert signal.price > 0

    def test_crypto_volatility_handling(self):
        """Test that strategy handles crypto volatility appropriately."""
        from strategies.equity.golden_cross import GoldenCrossStrategy

        strategy = GoldenCrossStrategy()

        # Create highly volatile crypto data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300, freq="D")

        # Simulate crypto boom and bust cycle
        prices = np.ones(len(dates)) * 50000
        prices[100:150] *= 2  # Boom period
        prices[200:250] *= 0.5  # Bust period

        volatile_data = pd.DataFrame(
            {
                "Open": prices,
                "High": prices * 1.05,
                "Low": prices * 0.95,
                "Close": prices,
                "Volume": np.random.randint(1000, 10000, len(dates)),
                "Adj Close": prices,
            },
            index=dates,
        )

        # Test that strategy can handle this volatility
        signal = strategy.should_enter_position("BTCUSD", volatile_data)

        # Should not crash and should return either None or valid signal
        assert signal is None or (
            hasattr(signal, "symbol") and signal.symbol == "BTCUSD"
        )

    def test_crypto_position_sizing(self, sample_crypto_data):
        """Test position sizing for crypto assets."""
        from strategies.equity.golden_cross import GoldenCrossStrategy
        from indicators.technical import TechnicalIndicators

        strategy = GoldenCrossStrategy()

        # Add technical indicators
        indicators = TechnicalIndicators(sample_crypto_data)
        indicators.add_sma(strategy.fast_ma_period, "Close")
        indicators.add_sma(strategy.slow_ma_period, "Close")
        enhanced_data = indicators.get_data()

        # Test position sizing calculation
        signal = strategy.should_enter_position("BTCUSD", enhanced_data)

        if signal is not None:
            # Verify position size is reasonable for crypto
            assert signal.position_size > 0
            assert signal.position_size <= strategy.max_position_size

            # Crypto should typically have smaller position sizes due to volatility
            assert signal.position_size <= 0.30  # Max 30% per position

    def test_crypto_trend_strength_calculation(self, sample_crypto_data):
        """Test trend strength calculation for crypto assets."""
        from strategies.equity.golden_cross import GoldenCrossStrategy
        from indicators.technical import TechnicalIndicators

        strategy = GoldenCrossStrategy()

        # Add technical indicators
        indicators = TechnicalIndicators(sample_crypto_data)
        indicators.add_sma(strategy.fast_ma_period, "Close")
        indicators.add_sma(strategy.slow_ma_period, "Close")
        enhanced_data = indicators.get_data()

        # Calculate trend strength
        latest = enhanced_data.iloc[-1]
        sma_50 = latest.get(f"SMA_{strategy.fast_ma_period}", 0)
        sma_200 = latest.get(f"SMA_{strategy.slow_ma_period}", 0)

        if sma_200 > 0:
            trend_strength = (sma_50 - sma_200) / sma_200

            # Trend strength should be reasonable for crypto (can be higher due to volatility)
            assert (
                abs(trend_strength) < 2.0
            )  # Allow up to 200% difference for crypto volatility

            # If trend strength is significant, should generate signal
            if abs(trend_strength) > strategy.min_trend_strength:
                signal = strategy.should_enter_position("BTCUSD", enhanced_data)
                # Signal might be generated, but not guaranteed due to other factors

    def test_crypto_data_quality_requirements(self):
        """Test that strategy handles crypto data quality requirements."""
        from strategies.equity.golden_cross import GoldenCrossStrategy

        strategy = GoldenCrossStrategy()

        # Test with insufficient data
        insufficient_data = pd.DataFrame(
            {
                "Open": [50000] * 50,  # Only 50 days, need 200+ for 200-day MA
                "High": [51000] * 50,
                "Low": [49000] * 50,
                "Close": [50000] * 50,
                "Volume": [1000] * 50,
                "Adj Close": [50000] * 50,
            }
        )

        # Should handle insufficient data gracefully
        signal = strategy.should_enter_position("BTCUSD", insufficient_data)
        assert signal is None  # Should not generate signal with insufficient data

    def test_crypto_strategy_summary(self):
        """Test strategy summary includes crypto information."""
        from strategies.equity.golden_cross import GoldenCrossStrategy

        strategy = GoldenCrossStrategy()
        summary = strategy.get_strategy_summary()

        # Verify summary structure
        assert "name" in summary
        assert "symbols" in summary
        assert "fast_ma_period" in summary
        assert "slow_ma_period" in summary

        # Verify crypto symbols are included in summary
        crypto_symbols = [sym for sym in strategy.symbols if sym.endswith("USD")]
        assert len(crypto_symbols) == 10

        # Verify all crypto symbols are in the summary
        for symbol in crypto_symbols:
            assert symbol in summary["symbols"]

    def test_crypto_crossover_state_tracking(self, sample_crypto_data):
        """Test crossover state tracking for crypto assets."""
        from strategies.equity.golden_cross import GoldenCrossStrategy
        from indicators.technical import TechnicalIndicators

        strategy = GoldenCrossStrategy()

        # Add technical indicators
        indicators = TechnicalIndicators(sample_crypto_data)
        indicators.add_sma(strategy.fast_ma_period, "Close")
        indicators.add_sma(strategy.slow_ma_period, "Close")
        enhanced_data = indicators.get_data()

        # Test crossover state tracking
        signal = strategy.should_enter_position("BTCUSD", enhanced_data)

        # Verify crossover states are tracked
        assert hasattr(strategy, "crossover_states")
        assert isinstance(strategy.crossover_states, dict)

        # If signal was generated, crossover state should be updated
        if signal is not None:
            assert "BTCUSD" in strategy.crossover_states
            assert strategy.crossover_states["BTCUSD"] in ["golden", "death", "none"]

    def test_crypto_signal_frequency(self, sample_crypto_data):
        """Test that crypto signals are generated at appropriate frequency."""
        from strategies.equity.golden_cross import GoldenCrossStrategy
        from indicators.technical import TechnicalIndicators

        strategy = GoldenCrossStrategy()

        # Add technical indicators
        indicators = TechnicalIndicators(sample_crypto_data)
        indicators.add_sma(strategy.fast_ma_period, "Close")
        indicators.add_sma(strategy.slow_ma_period, "Close")
        enhanced_data = indicators.get_data()

        # Test multiple signal generations
        signals = []
        for i in range(10):
            signal = strategy.should_enter_position("BTCUSD", enhanced_data)
            if signal is not None:
                signals.append(signal)

        # Crypto should not generate excessive signals due to volatility filtering
        # But should still generate some signals if conditions are met
        assert len(signals) <= 5  # Should not generate too many signals in short period
