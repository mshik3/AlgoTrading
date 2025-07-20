"""
Unit tests for technical indicators calculations.
"""

import pytest
import pandas as pd
import numpy as np
from indicators.technical import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_bollinger_bands,
    TechnicalIndicators,
)


class TestBasicIndicators:
    """Test basic technical indicators."""

    @pytest.fixture
    def sample_prices(self):
        """Sample price data for testing."""
        np.random.seed(42)
        prices = 100 + np.random.randn(50).cumsum() * 0.1
        return pd.Series(prices)

    def test_sma_calculation(self, sample_prices):
        """Test Simple Moving Average calculation."""
        sma = calculate_sma(sample_prices, window=5)

        # Check that we have the right number of values
        assert len(sma) == len(sample_prices)

        # Check that first 4 values are NaN (window=5)
        assert sma.iloc[:4].isna().all()

        # Check that 5th value equals mean of first 5 prices
        expected_5th = sample_prices.iloc[:5].mean()
        assert abs(sma.iloc[4] - expected_5th) < 1e-10

    def test_ema_calculation(self, sample_prices):
        """Test Exponential Moving Average calculation."""
        ema = calculate_ema(sample_prices, window=5)

        # Check that we have the right number of values
        assert len(ema) == len(sample_prices)

        # Check that first value is not NaN
        assert not np.isnan(ema.iloc[0])

        # EMA should be smoother than price series (less volatile)
        price_volatility = sample_prices.std()
        ema_volatility = ema.std()
        assert ema_volatility < price_volatility

    def test_rsi_calculation(self, sample_prices):
        """Test RSI calculation."""
        rsi = calculate_rsi(sample_prices, window=14)

        # Check that we have the right number of values
        assert len(rsi) == len(sample_prices)

        # Check that RSI values are between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

        # First 14 values should be NaN
        assert rsi.iloc[:14].isna().all()

    def test_bollinger_bands(self, sample_prices):
        """Test Bollinger Bands calculation."""
        bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(
            sample_prices, window=20, num_std=2
        )

        # Check lengths
        assert len(bb_upper) == len(sample_prices)
        assert len(bb_lower) == len(sample_prices)
        assert len(bb_middle) == len(sample_prices)

        # Upper band should be above lower band
        valid_data = ~(bb_upper.isna() | bb_lower.isna())
        assert (bb_upper[valid_data] > bb_lower[valid_data]).all()

        # Middle band should be between upper and lower
        assert (bb_middle[valid_data] >= bb_lower[valid_data]).all()
        assert (bb_middle[valid_data] <= bb_upper[valid_data]).all()


class TestTechnicalIndicatorsClass:
    """Test the TechnicalIndicators wrapper class."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        data = pd.DataFrame(
            {
                "Open": 100 + np.random.randn(50).cumsum() * 0.1,
                "High": 100 + np.random.randn(50).cumsum() * 0.1 + 1,
                "Low": 100 + np.random.randn(50).cumsum() * 0.1 - 1,
                "Close": 100 + np.random.randn(50).cumsum() * 0.1,
                "Volume": np.random.randint(1000000, 10000000, 50),
            },
            index=dates,
        )

        # Ensure price relationships are logical
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        return data

    def test_technical_indicators_initialization(self, sample_ohlcv_data):
        """Test TechnicalIndicators class initialization."""
        indicators = TechnicalIndicators(sample_ohlcv_data)
        assert indicators.data.equals(sample_ohlcv_data)

    def test_missing_columns_validation(self):
        """Test validation of required columns."""
        incomplete_data = pd.DataFrame({"Open": [1, 2, 3]})

        with pytest.raises(ValueError, match="Missing required columns"):
            TechnicalIndicators(incomplete_data)

    def test_method_chaining(self, sample_ohlcv_data):
        """Test that indicator methods can be chained."""
        indicators = TechnicalIndicators(sample_ohlcv_data)

        result = indicators.add_sma(20).add_rsi(14).add_ema(12)

        # Check that it returns the same instance (for chaining)
        assert isinstance(result, TechnicalIndicators)

        # Check that indicators were added
        assert "SMA_20" in indicators.data.columns
        assert "RSI_14" in indicators.data.columns
        assert "EMA_12" in indicators.data.columns

    def test_get_signals_basic(self, sample_ohlcv_data):
        ti = TechnicalIndicators(sample_ohlcv_data)
        ti.add_rsi().add_bollinger_bands()
        signals = ti.get_signals()
        # Accept either rsi_oversold or rsi_overbought as valid keys
        assert any(k in signals for k in ["rsi_oversold", "rsi_overbought"])


class TestIndicatorEdgeCases:
    """Test edge cases and error handling for indicators."""

    def test_empty_series(self):
        """Test indicators with empty data."""
        empty_series = pd.Series([])

        sma = calculate_sma(empty_series, 5)
        assert len(sma) == 0

    def test_insufficient_data(self):
        """Test indicators with insufficient data points."""
        short_series = pd.Series([1, 2, 3])

        # SMA with window larger than data should return all NaN except for valid windows
        sma = calculate_sma(short_series, window=5)
        assert len(sma) == 3
        assert sma.isna().all()

        # But smaller window should work
        sma_small = calculate_sma(short_series, window=2)
        assert not sma_small.iloc[-1:].isna().any()  # Last value should be valid

    def test_invalid_parameters(self):
        # Should raise ValueError for negative window
        with pytest.raises(ValueError):
            calculate_rsi(pd.Series([1, 2, 3]), window=-1)
        # Should raise ValueError for zero window
        with pytest.raises(ValueError):
            calculate_rsi(pd.Series([1, 2, 3]), window=0)
