"""
Unit tests for DataValidator class.
Tests data validation functionality with comprehensive edge cases and error conditions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from data.processors import DataValidator


class TestDataValidator:
    """Test DataValidator functionality."""

    @pytest.fixture
    def valid_ohlcv_data(self):
        """Create valid OHLCV data for testing."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = {
            "Open": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ],
            "High": [
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
            ],
            "Low": [
                99.0,
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
            ],
            "Close": [
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
            ],
            "Volume": [
                1000000,
                1100000,
                1200000,
                1300000,
                1400000,
                1500000,
                1600000,
                1700000,
                1800000,
                1900000,
            ],
            "Adj Close": [
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
            ],
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def invalid_ohlcv_data(self):
        """Create invalid OHLCV data for testing."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data = {
            "Open": [100.0, 101.0, np.nan, 103.0, 104.0],
            "High": [102.0, 103.0, 104.0, 105.0, 106.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            "Adj Close": [101.0, 102.0, 103.0, 104.0, 105.0],
        }
        return pd.DataFrame(data, index=dates)

    def test_validate_dataframe_valid_data(self, valid_ohlcv_data):
        """Test validation of valid OHLCV data."""
        is_valid, error_msg = DataValidator.validate_dataframe(valid_ohlcv_data, "AAPL")
        assert is_valid is True
        assert error_msg is None

    def test_validate_dataframe_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        empty_df = pd.DataFrame()
        is_valid, error_msg = DataValidator.validate_dataframe(empty_df, "AAPL")
        assert is_valid is False
        assert "Empty DataFrame" in error_msg

    def test_validate_dataframe_none_dataframe(self):
        """Test validation of None DataFrame."""
        is_valid, error_msg = DataValidator.validate_dataframe(None, "AAPL")
        assert is_valid is False
        assert "Empty DataFrame" in error_msg

    def test_validate_dataframe_missing_columns(self):
        """Test validation with missing required columns."""
        incomplete_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                # Missing Close, Volume, Adj Close
            }
        )
        is_valid, error_msg = DataValidator.validate_dataframe(incomplete_data, "AAPL")
        assert is_valid is False
        assert "Missing column" in error_msg

    def test_validate_dataframe_non_numeric_columns(self):
        """Test validation with non-numeric data."""
        non_numeric_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": ["invalid", "data"],  # Non-numeric
                "Adj Close": [101.0, 102.0],
            }
        )
        is_valid, error_msg = DataValidator.validate_dataframe(non_numeric_data, "AAPL")
        assert is_valid is False
        assert "is not numeric" in error_msg

    def test_validate_dataframe_nan_values(self, invalid_ohlcv_data):
        """Test validation with NaN values."""
        is_valid, error_msg = DataValidator.validate_dataframe(
            invalid_ohlcv_data, "AAPL"
        )
        # Should still be valid but with warning
        assert is_valid is True
        assert error_msg is None

    def test_validate_dataframe_negative_prices(self):
        """Test validation with negative prices."""
        negative_price_data = pd.DataFrame(
            {
                "Open": [100.0, -101.0],  # Negative price
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000000, 1100000],
                "Adj Close": [101.0, 102.0],
            }
        )
        is_valid, error_msg = DataValidator.validate_dataframe(
            negative_price_data, "AAPL"
        )
        # Should still be valid but with warning
        assert is_valid is True
        assert error_msg is None

    def test_validate_dataframe_negative_volume(self):
        """Test validation with negative volume."""
        negative_volume_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000000, -1100000],  # Negative volume
                "Adj Close": [101.0, 102.0],
            }
        )
        is_valid, error_msg = DataValidator.validate_dataframe(
            negative_volume_data, "AAPL"
        )
        # Should still be valid but with warning
        assert is_valid is True
        assert error_msg is None

    def test_validate_dataframe_zero_prices(self):
        """Test validation with zero prices."""
        zero_price_data = pd.DataFrame(
            {
                "Open": [100.0, 0.0],  # Zero price
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000000, 1100000],
                "Adj Close": [101.0, 102.0],
            }
        )
        is_valid, error_msg = DataValidator.validate_dataframe(zero_price_data, "AAPL")
        # Should still be valid but with warning
        assert is_valid is True
        assert error_msg is None

    def test_validate_dataframe_price_relationships(self):
        """Test validation of price relationships (High >= Low, etc.)."""
        invalid_price_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 99.0],  # High < Low
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000000, 1100000],
                "Adj Close": [101.0, 102.0],
            }
        )
        is_valid, error_msg = DataValidator.validate_dataframe(
            invalid_price_data, "AAPL"
        )
        # Should still be valid but with warning
        assert is_valid is True
        assert error_msg is None

    def test_validate_dataframe_single_row(self):
        """Test validation of single row DataFrame."""
        single_row_data = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000000],
                "Adj Close": [101.0],
            }
        )
        is_valid, error_msg = DataValidator.validate_dataframe(single_row_data, "AAPL")
        assert is_valid is True
        assert error_msg is None

    def test_validate_dataframe_large_dataset(self):
        """Test validation of large dataset."""
        dates = pd.date_range("2023-01-01", periods=1000, freq="D")
        large_data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 200, 1000),
                "High": np.random.uniform(200, 300, 1000),
                "Low": np.random.uniform(50, 100, 1000),
                "Close": np.random.uniform(100, 200, 1000),
                "Volume": np.random.randint(1000000, 10000000, 1000),
                "Adj Close": np.random.uniform(100, 200, 1000),
            },
            index=dates,
        )
        is_valid, error_msg = DataValidator.validate_dataframe(large_data, "AAPL")
        assert is_valid is True
        assert error_msg is None

    def test_validate_dataframe_mixed_data_types(self):
        """Test validation with mixed data types in columns."""
        mixed_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000000, 1100000],
                "Adj Close": [101.0, 102.0],
            }
        )
        # Convert some columns to different types
        mixed_data["Open"] = mixed_data["Open"].astype("float32")
        mixed_data["Volume"] = mixed_data["Volume"].astype("int64")

        is_valid, error_msg = DataValidator.validate_dataframe(mixed_data, "AAPL")
        assert is_valid is True
        assert error_msg is None

    def test_validate_dataframe_with_index(self, valid_ohlcv_data):
        """Test validation with DataFrame that has a datetime index."""
        is_valid, error_msg = DataValidator.validate_dataframe(valid_ohlcv_data, "AAPL")
        assert is_valid is True
        assert error_msg is None

    def test_validate_dataframe_symbol_logging(self, valid_ohlcv_data, caplog):
        """Test that symbol is properly logged during validation."""
        with caplog.at_level("WARNING"):
            DataValidator.validate_dataframe(valid_ohlcv_data, "TEST_SYMBOL")

        # Should not have any warnings for valid data
        assert len(caplog.records) == 0

    def test_validate_dataframe_warning_logging(self, invalid_ohlcv_data, caplog):
        """Test that warnings are properly logged for invalid data."""
        with caplog.at_level("WARNING"):
            DataValidator.validate_dataframe(invalid_ohlcv_data, "TEST_SYMBOL")

        # Should have warnings for NaN values
        assert len(caplog.records) > 0
        assert any("NaN values" in record.message for record in caplog.records)
