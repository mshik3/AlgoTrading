"""
Unit tests for DataCleaner class.
Tests data cleaning operations, outlier detection, and data transformation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from data.processors import DataCleaner


class TestDataCleaner:
    """Test DataCleaner functionality."""

    @pytest.fixture
    def dirty_ohlcv_data(self):
        """Create dirty OHLCV data for testing."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = {
            "Open": [
                100.0,
                101.0,
                np.nan,
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
    def outlier_data(self):
        """Create data with outliers for testing."""
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
        df = pd.DataFrame(data, index=dates)
        # Add outliers
        df.loc[df.index[2], "High"] = 1000.0  # Extreme outlier
        df.loc[df.index[5], "Volume"] = 0  # Zero volume
        return df

    def test_clean_dataframe_basic_cleaning(self, dirty_ohlcv_data):
        """Test basic data cleaning functionality."""
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_dataframe(dirty_ohlcv_data, "AAPL")

        # Should handle NaN values
        assert not cleaned_df.isnull().values.any()
        assert len(cleaned_df) > 0

    def test_clean_dataframe_remove_outliers(self, outlier_data):
        """Test outlier removal functionality."""
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_dataframe(outlier_data, "AAPL")

        # Should remove extreme outliers
        assert cleaned_df["High"].max() < 1000.0
        assert len(cleaned_df) < len(outlier_data)

    def test_clean_dataframe_empty_dataframe(self):
        """Test cleaning of empty DataFrame."""
        cleaner = DataCleaner()
        empty_df = pd.DataFrame()
        cleaned_df = cleaner.clean_dataframe(empty_df, "AAPL")

        assert cleaned_df.empty

    def test_clean_dataframe_none_dataframe(self):
        """Test cleaning of None DataFrame."""
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_dataframe(None, "AAPL")

        assert cleaned_df is None

    def test_clean_dataframe_missing_columns(self):
        """Test cleaning with missing required columns."""
        cleaner = DataCleaner()
        incomplete_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                # Missing other columns
            }
        )

        with pytest.raises(ValueError):
            cleaner.clean_dataframe(incomplete_data, "AAPL")

    def test_clean_dataframe_handle_nan_values(self, dirty_ohlcv_data):
        """Test handling of NaN values."""
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_dataframe(dirty_ohlcv_data, "AAPL")

        # Should fill or remove NaN values
        assert not cleaned_df.isnull().values.any()

    def test_clean_dataframe_handle_negative_values(self):
        """Test handling of negative values."""
        cleaner = DataCleaner()
        negative_data = pd.DataFrame(
            {
                "Open": [100.0, -101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000000, 1100000],
                "Adj Close": [101.0, 102.0],
            }
        )
        cleaned_df = cleaner.clean_dataframe(negative_data, "AAPL")

        # Should handle negative values appropriately
        assert len(cleaned_df) > 0

    def test_clean_dataframe_handle_zero_volume(self, outlier_data):
        """Test handling of zero volume."""
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_dataframe(outlier_data, "AAPL")

        # Should handle zero volume appropriately
        assert len(cleaned_df) > 0

    def test_clean_dataframe_preserve_data_integrity(self, dirty_ohlcv_data):
        """Test that cleaning preserves data integrity."""
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_dataframe(dirty_ohlcv_data, "AAPL")

        # Should maintain price relationships
        assert (cleaned_df["High"] >= cleaned_df["Low"]).all()
        assert (cleaned_df["High"] >= cleaned_df["Open"]).all()
        assert (cleaned_df["High"] >= cleaned_df["Close"]).all()

    def test_clean_dataframe_large_dataset(self):
        """Test cleaning of large dataset."""
        cleaner = DataCleaner()
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

        # Add some noise
        large_data.loc[large_data.index[::100], "Open"] = np.nan

        cleaned_df = cleaner.clean_dataframe(large_data, "AAPL")
        assert len(cleaned_df) > 0
        assert not cleaned_df.isnull().values.any()

    def test_clean_dataframe_duplicate_removal(self):
        """Test removal of duplicate rows."""
        cleaner = DataCleaner()
        duplicate_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 100.0],  # Duplicate
                "High": [102.0, 103.0, 102.0],  # Duplicate
                "Low": [99.0, 100.0, 99.0],  # Duplicate
                "Close": [101.0, 102.0, 101.0],  # Duplicate
                "Volume": [1000000, 1100000, 1000000],  # Duplicate
                "Adj Close": [101.0, 102.0, 101.0],  # Duplicate
            }
        )
        cleaned_df = cleaner.clean_dataframe(duplicate_data, "AAPL")

        # Should remove duplicates
        assert len(cleaned_df) < len(duplicate_data)

    def test_clean_dataframe_sort_by_date(self):
        """Test that data is sorted by date."""
        cleaner = DataCleaner()
        unsorted_dates = pd.date_range("2023-01-01", periods=5, freq="D")
        unsorted_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [102.0, 103.0, 104.0, 105.0, 106.0],
                "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
                "Adj Close": [101.0, 102.0, 103.0, 104.0, 105.0],
            },
            index=unsorted_dates[::-1],
        )  # Reverse order

        cleaned_df = cleaner.clean_dataframe(unsorted_data, "AAPL")

        # Should be sorted by date
        assert cleaned_df.index.is_monotonic_increasing

    def test_clean_dataframe_handle_extreme_outliers(self):
        """Test handling of extreme outliers."""
        cleaner = DataCleaner()
        extreme_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [102.0, 103.0, 104.0, 105.0, 106.0],
                "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
                "Adj Close": [101.0, 102.0, 103.0, 104.0, 105.0],
            }
        )
        # Add extreme outlier
        extreme_data.loc[extreme_data.index[2], "High"] = 1000000.0

        cleaned_df = cleaner.clean_dataframe(extreme_data, "AAPL")

        # Should handle extreme outlier
        assert cleaned_df["High"].max() < 1000000.0

    def test_clean_dataframe_logging(self, dirty_ohlcv_data, caplog):
        """Test that cleaning operations are properly logged."""
        cleaner = DataCleaner()
        with caplog.at_level("INFO"):
            cleaner.clean_dataframe(dirty_ohlcv_data, "TEST_SYMBOL")

        # Should have logging messages
        assert len(caplog.records) > 0

    def test_clean_dataframe_performance(self, dirty_ohlcv_data):
        """Test cleaning performance with timing."""
        cleaner = DataCleaner()
        import time

        start_time = time.time()
        cleaned_df = cleaner.clean_dataframe(dirty_ohlcv_data, "AAPL")
        end_time = time.time()

        # Should complete within reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second
        assert len(cleaned_df) > 0
