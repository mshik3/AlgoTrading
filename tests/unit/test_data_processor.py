"""
Unit tests for DataProcessor class.
Tests data processing pipeline, validation integration, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from data.processors import DataProcessor, DataValidator, DataCleaner


class TestDataProcessor:
    """Test DataProcessor functionality."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
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

    def test_processor_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        assert processor is not None
        assert hasattr(processor, "validator")
        assert hasattr(processor, "cleaner")

    def test_process_data_valid_data(self, sample_ohlcv_data):
        """Test processing of valid data."""
        processor = DataProcessor()
        result = processor.process_data(sample_ohlcv_data, "AAPL")

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_process_data_dirty_data(self, dirty_ohlcv_data):
        """Test processing of dirty data."""
        processor = DataProcessor()
        result = processor.process_data(dirty_ohlcv_data, "AAPL")

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Should handle NaN values
        assert not result.isnull().values.any()

    def test_process_data_empty_dataframe(self):
        """Test processing of empty DataFrame."""
        processor = DataProcessor()
        empty_df = pd.DataFrame()
        result = processor.process_data(empty_df, "AAPL")

        assert result is not None
        assert result.empty

    def test_process_data_none_dataframe(self):
        """Test processing of None DataFrame."""
        processor = DataProcessor()
        result = processor.process_data(None, "AAPL")

        assert result is None

    def test_process_data_missing_columns(self):
        """Test processing with missing required columns."""
        processor = DataProcessor()
        incomplete_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                # Missing other columns
            }
        )

        with pytest.raises(ValueError):
            processor.process_data(incomplete_data, "AAPL")

    def test_process_data_validation_failure(self):
        """Test processing when validation fails."""
        processor = DataProcessor()
        invalid_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": ["invalid", "data"],  # Non-numeric
                "Adj Close": [101.0, 102.0],
            }
        )

        with pytest.raises(ValueError):
            processor.process_data(invalid_data, "AAPL")

    def test_process_data_with_custom_config(self, sample_ohlcv_data):
        """Test processing with custom configuration."""
        config = {
            "remove_outliers": True,
            "fill_method": "forward",
            "min_data_points": 5,
        }
        processor = DataProcessor(**config)
        result = processor.process_data(sample_ohlcv_data, "AAPL")

        assert result is not None
        assert len(result) >= config["min_data_points"]

    def test_process_data_pipeline_steps(self, dirty_ohlcv_data):
        """Test that all pipeline steps are executed."""
        processor = DataProcessor()

        # Mock the validator and cleaner to track calls
        with patch.object(
            processor.validator, "validate_dataframe"
        ) as mock_validate, patch.object(
            processor.cleaner, "clean_dataframe"
        ) as mock_clean:

            mock_validate.return_value = (True, None)
            mock_clean.return_value = dirty_ohlcv_data.dropna()

            result = processor.process_data(dirty_ohlcv_data, "AAPL")

            # Verify both validation and cleaning were called
            mock_validate.assert_called_once()
            mock_clean.assert_called_once()

    def test_process_data_error_handling(self, sample_ohlcv_data):
        """Test error handling during processing."""
        processor = DataProcessor()

        # Mock validator to raise an exception
        with patch.object(
            processor.validator,
            "validate_dataframe",
            side_effect=Exception("Validation error"),
        ):
            with pytest.raises(Exception):
                processor.process_data(sample_ohlcv_data, "AAPL")

    def test_process_data_logging(self, sample_ohlcv_data, caplog):
        """Test that processing operations are properly logged."""
        processor = DataProcessor()
        with caplog.at_level("INFO"):
            processor.process_data(sample_ohlcv_data, "TEST_SYMBOL")

        # Should have logging messages
        assert len(caplog.records) > 0

    def test_process_data_performance(self, sample_ohlcv_data):
        """Test processing performance."""
        processor = DataProcessor()
        import time

        start_time = time.time()
        result = processor.process_data(sample_ohlcv_data, "AAPL")
        end_time = time.time()

        # Should complete within reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second
        assert result is not None

    def test_process_data_large_dataset(self):
        """Test processing of large dataset."""
        processor = DataProcessor()
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

        result = processor.process_data(large_data, "AAPL")
        assert result is not None
        assert len(result) > 0

    def test_process_data_preserve_index(self, sample_ohlcv_data):
        """Test that processing preserves the index."""
        processor = DataProcessor()
        result = processor.process_data(sample_ohlcv_data, "AAPL")

        assert result.index.equals(sample_ohlcv_data.index)

    def test_process_data_column_order(self, sample_ohlcv_data):
        """Test that processing preserves column order."""
        processor = DataProcessor()
        result = processor.process_data(sample_ohlcv_data, "AAPL")

        expected_columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        assert list(result.columns) == expected_columns

    def test_process_data_with_outliers(self):
        """Test processing with outlier data."""
        processor = DataProcessor()
        outlier_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [102.0, 103.0, 1000.0, 105.0, 106.0],  # Extreme outlier
                "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
                "Adj Close": [101.0, 102.0, 103.0, 104.0, 105.0],
            }
        )

        result = processor.process_data(outlier_data, "AAPL")
        assert result is not None
        assert len(result) > 0

    def test_process_data_multiple_symbols(self, sample_ohlcv_data):
        """Test processing multiple symbols."""
        processor = DataProcessor()
        symbols = ["AAPL", "MSFT", "GOOGL"]

        for symbol in symbols:
            result = processor.process_data(sample_ohlcv_data, symbol)
            assert result is not None
            assert len(result) > 0

    def test_process_data_config_validation(self):
        """Test configuration validation."""
        # Test with invalid config
        with pytest.raises(ValueError):
            DataProcessor(invalid_option="value")

    def test_process_data_memory_efficiency(self, sample_ohlcv_data):
        """Test memory efficiency of processing."""
        processor = DataProcessor()
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process data multiple times
        for _ in range(10):
            result = processor.process_data(sample_ohlcv_data, "AAPL")

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
