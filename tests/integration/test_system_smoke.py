"""
Smoke tests for basic system integration.
These tests validate that the core components work together.
"""

import pytest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
import os


class TestSystemSmoke:
    """Smoke tests for basic system functionality."""

    def test_imports_work(self):
        """Test that all critical modules can be imported."""
        # Test data modules
        from data.collectors import YahooFinanceCollector
        from data.processors import DataProcessor
        from data.storage import MarketData, Symbol

        # Test indicator modules
        from indicators.technical import TechnicalIndicators, calculate_rsi

        # Test strategy modules
        from strategies.base import BaseStrategy, StrategySignal
        from strategies.equity.mean_reversion import MeanReversionStrategy

        # Test utility modules
        from utils.config import get_env_var, load_environment
        from utils.validators import validate_symbols, validate_period
        from utils.exceptions import AlgoTradingException

        # If we get here, all imports worked
        assert True

    def test_configuration_system(self):
        """Test that configuration system works with environment variables."""
        from utils.config import get_env_var, validate_required_env_vars

        # Test with mock environment
        with patch.dict(
            os.environ,
            {"DB_HOST": "test_host", "DB_NAME": "test_db", "DB_USER": "test_user"},
        ):
            # Should be able to get values
            assert get_env_var("DB_HOST") == "test_host"
            assert get_env_var("NONEXISTENT", "default") == "default"

            # Validation should pass with required vars present
            result = validate_required_env_vars()
            assert result["valid"] is True

    def test_data_pipeline_basic(self):
        """Test basic data collection and processing pipeline."""
        from data.collectors import YahooFinanceCollector
        from data.processors import DataProcessor

        # Create mock data
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 99],
                "High": [102, 103, 101],
                "Low": [98, 99, 97],
                "Close": [101, 100, 98],
                "Volume": [1000000, 1200000, 800000],
                "Adj Close": [101, 100, 98],
            }
        )

        # Test data processor
        processor = DataProcessor()

        # Should not raise exceptions
        processed = processor.process_dataframe(mock_data, symbol="TEST")
        assert processed is not None
        assert len(processed) == 3

    def test_technical_indicators_pipeline(self):
        """Test technical indicators calculation pipeline."""
        from indicators.technical import TechnicalIndicators

        # Create sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        data = pd.DataFrame(
            {
                "Open": 100 + np.random.randn(30).cumsum() * 0.1,
                "High": 100 + np.random.randn(30).cumsum() * 0.1 + 1,
                "Low": 100 + np.random.randn(30).cumsum() * 0.1 - 1,
                "Close": 100 + np.random.randn(30).cumsum() * 0.1,
                "Volume": np.random.randint(1000000, 10000000, 30),
            },
            index=dates,
        )

        # Ensure logical price relationships
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        # Test indicators pipeline
        indicators = TechnicalIndicators(data)
        result = indicators.add_sma(10).add_rsi(14).add_bollinger_bands(20)

        # Check that indicators were added
        assert "SMA_10" in indicators.data.columns
        assert "RSI_14" in indicators.data.columns
        assert "BB_upper" in indicators.data.columns
        assert "BB_lower" in indicators.data.columns

    def test_strategy_basic_functionality(self):
        """Test basic strategy functionality."""
        from strategies.equity.mean_reversion import MeanReversionStrategy

        # Create sample data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        sample_data = {
            "TEST": pd.DataFrame(
                {
                    "Open": 100 + np.random.randn(50).cumsum() * 0.2,
                    "High": 102 + np.random.randn(50).cumsum() * 0.2,
                    "Low": 98 + np.random.randn(50).cumsum() * 0.2,
                    "Close": 100 + np.random.randn(50).cumsum() * 0.2,
                    "Volume": np.random.randint(1000000, 10000000, 50),
                },
                index=dates,
            )
        }

        # Ensure logical relationships
        for symbol in sample_data:
            df = sample_data[symbol]
            df["High"] = df[["Open", "High", "Close"]].max(axis=1)
            df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

        # Test strategy
        strategy = MeanReversionStrategy(["TEST"])
        signals = strategy.generate_signals(sample_data)

        # Should return a list (even if empty)
        assert isinstance(signals, list)

    def test_validation_pipeline(self):
        """Test input validation pipeline."""
        from utils.validators import validate_symbols, validate_period

        # Test symbol validation
        valid_symbols = validate_symbols(["AAPL", "MSFT", "test"])
        assert valid_symbols == ["AAPL", "MSFT", "TEST"]

        # Test period validation
        valid_period = validate_period("5Y")
        assert valid_period == "5y"

        # Test error handling
        with pytest.raises(ValueError):
            validate_symbols([""])

        with pytest.raises(ValueError):
            validate_period("invalid")

    @pytest.mark.slow
    def test_pipeline_main_function_mock(self):
        """Test main pipeline function with mocked dependencies."""
        from pipeline import main

        # Mock all external dependencies
        with patch("pipeline.run_data_collection") as mock_run, patch(
            "pipeline.load_environment"
        ) as mock_load_env, patch(
            "pipeline.validate_required_env_vars"
        ) as mock_validate, patch(
            "sys.argv",
            [
                "pipeline.py",
                "--task",
                "collect",
                "--symbols",
                "AAPL",
                "--period",
                "1mo",
            ],
        ):

            # Configure mocks
            mock_load_env.return_value = True
            mock_validate.return_value = {"valid": True, "missing_required": []}
            mock_run.return_value = True

            # Should complete without errors
            result = main()
            assert result == 0  # Success exit code

            # Verify mocks were called
            mock_load_env.assert_called_once()
            mock_validate.assert_called_once()
            mock_run.assert_called_once()


class TestErrorHandling:
    """Test error handling in system components."""

    def test_database_exception_handling(self):
        """Test custom database exception handling."""
        from utils.exceptions import DatabaseException, log_exception
        import logging

        # Create a test exception
        exc = DatabaseException("Test error", operation="test_op")
        assert exc.operation == "test_op"
        assert exc.message == "Test error"

        # Test logging (should not raise exceptions)
        logger = logging.getLogger("test")
        log_exception(logger, exc, "test context")

    def test_data_validation_errors(self):
        """Test data validation error handling."""
        from data.processors import DataValidator
        import pandas as pd

        validator = DataValidator()

        # Test with invalid data
        invalid_data = pd.DataFrame({"Close": ["not_a_number", "also_invalid"]})
        is_valid, error = validator.validate_dataframe(invalid_data)

        assert not is_valid
        assert "not numeric" in error or "Missing column" in error

    def test_configuration_error_handling(self):
        """Test configuration error handling."""
        from utils.config import get_env_var

        # Test required variable that doesn't exist
        with pytest.raises(ValueError, match="Required environment variable"):
            get_env_var("NONEXISTENT_REQUIRED_VAR", required=True)
