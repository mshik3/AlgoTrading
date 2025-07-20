"""
Unit tests for input validation utilities.
"""

import pytest
from utils.validators import (
    validate_symbols,
    validate_period,
    validate_positive_integer,
    validate_float_range,
    sanitize_string,
)


class TestValidateSymbols:
    """Test symbol validation functionality."""

    def test_valid_single_symbol(self):
        result = validate_symbols("AAPL")
        assert result == ["AAPL"]

    def test_valid_multiple_symbols(self):
        result = validate_symbols(["aapl", "MSFT", "googl"])
        assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_empty_symbols(self):
        result = validate_symbols([])
        assert result == []

        result = validate_symbols(None)
        assert result == []

    def test_invalid_symbols(self):
        with pytest.raises(ValueError, match="Invalid symbols found"):
            validate_symbols([""])

        with pytest.raises(ValueError, match="Invalid symbols found"):
            validate_symbols(["TOOLONGSYMBOL"])

        with pytest.raises(ValueError, match="Invalid symbols found"):
            validate_symbols(["123"])

    def test_duplicate_removal(self):
        result = validate_symbols(["AAPL", "aapl", "AAPL"])
        assert result == ["AAPL"]

    def test_too_many_symbols(self):
        # Create 51 unique valid symbols using only letters
        many_symbols = []
        for i in range(51):
            # Create unique 3-letter symbols
            first = chr(65 + (i // 26))  # A-Z for first letter
            second = chr(65 + (i % 26))  # A-Z for second letter
            third = chr(65 + ((i + 1) % 26))  # A-Z for third letter
            symbol = first + second + third
            many_symbols.append(symbol)

        with pytest.raises(ValueError, match="Too many symbols"):
            validate_symbols(many_symbols)


class TestValidatePeriod:
    """Test period validation functionality."""

    def test_valid_periods(self):
        valid_periods = [
            "1d",
            "5d",
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y",
            "10y",
            "ytd",
            "max",
        ]
        for period in valid_periods:
            result = validate_period(period)
            assert result == period.lower()

    def test_case_insensitive(self):
        result = validate_period("5Y")
        assert result == "5y"

    def test_invalid_period(self):
        with pytest.raises(ValueError, match="Invalid period"):
            validate_period("invalid")

    def test_empty_period(self):
        with pytest.raises(ValueError, match="Period cannot be empty"):
            validate_period("")


class TestValidatePositiveInteger:
    """Test positive integer validation."""

    def test_valid_integer(self):
        result = validate_positive_integer(5, "test_field")
        assert result == 5

    def test_valid_string_integer(self):
        result = validate_positive_integer("10", "test_field")
        assert result == 10

    def test_minimum_value(self):
        with pytest.raises(ValueError, match="must be >= 5"):
            validate_positive_integer(3, "test_field", min_value=5)

    def test_maximum_value(self):
        with pytest.raises(ValueError, match="must be <= 10"):
            validate_positive_integer(15, "test_field", max_value=10)

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="not a valid integer"):
            validate_positive_integer("not_a_number", "test_field")


class TestValidateFloatRange:
    """Test float range validation."""

    def test_valid_float(self):
        result = validate_float_range(3.14, "test_field")
        assert result == 3.14

    def test_valid_string_float(self):
        result = validate_float_range("2.5", "test_field")
        assert result == 2.5

    def test_range_validation(self):
        with pytest.raises(ValueError, match="must be >= 0.0"):
            validate_float_range(-1.0, "test_field", min_value=0.0)

        with pytest.raises(ValueError, match="must be <= 1.0"):
            validate_float_range(2.0, "test_field", max_value=1.0)


class TestSanitizeString:
    """Test string sanitization."""

    def test_basic_sanitization(self):
        result = sanitize_string("  hello world  ")
        assert result == "hello world"

    def test_max_length(self):
        with pytest.raises(ValueError, match="String too long"):
            sanitize_string("a" * 256, max_length=255)

    def test_empty_string(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_string("", allow_empty=False)

        result = sanitize_string("", allow_empty=True)
        assert result == ""

    def test_control_character_removal(self):
        # Test that control characters are removed
        input_str = "hello\x00\x01world"
        result = sanitize_string(input_str)
        assert result == "helloworld"
