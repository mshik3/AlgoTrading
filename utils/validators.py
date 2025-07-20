"""
Input validation utilities for the algorithmic trading system.
Provides validation for user inputs, symbols, dates, and other parameters.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Union, Dict, Any

logger = logging.getLogger(__name__)

# Valid stock symbol pattern (1-5 letters, optionally followed by numbers/dots)
SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")

# Valid period patterns for data collection
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


def validate_symbols(symbols: Union[str, List[str]]) -> List[str]:
    """
    Validate and clean stock symbols.

    Args:
        symbols: Single symbol string or list of symbols

    Returns:
        List of validated symbols

    Raises:
        ValueError: If any symbol is invalid
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    elif symbols is None:
        return []
    elif not isinstance(symbols, list):
        raise ValueError(f"Symbols must be string or list, got {type(symbols)}")

    validated_symbols = []
    invalid_symbols = []

    for symbol in symbols:
        if not isinstance(symbol, str):
            invalid_symbols.append(f"{symbol} (not a string)")
            continue

        # Clean and uppercase the symbol
        cleaned_symbol = symbol.strip().upper()

        # Validate format
        if not cleaned_symbol:
            invalid_symbols.append("empty symbol")
            continue

        if len(cleaned_symbol) > 10:
            invalid_symbols.append(f"{cleaned_symbol} (too long)")
            continue

        if not SYMBOL_PATTERN.match(cleaned_symbol):
            invalid_symbols.append(f"{cleaned_symbol} (invalid format)")
            continue

        validated_symbols.append(cleaned_symbol)

    if invalid_symbols:
        raise ValueError(f"Invalid symbols found: {', '.join(invalid_symbols)}")

    # Remove duplicates while preserving order
    unique_symbols = []
    seen = set()
    for symbol in validated_symbols:
        if symbol not in seen:
            unique_symbols.append(symbol)
            seen.add(symbol)

    if len(unique_symbols) > 50:
        raise ValueError(f"Too many symbols provided: {len(unique_symbols)} (max 50)")

    logger.debug(f"Validated {len(unique_symbols)} symbols: {unique_symbols}")
    return unique_symbols


def validate_period(period: str) -> str:
    """
    Validate data collection period.

    Args:
        period: Period string (e.g., '5y', '1mo')

    Returns:
        Validated period string

    Raises:
        ValueError: If period is invalid
    """
    if not isinstance(period, str):
        raise ValueError(f"Period must be string, got {type(period)}")

    cleaned_period = period.strip().lower()

    if not cleaned_period:
        raise ValueError("Period cannot be empty")

    if cleaned_period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period '{period}'. Valid periods: {', '.join(sorted(VALID_PERIODS))}"
        )

    return cleaned_period


def validate_date_range(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> Dict[str, Optional[datetime]]:
    """
    Validate date range parameters.

    Args:
        start_date: Start date string in YYYY-MM-DD format
        end_date: End date string in YYYY-MM-DD format

    Returns:
        Dictionary with parsed datetime objects

    Raises:
        ValueError: If dates are invalid or inconsistent
    """
    result = {"start_date": None, "end_date": None}

    def parse_date(date_str: str, field_name: str) -> datetime:
        try:
            return datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Invalid {field_name} format '{date_str}'. Use YYYY-MM-DD format."
            )

    if start_date:
        if not isinstance(start_date, str):
            raise ValueError(f"Start date must be string, got {type(start_date)}")
        result["start_date"] = parse_date(start_date, "start_date")

    if end_date:
        if not isinstance(end_date, str):
            raise ValueError(f"End date must be string, got {type(end_date)}")
        result["end_date"] = parse_date(end_date, "end_date")

    # Validate date logic
    if result["start_date"] and result["end_date"]:
        if result["start_date"] >= result["end_date"]:
            raise ValueError("Start date must be before end date")

        # Check for reasonable date range (not too far back, not in future)
        today = datetime.now()
        max_history = today - timedelta(days=365 * 20)  # 20 years max

        if result["start_date"] < max_history:
            raise ValueError(f"Start date too far in the past (max 20 years)")

        if result["end_date"] > today:
            raise ValueError("End date cannot be in the future")

    return result


def validate_positive_integer(
    value: Union[str, int], field_name: str, min_value: int = 1, max_value: int = None
) -> int:
    """
    Validate positive integer input.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        min_value: Minimum allowed value (default: 1)
        max_value: Maximum allowed value (default: None)

    Returns:
        Validated integer value

    Raises:
        ValueError: If value is invalid
    """
    try:
        if isinstance(value, str):
            int_value = int(value.strip())
        elif isinstance(value, int):
            int_value = value
        else:
            raise ValueError(
                f"{field_name} must be integer or string, got {type(value)}"
            )
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid {field_name}: '{value}' is not a valid integer")

    if int_value < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}, got {int_value}")

    if max_value is not None and int_value > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}, got {int_value}")

    return int_value


def validate_float_range(
    value: Union[str, float],
    field_name: str,
    min_value: float = None,
    max_value: float = None,
) -> float:
    """
    Validate float input within specified range.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        min_value: Minimum allowed value (default: None)
        max_value: Maximum allowed value (default: None)

    Returns:
        Validated float value

    Raises:
        ValueError: If value is invalid
    """
    try:
        if isinstance(value, str):
            float_value = float(value.strip())
        elif isinstance(value, (int, float)):
            float_value = float(value)
        else:
            raise ValueError(
                f"{field_name} must be number or string, got {type(value)}"
            )
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid {field_name}: '{value}' is not a valid number")

    if min_value is not None and float_value < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}, got {float_value}")

    if max_value is not None and float_value > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}, got {float_value}")

    return float_value


def sanitize_string(
    value: str, max_length: int = 255, allow_empty: bool = False
) -> str:
    """
    Sanitize string input by removing dangerous characters and limiting length.

    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        allow_empty: Whether empty strings are allowed

    Returns:
        Sanitized string

    Raises:
        ValueError: If string is invalid
    """
    if not isinstance(value, str):
        raise ValueError(f"Expected string, got {type(value)}")

    # Strip whitespace
    sanitized = value.strip()

    if not sanitized and not allow_empty:
        raise ValueError("String cannot be empty")

    # Remove/escape potentially dangerous characters
    # Remove control characters but allow basic punctuation
    sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)

    # Limit length
    if len(sanitized) > max_length:
        raise ValueError(f"String too long: {len(sanitized)} chars (max {max_length})")

    return sanitized
