"""
Custom exceptions and error handling utilities for the algorithmic trading system.
"""

import logging
from typing import Optional, Any, Dict


class AlgoTradingException(Exception):
    """Base exception for all algorithmic trading system errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatabaseException(AlgoTradingException):
    """Exception raised for database-related errors."""

    def __init__(
        self,
        message: str,
        operation: str = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.operation = operation


class DataException(AlgoTradingException):
    """Exception raised for data-related errors."""

    def __init__(
        self, message: str, symbol: str = None, details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.symbol = symbol


class DataCollectionError(DataException):
    """Exception raised when data collection fails."""

    def __init__(
        self,
        message: str,
        symbol: str = None,
        source: str = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, symbol, details)
        self.source = source


class DataValidationError(DataException):
    """Exception raised when data validation fails."""

    def __init__(
        self,
        message: str,
        symbol: str = None,
        validation_type: str = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, symbol, details)
        self.validation_type = validation_type


class RateLimitExceeded(DataCollectionError):
    """Exception raised when API rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        symbol: str = None,
        source: str = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, symbol, source, details)
        self.retry_after = retry_after


class ConfigurationException(AlgoTradingException):
    """Exception raised for configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: str = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.config_key = config_key


class StrategyException(AlgoTradingException):
    """Exception raised for strategy-related errors."""

    def __init__(
        self,
        message: str,
        strategy_name: str = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.strategy_name = strategy_name


def log_exception(
    logger: logging.Logger, exception: Exception, context: str = None
) -> None:
    """
    Log an exception with additional context information.

    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context about where the exception occurred
    """
    if isinstance(exception, AlgoTradingException):
        # Custom exceptions have structured information
        log_message = (
            f"{context}: {exception.message}" if context else exception.message
        )
        # Filter out reserved keys to avoid conflicts
        extra_data = {
            "exception_type": type(exception).__name__,
            "details": exception.details,
        }
        # Add non-reserved attributes
        for key, value in getattr(exception, "__dict__", {}).items():
            if key not in ["message", "asctime"]:
                extra_data[key] = value
        logger.error(log_message, extra=extra_data)
    else:
        # Standard exceptions
        log_message = f"{context}: {str(exception)}" if context else str(exception)
        logger.error(log_message, extra={"exception_type": type(exception).__name__})


def safe_database_operation(logger: logging.Logger, operation_name: str):
    """
    Decorator for database operations that provides consistent error handling.

    Args:
        logger: Logger instance
        operation_name: Name of the database operation for logging
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            session = None
            try:
                # Try to extract session from arguments
                if args and hasattr(args[0], "query"):
                    session = args[0]
                elif "session" in kwargs:
                    session = kwargs["session"]

                return func(*args, **kwargs)

            except Exception as e:
                # Rollback transaction if session is available
                if session:
                    try:
                        session.rollback()
                        logger.debug(f"Rolled back transaction for {operation_name}")
                    except Exception as rollback_error:
                        logger.error(
                            f"Failed to rollback transaction: {rollback_error}"
                        )

                # Log the exception with context
                log_exception(
                    logger, e, f"Database operation '{operation_name}' failed"
                )

                # Re-raise as DatabaseException if not already a custom exception
                if not isinstance(e, AlgoTradingException):
                    raise DatabaseException(
                        f"Database operation '{operation_name}' failed: {str(e)}",
                        operation=operation_name,
                        details={"original_exception": type(e).__name__},
                    )
                else:
                    raise

        return wrapper

    return decorator
