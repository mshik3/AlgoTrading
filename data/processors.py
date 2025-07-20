"""
Data processing module for the algorithmic trading system.
Handles data validation, cleaning, and transformation.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates market data for quality and completeness."""

    @staticmethod
    def validate_dataframe(df, symbol=None):
        """
        Validate a DataFrame of market data.

        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            symbol (str): Optional symbol for logging

        Returns:
            tuple: (bool, str) indicating if data is valid and any error message
        """
        if df is None or df.empty:
            return False, "Empty DataFrame"

        # Check for required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        for col in required_columns:
            if col not in df.columns:
                return False, f"Missing column: {col}"

        # Check data types
        for col in required_columns:
            if not np.issubdtype(df[col].dtype, np.number):
                return False, f"Column {col} is not numeric"

        # Check for NaN values
        if df.isnull().values.any():
            logger.warning(f"DataFrame for {symbol} contains NaN values")

        # Check for negative prices
        for col in ["Open", "High", "Low", "Close", "Adj Close"]:
            if (df[col] <= 0).any():
                logger.warning(
                    f"DataFrame for {symbol} contains non-positive {col} values"
                )

        # Check for negative volume
        if (df["Volume"] < 0).any():
            logger.warning(f"DataFrame for {symbol} contains negative Volume values")

        # Check OHLC relationships
        violations = (
            (df["High"] < df["Low"]).any()
            or (df["High"] < df["Open"]).any()
            or (df["High"] < df["Close"]).any()
            or (df["Low"] > df["Open"]).any()
            or (df["Low"] > df["Close"]).any()
        )

        if violations:
            logger.warning(
                f"DataFrame for {symbol} contains OHLC relationship violations"
            )

        return True, None

    @staticmethod
    def check_continuity(df, max_gap_days=5):
        """
        Check if the DataFrame has continuity in dates.

        Args:
            df (pandas.DataFrame): DataFrame with DatetimeIndex
            max_gap_days (int): Maximum allowed gap in trading days

        Returns:
            tuple: (bool, list) indicating continuity and list of gaps
        """
        if df is None or len(df) <= 1:
            return True, []

        # Make sure index is sorted
        df = df.sort_index()

        # Get all dates
        dates = df.index

        # Find gaps
        gaps = []
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i - 1]).days

            # Exclude weekends from gap calculation
            # (this is a simplified approach; doesn't account for holidays)
            week_delta = len(
                [
                    d
                    for d in range(1, delta)
                    if (dates[i - 1] + timedelta(days=d)).weekday() < 5
                ]
            )

            if week_delta > max_gap_days:
                gaps.append((dates[i - 1], dates[i], week_delta))

        return len(gaps) == 0, gaps


class DataCleaner:
    """Cleans market data to ensure quality and consistency."""

    @staticmethod
    def clean_dataframe(
        df, symbol=None, fill_missing=True, remove_outliers=True, adjust_prices=False
    ):
        """
        Clean a DataFrame of market data.

        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            symbol (str): Symbol for logging purposes
            fill_missing (bool): Whether to fill missing values
            remove_outliers (bool): Whether to remove outliers
            adjust_prices (bool): Whether to adjust OHLC based on Adj Close

        Returns:
            pandas.DataFrame: Cleaned DataFrame
        """
        if df is None or df.empty:
            logger.info(f"Empty DataFrame for {symbol}, returning as is")
            return df

        logger.info(f"Starting data cleaning for {symbol} with {len(df)} rows")

        result = df.copy()

        # Validate required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        missing_columns = [col for col in required_columns if col not in result.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Sort by date (index)
        if not result.index.is_monotonic_increasing:
            result = result.sort_index()
            logger.info(f"Sorted data by date for {symbol}")

        # Remove duplicates
        initial_rows = len(result)
        result = result.drop_duplicates()
        if len(result) < initial_rows:
            logger.info(
                f"Removed {initial_rows - len(result)} duplicate rows for {symbol}"
            )

        # Fill missing values
        if fill_missing:
            result = DataCleaner.fill_missing_values(result)
            logger.info(f"Filled missing values for {symbol}")

        # Remove outliers
        if remove_outliers:
            for col in ["Open", "High", "Low", "Close", "Adj Close"]:
                result = DataCleaner.remove_outliers(result, column=col, symbol=symbol)

        # Apply adjustments
        if adjust_prices:
            result = DataCleaner.apply_adjustments(result)
            logger.info(f"Applied price adjustments for {symbol}")

        logger.info(
            f"Completed data cleaning for {symbol}: {len(result)} rows remaining"
        )
        return result

    @staticmethod
    def fill_missing_values(df, method="ffill"):
        """
        Fill missing values in a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            method (str): Method to fill values ('ffill', 'bfill', or 'interpolate')

        Returns:
            pandas.DataFrame: DataFrame with filled values
        """
        if df is None or df.empty:
            return df

        if method == "ffill":
            return df.ffill().bfill()
        elif method == "bfill":
            return df.bfill().ffill()
        elif method == "interpolate":
            return df.interpolate(method="linear").ffill().bfill()
        else:
            logger.warning(f"Unknown fill method: {method}, using 'ffill'")
            return df.ffill().bfill()

    @staticmethod
    def remove_outliers(df, column="Close", window=20, threshold=3, symbol=None):
        """
        Remove outliers from a DataFrame using multiple detection methods.

        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            column (str): Column to check for outliers
            window (int): Rolling window size
            threshold (float): Z-score threshold for outlier detection
            symbol (str): Symbol for logging purposes

        Returns:
            pandas.DataFrame: DataFrame with outliers removed
        """
        if df is None or df.empty:
            return df

        result = df.copy()
        initial_rows = len(result)
        outliers_mask = pd.Series([False] * len(df), index=df.index)

        # Method 1: Simple statistical outlier detection (works for small datasets)
        if len(df) >= 3:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Detect extreme outliers (beyond 3x IQR)
            extreme_outliers = (df[column] < lower_bound - 2 * IQR) | (
                df[column] > upper_bound + 2 * IQR
            )
            outliers_mask |= extreme_outliers

        # Method 2: Rolling z-score (works for larger datasets)
        if len(df) >= window and window < len(df):
            # Calculate rolling mean and std
            rolling_mean = df[column].rolling(window=window, center=True).mean()
            rolling_std = df[column].rolling(window=window, center=True).std()

            # Calculate z-scores
            z_scores = np.abs((df[column] - rolling_mean) / rolling_std)

            # Identify outliers
            rolling_outliers = z_scores > threshold
            outliers_mask |= rolling_outliers

        # Method 3: Percentage-based outlier detection for extreme cases
        if len(df) >= 5:
            mean_val = df[column].mean()
            std_val = df[column].std()

            # Detect values that are more than 5 standard deviations from mean
            extreme_deviation = np.abs(df[column] - mean_val) > 5 * std_val
            outliers_mask |= extreme_deviation

        # Remove outlier rows
        if outliers_mask.any():
            logger.info(
                f"Found {outliers_mask.sum()} outliers in {column} for {symbol}"
            )
            result = result[~outliers_mask]

            if symbol:
                logger.info(
                    f"Removed {initial_rows - len(result)} outlier rows from {column} for {symbol}"
                )

        return result

    @staticmethod
    def apply_adjustments(df):
        """
        Apply adjustments to OHLC based on Adj Close.

        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data

        Returns:
            pandas.DataFrame: DataFrame with adjusted values
        """
        if df is None or df.empty:
            return df

        # Calculate adjustment ratio
        df["ratio"] = df["Adj Close"] / df["Close"]

        # Apply adjustment to OHLC
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col] * df["ratio"]

        # Drop ratio column
        df = df.drop("ratio", axis=1)

        return df


class DataProcessor:
    """Processes market data by validating and cleaning."""

    def __init__(self, **config):
        """Initialize the data processor."""
        self.validator = DataValidator()
        self.cleaner = DataCleaner()

        # Validate configuration
        valid_config_keys = {
            "remove_outliers",
            "fill_method",
            "min_data_points",
            "validate",
            "fill_missing",
            "adjust_prices",
        }

        for key in config:
            if key not in valid_config_keys:
                raise ValueError(f"Invalid configuration key: {key}")

        # Store configuration
        self.config = config

    def process_dataframe(
        self,
        df,
        symbol=None,
        validate=True,
        fill_missing=True,
        remove_outliers=True,
        adjust_prices=False,
    ):
        """
        Process a DataFrame of market data.

        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            symbol (str): Symbol for logging
            validate (bool): Whether to validate the data
            fill_missing (bool): Whether to fill missing values
            remove_outliers (bool): Whether to remove outliers
            adjust_prices (bool): Whether to adjust OHLC based on Adj Close

        Returns:
            pandas.DataFrame: Processed DataFrame
        """
        if df is None or df.empty:
            logger.warning(f"Empty DataFrame for {symbol}")
            return df

        # Validate data
        if validate:
            is_valid, error_msg = self.validator.validate_dataframe(df, symbol)
            if not is_valid:
                logger.error(f"Invalid data for {symbol}: {error_msg}")
                raise ValueError(f"Invalid data for {symbol}: {error_msg}")

        # Process data using clean_dataframe method
        df = self.cleaner.clean_dataframe(
            df,
            fill_missing=fill_missing,
            remove_outliers=remove_outliers,
            adjust_prices=adjust_prices,
        )

        logger.info(f"Successfully processed data for {symbol}: {len(df)} rows")
        return df

    def process_data(self, df, symbol=None, **kwargs):
        """
        Alias for process_dataframe for backward compatibility.

        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            symbol (str): Symbol for logging
            **kwargs: Additional arguments passed to process_dataframe

        Returns:
            pandas.DataFrame: Processed DataFrame
        """
        return self.process_dataframe(df, symbol, **kwargs)

    def check_data_quality(self, df, symbol=None, max_gap_days=5):
        """
        Comprehensive check of data quality.

        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            symbol (str): Symbol for logging
            max_gap_days (int): Maximum allowed gap in trading days

        Returns:
            dict: Dictionary with quality metrics
        """
        if df is None or df.empty:
            return {"valid": False, "error": "Empty DataFrame"}

        # Validate basic structure
        is_valid, error_msg = self.validator.validate_dataframe(df, symbol)
        if not is_valid:
            return {"valid": False, "error": error_msg}

        # Check date continuity
        has_continuity, gaps = self.validator.check_continuity(df, max_gap_days)

        # Calculate other quality metrics
        null_pct = df.isnull().mean().to_dict()
        total_rows = len(df)
        date_range = (df.index.min(), df.index.max())
        total_days = (date_range[1] - date_range[0]).days
        coverage_pct = total_rows / total_days if total_days > 0 else 0

        return {
            "valid": is_valid,
            "continuity": has_continuity,
            "gaps": gaps,
            "null_percentage": null_pct,
            "row_count": total_rows,
            "date_range": date_range,
            "coverage_percentage": coverage_pct,
        }
