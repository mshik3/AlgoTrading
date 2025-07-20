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

        return True, ""

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
            return df.fillna(method="ffill").fillna(method="bfill")
        elif method == "bfill":
            return df.fillna(method="bfill").fillna(method="ffill")
        elif method == "interpolate":
            return (
                df.interpolate(method="linear")
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
        else:
            logger.warning(f"Unknown fill method: {method}, using 'ffill'")
            return df.fillna(method="ffill").fillna(method="bfill")

    @staticmethod
    def remove_outliers(df, column="Close", window=20, threshold=3):
        """
        Remove outliers from a DataFrame using rolling z-score.

        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            column (str): Column to check for outliers
            window (int): Rolling window size
            threshold (float): Z-score threshold for outlier detection

        Returns:
            pandas.DataFrame: DataFrame with outliers replaced
        """
        if df is None or df.empty or window >= len(df):
            return df

        result = df.copy()

        # Calculate rolling mean and std
        rolling_mean = df[column].rolling(window=window, center=True).mean()
        rolling_std = df[column].rolling(window=window, center=True).std()

        # Calculate z-scores
        z_scores = np.abs((df[column] - rolling_mean) / rolling_std)

        # Identify outliers
        outliers = z_scores > threshold

        if outliers.any():
            logger.info(f"Found {outliers.sum()} outliers in {column}")

            # Replace outliers with rolling mean
            result.loc[outliers, column] = rolling_mean[outliers]

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

    def __init__(self):
        """Initialize the data processor."""
        self.validator = DataValidator()
        self.cleaner = DataCleaner()

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
                return None

        # Process data
        if fill_missing:
            df = self.cleaner.fill_missing_values(df)

        if remove_outliers:
            for col in ["Open", "High", "Low", "Close", "Adj Close"]:
                df = self.cleaner.remove_outliers(df, column=col)

        if adjust_prices:
            df = self.cleaner.apply_adjustments(df)

        return df

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
