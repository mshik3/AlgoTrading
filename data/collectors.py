"""
Data collectors module for the algorithmic trading system.
Handles fetching data from various sources.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from .storage import MarketData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YahooFinanceCollector:
    """Collector for Yahoo Finance data."""

    def __init__(self):
        """Initialize the Yahoo Finance collector."""
        self.source_name = "Yahoo Finance"

    def fetch_daily_data(self, symbol, start_date=None, end_date=None, period=None):
        """
        Fetch daily OHLCV data for a symbol from Yahoo Finance.

        Args:
            symbol (str): The ticker symbol to fetch data for
            start_date (datetime or str): Start date for data collection
            end_date (datetime or str): End date for data collection
            period (str): Optional period string (e.g., "5y" for 5 years) instead of dates

        Returns:
            pandas.DataFrame: DataFrame with the OHLCV data
        """
        try:
            logger.info(f"Fetching data for {symbol} from {self.source_name}")

            if not end_date:
                end_date = datetime.now()

            if not start_date and not period:
                # Default to 5 years of data if no start_date or period provided
                start_date = end_date - timedelta(days=5 * 365)

            # Fetch data
            if period:
                data = yf.download(symbol, period=period, progress=False)
            else:
                data = yf.download(
                    symbol, start=start_date, end=end_date, progress=False
                )

            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            # Handle MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                logger.info(f"MultiIndex detected, flattening DataFrame for {symbol}")
                # If we have a MultiIndex due to multiple tickers, select just this ticker
                if len(data.columns.levels) > 1 and symbol in data.columns.levels[1]:
                    data = data.xs(symbol, level=1, axis=1)
                else:
                    # If it's a different MultiIndex structure, flatten it
                    data.columns = [col[0] for col in data.columns]

            # Make sure all necessary columns are present
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            adj_close_column = (
                "Adj Close"
                if "Adj Close" in data.columns
                else "Adj_Close" if "Adj_Close" in data.columns else None
            )

            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                logger.error(f"Missing required columns for {symbol}: {missing}")
                return None

            # If Adj Close is missing, use Close
            if adj_close_column is None:
                logger.warning(
                    f"Adjusted Close column missing for {symbol}, using Close instead"
                )
                data["Adj Close"] = data["Close"]

            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def transform_to_market_data(self, symbol, df):
        """
        Transform a pandas DataFrame from Yahoo Finance to a list of MarketData objects.

        Args:
            symbol (str): The ticker symbol
            df (pandas.DataFrame): DataFrame with OHLCV data

        Returns:
            list: List of MarketData objects
        """
        if df is None or df.empty:
            return []

        market_data_list = []

        for index, row in df.iterrows():
            try:
                # Get adjusted close column
                adj_close_col = "Adj Close" if "Adj Close" in df.columns else "Close"

                market_data = MarketData(
                    symbol=symbol,
                    date=index.date(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                    adjusted_close=float(row[adj_close_col]),
                )
                market_data_list.append(market_data)
            except Exception as e:
                logger.error(
                    f"Error transforming data for {symbol} at {index}: {str(e)}"
                )
                continue

        return market_data_list

    def collect_and_transform(
        self, symbol, start_date=None, end_date=None, period=None
    ):
        """
        Collect data from Yahoo Finance and transform it to MarketData objects.

        Args:
            symbol (str): The ticker symbol
            start_date (datetime or str): Start date for data collection
            end_date (datetime or str): End date for data collection
            period (str): Optional period string instead of dates

        Returns:
            list: List of MarketData objects
        """
        df = self.fetch_daily_data(symbol, start_date, end_date, period)
        return self.transform_to_market_data(symbol, df)


class AlphaVantageCollector:
    """Collector for Alpha Vantage data."""

    def __init__(self, api_key=None):
        """
        Initialize the Alpha Vantage collector.

        Args:
            api_key (str): Alpha Vantage API key
        """
        self.api_key = api_key or "YOUR_API_KEY"
        self.source_name = "Alpha Vantage"

    # Implementation to be added in future phases
    def fetch_daily_data(self, symbol, start_date=None, end_date=None):
        """
        Fetch daily OHLCV data for a symbol from Alpha Vantage.

        Note: This is a placeholder for future implementation.
        """
        logger.info(f"Alpha Vantage collector not yet implemented")
        return None


def get_collector(source="yahoo"):
    """
    Factory function to create a data collector.

    Args:
        source (str): The data source to use ('yahoo' or 'alphavantage')

    Returns:
        object: A data collector instance
    """
    if source.lower() == "yahoo":
        return YahooFinanceCollector()
    elif source.lower() == "alphavantage":
        return AlphaVantageCollector()
    else:
        raise ValueError(f"Unsupported data source: {source}")
