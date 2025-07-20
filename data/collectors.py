"""
Data collectors module for the algorithmic trading system.
Handles fetching data from various sources.
"""

import logging
import time
import random
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .storage import MarketData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User agents for rotation to avoid rate limiting
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
]


class YahooFinanceCollector:
    """Collector for Yahoo Finance data with rate limiting protection."""

    def __init__(self, delay_range=(1, 3), max_retries=3, backoff_factor=2):
        """
        Initialize the Yahoo Finance collector with rate limiting protection.

        Args:
            delay_range: Tuple of (min, max) seconds to wait between requests
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Multiplier for exponential backoff between retries
        """
        self.source_name = "Yahoo Finance"
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = self._create_session()
        self.last_request_time = 0

    def _create_session(self):
        """Create a configured requests session with retry strategy."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
            allowed_methods=["GET"],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_connections=10, pool_maxsize=20
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set random user agent
        session.headers.update(
            {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Cache-Control": "no-cache",
            }
        )

        return session

    def _rate_limit_delay(self):
        """Implement rate limiting with random delays."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        # Calculate delay needed
        min_delay, max_delay = self.delay_range
        required_delay = random.uniform(min_delay, max_delay)

        if time_since_last < required_delay:
            sleep_time = required_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                self._rate_limit_delay()
                return func(*args, **kwargs)
            except Exception as e:
                if (
                    "rate limit" in str(e).lower()
                    or "too many requests" in str(e).lower()
                ):
                    if attempt < self.max_retries:
                        backoff_time = (
                            self.backoff_factor**attempt
                        ) * random.uniform(5, 10)
                        logger.warning(
                            f"Rate limited, backing off for {backoff_time:.2f} seconds (attempt {attempt + 1}/{self.max_retries + 1})"
                        )
                        time.sleep(backoff_time)

                        # Rotate user agent for next attempt
                        self.session.headers.update(
                            {"User-Agent": random.choice(USER_AGENTS)}
                        )
                        continue
                    else:
                        logger.error(f"Max retries exceeded due to rate limiting")
                        raise
                else:
                    # Non-rate limiting error, re-raise immediately
                    raise

        return None

    def fetch_daily_data(self, symbol, start_date=None, end_date=None, period=None):
        """
        Fetch daily OHLCV data for a symbol from Yahoo Finance with rate limiting protection.

        Args:
            symbol (str): The ticker symbol to fetch data for
            start_date (datetime or str): Start date for data collection
            end_date (datetime or str): End date for data collection
            period (str): Optional period string (e.g., "5y" for 5 years) instead of dates

        Returns:
            pandas.DataFrame: DataFrame with the OHLCV data
        """

        def _fetch_data():
            logger.info(f"Fetching data for {symbol} from {self.source_name}")

            if not end_date:
                end_date_val = datetime.now()
            else:
                end_date_val = end_date

            if not start_date and not period:
                # Default to 5 years of data if no start_date or period provided
                start_date_val = end_date_val - timedelta(days=5 * 365)
            else:
                start_date_val = start_date

            # Create ticker with our configured session
            ticker = yf.Ticker(symbol, session=self.session)

            # Fetch data
            if period:
                data = ticker.history(period=period, progress=False, auto_adjust=True)
            else:
                data = ticker.history(
                    start=start_date_val,
                    end=end_date_val,
                    progress=False,
                    auto_adjust=True,
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

        try:
            return self._retry_with_backoff(_fetch_data)
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
