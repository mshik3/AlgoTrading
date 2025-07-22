"""
Data collectors module for the algorithmic trading system.
Handles fetching data from Alpaca Markets API.
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union
import pandas as pd

# Alpaca SDK
try:
    from alpaca.data import (
        StockHistoricalDataClient,
        CryptoHistoricalDataClient,
        TimeFrame,
    )
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("Alpaca SDK not installed. Install with: pip install alpaca-py")

from .storage import MarketData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- PATCH: Add safe_date helper ---
def _safe_date(dt):
    from datetime import datetime, date

    if isinstance(dt, datetime):
        return dt.date()
    elif isinstance(dt, date):
        return dt
    else:
        raise TypeError(f"Expected datetime or date, got {type(dt)}")


class AlpacaDataCollector:
    """
    High-quality data collector using Alpaca Markets API.

    Advantages:
    - No rate limiting issues
    - Direct exchange data
    - Real-time capabilities
    - Perfect integration with Alpaca trading
    - Professional-grade reliability
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca data collector.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Whether to use paper trading data
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.stock_client = None
        self.crypto_client = None

        if ALPACA_AVAILABLE:
            self.stock_client = StockHistoricalDataClient(
                api_key=api_key, secret_key=secret_key
            )
            self.crypto_client = CryptoHistoricalDataClient(
                api_key=api_key, secret_key=secret_key
            )

        # Dynamic crypto symbols mapping - will be populated from Alpaca Assets API
        self.crypto_symbols = {}
        self._load_crypto_symbols()

        logger.info("Alpaca data collector initialized")

    def _load_crypto_symbols(self):
        """Load available crypto symbols from Alpaca Assets API."""
        try:
            from .alpaca_assets import get_available_crypto_symbols

            available_symbols = get_available_crypto_symbols()

            # Create mapping from our format to Alpaca format
            for alpaca_symbol in available_symbols:
                # Convert "BTC/USD" to "BTCUSD"
                our_symbol = alpaca_symbol.replace("/", "")
                self.crypto_symbols[our_symbol] = alpaca_symbol

            logger.info(
                f"Loaded {len(self.crypto_symbols)} crypto symbols from Alpaca Assets API"
            )

        except Exception as e:
            logger.warning(f"Failed to load crypto symbols from Assets API: {e}")
            # Fallback to a minimal set of known working symbols
            self.crypto_symbols = {
                "BTCUSD": "BTC/USD",
                "ETHUSD": "ETH/USD",
                "SOLUSD": "SOL/USD",
                "LINKUSD": "LINK/USD",
            }
            logger.info("Using fallback crypto symbols")

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a crypto asset."""
        return symbol in self.crypto_symbols

    def _get_alpaca_symbol(self, symbol: str) -> str:
        """Convert our symbol format to Alpaca's format."""
        if self._is_crypto_symbol(symbol):
            return self.crypto_symbols[symbol]
        return symbol

    def _get_client(self, symbol: str):
        """Get the appropriate client for the symbol type."""
        if self._is_crypto_symbol(symbol):
            return self.crypto_client
        return self.stock_client

    def fetch_daily_data(
        self,
        symbol: str,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        period: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data from Alpaca.

        Args:
            symbol: Stock or crypto symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            period: Period string (e.g., '5y', '1y', '6mo')

        Returns:
            pandas.DataFrame with OHLCV data
        """
        try:
            # Convert period to dates if provided
            if period:
                end_date = datetime.now()
                if period == "5y":
                    start_date = end_date - timedelta(days=5 * 365)
                elif period == "1y":
                    start_date = end_date - timedelta(days=365)
                elif period == "6mo":
                    start_date = end_date - timedelta(days=180)
                elif period == "3mo":
                    start_date = end_date - timedelta(days=90)
                elif period == "1mo":
                    start_date = end_date - timedelta(days=30)
                else:
                    # Default to 1 year for unknown periods
                    start_date = end_date - timedelta(days=365)

            # Ensure we have start and end dates
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=365)  # Default to 1 year

            # Convert string dates to datetime
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

            logger.info(
                f"Fetching {symbol} data from {_safe_date(start_date)} to {_safe_date(end_date)}"
            )

            if not ALPACA_AVAILABLE:
                logger.error("Alpaca SDK not available")
                return None

            # Get appropriate client and symbol format
            client = self._get_client(symbol)
            alpaca_symbol = self._get_alpaca_symbol(symbol)

            # Create appropriate request based on asset type
            if self._is_crypto_symbol(symbol):
                request = CryptoBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                )
                bars = client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                )
                bars = client.get_stock_bars(request)

            if bars and alpaca_symbol in bars:
                df = bars[alpaca_symbol].df

                # Rename columns to match expected format
                df = df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    }
                )

                # Add Adj Close column (same as Close for now)
                df["Adj Close"] = df["Close"]

                logger.info(f"Fetched {len(df)} days of data for {symbol}")
                return df
            else:
                logger.warning(f"No data returned for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def incremental_fetch_daily_data(
        self,
        session,
        symbol: str,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        period: Optional[str] = None,
        force_update: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily data incrementally, only downloading missing data.

        Args:
            session: SQLAlchemy session for database operations
            symbol: Stock symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            period: Period string (e.g., '5y' for 5 years)
            force_update: Whether to force update existing data

        Returns:
            DataFrame with complete OHLCV data or None if failed
        """
        from datetime import datetime, timedelta
        from .storage import (
            get_symbol_data_range,
            get_missing_date_ranges,
            get_cached_market_data,
            set_cached_market_data,
            invalidate_symbol_cache,
        )

        try:
            # Convert period to dates if provided
            if period:
                end_date = datetime.now()
                if period == "5y":
                    start_date = end_date - timedelta(days=5 * 365)
                elif period == "2y":
                    start_date = end_date - timedelta(days=2 * 365)
                elif period == "1y":
                    start_date = end_date - timedelta(days=365)
                elif period == "6mo":
                    start_date = end_date - timedelta(days=180)
                elif period == "3mo":
                    start_date = end_date - timedelta(days=90)
                elif period == "1mo":
                    start_date = end_date - timedelta(days=30)
                else:
                    start_date = end_date - timedelta(days=365)

            # Ensure we have start and end dates
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=365)

            # Convert string dates to datetime
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

            # Check cache first
            cached_data = get_cached_market_data(
                symbol, _safe_date(start_date), _safe_date(end_date)
            )
            if cached_data and not force_update:
                logger.info(f"Using cached data for {symbol}")
                return self._convert_market_data_to_dataframe(cached_data)

            # Check if we need to fetch any data
            if not force_update:
                data_range = get_symbol_data_range(session, symbol)

                if data_range["has_data"]:
                    # Check if we have all the data we need
                    if data_range["earliest_date"] <= _safe_date(
                        start_date
                    ) and data_range["latest_date"] >= _safe_date(end_date):
                        logger.info(
                            f"All data for {symbol} already available in database"
                        )
                        # Get data from database
                        from .storage import get_market_data

                        db_data = get_market_data(
                            session,
                            symbol,
                            _safe_date(start_date),
                            _safe_date(end_date),
                        )
                        if db_data:
                            df = self._convert_market_data_to_dataframe(db_data)
                            # Cache the result
                            set_cached_market_data(
                                symbol,
                                _safe_date(start_date),
                                _safe_date(end_date),
                                db_data,
                            )
                            return df

            # Calculate missing date ranges
            missing_ranges = get_missing_date_ranges(
                session, symbol, _safe_date(start_date), _safe_date(end_date)
            )

            if not missing_ranges and not force_update:
                logger.info(f"No missing data for {symbol}")
                # Get existing data from database
                from .storage import get_market_data

                db_data = get_market_data(
                    session, symbol, _safe_date(start_date), _safe_date(end_date)
                )
                if db_data:
                    df = self._convert_market_data_to_dataframe(db_data)
                    set_cached_market_data(
                        symbol, _safe_date(start_date), _safe_date(end_date), db_data
                    )
                    return df

            # Fetch missing data
            all_fetched_data = []
            for range_start, range_end in missing_ranges:
                logger.info(
                    f"Fetching missing data for {symbol}: {_safe_date(range_start)} to {_safe_date(range_end)}"
                )

                try:
                    # Convert date objects to datetime objects for fetch_daily_data
                    range_start_dt = datetime.combine(range_start, datetime.min.time())
                    range_end_dt = datetime.combine(range_end, datetime.min.time())

                    # Fetch data for this range
                    range_data = self.fetch_daily_data(
                        symbol, start_date=range_start_dt, end_date=range_end_dt
                    )

                    if range_data is not None and not range_data.empty:
                        all_fetched_data.append(range_data)
                        logger.info(
                            f"✓ Fetched {len(range_data)} records for {symbol} ({range_start} to {range_end})"
                        )
                    else:
                        logger.warning(
                            f"No data fetched for {symbol} ({range_start} to {range_end})"
                        )

                except Exception as e:
                    logger.error(
                        f"Error fetching data for {symbol} ({range_start} to {range_end}): {str(e)}"
                    )

            # Combine all fetched data
            if all_fetched_data:
                combined_fetched = pd.concat(all_fetched_data, ignore_index=True)
                combined_fetched = combined_fetched.drop_duplicates(
                    subset=["Date"]
                ).sort_values("Date")

                # Save to database
                from .storage import save_market_data

                # Convert to MarketData objects and save
                market_data_objects = self.transform_to_market_data(
                    symbol, combined_fetched
                )
                if market_data_objects:
                    new_records, updated_records = save_market_data(
                        session, market_data_objects
                    )
                    logger.info(
                        f"Saved {new_records} new, {updated_records} updated records for {symbol}"
                    )

                    # Invalidate cache for this symbol
                    invalidate_symbol_cache(symbol)

            # Get complete dataset from database
            from .storage import get_market_data

            complete_data = get_market_data(
                session, symbol, _safe_date(start_date), _safe_date(end_date)
            )

            if complete_data:
                df = self._convert_market_data_to_dataframe(complete_data)
                # Cache the complete result
                set_cached_market_data(
                    symbol, _safe_date(start_date), _safe_date(end_date), complete_data
                )
                return df
            else:
                logger.warning(f"No complete data available for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error in incremental fetch for {symbol}: {str(e)}")
            return None

    def _convert_market_data_to_dataframe(self, market_data_objects):
        """
        Convert MarketData objects to pandas DataFrame.

        Args:
            market_data_objects: List of MarketData objects

        Returns:
            DataFrame with OHLCV data
        """
        if not market_data_objects:
            return pd.DataFrame()

        data = []
        for record in market_data_objects:
            data.append(
                {
                    "Date": record.date,
                    "Open": float(record.open_price),
                    "High": float(record.high_price),
                    "Low": float(record.low_price),
                    "Close": float(record.close_price),
                    "Volume": int(record.volume),
                    "Adj Close": float(record.adj_close),
                }
            )

        df = pd.DataFrame(data)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        return df

    def incremental_collect_and_transform(
        self,
        session,
        symbol: str,
        start_date=None,
        end_date=None,
        period=None,
        force_update=False,
    ) -> List[MarketData]:
        """
        Collect and transform data incrementally, only fetching missing data.

        Args:
            session: SQLAlchemy session for database operations
            symbol: Stock or crypto symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            period: Period string (optional)
            force_update: Whether to force update existing data

        Returns:
            List of MarketData objects
        """
        # Use the incremental fetch method from alpaca_collector
        df = self.incremental_fetch_daily_data(
            session=session,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            force_update=force_update,
        )

        if df is not None and not df.empty:
            return self.transform_to_market_data(symbol, df)
        else:
            return []

    def transform_to_market_data(
        self, symbol: str, df: pd.DataFrame
    ) -> List[MarketData]:
        """
        Transform DataFrame to MarketData objects.

        Args:
            symbol: Stock or crypto symbol
            df: DataFrame with OHLCV data

        Returns:
            List of MarketData objects
        """
        if df is None or df.empty:
            return []

        market_data_list = []
        for date, row in df.iterrows():
            market_data = MarketData(
                symbol=symbol,
                date=date.date(),
                open_price=float(row["Open"]),
                high_price=float(row["High"]),
                low_price=float(row["Low"]),
                close_price=float(row["Close"]),
                volume=int(row["Volume"]),
                adj_close=float(row["Adj Close"]),
            )
            market_data_list.append(market_data)

        return market_data_list

    def collect_and_transform(
        self, symbol: str, start_date=None, end_date=None, period=None
    ) -> List[MarketData]:
        """
        Collect and transform data in one step.

        Args:
            symbol: Stock or crypto symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            period: Period string (optional)

        Returns:
            List of MarketData objects
        """
        df = self.fetch_daily_data(symbol, start_date, end_date, period)
        return self.transform_to_market_data(symbol, df)


def get_collector(source: str = "alpaca", **kwargs) -> AlpacaDataCollector:
    """
    Get a data collector instance.

    Args:
        source: Data source (only 'alpaca' supported)
        **kwargs: Additional arguments for collector initialization

    Returns:
        Data collector instance
    """
    if source.lower() == "alpaca":
        api_key = kwargs.get("api_key")
        secret_key = kwargs.get("secret_key")
        paper = kwargs.get("paper", True)

        if not api_key or not secret_key:
            raise ValueError("Alpaca API key and secret key are required")

        return AlpacaDataCollector(api_key=api_key, secret_key=secret_key, paper=paper)
    else:
        raise ValueError(
            f"Unsupported data source: {source}. Only 'alpaca' is supported."
        )
