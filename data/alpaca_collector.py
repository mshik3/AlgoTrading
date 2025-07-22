#!/usr/bin/env python3
"""
Alpaca Markets Data Collector

This module provides reliable market data collection using Alpaca's API.
Alpaca offers free real-time and historical data with generous rate limits,
making it perfect for algorithmic trading systems.

Key Features:
- Real-time and historical data
- 200 requests/minute (free tier)
- Direct exchange data (not scraped)
- Perfect integration with Alpaca trading
- No rate limiting issues
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# --- PATCH: Add safe_date helper ---
def _safe_date(dt):
    """
    Safely convert datetime to date object.

    Args:
        dt: datetime, date, or compatible object

    Returns:
        date object

    Raises:
        TypeError: If dt is not a datetime or date object
    """
    from datetime import datetime, date

    if isinstance(dt, datetime):
        return dt.date()
    elif isinstance(dt, date):
        return dt
    else:
        raise TypeError(f"Expected datetime or date, got {type(dt)}")


# Alpaca SDK
try:
    from alpaca.data import (
        StockHistoricalDataClient,
        CryptoHistoricalDataClient,
        TimeFrame,
    )
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    import alpaca

    ALPACA_AVAILABLE = True
    ALPACA_VERSION = alpaca.__version__
    logger.info(f"Alpaca SDK version: {ALPACA_VERSION}")
except ImportError:
    ALPACA_AVAILABLE = False
    ALPACA_VERSION = None
    logging.warning("Alpaca SDK not installed. Install with: pip install alpaca-py")


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca data collection."""

    api_key: str
    secret_key: str
    base_url: str = "https://data.alpaca.markets"
    paper: bool = True  # Use paper trading data


class AlpacaDataCollector:
    """
    High-quality data collector using Alpaca Markets API.

    Advantages over Yahoo Finance:
    - No rate limiting issues
    - Direct exchange data
    - Real-time capabilities
    - Perfect integration with Alpaca trading
    - Professional-grade reliability
    """

    def __init__(self, config: Optional[AlpacaConfig] = None):
        """
        Initialize Alpaca data collector.

        Args:
            config: Alpaca configuration (if None, loads from environment)
        """
        if config is None:
            config = self._load_config_from_env()

        self.config = config
        self.stock_client = None
        self.crypto_client = None

        if ALPACA_AVAILABLE:
            self.stock_client = StockHistoricalDataClient(
                api_key=config.api_key, secret_key=config.secret_key
            )
            self.crypto_client = CryptoHistoricalDataClient(
                api_key=config.api_key, secret_key=config.secret_key
            )

        # Rate limiting tracking
        self.request_count = 0
        self.last_reset = datetime.now()
        self.max_requests_per_minute = 200  # Free tier limit

        # Dynamic crypto symbols mapping - will be populated from Alpaca Assets API
        self.crypto_symbols = {}
        self._load_crypto_symbols()

    def _load_config_from_env(self) -> AlpacaConfig:
        """Load Alpaca configuration from environment variables."""
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or provide AlpacaConfig object."
            )

        # Validate API key format (basic validation)
        if len(api_key) < 10 or len(secret_key) < 10:
            raise ValueError(
                "Invalid API key format. Keys should be at least 10 characters long."
            )

        return AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,  # Use paper trading for safety
        )

    def _load_crypto_symbols(self):
        """Load available crypto symbols from Alpaca Assets API."""
        try:
            from .alpaca_assets import get_available_crypto_symbols
            from .crypto_universe import get_alpaca_crypto_symbols

            # Try to get symbols from our crypto universe first
            try:
                alpaca_symbols = get_alpaca_crypto_symbols()
                for alpaca_symbol in alpaca_symbols:
                    # Convert "BTC/USD" to "BTCUSD"
                    our_symbol = alpaca_symbol.replace("/", "")
                    self.crypto_symbols[our_symbol] = alpaca_symbol

                logger.info(
                    f"Loaded {len(self.crypto_symbols)} crypto symbols from crypto universe"
                )

            except Exception as e:
                logger.warning(f"Failed to load from crypto universe: {e}")
                # Fallback to Alpaca Assets API
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
            # Fallback to only symbols that are actually available in Alpaca's API
            self.crypto_symbols = {
                "BTCUSD": "BTC/USD",
                "ETHUSD": "ETH/USD",
                "SOLUSD": "SOL/USD",
                "LINKUSD": "LINK/USD",
                "DOTUSD": "DOT/USD",
                "AVAXUSD": "AVAX/USD",
                "UNIUSD": "UNI/USD",
                "LTCUSD": "LTC/USD",
                "BCHUSD": "BCH/USD",
                "DOGEUSD": "DOGE/USD",
                "SHIBUSD": "SHIB/USD",
                "XRPUSD": "XRP/USD",
                "AAVEUSD": "AAVE/USD",
                "BATUSD": "BAT/USD",
                "CRVUSD": "CRV/USD",
                "GRTUSD": "GRT/USD",
                "MKRUSD": "MKR/USD",
                "PEPEUSD": "PEPE/USD",
                "SUSHIUSD": "SUSHI/USD",
                "XTZUSD": "XTZ/USD",
                "YFIUSD": "YFI/USD",
            }
            logger.info(
                f"Using fallback crypto symbols: {list(self.crypto_symbols.keys())}"
            )

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a crypto asset."""
        return symbol in self.crypto_symbols

    def _get_alpaca_symbol(self, symbol: str) -> str:
        """Convert our symbol format to Alpaca's format."""
        if self._is_crypto_symbol(symbol):
            return self.crypto_symbols[symbol]

        # Normalize stock symbols for Alpaca API compatibility
        from utils.symbol_normalization import normalize_symbol_for_alpaca

        return normalize_symbol_for_alpaca(symbol)

    def get_available_crypto_symbols(self) -> list:
        """Get list of available crypto symbols in our format."""
        return list(self.crypto_symbols.keys())

    def is_crypto_symbol_available(self, symbol: str) -> bool:
        """Check if a crypto symbol is available in Alpaca's API."""
        return symbol in self.crypto_symbols

    def suggest_alternative_crypto_symbols(
        self, requested_symbol: str, limit: int = 5
    ) -> list:
        """Suggest alternative crypto symbols when the requested one is not available."""
        available_symbols = self.get_available_crypto_symbols()

        # Popular alternatives to suggest
        popular_alternatives = ["BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "DOTUSD"]

        suggestions = []
        for alt in popular_alternatives:
            if alt in available_symbols and alt != requested_symbol:
                suggestions.append(alt)
                if len(suggestions) >= limit:
                    break

        return suggestions

    def _get_client(self, symbol: str):
        """Get the appropriate client for the symbol type."""
        if self._is_crypto_symbol(symbol):
            return self.crypto_client
        return self.stock_client

    def _validate_bars_response(self, bars, symbol: str, alpaca_symbol: str) -> bool:
        """
        Validate the bars response structure based on Alpaca SDK version.

        Args:
            bars: The bars response from Alpaca API
            symbol: Original symbol
            alpaca_symbol: Alpaca-formatted symbol

        Returns:
            True if response is valid, False otherwise
        """
        if not bars:
            logger.error(f"bars response is None for {symbol} ({alpaca_symbol})")
            return False

        # Log the response structure for debugging
        logger.info(f"bars type: {type(bars)}")
        logger.info(
            f"bars attributes: {[attr for attr in dir(bars) if not attr.startswith('_')]}"
        )

        # Check if bars has a data attribute
        if not hasattr(bars, "data"):
            logger.error(
                f"bars object missing 'data' attribute for {symbol} ({alpaca_symbol})"
            )
            return False

        if bars.data is None:
            logger.error(f"bars.data is None for {symbol} ({alpaca_symbol})")
            return False

        # Check if bars.data is a dictionary-like object
        if not hasattr(bars.data, "items"):
            logger.error(
                f"bars.data is not a dictionary-like object for {symbol} ({alpaca_symbol}). Type: {type(bars.data)}"
            )
            return False

        # Check if the symbol exists in the data
        if alpaca_symbol not in bars.data:
            logger.warning(
                f"Symbol {alpaca_symbol} not found in bars.data. Available keys: {list(bars.data.keys())}"
            )
            return False

        return True

    def _check_rate_limit(self):
        """Check and manage rate limits."""
        now = datetime.now()

        # Reset counter every minute
        if (now - self.last_reset).seconds >= 60:
            self.request_count = 0
            self.last_reset = now

        # Check if we're approaching the limit
        if self.request_count >= self.max_requests_per_minute * 0.9:
            wait_time = 60 - (now - self.last_reset).seconds
            if wait_time > 0:
                logger.warning(f"Rate limit approaching, waiting {wait_time} seconds")
                import time

                time.sleep(wait_time)
                self.request_count = 0
                self.last_reset = datetime.now()

    def validate_symbol_availability(self, symbol: str) -> bool:
        """
        Validate if a symbol is available for data collection.

        Args:
            symbol: Symbol to validate

        Returns:
            True if symbol is available, False otherwise
        """
        # Check if it's a crypto symbol first
        if self._is_crypto_symbol(symbol):
            if not self.is_crypto_symbol_available(symbol):
                logger.warning(f"Crypto symbol {symbol} is not available in Alpaca API")
                suggestions = self.suggest_alternative_crypto_symbols(symbol)
                if suggestions:
                    logger.info(f"Available alternatives: {suggestions}")
                return False
            return True

        # For symbols ending with USD that are not in crypto_symbols, they might be unavailable crypto
        if symbol.endswith("USD") and not self._is_crypto_symbol(symbol):
            logger.warning(
                f"Symbol {symbol} appears to be crypto but is not available in Alpaca API"
            )
            return False

        # Check if symbol is known to be unavailable in Alpaca
        from utils.symbol_normalization import (
            normalize_symbol_for_alpaca,
            is_symbol_normalized,
            is_symbol_available_for_alpaca,
            get_fallback_symbol,
        )

        if not is_symbol_available_for_alpaca(symbol):
            fallback_symbol = get_fallback_symbol(symbol)
            if fallback_symbol:
                logger.warning(
                    f"Symbol {symbol} is not available in Alpaca API. "
                    f"Will use fallback: {fallback_symbol}"
                )
                # Return True to allow the fetch to proceed with fallback
                return True
            else:
                logger.error(
                    f"Symbol {symbol} is not available in Alpaca API and no fallback is configured"
                )
                return False

        # For non-crypto symbols, normalize and assume they're available (stocks/ETFs)
        if is_symbol_normalized(symbol):
            normalized = normalize_symbol_for_alpaca(symbol)
            logger.debug(f"Validating normalized symbol: {symbol} -> {normalized}")

        return True

    def fetch_daily_data(
        self,
        symbol: str,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        period: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'SPY', 'AAPL')
            start_date: Start date for data collection
            end_date: End date for data collection
            period: Period string (e.g., '5y' for 5 years) - alternative to dates

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Validate symbol availability first
            if not self.validate_symbol_availability(symbol):
                return None

            self._check_rate_limit()

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

            # Log symbol normalization if it occurred
            from utils.symbol_normalization import is_symbol_normalized

            if is_symbol_normalized(symbol):
                logger.info(
                    f"Normalized symbol for data fetch: {symbol} -> {alpaca_symbol}"
                )

            # Create appropriate request based on asset type
            if self._is_crypto_symbol(symbol):
                request = CryptoBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                )
                logger.info(f"Crypto request created for {symbol} ({alpaca_symbol})")
                bars = client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                )
                logger.info(f"Stock request created for {symbol} ({alpaca_symbol})")
                bars = client.get_stock_bars(request)

                # Validate the bars response structure
            if not self._validate_bars_response(bars, symbol, alpaca_symbol):
                # Provide better error reporting for crypto symbols
                if self._is_crypto_symbol(symbol):
                    if not self.is_crypto_symbol_available(symbol):
                        suggestions = self.suggest_alternative_crypto_symbols(symbol)
                        logger.error(
                            f"Crypto symbol {symbol} is not available in Alpaca's API. "
                            f"Available crypto symbols: {', '.join(self.get_available_crypto_symbols())}"
                        )
                        if suggestions:
                            logger.info(
                                f"Suggested alternatives: {', '.join(suggestions)}"
                            )
                    else:
                        logger.error(
                            f"Crypto symbol {symbol} is mapped but returned no data from API"
                        )
                else:
                    logger.error(f"Stock symbol {symbol} returned no data from API")
                return None

            # Try to access the data safely
            try:
                # Access data by symbol
                if alpaca_symbol in bars.data:
                    bar_list = bars.data[alpaca_symbol]
                    if bar_list and len(bar_list) > 0:
                        # Convert list of Bar objects to DataFrame
                        data_list = []
                        for bar in bar_list:
                            if hasattr(bar, "__dict__"):
                                data_list.append(bar.__dict__)
                            else:
                                data_list.append(bar)

                        df = pd.DataFrame(data_list)
                        logger.info(
                            f"Found data for {symbol} ({alpaca_symbol}) with {len(df)} rows"
                        )
                    else:
                        logger.warning(
                            f"Empty data list for {symbol} ({alpaca_symbol})"
                        )
                        return None
                else:
                    logger.warning(f"Symbol {alpaca_symbol} not found in bars data")
                    return None

            except Exception as e:
                logger.error(
                    f"Error accessing data for {symbol} ({alpaca_symbol}): {str(e)}"
                )
                return None

            if df.empty:
                logger.warning(f"Empty DataFrame for {symbol} ({alpaca_symbol})")
                return None

            # Convert timestamp to Date column and set as index
            if "timestamp" in df.columns:
                df["Date"] = pd.to_datetime(df["timestamp"]).dt.date
                df.set_index("Date", inplace=True)

            # Rename columns to match expected format (proper case for consistency)
            df = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )

            # Add adj_close column (use close for now, Alpaca doesn't provide adjusted)
            df["Adj Close"] = df["Close"]

            # Ensure all required columns are present
            required_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                logger.error(f"Missing required columns for {symbol}: {missing}")
                return None

            self.request_count += 1
            logger.info(f"✓ Successfully fetched {len(df)} days of data for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Stock or crypto symbol

        Returns:
            Current price or None if failed
        """
        try:
            self._check_rate_limit()

            # Get appropriate client and symbol format
            client = self._get_client(symbol)
            alpaca_symbol = self._get_alpaca_symbol(symbol)

            # Fetch today's data to get current price
            today = datetime.now()

            # Create appropriate request based on asset type
            if self._is_crypto_symbol(symbol):
                request = CryptoBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=TimeFrame.Day,
                    start=today
                    - timedelta(days=5),  # Get last 5 days to ensure we have today
                    end=today,
                )
                bars = client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=TimeFrame.Day,
                    start=today
                    - timedelta(days=5),  # Get last 5 days to ensure we have today
                    end=today,
                )
                bars = client.get_stock_bars(request)

            # Validate the bars response structure
            if not self._validate_bars_response(bars, symbol, alpaca_symbol):
                return None

            # Get the latest bar data
            try:
                bar_list = bars.data[alpaca_symbol]
                if bar_list and len(bar_list) > 0:
                    latest_bar = bar_list[-1]
                    if hasattr(latest_bar, "close"):
                        self.request_count += 1
                        return float(latest_bar.close)
                    else:
                        logger.error(
                            f"Latest bar missing 'close' attribute for {symbol}"
                        )
                        return None
                else:
                    logger.warning(f"No bar data available for {symbol}")
                    return None
            except Exception as e:
                logger.error(f"Error accessing bar data for {symbol}: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None

    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols efficiently.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbol -> current price
        """
        prices = {}

        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price

        return prices

    def fetch_large_dataset_batch(
        self, symbols: List[str], period: str = "1y", batch_size: int = 50
    ) -> pd.DataFrame:
        """
        Fetch data for large datasets (920 assets) using batch processing.

        Args:
            symbols: List of symbols to fetch
            period: Time period for data
            batch_size: Number of symbols to process in each batch

        Returns:
            DataFrame with combined data for all symbols
        """
        import time

        logger.info(
            f"Starting batch processing for {len(symbols)} symbols with batch size {batch_size}"
        )

        all_data = []
        total_batches = (len(symbols) + batch_size - 1) // batch_size

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)"
            )

            for symbol in batch:
                try:
                    data = self.fetch_daily_data(symbol, period=period)
                    if data is not None and not data.empty:
                        data["symbol"] = symbol
                        all_data.append(data)
                        logger.debug(f"✓ Added data for {symbol}")
                    else:
                        logger.warning(f"✗ No data for {symbol}")
                except Exception as e:
                    logger.error(f"✗ Error fetching {symbol}: {e}")
                    continue

            # Rate limiting between batches
            if batch_num < total_batches:
                logger.info(f"Rate limiting: waiting 1 second before next batch")
                time.sleep(1)

        if not all_data:
            logger.warning("No data collected from any symbols")
            return pd.DataFrame()

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(
            f"✓ Successfully collected data for {len(combined_data['symbol'].unique())} symbols"
        )

        return combined_data

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

            # 🚀 OPTIMIZED: Use intelligent strategy to minimize API calls
            # Analyze gaps to determine the most efficient fetching approach
            strategy_analysis = self._analyze_gaps_for_optimal_strategy(
                missing_ranges, start_date, end_date
            )

            logger.info(
                f"📊 Strategy for {symbol}: {strategy_analysis['strategy']} - {strategy_analysis['reason']}"
            )
            if strategy_analysis.get("api_calls_saved", 0) > 0:
                logger.info(
                    f"💡 Optimization will save {strategy_analysis['api_calls_saved']} API calls!"
                )

            new_records_total = 0
            updated_records_total = 0

            if strategy_analysis["strategy"] == "bulk":
                # Use optimized bulk fetching strategy
                new_records, updated_records, total_fetched = (
                    self._bulk_fetch_with_deduplication(
                        session, symbol, start_date, end_date
                    )
                )
                new_records_total += new_records
                updated_records_total += updated_records

                if new_records > 0 or updated_records > 0:
                    # Invalidate cache for this symbol
                    invalidate_symbol_cache(symbol)

            elif strategy_analysis["strategy"] == "incremental":
                # Use traditional incremental approach for large consecutive gaps
                logger.info(
                    f"Using incremental strategy for {len(missing_ranges)} large gaps"
                )

                all_fetched_data = []
                for range_start, range_end in missing_ranges:
                    gap_days = (range_end - range_start).days + 1
                    logger.info(
                        f"Fetching gap for {symbol}: {_safe_date(range_start)} to {_safe_date(range_end)} ({gap_days} days)"
                    )

                    try:
                        # Fetch data for this range
                        range_data = self.fetch_daily_data(
                            symbol,
                            start_date=_safe_date(range_start),
                            end_date=_safe_date(range_end),
                        )

                        if range_data is not None and not range_data.empty:
                            all_fetched_data.append(range_data)
                            logger.info(
                                f"✓ Fetched {len(range_data)} records for gap ({_safe_date(range_start)} to {_safe_date(range_end)})"
                            )
                        else:
                            logger.warning(
                                f"No data fetched for gap ({_safe_date(range_start)} to {_safe_date(range_end)})"
                            )

                    except Exception as e:
                        logger.error(
                            f"Error fetching gap for {symbol} ({_safe_date(range_start)} to {_safe_date(range_end)}): {str(e)}"
                        )

                # Combine and save incremental data
                if all_fetched_data:
                    combined_fetched = pd.concat(all_fetched_data)
                    # Remove duplicates based on index (Date) and sort
                    combined_fetched = combined_fetched[
                        ~combined_fetched.index.duplicated(keep="last")
                    ]
                    combined_fetched = combined_fetched.sort_index()

                    # Convert to MarketData objects and save
                    market_data_objects = self.transform_to_market_data(
                        symbol, combined_fetched
                    )
                    if market_data_objects:
                        from .storage import save_market_data

                        new_records, updated_records = save_market_data(
                            session, market_data_objects
                        )
                        new_records_total += new_records
                        updated_records_total += updated_records
                        logger.info(
                            f"Saved {new_records} new, {updated_records} updated records for {symbol}"
                        )

                        # Invalidate cache for this symbol
                        invalidate_symbol_cache(symbol)

            # Log final optimization results
            if new_records_total > 0 or updated_records_total > 0:
                logger.info(
                    f"🎯 Optimization complete for {symbol}: {new_records_total} new, {updated_records_total} updated records using {strategy_analysis['strategy']} strategy"
                )

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

    def _analyze_gaps_for_optimal_strategy(
        self, missing_ranges, total_start_date, total_end_date
    ):
        """
        Analyze missing date ranges to determine the most efficient fetching strategy.

        This method determines whether to:
        1. Fetch the entire period in one API call (bulk strategy)
        2. Fetch only missing ranges incrementally (incremental strategy)

        Args:
            missing_ranges: List of (start_date, end_date) tuples for missing data
            total_start_date: Overall start date for the data request
            total_end_date: Overall end date for the data request

        Returns:
            dict: Strategy analysis with recommendation
        """
        from datetime import timedelta

        if not missing_ranges:
            return {"strategy": "none", "reason": "No missing data"}

        # Calculate total requested period
        total_days = (total_end_date - total_start_date).days + 1

        # Calculate total missing days
        missing_days = sum(
            (end_date - start_date).days + 1 for start_date, end_date in missing_ranges
        )

        # Calculate largest gap
        largest_gap = max(
            (end_date - start_date).days + 1 for start_date, end_date in missing_ranges
        )

        # Calculate number of separate gaps
        num_gaps = len(missing_ranges)

        # Missing data percentage
        missing_percentage = missing_days / total_days if total_days > 0 else 0

        logger.info(
            f"Gap analysis: {missing_days}/{total_days} days missing ({missing_percentage:.1%}), {num_gaps} gaps, largest gap: {largest_gap} days"
        )

        # Strategy decision logic
        strategy_reasons = []

        # Use bulk strategy if:
        # 1. More than 70% of data is missing (likely new symbol)
        if missing_percentage > 0.70:
            return {
                "strategy": "bulk",
                "reason": f"High missing percentage ({missing_percentage:.1%}) suggests new symbol - bulk fetch more efficient",
                "api_calls_saved": num_gaps - 1,
                "total_days": total_days,
                "missing_days": missing_days,
                "num_gaps": num_gaps,
            }

        # 2. Many small gaps (more than 5 gaps for periods > 6 months)
        if num_gaps > 5 and total_days > 180:
            return {
                "strategy": "bulk",
                "reason": f"{num_gaps} separate gaps would require {num_gaps} API calls - bulk fetch more efficient",
                "api_calls_saved": num_gaps - 1,
                "total_days": total_days,
                "missing_days": missing_days,
                "num_gaps": num_gaps,
            }

        # 3. Small total period (< 3 months) with any gaps
        if total_days < 90 and missing_days > 0:
            return {
                "strategy": "bulk",
                "reason": f"Short period ({total_days} days) - single API call more efficient than {num_gaps} incremental calls",
                "api_calls_saved": num_gaps - 1,
                "total_days": total_days,
                "missing_days": missing_days,
                "num_gaps": num_gaps,
            }

        # 4. Scattered small gaps (average gap size < 30 days)
        avg_gap_size = missing_days / num_gaps if num_gaps > 0 else 0
        if num_gaps > 2 and avg_gap_size < 30:
            return {
                "strategy": "bulk",
                "reason": f"Many small gaps (avg {avg_gap_size:.1f} days) - bulk fetch avoids {num_gaps} API calls",
                "api_calls_saved": num_gaps - 1,
                "total_days": total_days,
                "missing_days": missing_days,
                "num_gaps": num_gaps,
            }

        # Use incremental strategy for:
        # 1. Few large gaps where incremental makes sense
        # 2. Low missing percentage with large consecutive gaps
        return {
            "strategy": "incremental",
            "reason": f"Large consecutive gaps ({largest_gap} days max) - incremental fetch most efficient",
            "api_calls_saved": 0,
            "total_days": total_days,
            "missing_days": missing_days,
            "num_gaps": num_gaps,
        }

    def _bulk_fetch_with_deduplication(self, session, symbol, start_date, end_date):
        """
        Fetch entire period in one API call and deduplicate against existing database data.

        This method is used when the gap analysis determines that a bulk fetch
        is more efficient than multiple incremental API calls.

        Args:
            session: SQLAlchemy session for database operations
            symbol: Stock or crypto symbol
            start_date: Start date for data collection
            end_date: End date for data collection

        Returns:
            tuple: (new_records_saved, updated_records, total_fetched)
        """
        from .storage import get_market_data, save_market_data

        logger.info(
            f"🚀 Bulk fetching {symbol} from {_safe_date(start_date)} to {_safe_date(end_date)} (OPTIMIZED)"
        )

        try:
            # Fetch entire period in ONE API call
            bulk_data = self.fetch_daily_data(
                symbol, start_date=start_date, end_date=end_date
            )

            if bulk_data is None or bulk_data.empty:
                logger.warning(f"Bulk fetch returned no data for {symbol}")
                return 0, 0, 0

            logger.info(f"✓ Bulk fetched {len(bulk_data)} days of data for {symbol}")

            # Get existing data from database to identify what's already there
            existing_data = get_market_data(
                session, symbol, _safe_date(start_date), _safe_date(end_date)
            )

            # Create a set of existing dates for fast lookup
            existing_dates = set()
            if existing_data:
                existing_dates = {record.date for record in existing_data}
                logger.info(
                    f"Found {len(existing_dates)} existing dates for {symbol} in database"
                )

            # Filter bulk data to only include new/missing dates
            if existing_dates:
                # Only keep dates that are NOT in the database
                bulk_data_filtered = bulk_data[~bulk_data.index.isin(existing_dates)]
                logger.info(
                    f"After deduplication: {len(bulk_data_filtered)} new records to save for {symbol}"
                )
            else:
                bulk_data_filtered = bulk_data
                logger.info(
                    f"No existing data - will save all {len(bulk_data_filtered)} records for {symbol}"
                )

            if bulk_data_filtered.empty:
                logger.info(
                    f"No new data to save for {symbol} - all data already in database"
                )
                return 0, 0, len(bulk_data)

            # Convert to MarketData objects and save
            market_data_objects = self.transform_to_market_data(
                symbol, bulk_data_filtered
            )
            if market_data_objects:
                new_records, updated_records = save_market_data(
                    session, market_data_objects
                )
                logger.info(
                    f"✅ Bulk save complete for {symbol}: {new_records} new, {updated_records} updated records"
                )
                return new_records, updated_records, len(bulk_data)
            else:
                logger.warning(
                    f"Failed to convert DataFrame to MarketData objects for {symbol}"
                )
                return 0, 0, len(bulk_data)

        except Exception as e:
            logger.error(f"Error in bulk fetch for {symbol}: {str(e)}")
            return 0, 0, 0

    def incremental_fetch_batch(
        self,
        session,
        symbols: List[str],
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        period: Optional[str] = None,
        batch_size: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data incrementally for multiple symbols in batches.

        Args:
            session: SQLAlchemy session
            symbols: List of symbols to fetch
            start_date: Start date for data collection
            end_date: End date for data collection
            period: Period string
            batch_size: Number of symbols to process in each batch

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        import time

        logger.info(f"Starting incremental batch fetch for {len(symbols)} symbols")

        results = {}
        total_batches = (len(symbols) + batch_size - 1) // batch_size

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)"
            )

            for symbol in batch:
                try:
                    data = self.incremental_fetch_daily_data(
                        session, symbol, start_date, end_date, period
                    )

                    if data is not None and not data.empty:
                        results[symbol] = data
                        logger.debug(f"✓ {symbol}: {len(data)} days of data")
                    else:
                        logger.warning(f"✗ No data for {symbol}")

                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {str(e)}")

            # Rate limiting between batches
            if batch_num < total_batches:
                logger.info(f"Rate limiting: waiting 1 second before next batch")
                time.sleep(1)

        logger.info(
            f"Incremental batch fetch complete: {len(results)} symbols successful"
        )
        return results

    def test_connection(self) -> bool:
        """
        Test the connection to Alpaca API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not ALPACA_AVAILABLE:
                logger.error("Alpaca SDK not available")
                return False

            # Try to fetch a small amount of data for SPY
            test_data = self.fetch_daily_data("SPY", period="3mo")

            if test_data is not None and not test_data.empty:
                logger.info("✓ Alpaca API connection successful")
                return True
            else:
                logger.error("✗ Alpaca API connection failed - no data returned")
                return False

        except Exception as e:
            logger.error(f"✗ Alpaca API connection failed: {str(e)}")
            return False

    def transform_to_market_data(
        self, symbol: str, df: pd.DataFrame
    ) -> List["MarketData"]:
        """
        Transform DataFrame to MarketData objects.

        Args:
            symbol: Stock or crypto symbol
            df: DataFrame with OHLCV data

        Returns:
            List of MarketData objects
        """
        from .storage import MarketData

        if df is None or df.empty:
            return []

        market_data_list = []

        # Handle different DataFrame index/column structures
        if isinstance(df.index[0], str) and "Date" in df.columns:
            # If Date is a column, iterate through rows using that
            for idx, row in df.iterrows():
                date_val = pd.to_datetime(row["Date"]).date()
                market_data = MarketData(
                    symbol=symbol,
                    date=date_val,
                    open_price=float(row.get("Open", row.get("open", 0))),
                    high_price=float(row.get("High", row.get("high", 0))),
                    low_price=float(row.get("Low", row.get("low", 0))),
                    close_price=float(row.get("Close", row.get("close", 0))),
                    volume=int(row.get("Volume", row.get("volume", 0))),
                    adj_close=float(
                        row.get("Adj Close", row.get("adj_close", row.get("close", 0)))
                    ),
                )
                market_data_list.append(market_data)
        elif hasattr(df.index[0], "date"):
            # If index is already datetime/date objects
            for date_val, row in df.iterrows():
                market_data = MarketData(
                    symbol=symbol,
                    date=date_val.date() if hasattr(date_val, "date") else date_val,
                    open_price=float(row.get("Open", row.get("open", 0))),
                    high_price=float(row.get("High", row.get("high", 0))),
                    low_price=float(row.get("Low", row.get("low", 0))),
                    close_price=float(row.get("Close", row.get("close", 0))),
                    volume=int(row.get("Volume", row.get("volume", 0))),
                    adj_close=float(
                        row.get("Adj Close", row.get("adj_close", row.get("close", 0)))
                    ),
                )
                market_data_list.append(market_data)
        else:
            # Handle case where we need to extract date from timestamp column
            for idx, row in df.iterrows():
                # Try to get date from timestamp column if it exists
                if "timestamp" in row:
                    date_val = pd.to_datetime(row["timestamp"]).date()
                elif "Date" in row:
                    date_val = pd.to_datetime(row["Date"]).date()
                else:
                    # Fallback: use current date (this shouldn't happen in normal operation)
                    date_val = datetime.now().date()
                    logger.warning(
                        f"No timestamp found for {symbol}, using current date"
                    )

                market_data = MarketData(
                    symbol=symbol,
                    date=date_val,
                    open_price=float(row.get("Open", row.get("open", 0))),
                    high_price=float(row.get("High", row.get("high", 0))),
                    low_price=float(row.get("Low", row.get("low", 0))),
                    close_price=float(row.get("Close", row.get("close", 0))),
                    volume=int(row.get("Volume", row.get("volume", 0))),
                    adj_close=float(
                        row.get("Adj Close", row.get("adj_close", row.get("close", 0)))
                    ),
                )
                market_data_list.append(market_data)

        return market_data_list


def get_alpaca_collector() -> AlpacaDataCollector:
    """
    Factory function to create an Alpaca data collector.

    Returns:
        Configured AlpacaDataCollector instance
    """
    return AlpacaDataCollector()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Alpaca Data Collector...")

    try:
        collector = get_alpaca_collector()

        # Test connection
        if collector.test_connection():
            print("✓ Connection successful!")

            # Test data fetching
            symbols = ["SPY", "QQQ", "VTI"]

            for symbol in symbols:
                print(f"\nFetching data for {symbol}...")
                data = collector.fetch_daily_data(symbol, period="6mo")

                if data is not None:
                    print(f"✓ {symbol}: {len(data)} days of data")
                    print(f"  Latest price: ${data['Close'].iloc[-1]:.2f}")
                    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
                else:
                    print(f"✗ Failed to fetch data for {symbol}")

            # Test current prices
            print(f"\nGetting current prices...")
            prices = collector.get_multiple_prices(symbols)
            for symbol, price in prices.items():
                print(f"  {symbol}: ${price:.2f}")

        else:
            print("✗ Connection failed!")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTo use Alpaca data collector:")
        print("1. Sign up at https://alpaca.markets")
        print("2. Get your API keys from the dashboard")
        print("3. Set environment variables:")
        print("   export ALPACA_API_KEY=your_api_key")
        print("   export ALPACA_SECRET_KEY=your_secret_key")
        print("4. Install Alpaca SDK: pip install alpaca-py")
