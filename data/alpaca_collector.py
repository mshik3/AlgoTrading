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
                f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}"
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

            # Rename columns to match expected format (lowercase for consistency)
            df = df.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )

            # Add adj_close column (use close for now, Alpaca doesn't provide adjusted)
            df["adj_close"] = df["close"]

            # Ensure all required columns are present
            required_cols = ["open", "high", "low", "close", "volume", "adj_close"]
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
