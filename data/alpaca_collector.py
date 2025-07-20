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

logger = logging.getLogger(__name__)


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

        # Crypto symbols mapping (Alpaca uses different symbols for crypto)
        self.crypto_symbols = {
            "BTCUSD": "BTC/USD",
            "ETHUSD": "ETH/USD",
            "ADAUSD": "ADA/USD",
            "DOTUSD": "DOT/USD",
            "LINKUSD": "LINK/USD",
            "LTCUSD": "LTC/USD",
            "BCHUSD": "BCH/USD",
            "XRPUSD": "XRP/USD",
            "SOLUSD": "SOL/USD",
            "MATICUSD": "MATIC/USD",
        }

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
            raise ValueError("Invalid API key format. Keys should be at least 10 characters long.")

        return AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,  # Use paper trading for safety
        )

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
            self._check_rate_limit()

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
                f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}"
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
                logger.info(f"Crypto request created for {symbol} ({alpaca_symbol})")
                bars = client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                )
                logger.info(f"Stock request created for {symbol}")
                bars = client.get_stock_bars(request)

            logger.info(f"Response received: {type(bars)}")
            if bars:
                logger.info(
                    f"BarSet length: {len(bars) if hasattr(bars, '__len__') else 'No length'}"
                )
                logger.info(f"BarSet attributes: {dir(bars)}")

            if not bars:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Try different ways to access the data
            try:
                # Method 1: Direct access to BarSet df
                if hasattr(bars, "df"):
                    df = bars.df
                    logger.info(f"Found data with {len(df)} rows")
                # Method 2: Direct access by symbol (use alpaca_symbol for crypto)
                elif hasattr(bars, alpaca_symbol):
                    df = bars[alpaca_symbol].df
                    logger.info(f"Found data for {symbol} ({alpaca_symbol}) with {len(df)} rows")
                # Method 3: Iterate through bars
                elif hasattr(bars, "__iter__"):
                    for bar in bars:
                        if hasattr(bar, "symbol") and bar.symbol == alpaca_symbol:
                            df = bar.df
                            logger.info(f"Found data for {symbol} ({alpaca_symbol}) with {len(df)} rows")
                            break
                else:
                    logger.warning(f"No data found for {symbol} ({alpaca_symbol})")
                    return None
            except Exception as e:
                logger.error(f"Error accessing data for {symbol} ({alpaca_symbol}): {str(e)}")
                return None

            if df.empty:
                logger.warning(f"Empty DataFrame for {symbol}")
                return None

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

            # Add Adj Close column (use Close for now, Alpaca doesn't provide adjusted)
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
                    start=today - timedelta(days=5),  # Get last 5 days to ensure we have today
                    end=today,
                )
                bars = client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=TimeFrame.Day,
                    start=today - timedelta(days=5),  # Get last 5 days to ensure we have today
                    end=today,
                )
                bars = client.get_stock_bars(request)

            if bars and alpaca_symbol in bars:
                latest_bar = bars[alpaca_symbol].df.iloc[-1]
                self.request_count += 1
                return float(latest_bar["close"])

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
