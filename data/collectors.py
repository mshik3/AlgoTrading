"""
Data collectors module for the algorithmic trading system.
Handles fetching data from Alpaca Markets API.
"""

import logging
from datetime import datetime, timedelta
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

        logger.info("Alpaca data collector initialized")

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
