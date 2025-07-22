"""
Alpaca Data Adapter for Modern Portfolio Optimization.

This module provides Alpaca API integration for the modern portfolio
optimization libraries, replacing Yahoo Finance with reliable Alpaca data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST
import os

logger = logging.getLogger(__name__)


class AlpacaDataAdapter:
    """
    Alpaca API data adapter for modern portfolio optimization.

    Provides reliable market data from Alpaca API instead of Yahoo Finance,
    with proper error handling and rate limiting.
    """

    def __init__(
        self, api_key: str = None, secret_key: str = None, base_url: str = None
    ):
        """
        Initialize Alpaca API connection.

        Args:
            api_key: Alpaca API key (defaults to environment variable)
            secret_key: Alpaca secret key (defaults to environment variable)
            base_url: Alpaca base URL (defaults to paper trading)
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.base_url = base_url or "https://paper-api.alpaca.markets"

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )

        # Initialize Alpaca API
        self.api = tradeapi.REST(
            self.api_key, self.secret_key, self.base_url, api_version="v2"
        )

        logger.info(f"Initialized Alpaca API connection to {self.base_url}")

    def get_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str = None,
        end_date: str = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Get historical price data from Alpaca API.

        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date (YYYY-MM-DD), defaults to 2 years ago
            end_date: End date (YYYY-MM-DD), defaults to today
            timeframe: Data timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)

        Returns:
            DataFrame with OHLCV data, indexed by date
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Set default dates
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        try:
            # Get historical data from Alpaca
            bars = self.api.get_bars(
                symbols,
                timeframe,
                start=start_date,
                end=end_date,
                adjustment="all",  # Include dividends and splits
            )

            if not bars:
                raise ValueError(f"No data returned for symbols: {symbols}")

            # Convert to DataFrame
            df = bars.df

            # Handle multi-symbol data
            if len(symbols) > 1:
                # Multi-symbol data comes with multi-level columns
                if df.columns.nlevels > 1:
                    # Extract close prices for portfolio optimization
                    close_data = df["close"].unstack(level=0)
                    close_data.columns.name = "symbol"
                    return close_data
                else:
                    # Single symbol data
                    return df[["close"]].rename(columns={"close": symbols[0]})
            else:
                # Single symbol
                return df[["close"]].rename(columns={"close": symbols[0]})

        except Exception as e:
            logger.error(f"Error fetching data from Alpaca: {e}")
            raise

    def get_latest_prices(self, symbols: Union[str, List[str]]) -> pd.Series:
        """
        Get latest prices for portfolio optimization.

        Args:
            symbols: Single symbol or list of symbols

        Returns:
            Series with latest prices
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        try:
            # Get latest quotes
            quotes = self.api.get_latest_quotes(symbols)

            if isinstance(quotes, dict):
                # Single symbol
                return pd.Series([quotes["ap"]], index=[symbols[0]])
            else:
                # Multiple symbols
                prices = {}
                for symbol in symbols:
                    if symbol in quotes:
                        prices[symbol] = quotes[symbol]["ap"]  # Ask price
                    else:
                        # Fallback to last trade price
                        trades = self.api.get_latest_trades(symbol)
                        if trades:
                            prices[symbol] = trades[0].p

                return pd.Series(prices)

        except Exception as e:
            logger.error(f"Error fetching latest prices: {e}")
            # Fallback: get last close price from historical data
            return self.get_historical_data(
                symbols, end_date=datetime.now().strftime("%Y-%m-%d")
            ).iloc[-1]

    def get_assets_info(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get asset information for portfolio optimization.

        Args:
            symbols: List of symbols

        Returns:
            DataFrame with asset information
        """
        try:
            assets = self.api.list_assets(status="active")

            # Filter for requested symbols
            asset_info = []
            for asset in assets:
                if asset.symbol in symbols:
                    asset_info.append(
                        {
                            "symbol": asset.symbol,
                            "name": asset.name,
                            "exchange": asset.exchange,
                            "type": asset.asset_class,
                            "status": asset.status,
                            "tradable": asset.tradable,
                            "marginable": asset.marginable,
                            "shortable": asset.shortable,
                            "easy_to_borrow": asset.easy_to_borrow,
                        }
                    )

            return pd.DataFrame(asset_info)

        except Exception as e:
            logger.error(f"Error fetching asset info: {e}")
            return pd.DataFrame()

    def get_account_info(self) -> Dict:
        """
        Get account information for portfolio optimization.

        Returns:
            Dictionary with account details
        """
        try:
            account = self.api.get_account()
            return {
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "daytrade_count": account.daytrade_count,
                "status": account.status,
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {}

    def get_positions(self) -> Dict[str, float]:
        """
        Get current positions for portfolio optimization.

        Returns:
            Dictionary mapping symbols to position sizes
        """
        try:
            positions = self.api.list_positions()
            return {pos.symbol: float(pos.qty) for pos in positions}
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {}


class AlpacaPortfolioOptimizer:
    """
    Portfolio optimizer that uses Alpaca API data.

    Wraps the modern portfolio optimization libraries with Alpaca data.
    """

    def __init__(self, alpaca_adapter: AlpacaDataAdapter = None):
        """
        Initialize with Alpaca data adapter.

        Args:
            alpaca_adapter: AlpacaDataAdapter instance
        """
        self.alpaca = alpaca_adapter or AlpacaDataAdapter()

    def optimize_portfolio(
        self,
        symbols: List[str],
        method: str = "max_sharpe",
        start_date: str = None,
        end_date: str = None,
        **kwargs,
    ) -> Dict:
        """
        Optimize portfolio using Alpaca data.

        Args:
            symbols: List of symbols to optimize
            method: Optimization method ('max_sharpe', 'min_volatility', etc.)
            start_date: Start date for historical data
            end_date: End date for historical data
            **kwargs: Additional optimization parameters

        Returns:
            Optimization results
        """
        # Get price data from Alpaca
        price_data = self.alpaca.get_historical_data(
            symbols, start_date=start_date, end_date=end_date
        )

        # Get latest prices for discrete allocation
        latest_prices = self.alpaca.get_latest_prices(symbols)

        # Import and use modern portfolio optimizer
        from portfolio.modern_portfolio_optimization import create_portfolio_optimizer

        optimizer = create_portfolio_optimizer(price_data, method="pypfopt")

        # Run optimization
        if method == "max_sharpe":
            result = optimizer.optimize_max_sharpe(**kwargs)
        elif method == "min_volatility":
            result = optimizer.optimize_min_volatility(**kwargs)
        elif method == "efficient_risk":
            target_vol = kwargs.pop("target_volatility", 0.15)
            result = optimizer.optimize_efficient_risk(target_vol, **kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Add Alpaca-specific information
        result["alpaca_data"] = {
            "latest_prices": latest_prices.to_dict(),
            "account_info": self.alpaca.get_account_info(),
            "current_positions": self.alpaca.get_positions(),
            "assets_info": self.alpaca.get_assets_info(symbols).to_dict("records"),
        }

        return result

    def black_litterman_optimization(
        self,
        symbols: List[str],
        views: Dict[str, float],
        start_date: str = None,
        end_date: str = None,
        **kwargs,
    ) -> Dict:
        """
        Black-Litterman optimization using Alpaca data.

        Args:
            symbols: List of symbols
            views: Investor views {symbol: expected_return}
            start_date: Start date for historical data
            end_date: End date for historical data
            **kwargs: Additional parameters

        Returns:
            Black-Litterman optimization results
        """
        # Get price data from Alpaca
        price_data = self.alpaca.get_historical_data(
            symbols, start_date=start_date, end_date=end_date
        )

        # Import and use modern portfolio optimizer
        from portfolio.modern_portfolio_optimization import create_portfolio_optimizer

        optimizer = create_portfolio_optimizer(price_data, method="pypfopt")

        # Run Black-Litterman optimization
        result = optimizer.black_litterman_optimization(views, **kwargs)

        # Add Alpaca-specific information
        result["alpaca_data"] = {
            "latest_prices": self.alpaca.get_latest_prices(symbols).to_dict(),
            "account_info": self.alpaca.get_account_info(),
            "current_positions": self.alpaca.get_positions(),
        }

        return result


# Factory function for easy access
def create_alpaca_optimizer(
    api_key: str = None, secret_key: str = None
) -> AlpacaPortfolioOptimizer:
    """
    Create Alpaca portfolio optimizer instance.

    Args:
        api_key: Alpaca API key
        secret_key: Alpaca secret key

    Returns:
        AlpacaPortfolioOptimizer instance
    """
    adapter = AlpacaDataAdapter(api_key=api_key, secret_key=secret_key)
    return AlpacaPortfolioOptimizer(adapter)
