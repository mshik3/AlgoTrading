#!/usr/bin/env python3
"""
Alpaca Assets API module for dynamic asset discovery.

This module provides functionality to query Alpaca's Assets API to discover
available crypto assets dynamically, rather than relying on hardcoded lists.
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AlpacaAsset:
    """Represents an Alpaca asset."""

    id: str
    class_type: str  # 'crypto', 'us_equity', etc.
    exchange: str
    symbol: str
    name: str
    status: str
    tradable: bool
    marginable: bool
    shortable: bool
    easy_to_borrow: bool
    fractionable: bool


class AlpacaAssetsAPI:
    """
    Client for Alpaca's Assets API to discover available assets.

    This class provides methods to query and cache available crypto assets,
    allowing for dynamic symbol validation and discovery.
    """

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize the Alpaca Assets API client.

        Args:
            api_key: Alpaca API key (if None, loads from environment)
            secret_key: Alpaca secret key (if None, loads from environment)
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or provide them directly."
            )

        # Paper trading base URL
        self.base_url = "https://paper-api.alpaca.markets"

        # Cache for available assets
        self._cached_assets: Dict[str, AlpacaAsset] = {}
        self._cached_crypto_symbols: Set[str] = set()
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(hours=1)  # Cache for 1 hour

        logger.info("Alpaca Assets API client initialized")

    def _make_request(self, endpoint: str) -> List[Dict]:
        """
        Make a request to the Alpaca Assets API.

        Args:
            endpoint: API endpoint to call

        Returns:
            List of asset dictionaries

        Raises:
            requests.RequestException: If the API request fails
        """
        url = f"{self.base_url}/v2/assets"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch assets from Alpaca API: {e}")
            raise

    def get_all_assets(self, force_refresh: bool = False) -> Dict[str, AlpacaAsset]:
        """
        Get all available assets from Alpaca.

        Args:
            force_refresh: Whether to force refresh the cache

        Returns:
            Dictionary mapping symbol -> AlpacaAsset
        """
        # Check if cache is still valid
        if (
            not force_refresh
            and self._cache_timestamp
            and datetime.now() - self._cache_timestamp < self._cache_duration
        ):
            logger.debug("Using cached assets")
            return self._cached_assets

        logger.info("Fetching assets from Alpaca API...")

        try:
            assets_data = self._make_request("/v2/assets")

            # Parse assets
            assets = {}
            for asset_data in assets_data:
                asset = AlpacaAsset(
                    id=asset_data.get("id", ""),
                    class_type=asset_data.get("class", ""),
                    exchange=asset_data.get("exchange", ""),
                    symbol=asset_data.get("symbol", ""),
                    name=asset_data.get("name", ""),
                    status=asset_data.get("status", ""),
                    tradable=asset_data.get("tradable", False),
                    marginable=asset_data.get("marginable", False),
                    shortable=asset_data.get("shortable", False),
                    easy_to_borrow=asset_data.get("easy_to_borrow", False),
                    fractionable=asset_data.get("fractionable", False),
                )
                assets[asset.symbol] = asset

            # Update cache
            self._cached_assets = assets
            self._cache_timestamp = datetime.now()

            logger.info(f"Fetched {len(assets)} assets from Alpaca API")
            return assets

        except Exception as e:
            logger.error(f"Error fetching assets: {e}")
            # Return cached data if available, otherwise empty dict
            return self._cached_assets if self._cached_assets else {}

    def get_crypto_assets(self, force_refresh: bool = False) -> Dict[str, AlpacaAsset]:
        """
        Get available crypto assets from Alpaca.

        Args:
            force_refresh: Whether to force refresh the cache

        Returns:
            Dictionary mapping symbol -> AlpacaAsset for crypto assets only
        """
        all_assets = self.get_all_assets(force_refresh)

        # Filter for crypto assets that are tradable
        crypto_assets = {
            symbol: asset
            for symbol, asset in all_assets.items()
            if asset.class_type == "crypto"
            and asset.tradable
            and asset.status == "active"
        }

        # Update crypto symbols cache
        self._cached_crypto_symbols = set(crypto_assets.keys())

        logger.info(f"Found {len(crypto_assets)} tradable crypto assets")
        return crypto_assets

    def get_available_crypto_symbols(self, force_refresh: bool = False) -> Set[str]:
        """
        Get set of available crypto symbols.

        Args:
            force_refresh: Whether to force refresh the cache

        Returns:
            Set of available crypto symbols
        """
        crypto_assets = self.get_crypto_assets(force_refresh)
        return set(crypto_assets.keys())

    def is_crypto_symbol_available(self, symbol: str) -> bool:
        """
        Check if a crypto symbol is available for trading.

        Args:
            symbol: Crypto symbol to check (e.g., "BTC/USD")

        Returns:
            True if symbol is available, False otherwise
        """
        # Ensure we have fresh data
        available_symbols = self.get_available_crypto_symbols()
        return symbol in available_symbols

    def get_crypto_asset_info(self, symbol: str) -> Optional[AlpacaAsset]:
        """
        Get detailed information about a crypto asset.

        Args:
            symbol: Crypto symbol to get info for

        Returns:
            AlpacaAsset if found, None otherwise
        """
        crypto_assets = self.get_crypto_assets()
        return crypto_assets.get(symbol)

    def suggest_alternative_symbols(
        self, requested_symbol: str, limit: int = 5
    ) -> List[str]:
        """
        Suggest alternative crypto symbols when the requested one is not available.

        Args:
            requested_symbol: The symbol that was requested but not available
            limit: Maximum number of alternatives to suggest

        Returns:
            List of alternative symbols
        """
        available_symbols = self.get_available_crypto_symbols()

        # Extract base currency from requested symbol (e.g., "ADA" from "ADA/USD")
        base_currency = (
            requested_symbol.split("/")[0]
            if "/" in requested_symbol
            else requested_symbol
        )

        # Find symbols that might be similar or popular alternatives
        alternatives = []

        # First, try to find symbols with the same base currency
        for symbol in available_symbols:
            if symbol.startswith(f"{base_currency}/"):
                alternatives.append(symbol)
                if len(alternatives) >= limit:
                    break

        # If we don't have enough, add popular crypto symbols
        popular_symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "MATIC/USD", "LINK/USD"]
        for symbol in popular_symbols:
            if symbol in available_symbols and symbol not in alternatives:
                alternatives.append(symbol)
                if len(alternatives) >= limit:
                    break

        return alternatives[:limit]

    def refresh_cache(self):
        """Force refresh the asset cache."""
        logger.info("Refreshing asset cache...")
        self.get_all_assets(force_refresh=True)


# Global instance for easy access
_assets_api_instance: Optional[AlpacaAssetsAPI] = None


def get_assets_api() -> AlpacaAssetsAPI:
    """
    Get the global AlpacaAssetsAPI instance.

    Returns:
        AlpacaAssetsAPI instance
    """
    global _assets_api_instance
    if _assets_api_instance is None:
        _assets_api_instance = AlpacaAssetsAPI()
    return _assets_api_instance


def get_available_crypto_symbols() -> Set[str]:
    """
    Get available crypto symbols.

    Returns:
        Set of available crypto symbols
    """
    return get_assets_api().get_available_crypto_symbols()


def is_crypto_symbol_available(symbol: str) -> bool:
    """
    Check if a crypto symbol is available.

    Args:
        symbol: Crypto symbol to check

    Returns:
        True if available, False otherwise
    """
    return get_assets_api().is_crypto_symbol_available(symbol)
