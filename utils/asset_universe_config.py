"""
Asset Universe Configuration
Centralized configuration for the expanded 920-asset universe.
Combines Fortune 500 companies, ETFs, and cryptocurrencies.
"""

import logging
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

# Import asset data sources
try:
    from data.fortune500_data import Fortune500DataSource
    from data.etf_database import ETFDatabase
    from data.crypto_universe import CryptoUniverse
except ImportError:
    # Fallback for when modules aren't available yet
    Fortune500DataSource = None
    ETFDatabase = None
    CryptoUniverse = None

logger = logging.getLogger(__name__)


@dataclass
class AssetUniverseConfig:
    """Configuration for the asset universe."""

    fortune500_limit: int = 500
    etf_limit: int = 400
    crypto_limit: int = 20
    total_target: int = 920


class AssetUniverseManager:
    """
    Manages the comprehensive 920-asset universe.
    Combines Fortune 500 companies, ETFs, and cryptocurrencies.
    """

    def __init__(self, config: Optional[AssetUniverseConfig] = None):
        """Initialize the asset universe manager."""
        self.config = config or AssetUniverseConfig()

        # Initialize data sources if available
        if Fortune500DataSource is not None:
            self.fortune500_source = Fortune500DataSource()
        else:
            self.fortune500_source = None

        if ETFDatabase is not None:
            self.etf_database = ETFDatabase()
        else:
            self.etf_database = None

        if CryptoUniverse is not None:
            self.crypto_universe = CryptoUniverse()
        else:
            self.crypto_universe = None

        # Cache for the combined universe
        self._combined_symbols = None
        self._asset_categories = None

    def get_fortune500_symbols(self, limit: Optional[int] = None) -> List[str]:
        """Get Fortune 500 symbols."""
        if self.fortune500_source is None:
            # Fallback to basic Fortune 500 symbols
            return [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "TSLA",
                "NVDA",
                "NFLX",
                "JPM",
                "BAC",
            ]

        limit = limit or self.config.fortune500_limit
        companies = self.fortune500_source.get_top_companies(limit)
        return [company.symbol for company in companies if company.symbol]

    def get_etf_symbols(self, limit: Optional[int] = None) -> List[str]:
        """Get ETF symbols."""
        if self.etf_database is None:
            # Fallback to basic ETF symbols
            return [
                "SPY",
                "QQQ",
                "VTI",
                "IWM",
                "VEA",
                "VWO",
                "AGG",
                "TLT",
                "XLF",
                "XLK",
            ]

        limit = limit or self.config.etf_limit
        etfs = self.etf_database.get_all_etfs()
        # Prioritize by AUM and expense ratio
        sorted_etfs = sorted(
            etfs, key=lambda x: (x.aum_billions, -x.expense_ratio), reverse=True
        )
        return [etf.symbol for etf in sorted_etfs[:limit] if etf.symbol]

    def get_crypto_symbols(self, limit: Optional[int] = None) -> List[str]:
        """Get crypto symbols (trading cryptocurrencies only)."""
        if self.crypto_universe is None:
            # Fallback to basic crypto symbols
            return ["BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "DOTUSD"]

        limit = limit or self.config.crypto_limit
        cryptos = self.crypto_universe.get_trading_cryptocurrencies()
        # Sort by market cap rank
        sorted_cryptos = sorted(cryptos, key=lambda x: x.market_cap_rank)
        return [crypto.symbol for crypto in sorted_cryptos[:limit] if crypto.symbol]

    def get_combined_symbols(self) -> List[str]:
        """Get the complete 920-asset universe."""
        if self._combined_symbols is None:
            fortune500_symbols = self.get_fortune500_symbols()
            etf_symbols = self.get_etf_symbols()
            crypto_symbols = self.get_crypto_symbols()

            # Combine all symbols
            combined = fortune500_symbols + etf_symbols + crypto_symbols

            # Remove duplicates while preserving order
            seen = set()
            unique_symbols = []
            for symbol in combined:
                if symbol not in seen:
                    seen.add(symbol)
                    unique_symbols.append(symbol)

            # Filter out unavailable symbols
            from utils.symbol_normalization import is_symbol_available_for_alpaca
            original_count = len(unique_symbols)
            available_symbols = [s for s in unique_symbols if is_symbol_available_for_alpaca(s)]
            filtered_count = original_count - len(available_symbols)
            
            if filtered_count > 0:
                logger.warning(
                    f"Filtered out {filtered_count} unavailable symbols from asset universe. "
                    f"Final universe size: {len(available_symbols)}"
                )

            self._combined_symbols = available_symbols

            logger.info(f"Created {len(self._combined_symbols)}-asset universe:")
            logger.info(f"  - Fortune 500: {len(fortune500_symbols)}")
            logger.info(f"  - ETFs: {len(etf_symbols)}")
            logger.info(f"  - Crypto: {len(crypto_symbols)}")

        return self._combined_symbols

    def get_asset_categories(self) -> Dict[str, List[str]]:
        """Get assets organized by category."""
        if self._asset_categories is None:
            categories = {
                "Fortune 500": self.get_fortune500_symbols(),
                "US ETFs": self._get_us_etf_symbols(),
                "Sector ETFs": self._get_sector_etf_symbols(),
                "International ETFs": self._get_international_etf_symbols(),
                "Bond ETFs": self._get_bond_etf_symbols(),
                "Commodity ETFs": self._get_commodity_etf_symbols(),
                "Crypto": self.get_crypto_symbols(),
            }
            self._asset_categories = categories

        return self._asset_categories

    def _get_us_etf_symbols(self) -> List[str]:
        """Get US-focused ETF symbols."""
        us_categories = [
            "US Large Cap",
            "US Mid Cap",
            "US Small Cap",
            "US Total Market",
            "Growth",
            "Value",
            "Dividend",
        ]
        symbols = []
        for category in us_categories:
            etfs = self.etf_database.get_etfs_by_category(category)
            symbols.extend([etf.symbol for etf in etfs[:20]])  # Limit per category
        return symbols[:100]  # Total limit

    def _get_sector_etf_symbols(self) -> List[str]:
        """Get sector ETF symbols."""
        sector_categories = [
            "Technology",
            "Financials",
            "Healthcare",
            "Consumer Discretionary",
            "Consumer Staples",
            "Industrials",
            "Energy",
            "Materials",
            "Real Estate",
            "Utilities",
        ]
        symbols = []
        for category in sector_categories:
            etfs = self.etf_database.get_etfs_by_category(category)
            symbols.extend([etf.symbol for etf in etfs[:10]])  # Limit per sector
        return symbols[:100]  # Total limit

    def _get_international_etf_symbols(self) -> List[str]:
        """Get international ETF symbols."""
        intl_categories = [
            "International Developed",
            "International Emerging",
            "International Regional",
        ]
        symbols = []
        for category in intl_categories:
            etfs = self.etf_database.get_etfs_by_category(category)
            symbols.extend([etf.symbol for etf in etfs[:15]])  # Limit per category
        return symbols[:50]  # Total limit

    def _get_bond_etf_symbols(self) -> List[str]:
        """Get bond ETF symbols."""
        bond_categories = ["US Bonds", "International Bonds"]
        symbols = []
        for category in bond_categories:
            etfs = self.etf_database.get_etfs_by_category(category)
            symbols.extend([etf.symbol for etf in etfs[:20]])  # Limit per category
        return symbols[:50]  # Total limit

    def _get_commodity_etf_symbols(self) -> List[str]:
        """Get commodity ETF symbols."""
        commodity_categories = ["Commodities", "Currencies"]
        symbols = []
        for category in commodity_categories:
            etfs = self.etf_database.get_etfs_by_category(category)
            symbols.extend([etf.symbol for etf in etfs[:10]])  # Limit per category
        return symbols[:20]  # Total limit

    def get_strategy_asset_universe(self, strategy_type: str) -> List[str]:
        """Get asset universe optimized for specific strategy types."""
        if strategy_type == "golden_cross":
            # Golden Cross works well with liquid, large-cap assets
            return (
                self.get_fortune500_symbols(100)  # Top 100 Fortune 500
                + self._get_us_etf_symbols()[:50]  # Top 50 US ETFs
                + self.get_crypto_symbols(10)  # Top 10 crypto
            )

        elif strategy_type == "mean_reversion":
            # Mean reversion works well with volatile assets
            return (
                self.get_fortune500_symbols(200)  # More stocks for mean reversion
                + self._get_sector_etf_symbols()[:50]  # Sector ETFs
                + self.get_crypto_symbols(15)  # More crypto for volatility
            )

        elif strategy_type == "momentum":
            # Momentum strategies need diverse asset classes
            return (
                self.get_fortune500_symbols(150)
                + self._get_us_etf_symbols()[:75]
                + self._get_international_etf_symbols()[:25]
                + self.get_crypto_symbols(20)
            )

        elif strategy_type == "etf_rotation":
            # ETF rotation focuses on ETFs
            return (
                self._get_us_etf_symbols()[:100]
                + self._get_sector_etf_symbols()[:100]
                + self._get_international_etf_symbols()[:100]
                + self._get_bond_etf_symbols()[:50]
                + self._get_commodity_etf_symbols()[:20]
                + self.get_crypto_symbols(20)
            )

        else:
            # Default to full universe
            return self.get_combined_symbols()

    def get_asset_info(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive asset information."""
        # Check Fortune 500
        company = self.fortune500_source.get_company_by_symbol(symbol)
        if company:
            return {
                "symbol": symbol,
                "name": company.company_name,
                "type": "stock",
                "sector": company.sector,
                "industry": company.industry,
                "market_cap_billions": company.market_cap_billions,
                "source": "fortune500",
            }

        # Check ETFs
        etf = self.etf_database.get_etf_by_symbol(symbol)
        if etf:
            return {
                "symbol": symbol,
                "name": etf.name,
                "type": "etf",
                "category": etf.category,
                "asset_class": etf.asset_class,
                "expense_ratio": etf.expense_ratio,
                "aum_billions": etf.aum_billions,
                "source": "etf_database",
            }

        # Check Crypto
        crypto = self.crypto_universe.get_cryptocurrency_by_symbol(symbol)
        if crypto:
            return {
                "symbol": symbol,
                "name": crypto.name,
                "type": "crypto",
                "category": crypto.category,
                "market_cap_rank": crypto.market_cap_rank,
                "alpaca_symbol": crypto.alpaca_symbol,
                "source": "crypto_universe",
            }

        return None

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists in the universe."""
        return self.get_asset_info(symbol) is not None

    def get_universe_summary(self) -> Dict:
        """Get summary statistics of the asset universe."""
        combined_symbols = self.get_combined_symbols()
        categories = self.get_asset_categories()

        summary = {
            "total_assets": len(combined_symbols),
            "fortune500_count": len(categories["Fortune 500"]),
            "etf_count": len(categories["US ETFs"])
            + len(categories["Sector ETFs"])
            + len(categories["International ETFs"])
            + len(categories["Bond ETFs"])
            + len(categories["Commodity ETFs"]),
            "crypto_count": len(categories["Crypto"]),
            "categories": {k: len(v) for k, v in categories.items()},
            "target_universe_size": self.config.total_target,
        }

        return summary


# Global instance for easy access
_asset_universe_manager = None


def get_asset_universe_manager() -> AssetUniverseManager:
    """Get the global asset universe manager instance."""
    global _asset_universe_manager
    if _asset_universe_manager is None:
        _asset_universe_manager = AssetUniverseManager()
    return _asset_universe_manager


def get_920_asset_universe() -> List[str]:
    """Get the complete 920-asset universe."""
    manager = get_asset_universe_manager()
    return manager.get_combined_symbols()


def get_strategy_assets(strategy_type: str) -> List[str]:
    """Get assets optimized for a specific strategy."""
    manager = get_asset_universe_manager()
    return manager.get_strategy_asset_universe(strategy_type)


def get_asset_info(symbol: str) -> Optional[Dict]:
    """Get asset information."""
    manager = get_asset_universe_manager()
    return manager.get_asset_info(symbol)


def validate_asset_symbol(symbol: str) -> bool:
    """Validate asset symbol."""
    manager = get_asset_universe_manager()
    return manager.validate_symbol(symbol)


def get_universe_summary() -> Dict:
    """Get universe summary."""
    manager = get_asset_universe_manager()
    return manager.get_universe_summary()


if __name__ == "__main__":
    # Test the asset universe manager
    manager = AssetUniverseManager()

    # Get universe summary
    summary = manager.get_universe_summary()
    print("Asset Universe Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Get strategy-specific universes
    strategies = ["golden_cross", "mean_reversion", "momentum", "etf_rotation"]
    for strategy in strategies:
        symbols = manager.get_strategy_asset_universe(strategy)
        print(f"\n{strategy.upper()} Strategy Assets: {len(symbols)}")
        print(f"  Sample: {symbols[:10]}")

    # Test asset info
    test_symbols = ["AAPL", "SPY", "BTCUSD"]
    for symbol in test_symbols:
        info = manager.get_asset_info(symbol)
        if info:
            print(f"\n{symbol}: {info['name']} ({info['type']})")
        else:
            print(f"\n{symbol}: Not found in universe")
