"""
Crypto Universe Database
Provides comprehensive list of 20+ cryptocurrencies available on Alpaca.
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Cryptocurrency:
    """Represents a cryptocurrency."""

    symbol: str
    name: str
    alpaca_symbol: str  # Alpaca's format (e.g., "BTC/USD")
    category: str
    market_cap_rank: int
    description: str
    is_stablecoin: bool = False


class CryptoUniverse:
    """Database of cryptocurrencies available on Alpaca."""

    def __init__(self):
        """Initialize the crypto universe."""
        self.cryptocurrencies = self._load_crypto_data()

    def _load_crypto_data(self) -> List[Cryptocurrency]:
        """Load cryptocurrency data."""
        crypto_data = [
            # Major Cryptocurrencies
            Cryptocurrency(
                "BTCUSD",
                "Bitcoin",
                "BTC/USD",
                "Store of Value",
                1,
                "First and largest cryptocurrency by market cap",
            ),
            Cryptocurrency(
                "ETHUSD",
                "Ethereum",
                "ETH/USD",
                "Smart Contract Platform",
                2,
                "Leading smart contract platform",
            ),
            Cryptocurrency(
                "SOLUSD",
                "Solana",
                "SOL/USD",
                "Smart Contract Platform",
                3,
                "High-performance blockchain platform",
            ),
            Cryptocurrency(
                "LINKUSD",
                "Chainlink",
                "LINK/USD",
                "Oracle Network",
                4,
                "Decentralized oracle network",
            ),
            Cryptocurrency(
                "DOTUSD",
                "Polkadot",
                "DOT/USD",
                "Interoperability",
                5,
                "Multi-chain interoperability platform",
            ),
            Cryptocurrency(
                "AVAXUSD",
                "Avalanche",
                "AVAX/USD",
                "Smart Contract Platform",
                6,
                "High-throughput blockchain platform",
            ),
            Cryptocurrency(
                "UNIUSD",
                "Uniswap",
                "UNI/USD",
                "DeFi Protocol",
                7,
                "Decentralized exchange protocol",
            ),
            Cryptocurrency(
                "LTCUSD",
                "Litecoin",
                "LTC/USD",
                "Peer-to-Peer",
                8,
                "Peer-to-peer cryptocurrency",
            ),
            Cryptocurrency(
                "BCHUSD",
                "Bitcoin Cash",
                "BCH/USD",
                "Peer-to-Peer",
                9,
                "Bitcoin fork for faster transactions",
            ),
            Cryptocurrency(
                "XRPUSD",
                "Ripple",
                "XRP/USD",
                "Cross-Border Payments",
                10,
                "Cross-border payment protocol",
            ),
            # Meme Coins and Popular Tokens
            Cryptocurrency(
                "DOGEUSD",
                "Dogecoin",
                "DOGE/USD",
                "Meme Coin",
                11,
                "Popular meme cryptocurrency",
            ),
            Cryptocurrency(
                "SHIBUSD",
                "Shiba Inu",
                "SHIB/USD",
                "Meme Coin",
                12,
                "Dogecoin-inspired meme token",
            ),
            Cryptocurrency(
                "PEPEUSD",
                "Pepe",
                "PEPE/USD",
                "Meme Coin",
                13,
                "Pepe the Frog-inspired meme token",
            ),
            # DeFi Tokens
            Cryptocurrency(
                "AAVEUSD",
                "Aave",
                "AAVE/USD",
                "DeFi Protocol",
                14,
                "Decentralized lending protocol",
            ),
            Cryptocurrency(
                "MKRUSD",
                "Maker",
                "MKR/USD",
                "DeFi Protocol",
                15,
                "Decentralized autonomous organization",
            ),
            Cryptocurrency(
                "CRVUSD",
                "Curve",
                "CRV/USD",
                "DeFi Protocol",
                16,
                "Decentralized exchange for stablecoins",
            ),
            Cryptocurrency(
                "SUSHIUSD",
                "SushiSwap",
                "SUSHI/USD",
                "DeFi Protocol",
                17,
                "Decentralized exchange protocol",
            ),
            Cryptocurrency(
                "YFIUSD",
                "Yearn Finance",
                "YFI/USD",
                "DeFi Protocol",
                18,
                "Yield farming aggregator",
            ),
            # Utility Tokens
            Cryptocurrency(
                "BATUSD",
                "Basic Attention Token",
                "BAT/USD",
                "Utility",
                19,
                "Digital advertising token",
            ),
            Cryptocurrency(
                "GRTUSD",
                "The Graph",
                "GRT/USD",
                "Utility",
                20,
                "Decentralized indexing protocol",
            ),
            Cryptocurrency(
                "XTZUSD",
                "Tezos",
                "XTZ/USD",
                "Smart Contract Platform",
                21,
                "Self-amending blockchain platform",
            ),
            # Stablecoins (for reference, though not typically traded for profit)
            Cryptocurrency(
                "USDCUSD",
                "USD Coin",
                "USDC/USD",
                "Stablecoin",
                22,
                "USD-backed stablecoin",
                True,
            ),
            Cryptocurrency(
                "USDTUSD",
                "Tether",
                "USDT/USD",
                "Stablecoin",
                23,
                "USD-backed stablecoin",
                True,
            ),
            Cryptocurrency(
                "DAIUSD",
                "Dai",
                "DAI/USD",
                "Stablecoin",
                24,
                "Decentralized stablecoin",
                True,
            ),
        ]

        logger.info(f"Loaded {len(crypto_data)} cryptocurrencies")
        return crypto_data

    def get_all_cryptocurrencies(self) -> List[Cryptocurrency]:
        """Get all cryptocurrencies."""
        return self.cryptocurrencies

    def get_trading_cryptocurrencies(self) -> List[Cryptocurrency]:
        """Get cryptocurrencies suitable for trading (exclude stablecoins)."""
        return [crypto for crypto in self.cryptocurrencies if not crypto.is_stablecoin]

    def get_stablecoins(self) -> List[Cryptocurrency]:
        """Get stablecoins."""
        return [crypto for crypto in self.cryptocurrencies if crypto.is_stablecoin]

    def get_cryptocurrencies_by_category(self, category: str) -> List[Cryptocurrency]:
        """Get cryptocurrencies by category."""
        return [
            crypto
            for crypto in self.cryptocurrencies
            if crypto.category.lower() == category.lower()
        ]

    def get_top_cryptocurrencies(self, limit: int = 10) -> List[Cryptocurrency]:
        """Get top N cryptocurrencies by market cap rank."""
        return sorted(self.cryptocurrencies, key=lambda x: x.market_cap_rank)[:limit]

    def get_trading_symbols(self) -> List[str]:
        """Get all trading symbols."""
        return [crypto.symbol for crypto in self.cryptocurrencies]

    def get_alpaca_symbols(self) -> List[str]:
        """Get all Alpaca format symbols."""
        return [crypto.alpaca_symbol for crypto in self.cryptocurrencies]

    def get_cryptocurrency_by_symbol(self, symbol: str) -> Optional[Cryptocurrency]:
        """Get cryptocurrency by trading symbol."""
        for crypto in self.cryptocurrencies:
            if crypto.symbol == symbol:
                return crypto
        return None

    def get_cryptocurrency_by_alpaca_symbol(
        self, alpaca_symbol: str
    ) -> Optional[Cryptocurrency]:
        """Get cryptocurrency by Alpaca symbol."""
        for crypto in self.cryptocurrencies:
            if crypto.alpaca_symbol == alpaca_symbol:
                return crypto
        return None

    def get_categories(self) -> Set[str]:
        """Get all available categories."""
        return set(crypto.category for crypto in self.cryptocurrencies)

    def symbol_to_alpaca_symbol(self, symbol: str) -> Optional[str]:
        """Convert our symbol format to Alpaca format."""
        crypto = self.get_cryptocurrency_by_symbol(symbol)
        return crypto.alpaca_symbol if crypto else None

    def alpaca_symbol_to_symbol(self, alpaca_symbol: str) -> Optional[str]:
        """Convert Alpaca format to our symbol format."""
        crypto = self.get_cryptocurrency_by_alpaca_symbol(alpaca_symbol)
        return crypto.symbol if crypto else None

    def is_available_on_alpaca(self, symbol: str) -> bool:
        """Check if symbol is available on Alpaca."""
        return self.get_cryptocurrency_by_symbol(symbol) is not None


def get_crypto_universe() -> CryptoUniverse:
    """Get crypto universe instance."""
    return CryptoUniverse()


def get_crypto_symbols() -> List[str]:
    """Get all crypto trading symbols."""
    universe = CryptoUniverse()
    return universe.get_trading_symbols()


def get_alpaca_crypto_symbols() -> List[str]:
    """Get all Alpaca format crypto symbols."""
    universe = CryptoUniverse()
    return universe.get_alpaca_symbols()


def symbol_to_alpaca_symbol(symbol: str) -> Optional[str]:
    """Convert symbol to Alpaca format."""
    universe = CryptoUniverse()
    return universe.symbol_to_alpaca_symbol(symbol)


def is_crypto_available(symbol: str) -> bool:
    """Check if crypto symbol is available on Alpaca."""
    universe = CryptoUniverse()
    return universe.is_available_on_alpaca(symbol)


if __name__ == "__main__":
    # Test the crypto universe
    universe = CryptoUniverse()
    print(f"Loaded {len(universe.cryptocurrencies)} cryptocurrencies")

    # Get categories
    categories = universe.get_categories()
    print(f"\nAvailable categories: {sorted(categories)}")

    # Get trading symbols
    symbols = universe.get_trading_symbols()
    print(f"\nTrading symbols: {symbols}")

    # Get Alpaca symbols
    alpaca_symbols = universe.get_alpaca_symbols()
    print(f"\nAlpaca symbols: {alpaca_symbols}")

    # Test symbol conversion
    print(f"\nSymbol conversion examples:")
    print(f"BTCUSD -> {universe.symbol_to_alpaca_symbol('BTCUSD')}")
    print(f"ETH/USD -> {universe.alpaca_symbol_to_symbol('ETH/USD')}")

    # Get top 10 cryptocurrencies
    top_10 = universe.get_top_cryptocurrencies(10)
    print(f"\nTop 10 cryptocurrencies by market cap:")
    for crypto in top_10:
        print(
            f"{crypto.market_cap_rank}. {crypto.name} ({crypto.symbol}) - {crypto.category}"
        )
