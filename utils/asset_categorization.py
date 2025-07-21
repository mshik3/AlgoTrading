"""
Asset categorization utilities.
Provides functionality to categorize different types of financial assets.
"""

from typing import Dict, Set, List


def categorize_asset(symbol: str) -> str:
    """
    Categorize asset by type.

    Args:
        symbol: The asset symbol to categorize

    Returns:
        The asset category as a string

    Raises:
        TypeError: If symbol is None or not a string
    """
    # Handle None and invalid inputs
    if symbol is None:
        raise TypeError("Symbol cannot be None")

    if not isinstance(symbol, str):
        raise TypeError(f"Symbol must be a string, got {type(symbol)}")

    # Handle empty string
    if not symbol.strip():
        return "Other"

    # Major US ETFs
    us_etfs = {"SPY", "QQQ", "VTI", "IWM", "VEA", "VWO", "AGG", "TLT"}
    # Sector ETFs
    sector_etfs = {"XLF", "XLK", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB"}
    # Tech Stocks
    tech_stocks = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"}
    # Financial & Industrial
    financial_stocks = {"JPM", "BAC", "WFC", "GS", "UNH", "JNJ"}
    # International ETFs
    intl_etfs = {"EFA", "EEM", "FXI", "EWJ", "EWG", "EWU"}
    # Commodity ETFs
    commodity_etfs = {"GLD", "SLV", "USO", "DBA"}
    # Crypto - Only available in Alpaca API
    crypto = {
        "BTCUSD",
        "ETHUSD",
        "DOTUSD",
        "LINKUSD",
        "LTCUSD",
        "BCHUSD",
        "XRPUSD",
        "SOLUSD",
        "AVAXUSD",
        "UNIUSD",
    }

    if symbol in us_etfs:
        return "US ETFs"
    elif symbol in sector_etfs:
        return "Sector ETFs"
    elif symbol in tech_stocks:
        return "Tech Stocks"
    elif symbol in financial_stocks:
        return "Financial Stocks"
    elif symbol in intl_etfs:
        return "International ETFs"
    elif symbol in commodity_etfs:
        return "Commodity ETFs"
    elif symbol in crypto:
        return "Crypto"
    else:
        return "Other"


def get_asset_categories() -> Dict[str, Set[str]]:
    """
    Get all asset categories and their symbols.

    Returns:
        Dictionary mapping category names to sets of symbols
    """
    return {
        "US ETFs": {"SPY", "QQQ", "VTI", "IWM", "VEA", "VWO", "AGG", "TLT"},
        "Sector ETFs": {"XLF", "XLK", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB"},
        "Tech Stocks": {
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "NVDA",
            "NFLX",
        },
        "Financial Stocks": {"JPM", "BAC", "WFC", "GS", "UNH", "JNJ"},
        "International ETFs": {"EFA", "EEM", "FXI", "EWJ", "EWG", "EWU"},
        "Commodity ETFs": {"GLD", "SLV", "USO", "DBA"},
        "Crypto": {
            "BTCUSD",
            "ETHUSD",
            "DOTUSD",
            "LINKUSD",
            "LTCUSD",
            "BCHUSD",
            "XRPUSD",
            "SOLUSD",
            "AVAXUSD",
            "UNIUSD",
        },
    }


def get_etf_rotation_universes() -> Dict[str, Dict[str, List[str]]]:
    """
    Get comprehensive ETF universes for rotation strategies.

    Returns:
        Dictionary mapping strategy type -> ETF universe configuration
    """
    return {
        "dual_momentum": {
            "US_Equities": ["SPY", "QQQ", "VTI", "IWM"],
            "International": ["EFA", "EEM", "VEA", "VWO"],
            "Bonds": ["TLT", "AGG", "BND", "LQD"],
            "Real_Estate": ["VNQ", "IYR", "SCHH"],
            "Commodities": ["GLD", "SLV", "USO", "DBA"],
            "Cash_Equivalents": ["SHY", "BIL", "SHV"],
        },
        "sector_rotation": {
            "Technology": ["XLK", "VGT", "SMH"],
            "Financials": ["XLF", "VFH", "KBE"],
            "Healthcare": ["XLV", "VHT", "IHI"],
            "Consumer_Discretionary": ["XLY", "VCR", "XRT"],
            "Consumer_Staples": ["XLP", "VDC", "XLP"],
            "Industrials": ["XLI", "VIS", "XAR"],
            "Energy": ["XLE", "VDE", "XOP"],
            "Materials": ["XLB", "VAW", "XME"],
            "Real_Estate": ["XLRE", "VNQ", "IYR"],
            "Utilities": ["XLU", "VPU", "XLU"],
            "Communications": ["XLC", "VOX", "XLC"],
        },
        "multi_asset": {
            "US_Stocks": ["SPY", "QQQ", "VTI", "IWM"],
            "International_Stocks": ["EFA", "EEM", "VEA", "VWO"],
            "Bonds": ["TLT", "AGG", "BND", "LQD"],
            "Real_Estate": ["VNQ", "IYR", "SCHH"],
            "Commodities": ["GLD", "SLV", "USO", "DBA"],
            "Cash": ["SHY", "BIL", "SHV"],
        },
    }


def get_etf_universe_for_strategy(strategy_type: str) -> Dict[str, List[str]]:
    """
    Get ETF universe for a specific rotation strategy.

    Args:
        strategy_type: Type of rotation strategy ('dual_momentum', 'sector_rotation', 'multi_asset')

    Returns:
        ETF universe configuration for the strategy
    """
    universes = get_etf_rotation_universes()
    return universes.get(strategy_type, {})


def is_crypto_symbol(symbol: str) -> bool:
    """
    Check if a symbol is a crypto asset.

    Args:
        symbol: The asset symbol to check

    Returns:
        True if the symbol is a crypto asset, False otherwise
    """
    return categorize_asset(symbol) == "Crypto"


def is_etf_symbol(symbol: str) -> bool:
    """
    Check if a symbol is an ETF.

    Args:
        symbol: The asset symbol to check

    Returns:
        True if the symbol is an ETF, False otherwise
    """
    category = categorize_asset(symbol)
    return category in [
        "US ETFs",
        "Sector ETFs",
        "International ETFs",
        "Commodity ETFs",
    ]
