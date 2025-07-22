"""
Symbol normalization utilities for Alpaca API compatibility.
Handles special characters and symbol mapping for different data sources.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Stock symbol mapping for Alpaca API compatibility
# Maps symbols with special characters to their Alpaca-compatible equivalents
STOCK_SYMBOL_MAPPING = {
    # Berkshire Hathaway variants
    "BRK-B": "BRKB",
    "BRK.B": "BRKB",
    "BRK_A": "BRKA",
    "BRK-A": "BRKA",
    # Other common symbols with special characters
    "BF.B": "BFB",
    "BF-B": "BFB",
    # Add more mappings as needed
}

# Symbols known to be unavailable in Alpaca's API
UNAVAILABLE_SYMBOLS = {
    "BRK-B",
    "BRKB",
    "BRK.B",  # Berkshire Hathaway B shares
    "BRK-A",
    "BRKA",
    "BRK_A",  # Berkshire Hathaway A shares
    "BF.B",
    "BFB",
    "BF-B",  # Brown-Forman B shares
    "VWEHX",  # Vanguard High-Yield Corporate Fund
    # Add more unavailable symbols as discovered
}

# Fallback symbol mapping for unavailable symbols
# Maps unavailable symbols to alternative symbols that are available
FALLBACK_SYMBOL_MAPPING = {
    "BRK-B": "JPM",  # Replace with JPMorgan Chase (major financial)
    "BRKB": "JPM",  # Normalized version
    "BRK.B": "JPM",  # Alternative format
    "BRK-A": "JPM",  # A shares also map to JPM
    "BRKA": "JPM",  # Normalized version
    "BRK_A": "JPM",  # Alternative format
    "BF.B": "KO",  # Replace with Coca-Cola (consumer staples)
    "BFB": "KO",  # Normalized version
    "BF-B": "KO",  # Alternative format
    "VWEHX": "HYG",  # Replace with iShares High Yield Corporate Bond ETF
}

# Reverse mapping for display purposes
# Use the first occurrence when multiple symbols map to the same normalized symbol
REVERSE_SYMBOL_MAPPING = {}
for original, normalized in STOCK_SYMBOL_MAPPING.items():
    if normalized not in REVERSE_SYMBOL_MAPPING:
        REVERSE_SYMBOL_MAPPING[normalized] = original


def is_symbol_available_for_alpaca(symbol: str) -> bool:
    """
    Check if a symbol is available in Alpaca's API.

    Args:
        symbol: Symbol to check

    Returns:
        True if symbol is available, False if known to be unavailable
    """
    if not symbol:
        return False

    # Check if symbol is in our unavailable list
    if symbol in UNAVAILABLE_SYMBOLS:
        return False

    # Check if normalized symbol is unavailable
    normalized = normalize_symbol_for_alpaca(symbol)
    if normalized in UNAVAILABLE_SYMBOLS:
        return False

    return True


def get_fallback_symbol(symbol: str) -> Optional[str]:
    """
    Get a fallback symbol for an unavailable symbol.

    Args:
        symbol: Original symbol that is unavailable

    Returns:
        Fallback symbol if available, None otherwise
    """
    if not symbol:
        return None

    # Check direct mapping first
    if symbol in FALLBACK_SYMBOL_MAPPING:
        return FALLBACK_SYMBOL_MAPPING[symbol]

    # Check normalized symbol
    normalized = normalize_symbol_for_alpaca(symbol)
    if normalized in FALLBACK_SYMBOL_MAPPING:
        return FALLBACK_SYMBOL_MAPPING[normalized]

    return None


def normalize_symbol_for_alpaca(symbol: str) -> str:
    """
    Normalize a symbol for Alpaca API compatibility.

    Args:
        symbol: Original symbol (e.g., "BRK-B")

    Returns:
        Normalized symbol for Alpaca API (e.g., "BRKB")
    """
    if not symbol:
        return symbol

    # Check if symbol is unavailable and has a fallback
    if symbol in FALLBACK_SYMBOL_MAPPING:
        fallback = FALLBACK_SYMBOL_MAPPING[symbol]
        logger.info(f"Symbol {symbol} is unavailable, using fallback: {fallback}")
        return fallback

    # Check if we have a direct mapping
    if symbol in STOCK_SYMBOL_MAPPING:
        normalized = STOCK_SYMBOL_MAPPING[symbol]
        logger.debug(f"Normalized symbol {symbol} -> {normalized}")
        return normalized

    # For symbols not in our mapping, remove hyphens and dots
    # This handles most common cases
    normalized = symbol.replace("-", "").replace(".", "")

    if normalized != symbol:
        logger.debug(f"Auto-normalized symbol {symbol} -> {normalized}")

    return normalized


def denormalize_symbol_for_display(symbol: str) -> str:
    """
    Convert a normalized symbol back to its display format.

    Args:
        symbol: Normalized symbol (e.g., "BRKB")

    Returns:
        Display symbol (e.g., "BRK-B")
    """
    if not symbol:
        return symbol

    # Check if we have a reverse mapping
    if symbol in REVERSE_SYMBOL_MAPPING:
        display_symbol = REVERSE_SYMBOL_MAPPING[symbol]
        logger.debug(f"Denormalized symbol {symbol} -> {display_symbol}")
        return display_symbol

    return symbol


def is_symbol_normalized(symbol: str) -> bool:
    """
    Check if a symbol needs normalization for Alpaca API.

    Args:
        symbol: Symbol to check

    Returns:
        True if symbol contains special characters that need normalization
    """
    if not symbol:
        return False

    # Check if it's in our mapping
    if symbol in STOCK_SYMBOL_MAPPING:
        return True

    # Check for common special characters
    return "-" in symbol or "." in symbol


def get_symbol_mapping_info(symbol: str) -> Dict[str, str]:
    """
    Get information about symbol mapping.

    Args:
        symbol: Symbol to get info for

    Returns:
        Dictionary with mapping information
    """
    normalized = normalize_symbol_for_alpaca(symbol)
    display = denormalize_symbol_for_display(normalized)

    return {
        "original": symbol,
        "normalized": normalized,
        "display": display,
        "needs_normalization": is_symbol_normalized(symbol),
        "is_mapped": symbol in STOCK_SYMBOL_MAPPING
        or normalized in REVERSE_SYMBOL_MAPPING,
    }


def add_symbol_mapping(original_symbol: str, normalized_symbol: str) -> None:
    """
    Add a new symbol mapping to the system.

    Args:
        original_symbol: Original symbol with special characters
        normalized_symbol: Alpaca-compatible symbol
    """
    STOCK_SYMBOL_MAPPING[original_symbol] = normalized_symbol
    # Only add to reverse mapping if not already present (preserve first occurrence)
    if normalized_symbol not in REVERSE_SYMBOL_MAPPING:
        REVERSE_SYMBOL_MAPPING[normalized_symbol] = original_symbol
    logger.info(f"Added symbol mapping: {original_symbol} -> {normalized_symbol}")


def get_all_mappings() -> Dict[str, str]:
    """
    Get all current symbol mappings.

    Returns:
        Dictionary of all symbol mappings
    """
    return STOCK_SYMBOL_MAPPING.copy()
