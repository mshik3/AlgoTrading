"""
Tests for symbol normalization utilities.
"""

import pytest
from utils.symbol_normalization import (
    normalize_symbol_for_alpaca,
    denormalize_symbol_for_display,
    is_symbol_normalized,
    get_symbol_mapping_info,
    add_symbol_mapping,
    get_all_mappings,
    is_symbol_available_for_alpaca,
    get_fallback_symbol,
)


class TestSymbolNormalization:
    """Test symbol normalization functionality."""

    def test_brk_b_normalization(self):
        """Test BRK-B normalization to fallback symbol."""
        normalized = normalize_symbol_for_alpaca("BRK-B")
        assert normalized == "JPM"  # Now returns fallback symbol

        # Test reverse mapping (should still work for display)
        display = denormalize_symbol_for_display("BRKB")
        assert display == "BRK-B"

    def test_brk_dot_b_normalization(self):
        """Test BRK.B normalization to fallback symbol."""
        normalized = normalize_symbol_for_alpaca("BRK.B")
        assert normalized == "JPM"  # Now returns fallback symbol

    def test_bf_b_normalization(self):
        """Test BF.B normalization to fallback symbol."""
        normalized = normalize_symbol_for_alpaca("BF.B")
        assert normalized == "KO"  # Now returns fallback symbol

    def test_auto_normalization(self):
        """Test automatic normalization for symbols not in mapping."""
        # Test hyphen removal
        assert normalize_symbol_for_alpaca("TEST-A") == "TESTA"
        assert normalize_symbol_for_alpaca("TEST-B") == "TESTB"

        # Test dot removal
        assert normalize_symbol_for_alpaca("TEST.A") == "TESTA"
        assert normalize_symbol_for_alpaca("TEST.B") == "TESTB"

    def test_normal_symbols_unchanged(self):
        """Test that normal symbols are not changed."""
        normal_symbols = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
        for symbol in normal_symbols:
            assert normalize_symbol_for_alpaca(symbol) == symbol

    def test_is_symbol_normalized(self):
        """Test symbol normalization detection."""
        assert is_symbol_normalized("BRK-B") == True
        assert is_symbol_normalized("BF.B") == True
        assert is_symbol_normalized("TEST-A") == True
        assert is_symbol_normalized("AAPL") == False
        assert is_symbol_normalized("") == False

    def test_get_symbol_mapping_info(self):
        """Test symbol mapping info retrieval."""
        info = get_symbol_mapping_info("BRK-B")
        assert info["original"] == "BRK-B"
        assert info["normalized"] == "JPM"  # Now returns fallback symbol
        assert info["display"] == "JPM"  # Display is the same as normalized for fallbacks
        assert info["needs_normalization"] == True
        assert info["is_mapped"] == True

    def test_add_symbol_mapping(self):
        """Test adding new symbol mappings."""
        # Add a new mapping
        add_symbol_mapping("NEW-SYMBOL", "NEWSYMBOL")

        # Test the new mapping
        assert normalize_symbol_for_alpaca("NEW-SYMBOL") == "NEWSYMBOL"
        assert denormalize_symbol_for_display("NEWSYMBOL") == "NEW-SYMBOL"

        # Clean up - remove the test mapping
        from utils.symbol_normalization import (
            STOCK_SYMBOL_MAPPING,
            REVERSE_SYMBOL_MAPPING,
        )

        del STOCK_SYMBOL_MAPPING["NEW-SYMBOL"]
        del REVERSE_SYMBOL_MAPPING["NEWSYMBOL"]

    def test_get_all_mappings(self):
        """Test getting all symbol mappings."""
        mappings = get_all_mappings()
        assert isinstance(mappings, dict)
        assert "BRK-B" in mappings
        assert mappings["BRK-B"] == "BRKB"

    def test_symbol_availability(self):
        """Test symbol availability checking."""
        # Test unavailable symbols
        assert is_symbol_available_for_alpaca("BRK-B") == False
        assert is_symbol_available_for_alpaca("BRKB") == False
        assert is_symbol_available_for_alpaca("BF.B") == False
        assert is_symbol_available_for_alpaca("BFB") == False
        assert is_symbol_available_for_alpaca("VWEHX") == False

        # Test available symbols
        assert is_symbol_available_for_alpaca("AAPL") == True
        assert is_symbol_available_for_alpaca("MSFT") == True
        assert is_symbol_available_for_alpaca("SPY") == True

    def test_fallback_symbols(self):
        """Test fallback symbol functionality."""
        # Test fallback symbols
        assert get_fallback_symbol("BRK-B") == "JPM"
        assert get_fallback_symbol("BRKB") == "JPM"
        assert get_fallback_symbol("BF.B") == "KO"
        assert get_fallback_symbol("BFB") == "KO"
        assert get_fallback_symbol("VWEHX") == "HYG"

        # Test symbols without fallbacks
        assert get_fallback_symbol("AAPL") == None
        assert get_fallback_symbol("MSFT") == None

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty string
        assert normalize_symbol_for_alpaca("") == ""
        assert denormalize_symbol_for_display("") == ""

        # None (should handle gracefully)
        assert normalize_symbol_for_alpaca(None) == None
        assert denormalize_symbol_for_display(None) == None

        # Multiple special characters
        assert normalize_symbol_for_alpaca("TEST-A.B") == "TESTAB"
