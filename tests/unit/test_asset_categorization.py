from utils.asset_categorization import categorize_asset

"""
Unit tests for asset categorization functionality.
Tests the categorization of different asset types including crypto.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestAssetCategorization:
    """Test asset categorization functionality."""

    def test_us_etf_categorization(self):
        """Test categorization of US ETFs."""
        us_etfs = ["SPY", "QQQ", "VTI", "IWM", "VEA", "VWO", "AGG", "TLT"]

        for symbol in us_etfs:
            category = categorize_asset(symbol)
            assert (
                any(
                    x in category
                    for x in [
                        "Large Cap",
                        "Small Cap",
                        "Total Market",
                        "Bond",
                        "International Developed",
                        "International Emerging",
                    ]
                )
                and "ETFs" in category
            ), f"Symbol {symbol} should be categorized as an ETF type"

    def test_sector_etf_categorization(self):
        """Test categorization of sector ETFs."""
        sector_etfs = ["XLF", "XLK", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB"]

        for symbol in sector_etfs:
            category = categorize_asset(symbol)
            assert "ETFs" in category and any(
                sector in category
                for sector in [
                    "Financials",
                    "Technology",
                    "Healthcare",
                    "Energy",
                    "Industrials",
                    "Consumer",
                    "Utilities",
                    "Materials",
                ]
            ), f"Symbol {symbol} should be categorized as a sector ETF"

    def test_tech_stock_categorization(self):
        """Test categorization of tech stocks."""
        tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"]

        for symbol in tech_stocks:
            category = categorize_asset(symbol)
            assert (
                any(
                    x in category
                    for x in [
                        "Technology",
                        "Consumer Discretionary",
                        "Communication Services",
                    ]
                )
                and "Stocks" in category
            ), f"Symbol {symbol} should be categorized as a tech-related stock"

    def test_financial_stock_categorization(self):
        """Test categorization of financial stocks."""
        financial_stocks = ["JPM", "BAC", "WFC", "GS", "UNH", "JNJ"]

        for symbol in financial_stocks:
            category = categorize_asset(symbol)
            assert (
                any(x in category for x in ["Financials", "Healthcare"])
                and "Stocks" in category
            ), f"Symbol {symbol} should be categorized as a financial or healthcare stock"

    def test_international_etf_categorization(self):
        """Test categorization of international ETFs."""
        international_etfs = ["EFA", "EEM", "FXI", "EWJ", "EWG", "EWU"]

        for symbol in international_etfs:
            category = categorize_asset(symbol)
            assert (
                "International" in category and "ETFs" in category
            ), f"Symbol {symbol} should be categorized as an International ETF type"

    def test_commodity_etf_categorization(self):
        """Test categorization of commodity ETFs."""
        commodity_etfs = ["GLD", "SLV", "USO", "DBA"]

        for symbol in commodity_etfs:
            category = categorize_asset(symbol)
            assert (
                category == "Commodities ETFs"
            ), f"Symbol {symbol} should be categorized as Commodities ETFs"

    def test_crypto_categorization(self):
        """Test categorization of crypto assets."""
        crypto_symbols = [
            "BTCUSD",
            "ETHUSD",
            "SOLUSD",
            "DOTUSD",
            "LINKUSD",
            "LTCUSD",
            "BCHUSD",
            "XRPUSD",
            "SOLUSD",
            "MATICUSD",
        ]

        for symbol in crypto_symbols:
            category = categorize_asset(symbol)
            assert (
                "Crypto" in category
            ), f"Symbol {symbol} should be categorized as a Crypto type"

    def test_unknown_symbol_categorization(self):
        """Test categorization of unknown symbols."""

        unknown_symbols = ["UNKNOWN", "TEST123", "XYZ", "ABC", "123"]

        for symbol in unknown_symbols:
            category = categorize_asset(symbol)
            assert (
                category == "Other"
            ), f"Unknown symbol {symbol} should be categorized as Other"

    def test_case_sensitivity(self):
        """Test that categorization is case sensitive."""

        # Test that lowercase versions are not recognized
        assert categorize_asset("spy") == "Other"  # Should not match SPY
        assert categorize_asset("btcusd") == "Other"  # Should not match BTCUSD
        assert categorize_asset("aapl") == "Other"  # Should not match AAPL

    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs."""

        # Test empty string
        assert categorize_asset("") == "Other"

        # Test None (should raise TypeError)
        with pytest.raises(TypeError):
            categorize_asset(None)

    def test_symbol_patterns(self):
        """Test that symbol patterns are correctly identified."""
        # Test USD suffix pattern for crypto
        usd_symbols = ["BTCUSD", "ETHUSD", "DOTUSD", "LINKUSD", "SOLUSD"]
        for symbol in usd_symbols:
            category = categorize_asset(symbol)
            assert (
                "Crypto" in category
            ), f"Symbol {symbol} should be categorized as a Crypto type"

        # Test that non-crypto USD symbols are not misclassified
        non_crypto_usd = ["TESTUSD", "FAKEUSD", "MOCKUSD"]
        for symbol in non_crypto_usd:
            assert categorize_asset(symbol) == "Other"

    def test_categorization_completeness(self):
        """Test that all symbols in the strategy are properly categorized."""
        from strategies.equity.golden_cross import GoldenCrossStrategy

        strategy = GoldenCrossStrategy()

        # Test all symbols in the strategy
        for symbol in strategy.symbols:
            category = categorize_asset(symbol)

            # Verify category is one of the expected types
            assert (
                "Stocks" in category
                or "ETFs" in category
                or "Crypto" in category
                or category == "Other"
            ), f"Symbol {symbol} has unexpected category: {category}"

            # Verify category is not "Other" for known symbols
            assert (
                category != "Other"
            ), f"Known symbol {symbol} should not be categorized as Other"

    def test_categorization_distribution(self):
        """Test that categorization produces expected distribution."""
        from strategies.equity.golden_cross import GoldenCrossStrategy

        strategy = GoldenCrossStrategy()

        # Count categories by type
        stocks = 0
        etfs = 0
        crypto = 0
        other = 0

        for symbol in strategy.symbols:
            category = categorize_asset(symbol)
            if "Stocks" in category:
                stocks += 1
            elif "ETFs" in category:
                etfs += 1
            elif "Crypto" in category:
                crypto += 1
            else:
                other += 1

        # Verify expected distribution
        assert stocks >= 20, f"Expected at least 20 stocks, got {stocks}"
        assert etfs >= 15, f"Expected at least 15 ETFs, got {etfs}"
        assert crypto >= 8, f"Expected at least 8 crypto assets, got {crypto}"
        assert other == 0, f"Expected 0 other assets, got {other}"

        # Verify total count
        total = stocks + etfs + crypto + other
        assert total == len(
            strategy.symbols
        ), f"Expected {len(strategy.symbols)} total symbols, got {total}"

    def test_categorization_performance(self):
        """Test that categorization is performant."""
        import time

        # Test symbols
        test_symbols = [
            "SPY",
            "AAPL",
            "BTCUSD",
            "GLD",
            "EFA",
            "XLF",
        ] * 1000  # 6000 symbols

        # Time the categorization
        start_time = time.time()
        for symbol in test_symbols:
            categorize_asset(symbol)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second for 6000 symbols)
        duration = end_time - start_time
        assert duration < 1.0, f"Categorization took too long: {duration:.3f} seconds"

    def test_categorization_edge_cases(self):
        """Test edge cases in categorization."""

        # Test very long symbols
        long_symbol = "A" * 1000
        assert categorize_asset(long_symbol) == "Other"

        # Test symbols with special characters
        special_symbols = ["SPY!", "AAPL@", "BTC#USD", "GLD$", "EFA%"]
        for symbol in special_symbols:
            assert categorize_asset(symbol) == "Other"

        # Test numeric symbols
        numeric_symbols = ["123", "456", "789", "000"]
        for symbol in numeric_symbols:
            assert categorize_asset(symbol) == "Other"

    def test_categorization_consistency(self):
        """Test that categorization is consistent across multiple calls."""

        test_symbols = ["SPY", "AAPL", "BTCUSD", "GLD", "EFA", "XLF"]

        # Test multiple calls for each symbol
        for symbol in test_symbols:
            first_result = categorize_asset(symbol)

            # Call multiple times
            for _ in range(10):
                result = categorize_asset(symbol)
                assert (
                    result == first_result
                ), f"Inconsistent categorization for {symbol}: {first_result} vs {result}"

    def test_categorization_with_whitespace(self):
        """Test that categorization handles whitespace correctly."""

        # Test symbols with leading/trailing whitespace
        assert categorize_asset(" SPY") == "Other"  # Leading space
        assert categorize_asset("SPY ") == "Other"  # Trailing space
        assert categorize_asset(" SPY ") == "Other"  # Both spaces
        assert categorize_asset("  SPY  ") == "Other"  # Multiple spaces

        # Test symbols with internal whitespace
        assert categorize_asset("SP Y") == "Other"  # Internal space
        assert categorize_asset("S P Y") == "Other"  # Multiple internal spaces

    def test_categorization_with_unicode(self):
        """Test that categorization handles unicode characters correctly."""

        # Test unicode symbols
        unicode_symbols = ["SPY🚀", "AAPL📱", "BTCUSD₿", "GLD💰", "EFA🌍"]
        for symbol in unicode_symbols:
            assert categorize_asset(symbol) == "Other"

        # Test unicode-only symbols
        unicode_only = ["🚀", "📱", "₿", "💰", "🌍"]
        for symbol in unicode_only:
            assert categorize_asset(symbol) == "Other"
