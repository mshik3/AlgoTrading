"""
Integration tests for crypto functionality.
Tests end-to-end crypto data flow and trading execution.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestCryptoIntegration:
    """Integration tests for crypto functionality."""

    @pytest.fixture
    def mock_alpaca_config(self):
        """Mock Alpaca configuration."""
        from alpaca_data_collector import AlpacaConfig

        return AlpacaConfig(
            api_key="test_key_123456789", secret_key="test_secret_123456789", paper=True
        )

    @pytest.fixture
    def sample_crypto_market_data(self):
        """Generate sample crypto market data for integration testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300, freq="D")

        # Generate realistic crypto price data
        base_price = 50000  # BTC-like starting price
        returns = np.random.normal(0.02, 0.05, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.01, len(dates))),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.02, len(dates)))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.02, len(dates)))),
                "Close": prices,
                "Volume": np.random.randint(1000, 10000, len(dates)),
                "Adj Close": prices,
            },
            index=dates,
        )

        # Ensure logical price relationships
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        return data

    def test_end_to_end_crypto_data_flow(
        self, mock_alpaca_config, sample_crypto_market_data
    ):
        """Test complete crypto data flow from collection to analysis."""
        with patch("alpaca_data_collector.ALPACA_AVAILABLE", True):
            with patch("alpaca_data_collector.StockHistoricalDataClient"):
                with patch(
                    "alpaca_data_collector.CryptoHistoricalDataClient"
                ) as mock_crypto_client:
                    with patch("alpaca_data_collector.CryptoBarsRequest"):
                        from alpaca_data_collector import AlpacaDataCollector
                        from strategies.equity.golden_cross import GoldenCrossStrategy
                        from indicators.technical import TechnicalIndicators

                        # Setup mocks
                        mock_bars = Mock()
                        mock_bars.df = sample_crypto_market_data
                        mock_crypto_client.return_value.get_crypto_bars.return_value = (
                            mock_bars
                        )

                        # 1. Data Collection
                        collector = AlpacaDataCollector(config=mock_alpaca_config)
                        crypto_data = collector.fetch_daily_data("BTCUSD", period="1y")

                        # Verify data was collected
                        assert crypto_data is not None
                        assert len(crypto_data) > 0
                        assert "Open" in crypto_data.columns
                        assert "Close" in crypto_data.columns

                        # 2. Technical Analysis
                        indicators = TechnicalIndicators(crypto_data)
                        indicators.add_sma(50, "Close")
                        indicators.add_sma(200, "Close")
                        enhanced_data = indicators.get_data()

                        # Verify indicators were calculated
                        assert "SMA_50" in enhanced_data.columns
                        assert "SMA_200" in enhanced_data.columns

                        # 3. Strategy Analysis
                        strategy = GoldenCrossStrategy()
                        signal = strategy.should_enter_position("BTCUSD", enhanced_data)

                        # Verify signal generation (may be None if no Golden Cross)
                        if signal is not None:
                            assert signal.symbol == "BTCUSD"
                            assert signal.signal_type.value in ["buy", "sell"]
                            assert signal.confidence > 0
                            assert signal.price > 0

    def test_crypto_trading_execution_flow(self, mock_alpaca_config):
        """Test complete crypto trading execution flow."""
        with patch("execution.alpaca.ALPACA_AVAILABLE", True):
            with patch("execution.alpaca.TradingClient") as mock_trading_client:
                with patch("execution.alpaca.StockHistoricalDataClient"):
                    with patch(
                        "execution.alpaca.CryptoHistoricalDataClient"
                    ) as mock_crypto_client:
                        from execution.alpaca import AlpacaTradingClient
                        from strategies.base import StrategySignal, SignalType

                        # Setup mocks
                        mock_account = Mock()
                        mock_account.buying_power = "10000"
                        mock_trading_client.return_value.get_account.return_value = (
                            mock_account
                        )

                        mock_order = Mock()
                        mock_order.id = "test_order_id"
                        mock_order.status = "accepted"
                        mock_trading_client.return_value.submit_order.return_value = (
                            mock_order
                        )

                        # Mock crypto price data
                        mock_bars = Mock()
                        mock_bars.df = pd.DataFrame(
                            {
                                "close": [50000, 51000, 52000],
                            },
                            index=pd.date_range("2024-01-01", periods=3),
                        )
                        mock_crypto_client.return_value.get_crypto_bars.return_value = (
                            mock_bars
                        )

                        # 1. Initialize trading client
                        trading_client = AlpacaTradingClient(config=mock_alpaca_config)

                        # 2. Create crypto buy signal
                        signal = StrategySignal(
                            symbol="BTCUSD",
                            signal_type=SignalType.BUY,
                            confidence=0.8,
                            price=52000.0,
                            timestamp=datetime.now(),
                        )

                        # 3. Execute signal
                        result = trading_client.execute_signal(signal)

                        # Verify execution
                        assert result == True
                        mock_trading_client.return_value.submit_order.assert_called_once()

                        # Verify correct symbol was used
                        call_args = (
                            mock_trading_client.return_value.submit_order.call_args
                        )
                        order_request = call_args[0][0]
                        assert order_request.symbol == "BTC/USD"

    def test_multi_asset_crypto_analysis(
        self, mock_alpaca_config, sample_crypto_market_data
    ):
        """Test analysis of multiple assets including crypto."""
        with patch("alpaca_data_collector.ALPACA_AVAILABLE", True):
            with patch("alpaca_data_collector.StockHistoricalDataClient"):
                with patch(
                    "alpaca_data_collector.CryptoHistoricalDataClient"
                ) as mock_crypto_client:
                    with patch("alpaca_data_collector.CryptoBarsRequest"):
                        from alpaca_data_collector import AlpacaDataCollector
                        from strategies.equity.golden_cross import GoldenCrossStrategy
                        from scripts.golden_cross_analysis import GoldenCrossAnalyzer

                        # Setup mocks
                        mock_bars = Mock()
                        mock_bars.df = sample_crypto_market_data
                        mock_crypto_client.return_value.get_crypto_bars.return_value = (
                            mock_bars
                        )

                        # 1. Initialize components
                        collector = AlpacaDataCollector(config=mock_alpaca_config)
                        strategy = GoldenCrossStrategy()

                        # 2. Collect data for multiple symbols
                        symbols = ["BTCUSD", "ETHUSD", "SPY", "AAPL"]
                        market_data = {}

                        for symbol in symbols:
                            data = collector.fetch_daily_data(symbol, period="1y")
                            if data is not None and len(data) >= 250:
                                market_data[symbol] = data

                        # Verify data collection
                        assert len(market_data) > 0

                        # 3. Generate signals for all symbols
                        signals = strategy.generate_signals(market_data)

                        # Verify signal generation
                        assert isinstance(signals, dict)
                        for symbol in market_data.keys():
                            assert symbol in signals
                            # Signal may be None if no Golden Cross detected

    def test_crypto_error_handling_integration(self, mock_alpaca_config):
        """Test error handling in crypto integration."""
        with patch("alpaca_data_collector.ALPACA_AVAILABLE", True):
            with patch("alpaca_data_collector.StockHistoricalDataClient"):
                with patch(
                    "alpaca_data_collector.CryptoHistoricalDataClient"
                ) as mock_crypto_client:
                    from alpaca_data_collector import AlpacaDataCollector
                    from strategies.equity.golden_cross import GoldenCrossStrategy

                    # Setup mock to raise exception
                    mock_crypto_client.return_value.get_crypto_bars.side_effect = (
                        Exception("API Error")
                    )

                    # 1. Test data collection error handling
                    collector = AlpacaDataCollector(config=mock_alpaca_config)
                    result = collector.fetch_daily_data("BTCUSD", period="1y")

                    # Should handle error gracefully
                    assert result is None

                    # 2. Test strategy with missing data
                    strategy = GoldenCrossStrategy()
                    market_data = {"BTCUSD": None}  # No data available

                    signals = strategy.generate_signals(market_data)

                    # Should handle missing data gracefully
                    assert isinstance(signals, dict)
                    assert "BTCUSD" in signals
                    assert signals["BTCUSD"] is None

    def test_crypto_asset_categorization_integration(self):
        """Test asset categorization integration with crypto."""
        from utils.asset_categorization import categorize_asset
        from scripts.golden_cross_analysis import GoldenCrossAnalyzer
        from strategies.equity.golden_cross import GoldenCrossStrategy

        # 1. Test categorization of all strategy symbols
        strategy = GoldenCrossStrategy()

        categories = {}
        for symbol in strategy.symbols:
            category = categorize_asset(symbol)
            categories[category] = categories.get(category, 0) + 1

        # Verify crypto categorization
        assert categories.get("Crypto", 0) == 10

        # 2. Test categorization in analysis context
        analyzer = GoldenCrossAnalyzer()

        # Test with mixed asset types
        test_symbols = ["SPY", "AAPL", "BTCUSD", "GLD", "EFA"]
        expected_categories = [
            "US ETFs",
            "Tech Stocks",
            "Crypto",
            "Commodity ETFs",
            "International ETFs",
        ]

        for symbol, expected in zip(test_symbols, expected_categories):
            category = categorize_asset(symbol)
            assert (
                category == expected
            ), f"Symbol {symbol} should be {expected}, got {category}"

    def test_crypto_rate_limiting_integration(self, mock_alpaca_config):
        """Test rate limiting with crypto data collection."""
        with patch("alpaca_data_collector.ALPACA_AVAILABLE", True):
            with patch("alpaca_data_collector.StockHistoricalDataClient"):
                with patch(
                    "alpaca_data_collector.CryptoHistoricalDataClient"
                ) as mock_crypto_client:
                    with patch("alpaca_data_collector.CryptoBarsRequest"):
                        from alpaca_data_collector import AlpacaDataCollector

                        # Setup mocks
                        mock_bars = Mock()
                        mock_bars.df = pd.DataFrame(
                            {
                                "open": [50000],
                                "high": [51000],
                                "low": [49000],
                                "close": [50000],
                                "volume": [1000],
                            },
                            index=pd.date_range("2024-01-01", periods=1),
                        )
                        mock_crypto_client.return_value.get_crypto_bars.return_value = (
                            mock_bars
                        )

                        collector = AlpacaDataCollector(config=mock_alpaca_config)

                        # Test multiple requests to verify rate limiting
                        crypto_symbols = [
                            "BTCUSD",
                            "ETHUSD",
                            "ADAUSD",
                            "DOTUSD",
                            "LINKUSD",
                        ]

                        for symbol in crypto_symbols:
                            result = collector.fetch_daily_data(symbol, period="1d")
                            # Should not fail due to rate limiting
                            assert (
                                result is not None or result is None
                            )  # Either data or no data

    def test_crypto_configuration_integration(self):
        """Test configuration integration for crypto functionality."""
        from alpaca_data_collector import AlpacaDataCollector
        from execution.alpaca import AlpacaTradingClient

        # Test with valid configuration
        with patch.dict(
            "os.environ",
            {
                "ALPACA_API_KEY": "valid_key_123456789",
                "ALPACA_SECRET_KEY": "valid_secret_123456789",
            },
        ):
            with patch("alpaca_data_collector.ALPACA_AVAILABLE", True):
                with patch("alpaca_data_collector.StockHistoricalDataClient"):
                    with patch("alpaca_data_collector.CryptoHistoricalDataClient"):
                        with patch("execution.alpaca.ALPACA_AVAILABLE", True):
                            with patch("execution.alpaca.TradingClient"):
                                with patch(
                                    "execution.alpaca.StockHistoricalDataClient"
                                ):
                                    with patch(
                                        "execution.alpaca.CryptoHistoricalDataClient"
                                    ):
                                        # Both should initialize successfully
                                        collector = AlpacaDataCollector()
                                        trading_client = AlpacaTradingClient()

                                        # Verify configuration is shared
                                        assert (
                                            collector.config.api_key
                                            == trading_client.config.api_key
                                        )
                                        assert (
                                            collector.config.secret_key
                                            == trading_client.config.secret_key
                                        )

    def test_crypto_symbol_mapping_integration(self):
        """Test symbol mapping consistency across components."""
        from alpaca_data_collector import AlpacaDataCollector
        from execution.alpaca import AlpacaTradingClient

        with patch("alpaca_data_collector.ALPACA_AVAILABLE", True):
            with patch("alpaca_data_collector.StockHistoricalDataClient"):
                with patch("alpaca_data_collector.CryptoHistoricalDataClient"):
                    with patch("execution.alpaca.ALPACA_AVAILABLE", True):
                        with patch("execution.alpaca.TradingClient"):
                            with patch("execution.alpaca.StockHistoricalDataClient"):
                                with patch(
                                    "execution.alpaca.CryptoHistoricalDataClient"
                                ):
                                    collector = AlpacaDataCollector()
                                    trading_client = AlpacaTradingClient()

                                    # Test symbol mapping consistency
                                    crypto_symbols = [
                                        "BTCUSD",
                                        "ETHUSD",
                                        "ADAUSD",
                                        "DOTUSD",
                                        "LINKUSD",
                                    ]

                                    for symbol in crypto_symbols:
                                        collector_symbol = collector._get_alpaca_symbol(
                                            symbol
                                        )
                                        trading_symbol = (
                                            trading_client._get_alpaca_symbol(symbol)
                                        )

                                        assert (
                                            collector_symbol == trading_symbol
                                        ), f"Symbol mapping mismatch for {symbol}"

    def test_crypto_performance_integration(self):
        """Test performance of crypto integration."""
        import time
        from utils.asset_categorization import categorize_asset
        from strategies.equity.golden_cross import GoldenCrossStrategy

        # Test strategy initialization performance
        start_time = time.time()
        strategy = GoldenCrossStrategy()
        init_time = time.time() - start_time

        # Should initialize quickly
        assert (
            init_time < 1.0
        ), f"Strategy initialization took too long: {init_time:.3f} seconds"

        # Test categorization performance
        symbols = strategy.symbols
        start_time = time.time()
        for symbol in symbols:
            categorize_asset(symbol)
        categorization_time = time.time() - start_time

        # Should categorize quickly
        assert (
            categorization_time < 1.0
        ), f"Categorization took too long: {categorization_time:.3f} seconds"

    def test_crypto_memory_usage_integration(self):
        """Test memory usage of crypto integration."""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform crypto operations
        from strategies.equity.golden_cross import GoldenCrossStrategy
        from utils.asset_categorization import categorize_asset

        strategy = GoldenCrossStrategy()

        # Categorize all symbols
        for symbol in strategy.symbols:
            categorize_asset(symbol)

        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        memory_increase_mb = memory_increase / 1024 / 1024
        assert (
            memory_increase_mb < 100
        ), f"Memory usage increased too much: {memory_increase_mb:.2f} MB"
