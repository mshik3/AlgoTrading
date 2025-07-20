"""
Unit tests for Alpaca trading client crypto functionality.
Tests crypto trading execution, symbol handling, and client selection.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestAlpacaTradingCrypto:
    """Test Alpaca trading client crypto functionality."""

    @pytest.fixture
    def mock_alpaca_config(self):
        """Mock Alpaca configuration."""
        from execution.alpaca import AlpacaConfig

        return AlpacaConfig(
            api_key="test_key_123456789", secret_key="test_secret_123456789", paper=True
        )

    @pytest.fixture
    def trading_client(self, mock_alpaca_config):
        """Create trading client with mocked config."""
        with patch("execution.alpaca.ALPACA_AVAILABLE", True):
            with patch("execution.alpaca.TradingClient"):
                with patch("execution.alpaca.StockHistoricalDataClient"):
                    with patch("execution.alpaca.CryptoHistoricalDataClient"):
                        from execution.alpaca import AlpacaTradingClient

                        return AlpacaTradingClient(config=mock_alpaca_config)

    def test_crypto_symbol_detection(self, trading_client):
        """Test that crypto symbols are correctly identified in trading client."""
        # Test representative crypto symbol
        assert trading_client._is_crypto_symbol("BTCUSD") == True

        # Test representative non-crypto symbols
        assert trading_client._is_crypto_symbol("SPY") == False
        assert trading_client._is_crypto_symbol("AAPL") == False

    def test_crypto_symbol_conversion(self, trading_client):
        """Test symbol conversion to Alpaca format in trading client."""
        # Test representative crypto symbol conversion
        assert trading_client._get_alpaca_symbol("BTCUSD") == "BTC/USD"

        # Test representative non-crypto symbols (should remain unchanged)
        assert trading_client._get_alpaca_symbol("SPY") == "SPY"
        assert trading_client._get_alpaca_symbol("AAPL") == "AAPL"

    def test_data_client_selection(self, trading_client):
        """Test that appropriate data clients are selected for different symbol types."""
        # Test crypto data client selection
        crypto_data_client = trading_client._get_data_client("BTCUSD")
        assert crypto_data_client == trading_client.crypto_data_client

        # Test stock data client selection
        stock_data_client = trading_client._get_data_client("SPY")
        assert stock_data_client == trading_client.data_client

        # Test that clients are different
        assert crypto_data_client != stock_data_client

    def test_crypto_current_price_fetching(self, trading_client):
        """Test crypto current price fetching in trading client."""
        # Mock crypto bars response with proper structure
        mock_bar_data = Mock()
        mock_bar_data.df = pd.DataFrame(
            {
                "close": [50000, 51000, 52000],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        # Create a mock that behaves like a dictionary with "BTC/USD" key
        mock_bars = Mock()
        mock_bars.__getitem__ = Mock(return_value=mock_bar_data)
        mock_bars.__contains__ = Mock(return_value=True)

        # Mock the crypto data client
        trading_client.crypto_data_client.get_crypto_bars.return_value = mock_bars

        # Test crypto current price fetching
        result = trading_client.get_current_price("BTCUSD")

        # Verify crypto data client was used
        trading_client.crypto_data_client.get_crypto_bars.assert_called_once()

        # Verify result (should be the last close price)
        assert result == 52000.0

    def test_stock_current_price_fetching(self, trading_client):
        """Test stock current price fetching in trading client."""
        # Mock stock bars response with proper structure
        mock_bar_data = Mock()
        mock_bar_data.df = pd.DataFrame(
            {
                "close": [100, 101, 102],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        # Create a mock that behaves like a dictionary with "SPY" key
        mock_bars = Mock()
        mock_bars.__getitem__ = Mock(return_value=mock_bar_data)
        mock_bars.__contains__ = Mock(return_value=True)

        # Mock the stock data client
        trading_client.data_client.get_stock_bars.return_value = mock_bars

        # Test stock current price fetching
        result = trading_client.get_current_price("SPY")

        # Verify stock data client was used
        trading_client.data_client.get_stock_bars.assert_called_once()

        # Verify result (should be the last close price)
        assert result == 102.0

    def test_crypto_buy_order_execution(self, trading_client):
        """Test crypto buy order execution."""
        from strategies.base import StrategySignal, SignalType

        # Mock account info with sufficient buying power (increased to ensure non-zero quantity)
        mock_account = Mock()
        mock_account.buying_power = "1000000"  # $1M to ensure we can buy at least 1 BTC
        trading_client.trading_client.get_account.return_value = mock_account

        # Mock current price
        trading_client.get_current_price = Mock(return_value=50000.0)

        # Mock order submission
        mock_order = Mock()
        mock_order.id = "test_order_id"
        mock_order.status = "accepted"
        trading_client.trading_client.submit_order.return_value = mock_order

        # Create buy signal
        signal = StrategySignal(
            symbol="BTCUSD",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=50000.0,
            timestamp=datetime.now(),
        )

        # Test buy order execution
        result = trading_client.execute_signal(signal)

        # Verify order was submitted with correct symbol
        trading_client.trading_client.submit_order.assert_called_once()
        call_args = trading_client.trading_client.submit_order.call_args
        order_request = call_args[0][0]
        assert order_request.symbol == "BTC/USD"
        assert order_request.side.value == "buy"

        # Verify result
        assert result == True

    def test_crypto_sell_order_execution(self, trading_client):
        """Test crypto sell order execution."""
        from strategies.base import StrategySignal, SignalType

        # Mock positions with correct symbol format (matches signal.symbol)
        mock_positions = [{"symbol": "BTCUSD", "quantity": 0.1, "market_value": 5000.0}]
        trading_client.get_positions = Mock(return_value=mock_positions)

        # Mock order submission
        mock_order = Mock()
        mock_order.id = "test_order_id"
        mock_order.status = "accepted"
        trading_client.trading_client.submit_order.return_value = mock_order

        # Create sell signal
        signal = StrategySignal(
            symbol="BTCUSD",
            signal_type=SignalType.SELL,
            confidence=0.8,
            price=50000.0,
            timestamp=datetime.now(),
        )

        # Test sell order execution
        result = trading_client.execute_signal(signal)

        # Verify order was submitted with correct symbol
        trading_client.trading_client.submit_order.assert_called_once()
        call_args = trading_client.trading_client.submit_order.call_args
        order_request = call_args[0][0]
        assert order_request.symbol == "BTC/USD"
        assert order_request.side.value == "sell"

        # Verify result
        assert result == True

    def test_stock_buy_order_execution(self, trading_client):
        """Test stock buy order execution."""
        from strategies.base import StrategySignal, SignalType

        # Mock account info
        mock_account = Mock()
        mock_account.buying_power = "10000"
        trading_client.trading_client.get_account.return_value = mock_account

        # Mock current price
        trading_client.get_current_price = Mock(return_value=100.0)

        # Mock order submission
        mock_order = Mock()
        mock_order.id = "test_order_id"
        mock_order.status = "accepted"
        trading_client.trading_client.submit_order.return_value = mock_order

        # Create buy signal
        signal = StrategySignal(
            symbol="SPY",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=100.0,
            timestamp=datetime.now(),
        )

        # Test buy order execution
        result = trading_client.execute_signal(signal)

        # Verify order was submitted with correct symbol
        trading_client.trading_client.submit_order.assert_called_once()
        call_args = trading_client.trading_client.submit_order.call_args
        order_request = call_args[0][0]
        assert order_request.symbol == "SPY"
        assert order_request.side.value == "buy"

        # Verify result
        assert result == True

    def test_error_handling_crypto_order_execution(self, trading_client):
        """Test error handling when crypto order execution fails."""
        from strategies.base import StrategySignal, SignalType

        # Mock account info
        mock_account = Mock()
        mock_account.buying_power = "10000"
        trading_client.trading_client.get_account.return_value = mock_account

        # Mock current price to return None (error condition)
        trading_client.get_current_price = Mock(return_value=None)

        # Create buy signal
        signal = StrategySignal(
            symbol="BTCUSD",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=50000.0,
            timestamp=datetime.now(),
        )

        # Test that error is handled gracefully
        result = trading_client.execute_signal(signal)
        assert result == False

    def test_crypto_configuration_validation(self):
        """Test crypto-related configuration validation in trading client."""
        from execution.alpaca import AlpacaTradingClient

        # Test with valid keys
        with patch.dict(
            "os.environ",
            {
                "ALPACA_API_KEY": "valid_key_123456789",
                "ALPACA_SECRET_KEY": "valid_secret_123456789",
            },
        ):
            with patch("execution.alpaca.ALPACA_AVAILABLE", True):
                with patch("execution.alpaca.TradingClient"):
                    with patch("execution.alpaca.StockHistoricalDataClient"):
                        with patch("execution.alpaca.CryptoHistoricalDataClient"):
                            client = AlpacaTradingClient()
                            assert client.config.api_key == "valid_key_123456789"
                            assert client.config.secret_key == "valid_secret_123456789"

        # Test with invalid keys (too short)
        with patch.dict(
            "os.environ", {"ALPACA_API_KEY": "short", "ALPACA_SECRET_KEY": "short"}
        ):
            with patch("execution.alpaca.ALPACA_AVAILABLE", True):
                with pytest.raises(ValueError, match="Invalid API key format"):
                    AlpacaTradingClient()

    def test_crypto_edge_cases(self, trading_client):
        """Test edge cases for crypto functionality in trading client."""
        # Test unknown symbol (should not be treated as crypto)
        assert trading_client._is_crypto_symbol("UNKNOWNUSD") == False
        assert trading_client._get_alpaca_symbol("UNKNOWNUSD") == "UNKNOWNUSD"

        # Test empty and None symbols - should handle gracefully
        assert trading_client._is_crypto_symbol("") == False
        assert trading_client._get_alpaca_symbol("") == ""
        assert trading_client._is_crypto_symbol(None) == False
        assert trading_client._get_alpaca_symbol(None) == None
