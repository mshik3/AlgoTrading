"""
Alpaca Trading Client for live and paper trading execution.
Provides integration with Alpaca Markets API for order execution and portfolio management.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd

# Alpaca SDK
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    from alpaca.data import (
        StockHistoricalDataClient,
        CryptoHistoricalDataClient,
        TimeFrame,
    )
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("Alpaca SDK not installed. Install with: pip install alpaca-py")

from strategies.base import StrategySignal, SignalType
from .paper import PaperPosition, PaperTrade

logger = logging.getLogger(__name__)


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca trading."""

    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"  # Paper trading by default
    paper: bool = True


class AlpacaTradingClient:
    """
    Alpaca trading client for executing orders and managing positions.

    Features:
    - Paper trading support (default)
    - Live trading capability
    - Real-time portfolio data
    - Order execution and management
    - Position tracking
    """

    def __init__(self, config: Optional[AlpacaConfig] = None):
        """
        Initialize Alpaca trading client.

        Args:
            config: Alpaca configuration (if None, loads from environment)
        """
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "Alpaca SDK not available. Install with: pip install alpaca-py"
            )

        if config is None:
            config = self._load_config_from_env()

        self.config = config

        # Initialize clients
        self.trading_client = TradingClient(
            api_key=config.api_key, secret_key=config.secret_key, paper=config.paper
        )

        self.data_client = StockHistoricalDataClient(
            api_key=config.api_key, secret_key=config.secret_key
        )

        self.crypto_data_client = CryptoHistoricalDataClient(
            api_key=config.api_key, secret_key=config.secret_key
        )

        # Crypto symbols mapping (Alpaca uses different symbols for crypto)
        self.crypto_symbols = {
            "BTCUSD": "BTC/USD",
            "ETHUSD": "ETH/USD",
            "ADAUSD": "ADA/USD",
            "DOTUSD": "DOT/USD",
            "LINKUSD": "LINK/USD",
            "LTCUSD": "LTC/USD",
            "BCHUSD": "BCH/USD",
            "XRPUSD": "XRP/USD",
            "SOLUSD": "SOL/USD",
            "MATICUSD": "MATIC/USD",
        }

        logger.info(f"Alpaca trading client initialized (paper: {config.paper})")

    def _load_config_from_env(self) -> AlpacaConfig:
        """Load Alpaca configuration from environment variables."""
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or provide AlpacaConfig object."
            )

        # Validate API key format (basic validation)
        if len(api_key) < 10 or len(secret_key) < 10:
            raise ValueError(
                "Invalid API key format. Keys should be at least 10 characters long."
            )

        return AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
            paper=True,  # Default to paper trading for safety
        )

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a crypto asset."""
        return symbol in self.crypto_symbols

    def _get_alpaca_symbol(self, symbol: str) -> str:
        """Convert our symbol format to Alpaca's format."""
        if self._is_crypto_symbol(symbol):
            return self.crypto_symbols[symbol]
        return symbol

    def _get_data_client(self, symbol: str):
        """Get the appropriate data client for the symbol type."""
        if self._is_crypto_symbol(symbol):
            return self.crypto_data_client
        return self.data_client

    def test_connection(self) -> bool:
        """
        Test connection to Alpaca API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test trading client
            account = self.trading_client.get_account()
            logger.info(f"✓ Connected to Alpaca (Paper: {self.config.paper})")
            logger.info(f"  Account ID: {account.id}")
            logger.info(f"  Status: {account.status}")
            logger.info(f"  Cash: ${float(account.cash):,.2f}")
            logger.info(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
            return True
        except Exception as e:
            logger.error(f"✗ Alpaca connection failed: {str(e)}")
            return False

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information.

        Returns:
            Dictionary with account details
        """
        try:
            account = self.trading_client.get_account()
            return {
                "id": account.id,
                "status": account.status,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "daytrade_count": account.daytrade_count,
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
                "created_at": account.created_at,
                "currency": account.currency,
            }
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {}

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.

        Returns:
            List of position dictionaries
        """
        try:
            positions = self.trading_client.get_all_positions()
            position_list = []

            for position in positions:
                position_list.append(
                    {
                        "symbol": position.symbol,
                        "quantity": int(position.qty),
                        "avg_price": float(position.avg_entry_price),
                        "market_value": float(position.market_value),
                        "unrealized_pnl": float(position.unrealized_pl),
                        "unrealized_pnl_pct": float(position.unrealized_plpc),
                        "side": position.side,
                        "current_price": float(position.current_price),
                    }
                )

            return position_list
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Stock or crypto symbol

        Returns:
            Current price or None if failed
        """
        try:
            # Get appropriate client and symbol format
            client = self._get_data_client(symbol)
            alpaca_symbol = self._get_alpaca_symbol(symbol)

            # Get today's data to get current price
            today = datetime.now()

            # Create appropriate request based on asset type
            if self._is_crypto_symbol(symbol):
                request = CryptoBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=TimeFrame.Day,
                    start=today - timedelta(days=5),  # Get last 5 days
                    end=today,
                )
                bars = client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=TimeFrame.Day,
                    start=today - timedelta(days=5),  # Get last 5 days
                    end=today,
                )
                bars = client.get_stock_bars(request)

            if bars and alpaca_symbol in bars:
                latest_bar = bars[alpaca_symbol].df.iloc[-1]
                return float(latest_bar["close"])

            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None

    def execute_signal(self, signal: StrategySignal) -> bool:
        """
        Execute a trading signal.

        Args:
            signal: The trading signal to execute

        Returns:
            True if order placed successfully, False otherwise
        """
        try:
            if signal.signal_type == SignalType.BUY:
                return self._execute_buy_order(signal)
            elif signal.signal_type == SignalType.SELL:
                return self._execute_sell_order(signal)
            else:
                logger.warning(f"Unknown signal type: {signal.signal_type}")
                return False
        except Exception as e:
            logger.error(f"Error executing signal: {str(e)}")
            return False

    def _execute_buy_order(self, signal: StrategySignal) -> bool:
        """
        Execute a buy order.

        Args:
            signal: Buy signal

        Returns:
            True if order placed successfully
        """
        try:
            # Get current price
            current_price = self.get_current_price(signal.symbol)
            if current_price is None:
                logger.error(f"Could not get current price for {signal.symbol}")
                return False

            # Calculate position size (use 5% of available cash for safety)
            account = self.trading_client.get_account()
            available_cash = float(account.buying_power)
            position_value = available_cash * 0.05  # 5% of available cash

            # Calculate quantity (round down to whole shares)
            quantity = int(position_value / current_price)

            if quantity <= 0:
                logger.warning(
                    f"Position size too small for {signal.symbol}: ${position_value:.2f}"
                )
                return False

            # Get appropriate symbol format for crypto
            alpaca_symbol = self._get_alpaca_symbol(signal.symbol)

            # Create market order
            order_request = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )

            # Place order
            order = self.trading_client.submit_order(order_request)

            logger.info(
                f"✓ Buy order placed for {signal.symbol}: {quantity} shares at ~${current_price:.2f}"
            )
            logger.info(f"  Order ID: {order.id}")
            logger.info(f"  Status: {order.status}")

            return True

        except Exception as e:
            logger.error(f"Error executing buy order for {signal.symbol}: {str(e)}")
            return False

    def _execute_sell_order(self, signal: StrategySignal) -> bool:
        """
        Execute a sell order.

        Args:
            signal: Sell signal

        Returns:
            True if order placed successfully
        """
        try:
            # Get current positions
            positions = self.get_positions()
            position = next(
                (p for p in positions if p["symbol"] == signal.symbol), None
            )

            if not position:
                logger.warning(f"No position found for {signal.symbol}")
                return False

            quantity = position["quantity"]

            if quantity <= 0:
                logger.warning(f"No shares to sell for {signal.symbol}")
                return False

            # Get appropriate symbol format for crypto
            alpaca_symbol = self._get_alpaca_symbol(signal.symbol)

            # Create market order
            order_request = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=quantity,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )

            # Place order
            order = self.trading_client.submit_order(order_request)

            logger.info(f"✓ Sell order placed for {signal.symbol}: {quantity} shares")
            logger.info(f"  Order ID: {order.id}")
            logger.info(f"  Status: {order.status}")

            return True

        except Exception as e:
            logger.error(f"Error executing sell order for {signal.symbol}: {str(e)}")
            return False

    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get orders with optional status filter.

        Args:
            status: Order status filter (e.g., 'open', 'closed', 'all')

        Returns:
            List of order dictionaries
        """
        try:
            orders = self.trading_client.get_orders(status=status)
            order_list = []

            for order in orders:
                order_list.append(
                    {
                        "id": order.id,
                        "symbol": order.symbol,
                        "quantity": int(order.qty),
                        "side": order.side,
                        "type": order.type,
                        "status": order.status,
                        "filled_at": order.filled_at,
                        "filled_avg_price": (
                            float(order.filled_avg_price)
                            if order.filled_avg_price
                            else None
                        ),
                        "created_at": order.created_at,
                    }
                )

            return order_list
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"✓ Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary.

        Returns:
            Portfolio summary dictionary
        """
        try:
            account = self.trading_client.get_account()
            positions = self.get_positions()

            total_position_value = sum(p["market_value"] for p in positions)
            total_unrealized_pnl = sum(p["unrealized_pnl"] for p in positions)

            return {
                "total_value": float(account.portfolio_value),
                "cash": float(account.cash),
                "positions_value": total_position_value,
                "unrealized_pnl": total_unrealized_pnl,
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "position_count": len(positions),
                "daytrade_count": account.daytrade_count,
            }
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
            return {}

    def get_transaction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get transaction history from Alpaca.

        Args:
            limit: Maximum number of transactions to return

        Returns:
            List of transaction dictionaries
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            # Get all orders (both open and closed)
            request_params = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            orders = self.trading_client.get_orders(filter=request_params)

            transactions = []
            for order in orders:
                if order.status.value == "filled":
                    transaction = {
                        "id": order.id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "quantity": int(order.qty),
                        "filled_price": float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                        "filled_at": order.filled_at,
                        "created_at": order.created_at,
                        "order_type": order.type.value,
                        "status": order.status.value,
                        "total_value": float(order.filled_avg_price * order.qty) if order.filled_avg_price else 0.0,
                    }
                    transactions.append(transaction)

            return transactions

        except Exception as e:
            logger.error(f"Error getting transaction history: {str(e)}")
            return []

    def get_portfolio_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Get portfolio performance metrics.

        Args:
            days: Number of days to analyze

        Returns:
            Portfolio performance dictionary
        """
        try:
            account = self.trading_client.get_account()
            positions = self.get_positions()
            
            # Calculate basic metrics
            total_value = float(account.portfolio_value)
            cash = float(account.cash)
            positions_value = total_value - cash
            total_unrealized_pnl = sum(p["unrealized_pnl"] for p in positions)
            
            # Get recent transactions for realized P&L
            recent_transactions = self.get_transaction_history(limit=50)
            realized_pnl = 0.0
            
            # Calculate realized P&L from recent transactions
            for tx in recent_transactions:
                if tx["side"] == "sell":
                    # This is a simplified calculation - in reality you'd need to track cost basis
                    realized_pnl += tx["total_value"] * 0.01  # Assume 1% profit for demo
            
            return {
                "total_value": total_value,
                "cash": cash,
                "positions_value": positions_value,
                "unrealized_pnl": total_unrealized_pnl,
                "realized_pnl": realized_pnl,
                "total_pnl": total_unrealized_pnl + realized_pnl,
                "position_count": len(positions),
                "recent_transactions": len(recent_transactions),
                "account_status": account.status,
                "buying_power": float(account.buying_power),
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {str(e)}")
            return {}


def get_alpaca_client() -> AlpacaTradingClient:
    """
    Factory function to create an Alpaca trading client.

    Returns:
        Configured AlpacaTradingClient instance
    """
    return AlpacaTradingClient()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Alpaca Trading Client...")

    try:
        client = get_alpaca_client()

        # Test connection
        if client.test_connection():
            print("✓ Connection successful!")

            # Get account info
            account_info = client.get_account_info()
            print(f"\nAccount Info:")
            print(f"  Cash: ${account_info.get('cash', 0):,.2f}")
            print(f"  Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
            print(f"  Buying Power: ${account_info.get('buying_power', 0):,.2f}")

            # Get positions
            positions = client.get_positions()
            print(f"\nPositions ({len(positions)}):")
            for pos in positions:
                print(
                    f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}"
                )
                print(
                    f"    Current: ${pos['current_price']:.2f} | P&L: ${pos['unrealized_pnl']:.2f}"
                )

            # Get current prices
            symbols = ["SPY", "QQQ", "VTI"]
            print(f"\nCurrent Prices:")
            for symbol in symbols:
                price = client.get_current_price(symbol)
                if price:
                    print(f"  {symbol}: ${price:.2f}")
                else:
                    print(f"  {symbol}: Unable to get price")

        else:
            print("✗ Connection failed!")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTo use Alpaca trading client:")
        print("1. Sign up at https://alpaca.markets")
        print("2. Get your API keys from the dashboard")
        print("3. Set environment variables:")
        print("   export ALPACA_API_KEY=your_api_key")
        print("   export ALPACA_SECRET_KEY=your_secret_key")
        print("4. Install Alpaca SDK: pip install alpaca-py")
