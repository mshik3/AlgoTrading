"""
Real Alpaca Account Service
Connects to actual Alpaca API to get real account data and positions.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List

# Add project root to path - more robust path resolution
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from utils.config import load_environment, get_env_var
from data.alpaca_collector import AlpacaConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlpacaAccountService:
    """
    Service class for connecting to real Alpaca account.
    Provides real account data, positions, and cash balance.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(AlpacaAccountService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the Alpaca account service (only once)."""
        if not self._initialized:
            self.setup_environment()
            self.setup_alpaca_client()
            AlpacaAccountService._initialized = True

    def setup_environment(self):
        """Load environment variables and validate configuration."""
        try:
            # Only load environment if not already loaded
            from utils.config import _environment_loaded

            if not _environment_loaded:
                load_environment()

            # Check for required environment variables
            alpaca_key = get_env_var("ALPACA_API_KEY", default=None)
            alpaca_secret = get_env_var("ALPACA_SECRET_KEY", default=None)

            if not alpaca_key or not alpaca_secret:
                logger.error(
                    "Alpaca API credentials not found - cannot connect to real account"
                )
                self.connected = False
            else:
                self.connected = True
                self.api_key = alpaca_key
                self.secret_key = alpaca_secret

        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            self.connected = False

    def setup_alpaca_client(self):
        """Initialize the Alpaca trading client."""
        try:
            if self.connected:
                from alpaca.trading.client import TradingClient
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus

                config = AlpacaConfig(
                    api_key=self.api_key, secret_key=self.secret_key, paper=True
                )

                self.trading_client = TradingClient(
                    api_key=config.api_key,
                    secret_key=config.secret_key,
                    paper=config.paper,
                )

                logger.info("✓ Alpaca trading client initialized")
            else:
                self.trading_client = None
                logger.warning(
                    "✗ Alpaca trading client not initialized - using fallback"
                )

        except Exception as e:
            logger.error(f"Alpaca client setup failed: {e}")
            self.trading_client = None
            self.connected = False

    def get_account_summary(self) -> Dict:
        """
        Get real account summary from Alpaca.

        Returns:
            Dictionary with account information
        """
        if not self.connected or not self.trading_client:
            raise Exception("Alpaca connection required for account data")

        try:
            account = self.trading_client.get_account()

            # Calculate total P&L properly
            # For a new account with no positions, P&L should be 0
            # For accounts with positions, P&L = current equity - initial equity
            initial_margin = (
                float(account.initial_margin) if account.initial_margin else 0.0
            )
            current_equity = float(account.equity)

            # If initial margin is 0 (new account), P&L should be 0
            # Otherwise, calculate as equity - initial margin
            total_pnl = current_equity - initial_margin if initial_margin > 0 else 0.0

            return {
                "total_value": float(account.equity),
                "cash": float(account.cash),
                "total_pnl": total_pnl,
                "positions_value": float(account.equity) - float(account.cash),
                "buying_power": float(account.buying_power),
                "day_trade_count": account.daytrade_count,
                "account_status": account.status,
                "last_updated": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            raise Exception(f"Failed to get account summary: {e}")

    def get_positions(self) -> List[Dict]:
        """
        Get real positions from Alpaca.

        Returns:
            List of position dictionaries
        """
        if not self.connected or not self.trading_client:
            raise Exception("Alpaca connection required for position data")

        try:
            positions = self.trading_client.get_all_positions()

            real_positions = []
            for position in positions:
                # Get current market price for P&L calculation
                try:
                    from alpaca.data.historical import StockHistoricalDataClient
                    from alpaca.data.requests import StockLatestQuoteRequest

                    data_client = StockHistoricalDataClient(
                        api_key=self.api_key, secret_key=self.secret_key
                    )

                    quote_request = StockLatestQuoteRequest(
                        symbol_or_symbols=position.symbol
                    )
                    quote = data_client.get_stock_latest_quote(quote_request)
                    current_price = float(quote[position.symbol].ask_price)

                except Exception:
                    # Fallback to position's current price if quote fails
                    current_price = float(position.current_price)

                real_positions.append(
                    {
                        "symbol": position.symbol,
                        "quantity": int(position.qty),
                        "avg_price": float(position.avg_entry_price),
                        "current_price": current_price,
                        "market_value": float(position.market_value),
                        "unrealized_pnl": float(position.unrealized_pl),
                        "pnl_percent": (
                            (
                                float(position.unrealized_pl)
                                / float(position.market_value)
                            )
                            * 100
                            if float(position.market_value) > 0
                            else 0
                        ),
                    }
                )

            return real_positions

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_recent_orders(self, limit: int = 10) -> List[Dict]:
        """
        Get recent orders from Alpaca.

        Args:
            limit: Maximum number of orders to return

        Returns:
            List of order dictionaries
        """
        if not self.connected or not self.trading_client:
            raise Exception("Alpaca connection required for order data")

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request_params = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)

            orders = self.trading_client.get_orders(filter=request_params)

            recent_orders = []
            for order in orders:
                recent_orders.append(
                    {
                        "id": order.id,
                        "timestamp": order.created_at,
                        "action": order.side.value,
                        "symbol": order.symbol,
                        "quantity": int(order.qty),
                        "price": (
                            float(order.filled_avg_price)
                            if order.filled_avg_price
                            else float(order.limit_price) if order.limit_price else 0
                        ),
                        "status": order.status.value,
                        "order_type": (
                            order.order_class.value if order.order_class else "simple"
                        ),
                    }
                )

            return recent_orders

        except Exception as e:
            logger.error(f"Error getting recent orders: {e}")
            return []

    def get_strategy_signals(self) -> List[Dict]:
        """
        Get recent strategy signals (not actual trades).

        Returns:
            List of signal dictionaries
        """
        # This would be populated by the analysis service
        # For now, return empty list - signals will come from analysis
        return []

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by ID.

        Args:
            order_id: The order ID to cancel

        Returns:
            True if cancelled successfully, False otherwise
        """
        if not self.connected or not self.trading_client:
            raise Exception("Alpaca connection required for order cancellation")

        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"✓ Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def is_connected(self) -> bool:
        """
        Check if connected to Alpaca.

        Returns:
            True if connected, False otherwise
        """
        return self.connected
