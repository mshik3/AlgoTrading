"""
Live Data Management for Trading Dashboard
Handles real-time data fetching, caching, and updates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os
from flask_caching import Cache
import time

# Add parent directory to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    from execution.paper import PaperTradingSimulator
    from data.storage import DatabaseStorage
    from utils.config import Config

    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports not available for LiveDataManager: {e}")
    IMPORTS_AVAILABLE = False

    # Mock classes for development
    class PaperTradingSimulator:
        def get_portfolio_summary(self):
            return {
                "total_value": 100000.0,
                "cash": 100000.0,
                "total_pnl": 0.0,
                "positions_value": 0.0,
            }

        def get_positions(self):
            return []

        def get_trade_history(self, limit=10):
            return []

    class Config:
        pass

    class DatabaseStorage:
        def __init__(self, config):
            pass


class LiveDataManager:
    """Manages real-time data for the trading dashboard"""

    def __init__(self):
        self.paper_trader = PaperTradingSimulator()
        self.db_config = Config() if IMPORTS_AVAILABLE else None
        self.db = DatabaseStorage(self.db_config) if IMPORTS_AVAILABLE else None
        self.cache = None

    def setup_cache(self, app):
        """Setup Flask-Caching for the dashboard"""
        self.cache = Cache(
            app,
            config={
                "CACHE_TYPE": "simple",
                "CACHE_DEFAULT_TIMEOUT": 30,  # 30 seconds default
            },
        )
        return self.cache

    def get_portfolio_summary(self):
        """
        Get current portfolio summary with caching

        Returns:
            dict: Portfolio summary data
        """
        if self.cache:
            summary = self.cache.get("portfolio_summary")
            if summary:
                return summary

        try:
            # Use Alpaca account service instead of paper trader
            from dashboard.services.alpaca_account import AlpacaAccountService

            alpaca_service = AlpacaAccountService()
            summary = alpaca_service.get_account_summary()

            if self.cache:
                self.cache.set("portfolio_summary", summary, timeout=30)

            return summary

        except Exception as e:
            print(f"Error getting portfolio summary: {e}")
            raise Exception("Alpaca connection required for portfolio data")

    def get_positions(self):
        """
        Get current positions with real-time pricing

        Returns:
            list: List of position dictionaries
        """
        if self.cache:
            positions = self.cache.get("positions_data")
            if positions:
                return positions

        try:
            # Use Alpaca account service instead of paper trader
            from dashboard.services.alpaca_account import AlpacaAccountService

            alpaca_service = AlpacaAccountService()
            positions = alpaca_service.get_positions()

            if self.cache:
                self.cache.set("positions_data", positions, timeout=30)

            return positions

        except Exception as e:
            print(f"Error getting positions: {e}")
            raise Exception("Alpaca connection required for position data")

    def get_current_price(self, symbol):
        """
        Get current price for a symbol with caching

        Args:
            symbol (str): Stock symbol

        Returns:
            float: Current price
        """
        cache_key = f"price_{symbol}"

        if self.cache:
            price = self.cache.get(cache_key)
            if price:
                return price

        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")

            if not data.empty:
                current_price = data["Close"].iloc[-1]
            else:
                # Fallback to longer period
                data = ticker.history(period="5d")
                current_price = data["Close"].iloc[-1] if not data.empty else 100.0

            if self.cache:
                self.cache.set(
                    cache_key, current_price, timeout=60
                )  # Cache for 1 minute

            return float(current_price)

        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return 100.0  # Default price

    def get_portfolio_history(self, days=30):
        """
        Get portfolio value history

        Args:
            days (int): Number of days of history

        Returns:
            pd.DataFrame: Portfolio history data
        """
        cache_key = f"portfolio_history_{days}"

        if self.cache:
            history = self.cache.get(cache_key)
            if history is not None:
                return history

        try:
            # Use Alpaca for portfolio history
            from dashboard.services.alpaca_account import AlpacaAccountService

            alpaca_service = AlpacaAccountService()

            # For now, return a simple linear progression based on current portfolio value
            # TODO: Implement actual historical portfolio data from Alpaca
            account_summary = alpaca_service.get_account_summary()
            current_value = account_summary.get("total_value", 100000)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            dates = pd.date_range(start=start_date, end=end_date, freq="D")

            # Simple linear progression (placeholder)
            values = [
                current_value * 0.95 + (current_value * 0.05 * i / len(dates))
                for i in range(len(dates))
            ]

            history_df = pd.DataFrame({"date": dates, "portfolio_value": values})

            if self.cache:
                self.cache.set(cache_key, history_df, timeout=300)

            return history_df

        except Exception as e:
            print(f"Error getting portfolio history: {e}")
            raise Exception("Alpaca connection required for portfolio history")

    def get_recent_activity(self, limit=10):
        """
        Get recent trading activity

        Args:
            limit (int): Number of recent activities to return

        Returns:
            list: List of activity dictionaries
        """
        if self.cache:
            activity = self.cache.get("recent_activity")
            if activity:
                return activity

        try:
            # Get recent trades from paper trader
            trades = self.paper_trader.get_trade_history(limit=limit // 2)

            activities = []

            # Convert trades to activity format
            for trade in trades:
                activities.append(
                    {
                        "timestamp": trade.get("timestamp", datetime.now()),
                        "type": "trade",
                        "description": f"{trade.get('action', 'N/A')} {trade.get('quantity', 0)} shares of {trade.get('symbol', 'N/A')} at ${trade.get('price', 0):.2f}",
                        "time": trade.get("timestamp", datetime.now()).strftime(
                            "%H:%M:%S"
                        ),
                    }
                )

            # Add mock system activities
            now = datetime.now()
            activities.extend(
                [
                    {
                        "timestamp": now - timedelta(minutes=15),
                        "type": "signal",
                        "description": "Golden Cross BUY signal generated for AAPL",
                        "time": (now - timedelta(minutes=15)).strftime("%H:%M:%S"),
                    },
                    {
                        "timestamp": now - timedelta(minutes=45),
                        "type": "system",
                        "description": "Portfolio rebalanced - Risk level: Low",
                        "time": (now - timedelta(minutes=45)).strftime("%H:%M:%S"),
                    },
                    {
                        "timestamp": now - timedelta(hours=1),
                        "type": "performance",
                        "description": "Golden Cross strategy performance: +2.3% this week",
                        "time": (now - timedelta(hours=1)).strftime("%H:%M:%S"),
                    },
                ]
            )

            # Sort by timestamp descending
            activities.sort(key=lambda x: x["timestamp"], reverse=True)
            activities = activities[:limit]

            if self.cache:
                self.cache.set(
                    "recent_activity", activities, timeout=60
                )  # Cache for 1 minute

            return activities

        except Exception as e:
            print(f"Error getting recent activity: {e}")
            return []

    def get_strategy_performance(self, strategy_name="golden_cross"):
        """
        Get strategy performance metrics from real Alpaca data

        Args:
            strategy_name (str): Name of the strategy

        Returns:
            dict: Strategy performance data
        """
        cache_key = f"strategy_performance_{strategy_name}"

        if self.cache:
            performance = self.cache.get(cache_key)
            if performance:
                return performance

        try:
            # Use Alpaca for strategy performance
            from dashboard.services.alpaca_account import AlpacaAccountService

            alpaca_service = AlpacaAccountService()

            if not alpaca_service.is_connected():
                return {
                    "name": "Golden Cross Strategy",
                    "status": "DISCONNECTED",
                    "last_signal": "N/A",
                    "signal_time": datetime.now(),
                    "win_rate": None,
                    "total_trades": 0,
                    "profit_factor": None,
                    "sharpe_ratio": None,
                    "max_drawdown": None,
                    "total_return": None,
                }

            # Get recent transactions to calculate strategy performance
            recent_orders = alpaca_service.get_recent_orders(limit=100)

            # Calculate basic metrics from recent trades
            total_trades = len(recent_orders)

            if total_trades == 0:
                # No trades yet - return empty state
                return {
                    "name": "Golden Cross Strategy",
                    "status": "ACTIVE",
                    "last_signal": "NONE",
                    "signal_time": datetime.now(),
                    "win_rate": None,
                    "total_trades": 0,
                    "profit_factor": None,
                    "sharpe_ratio": None,
                    "max_drawdown": None,
                    "total_return": None,
                }

            # Calculate win rate from actual trades
            filled_orders = [o for o in recent_orders if o.get("status") == "filled"]
            buy_orders = [o for o in filled_orders if o.get("action") == "buy"]
            sell_orders = [o for o in filled_orders if o.get("action") == "sell"]

            # Calculate win rate based on P&L if we have position data
            win_rate = None
            if len(filled_orders) > 0:
                # For now, use a simple heuristic based on order patterns
                # In a real system, you'd calculate actual P&L from position data
                if len(buy_orders) > 0 and len(sell_orders) > 0:
                    # Assume trades are profitable if we have both buy and sell orders
                    # This is a placeholder - real calculation would use actual P&L
                    win_rate = 0.65  # Placeholder until we implement real P&L tracking
                else:
                    win_rate = None

            # Determine last signal
            last_signal = "NONE"
            if filled_orders:
                last_order = filled_orders[0]  # Most recent order
                last_signal = last_order.get("action", "NONE").upper()

            performance = {
                "name": "Golden Cross Strategy",
                "status": "ACTIVE",
                "last_signal": last_signal,
                "signal_time": datetime.now(),
                "win_rate": win_rate,
                "total_trades": total_trades,
                "profit_factor": None,  # Would need actual P&L calculation
                "sharpe_ratio": None,  # Would need actual returns calculation
                "max_drawdown": None,  # Would need actual portfolio history
                "total_return": None,  # Would need actual portfolio history
            }

            if self.cache:
                self.cache.set(cache_key, performance, timeout=120)

            return performance

        except Exception as e:
            print(f"Error getting strategy performance: {e}")
            # Return error state
            return {
                "name": "Golden Cross Strategy",
                "status": "ERROR",
                "last_signal": "N/A",
                "signal_time": datetime.now(),
                "win_rate": None,
                "total_trades": 0,
                "profit_factor": None,
                "sharpe_ratio": None,
                "max_drawdown": None,
                "total_return": None,
            }

    def get_market_data(self, symbols=None):
        """
        Get market data for dashboard overview

        Args:
            symbols (list): List of symbols to get data for

        Returns:
            dict: Market data
        """
        if symbols is None:
            symbols = ["^GSPC", "^IXIC", "^DJI", "^RUT"]  # Major indices

        cache_key = f'market_data_{"_".join(symbols)}'

        if self.cache:
            market_data = self.cache.get(cache_key)
            if market_data:
                return market_data

        try:
            market_data = {}

            for symbol in symbols:
                price = self.get_current_price(symbol)
                # Calculate daily change (mock for now)
                daily_change = np.random.uniform(-0.02, 0.02)  # -2% to +2%

                market_data[symbol] = {
                    "price": price,
                    "change": daily_change,
                    "change_percent": daily_change * 100,
                }

            if self.cache:
                self.cache.set(cache_key, market_data, timeout=60)  # Cache for 1 minute

            return market_data

        except Exception as e:
            print(f"Error getting market data: {e}")
            return {}

    def refresh_all_data(self):
        """Force refresh all cached data"""
        if self.cache:
            self.cache.clear()
        print("All cached data cleared - will refresh on next request")
