"""
Strategy Metrics Service for Dashboard
Handles multi-strategy performance tracking and metrics collection.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import project modules
from utils.config import load_environment, get_env_var
from dashboard.services.alpaca_account import AlpacaAccountService
from strategies.equity.golden_cross import GoldenCrossStrategy
from strategies.equity.mean_reversion import MeanReversionStrategy
from strategies.etf.dual_momentum import DualMomentumStrategy
from strategies.etf.sector_rotation import SectorRotationStrategy
from utils.asset_categorization import get_etf_universe_for_strategy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyMetricsService:
    """
    Service for collecting and managing metrics for multiple trading strategies.

    Provides real-time performance tracking for:
    - Golden Cross Strategy
    - Mean Reversion Strategy
    - Future strategies can be easily added
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(StrategyMetricsService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the strategy metrics service (only once)."""
        if not self._initialized:
            self.alpaca_service = AlpacaAccountService()
            self.strategies = {
            "golden_cross": {
                "name": "Golden Cross Strategy",
                "class": GoldenCrossStrategy,
                "symbols": ["SPY", "QQQ", "VTI"],  # Default symbols
                "status": "ACTIVE",
                "description": "Moving average crossover strategy",
            },
            "mean_reversion": {
                "name": "Mean Reversion Strategy",
                "class": MeanReversionStrategy,
                "symbols": [
                    "SPY",
                    "QQQ",
                    "VTI",
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                ],  # More symbols for mean reversion
                "status": "ACTIVE",
                "description": "Statistical mean reversion with O-U process",
            },
            "dual_momentum": {
                "name": "Dual Momentum ETF Rotation",
                "class": DualMomentumStrategy,
                "etf_universe": get_etf_universe_for_strategy("dual_momentum"),
                "status": "ACTIVE",
                "description": "Gary Antonacci's dual momentum approach with absolute/relative momentum",
            },
            "sector_rotation": {
                "name": "Sector ETF Rotation",
                "class": SectorRotationStrategy,
                "etf_universe": get_etf_universe_for_strategy("sector_rotation"),
                "status": "ACTIVE",
                "description": "Sector rotation based on relative strength and momentum analysis",
            },
        }

            # Initialize strategy instances
            self.strategy_instances = {}
            self._initialize_strategies()
            StrategyMetricsService._initialized = True

    def _initialize_strategies(self):
        """Initialize strategy instances for metrics collection."""
        try:
            for strategy_id, config in self.strategies.items():
                strategy_class = config["class"]

                # Handle ETF strategies differently
                if strategy_id in ["dual_momentum", "sector_rotation"]:
                    etf_universe = config["etf_universe"]
                    strategy_instance = strategy_class(etf_universe=etf_universe)
                else:
                    # Handle equity strategies
                    symbols = config["symbols"]
                    strategy_instance = strategy_class(symbols=symbols)

                self.strategy_instances[strategy_id] = strategy_instance
                logger.info(f"Initialized {config['name']} for metrics tracking")

        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")

    def get_all_strategy_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all strategies.

        Returns:
            Dictionary mapping strategy_id -> metrics dictionary
        """
        all_metrics = {}

        for strategy_id, config in self.strategies.items():
            try:
                metrics = self._get_strategy_metrics(strategy_id)
                all_metrics[strategy_id] = metrics
            except Exception as e:
                logger.error(f"Error getting metrics for {strategy_id}: {e}")
                # Provide fallback metrics
                all_metrics[strategy_id] = self._get_fallback_metrics(strategy_id)

        return all_metrics

    def _get_strategy_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get detailed metrics for a specific strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Dictionary with strategy metrics
        """
        config = self.strategies.get(strategy_id, {})
        strategy_instance = self.strategy_instances.get(strategy_id)

        if not strategy_instance:
            return self._get_fallback_metrics(strategy_id)

        try:
            # Get strategy performance summary - handle different strategy types
            if strategy_id in ["dual_momentum", "sector_rotation"]:
                # ETF strategies have different summary methods
                if strategy_id == "dual_momentum":
                    strategy_summary = strategy_instance.get_dual_momentum_summary()
                else:  # sector_rotation
                    strategy_summary = strategy_instance.get_sector_rotation_summary()
                
                # ETF strategies don't have the same metrics as equity strategies
                strategy_summary = {
                    "total_signals": 0,  # Will be calculated from actual trading
                    "buy_signals": 0,
                    "sell_signals": 0,
                    "current_positions": len(strategy_instance.positions),
                    "last_signal_time": None,
                    "active_symbols": list(strategy_instance.positions.keys()),
                    "is_active": True,
                }
            else:
                # Equity strategies use standard summary
                strategy_summary = strategy_instance.get_strategy_summary()

            # Get Alpaca account data for real-time metrics
            account_connected = self.alpaca_service.is_connected()

            if account_connected:
                # Get recent orders and positions
                recent_orders = self.alpaca_service.get_recent_orders(limit=100)
                current_positions = self.alpaca_service.get_positions()

                # Filter orders by strategy (this would need strategy tagging in real implementation)
                strategy_orders = self._filter_orders_by_strategy(
                    recent_orders, strategy_id
                )
                filled_orders = [
                    o for o in strategy_orders if o.get("status") == "filled"
                ]

                # Calculate metrics
                total_trades = len(filled_orders)
                win_rate = self._calculate_win_rate(filled_orders, current_positions)
                last_signal = self._get_last_signal(filled_orders)

            else:
                # Fallback to strategy instance data
                total_trades = strategy_summary.get("total_signals", 0)
                win_rate = None
                last_signal = "NONE"

            # Build metrics dictionary
            metrics = {
                "strategy_id": strategy_id,
                "name": config.get("name", "Unknown Strategy"),
                "status": config.get("status", "UNKNOWN"),
                "description": config.get("description", ""),
                "account_connected": account_connected,
                "total_trades": total_trades,
                "total_signals": strategy_summary.get("total_signals", 0),
                "buy_signals": strategy_summary.get("buy_signals", 0),
                "sell_signals": strategy_summary.get("sell_signals", 0),
                "current_positions": strategy_summary.get("current_positions", 0),
                "win_rate": win_rate,
                "last_signal": last_signal,
                "last_signal_time": strategy_summary.get("last_signal_time"),
                "active_symbols": strategy_summary.get("active_symbols", []),
                "is_active": strategy_summary.get("is_active", True),
                "strategy_specific": self._get_strategy_specific_metrics(
                    strategy_id, strategy_instance
                ),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error getting metrics for {strategy_id}: {e}")
            return self._get_fallback_metrics(strategy_id)

    def _get_strategy_specific_metrics(
        self, strategy_id: str, strategy_instance
    ) -> Dict[str, Any]:
        """Get strategy-specific metrics."""
        try:
            if strategy_id == "mean_reversion":
                # Get mean reversion specific metrics
                summary = strategy_instance.get_strategy_summary()
                return {
                    "avg_holding_period_days": summary.get("avg_holding_period_days"),
                    "statistical_validation": summary.get("statistical_validation"),
                    "ou_thresholds": summary.get("ou_thresholds"),
                    "enhancements": summary.get("enhancements", []),
                }
            elif strategy_id == "golden_cross":
                # Get golden cross specific metrics
                summary = strategy_instance.get_strategy_summary()
                return {
                    "crossover_signals": summary.get("total_signals", 0),
                    "ma_periods": summary.get("config", {}).get(
                        "ma_periods", [50, 200]
                    ),
                }
            elif strategy_id == "dual_momentum":
                # Get dual momentum specific metrics
                summary = strategy_instance.get_dual_momentum_summary()
                return {
                    "current_asset": summary.get("current_asset"),
                    "defensive_mode": summary.get("defensive_mode", False),
                    "absolute_momentum_scores": summary.get(
                        "absolute_momentum_scores", {}
                    ),
                    "relative_momentum_scores": summary.get(
                        "relative_momentum_scores", {}
                    ),
                    "qualified_assets_count": len(
                        summary.get("absolute_momentum_scores", {})
                    ),
                }
            elif strategy_id == "sector_rotation":
                # Get sector rotation specific metrics
                summary = strategy_instance.get_sector_rotation_summary()
                return {
                    "sector_rankings": summary.get("sector_rankings", {}),
                    "top_sectors": list(summary.get("sector_rankings", {}).keys())[:4],
                    "sector_scores": summary.get("sector_scores", {}),
                    "benchmark_symbol": summary.get("sector_rotation_config", {}).get(
                        "benchmark_symbol", "SPY"
                    ),
                }
            else:
                return {}
        except Exception as e:
            logger.error(
                f"Error getting strategy-specific metrics for {strategy_id}: {e}"
            )
            return {}

    def _filter_orders_by_strategy(
        self, orders: List[Dict], strategy_id: str
    ) -> List[Dict]:
        """
        Filter orders by strategy (placeholder implementation).

        In a real implementation, orders would be tagged with strategy information.
        For now, we'll use a simple heuristic based on order patterns.
        """
        if not orders:
            return []

        # Placeholder logic - in reality, orders would have strategy tags
        # For now, return all orders and let the dashboard handle the display
        return orders

    def _calculate_win_rate(
        self, filled_orders: List[Dict], current_positions: List[Dict]
    ) -> Optional[float]:
        """Calculate win rate from filled orders and current positions."""
        if not filled_orders:
            return None

        try:
            # Count profitable positions
            profitable_positions = 0
            total_positions = len(current_positions)

            if total_positions > 0:
                for position in current_positions:
                    if position.get("unrealized_pnl", 0) > 0:
                        profitable_positions += 1

                win_rate = (profitable_positions / total_positions) * 100
                return win_rate
            else:
                # No current positions, can't calculate win rate without historical P&L
                return None

        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return None

    def _get_last_signal(self, filled_orders: List[Dict]) -> str:
        """Get the last signal from filled orders."""
        if not filled_orders:
            return "NONE"

        try:
            last_order = filled_orders[0]  # Most recent order
            return last_order.get("action", "NONE").upper()
        except Exception as e:
            logger.error(f"Error getting last signal: {e}")
            return "NONE"

    def _get_fallback_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """Get fallback metrics when strategy data is unavailable."""
        config = self.strategies.get(strategy_id, {})

        return {
            "strategy_id": strategy_id,
            "name": config.get("name", "Unknown Strategy"),
            "status": "ERROR",
            "description": config.get("description", ""),
            "account_connected": False,
            "total_trades": 0,
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "current_positions": 0,
            "win_rate": None,
            "last_signal": "N/A",
            "last_signal_time": None,
            "active_symbols": [],
            "is_active": False,
            "strategy_specific": {},
        }

    def get_strategy_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all strategy statuses.

        Returns:
            Dictionary with overall strategy status information
        """
        all_metrics = self.get_all_strategy_metrics()

        total_strategies = len(all_metrics)
        active_strategies = sum(
            1 for m in all_metrics.values() if m.get("is_active", False)
        )
        total_trades = sum(m.get("total_trades", 0) for m in all_metrics.values())
        total_signals = sum(m.get("total_signals", 0) for m in all_metrics.values())

        return {
            "total_strategies": total_strategies,
            "active_strategies": active_strategies,
            "total_trades": total_trades,
            "total_signals": total_signals,
            "account_connected": self.alpaca_service.is_connected(),
            "last_updated": datetime.now().isoformat(),
        }
