"""
Modern Portfolio Optimization Service for Dashboard.

This service integrates the modern portfolio optimization libraries
with the dashboard, using Alpaca API data instead of Yahoo Finance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import json

# Import modern portfolio optimization
from portfolio.alpaca_data_adapter import create_alpaca_optimizer

logger = logging.getLogger(__name__)


class ModernPortfolioService:
    """
    Modern portfolio optimization service for the dashboard.

    Provides professional-grade portfolio optimization using:
    - Cvxportfolio (Stanford/BlackRock academic-grade)
    - PyPortfolioOpt (5k+ stars community-tested)
    - Alpaca API for reliable market data
    """

    def __init__(self):
        """Initialize the modern portfolio service."""
        try:
            self.alpaca_optimizer = create_alpaca_optimizer()
            logger.info("Modern portfolio service initialized with Alpaca API")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca optimizer: {e}")
            self.alpaca_optimizer = None

    def get_optimization_results(
        self,
        symbols: List[str],
        method: str = "max_sharpe",
        portfolio_value: float = 100000,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get portfolio optimization results for dashboard display.

        Args:
            symbols: List of symbols to optimize
            method: Optimization method
            portfolio_value: Portfolio value for allocation
            **kwargs: Additional optimization parameters

        Returns:
            Dictionary with optimization results formatted for dashboard
        """
        if not self.alpaca_optimizer:
            return {"error": "Alpaca optimizer not available"}

        try:
            # Run optimization
            result = self.alpaca_optimizer.optimize_portfolio(
                symbols=symbols,
                method=method,
                portfolio_value=portfolio_value,
                **kwargs,
            )

            # Format results for dashboard
            dashboard_result = {
                "success": True,
                "method": method,
                "portfolio_value": portfolio_value,
                "symbols": symbols,
                "weights": result["cleaned_weights"],
                "expected_return": result["expected_annual_return"],
                "volatility": result["annual_volatility"],
                "sharpe_ratio": result["sharpe_ratio"],
                "allocation": result.get("discrete_allocation", {}),
                "leftover_cash": result.get("leftover_cash", 0),
                "alpaca_data": result.get("alpaca_data", {}),
                "timestamp": datetime.now().isoformat(),
            }

            # Add performance metrics
            if "alpaca_data" in result and "account_info" in result["alpaca_data"]:
                account_info = result["alpaca_data"]["account_info"]
                dashboard_result["account_info"] = account_info

                # Calculate additional metrics
                if account_info.get("portfolio_value"):
                    dashboard_result["current_portfolio_value"] = account_info[
                        "portfolio_value"
                    ]
                    dashboard_result["cash_available"] = account_info.get("cash", 0)

            return dashboard_result

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_black_litterman_results(
        self,
        symbols: List[str],
        views: Dict[str, float],
        portfolio_value: float = 100000,
    ) -> Dict[str, Any]:
        """
        Get Black-Litterman optimization results.

        Args:
            symbols: List of symbols
            views: Investor views {symbol: expected_return}
            portfolio_value: Portfolio value for allocation

        Returns:
            Black-Litterman optimization results
        """
        if not self.alpaca_optimizer:
            return {"error": "Alpaca optimizer not available"}

        try:
            result = self.alpaca_optimizer.black_litterman_optimization(
                symbols=symbols, views=views, portfolio_value=portfolio_value
            )

            return {
                "success": True,
                "method": "black_litterman",
                "views": views,
                "weights": result["cleaned_weights"],
                "expected_return": result["expected_annual_return"],
                "volatility": result["annual_volatility"],
                "sharpe_ratio": result["sharpe_ratio"],
                "bl_returns": result.get("bl_returns", {}),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_portfolio_comparison(
        self, symbols: List[str], portfolio_value: float = 100000
    ) -> Dict[str, Any]:
        """
        Compare different optimization methods.

        Args:
            symbols: List of symbols
            portfolio_value: Portfolio value

        Returns:
            Comparison of different optimization methods
        """
        methods = ["max_sharpe", "min_volatility"]
        comparison = {}

        for method in methods:
            try:
                result = self.get_optimization_results(
                    symbols=symbols, method=method, portfolio_value=portfolio_value
                )
                comparison[method] = result
            except Exception as e:
                comparison[method] = {"error": str(e)}

        return {
            "success": True,
            "comparison": comparison,
            "symbols": symbols,
            "portfolio_value": portfolio_value,
            "timestamp": datetime.now().isoformat(),
        }

    def get_rebalancing_recommendations(
        self,
        symbols: List[str],
        current_weights: Dict[str, float],
        target_method: str = "max_sharpe",
    ) -> Dict[str, Any]:
        """
        Get rebalancing recommendations.

        Args:
            symbols: List of symbols
            current_weights: Current portfolio weights
            target_method: Target optimization method

        Returns:
            Rebalancing recommendations
        """
        try:
            # Get target weights
            target_result = self.get_optimization_results(
                symbols=symbols, method=target_method
            )

            if not target_result.get("success"):
                return target_result

            target_weights = target_result["weights"]

            # Calculate rebalancing trades
            rebalancing_trades = {}
            for symbol in symbols:
                current_weight = current_weights.get(symbol, 0)
                target_weight = target_weights.get(symbol, 0)
                weight_diff = target_weight - current_weight

                if abs(weight_diff) > 0.01:  # 1% threshold
                    rebalancing_trades[symbol] = {
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "weight_diff": weight_diff,
                        "action": "buy" if weight_diff > 0 else "sell",
                    }

            return {
                "success": True,
                "current_weights": current_weights,
                "target_weights": target_weights,
                "rebalancing_trades": rebalancing_trades,
                "method": target_method,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Rebalancing recommendations failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_risk_metrics(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get comprehensive risk metrics.

        Args:
            symbols: List of symbols

        Returns:
            Risk metrics for the portfolio
        """
        try:
            # Get optimization results for risk metrics
            result = self.alpaca_optimizer.optimize_portfolio(
                symbols=symbols, method="max_sharpe"
            )

            # Calculate additional risk metrics
            alpaca_data = result.get("alpaca_data", {})
            account_info = alpaca_data.get("account_info", {})
            current_positions = alpaca_data.get("current_positions", {})

            risk_metrics = {
                "expected_return": result["expected_annual_return"],
                "volatility": result["annual_volatility"],
                "sharpe_ratio": result["sharpe_ratio"],
                "portfolio_value": account_info.get("portfolio_value", 0),
                "cash_ratio": account_info.get("cash", 0)
                / max(account_info.get("portfolio_value", 1), 1),
                "position_count": len(current_positions),
                "concentration_risk": self._calculate_concentration_risk(
                    result["cleaned_weights"]
                ),
                "timestamp": datetime.now().isoformat(),
            }

            return {"success": True, "risk_metrics": risk_metrics, "symbols": symbols}

        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_concentration_risk(self, weights: Dict[str, float]) -> float:
        """Calculate concentration risk using Herfindahl index."""
        if not weights:
            return 0.0

        # Herfindahl-Hirschman Index for concentration
        hhi = sum(weight**2 for weight in weights.values())
        return hhi


# Global service instance
modern_portfolio_service = ModernPortfolioService()


def get_modern_portfolio_service() -> ModernPortfolioService:
    """Get the global modern portfolio service instance."""
    return modern_portfolio_service
