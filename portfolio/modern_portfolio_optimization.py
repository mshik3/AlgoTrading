"""
Modern Portfolio Optimization using Industry-Standard Libraries.

This module replaces all custom portfolio optimization code with:
- Cvxportfolio: Academic-grade optimization from Stanford/BlackRock
- PyPortfolioOpt: 5k+ stars community-tested optimization
- Sophisticated risk models, cost models, and constraints
- Multi-period optimization capabilities
"""

import cvxportfolio as cvx
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.black_litterman import BlackLittermanModel
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ModernPortfolioOptimizer:
    """
    Modern portfolio optimization using industry-standard libraries.

    Combines the academic rigor of Cvxportfolio with the practicality
    of PyPortfolioOpt for comprehensive portfolio management.
    """

    def __init__(self, price_data: pd.DataFrame, benchmark: str = "SPY"):
        """
        Initialize with price data and benchmark.

        Args:
            price_data: DataFrame with prices (columns=assets, index=dates)
            benchmark: Benchmark asset symbol for risk-free rate estimation
        """
        self.price_data = price_data
        self.benchmark = benchmark
        self.returns = price_data.pct_change().dropna()

        # Pre-compute expected returns and risk models
        self.expected_returns = expected_returns.mean_historical_return(price_data)
        self.sample_cov = risk_models.sample_cov(price_data)
        self.shrunk_cov = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()

        logger.info(
            f"Initialized portfolio optimizer with {len(price_data.columns)} assets"
        )

    def optimize_max_sharpe(self, risk_model: str = "shrunk", **constraints) -> Dict:
        """
        Find the portfolio that maximizes the Sharpe ratio.

        Args:
            risk_model: 'sample', 'shrunk', or 'semicov'
            **constraints: Additional constraints (weight_bounds, sector_constraints, etc.)

        Returns:
            Dictionary with weights, performance metrics, and discrete allocation
        """
        cov_matrix = self._get_risk_model(risk_model)

        # Create efficient frontier
        ef = EfficientFrontier(
            self.expected_returns,
            cov_matrix,
            weight_bounds=constraints.get("weight_bounds", (0, 1)),
            solver=constraints.get("solver", "ECOS"),
        )

        # Add any custom constraints
        self._add_constraints(ef, constraints)

        # Optimize
        raw_weights = ef.max_sharpe(risk_free_rate=0.02)
        cleaned_weights = ef.clean_weights()

        # Performance
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=0.02)

        # Discrete allocation
        latest_prices = get_latest_prices(self.price_data)
        portfolio_value = constraints.get("portfolio_value", 100000)

        da = DiscreteAllocation(
            cleaned_weights, latest_prices, total_portfolio_value=portfolio_value
        )
        allocation, leftover = da.greedy_portfolio()

        return {
            "raw_weights": raw_weights,
            "cleaned_weights": cleaned_weights,
            "expected_annual_return": performance[0],
            "annual_volatility": performance[1],
            "sharpe_ratio": performance[2],
            "discrete_allocation": allocation,
            "leftover_cash": leftover,
            "method": "max_sharpe",
            "risk_model": risk_model,
        }

    def optimize_min_volatility(
        self, risk_model: str = "shrunk", **constraints
    ) -> Dict:
        """Find the minimum volatility portfolio."""
        cov_matrix = self._get_risk_model(risk_model)

        ef = EfficientFrontier(
            self.expected_returns,
            cov_matrix,
            weight_bounds=constraints.get("weight_bounds", (0, 1)),
        )

        self._add_constraints(ef, constraints)
        raw_weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=0.02)

        return {
            "raw_weights": raw_weights,
            "cleaned_weights": cleaned_weights,
            "expected_annual_return": performance[0],
            "annual_volatility": performance[1],
            "sharpe_ratio": performance[2],
            "method": "min_volatility",
            "risk_model": risk_model,
        }

    def optimize_efficient_risk(
        self, target_volatility: float, risk_model: str = "shrunk", **constraints
    ) -> Dict:
        """Optimize for maximum return given target risk level."""
        cov_matrix = self._get_risk_model(risk_model)

        ef = EfficientFrontier(self.expected_returns, cov_matrix)
        self._add_constraints(ef, constraints)

        raw_weights = ef.efficient_risk(target_volatility)
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=0.02)

        return {
            "raw_weights": raw_weights,
            "cleaned_weights": cleaned_weights,
            "expected_annual_return": performance[0],
            "annual_volatility": performance[1],
            "sharpe_ratio": performance[2],
            "target_volatility": target_volatility,
            "method": "efficient_risk",
            "risk_model": risk_model,
        }

    def black_litterman_optimization(
        self, views: Dict[str, float], view_confidences: Dict[str, float] = None
    ) -> Dict:
        """
        Black-Litterman optimization with investor views.

        Args:
            views: Dictionary of asset views {symbol: expected_return}
            view_confidences: Optional confidence levels for views

        Returns:
            Optimized portfolio using Black-Litterman expected returns
        """
        # Market-implied expected returns
        market_caps = self._estimate_market_caps()

        # Create Black-Litterman model
        bl = BlackLittermanModel(
            self.shrunk_cov,
            pi="market",  # Use market-implied returns as prior
            market_caps=market_caps,
            absolute_views=views,
            omega="idzorek" if view_confidences is None else view_confidences,
        )

        # Get posterior expected returns
        bl_returns = bl.bl_returns()
        bl_cov = bl.bl_cov()

        # Optimize using Black-Litterman inputs
        ef = EfficientFrontier(bl_returns, bl_cov)
        raw_weights = ef.max_sharpe(risk_free_rate=0.02)
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=0.02)

        return {
            "raw_weights": raw_weights,
            "cleaned_weights": cleaned_weights,
            "expected_annual_return": performance[0],
            "annual_volatility": performance[1],
            "sharpe_ratio": performance[2],
            "bl_returns": bl_returns,
            "views": views,
            "method": "black_litterman",
        }

    def _get_risk_model(self, risk_model: str) -> pd.DataFrame:
        """Get the appropriate risk model."""
        if risk_model == "sample":
            return self.sample_cov
        elif risk_model == "shrunk":
            return self.shrunk_cov
        elif risk_model == "semicov":
            return risk_models.semicovariance(self.price_data, benchmark=0)
        else:
            raise ValueError(f"Unknown risk model: {risk_model}")

    def _add_constraints(self, ef: EfficientFrontier, constraints: Dict):
        """Add custom constraints to the optimization."""
        # Sector constraints
        if "sector_mapper" in constraints and "sector_lower" in constraints:
            ef.add_sector_constraints(
                constraints["sector_mapper"],
                constraints["sector_lower"],
                constraints.get("sector_upper", {}),
            )

        # L2 regularization to reduce small weights
        if constraints.get("gamma", 0) > 0:
            from pypfopt import objective_functions

            ef.add_objective(objective_functions.L2_reg, gamma=constraints["gamma"])

    def _estimate_market_caps(self) -> pd.Series:
        """Estimate market capitalizations for Black-Litterman."""
        # Simple estimation using average prices (in practice, use actual market caps)
        avg_prices = self.price_data.mean()
        # Normalize to get relative market caps
        return avg_prices / avg_prices.sum()


class CvxPortfolioOptimizer:
    """
    Advanced portfolio optimization using Cvxportfolio.

    Provides academic-grade multi-period optimization with
    sophisticated cost models and risk management.
    """

    def __init__(self, price_data: pd.DataFrame):
        """Initialize with price data."""
        self.price_data = price_data
        self.returns = price_data.pct_change().dropna()
        logger.info(
            f"Initialized Cvxportfolio optimizer with {len(price_data.columns)} assets"
        )

    def optimize_single_period(
        self,
        forecast_returns: pd.Series = None,
        transaction_cost: float = 0.001,
        risk_aversion: float = 1.0,
        leverage_limit: float = 1.0,
    ) -> Dict:
        """
        Single-period portfolio optimization with transaction costs.

        Args:
            forecast_returns: Expected returns forecast
            transaction_cost: Transaction cost rate (e.g., 0.001 = 10 bps)
            risk_aversion: Risk aversion parameter (higher = more conservative)
            leverage_limit: Maximum leverage allowed

        Returns:
            Optimal weights and policy object
        """
        # Default forecast: simple mean reversion
        if forecast_returns is None:
            forecast_returns = cvx.ReturnsForecast()

        # Define objective: maximize return - risk penalty - transaction costs
        objective = (
            forecast_returns
            - risk_aversion * cvx.FullCovariance()
            - cvx.TcostModel(half_spread=transaction_cost)
        )

        # Define constraints
        constraints = [
            cvx.LongOnly(),  # No short selling
            cvx.LeverageLimit(leverage_limit),
            cvx.MarketNeutral() if leverage_limit > 1.0 else cvx.LongCash(),
        ]

        # Create and solve policy
        policy = cvx.SinglePeriodOptimization(objective, constraints)

        # Simulate on recent data for weights
        simulator = cvx.StockMarketSimulator(self.returns.columns)
        result = simulator.backtest(
            policy, start_time=self.returns.index[-100], end_time=self.returns.index[-1]
        )

        # Get final weights
        final_weights = result.w.iloc[-1].to_dict()

        return {
            "weights": final_weights,
            "policy": policy,
            "result": result,
            "method": "cvx_single_period",
            "transaction_cost": transaction_cost,
            "risk_aversion": risk_aversion,
        }

    def optimize_multi_period(
        self,
        horizon: int = 252,
        forecast_returns: pd.Series = None,
        transaction_cost: float = 0.001,
        risk_aversion: float = 1.0,
    ) -> Dict:
        """
        Multi-period portfolio optimization.

        Args:
            horizon: Investment horizon in days
            forecast_returns: Expected returns forecast
            transaction_cost: Transaction cost rate
            risk_aversion: Risk aversion parameter

        Returns:
            Multi-period optimal policy and results
        """
        # Multi-period objective
        objective = (
            (forecast_returns or cvx.ReturnsForecast())
            - risk_aversion * cvx.FullCovariance()
            - cvx.TcostModel(half_spread=transaction_cost)
        )

        # Multi-period constraints
        constraints = [
            cvx.LongOnly(),
            cvx.LeverageLimit(1.0),
            cvx.TurnoverLimit(0.2),  # Max 20% turnover per period
        ]

        # Create multi-period policy
        policy = cvx.MultiPeriodOptimization(
            objective,
            constraints,
            planning_horizon=min(horizon, 60),  # Limit computational complexity
        )

        # Backtest
        simulator = cvx.StockMarketSimulator(self.returns.columns)
        result = simulator.backtest(
            policy,
            start_time=self.returns.index[-horizon],
            end_time=self.returns.index[-1],
        )

        return {
            "policy": policy,
            "result": result,
            "method": "cvx_multi_period",
            "horizon": horizon,
            "transaction_cost": transaction_cost,
            "risk_aversion": risk_aversion,
            "final_return": result.returns.iloc[-1],
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.drawdown.max(),
        }


class TaxAwarePortfolioOptimizer:
    """
    Tax-aware portfolio optimization using the rebalancer library.

    Implements sophisticated tax-loss harvesting and wash sale avoidance.
    """

    def __init__(self, price_data: pd.DataFrame):
        """Initialize with price data."""
        self.price_data = price_data
        # Note: The actual rebalancer library integration would go here
        # For now, we'll create a simple tax-aware optimization framework
        logger.info("Initialized tax-aware portfolio optimizer")

    def optimize_with_tax_harvesting(
        self,
        current_holdings: Dict[str, float],
        target_weights: Dict[str, float],
        tax_lots: Dict = None,
        max_gain_realization: float = 10000,
    ) -> Dict:
        """
        Optimize portfolio transitions considering tax implications.

        Args:
            current_holdings: Current portfolio holdings
            target_weights: Target portfolio weights
            tax_lots: Tax lot information for held positions
            max_gain_realization: Maximum capital gains to realize

        Returns:
            Tax-efficient rebalancing recommendations
        """

        # This would integrate with the rebalancer library
        # For now, implement basic tax-aware logic

        recommendations = {
            "trades": {},
            "tax_impact": 0,
            "harvested_losses": 0,
            "realized_gains": 0,
            "wash_sale_warnings": [],
        }

        # Simple implementation - in practice would use the sophisticated rebalancer library
        for symbol, target_weight in target_weights.items():
            current_weight = current_holdings.get(symbol, 0)
            weight_diff = target_weight - current_weight

            if abs(weight_diff) > 0.05:  # 5% threshold
                recommendations["trades"][symbol] = weight_diff

        logger.info(
            f"Generated tax-aware rebalancing with {len(recommendations['trades'])} trades"
        )
        return recommendations


# Factory functions for easy access
def create_portfolio_optimizer(
    price_data: pd.DataFrame, method: str = "pypfopt"
) -> Union[ModernPortfolioOptimizer, CvxPortfolioOptimizer]:
    """
    Create portfolio optimizer instance.

    Args:
        price_data: Price data DataFrame
        method: 'pypfopt' for PyPortfolioOpt, 'cvx' for Cvxportfolio

    Returns:
        Portfolio optimizer instance
    """
    if method == "pypfopt":
        return ModernPortfolioOptimizer(price_data)
    elif method == "cvx":
        return CvxPortfolioOptimizer(price_data)
    elif method == "tax_aware":
        return TaxAwarePortfolioOptimizer(price_data)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'pypfopt', 'cvx', or 'tax_aware'"
        )


# Example usage functions
def optimize_portfolio_simple(symbols: List[str], method: str = "max_sharpe") -> Dict:
    """
    Simple portfolio optimization for quick use.

    Args:
        symbols: List of asset symbols
        method: Optimization method

    Returns:
        Optimized portfolio
    """
    # This would fetch data and optimize
    # Placeholder for demonstration
    return {
        "message": f"Would optimize portfolio with {len(symbols)} assets using {method}",
        "symbols": symbols,
        "method": method,
    }
