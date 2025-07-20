"""
Performance metrics calculation for backtesting results.
Provides comprehensive trading performance analysis including risk-adjusted returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for trading strategies.

    Includes:
    - Risk-adjusted returns (Sharpe ratio)
    - Drawdown analysis
    - Trade-level statistics
    - Volatility measures
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(
        self, portfolio_values: pd.Series, trades: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics.

        Args:
            portfolio_values: Daily portfolio values (indexed by date)
            trades: List of completed trades (optional)

        Returns:
            Dictionary containing all performance metrics
        """
        if portfolio_values.empty:
            return {"error": "No portfolio data provided"}

        metrics = {}

        # Basic return metrics
        metrics.update(self._calculate_return_metrics(portfolio_values))

        # Risk metrics
        metrics.update(self._calculate_risk_metrics(portfolio_values))

        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(portfolio_values))

        # Trade-level metrics (if trades provided)
        if trades:
            metrics.update(self._calculate_trade_metrics(trades))

        # Additional performance ratios
        metrics.update(self._calculate_performance_ratios(portfolio_values))

        return metrics

    def _calculate_return_metrics(
        self, portfolio_values: pd.Series
    ) -> Dict[str, float]:
        """Calculate basic return metrics."""
        if len(portfolio_values) < 2:
            return {}

        # Calculate returns
        returns = portfolio_values.pct_change().dropna()

        # Total return
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

        # Annualized return (assuming daily data)
        trading_days = len(returns)
        years = trading_days / 252  # Approximate trading days per year
        annualized_return = (
            (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
        )

        # Average daily return
        avg_daily_return = returns.mean()

        return {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "avg_daily_return": avg_daily_return,
            "avg_daily_return_pct": avg_daily_return * 100,
        }

    def _calculate_risk_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """Calculate risk-related metrics."""
        if len(portfolio_values) < 2:
            return {}

        returns = portfolio_values.pct_change().dropna()

        # Volatility (annualized)
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)

        # Downside deviation (volatility of negative returns only)
        negative_returns = returns[returns < 0]
        downside_volatility = (
            negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        )

        return {
            "daily_volatility": daily_volatility,
            "annualized_volatility": annualized_volatility,
            "annualized_volatility_pct": annualized_volatility * 100,
            "downside_volatility": downside_volatility,
            "downside_volatility_pct": downside_volatility * 100,
        }

    def _calculate_drawdown_metrics(
        self, portfolio_values: pd.Series
    ) -> Dict[str, Any]:
        """Calculate drawdown-related metrics."""
        if len(portfolio_values) < 2:
            return {}

        # Calculate running maximum (peak values)
        rolling_max = portfolio_values.expanding().max()

        # Calculate drawdown percentage
        drawdowns = (portfolio_values - rolling_max) / rolling_max

        # Maximum drawdown
        max_drawdown = drawdowns.min()

        # Duration of maximum drawdown
        max_dd_start = None
        max_dd_end = None
        max_dd_duration = 0

        if max_drawdown < 0:
            max_dd_idx = drawdowns.idxmin()

            # Find the start of the max drawdown period
            pre_max_dd = drawdowns.loc[:max_dd_idx]
            max_dd_start_idx = (
                pre_max_dd[pre_max_dd == 0].index[-1]
                if (pre_max_dd == 0).any()
                else pre_max_dd.index[0]
            )
            max_dd_start = max_dd_start_idx

            # Find the end of the max drawdown period (recovery)
            post_max_dd = drawdowns.loc[max_dd_idx:]
            recovery_points = post_max_dd[post_max_dd == 0]
            max_dd_end = (
                recovery_points.index[0]
                if not recovery_points.empty
                else post_max_dd.index[-1]
            )

            max_dd_duration = (max_dd_end - max_dd_start).days

        # Average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0

        # Time underwater (percentage of time in drawdown)
        underwater_periods = (drawdowns < 0).sum()
        time_underwater = (
            underwater_periods / len(drawdowns) if len(drawdowns) > 0 else 0
        )

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "max_drawdown_start": max_dd_start,
            "max_drawdown_end": max_dd_end,
            "max_drawdown_duration_days": max_dd_duration,
            "avg_drawdown": avg_drawdown,
            "avg_drawdown_pct": avg_drawdown * 100,
            "time_underwater_pct": time_underwater * 100,
        }

    def _calculate_trade_metrics(self, trades: List[Any]) -> Dict[str, Any]:
        """Calculate trade-level performance metrics."""
        if not trades:
            return {}

        # Extract trade data
        profits = [t.profit_loss for t in trades]
        profit_pcts = [t.profit_loss_pct for t in trades]
        holding_periods = [t.holding_period_days for t in trades]

        # Separate winning and losing trades
        winning_trades = [t for t in trades if t.profit_loss > 0]
        losing_trades = [t for t in trades if t.profit_loss <= 0]

        # Win rate
        win_rate = len(winning_trades) / len(trades) if trades else 0

        # Profit factor (gross profit / gross loss)
        gross_profit = (
            sum(t.profit_loss for t in winning_trades) if winning_trades else 0
        )
        gross_loss = (
            abs(sum(t.profit_loss for t in losing_trades)) if losing_trades else 0.01
        )  # Avoid division by zero
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average trade metrics
        avg_trade_profit = np.mean(profits) if profits else 0
        avg_winning_trade = (
            np.mean([t.profit_loss for t in winning_trades]) if winning_trades else 0
        )
        avg_losing_trade = (
            np.mean([t.profit_loss for t in losing_trades]) if losing_trades else 0
        )

        # Best and worst trades
        best_trade = max(profits) if profits else 0
        worst_trade = min(profits) if profits else 0

        # Consecutive wins/losses
        consecutive_wins = self._calculate_max_consecutive(
            trades, lambda t: t.profit_loss > 0
        )
        consecutive_losses = self._calculate_max_consecutive(
            trades, lambda t: t.profit_loss <= 0
        )

        # Average holding period
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_trade_profit": avg_trade_profit,
            "avg_winning_trade": avg_winning_trade,
            "avg_losing_trade": avg_losing_trade,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "consecutive_wins_max": consecutive_wins,
            "consecutive_losses_max": consecutive_losses,
            "avg_holding_period_days": avg_holding_period,
        }

    def _calculate_performance_ratios(
        self, portfolio_values: pd.Series
    ) -> Dict[str, float]:
        """Calculate risk-adjusted performance ratios."""
        if len(portfolio_values) < 2:
            return {}

        returns = portfolio_values.pct_change().dropna()

        # Sharpe Ratio
        if returns.std() > 0:
            excess_returns = returns.mean() - (
                self.risk_free_rate / 252
            )  # Daily risk-free rate
            sharpe_ratio = (excess_returns / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Sortino Ratio (like Sharpe but only considers downside volatility)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            excess_returns = returns.mean() - (self.risk_free_rate / 252)
            sortino_ratio = (excess_returns / negative_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = float("inf") if returns.mean() > 0 else 0

        # Calmar Ratio (annualized return / max drawdown)
        annualized_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (
            252 / len(returns)
        ) - 1
        max_drawdown = self._calculate_drawdown_metrics(portfolio_values).get(
            "max_drawdown", -0.01
        )
        calmar_ratio = (
            abs(annualized_return / max_drawdown) if max_drawdown != 0 else float("inf")
        )

        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
        }

    def _calculate_max_consecutive(self, trades: List[Any], condition_func) -> int:
        """Calculate maximum consecutive trades meeting a condition."""
        if not trades:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for trade in trades:
            if condition_func(trade):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def format_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a readable report."""
        if not metrics or "error" in metrics:
            return "No metrics available"

        report_sections = []

        # Overall Performance
        if "total_return_pct" in metrics:
            report_sections.append("=== OVERALL PERFORMANCE ===")
            report_sections.append(f"Total Return: {metrics['total_return_pct']:.2f}%")
            if "annualized_return_pct" in metrics:
                report_sections.append(
                    f"Annualized Return: {metrics['annualized_return_pct']:.2f}%"
                )
            if "annualized_volatility_pct" in metrics:
                report_sections.append(
                    f"Annualized Volatility: {metrics['annualized_volatility_pct']:.2f}%"
                )

        # Risk Metrics
        if "sharpe_ratio" in metrics:
            report_sections.append("\n=== RISK-ADJUSTED RETURNS ===")
            report_sections.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            if "sortino_ratio" in metrics:
                report_sections.append(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
            if "calmar_ratio" in metrics:
                report_sections.append(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")

        # Drawdown
        if "max_drawdown_pct" in metrics:
            report_sections.append("\n=== DRAWDOWN ANALYSIS ===")
            report_sections.append(
                f"Maximum Drawdown: {metrics['max_drawdown_pct']:.2f}%"
            )
            if "max_drawdown_duration_days" in metrics:
                report_sections.append(
                    f"Max Drawdown Duration: {metrics['max_drawdown_duration_days']} days"
                )
            if "time_underwater_pct" in metrics:
                report_sections.append(
                    f"Time Underwater: {metrics['time_underwater_pct']:.1f}%"
                )

        # Trading Statistics
        if "total_trades" in metrics:
            report_sections.append("\n=== TRADING STATISTICS ===")
            report_sections.append(f"Total Trades: {metrics['total_trades']}")
            if "win_rate_pct" in metrics:
                report_sections.append(f"Win Rate: {metrics['win_rate_pct']:.1f}%")
            if "profit_factor" in metrics:
                pf = metrics["profit_factor"]
                pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
                report_sections.append(f"Profit Factor: {pf_str}")
            if "avg_holding_period_days" in metrics:
                report_sections.append(
                    f"Avg Holding Period: {metrics['avg_holding_period_days']:.1f} days"
                )

        return "\n".join(report_sections)
