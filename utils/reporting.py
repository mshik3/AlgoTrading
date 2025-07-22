"""
Performance monitoring and reporting utilities.
Generates comprehensive strategy performance reports and alerts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import json

from strategies.base import BaseStrategy, StrategySignal
from backtesting import BacktestingEngine, PerformanceMetrics, BacktestResult
from dashboard.data.live_data import PaperTradingSimulator

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """
    Generate comprehensive performance reports for trading strategies.

    Features:
    - Strategy performance analysis
    - Risk metrics calculation
    - Alert generation for significant events
    - Performance comparison between strategies
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize performance reporter.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        self.metrics_calculator = PerformanceMetrics()

    def generate_strategy_report(
        self,
        strategy: BaseStrategy,
        backtest_result: Optional[BacktestResult] = None,
        paper_trader: Optional[PaperTradingSimulator] = None,
    ) -> str:
        """
        Generate comprehensive strategy performance report.

        Args:
            strategy: The trading strategy
            backtest_result: Backtesting results (optional)
            paper_trader: Paper trading simulator (optional)

        Returns:
            Formatted performance report string
        """
        report_sections = []

        # Header
        report_sections.append("=" * 80)
        report_sections.append(f"STRATEGY PERFORMANCE REPORT: {strategy.name}")
        report_sections.append(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_sections.append("=" * 80)

        # Strategy overview
        strategy_summary = strategy.get_strategy_summary()
        report_sections.append("\n📊 STRATEGY OVERVIEW")
        report_sections.append("-" * 40)
        report_sections.append(f"Strategy Name: {strategy_summary['strategy_name']}")
        report_sections.append(
            f"Active Symbols: {', '.join(strategy_summary['active_symbols'])}"
        )
        report_sections.append(
            f"Current Positions: {strategy_summary['current_positions']}"
        )
        report_sections.append(f"Total Signals: {strategy_summary['total_signals']}")
        report_sections.append(
            f"Active Status: {'✅ Active' if strategy_summary['is_active'] else '❌ Inactive'}"
        )

        # Backtest results
        if backtest_result:
            report_sections.append("\n📈 BACKTESTING RESULTS")
            report_sections.append("-" * 40)
            report_sections.append(
                f"Test Period: {backtest_result.start_date.date()} to {backtest_result.end_date.date()}"
            )
            report_sections.append(
                f"Initial Capital: ${backtest_result.initial_capital:,.2f}"
            )
            report_sections.append(
                f"Final Capital: ${backtest_result.final_capital:,.2f}"
            )
            report_sections.append(
                f"Total Return: ${backtest_result.total_return:,.2f} ({backtest_result.total_return_pct:.2f}%)"
            )

            # Performance metrics
            if backtest_result.performance_metrics:
                metrics_report = self.metrics_calculator.format_metrics_report(
                    backtest_result.performance_metrics
                )
                report_sections.append(f"\n{metrics_report}")

            # Trade analysis
            if backtest_result.trades:
                report_sections.append(f"\n💰 TRADE ANALYSIS")
                report_sections.append("-" * 40)

                winning_trades = [
                    t for t in backtest_result.trades if t.profit_loss > 0
                ]
                losing_trades = [
                    t for t in backtest_result.trades if t.profit_loss <= 0
                ]

                report_sections.append(f"Total Trades: {len(backtest_result.trades)}")
                report_sections.append(
                    f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(backtest_result.trades)*100:.1f}%)"
                )
                report_sections.append(
                    f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(backtest_result.trades)*100:.1f}%)"
                )

                if winning_trades:
                    avg_win = np.mean([t.profit_loss for t in winning_trades])
                    report_sections.append(f"Average Winning Trade: ${avg_win:.2f}")

                if losing_trades:
                    avg_loss = np.mean([t.profit_loss for t in losing_trades])
                    report_sections.append(f"Average Losing Trade: ${avg_loss:.2f}")

                # Best and worst trades
                best_trade = max(backtest_result.trades, key=lambda t: t.profit_loss)
                worst_trade = min(backtest_result.trades, key=lambda t: t.profit_loss)

                report_sections.append(
                    f"Best Trade: {best_trade.symbol} ${best_trade.profit_loss:.2f} "
                    f"({best_trade.profit_loss_pct:.2f}%) on {best_trade.exit_date.date()}"
                )
                report_sections.append(
                    f"Worst Trade: {worst_trade.symbol} ${worst_trade.profit_loss:.2f} "
                    f"({worst_trade.profit_loss_pct:.2f}%) on {worst_trade.exit_date.date()}"
                )

        # Paper trading results
        if paper_trader:
            report_sections.append("\n🧪 PAPER TRADING RESULTS")
            report_sections.append("-" * 40)

            portfolio_summary = paper_trader.get_portfolio_summary()
            positions_summary = paper_trader.get_positions_summary()

            report_sections.append(
                f"Portfolio Value: ${portfolio_summary['total_value']:,.2f}"
            )
            report_sections.append(f"Cash: ${portfolio_summary['cash']:,.2f}")
            report_sections.append(
                f"Total P&L: ${portfolio_summary['total_pnl']:,.2f} ({portfolio_summary['total_return_pct']:+.2f}%)"
            )
            report_sections.append(
                f"Number of Trades: {portfolio_summary['num_trades']}"
            )
            report_sections.append(
                f"Open Positions: {portfolio_summary['num_positions']}"
            )

            if positions_summary:
                report_sections.append(f"\n📍 Current Positions:")
                for pos in positions_summary:
                    pnl_indicator = "🟢" if pos["unrealized_pnl"] >= 0 else "🔴"
                    report_sections.append(
                        f"  {pnl_indicator} {pos['symbol']}: {pos['quantity']} shares, "
                        f"P&L: ${pos['unrealized_pnl']:+.2f} ({pos['unrealized_pnl_pct']:+.1f}%)"
                    )

        # Risk assessment
        report_sections.append(f"\n⚠️  RISK ASSESSMENT")
        report_sections.append("-" * 40)

        risk_alerts = self._generate_risk_alerts(
            strategy, backtest_result, paper_trader
        )
        if risk_alerts:
            for alert in risk_alerts:
                report_sections.append(f"  {alert}")
        else:
            report_sections.append("  ✅ No significant risk alerts")

        # Recommendations
        recommendations = self._generate_recommendations(
            strategy, backtest_result, paper_trader
        )
        if recommendations:
            report_sections.append(f"\n💡 RECOMMENDATIONS")
            report_sections.append("-" * 40)
            for rec in recommendations:
                report_sections.append(f"  • {rec}")

        return "\n".join(report_sections)

    def _generate_risk_alerts(
        self,
        strategy: BaseStrategy,
        backtest_result: Optional[BacktestResult],
        paper_trader: Optional[PaperTradingSimulator],
    ) -> List[str]:
        """Generate risk alerts based on performance metrics."""
        alerts = []

        if backtest_result:
            metrics = backtest_result.performance_metrics

            # High drawdown alert
            if metrics.get("max_drawdown_pct", 0) < -20:
                alerts.append(
                    f"🚨 High maximum drawdown: {metrics['max_drawdown_pct']:.1f}%"
                )

            # Low Sharpe ratio alert
            if metrics.get("sharpe_ratio", 0) < 0.5:
                alerts.append(f"⚠️  Low Sharpe ratio: {metrics['sharpe_ratio']:.2f}")

            # High volatility alert
            if metrics.get("annualized_volatility_pct", 0) > 30:
                alerts.append(
                    f"⚠️  High volatility: {metrics['annualized_volatility_pct']:.1f}%"
                )

            # Low win rate alert
            if metrics.get("win_rate_pct", 0) < 40:
                alerts.append(f"⚠️  Low win rate: {metrics['win_rate_pct']:.1f}%")

            # Long underwater periods
            if metrics.get("time_underwater_pct", 0) > 50:
                alerts.append(
                    f"⚠️  High time underwater: {metrics['time_underwater_pct']:.1f}%"
                )

        if paper_trader:
            summary = paper_trader.get_portfolio_summary()

            # Significant losses in paper trading
            if summary["total_return_pct"] < -10:
                alerts.append(
                    f"🚨 Significant paper trading losses: {summary['total_return_pct']:+.1f}%"
                )

            # Too many positions
            if summary["num_positions"] > 5:
                alerts.append(
                    f"⚠️  High number of positions: {summary['num_positions']}"
                )

        return alerts

    def _generate_recommendations(
        self,
        strategy: BaseStrategy,
        backtest_result: Optional[BacktestResult],
        paper_trader: Optional[PaperTradingSimulator],
    ) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        if backtest_result:
            metrics = backtest_result.performance_metrics

            # Performance-based recommendations
            if metrics.get("total_return_pct", 0) > 0:
                recommendations.append(
                    "Strategy shows positive returns - consider live deployment"
                )
            else:
                recommendations.append(
                    "Strategy shows negative returns - review parameters before live trading"
                )

            if metrics.get("sharpe_ratio", 0) < 1.0:
                recommendations.append(
                    "Consider tightening risk management to improve risk-adjusted returns"
                )

            if metrics.get("max_drawdown_pct", 0) < -15:
                recommendations.append(
                    "Implement additional stop-loss mechanisms to reduce drawdowns"
                )

            # Trade frequency recommendations
            total_days = (backtest_result.end_date - backtest_result.start_date).days
            if len(backtest_result.trades) / total_days > 0.1:
                recommendations.append(
                    "High trade frequency - monitor transaction costs"
                )
            elif len(backtest_result.trades) / total_days < 0.01:
                recommendations.append(
                    "Low trade frequency - consider expanding symbol universe"
                )

        # Strategy-specific recommendations
        if strategy.name == "Golden Cross":
            recommendations.append(
                "Golden Cross works best in trending markets - monitor market conditions"
            )
            recommendations.append(
                "Consider adding volume confirmation for better signal quality"
            )

        return recommendations

    def save_report(self, report: str, filename: str = None) -> str:
        """
        Save report to file.

        Args:
            report: Report content
            filename: Custom filename (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.txt"

        # Create output directory if it doesn't exist
        import os

        os.makedirs(self.output_dir, exist_ok=True)

        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, "w") as f:
                f.write(report)
            logger.info(f"Performance report saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
            return ""

    def generate_alert_summary(
        self,
        strategies: List[BaseStrategy],
        backtest_results: Optional[List[BacktestResult]] = None,
    ) -> str:
        """
        Generate a summary of alerts across multiple strategies.

        Args:
            strategies: List of strategies to analyze
            backtest_results: Corresponding backtest results (optional)

        Returns:
            Alert summary string
        """
        alert_summary = []
        alert_summary.append("=" * 60)
        alert_summary.append("STRATEGY ALERT SUMMARY")
        alert_summary.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        alert_summary.append("=" * 60)

        for i, strategy in enumerate(strategies):
            backtest_result = (
                backtest_results[i]
                if backtest_results and i < len(backtest_results)
                else None
            )

            alerts = self._generate_risk_alerts(strategy, backtest_result, None)

            alert_summary.append(f"\n🎯 {strategy.name}")
            alert_summary.append("-" * 30)

            if alerts:
                for alert in alerts:
                    alert_summary.append(f"  {alert}")
            else:
                alert_summary.append("  ✅ No alerts")

        return "\n".join(alert_summary)


def create_performance_dashboard(strategies: List[BaseStrategy]) -> Dict[str, Any]:
    """
    Create a performance dashboard data structure.

    Args:
        strategies: List of strategies to include

    Returns:
        Dashboard data dictionary
    """
    dashboard = {
        "timestamp": datetime.now().isoformat(),
        "strategies": [],
        "summary": {
            "total_strategies": len(strategies),
            "active_strategies": sum(1 for s in strategies if s.is_active),
            "total_positions": sum(len(s.positions) for s in strategies),
            "total_signals": sum(len(s.signals_history) for s in strategies),
        },
    }

    for strategy in strategies:
        strategy_data = {
            "name": strategy.name,
            "symbols": strategy.symbols,
            "is_active": strategy.is_active,
            "positions": len(strategy.positions),
            "signals": len(strategy.signals_history),
            "last_signal": (
                strategy.signals_history[-1].timestamp.isoformat()
                if strategy.signals_history
                else None
            ),
        }
        dashboard["strategies"].append(strategy_data)

    return dashboard
