"""
Modern Backtesting Engine using Backtrader.

This module replaces the current backtesting engine with Backtrader,
the battle-tested framework used by banks and quantitative firms.

Features:
- Used by x2 EuroStoxx and x6 Quantitative Trading firms
- Superior performance and reliability
- Extensive community support and documentation
- Live trading capability when ready
"""

import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ModernBacktestEngine:
    """
    Modern backtesting engine using Backtrader.

    Provides professional-grade backtesting capabilities with
    sophisticated analytics and performance metrics.
    """

    def __init__(self, initial_cash: float = 100000, commission: float = 0.001):
        """
        Initialize the backtesting engine.

        Args:
            initial_cash: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
        """
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(initial_cash)
        self.cerebro.broker.setcommission(commission=commission)

        # Add comprehensive analyzers
        self._add_analyzers()

        self.initial_cash = initial_cash
        logger.info(
            f"Initialized Backtrader engine with ${initial_cash:,.0f} initial cash"
        )

    def _add_analyzers(self):
        """Add comprehensive performance analyzers."""
        analyzers = [
            ("returns", btanalyzers.Returns),
            ("sharpe", btanalyzers.SharpeRatio),
            ("drawdown", btanalyzers.DrawDown),
            ("trades", btanalyzers.TradeAnalyzer),
            ("sqn", btanalyzers.SQN),  # System Quality Number
            ("vwr", btanalyzers.VWR),  # Variability-Weighted Return
            ("calmar", btanalyzers.CalmarRatio),
            ("positions", btanalyzers.PositionsValue),
            ("transactions", btanalyzers.Transactions),
        ]

        for name, analyzer in analyzers:
            self.cerebro.addanalyzer(analyzer, _name=name)

    def add_data(
        self,
        symbol: str,
        data: pd.DataFrame = None,
        start_date: str = None,
        end_date: str = None,
    ):
        """
        Add data feed to the backtest.

        Args:
            symbol: Asset symbol
            data: Optional price data DataFrame
            start_date: Start date for data fetch
            end_date: End date for data fetch
        """
        if data is None:
            # Fetch data using Alpaca API
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")

            if not api_key or not secret_key:
                raise ValueError(
                    "Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
                )

            api = tradeapi.REST(
                api_key,
                secret_key,
                "https://paper-api.alpaca.markets",
                api_version="v2",
            )

            # Set default dates
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

            try:
                bars = api.get_bars(
                    symbol, "1Day", start=start_date, end=end_date, adjustment="all"
                )
                if bars:
                    data = bars.df
                else:
                    raise ValueError(f"No data returned for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data from Alpaca for {symbol}: {e}")
                raise

        # Ensure proper column names for Backtrader
        if data.columns.nlevels > 1:
            data.columns = data.columns.droplevel(
                1
            )  # Remove ticker level if multi-level

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        data = data[required_cols].dropna()

        # Create Backtrader data feed
        data_feed = bt.feeds.PandasData(dataname=data, name=symbol, plot=False)

        self.cerebro.adddata(data_feed, name=symbol)
        logger.info(
            f"Added {symbol} data: {len(data)} bars from {data.index[0]} to {data.index[-1]}"
        )

    def add_strategy(self, strategy_class: bt.Strategy, **kwargs):
        """
        Add strategy to the backtest.

        Args:
            strategy_class: Backtrader strategy class
            **kwargs: Strategy parameters
        """
        self.cerebro.addstrategy(strategy_class, **kwargs)
        logger.info(f"Added strategy: {strategy_class.__name__}")

    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the backtest and return comprehensive results.

        Returns:
            Dictionary containing all performance metrics and analysis
        """
        logger.info("Starting backtest execution...")

        # Run the backtest
        results = self.cerebro.run()

        if not results:
            raise RuntimeError("Backtest failed to execute")

        strat = results[0]  # Get first (and typically only) strategy

        # Extract comprehensive results
        backtest_results = self._extract_results(strat)

        # Calculate additional metrics
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100

        backtest_results.update(
            {
                "initial_cash": self.initial_cash,
                "final_value": final_value,
                "total_return": total_return,
                "total_return_pct": f"{total_return:.2f}%",
                "profit_loss": final_value - self.initial_cash,
            }
        )

        logger.info(
            f"Backtest completed. Final value: ${final_value:,.0f} ({total_return:+.2f}%)"
        )

        return backtest_results

    def _extract_results(self, strategy) -> Dict[str, Any]:
        """Extract comprehensive results from strategy analyzers."""
        results = {}

        # Returns analysis
        if hasattr(strategy.analyzers, "returns"):
            returns_analysis = strategy.analyzers.returns.get_analysis()
            results["returns_analysis"] = returns_analysis

        # Sharpe ratio
        if hasattr(strategy.analyzers, "sharpe"):
            sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
            results["sharpe_ratio"] = sharpe_analysis.get("sharperatio", None)

        # Drawdown analysis
        if hasattr(strategy.analyzers, "drawdown"):
            drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
            results["max_drawdown"] = drawdown_analysis.get("max", {}).get(
                "drawdown", 0
            )
            results["max_drawdown_length"] = drawdown_analysis.get("max", {}).get(
                "len", 0
            )

        # Trade analysis
        if hasattr(strategy.analyzers, "trades"):
            trade_analysis = strategy.analyzers.trades.get_analysis()
            results["trade_analysis"] = {
                "total_trades": trade_analysis.get("total", {}).get("total", 0),
                "won_trades": trade_analysis.get("won", {}).get("total", 0),
                "lost_trades": trade_analysis.get("lost", {}).get("total", 0),
                "win_rate": self._calculate_win_rate(trade_analysis),
                "avg_win": trade_analysis.get("won", {})
                .get("pnl", {})
                .get("average", 0),
                "avg_loss": trade_analysis.get("lost", {})
                .get("pnl", {})
                .get("average", 0),
                "profit_factor": self._calculate_profit_factor(trade_analysis),
                "largest_win": trade_analysis.get("won", {})
                .get("pnl", {})
                .get("max", 0),
                "largest_loss": trade_analysis.get("lost", {})
                .get("pnl", {})
                .get("max", 0),
            }

        # System Quality Number
        if hasattr(strategy.analyzers, "sqn"):
            sqn_analysis = strategy.analyzers.sqn.get_analysis()
            results["system_quality_number"] = sqn_analysis.get("sqn", None)

        # VWR (Variability-Weighted Return)
        if hasattr(strategy.analyzers, "vwr"):
            vwr_analysis = strategy.analyzers.vwr.get_analysis()
            results["vwr"] = vwr_analysis.get("vwr", None)

        # Calmar Ratio
        if hasattr(strategy.analyzers, "calmar"):
            calmar_analysis = strategy.analyzers.calmar.get_analysis()
            results["calmar_ratio"] = calmar_analysis.get("calmar", None)

        return results

    def _calculate_win_rate(self, trade_analysis: Dict) -> float:
        """Calculate win rate from trade analysis."""
        total_trades = trade_analysis.get("total", {}).get("total", 0)
        won_trades = trade_analysis.get("won", {}).get("total", 0)

        if total_trades == 0:
            return 0.0

        return (won_trades / total_trades) * 100

    def _calculate_profit_factor(self, trade_analysis: Dict) -> float:
        """Calculate profit factor from trade analysis."""
        total_won = trade_analysis.get("won", {}).get("pnl", {}).get("total", 0)
        total_lost = abs(trade_analysis.get("lost", {}).get("pnl", {}).get("total", 0))

        if total_lost == 0:
            return float("inf") if total_won > 0 else 0.0

        return total_won / total_lost

    def plot_results(self, save_path: str = None):
        """Plot backtest results."""
        try:
            self.cerebro.plot(style="candlestick", barup="green", bardown="red")
            if save_path:
                import matplotlib.pyplot as plt

                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Plot saved to: {save_path}")
        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")


# Example Backtrader Strategy Adapters
class GoldenCrossStrategy(bt.Strategy):
    """Golden Cross strategy for Backtrader."""

    params = (
        ("fast_period", 50),
        ("slow_period", 200),
        ("position_size", 0.95),  # 95% of available cash
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow_period
        )

        # Golden Cross signal
        self.golden_cross = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.golden_cross[0] == 1:  # Fast MA crosses above slow MA
                size = (
                    self.broker.get_cash()
                    * self.params.position_size
                    / self.data.close[0]
                )
                self.buy(size=size)
                self.log(f"BUY CREATE, Price: {self.data.close[0]:.2f}")

        elif self.golden_cross[0] == -1:  # Fast MA crosses below slow MA
            self.sell(size=self.position.size)
            self.log(f"SELL CREATE, Price: {self.data.close[0]:.2f}")

    def log(self, txt, dt=None):
        """Logging function for the strategy."""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f"{dt.isoformat()}: {txt}")


class MeanReversionStrategy(bt.Strategy):
    """Mean Reversion strategy for Backtrader."""

    params = (
        ("period", 20),
        ("z_threshold", 2.0),
        ("z_exit", 0.5),
        ("position_size", 0.2),  # 20% of portfolio per position
    )

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.period
        )
        self.std = bt.indicators.StandardDeviation(
            self.data.close, period=self.params.period
        )

        # Z-score calculation
        self.zscore = (self.data.close - self.sma) / self.std

    def next(self):
        current_zscore = self.zscore[0]

        if not self.position:
            # Enter long position when oversold (z-score < -threshold)
            if current_zscore < -self.params.z_threshold:
                portfolio_value = self.broker.get_value()
                size = (portfolio_value * self.params.position_size) / self.data.close[
                    0
                ]
                self.buy(size=size)
                self.log(
                    f"BUY CREATE, Z-Score: {current_zscore:.2f}, Price: {self.data.close[0]:.2f}"
                )

        else:
            # Exit when z-score approaches neutral
            if abs(current_zscore) < self.params.z_exit:
                self.sell(size=self.position.size)
                self.log(
                    f"SELL CREATE, Z-Score: {current_zscore:.2f}, Price: {self.data.close[0]:.2f}"
                )

    def log(self, txt, dt=None):
        """Logging function for the strategy."""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f"{dt.isoformat()}: {txt}")


# Factory function
def create_backtest_engine(
    initial_cash: float = 100000, commission: float = 0.001
) -> ModernBacktestEngine:
    """
    Create a modern backtesting engine.

    Args:
        initial_cash: Starting capital
        commission: Commission rate

    Returns:
        ModernBacktestEngine instance
    """
    return ModernBacktestEngine(initial_cash=initial_cash, commission=commission)


# Quick backtest function for convenience
def quick_backtest(
    strategy_name: str,
    symbols: List[str],
    start_date: str = None,
    end_date: str = None,
    initial_cash: float = 100000,
    **strategy_params,
) -> Dict[str, Any]:
    """
    Run a quick backtest with minimal setup.

    Args:
        strategy_name: 'golden_cross' or 'mean_reversion'
        symbols: List of symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_cash: Starting capital
        **strategy_params: Strategy-specific parameters

    Returns:
        Backtest results
    """
    engine = create_backtest_engine(initial_cash=initial_cash)

    # Add data for all symbols
    for symbol in symbols:
        engine.add_data(symbol, start_date=start_date, end_date=end_date)

    # Add strategy
    strategy_map = {
        "golden_cross": GoldenCrossStrategy,
        "mean_reversion": MeanReversionStrategy,
    }

    if strategy_name not in strategy_map:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {list(strategy_map.keys())}"
        )

    engine.add_strategy(strategy_map[strategy_name], **strategy_params)

    # Run backtest
    return engine.run_backtest()
