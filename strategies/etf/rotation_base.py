"""
Base ETF Rotation Strategy.

Provides abstract interface and common functionality for all ETF rotation strategies.
Implements momentum-based rotation logic that can be extended for specific approaches.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from ..base import BaseStrategy, StrategySignal, SignalType
from indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class BaseETFRotationStrategy(BaseStrategy):
    """
    Abstract base class for ETF rotation strategies.

    Provides common functionality for momentum-based ETF rotation including:
    - Momentum calculation and ranking
    - Asset allocation and rebalancing
    - Risk management and position sizing
    - Performance tracking and analysis
    """

    def __init__(self, name: str, etf_universe: Dict[str, List[str]], **config):
        """
        Initialize ETF rotation strategy.

        Args:
            name: Strategy name
            etf_universe: Dictionary mapping category -> list of ETF symbols
            **config: Strategy-specific configuration parameters
        """
        # Flatten ETF universe into single list for base strategy
        all_symbols = []
        for category, symbols in etf_universe.items():
            all_symbols.extend(symbols)

        # Default configuration for ETF rotation
        default_config = {
            "momentum_lookback": 252,  # 1 year for momentum calculation
            "rebalance_frequency": 21,  # Rebalance every 21 days (monthly)
            "max_positions": 5,  # Maximum number of ETFs to hold
            "position_size": 0.20,  # Equal weight per position (20% each)
            "min_momentum_threshold": 0.0,  # Minimum momentum to consider
            "risk_free_rate": 0.02,  # 2% risk-free rate for absolute momentum
            "momentum_ranking_method": "returns",  # 'returns', 'sharpe', 'sortino'
            "use_relative_strength": True,  # Use relative strength vs benchmark
            "benchmark_symbol": "SPY",  # Benchmark for relative strength
            "stop_loss_pct": 0.15,  # 15% stop loss
            "take_profit_pct": 0.30,  # 30% take profit
            "min_confidence": 0.6,  # Lower threshold for rotation strategies
        }

        # Merge with provided config
        merged_config = {**default_config, **config}

        super().__init__(name=name, symbols=all_symbols, **merged_config)

        # Store ETF universe for category-based analysis
        self.etf_universe = etf_universe
        self.etf_categories = list(etf_universe.keys())

        # Track rotation state
        self.current_allocations = {}  # symbol -> allocation percentage
        self.last_rebalance_date = None
        self.momentum_rankings = {}
        self.relative_strength_scores = {}

    def calculate_momentum(
        self,
        data: pd.DataFrame,
        lookback: Optional[int] = None,
        method: str = "returns",
    ) -> float:
        """
        Calculate momentum for a given dataset.

        Args:
            data: OHLCV data
            lookback: Lookback period (defaults to config value)
            method: Momentum calculation method ('returns', 'sharpe', 'sortino')

        Returns:
            Momentum score
        """
        if lookback is None:
            lookback = self.config["momentum_lookback"]

        if len(data) < lookback:
            return np.nan

        # Calculate returns
        returns = data["Close"].pct_change().dropna()
        recent_returns = returns.tail(lookback)

        if method == "returns":
            # Simple cumulative return
            return data["Close"].iloc[-1] / data["Close"].iloc[-lookback] - 1

        elif method == "sharpe":
            # Sharpe ratio (excess return / volatility)
            if len(recent_returns) < 2:
                return np.nan
            excess_returns = recent_returns - self.config["risk_free_rate"] / 252
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        elif method == "sortino":
            # Sortino ratio (excess return / downside deviation)
            if len(recent_returns) < 2:
                return np.nan
            excess_returns = recent_returns - self.config["risk_free_rate"] / 252
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0:
                return np.nan
            downside_deviation = downside_returns.std()
            return excess_returns.mean() / downside_deviation * np.sqrt(252)

        else:
            raise ValueError(f"Unknown momentum method: {method}")

    def calculate_relative_strength(
        self,
        symbol_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        lookback: int = 252,
    ) -> float:
        """
        Calculate relative strength vs benchmark.

        Args:
            symbol_data: ETF OHLCV data
            benchmark_data: Benchmark OHLCV data
            lookback: Lookback period

        Returns:
            Relative strength score
        """
        if len(symbol_data) < lookback or len(benchmark_data) < lookback:
            return np.nan

        # Calculate cumulative returns
        symbol_return = (
            symbol_data["Close"].iloc[-1] / symbol_data["Close"].iloc[-lookback] - 1
        )
        benchmark_return = (
            benchmark_data["Close"].iloc[-1] / benchmark_data["Close"].iloc[-lookback]
            - 1
        )

        # Relative strength = symbol return - benchmark return
        return symbol_return - benchmark_return

    def rank_etfs_by_momentum(
        self, market_data: Dict[str, pd.DataFrame], category: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank ETFs by momentum within a category or across all ETFs.

        Args:
            market_data: Dictionary of symbol -> OHLCV data
            category: ETF category to rank (None for all ETFs)

        Returns:
            List of (symbol, momentum_score) tuples, sorted by momentum
        """
        rankings = []

        # Determine symbols to rank
        if category:
            symbols = self.etf_universe.get(category, [])
        else:
            symbols = self.symbols

        for symbol in symbols:
            if symbol not in market_data or market_data[symbol].empty:
                continue

            try:
                momentum = self.calculate_momentum(
                    market_data[symbol], method=self.config["momentum_ranking_method"]
                )

                if (
                    not np.isnan(momentum)
                    and momentum >= self.config["min_momentum_threshold"]
                ):
                    rankings.append((symbol, momentum))

            except Exception as e:
                logger.warning(f"Error calculating momentum for {symbol}: {str(e)}")
                continue

        # Sort by momentum (highest first)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def should_rebalance(self, current_date: datetime) -> bool:
        """
        Determine if it's time to rebalance the portfolio.

        Args:
            current_date: Current date

        Returns:
            True if rebalancing is needed
        """
        if self.last_rebalance_date is None:
            return True

        days_since_rebalance = (current_date - self.last_rebalance_date).days
        return days_since_rebalance >= self.config["rebalance_frequency"]

    def calculate_optimal_allocations(
        self, market_data: Dict[str, pd.DataFrame], portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate optimal ETF allocations based on momentum rankings.

        Args:
            market_data: Dictionary of symbol -> OHLCV data
            portfolio_value: Current portfolio value

        Returns:
            Dictionary mapping symbol -> target allocation percentage
        """
        # Get top-ranked ETFs
        rankings = self.rank_etfs_by_momentum(market_data)
        top_etfs = rankings[: self.config["max_positions"]]

        # Calculate equal-weighted allocations
        allocations = {}
        if top_etfs:
            allocation_per_etf = 1.0 / len(top_etfs)
            for symbol, _ in top_etfs:
                allocations[symbol] = allocation_per_etf

        return allocations

    def generate_rebalancing_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        current_allocations: Dict[str, float],
        target_allocations: Dict[str, float],
    ) -> List[StrategySignal]:
        """
        Generate signals to rebalance portfolio to target allocations.

        Args:
            market_data: Dictionary of symbol -> OHLCV data
            current_allocations: Current portfolio allocations
            target_allocations: Target portfolio allocations

        Returns:
            List of rebalancing signals
        """
        signals = []

        # Calculate position sizes based on portfolio value
        portfolio_value = 100000  # Default for signal generation

        for symbol, target_allocation in target_allocations.items():
            current_allocation = current_allocations.get(symbol, 0.0)

            # Check if we need to adjust position
            allocation_diff = target_allocation - current_allocation

            if abs(allocation_diff) > 0.05:  # 5% threshold for rebalancing
                current_price = market_data[symbol]["Close"].iloc[-1]
                target_value = portfolio_value * target_allocation
                current_value = portfolio_value * current_allocation

                if allocation_diff > 0:
                    # Need to buy more
                    signal = StrategySignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.8,
                        price=current_price,
                        quantity=int(target_value / current_price),
                        strategy_name=self.name,
                        metadata={
                            "rebalancing": True,
                            "target_allocation": target_allocation,
                            "current_allocation": current_allocation,
                            "allocation_change": allocation_diff,
                        },
                    )
                    signals.append(signal)
                else:
                    # Need to sell some
                    signal = StrategySignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.8,
                        price=current_price,
                        quantity=int(
                            abs(allocation_diff) * portfolio_value / current_price
                        ),
                        strategy_name=self.name,
                        metadata={
                            "rebalancing": True,
                            "target_allocation": target_allocation,
                            "current_allocation": current_allocation,
                            "allocation_change": allocation_diff,
                        },
                    )
                    signals.append(signal)

        return signals

    @abstractmethod
    def generate_signals(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> List[StrategySignal]:
        """
        Generate trading signals for ETF rotation.

        This method should be implemented by subclasses to define specific
        rotation logic (e.g., dual momentum, sector rotation, etc.).

        Args:
            market_data: Dictionary mapping symbol -> OHLCV DataFrame

        Returns:
            List of StrategySignal objects
        """
        pass

    def get_rotation_summary(self) -> Dict[str, Any]:
        """Get ETF rotation specific summary information."""
        summary = self.get_performance_summary()

        # Add rotation-specific metrics
        summary.update(
            {
                "etf_categories": self.etf_categories,
                "current_allocations": self.current_allocations.copy(),
                "momentum_rankings": self.momentum_rankings.copy(),
                "relative_strength_scores": self.relative_strength_scores.copy(),
                "last_rebalance_date": self.last_rebalance_date,
                "rotation_config": {
                    "momentum_lookback": self.config["momentum_lookback"],
                    "rebalance_frequency": self.config["rebalance_frequency"],
                    "max_positions": self.config["max_positions"],
                    "momentum_ranking_method": self.config["momentum_ranking_method"],
                },
            }
        )

        return summary
