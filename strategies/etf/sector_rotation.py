"""
Sector ETF Rotation Strategy Implementation.

This strategy implements sector rotation using relative strength and momentum analysis
to identify and invest in the strongest performing sectors at any given time.

Strategy Logic:
1. Calculate relative strength for each sector ETF vs benchmark
2. Rank sectors by momentum and relative strength
3. Allocate to top N sectors with equal weights
4. Rebalance monthly to capture sector rotation opportunities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from .rotation_base import BaseETFRotationStrategy
from ..base import StrategySignal, SignalType

logger = logging.getLogger(__name__)


class SectorRotationStrategy(BaseETFRotationStrategy):
    """
    Sector Rotation Strategy for ETF trading.

    Strategy Logic:
    1. Relative Strength: Compare sector performance vs benchmark (SPY)
    2. Momentum Ranking: Rank sectors by recent performance
    3. Multi-Sector Allocation: Invest in top N sectors with equal weights
    4. Monthly Rebalancing: Capture changing sector leadership

    This approach aims to:
    - Outperform the market by being in leading sectors
    - Reduce risk through sector diversification
    - Capture economic cycle benefits
    - Provide systematic, rules-based sector selection
    """

    def __init__(self, etf_universe: Dict[str, List[str]] = None, **config):
        """
        Initialize Sector Rotation Strategy.

        Args:
            etf_universe: Dictionary mapping category -> list of ETF symbols
            **config: Strategy configuration parameters
        """
        # Default sector ETF universe (SPDR sector ETFs)
        if etf_universe is None:
            etf_universe = {
                "Technology": ["XLK", "VGT", "SMH"],
                "Financials": ["XLF", "VFH", "KBE"],
                "Healthcare": ["XLV", "VHT", "IHI"],
                "Consumer_Discretionary": ["XLY", "VCR", "XRT"],
                "Consumer_Staples": ["XLP", "VDC", "XLP"],
                "Industrials": ["XLI", "VIS", "XAR"],
                "Energy": ["XLE", "VDE", "XOP"],
                "Materials": ["XLB", "VAW", "XME"],
                "Real_Estate": ["XLRE", "VNQ", "IYR"],
                "Utilities": ["XLU", "VPU", "XLU"],
                "Communications": ["XLC", "VOX", "XLC"],
            }

        # Sector rotation specific configuration
        sector_config = {
            "momentum_lookback": 63,  # 3 months for sector momentum
            "relative_strength_lookback": 252,  # 1 year for relative strength
            "max_positions": 4,  # Hold top 4 sectors
            "rebalance_frequency": 21,  # Monthly rebalancing
            "min_relative_strength": -0.05,  # Minimum relative strength threshold
            "momentum_ranking_method": "returns",  # Use simple returns
            "use_volatility_adjustment": True,  # Adjust for sector volatility
            "benchmark_symbol": "SPY",  # S&P 500 as benchmark
            "equal_weight": True,  # Equal weight allocation
            "sector_momentum_weight": 0.6,  # Weight for momentum in ranking
            "relative_strength_weight": 0.4,  # Weight for relative strength in ranking
            "confidence_threshold": 0.6,  # Lower threshold for sector rotation
        }

        # Merge with provided config
        merged_config = {**sector_config, **config}

        super().__init__(
            name="Sector ETF Rotation", etf_universe=etf_universe, **merged_config
        )

        # Track sector rotation state
        self.sector_rankings = {}
        self.sector_scores = {}
        self.benchmark_data = None

    def calculate_sector_score(
        self,
        sector_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        sector_symbol: str,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive sector score combining momentum and relative strength.

        Args:
            sector_data: Sector ETF OHLCV data
            benchmark_data: Benchmark OHLCV data
            sector_symbol: Sector symbol for tracking

        Returns:
            Dictionary with momentum, relative strength, and combined scores
        """
        scores = {}

        try:
            # Calculate momentum score
            momentum = self.calculate_momentum(
                sector_data, lookback=self.config["momentum_lookback"]
            )
            scores["momentum"] = momentum if not np.isnan(momentum) else 0.0

            # Calculate relative strength
            relative_strength = self.calculate_relative_strength(
                sector_data,
                benchmark_data,
                lookback=self.config["relative_strength_lookback"],
            )
            scores["relative_strength"] = (
                relative_strength if not np.isnan(relative_strength) else 0.0
            )

            # Calculate volatility adjustment if enabled
            if self.config["use_volatility_adjustment"]:
                sector_vol = sector_data["Close"].pct_change().std() * np.sqrt(252)
                benchmark_vol = benchmark_data["Close"].pct_change().std() * np.sqrt(
                    252
                )
                vol_ratio = benchmark_vol / sector_vol if sector_vol > 0 else 1.0
                scores["volatility_adjustment"] = vol_ratio
            else:
                scores["volatility_adjustment"] = 1.0

            # Calculate combined score
            momentum_weight = self.config["sector_momentum_weight"]
            rs_weight = self.config["relative_strength_weight"]

            combined_score = (
                scores["momentum"] * momentum_weight
                + scores["relative_strength"] * rs_weight
            ) * scores["volatility_adjustment"]

            scores["combined_score"] = combined_score

        except Exception as e:
            logger.warning(
                f"Error calculating sector score for {sector_symbol}: {str(e)}"
            )
            scores = {
                "momentum": 0.0,
                "relative_strength": 0.0,
                "volatility_adjustment": 1.0,
                "combined_score": 0.0,
            }

        return scores

    def rank_sectors(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Rank sectors by combined momentum and relative strength scores.

        Args:
            market_data: Dictionary of symbol -> OHLCV data

        Returns:
            List of (symbol, combined_score, detailed_scores) tuples, sorted by score
        """
        rankings = []
        benchmark_symbol = self.config["benchmark_symbol"]

        # Get benchmark data
        if benchmark_symbol not in market_data:
            logger.warning(f"Benchmark {benchmark_symbol} not found in market data")
            return rankings

        benchmark_data = market_data[benchmark_symbol]

        # Calculate scores for each sector
        for category, symbols in self.etf_universe.items():
            for symbol in symbols:
                if symbol not in market_data or market_data[symbol].empty:
                    continue

                # Calculate sector scores
                scores = self.calculate_sector_score(
                    market_data[symbol], benchmark_data, symbol
                )

                # Check minimum relative strength threshold
                if scores["relative_strength"] >= self.config["min_relative_strength"]:
                    rankings.append((symbol, scores["combined_score"], scores))
                    self.sector_scores[symbol] = scores

        # Sort by combined score (highest first)
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def should_enter_position(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[StrategySignal]:
        """
        Determine if we should enter a position in the given sector.

        Args:
            symbol: Sector ETF symbol
            data: OHLCV data

        Returns:
            StrategySignal if should enter, None otherwise
        """
        # Skip if we already have a position
        if symbol in self.positions:
            return None

        # Check if symbol is in top-ranked sectors
        if symbol not in self.sector_scores:
            return None

        scores = self.sector_scores[symbol]

        # Calculate confidence based on combined score
        confidence = min(0.6 + (scores["combined_score"] * 0.4), 1.0)

        if confidence < self.config["confidence_threshold"]:
            return None

        current_price = data["Close"].iloc[-1]

        # Calculate stop loss and take profit
        stop_loss = current_price * (1 - self.config["stop_loss_pct"])
        take_profit = current_price * (1 + self.config["take_profit_pct"])

        signal = StrategySignal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=confidence,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self.name,
            metadata={
                "sector_scores": scores,
                "entry_reason": "sector_rotation_qualified",
                "momentum_lookback": self.config["momentum_lookback"],
                "relative_strength_lookback": self.config["relative_strength_lookback"],
            },
        )

        logger.info(
            f"Sector Rotation BUY signal for {symbol}: combined_score={scores['combined_score']:.3f}, "
            f"confidence={confidence:.3f}"
        )

        return signal

    def should_exit_position(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[StrategySignal]:
        """
        Determine if we should exit existing sector position.

        Args:
            symbol: Sector ETF symbol
            data: OHLCV data

        Returns:
            StrategySignal if should exit, None otherwise
        """
        if symbol not in self.positions:
            return None

        # Check if symbol is no longer in top-ranked sectors
        if symbol not in self.sector_scores:
            current_price = data["Close"].iloc[-1]
            position = self.positions[symbol]
            entry_price = position["entry_price"]
            profit_loss_pct = (current_price - entry_price) / entry_price * 100

            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.CLOSE_LONG,
                confidence=0.8,
                price=current_price,
                strategy_name=self.name,
                metadata={
                    "exit_reason": "sector_no_longer_ranked",
                    "entry_price": entry_price,
                    "profit_loss_pct": profit_loss_pct,
                },
            )

            logger.info(
                f"Sector Rotation SELL signal for {symbol}: no longer ranked, "
                f"P&L={profit_loss_pct:.2f}%"
            )

            return signal

        # Check if relative strength has deteriorated significantly
        scores = self.sector_scores[symbol]
        if scores["relative_strength"] < self.config["min_relative_strength"]:
            current_price = data["Close"].iloc[-1]
            position = self.positions[symbol]
            entry_price = position["entry_price"]
            profit_loss_pct = (current_price - entry_price) / entry_price * 100

            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.CLOSE_LONG,
                confidence=0.8,
                price=current_price,
                strategy_name=self.name,
                metadata={
                    "exit_reason": "relative_strength_breakdown",
                    "relative_strength": scores["relative_strength"],
                    "entry_price": entry_price,
                    "profit_loss_pct": profit_loss_pct,
                },
            )

            logger.info(
                f"Sector Rotation SELL signal for {symbol}: relative_strength={scores['relative_strength']:.3f}, "
                f"P&L={profit_loss_pct:.2f}%"
            )

            return signal

        return None

    def generate_signals(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> List[StrategySignal]:
        """
        Generate sector rotation trading signals.

        Args:
            market_data: Dictionary mapping symbol -> OHLCV DataFrame

        Returns:
            List of StrategySignal objects
        """
        signals = []
        current_date = datetime.now()

        # Check if it's time to rebalance
        if not self.should_rebalance(current_date):
            # Just check for exit signals on current positions
            for symbol in list(self.positions.keys()):
                if symbol in market_data:
                    exit_signal = self.should_exit_position(symbol, market_data[symbol])
                    if exit_signal and self.validate_signal(exit_signal):
                        signals.append(exit_signal)
                        self.add_signal_to_history(exit_signal)
            return signals

        # Full rebalancing logic
        logger.info("Performing sector rotation rebalancing...")

        # Step 1: Rank sectors by combined momentum and relative strength
        sector_rankings = self.rank_sectors(market_data)

        if not sector_rankings:
            logger.warning("No sectors meet ranking criteria")
            return signals

        # Step 2: Select top N sectors
        max_positions = self.config["max_positions"]
        top_sectors = sector_rankings[:max_positions]

        logger.info(f"Top {len(top_sectors)} sectors: {[s[0] for s in top_sectors]}")

        # Step 3: Calculate target allocations
        target_allocations = {}
        if self.config["equal_weight"]:
            allocation_per_sector = 1.0 / len(top_sectors)
            for symbol, _, _ in top_sectors:
                target_allocations[symbol] = allocation_per_sector

        # Step 4: Generate rebalancing signals
        current_allocations = {}
        for symbol in self.positions:
            current_allocations[symbol] = (
                1.0 / len(self.positions) if self.positions else 0.0
            )

        rebalancing_signals = self.generate_rebalancing_signals(
            market_data, current_allocations, target_allocations
        )

        for signal in rebalancing_signals:
            if self.validate_signal(signal):
                signals.append(signal)
                self.add_signal_to_history(signal)

        # Step 5: Exit positions in sectors no longer ranked
        current_symbols = set(self.positions.keys())
        target_symbols = set(target_allocations.keys())
        symbols_to_exit = current_symbols - target_symbols

        for symbol in symbols_to_exit:
            if symbol in market_data:
                exit_signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    confidence=0.8,
                    price=market_data[symbol]["Close"].iloc[-1],
                    strategy_name=self.name,
                    metadata={
                        "exit_reason": "sector_rotation_rebalancing",
                        "new_target_sectors": list(target_symbols),
                    },
                )
                signals.append(exit_signal)
                self.add_signal_to_history(exit_signal)

        # Update rebalancing date and rankings
        self.last_rebalance_date = current_date
        self.sector_rankings = {symbol: score for symbol, score, _ in sector_rankings}

        return signals

    def get_sector_rotation_summary(self) -> Dict[str, Any]:
        """Get sector rotation specific summary information."""
        summary = self.get_rotation_summary()

        # Add sector rotation specific metrics
        summary.update(
            {
                "sector_rankings": self.sector_rankings.copy(),
                "sector_scores": self.sector_scores.copy(),
                "sector_rotation_config": {
                    "momentum_lookback": self.config["momentum_lookback"],
                    "relative_strength_lookback": self.config[
                        "relative_strength_lookback"
                    ],
                    "max_positions": self.config["max_positions"],
                    "benchmark_symbol": self.config["benchmark_symbol"],
                    "sector_momentum_weight": self.config["sector_momentum_weight"],
                    "relative_strength_weight": self.config["relative_strength_weight"],
                },
            }
        )

        return summary
