"""
Dual Momentum ETF Strategy Implementation.

This strategy implements Gary Antonacci's proven dual momentum approach:
1. Absolute Momentum: Compare asset returns vs risk-free rate
2. Relative Momentum: Choose best performing asset among qualified candidates

Based on research showing this approach has historically outperformed buy-and-hold
with lower drawdowns and better risk-adjusted returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from .rotation_base import BaseETFRotationStrategy
from ..base import StrategySignal, SignalType

logger = logging.getLogger(__name__)


class DualMomentumStrategy(BaseETFRotationStrategy):
    """
    Dual Momentum Strategy for ETF rotation.

    Strategy Logic:
    1. Absolute Momentum: Only invest in assets with positive momentum vs risk-free rate
    2. Relative Momentum: Among qualified assets, choose the one with highest momentum
    3. Rebalance monthly to capture changing momentum patterns
    4. Use defensive positioning (bonds/cash) when no assets meet criteria

    This approach has historically provided:
    - Higher returns than buy-and-hold
    - Lower maximum drawdowns
    - Better risk-adjusted returns (Sharpe ratio)
    - Protection during major market downturns
    """

    def __init__(self, etf_universe: Dict[str, List[str]] = None, **config):
        """
        Initialize Dual Momentum Strategy.

        Args:
            etf_universe: Dictionary mapping category -> list of ETF symbols
            **config: Strategy configuration parameters
        """
        # Default ETF universe for dual momentum
        if etf_universe is None:
            etf_universe = {
                "US_Equities": ["SPY", "QQQ", "VTI", "IWM"],
                "International": ["EFA", "EEM", "VEA", "VWO"],
                "Bonds": ["TLT", "AGG", "BND", "LQD"],
                "Real_Estate": ["VNQ", "IYR", "SCHH"],
                "Commodities": ["GLD", "SLV", "USO", "DBA"],
                "Cash_Equivalents": ["SHY", "BIL", "SHV"],  # Short-term Treasuries
            }

        # Dual momentum specific configuration
        dual_momentum_config = {
            "absolute_momentum_lookback": 252,  # 1 year for absolute momentum
            "relative_momentum_lookback": 252,  # 1 year for relative momentum
            "risk_free_rate": 0.02,  # 2% annual risk-free rate
            "min_absolute_momentum": 0.0,  # Minimum absolute momentum threshold
            "defensive_asset": "SHY",  # Default to short-term Treasuries when defensive
            "max_positions": 1,  # Dual momentum typically holds 1 asset at a time
            "rebalance_frequency": 21,  # Monthly rebalancing
            "momentum_ranking_method": "returns",  # Use simple returns for momentum
            "use_volatility_adjustment": False,  # Don't adjust for volatility
            "confidence_threshold": 0.7,  # Higher confidence for dual momentum
        }

        # Merge with provided config
        merged_config = {**dual_momentum_config, **config}

        super().__init__(
            name="Dual Momentum ETF Rotation",
            etf_universe=etf_universe,
            **merged_config,
        )

        # Track dual momentum state
        self.current_asset = None
        self.defensive_mode = False
        self.absolute_momentum_scores = {}
        self.relative_momentum_scores = {}

    def calculate_absolute_momentum(
        self, data: pd.DataFrame, lookback: Optional[int] = None
    ) -> float:
        """
        Calculate absolute momentum (return vs risk-free rate).

        Args:
            data: OHLCV data
            lookback: Lookback period

        Returns:
            Absolute momentum score (excess return over risk-free rate)
        """
        if lookback is None:
            lookback = self.config["absolute_momentum_lookback"]

        if len(data) < lookback:
            return np.nan

        # Calculate total return over lookback period
        total_return = data["Close"].iloc[-1] / data["Close"].iloc[-lookback] - 1

        # Calculate risk-free return over same period
        risk_free_return = (1 + self.config["risk_free_rate"]) ** (lookback / 252) - 1

        # Absolute momentum = excess return over risk-free rate
        absolute_momentum = total_return - risk_free_return

        return absolute_momentum

    def calculate_relative_momentum(
        self, market_data: Dict[str, pd.DataFrame], qualified_assets: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Calculate relative momentum among qualified assets.

        Args:
            market_data: Dictionary of symbol -> OHLCV data
            qualified_assets: List of assets that passed absolute momentum test

        Returns:
            List of (symbol, relative_momentum) tuples, sorted by momentum
        """
        relative_momentums = []

        for symbol in qualified_assets:
            if symbol not in market_data or market_data[symbol].empty:
                continue

            try:
                # Calculate total return over relative momentum period
                lookback = self.config["relative_momentum_lookback"]
                if len(market_data[symbol]) < lookback:
                    continue

                total_return = (
                    market_data[symbol]["Close"].iloc[-1]
                    / market_data[symbol]["Close"].iloc[-lookback]
                    - 1
                )

                relative_momentums.append((symbol, total_return))

            except Exception as e:
                logger.warning(
                    f"Error calculating relative momentum for {symbol}: {str(e)}"
                )
                continue

        # Sort by relative momentum (highest first)
        relative_momentums.sort(key=lambda x: x[1], reverse=True)
        return relative_momentums

    def get_qualified_assets(self, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Get assets that pass the absolute momentum test.

        Args:
            market_data: Dictionary of symbol -> OHLCV data

        Returns:
            List of symbols that have positive absolute momentum
        """
        qualified_assets = []

        for symbol in self.symbols:
            if symbol not in market_data or market_data[symbol].empty:
                continue

            try:
                absolute_momentum = self.calculate_absolute_momentum(
                    market_data[symbol]
                )

                if (
                    not np.isnan(absolute_momentum)
                    and absolute_momentum >= self.config["min_absolute_momentum"]
                ):
                    qualified_assets.append(symbol)
                    self.absolute_momentum_scores[symbol] = absolute_momentum
                else:
                    self.absolute_momentum_scores[symbol] = absolute_momentum

            except Exception as e:
                logger.warning(
                    f"Error calculating absolute momentum for {symbol}: {str(e)}"
                )
                continue

        return qualified_assets

    def should_enter_position(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[StrategySignal]:
        """
        Determine if we should enter a position in the given symbol.

        Args:
            symbol: ETF symbol
            data: OHLCV data

        Returns:
            StrategySignal if should enter, None otherwise
        """
        # Skip if we already have a position
        if symbol in self.positions:
            return None

        # Check absolute momentum
        absolute_momentum = self.calculate_absolute_momentum(data)

        if (
            np.isnan(absolute_momentum)
            or absolute_momentum < self.config["min_absolute_momentum"]
        ):
            return None

        # Calculate confidence based on momentum strength
        confidence = min(0.7 + (absolute_momentum * 0.3), 1.0)

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
                "absolute_momentum": absolute_momentum,
                "entry_reason": "dual_momentum_qualified",
                "momentum_lookback": self.config["absolute_momentum_lookback"],
            },
        )

        logger.info(
            f"Dual Momentum BUY signal for {symbol}: absolute_momentum={absolute_momentum:.3f}, "
            f"confidence={confidence:.3f}"
        )

        return signal

    def should_exit_position(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[StrategySignal]:
        """
        Determine if we should exit existing position.

        Args:
            symbol: ETF symbol
            data: OHLCV data

        Returns:
            StrategySignal if should exit, None otherwise
        """
        if symbol not in self.positions:
            return None

        # Check if absolute momentum has turned negative
        absolute_momentum = self.calculate_absolute_momentum(data)

        if (
            not np.isnan(absolute_momentum)
            and absolute_momentum < self.config["min_absolute_momentum"]
        ):
            current_price = data["Close"].iloc[-1]
            position = self.positions[symbol]
            entry_price = position["entry_price"]
            profit_loss_pct = (current_price - entry_price) / entry_price * 100

            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.CLOSE_LONG,
                confidence=0.9,  # High confidence for momentum breakdown
                price=current_price,
                strategy_name=self.name,
                metadata={
                    "exit_reason": "absolute_momentum_breakdown",
                    "absolute_momentum": absolute_momentum,
                    "entry_price": entry_price,
                    "profit_loss_pct": profit_loss_pct,
                },
            )

            logger.info(
                f"Dual Momentum SELL signal for {symbol}: absolute_momentum={absolute_momentum:.3f}, "
                f"P&L={profit_loss_pct:.2f}%"
            )

            return signal

        return None

    def generate_signals(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> List[StrategySignal]:
        """
        Generate dual momentum trading signals.

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
        logger.info("Performing dual momentum rebalancing...")

        # Step 1: Get assets that pass absolute momentum test
        qualified_assets = self.get_qualified_assets(market_data)

        if not qualified_assets:
            # No qualified assets - go defensive
            logger.info(
                "No assets pass absolute momentum test - entering defensive mode"
            )
            self.defensive_mode = True

            # Exit all current positions
            for symbol in list(self.positions.keys()):
                if symbol in market_data:
                    exit_signal = StrategySignal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_LONG,
                        confidence=0.9,
                        price=market_data[symbol]["Close"].iloc[-1],
                        strategy_name=self.name,
                        metadata={
                            "exit_reason": "defensive_mode_no_qualified_assets",
                            "qualified_assets_count": 0,
                        },
                    )
                    signals.append(exit_signal)
                    self.add_signal_to_history(exit_signal)

            # Optionally enter defensive position
            defensive_asset = self.config["defensive_asset"]
            if defensive_asset in market_data:
                entry_signal = self.should_enter_position(
                    defensive_asset, market_data[defensive_asset]
                )
                if entry_signal and self.validate_signal(entry_signal):
                    signals.append(entry_signal)
                    self.add_signal_to_history(entry_signal)

        else:
            # Step 2: Calculate relative momentum among qualified assets
            relative_momentums = self.calculate_relative_momentum(
                market_data, qualified_assets
            )

            if relative_momentums:
                # Step 3: Select the asset with highest relative momentum
                best_asset, best_momentum = relative_momentums[0]

                logger.info(
                    f"Best asset by relative momentum: {best_asset} ({best_momentum:.3f})"
                )

                # Exit positions in other assets
                for symbol in list(self.positions.keys()):
                    if symbol != best_asset:
                        if symbol in market_data:
                            exit_signal = StrategySignal(
                                symbol=symbol,
                                signal_type=SignalType.CLOSE_LONG,
                                confidence=0.8,
                                price=market_data[symbol]["Close"].iloc[-1],
                                strategy_name=self.name,
                                metadata={
                                    "exit_reason": "rebalancing_to_better_asset",
                                    "new_best_asset": best_asset,
                                    "best_momentum": best_momentum,
                                },
                            )
                            signals.append(exit_signal)
                            self.add_signal_to_history(exit_signal)

                # Enter position in best asset if not already holding
                if best_asset not in self.positions:
                    entry_signal = self.should_enter_position(
                        best_asset, market_data[best_asset]
                    )
                    if entry_signal and self.validate_signal(entry_signal):
                        signals.append(entry_signal)
                        self.add_signal_to_history(entry_signal)

                self.defensive_mode = False
                self.current_asset = best_asset

                # Store momentum scores for analysis
                self.relative_momentum_scores = dict(relative_momentums)

        # Update rebalancing date
        self.last_rebalance_date = current_date

        return signals

    def get_dual_momentum_summary(self) -> Dict[str, Any]:
        """Get dual momentum specific summary information."""
        summary = self.get_rotation_summary()

        # Add dual momentum specific metrics
        summary.update(
            {
                "current_asset": self.current_asset,
                "defensive_mode": self.defensive_mode,
                "absolute_momentum_scores": self.absolute_momentum_scores.copy(),
                "relative_momentum_scores": self.relative_momentum_scores.copy(),
                "dual_momentum_config": {
                    "absolute_momentum_lookback": self.config[
                        "absolute_momentum_lookback"
                    ],
                    "relative_momentum_lookback": self.config[
                        "relative_momentum_lookback"
                    ],
                    "risk_free_rate": self.config["risk_free_rate"],
                    "defensive_asset": self.config["defensive_asset"],
                },
            }
        )

        return summary
