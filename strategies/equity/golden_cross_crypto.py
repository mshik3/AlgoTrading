"""
Golden Cross strategy implementation for crypto assets.
"""

from typing import Dict, List, Optional
import pandas as pd
from strategies.base import BaseStrategy, SignalType, StrategySignal
from indicators.technical import TechnicalIndicators
from utils.asset_categorization import is_crypto_symbol


class GoldenCrossCryptoStrategy(BaseStrategy):
    """Golden Cross strategy specifically for crypto assets."""

    def __init__(self, **kwargs):
        """Initialize the strategy with crypto-specific parameters."""
        # Initialize with crypto-specific defaults
        kwargs.setdefault("fast_ma_period", 20)  # Shorter for crypto volatility
        kwargs.setdefault("slow_ma_period", 100)  # Shorter for crypto
        kwargs.setdefault("min_trend_strength", 0.05)  # Higher for crypto
        kwargs.setdefault("max_position_size", 0.20)  # Smaller for risk

        # Get crypto symbols
        symbols = [
            "BTCUSD",  # Bitcoin
            "ETHUSD",  # Ethereum
            "SOLUSD",  # Solana
            "DOTUSD",  # Polkadot
            "LINKUSD",  # Chainlink
            "LTCUSD",  # Litecoin
            "BCHUSD",  # Bitcoin Cash
            "XRPUSD",  # Ripple
            "MATICUSD",  # Polygon
            "AVAXUSD",  # Avalanche
        ]

        super().__init__(name="GoldenCrossCrypto", symbols=symbols, **kwargs)

        # Strategy-specific attributes
        self.fast_ma_period = kwargs.get("fast_ma_period", 20)
        self.slow_ma_period = kwargs.get("slow_ma_period", 100)
        self.min_trend_strength = kwargs.get("min_trend_strength", 0.05)
        self.max_position_size = kwargs.get("max_position_size", 0.20)
        self.crossover_states = {}  # Track crossover states

        # Strategy configuration
        self.config = {
            "strategy_type": "crypto_golden_cross",
            "fast_ma_period": self.fast_ma_period,
            "slow_ma_period": self.slow_ma_period,
            "min_trend_strength": self.min_trend_strength,
            "max_position_size": self.max_position_size,
            "asset_class": "crypto",
        }

    def generate_signals(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> List[StrategySignal]:
        """
        Generate trading signals for all crypto symbols.

        Args:
            market_data: Dictionary mapping symbol -> OHLCV DataFrame

        Returns:
            List of trading signals
        """
        signals = []

        for symbol in self.symbols:
            if symbol not in market_data:
                continue

            data = market_data[symbol]
            if len(data) < self.slow_ma_period:
                continue

            # Check for entry signal
            entry_signal = self.should_enter_position(symbol, data)
            if entry_signal:
                signals.append(entry_signal)

            # Check for exit signal
            exit_signal = self.should_exit_position(symbol, data)
            if exit_signal:
                signals.append(exit_signal)

        return signals

    def should_enter_position(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[StrategySignal]:
        """
        Check if we should enter a position based on golden cross.

        Args:
            symbol: The crypto symbol
            data: Market data DataFrame

        Returns:
            StrategySignal if conditions met, None otherwise
        """
        if len(data) < self.slow_ma_period:
            return None

        # Add technical indicators
        indicators = TechnicalIndicators(data)
        indicators.add_sma(self.fast_ma_period, "Close")
        indicators.add_sma(self.slow_ma_period, "Close")
        enhanced_data = indicators.get_data()

        # Get latest values
        latest = enhanced_data.iloc[-1]
        sma_fast = latest.get(f"SMA_{self.fast_ma_period}", 0)
        sma_slow = latest.get(f"SMA_{self.slow_ma_period}", 0)

        # Previous values for crossover detection
        prev = enhanced_data.iloc[-2]
        prev_sma_fast = prev.get(f"SMA_{self.fast_ma_period}", 0)
        prev_sma_slow = prev.get(f"SMA_{self.slow_ma_period}", 0)

        # Check for golden cross (fast MA crosses above slow MA)
        if prev_sma_fast <= prev_sma_slow and sma_fast > sma_slow:
            # Calculate trend strength
            trend_strength = (sma_fast - sma_slow) / sma_slow

            if trend_strength > self.min_trend_strength:
                self.crossover_states[symbol] = "golden"
                return StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=latest["Close"],
                    confidence=min(trend_strength, 1.0),
                    strategy_name=self.name,
                    metadata={
                        "trend_strength": trend_strength,
                        "fast_ma": sma_fast,
                        "slow_ma": sma_slow,
                    },
                )

        return None

    def should_exit_position(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[StrategySignal]:
        """
        Check if we should exit a position based on death cross.

        Args:
            symbol: The crypto symbol
            data: Market data DataFrame

        Returns:
            StrategySignal if conditions met, None otherwise
        """
        if len(data) < self.slow_ma_period:
            return None

        # Add technical indicators
        indicators = TechnicalIndicators(data)
        indicators.add_sma(self.fast_ma_period, "Close")
        indicators.add_sma(self.slow_ma_period, "Close")
        enhanced_data = indicators.get_data()

        # Get latest values
        latest = enhanced_data.iloc[-1]
        sma_fast = latest.get(f"SMA_{self.fast_ma_period}", 0)
        sma_slow = latest.get(f"SMA_{self.slow_ma_period}", 0)

        # Previous values for crossover detection
        prev = enhanced_data.iloc[-2]
        prev_sma_fast = prev.get(f"SMA_{self.fast_ma_period}", 0)
        prev_sma_slow = prev.get(f"SMA_{self.slow_ma_period}", 0)

        # Check for death cross (fast MA crosses below slow MA)
        if prev_sma_fast >= prev_sma_slow and sma_fast < sma_slow:
            # Calculate trend strength
            trend_strength = abs((sma_fast - sma_slow) / sma_slow)

            if trend_strength > self.min_trend_strength:
                self.crossover_states[symbol] = "death"
                return StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=latest["Close"],
                    confidence=min(trend_strength, 1.0),
                    strategy_name=self.name,
                    metadata={
                        "trend_strength": trend_strength,
                        "fast_ma": sma_fast,
                        "slow_ma": sma_slow,
                    },
                )

        return None
