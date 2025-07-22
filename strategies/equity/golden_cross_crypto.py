"""
Crypto-Optimized Golden Cross Strategy Implementation.

This strategy implements a Golden Cross approach specifically optimized for cryptocurrencies
that have limited historical data. It uses shorter moving averages (20/50 instead of 50/200)
to work with newer crypto coins that may only have 100-200 days of data.

Key Features:
- 20-day and 50-day Simple Moving Average tracking (vs 50/200 for traditional)
- Faster signal generation for crypto volatility
- Optimized for crypto trading patterns
- Works with newer cryptocurrencies (PEPE, SHIB, etc.)
- Reduced data requirements (100+ days vs 220+ days)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

from ..base import BaseStrategy, StrategySignal, SignalType
from indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class CryptoGoldenCrossStrategy(BaseStrategy):
    """
    Crypto-Optimized Golden Cross Strategy.

    Strategy Logic:
    - Buy when 20-day SMA crosses above 50-day SMA (Golden Cross)
    - Sell when 20-day SMA crosses below 50-day SMA (Death Cross)
    - Optimized for crypto volatility and shorter timeframes
    - Works with newer cryptocurrencies that have limited historical data
    - Faster signal generation than traditional Golden Cross
    """

    def __init__(self, symbols: List[str] = None, **config):
        """
        Initialize Crypto Golden Cross Strategy.

        Args:
            symbols: List of symbols to trade (defaults to crypto universe)
            **config: Strategy configuration parameters
        """
        # Use crypto symbols if none provided
        if symbols is None:
            from data.crypto_universe import CryptoUniverse

            crypto_universe = CryptoUniverse()
            symbols = [
                crypto.symbol for crypto in crypto_universe.cryptocurrencies
            ]

        # Default configuration for Crypto Golden Cross
        default_config = {
            "fast_ma_period": 20,  # Fast moving average period (vs 50 for traditional)
            "slow_ma_period": 50,  # Slow moving average period (vs 200 for traditional)
            "min_trend_strength": 0.01,  # Minimum 1% separation between MAs (vs 2% for traditional)
            "volume_confirmation": True,  # Require volume confirmation
            "volume_multiplier": 1.05,  # Volume should be 1.05x average (vs 1.1x for traditional)
            "max_position_size": 0.25,  # 25% max per crypto position (vs 30% for traditional)
            "min_days_between_signals": 3,  # Minimum days between signals (vs 5 for traditional)
            "trend_confirmation_days": 2,  # Days to confirm trend before entry (vs 3 for traditional)
            "take_profit_pct": 0.20,  # 20% take profit target (vs None for traditional)
            "stop_loss_pct": 0.10,  # 10% stop loss (vs None for traditional)
            "min_confidence": 0.6,  # Lower confidence threshold for crypto (vs 0.7 for traditional)
        }

        # Merge default config with provided config
        merged_config = {**default_config, **config}

        super().__init__(name="Crypto Golden Cross", symbols=symbols, **merged_config)

        # Track last signal dates to avoid whipsaws
        self.last_signal_dates = {}

        # Track crossover states to detect changes
        self.crossover_states = {}  # symbol -> 'golden' or 'death' or 'none'

    def get_minimum_data_requirements(self) -> int:
        """
        Get minimum number of days of data required for Crypto Golden Cross strategy.

        Returns:
            Minimum number of days required (100 for 50-day MA + buffer)
        """
        return 100

    def get_strategy_type(self) -> str:
        """
        Get the strategy type for categorization.

        Returns:
            Strategy type string
        """
        return "crypto_golden_cross"

    def is_crypto_strategy(self) -> bool:
        """
        Check if this strategy is optimized for crypto trading.

        Returns:
            True (this is a crypto-optimized strategy)
        """
        return True

    def generate_signals(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> List[StrategySignal]:
        """
        Generate trading signals for all symbols.

        Args:
            market_data: Dictionary mapping symbol -> OHLCV DataFrame

        Returns:
            List of trading signals
        """
        signals = []

        for symbol in self.symbols:
            if symbol not in market_data or market_data[symbol].empty:
                logger.warning(f"No data available for {symbol}")
                continue

            try:
                # Check for entry signal
                entry_signal = self.should_enter_position(symbol, market_data[symbol])
                if entry_signal and self.validate_signal(entry_signal):
                    signals.append(entry_signal)
                    self.add_signal_to_history(entry_signal)

                # Check for exit signal if we have a position
                if symbol in self.positions:
                    exit_signal = self.should_exit_position(symbol, market_data[symbol])
                    if exit_signal and self.validate_signal(exit_signal):
                        signals.append(exit_signal)
                        self.add_signal_to_history(exit_signal)

            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {str(e)}")
                continue

        return signals

    def should_enter_position(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[StrategySignal]:
        """
        Determine if we should enter a position based on Crypto Golden Cross.

        Args:
            symbol: Crypto symbol
            data: OHLCV data

        Returns:
            StrategySignal if should enter, None otherwise
        """
        # Skip if we already have a position
        if symbol in self.positions:
            return None

        # Need at least 70 days of data for 50-day MA + buffer
        if len(data) < 70:
            logger.debug(
                f"{symbol}: Insufficient data for Crypto Golden Cross (need 70+ days)"
            )
            return None

        try:
            # Calculate technical indicators
            indicators = TechnicalIndicators(data.copy())
            indicators.add_sma(self.config["fast_ma_period"])
            indicators.add_sma(self.config["slow_ma_period"])
            if self.config["volume_confirmation"]:
                indicators.add_volume_sma(20)

            analysis_data = indicators.get_data()

            # Get latest values and recent history for crossover detection
            latest = analysis_data.iloc[-1]
            recent = analysis_data.tail(10)  # Last 10 days for crossover detection

            # Check for required indicators
            fast_ma_col = f"SMA_{self.config['fast_ma_period']}"
            slow_ma_col = f"SMA_{self.config['slow_ma_period']}"

            if fast_ma_col not in latest.index or slow_ma_col not in latest.index:
                logger.warning(f"Missing required MAs for {symbol}")
                return None

            fast_ma_current = latest[fast_ma_col]
            slow_ma_current = latest[slow_ma_col]
            close_price = latest["Close"]

            # Skip if MAs are not valid
            if pd.isna(fast_ma_current) or pd.isna(slow_ma_current):
                return None

            # Check if we recently sent a signal to avoid whipsaws
            if symbol in self.last_signal_dates:
                days_since_last = (datetime.now() - self.last_signal_dates[symbol]).days
                if days_since_last < self.config["min_days_between_signals"]:
                    return None

            # Detect Golden Cross (20-day MA crosses above 50-day MA)
            golden_cross_detected = self._detect_golden_cross(
                recent, fast_ma_col, slow_ma_col
            )

            if not golden_cross_detected:
                return None

            # Additional confirmation criteria
            conditions_met = []
            confidence_factors = []

            # 1. Golden Cross detected
            conditions_met.append("GOLDEN_CROSS")
            confidence_factors.append(0.4)

            # 2. Trend strength - MAs should be reasonably separated
            ma_separation = abs(fast_ma_current - slow_ma_current) / slow_ma_current
            if ma_separation >= self.config["min_trend_strength"]:
                conditions_met.append("STRONG_TREND_SEPARATION")
                confidence_factors.append(0.2)

            # 3. Price above both MAs (confirming uptrend)
            if close_price > fast_ma_current and close_price > slow_ma_current:
                conditions_met.append("PRICE_ABOVE_MAS")
                confidence_factors.append(0.2)

            # 4. Volume confirmation (if enabled)
            if self.config["volume_confirmation"]:
                volume = latest["Volume"]
                volume_sma = latest.get("Volume_SMA_20")
                if (
                    pd.notna(volume_sma)
                    and volume > volume_sma * self.config["volume_multiplier"]
                ):
                    conditions_met.append("VOLUME_CONFIRMATION")
                    confidence_factors.append(0.2)

            # Calculate confidence
            confidence = min(sum(confidence_factors), 1.0)

            # Must have at least the golden cross + one additional confirmation
            if len(conditions_met) >= 2 and confidence >= self.config["min_confidence"]:
                # Update tracking
                self.last_signal_dates[symbol] = datetime.now()
                self.crossover_states[symbol] = "golden"

                # Calculate stop loss and take profit
                stop_loss = (
                    close_price * (1 - self.config["stop_loss_pct"])
                    if self.config["stop_loss_pct"]
                    else None
                )
                take_profit = (
                    close_price * (1 + self.config["take_profit_pct"])
                    if self.config["take_profit_pct"]
                    else None
                )

                # Create signal
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    price=close_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy_name=self.name,
                    metadata={
                        "conditions_met": conditions_met,
                        "fast_ma": fast_ma_current,
                        "slow_ma": slow_ma_current,
                        "ma_separation_pct": ma_separation * 100,
                        "entry_reason": "crypto_golden_cross_bullish",
                        "crossover_type": "golden",
                        "strategy_type": "crypto_optimized",
                    },
                )

                logger.info(
                    f"Crypto Golden Cross BUY signal for {symbol}: confidence={confidence:.3f}, "
                    f"20MA={fast_ma_current:.2f}, 50MA={slow_ma_current:.2f}"
                )
                return signal

        except Exception as e:
            logger.error(
                f"Error analyzing Crypto Golden Cross entry for {symbol}: {str(e)}"
            )
            return None

        return None

    def should_exit_position(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[StrategySignal]:
        """
        Determine if we should exit existing position based on Death Cross.

        Args:
            symbol: Crypto symbol
            data: OHLCV data

        Returns:
            StrategySignal if should exit, None otherwise
        """
        if symbol not in self.positions:
            return None

        try:
            # Calculate technical indicators
            indicators = TechnicalIndicators(data.copy())
            indicators.add_sma(self.config["fast_ma_period"])
            indicators.add_sma(self.config["slow_ma_period"])

            analysis_data = indicators.get_data()

            # Get recent history for crossover detection
            recent = analysis_data.tail(10)
            latest = analysis_data.iloc[-1]
            current_price = latest["Close"]

            fast_ma_col = f"SMA_{self.config['fast_ma_period']}"
            slow_ma_col = f"SMA_{self.config['slow_ma_period']}"

            # Detect Death Cross (20-day MA crosses below 50-day MA)
            death_cross_detected = self._detect_death_cross(
                recent, fast_ma_col, slow_ma_col
            )

            if death_cross_detected:
                # Update tracking
                self.crossover_states[symbol] = "death"

                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=0.8,
                    price=current_price,
                    strategy_name=self.name,
                    metadata={
                        "exit_reason": "crypto_death_cross_bearish",
                        "crossover_type": "death",
                        "strategy_type": "crypto_optimized",
                    },
                )

                logger.info(f"Crypto Death Cross SELL signal for {symbol}")
                return signal

        except Exception as e:
            logger.error(
                f"Error analyzing Crypto Death Cross exit for {symbol}: {str(e)}"
            )
            return None

        return None

    def _detect_golden_cross(
        self, recent_data: pd.DataFrame, fast_ma_col: str, slow_ma_col: str
    ) -> bool:
        """
        Detect if a Golden Cross (20 MA crossing above 50 MA) occurred recently.

        Args:
            recent_data: Recent price data with MAs
            fast_ma_col: Fast MA column name
            slow_ma_col: Slow MA column name

        Returns:
            True if Golden Cross detected in recent data
        """
        if len(recent_data) < 2:
            return False

        try:
            # Get the last few days to detect crossover
            for i in range(
                1, min(self.config["trend_confirmation_days"] + 1, len(recent_data))
            ):
                current = recent_data.iloc[-1]
                previous = recent_data.iloc[-(i + 1)]

                current_fast = current[fast_ma_col]
                current_slow = current[slow_ma_col]
                prev_fast = previous[fast_ma_col]
                prev_slow = previous[slow_ma_col]

                # Check if all values are valid
                if (
                    pd.isna(current_fast)
                    or pd.isna(current_slow)
                    or pd.isna(prev_fast)
                    or pd.isna(prev_slow)
                ):
                    continue

                # Golden Cross: fast MA crosses from below to above slow MA
                if prev_fast <= prev_slow and current_fast > current_slow:
                    logger.debug(
                        f"Crypto Golden Cross detected: {prev_fast:.2f} <= {prev_slow:.2f} -> {current_fast:.2f} > {current_slow:.2f}"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Error detecting Crypto Golden Cross: {str(e)}")
            return False

    def _detect_death_cross(
        self, recent_data: pd.DataFrame, fast_ma_col: str, slow_ma_col: str
    ) -> bool:
        """
        Detect if a Death Cross (20 MA crossing below 50 MA) occurred recently.

        Args:
            recent_data: Recent price data with MAs
            fast_ma_col: Fast MA column name
            slow_ma_col: Slow MA column name

        Returns:
            True if Death Cross detected in recent data
        """
        if len(recent_data) < 2:
            return False

        try:
            # Get the last few days to detect crossover
            for i in range(
                1, min(self.config["trend_confirmation_days"] + 1, len(recent_data))
            ):
                current = recent_data.iloc[-1]
                previous = recent_data.iloc[-(i + 1)]

                current_fast = current[fast_ma_col]
                current_slow = current[slow_ma_col]
                prev_fast = previous[fast_ma_col]
                prev_slow = previous[slow_ma_col]

                # Check if all values are valid
                if (
                    pd.isna(current_fast)
                    or pd.isna(current_slow)
                    or pd.isna(prev_fast)
                    or pd.isna(prev_slow)
                ):
                    continue

                # Death Cross: fast MA crosses from above to below slow MA
                if prev_fast >= prev_slow and current_fast < current_slow:
                    logger.debug(
                        f"Crypto Death Cross detected: {prev_fast:.2f} >= {prev_slow:.2f} -> {current_fast:.2f} < {current_slow:.2f}"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Error detecting Crypto Death Cross: {str(e)}")
            return False

    def get_strategy_summary(self) -> Dict:
        """Get crypto strategy-specific summary information."""
        summary = self.get_performance_summary()

        # Add crypto-specific metrics
        summary.update(
            {
                "strategy_type": "crypto_golden_cross",
                "fast_ma_period": self.config["fast_ma_period"],
                "slow_ma_period": self.config["slow_ma_period"],
                "min_data_requirements": self.get_minimum_data_requirements(),
                "crypto_optimizations": [
                    "20/50 moving averages (vs 50/200 traditional)",
                    "Faster signal generation",
                    "Lower data requirements (100+ days)",
                    "Crypto-specific risk management",
                    "Take profit and stop loss targets",
                ],
            }
        )

        return summary
