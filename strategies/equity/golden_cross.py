"""
Golden Cross Strategy Implementation.

This strategy implements the classic trend-following approach that buys when
the 50-day moving average crosses above the 200-day moving average (Golden Cross)
and sells when it crosses below (Death Cross).

Key Components:
- 50-day and 200-day Simple Moving Average tracking
- Crossover detection for entry/exit signals
- Focus on broad market ETFs for reliable trend following
- Long-term trend following with minimal trades per year
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

from ..base import BaseStrategy, StrategySignal, SignalType
from indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class GoldenCrossStrategy(BaseStrategy):
    """
    Golden Cross Strategy for equity trading.

    Strategy Logic:
    - Buy when 50-day SMA crosses above 200-day SMA (Golden Cross)
    - Sell when 50-day SMA crosses below 200-day SMA (Death Cross)
    - Focus on broad market ETFs (SPY, QQQ, VTI) for reliable trends
    - Typically generates 2-4 trades per year
    - Captures major bull runs while avoiding bear markets
    """

    def __init__(self, symbols: List[str] = None, **config):
        """
        Initialize Golden Cross Strategy.

        Args:
            symbols: List of symbols to trade (defaults to broad market ETFs)
            **config: Strategy configuration parameters
        """
        # Default to comprehensive 50-asset universe if no symbols provided
        if symbols is None:
            symbols = [
                # Major US ETFs (8)
                "SPY", "QQQ", "VTI", "IWM", "VEA", "VWO", "AGG", "TLT",
                
                # Sector ETFs (8)
                "XLF", "XLK", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB",
                
                # Major Tech Stocks (8)
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
                
                # Financial & Industrial (6)
                "JPM", "BAC", "WFC", "GS", "UNH", "JNJ",
                
                # International ETFs (6)
                "EFA", "EEM", "FXI", "EWJ", "EWG", "EWU",
                
                # Commodity ETFs (4)
                "GLD", "SLV", "USO", "DBA",
                
                # Crypto (10) - Only available in Alpaca API
                "BTCUSD", "ETHUSD", "DOTUSD", "LINKUSD", 
                "LTCUSD", "BCHUSD", "XRPUSD", "SOLUSD", "AVAXUSD", "UNIUSD"
            ]  # 50 diverse assets across stocks, ETFs, crypto, commodities

        # Default configuration for Golden Cross
        default_config = {
            "fast_ma_period": 50,  # Fast moving average period
            "slow_ma_period": 200,  # Slow moving average period
            "min_trend_strength": 0.02,  # Minimum 2% separation between MAs
            "volume_confirmation": True,  # Require volume confirmation
            "volume_multiplier": 1.1,  # Volume should be 1.1x average
            "max_position_size": 0.30,  # 30% max per ETF position
            "min_days_between_signals": 5,  # Minimum days between signals
            "trend_confirmation_days": 3,  # Days to confirm trend before entry
            "take_profit_pct": None,  # No take profit - ride the trend
            "stop_loss_pct": None,  # No stop loss - trust the crossover
            "min_confidence": 0.7,  # Confidence threshold
        }

        # Merge default config with provided config
        merged_config = {**default_config, **config}

        super().__init__(name="Golden Cross", symbols=symbols, **merged_config)

        # Track last signal dates to avoid whipsaws
        self.last_signal_dates = {}

        # Track crossover states to detect changes
        self.crossover_states = {}  # symbol -> 'golden' or 'death' or 'none'

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

    @property
    def fast_ma_period(self):
        """Get fast moving average period."""
        return self.config["fast_ma_period"]

    @property
    def slow_ma_period(self):
        """Get slow moving average period."""
        return self.config["slow_ma_period"]

    @property
    def min_trend_strength(self):
        """Get minimum trend strength threshold."""
        return self.config["min_trend_strength"]

    @property
    def max_position_size(self):
        """Get maximum position size."""
        return self.config["max_position_size"]

    def should_enter_position(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[StrategySignal]:
        """
        Determine if we should enter a position based on Golden Cross.

        Args:
            symbol: ETF symbol
            data: OHLCV data

        Returns:
            StrategySignal if should enter, None otherwise
        """
        # Skip if we already have a position
        if symbol in self.positions:
            return None

        # Need at least 220 days of data for 200-day MA + buffer
        if len(data) < 220:
            logger.debug(
                f"{symbol}: Insufficient data for Golden Cross (need 220+ days)"
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

            # Detect Golden Cross (50-day MA crosses above 200-day MA)
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
            confidence_factors.append(0.5)

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
                    confidence_factors.append(0.1)

            # Calculate confidence
            confidence = min(sum(confidence_factors), 1.0)

            # Must have at least the golden cross + one additional confirmation
            if len(conditions_met) >= 2 and confidence >= self.config["min_confidence"]:
                # Update tracking
                self.last_signal_dates[symbol] = datetime.now()
                self.crossover_states[symbol] = "golden"

                # Create signal
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    price=close_price,
                    stop_loss=None,  # Golden Cross trusts the crossover
                    take_profit=None,  # Ride the trend
                    strategy_name=self.name,
                    metadata={
                        "conditions_met": conditions_met,
                        "fast_ma": fast_ma_current,
                        "slow_ma": slow_ma_current,
                        "ma_separation_pct": ma_separation * 100,
                        "entry_reason": "golden_cross_bullish",
                        "crossover_type": "golden",
                    },
                )

                logger.info(
                    f"Golden Cross BUY signal for {symbol}: confidence={confidence:.3f}, "
                    f"50MA={fast_ma_current:.2f}, 200MA={slow_ma_current:.2f}"
                )
                return signal

        except Exception as e:
            logger.error(f"Error analyzing Golden Cross entry for {symbol}: {str(e)}")
            return None

        return None

    def should_exit_position(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[StrategySignal]:
        """
        Determine if we should exit existing position based on Death Cross.

        Args:
            symbol: ETF symbol
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

            # Detect Death Cross (50-day MA crosses below 200-day MA)
            death_cross_detected = self._detect_death_cross(
                recent, fast_ma_col, slow_ma_col
            )

            if death_cross_detected:
                # Check if we recently sent a signal to avoid whipsaws
                if symbol in self.last_signal_dates:
                    days_since_last = (
                        datetime.now() - self.last_signal_dates[symbol]
                    ).days
                    if days_since_last < self.config["min_days_between_signals"]:
                        return None

                # Update tracking
                self.last_signal_dates[symbol] = datetime.now()
                self.crossover_states[symbol] = "death"

                # Calculate P&L
                position = self.positions[symbol]
                entry_price = position["entry_price"]
                profit_loss_pct = (current_price - entry_price) / entry_price * 100

                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    confidence=0.9,  # High confidence for death cross
                    price=current_price,
                    strategy_name=self.name,
                    metadata={
                        "exit_reason": "death_cross_bearish",
                        "entry_price": entry_price,
                        "profit_loss_pct": profit_loss_pct,
                        "fast_ma": latest[fast_ma_col],
                        "slow_ma": latest[slow_ma_col],
                        "crossover_type": "death",
                    },
                )

                logger.info(
                    f"Death Cross SELL signal for {symbol}: P&L={profit_loss_pct:.2f}%, "
                    f"50MA={latest[fast_ma_col]:.2f}, 200MA={latest[slow_ma_col]:.2f}"
                )
                return signal

        except Exception as e:
            logger.error(f"Error analyzing Golden Cross exit for {symbol}: {str(e)}")

        return None

    def _detect_golden_cross(
        self, recent_data: pd.DataFrame, fast_ma_col: str, slow_ma_col: str
    ) -> bool:
        """
        Detect if a Golden Cross (50 MA crossing above 200 MA) occurred recently.

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
                        f"Golden Cross detected: {prev_fast:.2f} <= {prev_slow:.2f} -> {current_fast:.2f} > {current_slow:.2f}"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Error detecting Golden Cross: {str(e)}")
            return False

    def _detect_death_cross(
        self, recent_data: pd.DataFrame, fast_ma_col: str, slow_ma_col: str
    ) -> bool:
        """
        Detect if a Death Cross (50 MA crossing below 200 MA) occurred recently.

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
                        f"Death Cross detected: {prev_fast:.2f} >= {prev_slow:.2f} -> {current_fast:.2f} < {current_slow:.2f}"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Error detecting Death Cross: {str(e)}")
            return False

    def get_strategy_summary(self) -> Dict:
        """Get strategy-specific summary information."""
        summary = self.get_performance_summary()

        # Add required fields for test compatibility
        summary.update({
            "name": self.name,
            "symbols": self.symbols,
            "fast_ma_period": self.fast_ma_period,
            "slow_ma_period": self.slow_ma_period,
        })

        # Add Golden Cross specific metrics
        buy_signals = [
            s for s in self.signals_history if s.signal_type == SignalType.BUY
        ]
        sell_signals = [
            s
            for s in self.signals_history
            if s.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]
        ]

        golden_crosses = len(
            [s for s in buy_signals if s.metadata.get("crossover_type") == "golden"]
        )
        death_crosses = len(
            [s for s in sell_signals if s.metadata.get("crossover_type") == "death"]
        )

        if buy_signals and sell_signals:
            # Calculate completed trades
            completed_trades = []
            for sell_signal in sell_signals:
                # Find corresponding buy signal
                buy_signal = None
                for buy_sig in reversed(buy_signals):
                    if (
                        buy_sig.symbol == sell_signal.symbol
                        and buy_sig.timestamp < sell_signal.timestamp
                    ):
                        buy_signal = buy_sig
                        break

                if buy_signal:
                    holding_period = (sell_signal.timestamp - buy_signal.timestamp).days
                    profit_loss_pct = sell_signal.metadata.get("profit_loss_pct", 0)
                    completed_trades.append(
                        {
                            "holding_period": holding_period,
                            "profit_loss_pct": profit_loss_pct,
                        }
                    )

            if completed_trades:
                avg_holding_period = np.mean(
                    [t["holding_period"] for t in completed_trades]
                )
                win_rate = len(
                    [t for t in completed_trades if t["profit_loss_pct"] > 0]
                ) / len(completed_trades)
                avg_return = np.mean([t["profit_loss_pct"] for t in completed_trades])

                summary.update(
                    {
                        "avg_holding_period_days": round(avg_holding_period, 1),
                        "win_rate": round(win_rate * 100, 1),
                        "avg_return_pct": round(avg_return, 2),
                        "completed_trades": len(completed_trades),
                        "golden_crosses": golden_crosses,
                        "death_crosses": death_crosses,
                    }
                )

        summary["strategy_config"] = self.config
        summary["current_crossover_states"] = self.crossover_states.copy()

        return summary
