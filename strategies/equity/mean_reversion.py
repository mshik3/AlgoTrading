"""
Mean Reversion Strategy Implementation.

This strategy identifies assets that have deviated significantly from their historical
averages and bets on them returning to normal levels. It uses multiple technical
indicators to confirm oversold/overbought conditions.

Key Components:
- RSI for momentum analysis
- Bollinger Bands for volatility-based signals
- Moving averages for trend confirmation
- Volume analysis for confirmation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

from ..base import BaseStrategy, StrategySignal, SignalType
from indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy for equity trading.

    Strategy Logic:
    - Buy when price is oversold (RSI < 30, below lower Bollinger Band)
    - Additional confirmation from price being >2 std dev from MA
    - Sell when price returns to mean or becomes overbought
    - Uses volume confirmation to avoid low-volume moves
    """

    def __init__(self, symbols: List[str], **config):
        """
        Initialize Mean Reversion Strategy.

        Args:
            symbols: List of symbols to trade
            **config: Strategy configuration parameters
        """
        # Default configuration for mean reversion
        default_config = {
            "rsi_oversold": 30,  # RSI threshold for oversold
            "rsi_overbought": 70,  # RSI threshold for overbought
            "bb_period": 20,  # Bollinger Bands period
            "bb_std": 2,  # Bollinger Bands standard deviations
            "sma_period": 50,  # Simple moving average period
            "volume_sma_period": 20,  # Volume SMA period for confirmation
            "volume_multiplier": 1.2,  # Volume must be X times average
            "min_std_devs": 1.5,  # Minimum std devs from MA to consider
            "max_holding_period": 30,  # Max days to hold position
            "take_profit_pct": 0.15,  # 15% take profit target
            "stop_loss_pct": 0.08,  # 8% stop loss
            "min_confidence": 0.6,  # Lower threshold for mean reversion
        }

        # Merge default config with provided config
        merged_config = {**default_config, **config}

        super().__init__(name="Mean Reversion", symbols=symbols, **merged_config)

        # Track when positions were opened for holding period limits
        self.position_entry_dates = {}

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
        Determine if we should enter a position based on mean reversion criteria.

        Args:
            symbol: Stock symbol
            data: OHLCV data

        Returns:
            StrategySignal if should enter, None otherwise
        """
        # Skip if we already have a position
        if symbol in self.positions:
            return None

        # Need at least 60 days of data for reliable signals
        if len(data) < 60:
            return None

        # Calculate technical indicators
        try:
            indicators = TechnicalIndicators(data.copy())
            indicators.add_all_basic()
            analysis_data = indicators.get_data()

            # Get latest values
            latest = analysis_data.iloc[-1]

            # Check for required indicators
            required_indicators = [
                "RSI_14",
                "BB_Lower_20",
                "BB_Upper_20",
                "SMA_50",
                "Volume_SMA_20",
            ]
            if not all(indicator in latest.index for indicator in required_indicators):
                logger.warning(f"Missing required indicators for {symbol}")
                return None

            # Mean reversion entry conditions
            conditions_met = []
            confidence_factors = []

            # 1. RSI Oversold Condition
            rsi = latest["RSI_14"]
            if pd.notna(rsi) and rsi < self.config["rsi_oversold"]:
                conditions_met.append("RSI_OVERSOLD")
                confidence_factors.append(0.3)
                logger.debug(f"{symbol}: RSI oversold ({rsi:.2f})")

            # 2. Below Lower Bollinger Band
            close_price = latest["Close"]
            bb_lower = latest["BB_Lower_20"]
            if pd.notna(bb_lower) and close_price < bb_lower:
                conditions_met.append("BELOW_BB_LOWER")
                confidence_factors.append(0.35)
                logger.debug(
                    f"{symbol}: Below lower BB ({close_price:.2f} < {bb_lower:.2f})"
                )

            # 3. Significant deviation from moving average
            sma_50 = latest["SMA_50"]
            if pd.notna(sma_50):
                deviation_pct = abs(close_price - sma_50) / sma_50
                std_devs = deviation_pct / (
                    analysis_data["Close"].rolling(20).std().iloc[-1] / sma_50
                )

                if std_devs > self.config["min_std_devs"] and close_price < sma_50:
                    conditions_met.append("BELOW_MA_THRESHOLD")
                    confidence_factors.append(0.25)
                    logger.debug(f"{symbol}: Below MA by {std_devs:.2f} std devs")

            # 4. Volume confirmation (optional but increases confidence)
            volume = latest["Volume"]
            volume_sma = latest["Volume_SMA_20"]
            if (
                pd.notna(volume_sma)
                and volume > volume_sma * self.config["volume_multiplier"]
            ):
                conditions_met.append("VOLUME_CONFIRMATION")
                confidence_factors.append(0.1)
                logger.debug(f"{symbol}: High volume confirmation")

            # Need at least 2 main conditions (RSI oversold or below BB + another)
            main_conditions = ["RSI_OVERSOLD", "BELOW_BB_LOWER", "BELOW_MA_THRESHOLD"]
            main_conditions_met = [c for c in conditions_met if c in main_conditions]

            if len(main_conditions_met) >= 2:
                # Calculate confidence based on conditions met
                confidence = min(sum(confidence_factors), 1.0)

                # Calculate stop loss and take profit
                stop_loss = close_price * (1 - self.config["stop_loss_pct"])
                take_profit = close_price * (1 + self.config["take_profit_pct"])

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
                        "rsi": rsi,
                        "bb_lower": bb_lower,
                        "sma_50": sma_50,
                        "entry_reason": "mean_reversion_oversold",
                    },
                )

                logger.info(
                    f"Mean reversion BUY signal for {symbol}: confidence={confidence:.3f}, conditions={conditions_met}"
                )
                return signal

        except Exception as e:
            logger.error(f"Error analyzing entry for {symbol}: {str(e)}")
            return None

        return None

    def should_exit_position(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[StrategySignal]:
        """
        Determine if we should exit existing position.

        Args:
            symbol: Stock symbol
            data: OHLCV data

        Returns:
            StrategySignal if should exit, None otherwise
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        entry_price = position["entry_price"]

        try:
            # Calculate technical indicators
            indicators = TechnicalIndicators(data.copy())
            indicators.add_all_basic()
            analysis_data = indicators.get_data()

            # Get latest values
            latest = analysis_data.iloc[-1]
            current_price = latest["Close"]

            # Exit conditions
            exit_reasons = []
            confidence = 0.8  # High confidence for exits

            # 1. Take profit target hit
            if position["take_profit"] and current_price >= position["take_profit"]:
                exit_reasons.append("TAKE_PROFIT")
                confidence = 1.0

            # 2. Stop loss hit
            elif position["stop_loss"] and current_price <= position["stop_loss"]:
                exit_reasons.append("STOP_LOSS")
                confidence = 1.0

            # 3. Mean reversion completed - price returned to or above SMA
            elif pd.notna(latest.get("SMA_50")) and current_price >= latest["SMA_50"]:
                exit_reasons.append("MEAN_REVERSION_COMPLETE")

            # 4. RSI became overbought (mean reversion may have overcorrected)
            elif (
                pd.notna(latest.get("RSI_14"))
                and latest["RSI_14"] > self.config["rsi_overbought"]
            ):
                exit_reasons.append("RSI_OVERBOUGHT")

            # 5. Maximum holding period exceeded
            elif symbol in self.position_entry_dates:
                days_held = (datetime.now() - self.position_entry_dates[symbol]).days
                if days_held >= self.config["max_holding_period"]:
                    exit_reasons.append("MAX_HOLDING_PERIOD")

            # 6. Price above upper Bollinger Band (potential reversal)
            elif (
                pd.notna(latest.get("BB_Upper_20"))
                and current_price > latest["BB_Upper_20"]
            ):
                exit_reasons.append("ABOVE_BB_UPPER")

            if exit_reasons:
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    confidence=confidence,
                    price=current_price,
                    strategy_name=self.name,
                    metadata={
                        "exit_reasons": exit_reasons,
                        "entry_price": entry_price,
                        "profit_loss_pct": (current_price - entry_price)
                        / entry_price
                        * 100,
                        "rsi": latest.get("RSI_14"),
                        "sma_50": latest.get("SMA_50"),
                    },
                )

                logger.info(
                    f"Mean reversion SELL signal for {symbol}: reasons={exit_reasons}, P&L={(current_price-entry_price)/entry_price*100:.2f}%"
                )
                return signal

        except Exception as e:
            logger.error(f"Error analyzing exit for {symbol}: {str(e)}")

        return None

    def update_position(
        self, symbol: str, signal: StrategySignal, executed_price: float, quantity: int
    ):
        """Override to track entry dates."""
        super().update_position(symbol, signal, executed_price, quantity)

        if signal.signal_type == SignalType.BUY:
            self.position_entry_dates[symbol] = signal.timestamp
        elif signal.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]:
            if symbol in self.position_entry_dates:
                del self.position_entry_dates[symbol]

    def get_strategy_summary(self) -> Dict:
        """Get strategy-specific summary information."""
        summary = self.get_performance_summary()

        # Add mean reversion specific metrics
        buy_signals = [
            s for s in self.signals_history if s.signal_type == SignalType.BUY
        ]
        sell_signals = [
            s
            for s in self.signals_history
            if s.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]
        ]

        if buy_signals and sell_signals:
            # Calculate average holding period
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
                    }
                )

        summary["strategy_config"] = self.config
        return summary
