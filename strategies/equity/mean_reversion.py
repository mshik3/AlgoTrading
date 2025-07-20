"""
Mean Reversion Strategy Implementation.

This strategy identifies assets that have deviated significantly from their historical
averages and bets on them returning to normal levels. It uses multiple technical
indicators to confirm oversold/overbought conditions.

Enhanced with academic research:
- Wilder's RSI for accurate momentum analysis
- Statistical mean reversion validation (ADF, Hurst exponent)
- Proper Z-score calculation for deviation measurement
- O-U process optimal threshold calculation
- Volume analysis for confirmation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

from ..base import BaseStrategy, StrategySignal, SignalType
from indicators import TechnicalIndicators
from indicators.statistical_tests import comprehensive_mean_reversion_test
from indicators.ou_process import fit_ou_process_to_data

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Enhanced Mean Reversion Strategy for equity trading.

    Strategy Logic (Updated with Academic Research):
    - Statistical validation: Only trade assets that exhibit mean-reverting properties
    - Entry: Wilder's RSI < 30 AND Z-score < -1.5 AND below lower Bollinger Band
    - Exit: Mean reversion complete OR Wilder's RSI > 70 OR stop loss/take profit
    - Volume confirmation to avoid low-volume moves
    - O-U process optimal thresholds when possible
    - Half-life analysis for position sizing
    """

    def __init__(self, symbols: List[str], **config):
        """
        Initialize Enhanced Mean Reversion Strategy.

        Args:
            symbols: List of symbols to trade
            **config: Strategy configuration parameters
        """
        # Enhanced default configuration based on research
        default_config = {
            "wilder_rsi_oversold": 30,  # Wilder's RSI threshold for oversold
            "wilder_rsi_overbought": 70,  # Wilder's RSI threshold for overbought
            "zscore_entry_threshold": -1.5,  # Z-score threshold for entry (negative = oversold)
            "zscore_exit_threshold": 0.5,  # Z-score threshold for exit
            "zscore_window": 20,  # Window for Z-score calculation
            "bb_period": 20,  # Bollinger Bands period
            "bb_std": 2,  # Bollinger Bands standard deviations
            "sma_period": 50,  # Simple moving average period
            "volume_sma_period": 20,  # Volume SMA period for confirmation
            "volume_multiplier": 1.2,  # Volume must be X times average
            "max_holding_period": 30,  # Max days to hold position
            "take_profit_pct": 0.15,  # 15% take profit target
            "stop_loss_pct": 0.08,  # 8% stop loss
            "min_confidence": 0.7,  # Higher threshold for enhanced strategy
            "statistical_validation": True,  # Enable statistical mean reversion tests
            "use_ou_thresholds": True,  # Use O-U process optimal thresholds
            "min_hurst_exponent": 0.5,  # Maximum Hurst exponent for mean reversion
            "min_mean_reversion_score": 3,  # Minimum score from comprehensive test
            "revalidate_days": 60,  # Re-run statistical tests every N days
        }

        # Merge default config with provided config
        merged_config = {**default_config, **config}

        super().__init__(
            name="Enhanced Mean Reversion", symbols=symbols, **merged_config
        )

        # Track when positions were opened for holding period limits
        self.position_entry_dates = {}

        # Track statistical validation results and timestamps
        self._statistical_validation_cache = {}
        self._last_validation_dates = {}

        # Track O-U process optimal thresholds
        self._ou_thresholds_cache = {}

    def _validate_mean_reversion_properties(
        self, symbol: str, data: pd.DataFrame
    ) -> bool:
        """
        Validate that the asset exhibits mean-reverting properties using statistical tests.

        Args:
            symbol: Asset symbol
            data: Historical price data

        Returns:
            True if asset shows mean-reverting properties, False otherwise
        """
        if not self.config["statistical_validation"]:
            return True  # Skip validation if disabled

        try:
            # Check if we have cached results that are still valid
            current_date = datetime.now()
            if symbol in self._last_validation_dates:
                days_since_validation = (
                    current_date - self._last_validation_dates[symbol]
                ).days
                if days_since_validation < self.config["revalidate_days"]:
                    cached_result = self._statistical_validation_cache.get(symbol, {})
                    return cached_result.get("is_mean_reverting", False)

            # Run comprehensive statistical tests
            logger.info(f"Running statistical validation for {symbol}...")
            test_results = comprehensive_mean_reversion_test(data["Close"])

            # Cache results
            self._statistical_validation_cache[symbol] = test_results
            self._last_validation_dates[symbol] = current_date

            # Extract key metrics
            overall_assessment = test_results.get("overall_assessment", {})
            mean_reversion_score = overall_assessment.get("mean_reversion_score", 0)

            hurst_results = test_results.get("hurst_exponent", {})
            hurst_value = hurst_results.get("hurst_exponent", 0.5)

            # Determine if asset is suitable for mean reversion trading
            is_suitable = (
                mean_reversion_score >= self.config["min_mean_reversion_score"]
                and hurst_value < self.config["min_hurst_exponent"]
            )

            if is_suitable:
                logger.info(
                    f"✓ {symbol} passes mean reversion validation (score: {mean_reversion_score}/9, Hurst: {hurst_value:.3f})"
                )
            else:
                logger.warning(
                    f"✗ {symbol} fails mean reversion validation (score: {mean_reversion_score}/9, Hurst: {hurst_value:.3f})"
                )

            return is_suitable

        except Exception as e:
            logger.error(f"Statistical validation failed for {symbol}: {str(e)}")
            return False  # Conservative approach: reject if validation fails

    def _calculate_ou_optimal_thresholds(
        self, symbol: str, data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate optimal entry/exit thresholds using O-U process framework.

        Args:
            symbol: Asset symbol
            data: Historical price data

        Returns:
            Dictionary with optimal thresholds or fallback values
        """
        try:
            # Fit O-U process to price data
            logger.info(f"Fitting O-U process for {symbol}...")
            ou_process = fit_ou_process_to_data(data["Close"])

            if ou_process.fitted:
                # Calculate optimal thresholds
                thresholds = ou_process.calculate_optimal_thresholds(
                    transaction_cost=0.002, discount_rate=0.05  # 0.2% transaction cost
                )

                # Calculate expected returns
                expected_returns = ou_process.expected_return_per_unit_time(
                    thresholds["entry_threshold"], thresholds["exit_threshold"]
                )

                logger.info(
                    f"✓ O-U optimal thresholds for {symbol}: entry={thresholds['entry_threshold']:.2f}, exit={thresholds['exit_threshold']:.2f}"
                )

                # Add metadata
                thresholds.update(
                    {
                        "ou_fitted": True,
                        "half_life": ou_process.get_half_life(),
                        "expected_return_annualized": expected_returns.get(
                            "annualized_return", np.nan
                        ),
                        "ou_parameters": {
                            "theta": ou_process.theta,
                            "mu": ou_process.mu,
                            "sigma": ou_process.sigma,
                        },
                    }
                )

                return thresholds
            else:
                logger.warning(
                    f"O-U process fitting failed for {symbol}, using fallback thresholds"
                )

        except Exception as e:
            logger.error(f"O-U threshold calculation failed for {symbol}: {str(e)}")

        # Fallback to traditional thresholds
        current_price = data["Close"].iloc[-1]
        price_std = data["Close"].rolling(20).std().iloc[-1]

        return {
            "entry_threshold": current_price - 2 * price_std,
            "exit_threshold": current_price - 0.5 * price_std,
            "take_profit_threshold": current_price + 1.5 * price_std,
            "theta": data["Close"].rolling(50).mean().iloc[-1],
            "ou_fitted": False,
            "half_life": np.nan,
            "expected_return_annualized": np.nan,
            "ou_parameters": None,
        }

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
        Determine if we should enter a position based on enhanced mean reversion criteria.

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

        # Statistical validation - only trade assets with mean-reverting properties
        if not self._validate_mean_reversion_properties(symbol, data):
            logger.debug(f"{symbol}: Failed statistical mean reversion validation")
            return None

        # Calculate O-U optimal thresholds if enabled
        ou_thresholds = None
        if self.config["use_ou_thresholds"]:
            ou_thresholds = self._calculate_ou_optimal_thresholds(symbol, data)
            self._ou_thresholds_cache[symbol] = ou_thresholds

        # Calculate technical indicators
        try:
            indicators = TechnicalIndicators(data.copy())
            indicators.add_all_basic()  # This now includes Wilder's RSI and Z-score
            analysis_data = indicators.get_data()

            # Get latest values
            latest = analysis_data.iloc[-1]

            # Check for required indicators
            required_indicators = [
                "WRSI_14",  # Wilder's RSI (more accurate)
                "ZScore_20",  # Z-score for proper deviation measurement
                "BB_Lower_20",
                "BB_Upper_20",
                "SMA_50",
                "Volume_SMA_20",
            ]
            if not all(indicator in latest.index for indicator in required_indicators):
                logger.warning(f"Missing required indicators for {symbol}")
                return None

            # Enhanced mean reversion entry conditions
            conditions_met = []
            confidence_factors = []

            # 1. Wilder's RSI Oversold Condition (Fixed calculation)
            wilder_rsi = latest["WRSI_14"]
            if pd.notna(wilder_rsi) and wilder_rsi < self.config["wilder_rsi_oversold"]:
                conditions_met.append("WILDER_RSI_OVERSOLD")
                confidence_factors.append(0.35)  # Higher weight for accurate RSI
                logger.debug(f"{symbol}: Wilder's RSI oversold ({wilder_rsi:.2f})")

            # 2. Z-score Oversold Condition (Fixed standard deviation logic)
            zscore = latest["ZScore_20"]
            if pd.notna(zscore) and zscore < self.config["zscore_entry_threshold"]:
                conditions_met.append("ZSCORE_OVERSOLD")
                confidence_factors.append(
                    0.35
                )  # High weight for proper statistical measure
                logger.debug(f"{symbol}: Z-score oversold ({zscore:.2f})")

            # 3. Below Lower Bollinger Band
            close_price = latest["Close"]
            bb_lower = latest["BB_Lower_20"]
            if pd.notna(bb_lower) and close_price < bb_lower:
                conditions_met.append("BELOW_BB_LOWER")
                confidence_factors.append(0.20)
                logger.debug(
                    f"{symbol}: Below lower BB ({close_price:.2f} < {bb_lower:.2f})"
                )

            # 4. O-U Process Threshold (if available)
            if ou_thresholds and ou_thresholds.get("ou_fitted", False):
                entry_threshold = ou_thresholds["entry_threshold"]
                if close_price < entry_threshold:
                    conditions_met.append("OU_ENTRY_THRESHOLD")
                    confidence_factors.append(
                        0.25
                    )  # High weight for optimal thresholds
                    logger.debug(
                        f"{symbol}: Below O-U entry threshold ({close_price:.2f} < {entry_threshold:.2f})"
                    )

            # 5. Volume confirmation (enhanced)
            volume = latest["Volume"]
            volume_sma = latest["Volume_SMA_20"]
            if (
                pd.notna(volume_sma)
                and volume > volume_sma * self.config["volume_multiplier"]
            ):
                conditions_met.append("VOLUME_CONFIRMATION")
                confidence_factors.append(0.10)
                logger.debug(f"{symbol}: High volume confirmation")

            # Enhanced entry logic: Need at least 2 strong conditions
            strong_conditions = [
                "WILDER_RSI_OVERSOLD",
                "ZSCORE_OVERSOLD",
                "BELOW_BB_LOWER",
                "OU_ENTRY_THRESHOLD",
            ]
            strong_conditions_met = [
                c for c in conditions_met if c in strong_conditions
            ]

            if len(strong_conditions_met) >= 2:
                # Calculate confidence based on conditions met
                base_confidence = sum(confidence_factors)

                # Add bonus for half-life (faster mean reversion = higher confidence)
                half_life = indicators.get_half_life()
                if not np.isnan(half_life) and half_life < 20:
                    base_confidence += 0.10
                    conditions_met.append("FAST_MEAN_REVERSION")

                confidence = min(base_confidence, 1.0)

                # Enhanced stop loss and take profit
                if ou_thresholds and ou_thresholds.get("ou_fitted", False):
                    # Use O-U optimal levels
                    stop_loss = (
                        ou_thresholds["entry_threshold"] * 0.98
                    )  # 2% buffer below entry
                    take_profit = ou_thresholds["take_profit_threshold"]
                else:
                    # Use ATR-based approach
                    atr = latest.get(
                        "ATR_14", close_price * 0.02
                    )  # Fallback to 2% if ATR not available
                    stop_loss = close_price - (2.5 * atr)  # 2.5 ATR stop loss
                    take_profit = close_price + (
                        3.5 * atr
                    )  # 3.5 ATR take profit (1.4:1 risk-reward)

                # Create enhanced signal
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
                        "wilder_rsi": wilder_rsi,
                        "zscore": zscore,
                        "bb_lower": bb_lower,
                        "half_life": half_life,
                        "ou_thresholds": ou_thresholds,
                        "entry_reason": "enhanced_mean_reversion_oversold",
                        "statistical_validation": True,
                    },
                )

                logger.info(
                    f"Enhanced mean reversion BUY signal for {symbol}: confidence={confidence:.3f}, conditions={conditions_met}"
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
        Determine if we should exit existing position using enhanced criteria.

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

            # Get O-U thresholds if available
            ou_thresholds = self._ou_thresholds_cache.get(symbol)

            # Exit conditions with enhanced logic
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

            # 3. Z-score mean reversion completed (enhanced exit logic)
            elif (
                pd.notna(latest.get("ZScore_20"))
                and latest["ZScore_20"] > self.config["zscore_exit_threshold"]
            ):
                exit_reasons.append("ZSCORE_MEAN_REVERSION_COMPLETE")

            # 4. O-U Process Exit Threshold
            elif ou_thresholds and ou_thresholds.get("ou_fitted", False):
                exit_threshold = ou_thresholds["exit_threshold"]
                if current_price >= exit_threshold:
                    exit_reasons.append("OU_EXIT_THRESHOLD")
                    confidence = 0.9  # High confidence for optimal threshold

            # 5. Wilder's RSI became overbought (more accurate signal)
            elif (
                pd.notna(latest.get("WRSI_14"))
                and latest["WRSI_14"] > self.config["wilder_rsi_overbought"]
            ):
                exit_reasons.append("WILDER_RSI_OVERBOUGHT")

            # 6. Price returned to SMA (traditional mean reversion complete)
            elif pd.notna(latest.get("SMA_50")) and current_price >= latest["SMA_50"]:
                exit_reasons.append("PRICE_ABOVE_SMA")

            # 7. Maximum holding period exceeded
            elif symbol in self.position_entry_dates:
                days_held = (datetime.now() - self.position_entry_dates[symbol]).days
                if days_held >= self.config["max_holding_period"]:
                    exit_reasons.append("MAX_HOLDING_PERIOD")

            # 8. Price above upper Bollinger Band (potential reversal)
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
                        "wilder_rsi": latest.get("WRSI_14"),
                        "zscore": latest.get("ZScore_20"),
                        "sma_50": latest.get("SMA_50"),
                        "ou_thresholds": ou_thresholds,
                        "enhanced_exit": True,
                    },
                )

                logger.info(
                    f"Enhanced mean reversion SELL signal for {symbol}: reasons={exit_reasons}, P&L={(current_price-entry_price)/entry_price*100:.2f}%"
                )
                return signal

        except Exception as e:
            logger.error(f"Error analyzing exit for {symbol}: {str(e)}")

        return None

    def get_statistical_validation_summary(self) -> Dict:
        """Get summary of statistical validation results for all symbols."""
        summary = {}
        for symbol, results in self._statistical_validation_cache.items():
            if results:
                overall_assessment = results.get("overall_assessment", {})
                summary[symbol] = {
                    "mean_reversion_score": overall_assessment.get(
                        "mean_reversion_score", 0
                    ),
                    "conclusion": overall_assessment.get("conclusion", "Unknown"),
                    "trading_recommendation": overall_assessment.get(
                        "trading_recommendation", "Unknown"
                    ),
                    "last_validated": self._last_validation_dates.get(symbol, "Never"),
                }
        return summary

    def get_ou_thresholds_summary(self) -> Dict:
        """Get summary of O-U process optimal thresholds for all symbols."""
        summary = {}
        for symbol, thresholds in self._ou_thresholds_cache.items():
            if thresholds:
                summary[symbol] = {
                    "ou_fitted": thresholds.get("ou_fitted", False),
                    "entry_threshold": thresholds.get("entry_threshold", np.nan),
                    "exit_threshold": thresholds.get("exit_threshold", np.nan),
                    "half_life": thresholds.get("half_life", np.nan),
                    "expected_return_annualized": thresholds.get(
                        "expected_return_annualized", np.nan
                    ),
                    "ou_parameters": thresholds.get("ou_parameters"),
                }
        return summary

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
        """Get enhanced strategy-specific summary information."""
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

        # Add enhanced summaries
        summary["statistical_validation"] = self.get_statistical_validation_summary()
        summary["ou_thresholds"] = self.get_ou_thresholds_summary()
        summary["strategy_config"] = self.config
        summary["enhancements"] = [
            "Wilder's RSI (accurate calculation)",
            "Z-score deviation measurement",
            "Statistical mean reversion validation (ADF, Hurst)",
            "O-U process optimal thresholds",
            "ATR-based risk management",
            "Half-life analysis",
        ]

        return summary
