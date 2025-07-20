"""
Technical indicators module for equity trading strategies.
Provides optimized pandas-based implementations of common technical indicators.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Union, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)


def calculate_sma(data: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        data: Price series (typically close prices)
        window: Number of periods for the moving average

    Returns:
        Series with SMA values
    """
    return data.rolling(window=window).mean()


def calculate_ema(
    data: pd.Series, window: int = 20, alpha: Optional[float] = None
) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        data: Price series (typically close prices)
        window: Number of periods for the EMA
        alpha: Smoothing factor. If None, calculated as 2/(window+1)

    Returns:
        Series with EMA values
    """
    if alpha is None:
        alpha = 2.0 / (window + 1)

    return data.ewm(alpha=alpha, adjust=False).mean()


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    if window <= 0:
        raise ValueError("Window must be positive for RSI calculation.")
    """
    Calculate Relative Strength Index (RSI).

    Args:
        data: Price series (typically close prices)
        window: Number of periods for RSI calculation

    Returns:
        Series with RSI values (0-100)
    """
    # Calculate price changes
    delta = data.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    rsi.iloc[:window] = np.nan  # Ensure first `window` values are NaN

    return rsi


def calculate_wilder_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Wilder's RSI using proper exponential smoothing.

    This is the original RSI calculation by J. Welles Wilder using his smoothing method:
    Smoothed value = Previous smoothed value + (1/n) * (Current value - Previous smoothed value)

    This method is more accurate than the simple moving average approach for RSI.

    Args:
        data: Price series (typically close prices)
        window: Number of periods for RSI calculation

    Returns:
        Series with Wilder's RSI values (0-100)
    """
    if window <= 0:
        raise ValueError("Window must be positive for Wilder's RSI calculation.")

    delta = data.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Initialize result series
    rsi = pd.Series(index=data.index, dtype=float)
    rsi.iloc[:window] = np.nan

    # Calculate initial averages using SMA for first window
    initial_avg_gain = gains.iloc[1 : window + 1].mean()
    initial_avg_loss = losses.iloc[1 : window + 1].mean()

    # Start Wilder's smoothing from the window+1 position
    smoothed_gains = [initial_avg_gain]
    smoothed_losses = [initial_avg_loss]

    for i in range(window + 1, len(data)):
        # Wilder's smoothing formula
        smoothed_gain = smoothed_gains[-1] + (1 / window) * (
            gains.iloc[i] - smoothed_gains[-1]
        )
        smoothed_loss = smoothed_losses[-1] + (1 / window) * (
            losses.iloc[i] - smoothed_losses[-1]
        )

        smoothed_gains.append(smoothed_gain)
        smoothed_losses.append(smoothed_loss)

        # Calculate RSI
        if smoothed_loss == 0:
            rsi.iloc[i] = 100
        else:
            rs = smoothed_gain / smoothed_loss
            rsi.iloc[i] = 100 - (100 / (1 + rs))

    return rsi


def calculate_zscore(
    data: pd.Series, window: int = 20, min_periods: int = None
) -> pd.Series:
    """
    Calculate Z-score for mean reversion analysis.

    Z-score measures how many standard deviations a value is from the mean.
    This is the correct way to measure mean reversion strength.

    Args:
        data: Price series
        window: Rolling window for mean and std calculation
        min_periods: Minimum periods required for calculation

    Returns:
        Series with Z-score values
    """
    if min_periods is None:
        min_periods = window

    # Calculate rolling mean and standard deviation
    rolling_mean = data.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = data.rolling(window=window, min_periods=min_periods).std()

    # Calculate Z-score: (current_value - mean) / std_dev
    zscore = (data - rolling_mean) / rolling_std

    return zscore


def calculate_half_life(data: pd.Series) -> float:
    """
    Calculate the half-life of mean reversion for a time series.

    Half-life indicates how long it takes for a deviation from the mean
    to decay by half. Shorter half-life indicates stronger mean reversion.

    Args:
        data: Price series

    Returns:
        Half-life in periods (e.g., days if daily data)
    """
    # Calculate price differences (lag-1 regression)
    lagged_data = data.shift(1)
    price_diff = data.diff()

    # Remove NaN values
    valid_indices = ~(lagged_data.isna() | price_diff.isna())
    y = price_diff[valid_indices]
    x = lagged_data[valid_indices]

    if len(y) < 2:
        return np.nan

    # Perform linear regression: Δp_t = α + β * p_{t-1} + ε_t
    # The coefficient β tells us the mean reversion speed
    x_mean = x.mean()
    y_mean = y.mean()

    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()

    if denominator == 0:
        return np.nan

    beta = numerator / denominator

    # Half-life = -ln(2) / ln(1 + β)
    # For mean-reverting series, β should be negative
    if beta >= 0:
        return np.nan  # No mean reversion detected

    try:
        half_life = -np.log(2) / np.log(1 + beta)
        return half_life
    except (ValueError, ZeroDivisionError):
        return np.nan


def calculate_macd(
    data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        data: Price series (typically close prices)
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period

    Returns:
        DataFrame with columns: macd, signal, histogram
    """
    # Calculate EMAs
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line
    signal_line = calculate_ema(macd_line, signal)

    # Calculate histogram
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def calculate_bollinger_bands(
    data: pd.Series, window: int = 20, num_std: float = 2
) -> tuple:
    middle = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, lower, middle


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_window: Number of periods for %K calculation
        d_window: Number of periods for %D smoothing

    Returns:
        DataFrame with columns: k_percent, d_percent
    """
    # Calculate the lowest low and highest high over the window
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()

    # Calculate %K
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))

    # Calculate %D (smoothed %K)
    d_percent = k_percent.rolling(window=d_window).mean()

    return pd.DataFrame({"k_percent": k_percent, "d_percent": d_percent})


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Number of periods for ATR calculation

    Returns:
        Series with ATR values
    """
    # Calculate True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    # True Range is the maximum of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR as EMA of True Range
    atr = calculate_ema(true_range, window)

    return atr


def calculate_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """
    Calculate Williams %R.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Number of periods for calculation

    Returns:
        Series with Williams %R values (-100 to 0)
    """
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()

    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))

    return williams_r


def calculate_volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average of volume.

    Args:
        volume: Volume series
        window: Number of periods for the moving average

    Returns:
        Series with volume SMA values
    """
    return volume.rolling(window=window).mean()


class TechnicalIndicators:
    """
    Class for calculating multiple technical indicators on market data.
    Provides a convenient interface for strategy development.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data.

        Args:
            data: DataFrame with columns: Open, High, Low, Close, Volume
        """
        self.data = data
        self._validate_data()

    def _validate_data(self):
        """Validate that required columns are present."""
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def add_sma(self, period: int = 20, column: str = "Close") -> "TechnicalIndicators":
        """Add Simple Moving Average to the data."""
        self.data[f"SMA_{period}"] = calculate_sma(self.data[column], period)
        return self

    def add_ema(self, period: int = 20, column: str = "Close") -> "TechnicalIndicators":
        """Add Exponential Moving Average to the data."""
        self.data[f"EMA_{period}"] = calculate_ema(self.data[column], period)
        return self

    def add_rsi(self, period: int = 14, column: str = "Close") -> "TechnicalIndicators":
        """Add Relative Strength Index to the data."""
        self.data[f"RSI_{period}"] = calculate_rsi(self.data[column], period)
        return self

    def add_wilder_rsi(self, period: int = 14, column: str = "Close") -> "TechnicalIndicators":
        """Add Wilder's RSI (more accurate) to the data."""
        self.data[f"WRSI_{period}"] = calculate_wilder_rsi(self.data[column], period)
        return self

    def add_zscore(self, window: int = 20, column: str = "Close") -> "TechnicalIndicators":
        """Add Z-score for mean reversion analysis."""
        self.data[f"ZScore_{window}"] = calculate_zscore(self.data[column], window)
        return self

    def add_half_life_analysis(self, column: str = "Close") -> "TechnicalIndicators":
        """Add half-life of mean reversion."""
        half_life = calculate_half_life(self.data[column])
        # Store as metadata - half-life is a single value for the entire series
        self._half_life = half_life
        return self

    def get_half_life(self) -> float:
        """Get the calculated half-life of mean reversion."""
        return getattr(self, '_half_life', np.nan)

    def add_macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "Close"
    ) -> "TechnicalIndicators":
        """Add MACD indicators to the data."""
        macd_data = calculate_macd(self.data[column], fast, slow, signal)
        self.data[f"MACD_{fast}_{slow}_{signal}"] = macd_data["macd"]
        self.data[f"MACD_Signal_{fast}_{slow}_{signal}"] = macd_data["signal"]
        self.data[f"MACD_Histogram_{fast}_{slow}_{signal}"] = macd_data["histogram"]
        return self

    def add_bollinger_bands(
        self, period: int = 20, std_dev: float = 2, column: str = "Close"
    ) -> "TechnicalIndicators":
        """Add Bollinger Bands to the data."""
        bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(
            self.data[column], period, std_dev
        )
        self.data[f"BB_Upper_{period}"] = bb_upper
        self.data[f"BB_Lower_{period}"] = bb_lower
        self.data[f"BB_Middle_{period}"] = bb_middle
        return self

    def add_stochastic(
        self, k_period: int = 14, d_period: int = 3
    ) -> "TechnicalIndicators":
        """Add Stochastic Oscillator to the data."""
        stoch_data = calculate_stochastic(
            self.data["High"], self.data["Low"], self.data["Close"], k_period, d_period
        )
        self.data[f"Stoch_K_{k_period}"] = stoch_data["k_percent"]
        self.data[f"Stoch_D_{k_period}_{d_period}"] = stoch_data["d_percent"]
        return self

    def add_atr(self, period: int = 14) -> "TechnicalIndicators":
        """Add Average True Range to the data."""
        self.data[f"ATR_{period}"] = calculate_atr(
            self.data["High"], self.data["Low"], self.data["Close"], period
        )
        return self

    def add_williams_r(self, period: int = 14) -> "TechnicalIndicators":
        """Add Williams %R to the data."""
        self.data[f"Williams_R_{period}"] = calculate_williams_r(
            self.data["High"], self.data["Low"], self.data["Close"], period
        )
        return self

    def add_volume_sma(self, period: int = 20) -> "TechnicalIndicators":
        """Add Volume Simple Moving Average to the data."""
        self.data[f"Volume_SMA_{period}"] = calculate_volume_sma(
            self.data["Volume"], period
        )
        return self

    def add_all_basic(self) -> "TechnicalIndicators":
        """Add all basic indicators commonly used in equity strategies."""
        return (
            self.add_sma(20)
            .add_sma(50)
            .add_sma(200)  # Key moving averages
            .add_ema(12)
            .add_ema(26)  # MACD components
            .add_rsi(14)  # Standard RSI
            .add_wilder_rsi(14)  # Wilder's RSI (more accurate)
            .add_zscore(20)  # Z-score for mean reversion
            .add_half_life_analysis()  # Half-life analysis
            .add_macd()  # MACD
            .add_bollinger_bands(20, 2)  # Bollinger Bands
            .add_stochastic(14, 3)  # Stochastic
            .add_atr(14)  # ATR for volatility
            .add_volume_sma(20)
        )  # Volume trend

    def get_data(self) -> pd.DataFrame:
        """Return the data with all calculated indicators."""
        return self.data

    def get_latest_values(self) -> pd.Series:
        """Get the most recent values of all indicators."""
        return self.data.iloc[-1]

    def is_oversold(
        self, rsi_threshold: float = 30, stoch_threshold: float = 20
    ) -> bool:
        """
        Check if the latest data point indicates oversold conditions.

        Args:
            rsi_threshold: RSI threshold for oversold (default 30)
            stoch_threshold: Stochastic %K threshold for oversold (default 20)

        Returns:
            True if conditions suggest oversold
        """
        latest = self.get_latest_values()

        # Check RSI
        rsi_oversold = False
        if "RSI_14" in latest and pd.notna(latest["RSI_14"]):
            rsi_oversold = latest["RSI_14"] < rsi_threshold

        # Check Stochastic
        stoch_oversold = False
        if "Stoch_K_14" in latest and pd.notna(latest["Stoch_K_14"]):
            stoch_oversold = latest["Stoch_K_14"] < stoch_threshold

        return rsi_oversold or stoch_oversold

    def is_overbought(
        self, rsi_threshold: float = 70, stoch_threshold: float = 80
    ) -> bool:
        """
        Check if the latest data point indicates overbought conditions.

        Args:
            rsi_threshold: RSI threshold for overbought (default 70)
            stoch_threshold: Stochastic %K threshold for overbought (default 80)

        Returns:
            True if conditions suggest overbought
        """
        latest = self.get_latest_values()

        # Check RSI
        rsi_overbought = False
        if "RSI_14" in latest and pd.notna(latest["RSI_14"]):
            rsi_overbought = latest["RSI_14"] > rsi_threshold

        # Check Stochastic
        stoch_overbought = False
        if "Stoch_K_14" in latest and pd.notna(latest["Stoch_K_14"]):
            stoch_overbought = latest["Stoch_K_14"] > stoch_threshold

        return rsi_overbought or stoch_overbought

    def is_golden_cross(
        self, fast_ma: str = "SMA_50", slow_ma: str = "SMA_200"
    ) -> bool:
        """
        Check for golden cross (fast MA crossing above slow MA).

        Args:
            fast_ma: Fast moving average column name
            slow_ma: Slow moving average column name

        Returns:
            True if golden cross occurred in the latest data
        """
        if len(self.data) < 2:
            return False

        if fast_ma not in self.data.columns or slow_ma not in self.data.columns:
            return False

        current_fast = self.data[fast_ma].iloc[-1]
        current_slow = self.data[slow_ma].iloc[-1]
        prev_fast = self.data[fast_ma].iloc[-2]
        prev_slow = self.data[slow_ma].iloc[-2]

        # Check if fast MA crossed above slow MA
        return (prev_fast <= prev_slow) and (current_fast > current_slow)

    def is_death_cross(self, fast_ma: str = "SMA_50", slow_ma: str = "SMA_200") -> bool:
        """
        Check for death cross (fast MA crossing below slow MA).

        Args:
            fast_ma: Fast moving average column name
            slow_ma: Slow moving average column name

        Returns:
            True if death cross occurred in the latest data
        """
        if len(self.data) < 2:
            return False

        if fast_ma not in self.data.columns or slow_ma not in self.data.columns:
            return False

        current_fast = self.data[fast_ma].iloc[-1]
        current_slow = self.data[slow_ma].iloc[-1]
        prev_fast = self.data[fast_ma].iloc[-2]
        prev_slow = self.data[slow_ma].iloc[-2]

        # Check if fast MA crossed below slow MA
        return (prev_fast >= prev_slow) and (current_fast < current_slow)

    def get_signals(self) -> Dict[str, Any]:
        """
        Get trading signals based on technical indicators.

        Returns:
            Dictionary with signal information
        """
        signals = {}

        if len(self.data) < 2:
            return signals

        latest = self.get_latest_values()

        # RSI signals
        if "RSI_14" in latest and pd.notna(latest["RSI_14"]):
            rsi = latest["RSI_14"]
            if rsi < 30:
                signals["rsi_oversold"] = True
            elif rsi > 70:
                signals["rsi_overbought"] = True

        # MACD signals
        if "MACD_12_26_9" in latest and "MACD_Signal_12_26_9" in latest:
            macd = latest["MACD_12_26_9"]
            signal = latest["MACD_Signal_12_26_9"]
            if pd.notna(macd) and pd.notna(signal):
                if macd > signal:
                    signals["macd_bullish"] = True
                else:
                    signals["macd_bearish"] = True

        # Moving average signals
        if "SMA_50" in latest and "SMA_200" in latest:
            sma_50 = latest["SMA_50"]
            sma_200 = latest["SMA_200"]
            if pd.notna(sma_50) and pd.notna(sma_200):
                if sma_50 > sma_200:
                    signals["ma_bullish"] = True
                else:
                    signals["ma_bearish"] = True

        return signals
