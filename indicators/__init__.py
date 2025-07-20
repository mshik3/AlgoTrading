"""
Technical indicators package for the algorithmic trading system.
Provides technical analysis tools for equity strategies.
"""

from .technical import (
    calculate_rsi,
    calculate_wilder_rsi,
    calculate_zscore,
    calculate_half_life,
    calculate_macd,
    calculate_sma,
    calculate_ema,
    calculate_bollinger_bands,
    calculate_stochastic,
    calculate_atr,
    TechnicalIndicators,
)

# Statistical analysis modules
from .statistical_tests import (
    calculate_hurst_exponent,
    adf_test,
    calculate_variance_ratio,
    calculate_autocorrelation,
    comprehensive_mean_reversion_test,
)

from .ou_process import (
    OUProcess,
    fit_ou_process_to_data,
    calculate_ou_optimal_strategy,
)

__all__ = [
    # Technical indicators
    "calculate_rsi",
    "calculate_wilder_rsi",
    "calculate_zscore",
    "calculate_half_life",
    "calculate_macd",
    "calculate_sma",
    "calculate_ema",
    "calculate_bollinger_bands",
    "calculate_stochastic",
    "calculate_atr",
    "TechnicalIndicators",
    # Statistical tests
    "calculate_hurst_exponent",
    "adf_test",
    "calculate_variance_ratio",
    "calculate_autocorrelation",
    "comprehensive_mean_reversion_test",
    # O-U process
    "OUProcess",
    "fit_ou_process_to_data",
    "calculate_ou_optimal_strategy",
]
