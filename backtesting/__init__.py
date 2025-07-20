"""
Backtesting module for algorithmic trading strategies.
Provides historical data replay and performance analysis capabilities.
"""

from .engine import BacktestingEngine
from .metrics import PerformanceMetrics

__all__ = ["BacktestingEngine", "PerformanceMetrics"]
