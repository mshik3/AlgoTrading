"""
Trading strategies package for the algorithmic trading system.
Provides base classes and implementations for equity and ETF strategies.
"""

from .base import BaseStrategy, StrategySignal, SignalType
from .equity.golden_cross import GoldenCrossStrategy
from .equity.mean_reversion import MeanReversionStrategy
from .etf.dual_momentum import DualMomentumStrategy
from .etf.sector_rotation import SectorRotationStrategy

__all__ = [
    "BaseStrategy", 
    "StrategySignal", 
    "SignalType",
    "GoldenCrossStrategy",
    "MeanReversionStrategy",
    "DualMomentumStrategy",
    "SectorRotationStrategy",
]
