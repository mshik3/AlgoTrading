"""
Trading strategies package for the algorithmic trading system.
Provides base classes and implementations for equity and options strategies.
"""

from .base import BaseStrategy, StrategySignal, SignalType

__all__ = ["BaseStrategy", "StrategySignal", "SignalType"]
