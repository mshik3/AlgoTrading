"""
Execution module for trade execution and paper trading simulation.
Provides interfaces for different execution methods including paper trading.
"""

from .paper import PaperTradingSimulator

__all__ = ["PaperTradingSimulator"]
