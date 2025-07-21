"""
ETF Rotation Strategies Package.

This package contains ETF rotation strategies that implement various momentum-based
approaches for tactical asset allocation and sector rotation.
"""

from .rotation_base import BaseETFRotationStrategy
from .dual_momentum import DualMomentumStrategy
from .sector_rotation import SectorRotationStrategy

__all__ = [
    "BaseETFRotationStrategy",
    "DualMomentumStrategy",
    "SectorRotationStrategy",
]
