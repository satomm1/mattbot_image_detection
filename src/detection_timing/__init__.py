"""Frame timing helpers for the detector: the turning gate and RGB/depth pairing statistics.

ROS-free and clocked only by the stamps passed in, so they are unit-tested directly
(``test/test_detection_timing.py``).
"""

from .gating import PairStats, TurningGate

__all__ = ["PairStats", "TurningGate"]
