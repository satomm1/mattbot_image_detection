"""When to skip a camera frame because the robot is turning, and how well RGB and depth pair up.

:class:`TurningGate` decides from the robot's MEASURED yaw rate (odometry), at the image's
own time. The detector used to gate on the commanded rate (``/cmd_vel``) above 0.10 rad/s,
which drops frames on every small steering correction, and its state froze whenever
``/cmd_vel`` stopped publishing. Odometry says what the robot actually does, and the gate
now only has to cover what is still unreliable while turning fast: the colour and depth
images of a pair are taken at slightly different times, so a fast turn shifts one against
the other. (The camera pose is looked up at the image time, so the pose itself is right.)

A frame at time ``t`` is skipped if the robot turned faster than ``threshold`` (rad/s) at any
odometry sample within ``cooldown`` (s) before ``t``. Without recent odometry (none within
``max_odom_age`` of ``t``) frames are never skipped: losing all detections is worse than a few
misaligned ones.

:class:`PairStats` keeps the time gaps between the colour and depth images of the pairs the
synchronizer formed, to check the pairing slop against the camera's real timing.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class TurningGate:
    def __init__(self, threshold: float = 0.8, cooldown: float = 0.15, max_odom_age: float = 0.5,
                 history_s: float = 2.0) -> None:
        if threshold <= 0.0 or cooldown < 0.0 or max_odom_age <= 0.0:
            raise ValueError("threshold and max_odom_age must be positive, cooldown non-negative")
        self.threshold, self.cooldown, self.max_odom_age = float(threshold), float(cooldown), float(max_odom_age)
        self.history_s = max(float(history_s), self.cooldown + self.max_odom_age)
        self._samples: Deque[Tuple[float, float]] = deque()  # (stamp, |yaw rate|)
        self._lock = threading.Lock()  # odometry and images arrive on different threads

    def add_odom(self, stamp: float, yaw_rate: float) -> None:
        stamp = float(stamp)
        with self._lock:
            if self._samples and stamp < self._samples[-1][0]:
                return  # out of order
            self._samples.append((stamp, abs(float(yaw_rate))))
            while self._samples and self._samples[0][0] < stamp - self.history_s:
                self._samples.popleft()

    def turning(self, t: float) -> bool:
        """True if the frame at time ``t`` should be skipped."""
        t = float(t)
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return False
        near = [s for s in samples if abs(t - s[0]) <= self.max_odom_age]
        if not near:
            return False  # no recent odometry: never skip
        return any(w > self.threshold for s, w in samples if t - self.cooldown <= s <= t + 1e-9)


class PairStats:
    """Gaps (s) between the colour and depth stamps of the formed pairs, plus arrival counts."""

    def __init__(self, keep: int = 300) -> None:
        self.gaps: Deque[float] = deque(maxlen=int(keep))
        self.n_rgb = self.n_depth = self.n_pairs = 0

    def add_pair(self, rgb_stamp: float, depth_stamp: float) -> None:
        self.gaps.append(abs(float(rgb_stamp) - float(depth_stamp)))
        self.n_pairs += 1

    def summary(self) -> Optional[Tuple[float, float]]:
        """``(median, max)`` gap of the recent pairs, or None before the first pair."""
        if not self.gaps:
            return None
        g = np.asarray(self.gaps)
        return float(np.median(g)), float(g.max())
