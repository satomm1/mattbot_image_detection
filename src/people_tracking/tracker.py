"""Map-frame people tracker: one constant-velocity Kalman filter per person.

Input is one detection frame at a time: the map-frame positions (and widths) of the
people the detector saw, stamped with the image time. Output is the confirmed tracks
with position AND velocity, which the detector cannot measure. ROS-free and clocked
only by the stamps passed in, so it runs the same live, under sim time and on replay.

Per frame (:meth:`PeopleTracker.update`):

1. predict every track to the frame time (state ``[x, y, vx, vy]``, white-acceleration
   process noise ``accel_std``);
2. associate detections to tracks with the Hungarian algorithm on the Mahalanobis
   distance of the innovation, after a hard Euclidean gate ``gate_m``;
3. update matched tracks (measurement noise ``meas_std``); unmatched detections start
   tentative tracks (zero velocity, large velocity uncertainty);
4. confirm a track after ``min_hits`` detections; delete one unseen for ``max_coast_s``
   (and a tentative one at its first miss frame, so one false detection never confirms).

Speeds are clamped to ``max_speed``. Frames with no people still count: they advance
time and age the tracks. No frames at all (the detector skips frames while the robot
turns) just leaves tracks coasting until they expire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


@dataclass
class TrackerParams:
    accel_std: float = 1.0      # m/s^2: how hard people may change velocity (process noise)
    meas_std: float = 0.15      # m: detector position noise (1 sigma, per axis)
    init_vel_std: float = 1.0   # m/s: velocity uncertainty of a new track
    gate_m: float = 1.0         # m: never associate a detection farther than this from a track's prediction
    gate_chi2: float = 13.8     # Mahalanobis^2 gate (chi^2, 2 dof, 99.9 %)
    min_hits: int = 2           # detections before a track is confirmed (reported)
    max_coast_s: float = 1.0    # s: delete a track not detected for this long
    max_speed: float = 2.0      # m/s: clamp (people walk at up to ~1.5 m/s)
    width_alpha: float = 0.3    # EMA weight of the newest width (radius smoothing)
    min_radius: float = 0.1
    max_radius: float = 0.5

    def __post_init__(self) -> None:
        for name in ("accel_std", "meas_std", "init_vel_std", "gate_m", "gate_chi2", "max_coast_s", "max_speed"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.min_hits < 1:
            raise ValueError("min_hits must be >= 1")


@dataclass
class Track:
    id: int
    x: np.ndarray               # (4,) [x, y, vx, vy]
    P: np.ndarray               # (4, 4)
    t: float                    # time the state is valid at
    last_seen: float
    radius: float
    hits: int = 1
    confirmed: bool = False
    _missed_frames: int = field(default=0, repr=False)

    @property
    def pos_std(self) -> float:
        return float(math.sqrt(max(0.5 * (self.P[0, 0] + self.P[1, 1]), 0.0)))

    @property
    def vel_std(self) -> float:
        return float(math.sqrt(max(0.5 * (self.P[2, 2] + self.P[3, 3]), 0.0)))


class PeopleTracker:
    def __init__(self, params: Optional[TrackerParams] = None) -> None:
        self.p = params or TrackerParams()
        self.tracks: List[Track] = []
        self._next_id = 1
        self.t: Optional[float] = None

    # ------------------------------------------------------------------ filter

    def _predicted(self, tr: Track, t: float):
        dt = max(t - tr.t, 0.0)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.p.accel_std ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q = q * np.array([[dt4 / 4, 0, dt3 / 2, 0], [0, dt4 / 4, 0, dt3 / 2],
                          [dt3 / 2, 0, dt2, 0], [0, dt3 / 2, 0, dt2]])
        return F @ tr.x, F @ tr.P @ F.T + Q

    def _clamp_speed(self, x: np.ndarray) -> np.ndarray:
        s = math.hypot(x[2], x[3])
        if s > self.p.max_speed:
            x = x.copy()
            x[2:] *= self.p.max_speed / s
        return x

    # ------------------------------------------------------------------ frames

    def update(self, t: float, xy: Sequence[Sequence[float]], widths: Optional[Sequence[float]] = None) -> None:
        """One detection frame at time ``t``: ``xy`` (N, 2) map-frame positions, ``widths`` (N,) m."""
        t = float(t)
        if self.t is not None and t < self.t:
            return  # an out-of-order (older) frame: drop it rather than rewind the filters
        self.t = t
        z = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        keep = np.all(np.isfinite(z), axis=1)
        w = np.full(len(z), 2 * 0.25) if widths is None else np.asarray(widths, dtype=np.float64).reshape(-1)
        z, w = z[keep], w[keep]

        R = (self.p.meas_std ** 2) * np.eye(2)
        preds = [self._predicted(tr, t) for tr in self.tracks]
        n_t, n_d = len(self.tracks), len(z)
        matches = []
        if n_t and n_d:
            cost = np.full((n_t, n_d), 1e9)
            for i, (xp, Pp) in enumerate(preds):
                S = _H @ Pp @ _H.T + R
                Si = np.linalg.inv(S)
                for j in range(n_d):
                    nu = z[j] - xp[:2]
                    if math.hypot(nu[0], nu[1]) > self.p.gate_m:
                        continue
                    d2 = float(nu @ Si @ nu)
                    if d2 <= self.p.gate_chi2:
                        cost[i, j] = d2
            rows, cols = linear_sum_assignment(cost)
            matches = [(i, j) for i, j in zip(rows, cols) if cost[i, j] < 1e9]

        matched_t = {i for i, _ in matches}
        matched_d = {j for _, j in matches}
        for i, (xp, Pp) in enumerate(preds):
            tr = self.tracks[i]
            tr.x, tr.P, tr.t = self._clamp_speed(xp), Pp, t
            if i not in matched_t:
                tr._missed_frames += 1
        for i, j in matches:
            tr = self.tracks[i]
            S = _H @ tr.P @ _H.T + R
            K = tr.P @ _H.T @ np.linalg.inv(S)
            tr.x = self._clamp_speed(tr.x + K @ (z[j] - tr.x[:2]))
            tr.P = (np.eye(4) - K @ _H) @ tr.P
            tr.P = 0.5 * (tr.P + tr.P.T)
            tr.last_seen = t
            tr.hits += 1
            tr._missed_frames = 0
            r = float(np.clip(0.5 * w[j], self.p.min_radius, self.p.max_radius))
            tr.radius = self.p.width_alpha * r + (1.0 - self.p.width_alpha) * tr.radius
            if tr.hits >= self.p.min_hits:
                tr.confirmed = True

        self.tracks = [tr for tr in self.tracks
                       if t - tr.last_seen <= self.p.max_coast_s and (tr.confirmed or tr._missed_frames == 0)]
        for j in range(n_d):
            if j in matched_d:
                continue
            P0 = np.diag([self.p.meas_std ** 2] * 2 + [self.p.init_vel_std ** 2] * 2)
            r = float(np.clip(0.5 * w[j], self.p.min_radius, self.p.max_radius))
            tr = Track(self._next_id, np.array([z[j, 0], z[j, 1], 0.0, 0.0]), P0, t, t, r)
            tr.confirmed = self.p.min_hits <= 1
            self._next_id += 1
            self.tracks.append(tr)

    def confirmed(self, t: Optional[float] = None) -> List[Track]:
        """Copies of the confirmed tracks; with ``t``, predicted to it (the filters are not advanced)."""
        out = []
        for tr in self.tracks:
            if not tr.confirmed:
                continue
            x, P, at = tr.x.copy(), tr.P.copy(), tr.t
            if t is not None and t > tr.t:
                x, P = self._predicted(tr, float(t))
                x, at = self._clamp_speed(x), float(t)
            out.append(Track(tr.id, x, P, at, tr.last_seen, tr.radius, tr.hits, True))
        return out
