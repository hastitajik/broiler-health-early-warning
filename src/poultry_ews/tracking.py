"""Multi-object tracking of birds on the FLOOR PLANE (metres).

Detections are reduced to their box centre, projected to floor metres with the
camera homography, and linked frame to frame with the Hungarian algorithm on
Euclidean distance (gated). Tracking in metres instead of pixels makes the
gate the same everywhere in the image despite the oblique camera.

Track identities are only kept within one clip. Across clips, still birds are
linked by position (see signals/mortality.py and pipeline._track_mortality),
which is what the mortality signal needs.

MVBroTrack's own trackers (SORT / muSSP, ground-plane fusion across cameras)
are more elaborate; their output can be used instead through the
`mvbrotrack` detector kind plus this tracker, or by extending this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    t: list[float] = field(default_factory=list)      # seconds from clip start
    x: list[float] = field(default_factory=list)      # metres
    y: list[float] = field(default_factory=list)
    area: list[float] = field(default_factory=list)   # m^2 (box area on the floor plane)
    cls: list[int] = field(default_factory=list)
    missed: int = 0

    @property
    def duration(self) -> float:
        return self.t[-1] - self.t[0] if len(self.t) > 1 else 0.0

    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])


class FloorTracker:
    def __init__(self, max_dist_m: float = 0.35, max_missed: int = 4):
        self.max_dist = max_dist_m
        self.max_missed = max_missed
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next = 1

    def update(self, t: float, pts: np.ndarray, areas: np.ndarray, classes: np.ndarray):
        """pts (N,2) metres, areas (N,), classes (N,) for one frame."""
        n = len(pts)
        assigned_det: set[int] = set()
        if self.active and n:
            last = np.array([[tr.x[-1], tr.y[-1]] for tr in self.active])
            cost = np.linalg.norm(last[:, None, :] - pts[None, :, :], axis=2)
            # allow a larger jump for tracks that were missed for a few frames
            gate = self.max_dist * np.array([1 + tr.missed for tr in self.active])[:, None]
            big = 1e6
            cost_g = np.where(cost <= gate, cost, big)
            rows, cols = linear_sum_assignment(cost_g)
            matched_tracks = set()
            for r, c in zip(rows, cols):
                if cost_g[r, c] >= big:
                    continue
                tr = self.active[r]
                tr.t.append(t); tr.x.append(float(pts[c, 0])); tr.y.append(float(pts[c, 1]))
                tr.area.append(float(areas[c])); tr.cls.append(int(classes[c]))
                tr.missed = 0
                matched_tracks.add(r)
                assigned_det.add(c)
            for i, tr in enumerate(self.active):
                if i not in matched_tracks:
                    tr.missed += 1
        else:
            for tr in self.active:
                tr.missed += 1
        for c in range(n):
            if c not in assigned_det:
                self.active.append(Track(self._next, [t], [float(pts[c, 0])], [float(pts[c, 1])],
                                         [float(areas[c])], [int(classes[c])]))
                self._next += 1
        still_active = []
        for tr in self.active:
            (self.finished if tr.missed > self.max_missed else still_active).append(tr)
        self.active = still_active

    def tracks(self) -> list[Track]:
        return self.finished + self.active
