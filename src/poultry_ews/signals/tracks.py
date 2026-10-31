"""Track-based signals (used when a per-bird detector is configured).

From bird tracks on the floor plane we compute, per camera and clip, SUMS that
can be added across cameras and turned into pen-level metrics:

Signal 1, activity (after Hernandez et al., Poultry Science 2026):
  inactive_frac   share of 10-s track segments whose net speed is below
                  `inactive_speed_m_s` (proportion of inactivity)
  active_speed    mean net speed of the active segments (m/s)

Signal 2, distribution:
  count_frac_<zone>     share of bird detections in feeder / drinker / wall / corner / open
  resource_count_frac   feeder + drinker
  periphery_count_frac  wall + corner
  feeder_time_frac      mean share of time each bird (track >= 20 s) spends in a
                        feeder zone. Time near food was the strongest difference
                        between infected and healthy birds in Scheidwasser et al.
                        (R. Soc. Open Sci. 2025)
  bird_count            mean number of birds detected per frame

Signal 3, mortality: tracks that cover most of the clip without moving, and
tracks mostly classified as "dead", become still points for the
cross-clip candidate tracker.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..detection import CLASS_DEAD
from ..geometry import FEEDER, ZONE_NAMES
from ..tracking import Track


def zone_of(zone_map: np.ndarray, ppm: float, x: float, y: float) -> int:
    r = min(max(int(y * ppm), 0), zone_map.shape[0] - 1)
    c = min(max(int(x * ppm), 0), zone_map.shape[1] - 1)
    return int(zone_map[r, c])


def _segments(tr: Track, segment_s: float) -> list[float]:
    """Net speed (m/s) of consecutive windows of `segment_s` seconds."""
    if len(tr.t) < 2:
        return []
    t = np.asarray(tr.t)
    xy = tr.xy()
    out = []
    start = 0
    for i in range(1, len(t)):
        if t[i] - t[start] >= segment_s:
            d = float(np.linalg.norm(xy[i] - xy[start]))
            out.append(d / (t[i] - t[start]))
            start = i
    return out


def compute_track_stats(tracks: list[Track], n_frames: int, n_dets: int, zone_counts: dict[int, int],
                        clip_s: float, zone_map: np.ndarray, ppm: float, *,
                        segment_s: float = 10.0, inactive_speed_m_s: float = 0.02,
                        min_track_s: float = 20.0, still_radius_m: float = 0.08,
                        min_presence_frac: float = 0.9) -> dict[str, Any]:
    seg_total = seg_inactive = 0
    active_speed_sum = 0.0
    feeder_time_sum = 0.0
    n_tracks_q = 0
    still_points: list[list[float | str]] = []

    for tr in tracks:
        for v in _segments(tr, segment_s):
            seg_total += 1
            if v < inactive_speed_m_s:
                seg_inactive += 1
            else:
                active_speed_sum += v
        if tr.duration >= min_track_s:
            zones = [zone_of(zone_map, ppm, x, y) for x, y in zip(tr.x, tr.y)]
            feeder_time_sum += sum(1 for z in zones if z == FEEDER) / len(zones)
            n_tracks_q += 1
        # still / dead points for the mortality signal
        if len(tr.t) >= 3:
            xy = tr.xy()
            med = np.median(xy, axis=0)
            dead_share = sum(1 for c in tr.cls if c == CLASS_DEAD) / len(tr.cls)
            area = float(np.median(tr.area))
            if dead_share >= 0.5:
                still_points.append([float(med[0]), float(med[1]), area, "dead"])
            elif clip_s > 0 and tr.duration >= min_presence_frac * clip_s and \
                    float(np.max(np.linalg.norm(xy - med, axis=1))) <= still_radius_m:
                still_points.append([float(med[0]), float(med[1]), area, "still"])

    return {"n_frames": n_frames, "n_dets": n_dets,
            "zone_counts": {ZONE_NAMES[k]: int(v) for k, v in zone_counts.items()},
            "seg_total": seg_total, "seg_inactive": seg_inactive,
            "active_speed_sum": active_speed_sum, "feeder_time_sum": feeder_time_sum,
            "n_tracks_q": n_tracks_q, "still_points": still_points}


def metrics_from_stats(stats: list[dict[str, Any]]) -> dict[str, float | None]:
    """Pen-level metrics from the per-camera sums of one slot."""
    out: dict[str, float | None] = {}
    if not stats:
        return out
    n_dets = sum(s["n_dets"] for s in stats)
    out["bird_count"] = sum(s["n_dets"] / s["n_frames"] for s in stats if s["n_frames"]) \
        if any(s["n_frames"] for s in stats) else None
    zones: dict[str, int] = {}
    for s in stats:
        for k, v in s["zone_counts"].items():
            zones[k] = zones.get(k, 0) + v
    for name in ZONE_NAMES.values():
        out[f"count_frac_{name}"] = zones.get(name, 0) / n_dets if n_dets else None
    if n_dets:
        out["resource_count_frac"] = (zones.get("feeder", 0) + zones.get("drinker", 0)) / n_dets
        out["periphery_count_frac"] = (zones.get("wall", 0) + zones.get("corner", 0)) / n_dets
    else:
        out["resource_count_frac"] = out["periphery_count_frac"] = None
    seg = sum(s["seg_total"] for s in stats)
    ina = sum(s["seg_inactive"] for s in stats)
    out["inactive_frac"] = ina / seg if seg else None
    act = seg - ina
    out["active_speed"] = sum(s["active_speed_sum"] for s in stats) / act if act else None
    nq = sum(s["n_tracks_q"] for s in stats)
    out["feeder_time_frac"] = sum(s["feeder_time_sum"] for s in stats) / nq if nq else None
    return out
