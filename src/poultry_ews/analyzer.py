"""Analyse ONE clip from ONE camera -> per-camera result for one time slot.

Two modes, chosen by `analysis.segmenter.kind`:
  color                      colour segmentation (stage 1, no training)
  yolo | mvbrotrack | external  per-bird detections (stage 2/3): additionally
                             tracks birds on the floor plane and computes the
                             track-based metrics of signals/tracks.py
Optical flow (signal 1 fallback) and the occupancy map (signal 2) are computed
in both modes; in detection mode the bird mask comes from the boxes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import CameraConfig, Config
from .detection import Detector, boxes_to_mask, build_detector
from .geometry import CameraGeometry, FloorCanvas
from .segmentation import Segmenter, build_segmenter
from .signals.activity import ActivityAccumulator, flow_speed
from .signals.mortality import StillBlob, StillnessAccumulator
from .signals.tracks import compute_track_stats, zone_of
from .tracking import FloorTracker
from .video import effective_fps, iter_frames


@dataclass
class CameraClipResult:
    camera: str
    pen: str
    start: datetime
    duration_s: float
    n_frames: int
    bird_px: float
    moving_px: float
    speed_sum: float
    occupancy: np.ndarray          # float32 floor map, mean bird presence 0..1
    coverage: np.ndarray           # bool floor map
    blobs: list[StillBlob] = field(default_factory=list)
    snapshot: np.ndarray | None = None   # BGR working-size frame (last frame of the clip)
    geometry: CameraGeometry | None = None
    track_stats: dict[str, Any] | None = None   # detection mode only


def camera_geometry(cfg: Config, cam: CameraConfig, when: datetime) -> CameraGeometry:
    pen = cfg.pen(cam.pen)
    canvas = FloorCanvas(pen.width_m, pen.length_m, float(cfg.analysis["floor_px_per_m"]))
    return CameraGeometry(cam.calibration_for(when), canvas, int(cfg.analysis["work_width"]))


def _box_floor(geo: CameraGeometry, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Box centres -> floor metres (N,2); box footprint area on the floor (N,) m^2."""
    if len(boxes) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    centres = np.column_stack([(x1 + x2) / 2, (y1 + y2) / 2]).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(centres.astype(np.float64), geo.H_img_to_m).reshape(-1, 2)
    corners = np.stack([np.column_stack([x1, y1]), np.column_stack([x2, y1]),
                        np.column_stack([x2, y2]), np.column_stack([x1, y2])], axis=1)
    cm = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float64),
                                  geo.H_img_to_m).reshape(-1, 4, 2)
    xs, ys = cm[..., 0], cm[..., 1]
    area = 0.5 * np.abs(np.sum(xs * np.roll(ys, -1, axis=1) - ys * np.roll(xs, -1, axis=1), axis=1))
    return pts, area


def analyze_clip(cfg: Config, camera_id: str, clip: str | Path, start: datetime,
                 segmenter: Segmenter | Detector | None = None,
                 max_seconds: float | None = None, detector: Detector | None = None
                 ) -> CameraClipResult:
    cam = cfg.camera(camera_id)
    a = cfg.analysis
    geo = camera_geometry(cfg, cam, start)
    ppm = geo.canvas.ppm
    fps = effective_fps(clip, float(a["analysis_fps"]), cam.source_fps)
    seg_cfg = a["segmenter"]
    if detector is None and segmenter is None:
        detector = build_detector(seg_cfg, Path(clip), geo.cal.image_size[0],
                                  int(a["work_width"]), cfg.root)
    if detector is None and segmenter is None:
        segmenter = build_segmenter(seg_cfg)

    act = ActivityAccumulator(float(a["activity"]["moving_speed_m_s"]))
    still = StillnessAccumulator(geo.canvas.shape, float(a["mortality"]["still_speed_m_s"]))
    occ_sum = np.zeros(geo.canvas.shape, np.float32)
    cov = geo.coverage
    roi = geo.image_mask
    pen = cfg.pen(cam.pen).at(start)
    zone_map = geo.canvas.zone_map(pen)

    tr_cfg = a.get("tracking", {})
    tracker = FloorTracker(float(tr_cfg.get("max_dist_m", 0.35)),
                           int(tr_cfg.get("max_missed", 4))) if detector is not None else None
    n_dets = 0
    zone_counts: dict[int, int] = {}

    prev_gray = None
    n = 0
    t_last = 0.0
    snapshot = None
    for fr in iter_frames(clip, float(a["analysis_fps"]), int(a["work_width"]),
                          cam.source_fps, max_seconds):
        img = fr.image
        if img.shape[1] != geo.work_size[0] or img.shape[0] != geo.work_size[1]:
            img = cv2.resize(img, geo.work_size, interpolation=cv2.INTER_AREA)
        if detector is not None:
            boxes = detector(img, fr.index)
            if len(boxes):
                cx = ((boxes[:, 0] + boxes[:, 2]) / 2).astype(int).clip(0, roi.shape[1] - 1)
                cy = ((boxes[:, 1] + boxes[:, 3]) / 2).astype(int).clip(0, roi.shape[0] - 1)
                boxes = boxes[roi[cy, cx] > 0]       # ignore boxes outside the ROI / on equipment
            mask = cv2.bitwise_and(boxes_to_mask(boxes, roi.shape), roi)
            pts, areas = _box_floor(geo, boxes)
            tracker.update(fr.t_sec, pts, areas, boxes[:, 5] if len(boxes) else np.zeros(0))
            n_dets += len(pts)
            for x, y in pts:
                z = zone_of(zone_map, ppm, x, y)
                zone_counts[z] = zone_counts.get(z, 0) + 1
        else:
            mask = cv2.bitwise_and(segmenter(img, fr.index), roi)
        bird = (geo.to_floor(mask, nearest=True) > 127) & cov
        gray = geo.to_floor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        speed = None
        if prev_gray is not None:
            speed = flow_speed(prev_gray, gray, ppm, fps)
            act.update(speed, bird)
        still.update(bird, speed)
        occ_sum += bird
        prev_gray = gray
        n += 1
        t_last = fr.t_sec
        snapshot = img  # last frame of the clip = most recent view
    if n == 0:
        raise IOError(f"no frames read from {clip}")

    age = cfg.age_days(start)
    m = a["mortality"]
    duration = t_last + 1.0 / fps if n > 1 else 0.0
    track_stats = None
    if tracker is not None:
        track_stats = compute_track_stats(
            tracker.tracks(), n, n_dets, zone_counts, t_last, zone_map, ppm,
            segment_s=float(tr_cfg.get("segment_s", 10)),
            inactive_speed_m_s=float(tr_cfg.get("inactive_speed_m_s", 0.02)),
            min_track_s=float(tr_cfg.get("min_track_s", 20)),
            still_radius_m=float(tr_cfg.get("still_radius_m", 0.08)),
            min_presence_frac=float(m["min_presence_frac"]))
        blobs = [StillBlob(x, y, ar, kind) for x, y, ar, kind in track_stats["still_points"]]
    else:
        blobs = still.blobs(ppm, cfg.bird_area_m2(age), float(m["min_presence_frac"]),
                            float(m["max_moving_frac"]), float(m["min_area_factor"]),
                            float(m["max_area_factor"]))
    return CameraClipResult(
        camera=camera_id, pen=cam.pen, start=start, duration_s=duration, n_frames=n,
        bird_px=act.bird_px, moving_px=act.moving_px, speed_sum=act.speed_sum,
        occupancy=occ_sum / n, coverage=cov, blobs=blobs, snapshot=snapshot, geometry=geo,
        track_stats=track_stats)
