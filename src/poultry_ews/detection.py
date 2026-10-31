"""Per-bird detection sources (stage 2 and 3).

Every detector returns an (N, 6) float array per frame:
    x1, y1, x2, y2, confidence, class_id
in WORKING image pixels. Class ids follow the dataset built by
`pews build-dataset`: 0 = broiler, 1 = dead (optional class).

Sources
  YoloDetector         a YOLO model run inline on the sampled frames
                       (weights fine-tuned with `pews train`, or the MVBroTrack
                       models yolo11x_1920.pt / yolov8x_1920.pt)
  MVBroTrackTxtDetections
                       per-frame text files written by the MVBroTrack code
                       (single_camera_detection/run_inference_functions.py,
                       process_video_with_object_detection): a folder
                       <clip>.dets/ with 000001.txt, 000002.txt, ... one line
                       per box: "cls cx cy w h conf" in full-resolution pixels
  JsonlDetections      <clip>.dets.jsonl, one JSON object per frame:
                       {"frame": 0, "boxes": [[x1, y1, x2, y2], ...]} full-res pixels
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

EMPTY = np.zeros((0, 6), np.float32)
CLASS_BROILER, CLASS_DEAD = 0, 1


class Detector(Protocol):
    def __call__(self, frame_bgr: np.ndarray, frame_index: int = 0) -> np.ndarray: ...


def boxes_to_mask(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Filled ellipses inside the boxes -> uint8 mask (0/255)."""
    m = np.zeros(shape[:2], np.uint8)
    for x1, y1, x2, y2, *_ in boxes:
        c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        axes = (max(1, int((x2 - x1) / 2)), max(1, int((y2 - y1) / 2)))
        cv2.ellipse(m, c, axes, 0, 0, 360, 255, -1)
    return m


class YoloDetector:
    """Ultralytics YOLO model (detect or segment). Imported lazily."""

    def __init__(self, weights: str | Path, conf: float = 0.35, imgsz: int = 1280,
                 device: str | None = None):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("YoloDetector needs `pip install ultralytics`") from e
        if not Path(weights).exists():
            raise FileNotFoundError(f"model weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def __call__(self, frame_bgr: np.ndarray, frame_index: int = 0) -> np.ndarray:  # pragma: no cover
        res = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return EMPTY
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()[:, None]
        cls = res.boxes.cls.cpu().numpy()[:, None]
        return np.hstack([xyxy, conf, cls]).astype(np.float32)


class _PerFrameFile:
    """Shared logic: boxes stored per original frame index; nearest earlier frame is reused."""

    def __init__(self, by_frame: dict[int, np.ndarray], scale: float):
        self.by_frame = by_frame
        self.keys = sorted(by_frame)
        self.scale = scale

    def __call__(self, frame_bgr: np.ndarray, frame_index: int = 0) -> np.ndarray:
        if not self.keys:
            return EMPTY
        i = int(np.searchsorted(self.keys, frame_index, side="right")) - 1
        b = self.by_frame[self.keys[max(i, 0)]].copy()
        b[:, :4] *= self.scale
        return b


class MVBroTrackTxtDetections(_PerFrameFile):
    def __init__(self, clip_path: Path, full_width: int, work_width: int, suffix: str = ".dets",
                 min_conf: float = 0.0):
        folder = Path(str(clip_path) + suffix)
        if not folder.is_dir():
            raise FileNotFoundError(f"MVBroTrack detection folder not found: {folder}")
        by_frame: dict[int, np.ndarray] = {}
        for f in folder.glob("*.txt"):
            try:
                idx = int(f.stem) - 1  # MVBroTrack numbers frames from 1
            except ValueError:
                continue
            by_frame[idx] = parse_mvbrotrack_txt(f.read_text(), min_conf)
        super().__init__(by_frame, work_width / full_width)


class JsonlDetections(_PerFrameFile):
    def __init__(self, clip_path: Path, full_width: int, work_width: int,
                 suffix: str = ".dets.jsonl"):
        f = Path(str(clip_path) + suffix)
        if not f.exists():
            raise FileNotFoundError(f"external detections not found: {f}")
        by_frame: dict[int, np.ndarray] = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows = []
            for b in rec.get("boxes", []):
                x1, y1, x2, y2 = b[:4]
                conf = b[4] if len(b) > 4 else 1.0
                cls = b[5] if len(b) > 5 else CLASS_BROILER
                rows.append([x1, y1, x2, y2, conf, cls])
            by_frame[int(rec["frame"])] = np.asarray(rows, np.float32).reshape(-1, 6)
        super().__init__(by_frame, work_width / full_width)


def parse_mvbrotrack_txt(text: str, min_conf: float = 0.0) -> np.ndarray:
    """'cls cx cy w h conf' lines (pixel units) -> (N,6) x1,y1,x2,y2,conf,cls."""
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
        conf = float(p[5]) if len(p) > 5 else 1.0
        if conf < min_conf:
            continue
        rows.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf, cls])
    return np.asarray(rows, np.float32).reshape(-1, 6)


def build_detector(cfg_seg: dict, clip_path: Path | None, full_width: int, work_width: int,
                   root: Path | None = None) -> Detector | None:
    """Return a detector for kinds yolo / mvbrotrack / external, or None for 'color'."""
    kind = cfg_seg.get("kind", "color")
    if kind == "color":
        return None
    if kind == "yolo":
        w = Path(cfg_seg["weights"])
        if root is not None and not w.is_absolute():
            w = root / w
        imgsz = int(cfg_seg.get("imgsz", 1280))
        return YoloDetector(w, conf=float(cfg_seg.get("conf", 0.35)), imgsz=imgsz,
                            device=cfg_seg.get("device"))
    if clip_path is None:
        raise ValueError(f"detector kind {kind!r} needs the clip path")
    if kind == "mvbrotrack":
        return MVBroTrackTxtDetections(clip_path, full_width, work_width,
                                       cfg_seg.get("mvbrotrack_suffix", ".dets"),
                                       float(cfg_seg.get("conf", 0.0)))
    if kind == "external":
        return JsonlDetections(clip_path, full_width, work_width,
                               cfg_seg.get("external_suffix", ".dets.jsonl"))
    raise ValueError(f"unknown detector kind {kind!r}")
