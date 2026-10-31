"""Stage 1 bird segmentation (no training): which pixels are birds?

White/yellow birds are brighter and less saturated than the litter. Good
enough for flock-level activity and distribution; it cannot separate
individual birds in huddles. Per-bird detectors (stage 2/3) live in
detection.py and are used instead when `analysis.segmenter.kind` is
yolo / mvbrotrack / external.

The segmenter returns a uint8 mask (0/255) in WORKING image coordinates.
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class Segmenter(Protocol):
    def __call__(self, frame_bgr: np.ndarray, frame_index: int = 0) -> np.ndarray: ...


class ColorSegmenter:
    def __init__(self, contrast: float = 18, v_min: int = 140, s_max: int = 120,
                 bg_kernel_frac: float = 0.09, open_px: int = 3):
        self.contrast = contrast
        self.v_min = v_min
        self.s_max = s_max
        self.bg_kernel_frac = bg_kernel_frac
        self.open_px = open_px

    def __call__(self, frame_bgr: np.ndarray, frame_index: int = 0) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        L = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)[..., 0]
        k = max(int(w * self.bg_kernel_frac) | 1, 3)
        # median of a downscaled image ~ local floor brightness, cheap even for big kernels
        small = cv2.resize(L, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        bg = cv2.medianBlur(small, max(3, (k // 4) | 1))
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
        Hc, S, V = cv2.split(hsv)
        bright = (L.astype(np.int16) - bg.astype(np.int16)) > self.contrast
        pale = (V >= self.v_min) & (S <= self.s_max)
        red = ((Hc < 12) | (Hc > 165)) & (S > 110)
        m = (bright & pale & ~red).astype(np.uint8) * 255
        if self.open_px > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((self.open_px, self.open_px), np.uint8))
        return m


def build_segmenter(cfg_seg: dict, **_ignored) -> Segmenter:
    """Colour segmenter from the `analysis.segmenter` config block."""
    return ColorSegmenter(contrast=cfg_seg.get("contrast", 18),
                          v_min=cfg_seg.get("v_min", 140), s_max=cfg_seg.get("s_max", 120))
