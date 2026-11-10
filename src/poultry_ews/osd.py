"""Read the on-screen-display (OSD) timestamp burned into DVR frames.

Frames exported from raw .h264 clips carry only the time of day in their file
name ("05.45.00-05.50.00RO000.h264_<export date>.png"); the recording DATE is
only in the OSD text, e.g. "2025-12-08 09:45:18" in the top-right corner.
This module reads it with Tesseract (`apt install tesseract-ocr`,
`pip install pytesseract`).

The DVR font is white with a dark outline; it is binarised and the digits are
read with a character whitelist. A result is only accepted if it parses as a
valid date and time.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_RX = re.compile(r"(20\d\d)\D{0,2}(\d\d)\D{0,2}(\d\d)\D{0,3}(\d\d)\D{0,2}(\d\d)\D{0,2}(\d\d)")

# OSD box as fractions of the frame (x0, y0, x1, y1): top-right corner
DEFAULT_BOX = (0.68, 0.0, 1.0, 0.06)


def _candidates(img: np.ndarray, box) -> list[np.ndarray]:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    crop = img[int(y0 * h):max(int(y1 * h), int(y0 * h) + 8), int(x0 * w):int(x1 * w)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    out = []
    for thr in (200, 170, 230):
        _, b = cv2.threshold(big, thr, 255, cv2.THRESH_BINARY_INV)  # dark text on white
        out.append(cv2.copyMakeBorder(b, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255))
    _, otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out.append(cv2.copyMakeBorder(otsu, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255))
    return out


def read_osd_time(image: str | Path | np.ndarray, box=DEFAULT_BOX) -> datetime | None:
    """Timestamp from the OSD text, or None if it cannot be read reliably."""
    import pytesseract  # type: ignore

    img = cv2.imread(str(image)) if not isinstance(image, np.ndarray) else image
    if img is None:
        raise IOError(f"cannot read {image}")
    votes: dict[datetime, int] = {}
    for cand in _candidates(img, box):
        txt = pytesseract.image_to_string(
            cand, config="--psm 7 -c tessedit_char_whitelist=0123456789-:. ")
        m = _RX.search(txt.replace(" ", " "))
        if not m:
            continue
        try:
            dt = datetime(*map(int, m.groups()))
        except ValueError:
            continue
        votes[dt] = votes.get(dt, 0) + 1
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: kv[1])[0]
