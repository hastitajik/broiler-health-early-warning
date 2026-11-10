"""Fine-tuning workflow for the per-bird detector (stage 2).

    1. pews sample-frames   pick ~300 frames spread over cameras, ages and day/night
    2. pews prelabel        draft boxes with an existing model (e.g. MVBroTrack's
                            yolo11x_1920.pt) so annotators only correct them
    3. (annotate)           fix the boxes in CVAT or Label Studio, export YOLO format
    4. pews build-dataset   split by DAY (no near-duplicate frames across splits),
                            stratified by growth phase, + per-phase test sets
    5. pews train           fine-tune with ultralytics
    6. pews evaluate        precision / recall / mAP on the test set, per phase

Growth phases follow the MVBroTrack code (single_camera_detection/
generate_yolo_datasets.py): starter < day 11, grower day 11-23, finisher >= day 24.
"""
from __future__ import annotations

import csv
import json
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2

from .config import Config
from .pipeline import find_clips
from .timeparse import TimestampError, parse_clip_time
from .video import IMAGE_EXT

PHASES = ("starter", "grower", "finisher")
DEFAULT_CLASSES = ["broiler", "dead"]
MANIFEST = "manifest.csv"
MVBROTRACK_WEIGHTS = Path("DATA/MVBroTrack/Single-View-Detection/trained_models/yolo11x_1920.pt")


def phase_of(age_days: int) -> str:
    return "starter" if age_days < 11 else "grower" if age_days < 24 else "finisher"


# --------------------------------------------------------------------------- 1. sampling
@dataclass
class FrameSpec:
    camera: str
    clip: Path
    start: datetime
    offset_s: float


def sample_frames(cfg: Config, out_dir: Path, n_total: int = 300, seed: int = 42,
                  cameras: list[str] | None = None, age_bucket_days: int = 3) -> list[dict[str, Any]]:
    """Pick frames evenly across (camera, age bucket, lights on/off) and save them as JPG.

    Writes <out_dir>/images/*.jpg and <out_dir>/manifest.csv.
    """
    rng = random.Random(seed)
    groups: dict[tuple, list[FrameSpec]] = defaultdict(list)
    for cam in cfg.cameras:
        if cameras and cam.id not in cameras:
            continue
        for clip in find_clips([cam.input_dir]):
            try:
                start = parse_clip_time(clip, cam.filename_regex)
            except TimestampError:
                continue
            age = cfg.age_days(start)
            key = (cam.id, age // age_bucket_days, cfg.lights_on(start))
            groups[key].append(FrameSpec(cam.id, clip, start, 0.0))
    if not groups:
        raise FileNotFoundError("no clips found in the cameras' input_dir folders")

    for g in groups.values():
        rng.shuffle(g)
    picked: list[FrameSpec] = []
    keys = sorted(groups)
    while len(picked) < n_total and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(picked) < n_total:
                picked.append(groups[k].pop())

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in picked:
        frame, offset = _read_frame(spec.clip, rng)
        if frame is None:
            continue
        ts = spec.start + timedelta(seconds=offset)
        name = f"{spec.camera}_{ts:%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(img_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        age = cfg.age_days(ts)
        rows.append({"image": name, "camera": spec.camera, "clip": str(spec.clip),
                     "ts": ts.isoformat(timespec="seconds"), "age_days": age,
                     "phase": phase_of(age), "lights_on": int(cfg.lights_on(ts))})
    _write_manifest(out_dir / MANIFEST, rows, append=True)
    return rows


def _read_frame(clip: Path, rng: random.Random):
    if clip.suffix.lower() in IMAGE_EXT:
        return cv2.imread(str(clip)), 0.0
    cap = cv2.VideoCapture(str(clip))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        if n > 10:
            idx = rng.randint(int(n * 0.1), int(n * 0.9))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                return frame, idx / fps
        # raw .h264 streams often cannot seek: read forward a random number of frames
        cap.release()
        cap = cv2.VideoCapture(str(clip))
        skip = rng.randint(0, 300)
        frame, k = None, 0
        while k <= skip:
            ok, f = cap.read()
            if not ok:
                break
            frame, k = f, k + 1
        return frame, max(k - 1, 0) / fps
    finally:
        cap.release()


def _write_manifest(path: Path, rows: list[dict], append: bool):
    fields = ["image", "camera", "clip", "ts", "age_days", "phase", "lights_on", "ts_source"]
    exists = path.exists() and append
    with open(path, "a" if exists else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["image"]: r for r in csv.DictReader(f)}


# --------------------------------------------------------------------------- 1b. import existing frames
def import_frames(src_dirs: list[Path], out_dir: Path, placement_date: str, camera: str = "cam1",
                  cfg: Config | None = None, use_osd: bool = True) -> dict[str, Any]:
    """Copy already-exported frames into <out_dir>/images and write the manifest.

    Recording time: OSD text burned into the frame (osd.py) first, file name as
    fallback. Frames are renamed <camera>_<YYYYmmdd_HHMMSS>.<ext> so the day is
    visible; the original name is kept in the manifest `clip` column. Any
    existing train/val/test folder structure is ignored on purpose: the split is
    redone by day in `build-dataset`.
    """
    from .osd import read_osd_time
    placed = datetime.strptime(placement_date, "%Y-%m-%d")
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows, unreadable, mismatched = [], [], []
    seen: set[str] = set()
    files = sorted(f for d in src_dirs for f in Path(d).rglob("*")
                   if f.is_file() and f.suffix.lower() in IMAGE_EXT)
    for f in files:
        ts, src = None, ""
        if use_osd:
            try:
                ts = read_osd_time(f)
                src = "osd" if ts else ""
            except Exception:  # pragma: no cover - tesseract missing / unreadable image
                ts = None
        try:
            ts_name = parse_clip_time(f)
        except (TimestampError, ValueError):
            ts_name = None
        if ts_name is not None and ts is not None:
            # names with only a time range ("05.45.00-05.50.00") have no date: compare times of day
            d = abs((datetime.combine(ts.date(), ts_name.time()) - ts).total_seconds())
            if d > 15 * 60 and abs(d - 86400) > 15 * 60:
                mismatched.append(f.name)
        if ts is None and ts_name is not None and ts_name.year >= 2000 and not \
                re.match(r"\d\d[._]\d\d[._]\d\d-", f.name):
            ts, src = ts_name, "filename"
        if ts is None:
            unreadable.append(f.name)
            continue
        base = f"{camera}_{ts:%Y%m%d_%H%M%S}"
        name, k = base + f.suffix.lower(), 1
        while name in seen:
            name, k = f"{base}_{k}{f.suffix.lower()}", k + 1
        seen.add(name)
        shutil.copy2(f, img_dir / name)
        age = (ts.date() - placed.date()).days
        rows.append({"image": name, "camera": camera, "clip": f.name,
                     "ts": ts.isoformat(timespec="seconds"), "age_days": age, "phase": phase_of(age),
                     "lights_on": int(cfg.lights_on(ts)) if cfg else "", "ts_source": src})
    _write_manifest(out_dir / MANIFEST, rows, append=False)
    days = sorted({r["ts"][:10] for r in rows})
    phases: dict[str, int] = {}
    for r in rows:
        phases[r["phase"]] = phases.get(r["phase"], 0) + 1
    return {"imported": len(rows), "unreadable": unreadable, "time_mismatch": mismatched,
            "days": days, "phases": phases}


# --------------------------------------------------------------------------- 2. pre-labelling
def prelabel(weights: Path, images_dir: Path, labels_dir: Path, conf: float = 0.25,
             imgsz: int = 1920, overwrite: bool = False, previews_dir: Path | None = None,
             device: str | None = None) -> int:  # pragma: no cover - needs ultralytics
    """Draft YOLO labels (normalised 'cls cx cy w h') for every image without a label yet.

    With `previews_dir`, also writes a JPG per image with the drafted boxes drawn,
    to judge the pre-trained model at a glance before correcting.
    """
    from ultralytics import YOLO  # type: ignore
    model = YOLO(str(weights))
    labels_dir.mkdir(parents=True, exist_ok=True)
    # class list for labelling tools (LabelImg, Label Studio converters)
    (labels_dir / "classes.txt").write_text("\n".join(DEFAULT_CLASSES) + "\n")
    n = 0
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXT:
            continue
        out = labels_dir / (img.stem + ".txt")
        if out.exists() and not overwrite:
            continue
        res = model.predict(str(img), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        lines = []
        if res.boxes is not None:
            for (cx, cy, w, h), c in zip(res.boxes.xywhn.tolist(), res.boxes.cls.tolist()):
                # MVBroTrack models have a single class; everything becomes "broiler" (0)
                lines.append(f"{0 if c != 1 else 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        out.write_text("\n".join(lines))
        if previews_dir is not None:
            draw_labels(img, out, previews_dir / (img.stem + ".jpg"))
        n += 1
    return n


def draw_labels(image: Path, label: Path, out: Path, width: int = 1600):
    """Draw YOLO boxes (green = broiler, red = dead) on an image and save a JPG."""
    im = cv2.imread(str(image))
    h, w = im.shape[:2]
    n = 0
    for line in label.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255) if c == 1 else (0, 220, 0), 2)
        n += 1
    cv2.putText(im, f"{n} boxes", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
    s = width / w
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(im, (width, int(h * s))), [cv2.IMWRITE_JPEG_QUALITY, 85])


def export_cvat(images_dir: Path, labels_dir: Path, out_zip: Path,
                classes: list[str] | None = None) -> int:
    """Pack images + YOLO labels as a CVAT "YOLO 1.1" archive (import as annotations).

    In CVAT: create a task with the same images, then Actions > Upload annotations >
    YOLO 1.1 > this zip. After correcting: Export dataset > YOLO 1.1, and unzip the
    .txt files into a labels folder for `build-dataset`.
    """
    import zipfile
    classes = classes or DEFAULT_CLASSES
    imgs = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXT)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(classes) + "\n")
        z.writestr("obj.data", f"classes = {len(classes)}\nnames = data/obj.names\n"
                               "train = data/train.txt\n")
        z.writestr("train.txt", "\n".join(f"data/obj_train_data/{f.name}" for f in imgs) + "\n")
        for f in imgs:
            lab = labels_dir / (f.stem + ".txt")
            z.writestr(f"obj_train_data/{f.stem}.txt", lab.read_text() if lab.exists() else "")
    return len(imgs)


# --------------------------------------------------------------------------- 4. dataset
def build_dataset(images_dir: Path, labels_dir: Path, manifest: Path, out_dir: Path,
                  classes: list[str] | None = None, test_frac: float = 0.2, val_frac: float = 0.15,
                  seed: int = 42) -> dict[str, Any]:
    """Split labelled frames by day within each growth phase and write YOLO dataset files."""
    classes = classes or DEFAULT_CLASSES
    meta = read_manifest(manifest)
    items = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXT:
            continue
        lab = labels_dir / (img.stem + ".txt")
        if not lab.exists():
            continue  # not annotated yet
        m = meta.get(img.name)
        if m is None:
            continue
        items.append({"img": img, "lab": lab, "day": m["ts"][:10], "phase": m["phase"],
                      "camera": m["camera"]})
    if len(items) < 10:
        raise ValueError(f"only {len(items)} labelled images found; label more frames first")

    rng = random.Random(seed)
    split_of: dict[str, str] = {}
    for ph in PHASES:
        ph_items = [it for it in items if it["phase"] == ph]
        days = sorted({it["day"] for it in ph_items})
        rng.shuffle(days)
        n = len(ph_items)
        per_day = {d: sum(1 for it in ph_items if it["day"] == d) for d in days}
        if len(days) >= 3:
            acc = 0
            for d in days:
                if acc < test_frac * n:
                    split_of[(ph, d)] = "test"
                elif acc < (test_frac + val_frac) * n:
                    split_of[(ph, d)] = "val"
                else:
                    split_of[(ph, d)] = "train"
                acc += per_day[d]
            if "train" not in {split_of[(ph, d)] for d in days}:
                split_of[(ph, days[-1])] = "train"
        else:  # too few days in this phase: split by image instead
            idx = list(range(n))
            rng.shuffle(idx)
            for rank, i in enumerate(idx):
                ph_items[i]["split"] = ("test" if rank < test_frac * n else
                                        "val" if rank < (test_frac + val_frac) * n else "train")
    for it in items:
        it.setdefault("split", split_of.get((it["phase"], it["day"]), "train"))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sp in ("train", "val", "test"):
        (out_dir / "images" / sp).mkdir(parents=True)
        (out_dir / "labels" / sp).mkdir(parents=True)
    for it in items:
        shutil.copy2(it["img"], out_dir / "images" / it["split"] / it["img"].name)
        shutil.copy2(it["lab"], out_dir / "labels" / it["split"] / it["lab"].name)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    base = f"path: {out_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
    (out_dir / "data.yaml").write_text(base + f"test: images/test\nnames:\n{names}\n", encoding="utf-8")
    summary: dict[str, Any] = {"classes": classes, "total": len(items), "splits": {}, "phases": {}}
    for sp in ("train", "val", "test"):
        summary["splits"][sp] = sum(1 for it in items if it["split"] == sp)
    for ph in PHASES:
        test_imgs = [it for it in items if it["split"] == "test" and it["phase"] == ph]
        summary["phases"][ph] = {sp: sum(1 for it in items if it["split"] == sp and it["phase"] == ph)
                                 for sp in ("train", "val", "test")}
        if test_imgs:
            lst = out_dir / f"test_{ph}.txt"
            lst.write_text("\n".join(f"./images/test/{it['img'].name}" for it in test_imgs) + "\n")
            (out_dir / f"test_{ph}.yaml").write_text(
                base + f"test: test_{ph}.txt\nnames:\n{names}\n", encoding="utf-8")
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- 5-6. train / evaluate
def default_base_weights(root: Path) -> str:
    p = root / MVBROTRACK_WEIGHTS
    return str(p) if p.exists() else "yolo11s.pt"


def train(data_yaml: Path, base: str, out_dir: Path, imgsz: int = 1280, epochs: int = 100,
          batch: float = 0.6, patience: int = 30, device: str | None = None,
          name: str | None = None) -> Path:  # pragma: no cover - needs ultralytics
    from ultralytics import YOLO  # type: ignore
    model = YOLO(base)
    name = name or f"{Path(base).stem}_{imgsz}_{datetime.now():%Y%m%d_%H%M}"
    model.train(data=str(data_yaml), imgsz=imgsz, epochs=epochs, batch=batch, patience=patience,
                project=str(out_dir), name=name, device=device,
                # top-down camera: vertical flips are valid, colour shifts mimic lighting changes
                flipud=0.5, fliplr=0.5, hsv_v=0.5)
    return out_dir / name / "weights" / "best.pt"


def evaluate(weights: Path, dataset_dir: Path, imgsz: int = 1280, conf: float = 0.001,
             device: str | None = None) -> dict[str, Any]:  # pragma: no cover - needs ultralytics
    from ultralytics import YOLO  # type: ignore
    model = YOLO(str(weights))
    out: dict[str, Any] = {"weights": str(weights)}
    for label, yaml_file in [("all", dataset_dir / "data.yaml")] + \
            [(ph, dataset_dir / f"test_{ph}.yaml") for ph in PHASES]:
        if not yaml_file.exists():
            continue
        m = model.val(data=str(yaml_file), split="test", imgsz=imgsz, conf=conf, device=device,
                      plots=False, verbose=False)
        out[label] = {"precision": float(m.box.mp), "recall": float(m.box.mr),
                      "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}
    return out


def evaluation_markdown(r: dict[str, Any]) -> str:
    lines = [f"# Detector evaluation", "", f"Weights: `{r['weights']}`", "",
             "| Test set | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
    for k in ("all",) + PHASES:
        if k in r:
            v = r[k]
            lines.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | "
                         f"{v['mAP50_95']:.3f} |")
    return "\n".join(lines) + "\n"
