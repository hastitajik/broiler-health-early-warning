"""Fine-tuning data preparation (no ultralytics needed)."""
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from poultry_ews.finetune import build_dataset, phase_of, read_manifest, sample_frames

from synth import make_config, moving_flock, write_video


def test_phase_of_matches_mvbrotrack():
    assert phase_of(0) == "starter" and phase_of(10) == "starter"
    assert phase_of(11) == "grower" and phase_of(23) == "grower"
    assert phase_of(24) == "finisher"


def test_sample_frames_spreads_over_ages_and_day_night(tmp_path):
    cfg = make_config(tmp_path)
    for d in (1, 5, 15, 30):
        for hh in (9, 23):
            ts = datetime(2025, 11, 1) + timedelta(days=d, hours=hh)
            folder = cfg.camera("cam1").input_dir / f"{ts:%Y-%m-%d}"
            folder.mkdir(parents=True, exist_ok=True)
            write_video(folder / f"{ts:%H.%M.%S}-{ts:%H}.05.00RO000.mp4", moving_flock(4), seconds=2)
    rows = sample_frames(cfg, tmp_path / "lab", n_total=6)
    assert len(rows) == 6
    assert {r["phase"] for r in rows} == {"starter", "grower", "finisher"}
    m = read_manifest(tmp_path / "lab" / "manifest.csv")
    assert all((tmp_path / "lab" / "images" / k).exists() for k in m)


def _fake_labelled(tmp_path, days_per_phase=4, per_day=3):
    img_dir, lab_dir = tmp_path / "images", tmp_path / "labels"
    img_dir.mkdir()
    lab_dir.mkdir()
    rows = ["image,camera,clip,ts,age_days,phase,lights_on"]
    start = datetime(2025, 11, 1)
    for age in list(range(0, days_per_phase)) + list(range(12, 12 + days_per_phase)) + \
            list(range(25, 25 + days_per_phase)):
        for k in range(per_day):
            ts = start + timedelta(days=age, hours=k)
            name = f"cam1_{ts:%Y%m%d_%H%M%S}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), np.zeros((32, 32, 3), np.uint8))
            (lab_dir / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1")
            rows.append(f"{name}.jpg,cam1,x,{ts.isoformat()},{age},{phase_of(age)},1")
    (tmp_path / "manifest.csv").write_text("\n".join(rows))
    return img_dir, lab_dir


def test_build_dataset_splits_by_day(tmp_path):
    img_dir, lab_dir = _fake_labelled(tmp_path)
    out = tmp_path / "ds"
    s = build_dataset(img_dir, lab_dir, tmp_path / "manifest.csv", out)
    assert s["total"] == 36 and s["splits"]["test"] > 0 and s["splits"]["train"] > 0
    days = {}
    for sp in ("train", "val", "test"):
        for f in (out / "images" / sp).iterdir():
            days.setdefault(f.name[5:13], set()).add(sp)
    assert all(len(v) == 1 for v in days.values()), "a day leaked into two splits"
    for ph in ("starter", "grower", "finisher"):
        assert (out / f"test_{ph}.yaml").exists()
        assert s["phases"][ph]["test"] > 0
    assert "names:\n  0: broiler\n  1: dead" in (out / "data.yaml").read_text()


def test_build_dataset_needs_labels(tmp_path):
    img_dir, lab_dir = _fake_labelled(tmp_path, days_per_phase=1, per_day=2)
    for f in list(lab_dir.iterdir())[:4]:
        f.unlink()
    with pytest.raises(ValueError):
        build_dataset(img_dir, lab_dir, tmp_path / "manifest.csv", tmp_path / "ds")


def test_import_frames_reads_time_and_renames(tmp_path):
    from poultry_ews.finetune import import_frames
    src = tmp_path / "src" / "train"
    src.mkdir(parents=True)
    img = np.zeros((72, 128, 3), np.uint8)
    cv2.imwrite(str(src / "20251116020735157_fps15.mp4_20260425_220611.666.png"), img)
    cv2.imwrite(str(src / "05.45.00-05.50.00[RO][0@0][0].h264_20260426_222529.516.png"), img)
    r = import_frames([tmp_path / "src"], tmp_path / "lab", "2025-11-13", use_osd=False)
    # the mp4 name carries the date; the h264 name only a time of day -> skipped without OSD
    assert r["imported"] == 1 and len(r["unreadable"]) == 1
    m = read_manifest(tmp_path / "lab" / "manifest.csv")
    row = m["cam1_20251116_020735.png"]
    assert row["age_days"] == "3" and row["phase"] == "starter" and row["ts_source"] == "filename"


def test_export_cvat_and_draw(tmp_path):
    import zipfile
    from poultry_ews.finetune import draw_labels, export_cvat
    img_dir, lab_dir = _fake_labelled(tmp_path, days_per_phase=1, per_day=1)
    n = export_cvat(img_dir, lab_dir, tmp_path / "cvat.zip")
    z = zipfile.ZipFile(tmp_path / "cvat.zip")
    names = z.namelist()
    assert n == 3 and "obj.names" in names and "train.txt" in names
    assert sum(1 for x in names if x.startswith("obj_train_data/")) == 3
    first = sorted(img_dir.iterdir())[0]
    draw_labels(first, lab_dir / (first.stem + ".txt"), tmp_path / "p.jpg", width=64)
    assert (tmp_path / "p.jpg").exists()


def test_osd_reader_on_sample_frame():
    pytest.importorskip("pytesseract")
    import shutil as _sh
    if _sh.which("tesseract") is None:
        pytest.skip("tesseract not installed")
    from poultry_ews.osd import read_osd_time
    p = Path(__file__).resolve().parents[1] / "farm_data" / "samples" / "cam1_2025-12-09_054754.jpg"
    assert read_osd_time(p) == datetime(2025, 12, 9, 5, 47, 54)
