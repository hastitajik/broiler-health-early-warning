"""Command line: pews <command> --config configs/farm.yaml ...

  process       analyse clips (files or folders) of one camera, in time order
  watch         keep polling every camera's input_dir and process new clips
  serve         start the web panel / API
  preview       draw calibration + segmentation on a frame (check before trusting signals)
  calibrate     click calibration points on a frame (needs a desktop OpenCV)
  to-metres     convert image pixels to floor metres (for feeder / drinker positions)
  extract-frames  sample frames from clips for labelling (stage 2 detector)
  log-mortality   record daily deaths / culls
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path


def _cfg(args):
    from .config import load_config
    return load_config(args.config, getattr(args, "rules", None))


def cmd_process(args):
    from .pipeline import Pipeline, find_clips, process_many
    cfg = _cfg(args)
    pipe = Pipeline(cfg)
    clips = find_clips([Path(p) for p in args.paths])
    if not clips:
        print("no video/image files found", file=sys.stderr)
        return 1
    ok, err = process_many(pipe, args.camera, clips, args.max_seconds)
    print(f"processed {ok} clips, {err} errors -> {cfg.db_path}")
    return 0 if err == 0 else 2


def cmd_watch(args):
    from .pipeline import Pipeline, watch
    cfg = _cfg(args)
    watch(Pipeline(cfg), poll_s=args.poll, once=args.once)
    return 0


def cmd_serve(args):
    import uvicorn
    from .api.app import create_app
    uvicorn.run(create_app(_cfg(args)), host=args.host, port=args.port)
    return 0


def cmd_preview(args):
    import cv2
    from .preview import render_preview
    from .timeparse import TimestampError, parse_clip_time
    cfg = _cfg(args)
    when = datetime.strptime(args.date, "%Y-%m-%d") if args.date else None
    if when is None:
        try:
            when = parse_clip_time(args.image)
        except TimestampError:
            when = datetime.now()
    out = render_preview(cfg, args.camera, args.image, when)
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out} (calibration valid on {when.date()})")
    return 0


def cmd_calibrate(args):
    from .calibrate import run_calibration
    cfg = _cfg(args)
    cam = cfg.camera(args.camera)
    pen = cfg.pen(cam.pen)
    run_calibration(args.image, args.out or cam.calibration_file, cam.id, pen.width_m, pen.length_m,
                    datetime.strptime(args.valid_from, "%Y-%m-%d").date())
    return 0


def cmd_to_metres(args):
    import numpy as np
    from .analyzer import camera_geometry
    cfg = _cfg(args)
    geo = camera_geometry(cfg, cfg.camera(args.camera), datetime.strptime(args.date, "%Y-%m-%d"))
    pts = np.array([[float(v) for v in p.split(",")] for p in args.px])
    for p, m in zip(pts, geo.image_full_to_m(pts)):
        print(f"px ({p[0]:.0f}, {p[1]:.0f}) -> x={m[0]:.2f} m, y={m[1]:.2f} m")
    return 0


def cmd_extract(args):
    import cv2
    from .pipeline import find_clips
    from .video import iter_frames
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for clip in find_clips([Path(p) for p in args.paths]):
        for fr in iter_frames(clip, 1.0 / args.every, args.width, fallback_fps=args.fps):
            name = out / f"{clip.stem}_{int(fr.t_sec):05d}s.jpg"
            cv2.imwrite(str(name), fr.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
    print(f"wrote {n} frames to {out}")
    return 0


def cmd_sample(args):
    from .finetune import sample_frames
    cfg = _cfg(args)
    out = Path(args.out)
    rows = sample_frames(cfg, out, args.n, args.seed, args.camera or None)
    phases = {}
    for r in rows:
        phases[r["phase"]] = phases.get(r["phase"], 0) + 1
    print(f"wrote {len(rows)} frames to {out / 'images'} ({phases}); manifest: {out / 'manifest.csv'}")
    return 0


def cmd_prelabel(args):
    from .finetune import default_base_weights, export_cvat, prelabel
    root = Path(args.config).resolve().parent.parent if Path(args.config).exists() else Path.cwd()
    weights = args.weights or default_base_weights(root)
    d = Path(args.dir)
    n = prelabel(Path(weights), d / "images", d / "labels", args.conf, args.imgsz, args.overwrite,
                 None if args.no_previews else d / "previews", args.device)
    print(f"drafted labels for {n} images in {d / 'labels'} with {weights}")
    k = export_cvat(d / "images", d / "labels", d / "cvat_prelabels_yolo1.1.zip")
    print(f"CVAT import archive ({k} images): {d / 'cvat_prelabels_yolo1.1.zip'}")
    return 0


def cmd_import_frames(args):
    from .finetune import import_frames
    cfg = None
    if Path(args.config).exists():
        cfg = _cfg(args)
    r = import_frames([Path(p) for p in args.src], Path(args.out), args.placement_date,
                      args.camera, cfg, not args.no_osd)
    print(f"imported {r['imported']} frames into {Path(args.out) / 'images'}")
    print(f"days: {', '.join(r['days'])}")
    print(f"per growth phase: {r['phases']}")
    if r["unreadable"]:
        print(f"no recording time found for {len(r['unreadable'])} frames (skipped): {r['unreadable']}")
    if r["time_mismatch"]:
        print(f"OSD time differs from the file name for {len(r['time_mismatch'])} frames, check them: "
              f"{r['time_mismatch']}")
    return 0


def cmd_build_dataset(args):
    from .finetune import build_dataset
    d = Path(args.dir)
    s = build_dataset(d / "images", Path(args.labels) if args.labels else d / "labels",
                      d / "manifest.csv", Path(args.out), args.classes.split(","),
                      args.test_frac, args.val_frac, args.seed)
    print(f"dataset in {args.out}: {s['splits']} (per phase: {s['phases']})")
    return 0


def cmd_train(args):
    from .finetune import default_base_weights, train
    cfg = _cfg(args)
    base = args.base or default_base_weights(cfg.root)
    best = train(Path(args.data), base, Path(args.out), args.imgsz, args.epochs, args.batch,
                 args.patience, args.device)
    print(f"best weights: {best}")
    print("use them: in farm.yaml set analysis.segmenter.kind: yolo and weights: <that path>")
    return 0


def cmd_evaluate(args):
    from .finetune import evaluate, evaluation_markdown
    r = evaluate(Path(args.weights), Path(args.dataset), args.imgsz, device=args.device)
    md = evaluation_markdown(r)
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(json.dumps(r, indent=2))
    return 0


def cmd_observe(args):
    from .store import Store
    cfg = _cfg(args)
    ts = datetime.strptime(args.ts, "%Y-%m-%d %H:%M") if args.ts else datetime.now()
    oid = Store(cfg.db_path).add_observation(args.pen, ts, args.kind, args.severity, args.note, args.by)
    print(f"observation {oid} saved")
    return 0


def cmd_validate(args):
    from .store import Store
    from .validation import dump, report_markdown, validate
    cfg = _cfg(args)
    since = datetime.strptime(args.since, "%Y-%m-%d") if args.since else \
        datetime.combine(cfg.placement_date, datetime.min.time())
    until = datetime.strptime(args.until, "%Y-%m-%d") + timedelta(days=1) if args.until else datetime.now()
    r = validate(Store(cfg.db_path), args.pen, since, until, args.lead_h, args.lag_h)
    text = dump(r) if args.json else report_markdown(r)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_log(args):
    from .pipeline import Pipeline
    pct = Pipeline(_cfg(args)).daily_mortality(args.pen, args.date, args.deaths, args.culls, args.note)
    print(f"{args.pen} {args.date}: daily mortality {pct:.2f}%" if pct is not None else "logged")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="pews", description="Poultry disease early-warning system")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", default="configs/farm.yaml")
        sp.add_argument("--rules", default=None, help="rules.yaml (default: configs/rules.yaml)")
        return sp

    s = common(sub.add_parser("process", help="analyse clips of one camera"))
    s.add_argument("--camera", required=True)
    s.add_argument("--max-seconds", type=float, default=None, help="analyse only the start of each clip")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=cmd_process)

    s = common(sub.add_parser("watch", help="process new clips as they arrive"))
    s.add_argument("--poll", type=float, default=30)
    s.add_argument("--once", action="store_true", help="one pass, then exit")
    s.set_defaults(fn=cmd_watch)

    s = common(sub.add_parser("serve", help="web panel"))
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = common(sub.add_parser("preview", help="check calibration on a frame"))
    s.add_argument("--camera", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--date", default=None, help="YYYY-MM-DD (which calibration version)")
    s.add_argument("--out", default="preview.jpg")
    s.set_defaults(fn=cmd_preview)

    s = common(sub.add_parser("calibrate", help="click calibration on a frame"))
    s.add_argument("--camera", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--valid-from", required=True, help="YYYY-MM-DD from which this framing is valid")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_calibrate)

    s = common(sub.add_parser("to-metres", help="image pixels -> floor metres"))
    s.add_argument("--camera", required=True)
    s.add_argument("--date", required=True)
    s.add_argument("px", nargs="+", help="x,y in full-resolution pixels")
    s.set_defaults(fn=cmd_to_metres)

    s = sub.add_parser("extract-frames", help="sample frames for labelling")
    s.add_argument("paths", nargs="+")
    s.add_argument("--every", type=float, default=60, help="seconds between frames")
    s.add_argument("--width", type=int, default=2304)
    s.add_argument("--fps", type=float, default=15, help="fallback fps for raw .h264")
    s.add_argument("--out", default="farm_data/labeling")
    s.set_defaults(fn=cmd_extract)

    s = common(sub.add_parser("log-mortality", help="record daily deaths"))
    s.add_argument("--pen", required=True)
    s.add_argument("--date", required=True)
    s.add_argument("--deaths", type=int, required=True)
    s.add_argument("--culls", type=int, default=0)
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_log)

    # ---- detector fine-tuning (docs/finetune.md) ----
    s = common(sub.add_parser("sample-frames", help="1) pick frames to label"))
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--camera", action="append", help="limit to camera id (repeatable)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="farm_data/labeling")
    s.set_defaults(fn=cmd_sample)

    s = common(sub.add_parser("import-frames", help="1b) use frames you already exported"))
    s.add_argument("src", nargs="+", help="folders with frames (sub-folders are searched too)")
    s.add_argument("--placement-date", required=True, help="YYYY-MM-DD, day 0 of the flock")
    s.add_argument("--camera", default="cam1")
    s.add_argument("--out", default="farm_data/labeling")
    s.add_argument("--no-osd", action="store_true", help="do not read the timestamp burned into the frame")
    s.set_defaults(fn=cmd_import_frames)

    s = common(sub.add_parser("prelabel", help="2) draft boxes with an existing model"))
    s.add_argument("--dir", default="farm_data/labeling")
    s.add_argument("--device", default=None, help="'0' = first GPU, 'cpu'")
    s.add_argument("--no-previews", action="store_true")
    s.add_argument("--weights", default=None, help="default: MVBroTrack yolo11x_1920.pt if downloaded")
    s.add_argument("--conf", type=float, default=0.25)
    s.add_argument("--imgsz", type=int, default=1920)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=cmd_prelabel)

    s = sub.add_parser("build-dataset", help="4) split labelled frames into a YOLO dataset")
    s.add_argument("--dir", default="farm_data/labeling")
    s.add_argument("--labels", default=None, help="corrected labels folder (default: <dir>/labels)")
    s.add_argument("--out", default="farm_data/yolo_dataset")
    s.add_argument("--classes", default="broiler,dead")
    s.add_argument("--test-frac", type=float, default=0.2)
    s.add_argument("--val-frac", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_build_dataset)

    s = common(sub.add_parser("train", help="5) fine-tune the detector"))
    s.add_argument("--data", default="farm_data/yolo_dataset/data.yaml")
    s.add_argument("--base", default=None, help="default: MVBroTrack yolo11x_1920.pt, else yolo11s.pt")
    s.add_argument("--out", default="farm_data/runs")
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--batch", type=float, default=0.6, help="<1 = fraction of GPU memory")
    s.add_argument("--patience", type=int, default=30)
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="6) test-set accuracy, overall and per growth phase")
    s.add_argument("--weights", required=True)
    s.add_argument("--dataset", default="farm_data/yolo_dataset")
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--device", default=None)
    s.add_argument("--out", default=None, help="write a markdown report (+ .json)")
    s.set_defaults(fn=cmd_evaluate)

    s = common(sub.add_parser("observe", help="record a vet / farmer observation (reference data)"))
    s.add_argument("--pen", required=True)
    s.add_argument("--kind", required=True,
                   help="clinical_sign | diagnosis | treatment | vaccination | death | "
                        "lighting_change | feed_change | equipment_problem | other")
    s.add_argument("--ts", default=None, help="'YYYY-MM-DD HH:MM' (default: now)")
    s.add_argument("--severity", default="")
    s.add_argument("--note", default="")
    s.add_argument("--by", default="")
    s.set_defaults(fn=cmd_observe)

    s = common(sub.add_parser("validate", help="compare alerts with vet observations"))
    s.add_argument("--pen", required=True)
    s.add_argument("--since", default=None, help="YYYY-MM-DD (default: placement date)")
    s.add_argument("--until", default=None, help="YYYY-MM-DD (inclusive, default: now)")
    s.add_argument("--lead-h", type=float, default=72.0, help="alert may precede the problem by this much")
    s.add_argument("--lag-h", type=float, default=24.0, help="alert may follow the problem by this much")
    s.add_argument("--json", action="store_true")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_validate)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
