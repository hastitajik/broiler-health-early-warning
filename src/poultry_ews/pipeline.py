"""Pipeline: clip -> camera result -> pen fusion -> metrics + baseline -> alerts.

    process_clip()   analyse one clip of one camera and store its result
    fuse_slot()      combine all cameras of a pen for one 5-minute slot,
                     compute the 4 signals, z-scores, mortality tracking, alerts
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .alerts import AlertEngine
from .analyzer import CameraClipResult, analyze_clip, camera_geometry
from .baseline import compute_baseline
from .config import Config
from .geometry import FloorCanvas
from .segmentation import Segmenter
from .signals.activity import ActivityAccumulator
from .signals.circadian import baseline_profile, hourly_means, rhythm_metrics
from .signals.distribution import distribution_metrics, fuse_occupancy
from .signals.mortality import merge_blobs
from .signals.tracks import metrics_from_stats
from .store import Store, iso, parse_iso
from .timeparse import parse_clip_time, slot_start
from .video import is_media

log = logging.getLogger("pews.pipeline")

# metrics that get a baseline + z-score
ZSCORED = ("activity_index", "mean_speed", "resource_frac", "periphery_frac",
           "frac_feeder", "frac_drinker", "frac_wall", "frac_corner", "dispersion_entropy",
           # detection mode (signals/tracks.py)
           "inactive_frac", "active_speed", "resource_count_frac", "periphery_count_frac",
           "feeder_time_frac", "bird_count")


class Pipeline:
    def __init__(self, cfg: Config, store: Store | None = None, segmenter: Segmenter | None = None):
        self.cfg = cfg
        self.store = store or Store(cfg.db_path)
        self.segmenter = segmenter
        self.alerts = AlertEngine(cfg, self.store)
        self.slot_min = int(cfg.analysis["slot_minutes"])
        for sub in ("grids", "media", "media/events"):
            (cfg.data_dir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def _grid_path(self, pen: str, slot: datetime, cam: str) -> Path:
        p = self.cfg.data_dir / "grids" / pen / f"{slot:%Y%m%d_%H%M}_{cam}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _snap_path(self, slot: datetime, cam: str) -> Path:
        p = self.cfg.data_dir / "media" / "slots" / f"{slot:%Y%m%d}" / f"{slot:%H%M}_{cam}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def rel(self, p: Path) -> str:
        """Path relative to data_dir, as served by the API under /media."""
        return str(Path(p).relative_to(self.cfg.data_dir / "media")).replace("\\", "/")

    # ------------------------------------------------------------- process
    def process_clip(self, camera_id: str, clip: str | Path, start: datetime | None = None,
                     max_seconds: float | None = None, run_fusion: bool = True) -> CameraClipResult:
        clip = Path(clip)
        cam = self.cfg.camera(camera_id)
        if start is None:
            start = parse_clip_time(clip, cam.filename_regex)
        slot = slot_start(start, self.slot_min)
        try:
            res = analyze_clip(self.cfg, camera_id, clip, start, self.segmenter, max_seconds)
        except Exception as e:
            self.store.mark_clip(str(clip), camera_id, start, slot, "error", repr(e))
            raise
        grid = self._grid_path(cam.pen, slot, camera_id)
        np.savez_compressed(grid, occupancy=res.occupancy.astype(np.float16), coverage=res.coverage)
        snap = self._snap_path(slot, camera_id)
        if res.snapshot is not None:
            cv2.imwrite(str(snap), res.snapshot, [cv2.IMWRITE_JPEG_QUALITY, 75])
            shutil.copyfile(snap, self.cfg.data_dir / "media" / f"latest_{camera_id}.jpg")
        self.store.put_camera_result(
            slot, camera_id, cam.pen, start, res.duration_s, res.n_frames, res.bird_px,
            res.moving_px, res.speed_sum, str(grid), self.rel(snap),
            [(b.x_m, b.y_m, b.area_m2, b.kind) for b in res.blobs], res.track_stats)
        self.store.mark_clip(str(clip), camera_id, start, slot, "done")
        log.info("%s %s slot %s: %d frames, bird_px=%.0f, still blobs=%d",
                 camera_id, clip.name, slot, res.n_frames, res.bird_px, len(res.blobs))
        if run_fusion:
            self.fuse_slot(cam.pen, slot)
        return res

    # -------------------------------------------------------------- fusion
    def fuse_slot(self, pen_id: str, slot: datetime) -> dict[str, float | None]:
        cfg, st = self.cfg, self.store
        pen = cfg.pen(pen_id).at(slot)
        rows = st.camera_results(pen_id, slot)
        if not rows:
            return {}
        age = cfg.age_days(slot)

        bird_px = sum(r["bird_px"] for r in rows)
        moving_px = sum(r["moving_px"] for r in rows)
        speed_sum = sum(r["speed_sum"] for r in rows)
        metrics: dict[str, float | None] = dict(ActivityAccumulator.metrics(bird_px, moving_px, speed_sum))

        maps, covs = [], []
        for r in rows:
            g = np.load(r["grid_path"])
            maps.append(g["occupancy"].astype(np.float32))
            covs.append(g["coverage"])
        occ, cov = fuse_occupancy(maps, covs)
        canvas = FloorCanvas(pen.width_m, pen.length_m, float(cfg.analysis["floor_px_per_m"]))
        zone_map = canvas.zone_map(pen)
        metrics.update(distribution_metrics(occ, cov, zone_map, canvas.ppm))
        fused_path = self._grid_path(pen_id, slot, "fused")
        np.savez_compressed(fused_path, occupancy=occ.astype(np.float16), coverage=cov)

        stats = [json.loads(r["track_stats"]) for r in rows if r["track_stats"]]
        if stats:
            metrics.update(metrics_from_stats(stats))

        metrics.update(self._circadian(pen_id, slot, metrics))

        for name, value in metrics.items():
            if value is None:
                st.put_metric(cfg.flock_id, pen_id, slot, age, name, None)
            elif name in ZSCORED:
                b = compute_baseline(st, cfg.baseline, cfg.flock_id, pen_id, name, slot, age)
                st.put_metric(cfg.flock_id, pen_id, slot, age, name, value, b.median, b.mad,
                              b.z(value), b.n)
            else:
                st.put_metric(cfg.flock_id, pen_id, slot, age, name, value)

        snaps = [r["snapshot_path"] for r in rows if r["snapshot_path"]]
        activity_now = metrics.get("activity_index")
        if metrics.get("inactive_frac") is not None:
            activity_now = 1.0 - metrics["inactive_frac"]
        self._track_mortality(pen_id, slot, activity_now, rows)
        n_watch = len(st.candidates(pen_id, ("alerted",)))
        st.put_metric(cfg.flock_id, pen_id, slot, age, "still_candidates", float(n_watch))
        # snapshots are copied to media/events only when an event is actually created
        self.alerts.evaluate(pen_id, slot, lambda: self._keep_snapshots(snaps))
        return metrics

    # ------------------------------------------------------------ circadian
    def _circadian(self, pen: str, slot: datetime, metrics: dict) -> dict[str, float | None]:
        """Rhythm of the last 24 h vs the usual hourly profile.

        Uses the track-based activity (1 - inactive_frac) when a detector is
        configured, otherwise the optical-flow activity_index. Dark hours of the
        lighting programme are left out (the camera may not see, and birds rest),
        and no rhythm is reported for 48 h after a lighting change.
        """
        c = self.cfg.analysis["circadian"]
        none = {"rhythm_corr": None, "rhythm_amplitude": None}
        use_tracks = metrics.get("inactive_frac") is not None
        name = "inactive_frac" if use_tracks else "activity_index"
        current = metrics.get(name)
        conv = (lambda v: None if v is None else 1.0 - v) if use_tracks else (lambda v: v)

        period = self.cfg.lighting_at(slot)
        if period is not None and slot.date() <= period.valid_from + timedelta(days=2) and \
                any(p.valid_from < period.valid_from for p in self.cfg.lighting):
            return none  # lighting programme changed recently: the old rhythm is not comparable

        def rows(since, until):
            out = []
            for r in self.store.metric_series(pen, name, since=since, until=until,
                                              flock_id=self.cfg.flock_id):
                ts = parse_iso(r["slot_ts"])
                if self.cfg.lights_on(ts):
                    out.append((ts, conv(r["value"])))
            return out

        recent_rows = rows(slot - timedelta(hours=24), slot)
        if current is not None and not any(t == slot for t, _ in recent_rows) and self.cfg.lights_on(slot):
            recent_rows.append((slot, conv(current)))
        d0, d1 = c["baseline_days"]
        usual = baseline_profile(rows(slot - timedelta(days=int(d1)), slot - timedelta(days=int(d0) - 1)))
        lit_hours = sum(1 for h in range(24) if self.cfg.lights_on(slot.replace(hour=h, minute=30)))
        min_hours = min(int(c["min_hours"]), max(6, lit_hours - 2))
        return rhythm_metrics(hourly_means(recent_rows), usual, min_hours)

    # ------------------------------------------------------------ mortality
    def _track_mortality(self, pen: str, slot: datetime, activity: float | None, rows):
        m = self.cfg.analysis["mortality"]
        radius = float(m["match_radius_m"])
        blobs = merge_blobs([(b["x_m"], b["y_m"], b["area_m2"], b["kind"] or "still")
                             for b in self.store.blobs(pen, slot)], radius)
        active = activity is not None and activity >= float(m["min_flock_activity"])
        gap = timedelta(minutes=2 * self.slot_min)
        cands = [dict(c) for c in self.store.candidates(pen, ("open", "alerted"))]
        matched: set[int] = set()

        dead_min = int(m.get("dead_class_min_clips", 2))
        for x, y, a, kind in blobs:
            best, best_d = None, radius
            for c in cands:
                if c["id"] in matched:
                    continue
                last = parse_iso(c["last_slot"])
                if last > slot or last < slot - gap:
                    continue  # from the future (re-processing) or lost for too long
                d = ((c["x_m"] - x) ** 2 + (c["y_m"] - y) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = c, d
            if best is None:
                cid = self.store.add_candidate(pen, slot, x, y, a, float(self.slot_min), active)
                if kind == "dead":
                    self.store.update_candidate(cid, dead_votes=1)
                matched.add(cid)
                continue
            matched.add(best["id"])
            if parse_iso(best["last_slot"]) == slot:
                continue  # slot fused again (another camera arrived): already counted
            best["still_min"] += self.slot_min
            best["active_seen"] = int(best["active_seen"] or active)
            best["dead_votes"] = int(best.get("dead_votes") or 0) + (1 if kind == "dead" else 0)
            self.store.update_candidate(best["id"], last_slot=iso(slot), still_min=best["still_min"],
                                        active_seen=best["active_seen"], dead_votes=best["dead_votes"])
            by_stillness = best["still_min"] >= float(m["alert_after_min"]) and best["active_seen"]
            by_dead_class = best["dead_votes"] >= dead_min
            if best["status"] == "open" and (by_stillness or by_dead_class):
                snaps = self._annotate_candidate(pen, slot, best, rows)
                eid = self.alerts.raise_mortality_candidate(pen, slot, best, snaps)
                self.store.update_candidate(best["id"], status="alerted", event_id=eid)

        # open candidates not seen recently -> the bird moved: not dead
        for c in cands:
            if c["id"] not in matched and c["status"] == "open" and parse_iso(c["last_slot"]) < slot - gap:
                self.store.update_candidate(c["id"], status="moved")

    def _annotate_candidate(self, pen: str, slot: datetime, cand: dict, rows) -> list[str]:
        out = []
        for r in rows:
            src = self.cfg.data_dir / "media" / r["snapshot_path"] if r["snapshot_path"] else None
            if not src or not src.exists():
                continue
            geo = camera_geometry(self.cfg, self.cfg.camera(r["camera"]), slot)
            px = geo.floor_px_to_image(np.array([[cand["x_m"] * geo.canvas.ppm,
                                                  cand["y_m"] * geo.canvas.ppm]]))[0]
            img = cv2.imread(str(src))
            h, w = img.shape[:2]
            if not (0 <= px[0] < w and 0 <= px[1] < h) or not geo.image_mask[int(px[1]), int(px[0])]:
                continue
            rad = max(18, int(0.25 * geo.canvas.ppm * geo.scale * 2))
            cv2.circle(img, (int(px[0]), int(px[1])), rad, (0, 0, 255), 4)
            cv2.putText(img, f"#{cand['id']}", (int(px[0]) + rad, int(px[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            dst = self.cfg.data_dir / "media" / "events" / f"cand{cand['id']}_{slot:%Y%m%d_%H%M}_{r['camera']}.jpg"
            cv2.imwrite(str(dst), img)
            out.append(self.rel(dst))
        return out

    def _keep_snapshots(self, snaps: list[str]) -> list[str]:
        """Copy slot snapshots to media/events so pruning never deletes evidence."""
        out = []
        for s in snaps:
            src = self.cfg.data_dir / "media" / s
            if src.exists():
                dst = self.cfg.data_dir / "media" / "events" / src.name.replace(".jpg", f"_{src.parent.name}.jpg")
                if not dst.exists():
                    shutil.copyfile(src, dst)
                out.append(self.rel(dst))
        return out

    # ------------------------------------------------------------ daily log
    def daily_mortality(self, pen_id: str, date: str, deaths: int, culls: int, note: str = ""):
        cfg, st = self.cfg, self.store
        st.put_daily_log(pen_id, date, deaths, culls, note)
        pen_count = int(cfg.flock.get("initial_count", 0)) / max(1, len(cfg.pens))
        prev = st.q1("SELECT COALESCE(SUM(deaths+culls),0) AS n FROM daily_log WHERE pen=? AND date<?",
                     (pen_id, date))
        alive = pen_count - (prev["n"] if prev else 0)
        slot = datetime.strptime(date, "%Y-%m-%d").replace(hour=23, minute=55)
        pct = 100.0 * (deaths + culls) / alive if alive > 0 else None
        st.put_metric(cfg.flock_id, pen_id, slot, cfg.age_days(slot), "daily_mortality_pct", pct)
        for rule in cfg.rules:
            if rule.metric == "daily_mortality_pct":
                self.alerts._evaluate_rule(rule, pen_id, slot, [])
        return pct

    # ------------------------------------------------------------ housekeeping
    def prune(self, keep_hours: int = 48):
        """Delete per-slot snapshots and camera grids older than keep_hours (event copies stay)."""
        cutoff = datetime.now() - timedelta(hours=keep_hours)
        slots = self.cfg.data_dir / "media" / "slots"
        if slots.exists():
            for d in slots.iterdir():
                try:
                    if datetime.strptime(d.name, "%Y%m%d") < cutoff - timedelta(days=1):
                        shutil.rmtree(d, ignore_errors=True)
                except ValueError:
                    pass


# ---------------------------------------------------------------- batch helpers
def find_clips(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.is_dir():
            out += [q for q in sorted(p.rglob("*")) if q.is_file() and is_media(q)]
        elif is_media(p):
            out.append(p)
    return out


def process_many(pipe: Pipeline, camera_id: str, clips: list[Path],
                 max_seconds: float | None = None) -> tuple[int, int]:
    """Process clips in recording-time order (baselines depend on order)."""
    cam = pipe.cfg.camera(camera_id)
    dated = []
    for c in clips:
        try:
            dated.append((parse_clip_time(c, cam.filename_regex), c))
        except ValueError as e:
            log.error("skip %s: %s", c, e)
    ok = err = 0
    for start, c in sorted(dated):
        if pipe.store.clip_done(str(c)):
            continue
        try:
            pipe.process_clip(camera_id, c, start, max_seconds)
            ok += 1
        except Exception as e:
            log.exception("failed %s: %s", c, e)
            err += 1
    return ok, err


def watch(pipe: Pipeline, poll_s: float = 30.0, settle_s: float = 20.0, once: bool = False):
    """Poll every camera's input_dir; process clips whose size stopped changing."""
    sizes: dict[str, tuple[int, float]] = {}
    while True:
        for cam in pipe.cfg.cameras:
            if not cam.input_dir.exists():
                continue
            for c in find_clips([cam.input_dir]):
                key = str(c)
                if pipe.store.clip_done(key):
                    continue
                size = c.stat().st_size
                prev = sizes.get(key)
                now = time.time()
                if prev is None or prev[0] != size:
                    sizes[key] = (size, now)
                    if not once:
                        continue
                elif now - prev[1] < settle_s and not once:
                    continue
                try:
                    pipe.process_clip(cam.id, c)
                except Exception as e:
                    log.error("failed %s: %s", c, e)
        pipe.prune()
        if once:
            return
        time.sleep(poll_s)


def dump_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
