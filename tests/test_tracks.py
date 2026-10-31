"""Detection mode: tracking on the floor plane and the track-based signals."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from poultry_ews.analyzer import analyze_clip
from poultry_ews.detection import parse_mvbrotrack_txt
from poultry_ews.pipeline import Pipeline
from poultry_ews.signals.tracks import metrics_from_stats
from poultry_ews.tracking import FloorTracker

from synth import make_config, moving_flock, still_flock, write_video_with_dets


@pytest.fixture()
def cfg(tmp_path):
    c = make_config(tmp_path)
    c.analysis["segmenter"]["kind"] = "mvbrotrack"
    c.analysis["tracking"].update(segment_s=2.0, min_track_s=3.0)
    return c


def test_parse_mvbrotrack_txt():
    b = parse_mvbrotrack_txt("0 100 50 20 10 0.9\n1 10 10 4 4 0.2\n", min_conf=0.5)
    assert b.shape == (1, 6)
    assert np.allclose(b[0], [90, 45, 110, 55, 0.9, 0])


def test_tracker_keeps_identities_of_two_birds():
    tr = FloorTracker(max_dist_m=0.3)
    for k in range(10):
        t = k * 0.5
        pts = np.array([[0.5 + 0.05 * k, 1.0], [1.5 - 0.05 * k, 2.0]])
        tr.update(t, pts, np.full(2, 0.01), np.zeros(2))
    tracks = tr.tracks()
    assert len(tracks) == 2 and all(len(t.t) == 10 for t in tracks)


def test_activity_from_tracks(cfg, tmp_path):
    t = datetime(2025, 11, 20, 8, 0)
    mv = analyze_clip(cfg, "cam1", write_video_with_dets(tmp_path / "m.mp4", moving_flock(10)), t)
    st = analyze_clip(cfg, "cam1", write_video_with_dets(tmp_path / "s.mp4", still_flock(10)), t)
    m_mv = metrics_from_stats([mv.track_stats])
    m_st = metrics_from_stats([st.track_stats])
    assert m_mv["inactive_frac"] < 0.2 and m_mv["active_speed"] > 0.1
    assert m_st["inactive_frac"] > 0.95
    assert abs(m_mv["bird_count"] - 10) < 0.5


def test_feeder_time(cfg, tmp_path):
    # feeder zone in the test config: floor (1.0 m, 1.0 m), r 0.4 m.
    # Test calibration: image x -> floor y (4 m over 640 px), image y -> floor x (2 m over 360 px)
    x_px, y_px = 1.0 / 4.0 * 640, 1.0 / 2.0 * 360
    at_feeder = [{"x": x_px + dx, "y": y_px + dy, "r": 8} for dx, dy in ((0, 0), (20, 10), (-20, -10))]
    away = [{"x": 560, "y": 300, "r": 8}, {"x": 600, "y": 60, "r": 8}, {"x": 520, "y": 200, "r": 8}]
    t = datetime(2025, 11, 20, 8, 0)
    r = analyze_clip(cfg, "cam1", write_video_with_dets(tmp_path / "f.mp4", at_feeder + away), t)
    m = metrics_from_stats([r.track_stats])
    assert abs(m["feeder_time_frac"] - 0.5) < 0.05
    assert abs(m["count_frac_feeder"] - 0.5) < 0.05


def test_dead_class_raises_candidate_after_two_clips(cfg, tmp_path):
    pipe = Pipeline(cfg)
    dead = {"x": 500, "y": 100, "r": 10, "cls": 1}
    t0 = datetime(2025, 11, 20, 9, 0)
    for i in range(2):
        clip = write_video_with_dets(tmp_path / f"d{i}.mp4", moving_flock(6, seed=i + 3) + [dead], seed=i)
        pipe.process_clip("cam1", clip, t0 + timedelta(minutes=5 * i))
    ev = pipe.store.q("SELECT * FROM events WHERE rule_id='mortality_candidate'")
    assert len(ev) == 1
    assert "classified this bird as dead" in ev[0]["message"]


def test_still_track_raises_candidate(cfg, tmp_path):
    pipe = Pipeline(cfg)
    still = {"x": 500, "y": 100, "r": 10}
    t0 = datetime(2025, 11, 20, 9, 0)
    for i in range(4):
        clip = write_video_with_dets(tmp_path / f"c{i}.mp4", moving_flock(6, seed=i + 3) + [still], seed=i)
        pipe.process_clip("cam1", clip, t0 + timedelta(minutes=5 * i))
    assert len(pipe.store.q("SELECT * FROM events WHERE rule_id='mortality_candidate'")) == 1
    r = pipe.store.latest_metric("pen1", "inactive_frac")
    assert r["value"] is not None
