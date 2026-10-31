from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from poultry_ews.api.app import create_app
from poultry_ews.pipeline import Pipeline
from poultry_ews.validation import report_markdown, validate

from synth import make_config


def _event(st, ts, signal="activity", rule="activity_drop"):
    return st.create_event(pen="pen1", rule_id=rule, signal=signal, severity="warning",
                           opened_ts=ts.isoformat(timespec="seconds"),
                           updated_ts=ts.isoformat(timespec="seconds"),
                           slot_ts=ts.isoformat(timespec="seconds"), value=0.1, z=-2.5, message="m")


def test_validate_precision_recall_lead(tmp_path):
    cfg = make_config(tmp_path)
    st = Pipeline(cfg).store
    t0 = datetime(2025, 12, 1, 8)
    _event(st, t0)                                  # vet finds a problem 30 h later -> match
    _event(st, t0 + timedelta(days=5))              # nothing follows -> false alert
    _event(st, t0 + timedelta(days=10), "distribution", "periphery_crowding")
    st.add_observation("pen1", t0 + timedelta(hours=30), "clinical_sign", "moderate", "lethargy", "vet")
    st.add_observation("pen1", t0 + timedelta(days=10, hours=-5), "diagnosis", note="ND confirmed")
    st.add_observation("pen1", t0 + timedelta(days=20), "death", note="3 dead, no alert")
    st.add_observation("pen1", t0 + timedelta(days=2), "vaccination", note="not a problem")
    r = validate(st, "pen1", t0 - timedelta(days=1), t0 + timedelta(days=25))
    assert r["alerts"] == 3 and r["vet_problems"] == 3
    assert abs(r["precision"] - 2 / 3) < 1e-9
    assert abs(r["recall"] - 2 / 3) < 1e-9
    assert r["per_signal"]["activity"]["matched"] == 1
    assert r["per_signal"]["activity"]["median_lead_h"] == pytest.approx(30.0)
    assert r["per_signal"]["distribution"]["median_lead_h"] == pytest.approx(-5.0)
    md = report_markdown(r)
    assert "| Alerts | 3 |" in md


def test_observation_api_and_validation_endpoint(tmp_path):
    cfg = make_config(tmp_path)
    c = TestClient(create_app(cfg))
    r = c.post("/api/observations", json={"pen": "pen1", "kind": "clinical_sign", "note": "tremors",
                                         "by": "vet", "ts": "2025-12-03T10:00:00"})
    assert r.status_code == 200 and r.json()["kind"] == "clinical_sign"
    assert c.post("/api/observations", json={"pen": "pen1", "kind": "bogus", "note": "x"}).status_code == 400
    assert "lighting_change" in c.get("/api/observation-kinds").json()
    v = c.get("/api/pens/pen1/validation?since=2025-12-01T00:00:00&until=2025-12-10T00:00:00").json()
    assert v["vet_problems"] == 1 and v["recall"] == 0.0
    assert "Vet log" in c.get("/").text


def test_lighting_schedule(tmp_path):
    from poultry_ews.config import LightingPeriod
    from datetime import date, time
    cfg = make_config(tmp_path)
    cfg.lighting = [LightingPeriod(date(2025, 11, 1), time(5, 0), time(23, 0)),
                    LightingPeriod(date(2025, 12, 1), time(6, 0), time(2, 0))]  # lights off after midnight
    assert cfg.lights_on(datetime(2025, 11, 20, 12)) and not cfg.lights_on(datetime(2025, 11, 20, 23, 30))
    assert cfg.lights_on(datetime(2025, 12, 5, 12)) and cfg.lights_on(datetime(2025, 12, 5, 1))
    assert not cfg.lights_on(datetime(2025, 12, 5, 3))
    # rhythm is suppressed for 2 days after a lighting change
    pipe = Pipeline(cfg)
    assert pipe._circadian("pen1", datetime(2025, 12, 2, 10), {"activity_index": 0.2})["rhythm_corr"] is None
