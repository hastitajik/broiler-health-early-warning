"""Measure how well the alerts match what the veterinarian actually saw.

No paper gives the accuracy of this system on THIS farm; it has to be measured
here. Reference data comes from the vet / farmer log (observations table:
clinical signs, diagnoses, deaths) and from the outcome buttons on mortality
candidates (dead / alive).

Matching rule for an early-warning system: an alert counts as correct if a
vet-recorded problem follows it within `lead_h` hours, or preceded it by at
most `lag_h` hours (the alert came late but pointed at a real problem).

    pews validate --pen pen1 --since 2025-11-17 --until 2025-12-31 --out report.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median
from typing import Any

from .store import Store, parse_iso

PROBLEM_KINDS = ("clinical_sign", "diagnosis", "death")


@dataclass
class SignalReport:
    signal: str
    alerts: int = 0
    alerts_matched: int = 0
    lead_hours: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        return self.alerts_matched / self.alerts if self.alerts else None


def validate(store: Store, pen: str, since: datetime, until: datetime,
             lead_h: float = 72.0, lag_h: float = 24.0) -> dict[str, Any]:
    events = [dict(e) for e in store.q(
        "SELECT * FROM events WHERE pen=? AND opened_ts>=? AND opened_ts<=? ORDER BY opened_ts",
        (pen, since.isoformat(timespec="seconds"), until.isoformat(timespec="seconds")))]
    obs = [dict(o) for o in store.observations(pen, since - timedelta(hours=lag_h),
                                               until + timedelta(hours=lead_h), PROBLEM_KINDS)]
    obs_in_window = [o for o in obs if since <= parse_iso(o["ts"]) <= until]

    per_signal: dict[str, SignalReport] = {}
    for e in events:
        sr = per_signal.setdefault(e["signal"], SignalReport(e["signal"]))
        sr.alerts += 1
        t = parse_iso(e["opened_ts"])
        diffs = [(parse_iso(o["ts"]) - t).total_seconds() / 3600 for o in obs]
        ok = [d for d in diffs if -lag_h <= d <= lead_h]
        if ok:
            sr.alerts_matched += 1
            sr.lead_hours.append(max(ok) if all(d < 0 for d in ok) else min(d for d in ok if d >= 0))

    # recall: share of vet-recorded problems preceded (or shortly followed) by any alert
    detected = 0
    first_lead: list[float] = []
    for o in obs_in_window:
        t = parse_iso(o["ts"])
        diffs = [(t - parse_iso(e["opened_ts"])).total_seconds() / 3600 for e in events]
        ok = [d for d in diffs if -lag_h <= d <= lead_h]
        if ok:
            detected += 1
            first_lead.append(max(ok))

    cands = store.q("SELECT status FROM candidates WHERE pen=? AND status IN "
                    "('confirmed_dead','alive','other') AND last_slot>=? AND first_slot<=?",
                    (pen, since.isoformat(timespec="seconds"), until.isoformat(timespec="seconds")))
    dead = sum(1 for c in cands if c["status"] == "confirmed_dead")
    alive = sum(1 for c in cands if c["status"] == "alive")

    weeks = max((until - since).total_seconds() / (7 * 86400), 1e-9)
    all_alerts = sum(s.alerts for s in per_signal.values())
    all_matched = sum(s.alerts_matched for s in per_signal.values())
    return {
        "pen": pen, "since": since.isoformat(), "until": until.isoformat(),
        "lead_h": lead_h, "lag_h": lag_h,
        "vet_problems": len(obs_in_window),
        "vet_problems_detected": detected,
        "recall": detected / len(obs_in_window) if obs_in_window else None,
        "median_warning_hours": median(first_lead) if first_lead else None,
        "alerts": all_alerts,
        "precision": all_matched / all_alerts if all_alerts else None,
        "false_alerts_per_week": (all_alerts - all_matched) / weeks,
        "mortality_candidates_checked": dead + alive,
        "mortality_candidate_precision": dead / (dead + alive) if dead + alive else None,
        "per_signal": {k: {"alerts": v.alerts, "matched": v.alerts_matched,
                           "precision": v.precision,
                           "median_lead_h": median(v.lead_hours) if v.lead_hours else None}
                       for k, v in sorted(per_signal.items())},
    }


def _fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{100 * v:.0f}%" if pct else (f"{v:.1f}" if isinstance(v, float) else str(v))


def report_markdown(r: dict[str, Any]) -> str:
    lines = [
        f"# Alert validation: {r['pen']}",
        "",
        f"Period: {r['since'][:10]} to {r['until'][:10]}. An alert is correct when a vet-recorded "
        f"problem follows it within {r['lead_h']:.0f} h or preceded it by at most {r['lag_h']:.0f} h.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Vet-recorded problems | {r['vet_problems']} |",
        f"| ... preceded by an alert (recall) | {r['vet_problems_detected']} ({_fmt(r['recall'], True)}) |",
        f"| Median warning time before the vet record | {_fmt(r['median_warning_hours'])} h |",
        f"| Alerts | {r['alerts']} |",
        f"| Alerts followed by a real problem (precision) | {_fmt(r['precision'], True)} |",
        f"| False alerts per week | {_fmt(r['false_alerts_per_week'])} |",
        f"| Mortality candidates checked on site | {r['mortality_candidates_checked']} |",
        f"| ... that were really dead | {_fmt(r['mortality_candidate_precision'], True)} |",
        "",
        "| Signal | Alerts | Matched | Precision | Median lead (h) |",
        "|---|---|---|---|---|",
    ]
    for k, v in r["per_signal"].items():
        lines.append(f"| {k} | {v['alerts']} | {v['matched']} | {_fmt(v['precision'], True)} | "
                     f"{_fmt(v['median_lead_h'])} |")
    if r["vet_problems"] == 0:
        lines += ["", "No vet observations in this period: record clinical signs, diagnoses and "
                      "deaths in the panel (Vet log) so accuracy can be measured."]
    return "\n".join(lines) + "\n"


def dump(r: dict[str, Any]) -> str:
    return json.dumps(r, indent=2, default=str)
