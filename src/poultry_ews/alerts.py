"""Alert engine: rules -> events, with persistence, cooldown and notifications.

Messages are written for the farmer / vet. The recommended actions follow
the project PDF (signal 3): inspect the area, isolate sick birds, take samples,
call the vet, review feed / water / environment.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Callable

from .config import Config, Rule
from .store import Store, iso, parse_iso

log = logging.getLogger("pews.alerts")

SEV_ORDER = {"ok": 0, "warning": 1, "critical": 2}

SIGNAL_TITLES = {
    "activity": "Flock activity",
    "distribution": "Spatial distribution",
    "mortality": "Mortality",
    "circadian": "Daily rhythm",
}

MESSAGES = {
    "activity_drop": "Flock activity is lower than normal for this age and hour of day.",
    "activity_spike": "Flock activity suddenly rose above normal (panic, fear, disturbance?).",
    "feeder_drinker_avoidance": "Fewer birds than usual are at the feeders and drinkers.",
    "periphery_crowding": "More birds than usual are crowding at the walls and corners.",
    "daily_mortality": "Daily mortality is above the set limit.",
    "rhythm_loss": "The flock's daily activity pattern does not match previous days.",
    "rhythm_flattening": "The day/night activity difference has shrunk (the rhythm is flattening).",
    "mortality_candidate": "A bird has stayed in one place without moving for a long time. It may be dead.",
    "inactivity_rise": "More birds than normal for this age and hour are inactive.",
    "active_speed_drop": "Birds that do move are walking slower than normal for this age and hour.",
    "feeder_time_drop": "Birds are spending less time at the feeders than normal for this age and hour.",
    "periphery_count_rise": "More birds than usual are counted at the walls and corners.",
}

ACTIONS = {
    "activity": ["Walk the house and watch the birds' behaviour",
                 "Check feeders, drinkers, temperature, humidity and ventilation",
                 "If it continues: inform the veterinarian"],
    "distribution": ["Inspect the areas the birds are avoiding",
                     "Check temperature (heat or cold stress) and air flow",
                     "Look for neurological signs: tremors, twisted neck, circling"],
    "mortality": ["Inspect the marked spot on the image now",
                  "Isolate sick birds and remove dead ones",
                  "Take samples for the lab and call the veterinarian",
                  "Review recent changes in feed, water and environment"],
    "circadian": ["Compare with the house lighting programme (was it changed?)",
                  "Watch the flock closely over the next 24 hours",
                  "Inform the veterinarian if activity is also dropping"],
}


def rule_level(rule: Rule, value: float | None, z: float | None) -> str | None:
    """'ok' / 'warning' / 'critical', or None when not computable (learning)."""
    if rule.kind == "zscore":
        if z is None:
            return None
        s = -z if rule.direction == "low" else z
        return "critical" if s >= rule.crit else "warning" if s >= rule.warn else "ok"
    if value is None:
        return None
    if rule.op == ">":
        return "critical" if value > rule.crit else "warning" if value > rule.warn else "ok"
    return "critical" if value < rule.crit else "warning" if value < rule.warn else "ok"


class Notifier:
    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url

    def send(self, event: dict[str, Any]):
        log.warning("EVENT %s [%s] %s: %s", event.get("id"), event.get("severity"),
                    event.get("rule_id"), event.get("message"))
        if not self.webhook_url:
            return
        try:
            req = urllib.request.Request(
                self.webhook_url, data=json.dumps(event, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:  # never break the pipeline because of a notification
            log.error("webhook failed: %s", e)


def env_hint(store: Store, pen: str, slot: datetime) -> str:
    e = store.latest_environment(pen, before=slot)
    if not e or e["temp_c"] is None:
        return ""
    if (slot - parse_iso(e["ts"])) > timedelta(hours=3):
        return ""
    return f" Recorded temperature: {e['temp_c']:.1f} C" + (f", humidity {e['rh']:.0f}%" if e["rh"] is not None else "") + "."


class AlertEngine:
    def __init__(self, cfg: Config, store: Store, notifier: Notifier | None = None):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier or Notifier(cfg.notifications.get("webhook_url", ""))
        self.slot = timedelta(minutes=int(cfg.analysis["slot_minutes"]))

    # ------------------------------------------------------------------------
    def evaluate(self, pen: str, slot: datetime,
                 snapshots: list[str] | Callable[[], list[str]] | None = None) -> list[int]:
        """Evaluate all rules for a pen at a slot. Returns ids of new/escalated events.

        `snapshots` may be a callable so images are only copied when an event is created.
        """
        touched = []
        for rule in self.cfg.rules:
            eid = self._evaluate_rule(rule, pen, slot, snapshots or [])
            if eid:
                touched.append(eid)
        return touched

    def _recent(self, rule: Rule, pen: str, slot: datetime):
        rows = self.store.q(
            "SELECT * FROM metrics WHERE pen=? AND name=? AND slot_ts<=? ORDER BY slot_ts DESC LIMIT ?",
            (pen, rule.metric, iso(slot), rule.persist))
        return rows

    def _evaluate_rule(self, rule: Rule, pen: str, slot: datetime,
                       snapshots: list[str] | Callable[[], list[str]]) -> int | None:
        rows = self._recent(rule, pen, slot)
        if not rows or rows[0]["slot_ts"] != iso(slot):
            return None  # this metric was not produced at this slot
        levels = [rule_level(rule, r["value"], r["z"]) for r in rows]
        current = levels[0]
        existing = self.store.open_event(pen, rule.id)

        # consecutive slots? allow one missing clip between rows
        consecutive = True
        if rule.kind == "zscore" or rule.persist > 1:
            ts = [parse_iso(r["slot_ts"]) for r in rows]
            max_gap = self.slot * 2 if rule.metric != "daily_mortality_pct" else timedelta(days=2)
            consecutive = all(a - b <= max_gap for a, b in zip(ts, ts[1:]))

        breached = (len(rows) >= rule.persist and consecutive and
                    all(lv in ("warning", "critical") for lv in levels))

        if not breached:
            if existing and existing["active"] and current == "ok":
                self.store.update_event(existing["id"], active=0, updated_ts=iso(slot))
            return None

        r0 = rows[0]
        severity = current
        now = iso(slot)
        if existing and parse_iso(existing["updated_ts"]) >= slot - timedelta(minutes=rule.cooldown_min):
            escalated = SEV_ORDER[severity] > SEV_ORDER.get(existing["severity"], 0)
            self.store.update_event(existing["id"], updated_ts=now, slot_ts=now, value=r0["value"],
                                    z=r0["z"], active=1,
                                    severity=severity if escalated else existing["severity"])
            if escalated:
                self._notify(existing["id"])
                return int(existing["id"])
            return None

        msg = MESSAGES.get(rule.id, f"{rule.metric} is outside its normal range.") + \
            (env_hint(self.store, pen, slot) if rule.signal in ("distribution", "activity") else "")
        details = {
            "metric": rule.metric, "baseline_median": r0["baseline_median"],
            "n_baseline": r0["n_baseline"], "age_days": r0["age_days"],
            "actions": ACTIONS.get(rule.signal, []),
            "signal_title": SIGNAL_TITLES.get(rule.signal, rule.signal),
        }
        eid = self.store.create_event(
            pen=pen, rule_id=rule.id, signal=rule.signal, severity=severity, opened_ts=now,
            updated_ts=now, slot_ts=now, value=r0["value"], z=r0["z"], message=msg,
            details=details, snapshots=snapshots() if callable(snapshots) else snapshots)
        self._notify(eid)
        return eid

    # ------------------------------------------------------------------------
    def raise_mortality_candidate(self, pen: str, slot: datetime, cand: dict, snapshots: list[str]) -> int:
        details = {"candidate_id": cand["id"], "x_m": cand["x_m"], "y_m": cand["y_m"],
                   "still_min": cand["still_min"], "first_seen": cand["first_slot"],
                   "dead_votes": cand.get("dead_votes") or 0,
                   "actions": ACTIONS["mortality"], "signal_title": SIGNAL_TITLES["mortality"]}
        msg = (MESSAGES["mortality_candidate"] +
               f" Approximate location: {cand['x_m']:.1f} m from the left wall, {cand['y_m']:.1f} m from the"
               f" far end. Still for about {cand['still_min']:.0f} minutes.")
        if cand.get("dead_votes"):
            msg += f" The detector classified this bird as dead in {cand['dead_votes']} clip(s)."
        eid = self.store.create_event(
            pen=pen, rule_id="mortality_candidate", signal="mortality", severity="critical",
            opened_ts=iso(slot), updated_ts=iso(slot), slot_ts=iso(slot), value=cand["still_min"],
            z=None, message=msg, details=details, snapshots=snapshots)
        self._notify(eid)
        return eid

    def _notify(self, event_id: int):
        e = self.store.event(event_id)
        if e is None:
            return
        d = dict(e)
        d["details"] = json.loads(d["details"] or "{}")
        d["snapshots"] = json.loads(d["snapshots"] or "[]")
        self.notifier.send(d)


def signal_states(cfg: Config, store: Store, pen: str) -> dict[str, dict[str, Any]]:
    """Current state of each of the 4 signals for the dashboard."""
    out: dict[str, dict[str, Any]] = {}
    for sig in ("activity", "distribution", "mortality", "circadian"):
        out[sig] = {"title": SIGNAL_TITLES[sig], "state": "learning", "metrics": {}, "events": 0}
    for rule in cfg.rules:
        r = store.latest_metric(pen, rule.metric)
        st = out[rule.signal]
        lv = rule_level(rule, r["value"], r["z"]) if r else None
        if r:
            st["metrics"][rule.metric] = {
                "value": r["value"], "z": r["z"], "baseline": r["baseline_median"],
                "n_baseline": r["n_baseline"], "slot_ts": r["slot_ts"]}
        if lv is None:
            continue
        if st["state"] == "learning" or SEV_ORDER[lv] > SEV_ORDER.get(st["state"], -1):
            st["state"] = lv
    # active events push the state up
    for e in store.q("SELECT signal, severity FROM events WHERE pen=? AND status='open' AND active=1",
                     (pen,)):
        st = out.get(e["signal"])
        if st is None:
            continue
        st["events"] += 1
        if st["state"] == "learning" or SEV_ORDER[e["severity"]] > SEV_ORDER.get(st["state"], -1):
            st["state"] = e["severity"]
    # mortality: open candidates
    cands = store.candidates(pen, ("open", "alerted"))
    alerted = [c for c in cands if c["status"] == "alerted"]
    m = out["mortality"]
    m["candidates_watching"] = len(cands) - len(alerted)
    m["candidates_alerted"] = len(alerted)
    if alerted:
        m["state"] = "critical"
    elif m["state"] == "learning" and store.latest_metric(pen, "activity_index"):
        m["state"] = "ok"
    return out
