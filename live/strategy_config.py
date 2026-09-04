"""User-adjustable strategy thresholds and gate toggles, persisted to disk.

Read fresh (never cached in a module-level variable) by every consumer at
the moment of a signal check or gate decision - the whole point is that a
change made in the dashboard takes effect on the very next live check, not
just after a restart. Values live in data_cache/ alongside the other
runtime state (trade_log.json, disqualified_candidates.json), not in the
repo, since this is live-tunable operator state, not code.
"""
import json
from pathlib import Path
from threading import Lock

_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "strategy_config.json"
_lock = Lock()

DEFAULTS = {
    "vrp_threshold": 0.03,  # signals/options_quality.py's original hardcoded value
    "nvrp_threshold": 0.20,
    "z_threshold": 2.0,  # signals/mean_reversion.py's original hardcoded value
    "chop_threshold": 0.3,
    "earnings_gate_enabled": True,
    "news_gate_enabled": True,
}


def load() -> dict:
    with _lock:
        if not _PATH.exists():
            return dict(DEFAULTS)
        try:
            data = json.loads(_PATH.read_text())
        except json.JSONDecodeError:
            return dict(DEFAULTS)
    # Merge over DEFAULTS (not the other way) so a config file saved before
    # a new setting was added still picks up that setting's default instead
    # of a missing key blowing up a caller.
    return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}


def save(updates: dict) -> dict:
    current = load()
    current.update({k: v for k, v in updates.items() if k in DEFAULTS})
    with _lock:
        _PATH.parent.mkdir(exist_ok=True)
        _PATH.write_text(json.dumps(current, indent=2))
    return current
