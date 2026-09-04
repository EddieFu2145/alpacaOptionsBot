"""Short-term memory of candidates that just failed a hard, largely
time-stable gate (material news, earnings) - a real news event or a
scheduled earnings date doesn't change within the cooldown window, so
there's no reason for every research session to re-derive the same
rejection from scratch, burning tool calls and turns re-explaining a
conclusion already reached minutes ago.

Deliberately narrow: does NOT cover VRP/mean-reversion/liquidity, which
genuinely fluctuate intraday (a z-score or NVRP reading minutes old can
already be stale) and must stay freshly checked every time.
"""
import json
import time
from pathlib import Path
from typing import Optional

_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "disqualified_candidates.json"
COOLDOWN_SECONDS = 60 * 60  # long enough to stop churn across back-to-back sessions, short enough that a genuine same-day news break still gets picked up


def _load() -> dict:
    if not _PATH.exists():
        return {}
    try:
        return json.loads(_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def record(underlying: str, reason: str) -> None:
    data = _load()
    data[underlying] = {"reason": reason, "ts": time.time()}
    _PATH.parent.mkdir(exist_ok=True)
    _PATH.write_text(json.dumps(data))


def check(underlying: str) -> Optional[str]:
    entry = _load().get(underlying)
    if not entry or time.time() - entry["ts"] > COOLDOWN_SECONDS:
        return None
    return entry["reason"]


def recent() -> dict[str, str]:
    now = time.time()
    return {k: v["reason"] for k, v in _load().items() if now - v["ts"] <= COOLDOWN_SECONDS}
