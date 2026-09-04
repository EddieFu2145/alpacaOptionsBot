"""Durable, append-only record of what the live agent has been doing -
tool calls, results, and its own narration - so a dashboard can show real
activity without depending on an ephemeral captured-stdout stream from
whatever process happened to launch the bot.
"""
import json
import time
from pathlib import Path
from typing import Any

_LOG_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "agent_activity.jsonl"


def log_event(event_type: str, **fields: Any) -> None:
    entry = {"ts": time.time(), "type": event_type, **fields}
    _LOG_PATH.parent.mkdir(exist_ok=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def recent_events(n: int = 300) -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    with open(_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn write from a concurrent process - skip rather than crash the dashboard
    return events


def _truncate(value: Any, limit: int = 800) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 15] + "... [truncated]"
