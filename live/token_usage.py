"""Per-API-call LLM token-usage tracking.

Deliberately its own append-only log, not folded into activity_log.py:
activity_log is a bounded sliding window (meant for "what is the agent doing
right now"), and a single research session can already burn 20-40 turns -
tracking cumulative token spend needs every call kept forever, not whatever
happens to survive the window.

Stores whatever fields the API's usage object actually returns (via
model_dump()) rather than hardcoding attribute names - DeepSeek's
OpenAI-compatible usage object includes extra fields (prompt_cache_hit_tokens,
prompt_cache_miss_tokens) beyond the standard prompt/completion/total split,
and this way a schema change upstream doesn't silently drop data.
"""
import json
import time
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "token_usage.jsonl"


def record(model: str, usage: dict) -> None:
    entry = {"ts": time.time(), "model": model, **usage}
    _PATH.parent.mkdir(exist_ok=True)
    with open(_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def all_records() -> list[dict]:
    if not _PATH.exists():
        return []
    out = []
    with open(_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
