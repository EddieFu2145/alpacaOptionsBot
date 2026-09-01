"""A local, persistent record of trades this agent has opened.

Alpaca's own position data has no entry-timestamp field at all (confirmed:
absent from alpaca.trading.models.Position) - "how long has this been
held" has to be tracked here, not read from the broker.
"""
import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

_LOG_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "trade_log.json"
_LOCK_PATH = _LOG_PATH.with_suffix(".lock")


class _FileLock:
    """A portable mutex for read-modify-write access to the trade log.
    Without this, two agent sessions writing around the same moment (e.g.
    two different positions crossing an exit threshold seconds apart, each
    spawning its own agent run) can silently clobber each other's update -
    a real race, not a theoretical one, given the watcher can trigger
    overlapping sessions.
    """

    def __enter__(self):
        for _ in range(100):
            try:
                self._fd = open(_LOCK_PATH, "x")
                return self
            except FileExistsError:
                time.sleep(0.05)
        raise TimeoutError("Could not acquire the trade log lock after 5s")

    def __exit__(self, *exc_info):
        self._fd.close()
        _LOCK_PATH.unlink(missing_ok=True)


@dataclass
class LoggedTrade:
    short_symbol: str
    long_symbol: str
    underlying: str
    contracts: int
    entry_credit: float  # per spread, dollars per share
    entry_date: str  # ISO date
    expiration: str  # ISO date
    closed: bool = False


def _load() -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    return json.loads(_LOG_PATH.read_text())


def _save(trades: list[dict]) -> None:
    _LOG_PATH.parent.mkdir(exist_ok=True)
    _LOG_PATH.write_text(json.dumps(trades, indent=2))


def record_open(
    short_symbol: str, long_symbol: str, underlying: str, contracts: int, entry_credit: float, expiration: date
) -> None:
    with _FileLock():
        trades = _load()
        trades.append(
            asdict(
                LoggedTrade(
                    short_symbol=short_symbol,
                    long_symbol=long_symbol,
                    underlying=underlying,
                    contracts=contracts,
                    entry_credit=entry_credit,
                    entry_date=date.today().isoformat(),
                    expiration=expiration.isoformat(),
                )
            )
        )
        _save(trades)


def record_close(symbol: str) -> None:
    """Marks the logged trade closed if `symbol` matches either leg -
    closing either side of a defined-risk spread means it's being
    unwound, regardless of which leg the agent closes first."""
    with _FileLock():
        trades = _load()
        for t in trades:
            if symbol in (t["short_symbol"], t["long_symbol"]) and not t["closed"]:
                t["closed"] = True
        _save(trades)


def open_trades() -> list[dict]:
    return [t for t in _load() if not t["closed"]]
