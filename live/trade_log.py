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
from typing import Optional

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
            except (FileExistsError, PermissionError):
                # Confirmed live (data/fundamentals.py hit this first, under
                # 10 concurrent threads): a create racing another thread's
                # delete of the same lock file can surface as PermissionError
                # instead of FileExistsError on Windows - a documented NTFS
                # quirk during a tight create/unlink race, not a real
                # permissions problem. Same "someone else has it, retry"
                # condition either way.
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
    # Links two LoggedTrade records (a put spread + a call spread) that were
    # opened together as one 4-leg iron condor - trade_log's schema stays at
    # the proven 2-leg shape rather than being reworked for a variable leg
    # count; a shared group_id is how close_both_legs finds all 4 real legs
    # instead of just the 2 in whichever record a symbol happens to be in.
    # None (the default, and what every pre-condor record already on disk
    # implicitly has) means "just a plain 2-leg spread, no linked sibling."
    group_id: Optional[str] = None


def _load() -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    return json.loads(_LOG_PATH.read_text())


def _save(trades: list[dict]) -> None:
    _LOG_PATH.parent.mkdir(exist_ok=True)
    _LOG_PATH.write_text(json.dumps(trades, indent=2))


def record_open(
    short_symbol: str,
    long_symbol: str,
    underlying: str,
    contracts: int,
    entry_credit: float,
    expiration: date,
    group_id: Optional[str] = None,
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
                    group_id=group_id,
                )
            )
        )
        _save(trades)


def record_close(symbol: str, realized_pnl: Optional[float] = None) -> None:
    """Marks the logged trade closed if `symbol` matches either leg -
    closing either side of a defined-risk spread means it's being
    unwound, regardless of which leg the agent closes first.

    `realized_pnl` (total dollars, both legs, all contracts) is optional
    and best-effort - the caller only has it when both legs' closing fills
    were confirmed in time (see order_helpers.close_both_legs). Guarded by
    `not t["closed"]` same as the closed flag itself, so whichever of the
    two legs' record_close calls runs first is the one that sticks; the
    caller passes the same value keyed under both legs' symbols so it
    doesn't matter which one wins the race.
    """
    with _FileLock():
        trades = _load()
        for t in trades:
            if symbol in (t["short_symbol"], t["long_symbol"]) and not t["closed"]:
                t["closed"] = True
                if realized_pnl is not None:
                    t["realized_pnl"] = realized_pnl
        _save(trades)


def open_trades() -> list[dict]:
    return [t for t in _load() if not t["closed"]]


def all_trades() -> list[dict]:
    return _load()
