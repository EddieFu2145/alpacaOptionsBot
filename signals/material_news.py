"""Keyword-based material-news detector.

Built specifically to invalidate a mean-reversion thesis: a real,
significant news event (leadership change, M&A, legal/regulatory action,
guidance surprise, etc.) means an extreme price move may be a justified
repricing, not a statistical fluke likely to snap back - the exact
opposite of what mean reversion assumes.

Confirmed live why a simple "any recent news" check doesn't work: every
real large-cap tested (JPM, AAPL, TSLA, AVGO, PANW) had 3-15 news
articles in any given 3-day window just from routine ambient coverage
(analyst notes, "top stocks" roundups, minor mentions) - a raw count
threshold would reject every single candidate this system ever looks at.
Keyword matching on genuinely high-materiality event types is a coarser
but far more targeted signal. Confirmed working on a real, live case
found during development: Apple's actual CEO transition (Tim Cook to
John Ternus) showed up as 5 of 15 recent AAPL headlines, all matching
on "ceo" - exactly the kind of real catalyst this exists to catch.
"""
from datetime import datetime, timedelta

from data.news import get_news

LOOKBACK_DAYS = 3

_HIGH_IMPACT_KEYWORDS = [
    "ceo", "cfo", "resign", "steps down", "stepping down", "succeeds", "successor",
    "acquisition", "acquire", "acquires", "acquired", "merger", "merges", "buyout", "takeover",
    "lawsuit", "sues", "sued", "sec investigation", "ftc", "doj", "antitrust", "subpoena",
    "bankruptcy", "chapter 11", "restructuring", "recall", "data breach", "hack", "hacked",
    "fraud", "guidance cut", "profit warning", "restatement", "delisted", "delisting",
    "fda rejects", "fda approval", "clinical trial", "halted", "trading halt",
]


def material_news_check(underlying: str, lookback_days: int = LOOKBACK_DAYS) -> dict:
    """Args:
        underlying: Underlying ticker, e.g. AAPL.
        lookback_days: How many days back to scan (default 3, roughly the
            window a 2-sigma move would have played out over).
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    df = get_news(underlying, start, end, limit=50)

    matches = []
    for _, row in df.iterrows():
        text = f"{row.get('headline', '')} {row.get('summary', '')}".lower()
        hit_keywords = [kw for kw in _HIGH_IMPACT_KEYWORDS if kw in text]
        if hit_keywords:
            matches.append(
                {
                    "headline": row.get("headline"),
                    "matched_keywords": hit_keywords,
                    "created_at": str(row.get("created_at")),
                }
            )

    return {"underlying": underlying, "has_material_news": len(matches) > 0, "matches": matches}
