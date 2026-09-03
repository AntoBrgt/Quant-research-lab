"""News headline access, behind a small provider interface.

This is deliberately thin and best-effort: yfinance's free `Ticker.news` is
the only source wired in for now (no API key, no separate dependency -- it's
already required for prices). Headlines get routed through the exact same
cache-first signal-extraction path as SEC chunks (see `signal_extraction.py`),
so a stronger paid news provider can replace this later without touching the
extraction, research, or strategy layers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol

logger = logging.getLogger(__name__)


class NewsProvider(Protocol):
    def get_headlines(self, ticker: str, limit: int = 10) -> list[dict]:
        """Return recent headlines as {title, publisher, link, published_at}."""
        ...


class YFinanceNewsProvider:
    """Free, best-effort news provider backed by yfinance's Ticker.news."""

    def get_headlines(self, ticker: str, limit: int = 10) -> list[dict]:
        import yfinance as yf  # imported lazily so tests never need it installed

        try:
            raw_items = yf.Ticker(ticker.upper()).news or []
        except Exception:
            logger.exception("Failed to fetch news for %s", ticker)
            return []

        headlines: list[dict] = []
        for item in raw_items[:limit]:
            # yfinance has changed its news payload shape across versions;
            # content can be nested under "content" or present at the top level.
            content = item.get("content", item)
            title = content.get("title")
            if not title:
                continue

            published = content.get("pubDate") or content.get("providerPublishTime")
            published_at = None
            if isinstance(published, (int, float)):
                published_at = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
            elif isinstance(published, str):
                published_at = published

            headlines.append(
                {
                    "title": title,
                    "publisher": (content.get("provider") or {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else content.get("publisher"),
                    "link": (content.get("canonicalUrl") or {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else content.get("link"),
                    "published_at": published_at,
                }
            )

        return headlines
