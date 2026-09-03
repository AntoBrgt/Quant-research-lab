"""ISIN -> ticker resolution via OpenFIGI (free, no API key required).

This is deliberately separate from `portfolio_importers/` -- normalization
and position reconstruction stay network-free (see that package's docstring);
this module is an *optional*, explicit enrichment step, in the same spirit as
`price_provider.py`/`news_provider.py`. Nothing in `portfolio_importers/` or
`portfolio.py` imports this module or knows it exists.

Resolutions are cached on disk via `cache.py` (namespace "security_master",
keyed directly by the ISIN -- no hashing needed, ISINs are already short,
safe, inspectable filenames) so a given ISIN is only ever looked up once.

Honesty note: OpenFIGI's `ticker` field has no exchange suffix. For a US
listing that's directly usable as-is (e.g. "GOOGL"). For a non-US listing
it is NOT guaranteed to be a working Yahoo Finance ticker on its own --
Yahoo's exchange-suffix convention doesn't map cleanly from OpenFIGI's
`exchCode` for every exchange, and guessing wrong would produce a ticker
that looks valid but silently returns no price data, which is worse than
an honest "unresolved". So: a US-exchange match is returned with high
confidence; a non-US match is returned as a best-effort candidate, tagged
`confidence="best_effort"` -- callers should treat a subsequent failed
price lookup for one of these as expected, not a bug.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Protocol

import requests

import cache

logger = logging.getLogger(__name__)

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
# Unauthenticated limits: 25 requests/minute, 10 jobs/request. An API key
# (OPENFIGI_API_KEY) raises this to 25 requests/6s, 100 jobs/request -- see
# https://www.openfigi.com/api/documentation
BATCH_SIZE_NO_KEY = 10
BATCH_SIZE_WITH_KEY = 100
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES_ON_RATE_LIMIT = 2

# US-related OpenFIGI exchange codes -- a match against one of these is
# returned as a plain ticker with no suffix needed for Yahoo Finance / this
# project's TICKER_PATTERN, and is high-confidence.
_US_EXCH_CODES = {"US", "UN", "UW", "UQ", "UR", "UA"}


class SecurityMasterProvider(Protocol):
    def resolve_many(self, isins: list[str]) -> dict[str, Optional[dict]]:
        """Return {isin: {"ticker": str, "confidence": "high"|"best_effort", ...} | None}."""
        ...


def _api_key() -> Optional[str]:
    return os.getenv("OPENFIGI_API_KEY")


def _pick_candidate(candidates: list[dict]) -> Optional[dict]:
    """Choose one candidate from OpenFIGI's result list for one ISIN.

    Prefers a US-exchange listing (unambiguous, no suffix needed); otherwise
    takes the first candidate as a best-effort guess.
    """
    if not candidates:
        return None

    for candidate in candidates:
        if candidate.get("exchCode") in _US_EXCH_CODES:
            return {
                "ticker": candidate.get("ticker"),
                "name": candidate.get("name"),
                "exch_code": candidate.get("exchCode"),
                "security_type": candidate.get("securityType"),
                "confidence": "high",
            }

    first = candidates[0]
    return {
        "ticker": first.get("ticker"),
        "name": first.get("name"),
        "exch_code": first.get("exchCode"),
        "security_type": first.get("securityType"),
        "confidence": "best_effort",
    }


class OpenFIGIProvider:
    """Free, cached ISIN -> ticker resolution via OpenFIGI's /v3/mapping endpoint."""

    def __init__(self, session: Optional[requests.Session] = None):
        self._session = session or requests

    def _post(self, jobs: list[dict]) -> list[dict]:
        headers = {"Content-Type": "application/json"}
        api_key = _api_key()
        if api_key:
            headers["X-OPENFIGI-APIKEY"] = api_key

        for attempt in range(MAX_RETRIES_ON_RATE_LIMIT + 1):
            response = self._session.post(OPENFIGI_URL, json=jobs, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 429 and attempt < MAX_RETRIES_ON_RATE_LIMIT:
                wait = int(response.headers.get("retry-after", 6))
                logger.warning("OpenFIGI rate-limited, waiting %ss (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()

        raise RuntimeError("OpenFIGI rate limit exceeded after retries")

    def resolve_many(self, isins: list[str]) -> dict[str, Optional[dict]]:
        """Resolve a list of ISINs, using the disk cache for anything already looked up."""
        results: dict[str, Optional[dict]] = {}
        to_query: list[str] = []

        for isin in isins:
            cached = cache.get("security_master", isin)
            if cached is not None:
                results[isin] = cached["resolution"]
            else:
                to_query.append(isin)

        batch_size = BATCH_SIZE_WITH_KEY if _api_key() else BATCH_SIZE_NO_KEY
        for start in range(0, len(to_query), batch_size):
            batch = to_query[start : start + batch_size]
            jobs = [{"idType": "ID_ISIN", "idValue": isin} for isin in batch]

            try:
                responses = self._post(jobs)
            except (requests.RequestException, RuntimeError):
                logger.exception("OpenFIGI lookup failed for batch %s", batch)
                for isin in batch:
                    results[isin] = None
                continue

            for isin, job_response in zip(batch, responses):
                candidates = job_response.get("data")
                resolution = _pick_candidate(candidates) if candidates else None
                results[isin] = resolution
                cache.set("security_master", isin, {"resolution": resolution})

        return results


def resolve_isins(isins: list[str], provider: Optional[SecurityMasterProvider] = None) -> dict[str, Optional[dict]]:
    """Convenience function using the default (OpenFIGI) provider."""
    provider = provider or OpenFIGIProvider()
    return provider.resolve_many(isins)
