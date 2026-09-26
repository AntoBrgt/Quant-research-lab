"""Aggregate institutional mentions into an investable universe.

This module answers exactly one question: **why is this company in the
universe** (brief section 5.A) -- which institutions mention it, in which
reports, under which themes, and what direction they stated. It computes
NOTHING about whether the company is a good investment; that is
`research_engine.py` / `horizon.py` / `risk_reward.py`'s job (section 5.B),
using this module's output as one input alongside fundamentals/technicals/
signals, never as the final word.

A company enters the *universe* (an analyzable security) only if at least one
mention resolved to a `ticker` -- a theme/sector-level mention with no
identifiable company is real institutional context (surfaced via
`theme_summary()`) but isn't a security this project can price, chart, or
run fundamentals on.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DIRECTION_SCORE = {"POSITIVE": 1.0, "NEUTRAL": 0.0, "NEGATIVE": -1.0, "MENTIONED": 0.0}

# Coarse, documented-as-approximate country -> region mapping. yfinance's
# `.info` exposes `country`, not `region` -- this is a small, readable lookup,
# not a geopolitical authority. Anything not listed falls back to "Other".
_COUNTRY_TO_REGION = {
    "United States": "North America", "Canada": "North America", "Mexico": "North America",
    "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe", "Switzerland": "Europe",
    "Netherlands": "Europe", "Spain": "Europe", "Italy": "Europe", "Sweden": "Europe",
    "Ireland": "Europe", "Belgium": "Europe", "Denmark": "Europe", "Norway": "Europe", "Finland": "Europe",
    "Japan": "Asia", "China": "Asia", "South Korea": "Asia", "Taiwan": "Asia", "India": "Asia",
    "Singapore": "Asia", "Hong Kong": "Asia",
    "Australia": "Oceania", "New Zealand": "Oceania",
    "Brazil": "Latin America", "Argentina": "Latin America", "Chile": "Latin America",
}


def _region_for_country(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    return _COUNTRY_TO_REGION.get(country, "Other")


def _market_data_for(ticker: str) -> dict:
    """Best-effort company metadata from yfinance -- same honesty rules as
    elsewhere in this project: missing fields are None, never guessed.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
    except Exception:
        logger.warning("Could not fetch market data for %s", ticker)
        return {"country": None, "region": None, "sector": None, "industry": None, "market_cap": None}

    country = info.get("country")
    return {
        "country": country,
        "region": _region_for_country(country),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
    }


def aggregate_direction(directions: pd.Series) -> tuple[float, str]:
    scores = directions.map(_DIRECTION_SCORE).dropna()
    if scores.empty:
        return 0.0, "MENTIONED"
    mean_score = float(scores.mean())
    if mean_score > 0.25:
        label = "POSITIVE"
    elif mean_score < -0.25:
        label = "NEGATIVE"
    elif (directions.isin(["POSITIVE", "NEGATIVE"])).any():
        label = "NEUTRAL"  # real, but offsetting, directional views
    else:
        label = "MENTIONED"
    return round(mean_score, 4), label


UNIVERSE_COLUMNS = [
    "ticker", "company_name", "country", "region", "sector", "industry", "market_cap",
    "themes", "institution_count", "institutional_mentions",
    "institutional_direction", "institutional_direction_score", "latest_mention_date",
]


def build_universe(mentions: pd.DataFrame, enrich_market_data: bool = True) -> pd.DataFrame:
    """Aggregate mention rows (institutional_research.parser output) into one row per ticker.

    Rows with no `ticker` are theme/sector-level context only and are excluded
    from the returned universe (see `theme_summary()` for those) -- they are
    not silently lost, just not represented as a security here.
    """
    if mentions.empty:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)

    with_ticker = mentions[mentions["ticker"].notna() & (mentions["ticker"].astype(str).str.strip() != "")].copy()
    if with_ticker.empty:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)

    rows = []
    for ticker, group in with_ticker.groupby("ticker"):
        direction_score, direction_label = aggregate_direction(group["view_direction"])
        themes = sorted({t for t in group["theme"].dropna().unique().tolist() if t})
        company_name = group["company_name"].dropna().mode()
        publication_dates = pd.to_datetime(group["publication_date"], errors="coerce").dropna()

        market_data = _market_data_for(ticker) if enrich_market_data else {
            "country": None, "region": None, "sector": None, "industry": None, "market_cap": None,
        }
        # Fall back to whatever the mentions themselves stated when market
        # data enrichment is off or the lookup came back empty.
        mention_sector = group["sector"].dropna().mode()
        mention_region = group["region"].dropna().mode()

        rows.append(
            {
                "ticker": ticker,
                "company_name": company_name.iloc[0] if not company_name.empty else None,
                "country": market_data["country"],
                "region": market_data["region"] or (mention_region.iloc[0] if not mention_region.empty else None),
                "sector": market_data["sector"] or (mention_sector.iloc[0] if not mention_sector.empty else None),
                "industry": market_data["industry"],
                "market_cap": market_data["market_cap"],
                "themes": themes,
                "institution_count": group["institution"].nunique(),
                "institutional_mentions": len(group),
                "institutional_direction": direction_label,
                "institutional_direction_score": direction_score,
                "latest_mention_date": str(publication_dates.max().date()) if not publication_dates.empty else None,
            }
        )

    universe = pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)
    return universe.sort_values(["institution_count", "institutional_mentions"], ascending=False).reset_index(drop=True)


def theme_summary(mentions: pd.DataFrame) -> pd.DataFrame:
    """Theme -> institution count / mention count / companies exposed -- section 4's
    "institutional reports -> theme -> sub-themes -> companies" view, independent
    of whether a specific ticker was identified for every row.
    """
    if mentions.empty or "theme" not in mentions.columns:
        return pd.DataFrame(columns=["theme", "institution_count", "mention_count", "companies"])

    themed = mentions[mentions["theme"].notna() & (mentions["theme"].astype(str).str.strip() != "")]
    if themed.empty:
        return pd.DataFrame(columns=["theme", "institution_count", "mention_count", "companies"])

    rows = []
    for theme, group in themed.groupby("theme"):
        companies = sorted({t for t in group["ticker"].dropna().unique().tolist() if t})
        rows.append(
            {
                "theme": theme,
                "institution_count": group["institution"].nunique(),
                "mention_count": len(group),
                "companies": companies,
            }
        )
    return pd.DataFrame(rows).sort_values(["institution_count", "mention_count"], ascending=False).reset_index(drop=True)


def why_in_universe(ticker: str, mentions: pd.DataFrame) -> pd.DataFrame:
    """Every mention row that put this ticker in the universe -- the literal
    answer to "why is this company in the universe" for the company research
    page: which institution, which report, which theme, what view, when.
    """
    if mentions.empty:
        return mentions
    ticker = ticker.upper()
    return mentions[mentions["ticker"].astype(str).str.upper() == ticker].sort_values(
        "publication_date", ascending=False
    )
