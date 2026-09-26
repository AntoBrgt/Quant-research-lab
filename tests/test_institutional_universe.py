"""Universe aggregation -- pure pandas, no network (market data enrichment disabled)."""

import pandas as pd

from institutional_research import universe


def _mentions_df():
    return pd.DataFrame(
        [
            {
                "institution": "BlackRock", "report_title": "Outlook A", "publication_date": "2026-01-10",
                "theme": "AI infrastructure", "region": "North America", "sector": "Technology",
                "company_name": "Nvidia", "ticker": "NVDA", "view_direction": "POSITIVE",
            },
            {
                "institution": "J.P. Morgan Asset Management", "report_title": "Outlook B", "publication_date": "2026-02-01",
                "theme": "AI infrastructure", "region": "North America", "sector": "Technology",
                "company_name": "Nvidia", "ticker": "NVDA", "view_direction": "POSITIVE",
            },
            {
                "institution": "BlackRock", "report_title": "Outlook A", "publication_date": "2026-01-10",
                "theme": "European financials", "region": "Europe", "sector": "Financials",
                "company_name": None, "ticker": None, "view_direction": "POSITIVE",
            },
            {
                "institution": "Goldman Sachs", "report_title": "Sector Note", "publication_date": "2026-01-20",
                "theme": "Power demand", "region": "North America", "sector": "Utilities",
                "company_name": "NextEra Energy", "ticker": "NEE", "view_direction": "NEGATIVE",
            },
        ]
    )


def test_universe_excludes_theme_only_rows_with_no_ticker():
    result = universe.build_universe(_mentions_df(), enrich_market_data=False)
    assert set(result["ticker"]) == {"NVDA", "NEE"}


def test_universe_counts_institutions_and_mentions_per_company():
    result = universe.build_universe(_mentions_df(), enrich_market_data=False)
    nvda = result[result["ticker"] == "NVDA"].iloc[0]
    assert nvda["institution_count"] == 2  # BlackRock + JPM AM
    assert nvda["institutional_mentions"] == 2
    assert nvda["themes"] == ["AI infrastructure"]
    assert nvda["latest_mention_date"] == "2026-02-01"


def test_direction_aggregation_reflects_consistent_positive_view():
    result = universe.build_universe(_mentions_df(), enrich_market_data=False)
    nvda = result[result["ticker"] == "NVDA"].iloc[0]
    nee = result[result["ticker"] == "NEE"].iloc[0]
    assert nvda["institutional_direction"] == "POSITIVE"
    assert nee["institutional_direction"] == "NEGATIVE"


def test_a_company_can_belong_to_multiple_themes():
    mentions = pd.concat(
        [
            _mentions_df(),
            pd.DataFrame(
                [{
                    "institution": "UBS", "report_title": "Note C", "publication_date": "2026-01-25",
                    "theme": "Semiconductors", "region": "North America", "sector": "Technology",
                    "company_name": "Nvidia", "ticker": "NVDA", "view_direction": "MENTIONED",
                }]
            ),
        ],
        ignore_index=True,
    )
    result = universe.build_universe(mentions, enrich_market_data=False)
    nvda = result[result["ticker"] == "NVDA"].iloc[0]
    assert set(nvda["themes"]) == {"AI infrastructure", "Semiconductors"}


def test_empty_mentions_returns_empty_universe_with_expected_columns():
    result = universe.build_universe(pd.DataFrame(), enrich_market_data=False)
    assert result.empty
    assert list(result.columns) == universe.UNIVERSE_COLUMNS


def test_theme_summary_groups_across_companies_and_theme_only_rows():
    summary = universe.theme_summary(_mentions_df())
    ai_row = summary[summary["theme"] == "AI infrastructure"].iloc[0]
    assert ai_row["institution_count"] == 2
    assert ai_row["companies"] == ["NVDA"]

    financials_row = summary[summary["theme"] == "European financials"].iloc[0]
    assert financials_row["companies"] == []  # a real theme with no identified company -- not dropped


def test_why_in_universe_returns_only_that_tickers_rows():
    result = universe.why_in_universe("NVDA", _mentions_df())
    assert len(result) == 2
    assert set(result["institution"]) == {"BlackRock", "J.P. Morgan Asset Management"}
