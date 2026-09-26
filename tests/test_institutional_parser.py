"""Cache-first institutional mention extraction -- no network, FakeLLM double.

Mirrors `test_signal_extraction_cache.py`'s shape/assertions, adapted to the
mentions schema and its own cache namespace.
"""

import config
from institutional_research import parser


class _FakeMention:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return dict(self.__dict__)


class _FakeMentionExtraction:
    def __init__(self, mention_dicts):
        self.mentions = [_FakeMention(**d) for d in mention_dicts]


class FakeMentionLLM:
    def __init__(self, mentions: list[dict]):
        self._mentions = mentions
        self.call_count = 0

    def with_structured_output(self, schema, include_raw=False):
        return self

    def invoke(self, _prompt):
        self.call_count += 1
        return {"raw": None, "parsed": _FakeMentionExtraction(self._mentions), "parsing_error": None}


ITEM = {
    "institution": "BlackRock", "report_title": "2026 Global Outlook", "publication_date": "2026-01-15",
    "report_url": None, "report_type": "outlook", "source_path": "x.txt",
    "chunk_id": "blackrock_abc_000", "chunk_index": 0,
    "text": "We favor European financials given improving net interest margins.",
}

CANNED_MENTION = {
    "institution": "BlackRock", "report_title": "2026 Global Outlook", "publication_date": "2026-01-15",
    "report_url": None, "report_type": "outlook", "investment_horizon": "6-12 months",
    "theme": None, "region": "Europe", "sector": "Financials", "industry": None, "asset_class": "Equities",
    "company_name": None, "ticker": None, "isin": None,
    "institutional_view": "we favor European financials", "view_direction": "POSITIVE", "confidence": "High",
    "evidence": "We favor European financials given improving net interest margins.",
}


def test_first_run_is_cache_miss_and_calls_llm():
    llm = FakeMentionLLM([CANNED_MENTION])
    mentions, was_hit = parser.extract_mentions_for_chunk(ITEM, llm=llm)
    assert was_hit is False
    assert llm.call_count == 1
    assert mentions[0]["view_direction"] == "POSITIVE"


def test_second_identical_run_is_cache_hit():
    llm = FakeMentionLLM([CANNED_MENTION])
    parser.extract_mentions_for_chunk(ITEM, llm=llm)
    _, second_hit = parser.extract_mentions_for_chunk(ITEM, llm=llm)
    assert second_hit is True
    assert llm.call_count == 1


def test_runaway_output_is_capped(monkeypatch):
    monkeypatch.setattr(config, "MAX_MENTIONS_PER_CHUNK", 3)
    llm = FakeMentionLLM([CANNED_MENTION for _ in range(20)])
    mentions, _ = parser.extract_mentions_for_chunk(ITEM, llm=llm)
    assert len(mentions) == 3


def test_theme_level_mention_never_gains_a_company_or_action():
    """Section 3's core guarantee: no auto-conversion into a company recommendation."""
    llm = FakeMentionLLM([CANNED_MENTION])
    mentions, _ = parser.extract_mentions_for_chunk(ITEM, llm=llm)
    mention = mentions[0]
    assert mention["ticker"] is None
    assert mention["company_name"] is None
    assert "action" not in mention and "recommendation" not in mention


def test_chunk_report_splits_and_ids_are_stable():
    from institutional_research.schemas import InstitutionalReport

    report = InstitutionalReport(
        institution="BlackRock", report_title="2026 Global Outlook", source_path="a/b.txt",
        content_hash="hash", retrieved_at="2026-01-01T00:00:00+00:00", char_count=100,
    )
    items_1 = parser.chunk_report({"report": report, "text": "Short report text."})
    items_2 = parser.chunk_report({"report": report, "text": "Short report text."})
    assert items_1[0]["chunk_id"] == items_2[0]["chunk_id"]  # deterministic given the same source_path
    assert items_1[0]["institution"] == "BlackRock"


def test_run_extraction_reuses_cache_across_a_full_batch(monkeypatch):
    from institutional_research.schemas import InstitutionalReport

    report = InstitutionalReport(
        institution="BlackRock", report_title="2026 Global Outlook", source_path="a/b.txt",
        content_hash="hash", retrieved_at="2026-01-01T00:00:00+00:00", char_count=100,
    )
    loaded_reports = [{"report": report, "text": "We favor European financials given improving margins."}]

    llm = FakeMentionLLM([CANNED_MENTION])
    monkeypatch.setattr(parser.signal_extraction, "build_llm", lambda: llm)

    mentions_1, summary_1 = parser.run_extraction(loaded_reports)
    mentions_2, summary_2 = parser.run_extraction(loaded_reports)

    assert summary_1["llm_calls_made"] == 1
    assert summary_2["llm_calls_made"] == 0  # fully cached second run
    assert len(mentions_1) == len(mentions_2) == 1
