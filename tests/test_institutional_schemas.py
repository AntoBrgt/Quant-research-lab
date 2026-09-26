"""Institutional research schemas -- structural enforcement of "context, not a recommendation"."""

import pytest
from pydantic import ValidationError

from institutional_research.schemas import InstitutionalMention, InstitutionalReport, MentionExtraction


def _mention(**overrides):
    base = dict(
        institution="BlackRock",
        report_title="2026 Global Outlook",
        institutional_view="we favor European financials",
        view_direction="POSITIVE",
        confidence="High",
        evidence="We favor European financials given improving net interest margins.",
    )
    base.update(overrides)
    return InstitutionalMention(**base)


def test_theme_level_mention_needs_no_company_identifier():
    mention = _mention(region="Europe", sector="Financials")
    assert mention.company_name is None
    assert mention.ticker is None
    assert mention.view_direction == "POSITIVE"


def test_ticker_and_isin_are_normalized_to_uppercase():
    mention = _mention(ticker="bnp.pa", isin="fr0000131104")
    assert mention.ticker == "BNP.PA"
    assert mention.isin == "FR0000131104"


def test_invalid_view_direction_is_rejected():
    with pytest.raises(ValidationError):
        _mention(view_direction="BUY")  # never a recommendation label -- schema-enforced


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValidationError):
        _mention(confidence="0.87")  # categorical only, not a float probability


def test_mention_extraction_defaults_to_empty_list():
    extraction = MentionExtraction()
    assert extraction.mentions == []


def test_institutional_report_requires_traceability_fields():
    report = InstitutionalReport(
        institution="BlackRock",
        report_title="2026 Global Outlook",
        source_path="data/raw/institutional/BlackRock/outlook.pdf",
        content_hash="abc123",
        retrieved_at="2026-01-01T00:00:00+00:00",
        char_count=1000,
    )
    assert report.publication_date is None  # never guessed when not provided
    assert report.institution == "BlackRock"
