"""Pydantic schemas for institutional research ingestion.

Two levels, matching the brief's section 3/4:

- `InstitutionalReport`: one ingested document (metadata + a reference to
  where it came from). Its raw text is not duplicated elsewhere beyond what's
  already on disk under `data/raw/institutional/` -- "store metadata and
  references rather than unnecessarily copying entire reports" (brief,
  section 2).
- `InstitutionalMention`: one extracted institutional theme/exposure row, the
  output of `parser.py`. This is deliberately NOT a recommendation. Per the
  brief's example: "we favor European financials" becomes
  `region=Europe, sector=Financials, view_direction=POSITIVE` -- it must never
  be auto-converted into "BUY BNP". `parser.py`'s prompt enforces this at
  extraction time; this schema enforces it structurally by having no
  action/rating field at all.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

ViewDirection = Literal["POSITIVE", "NEUTRAL", "NEGATIVE", "MENTIONED"]
# Categorical, not a float -- consistent with this project's "confidence is
# not probability of return" position (see recommendations.py / brief section
# 15). A model-assigned 0.87 would look like a precise probability; it isn't.
Confidence = Literal["High", "Medium", "Low"]


class InstitutionalReport(BaseModel):
    """Metadata for one ingested institutional research document."""

    institution: str
    report_title: str
    publication_date: Optional[str] = None  # ISO date string when parseable, else None -- never guessed
    report_url: Optional[str] = None
    report_type: Optional[str] = None  # free text (e.g. "outlook", "thematic note") -- providers aren't uniform
    source_path: str  # local file this was read from, for traceability
    content_hash: str  # sha256 of the extracted text, used as part of the LLM cache key
    retrieved_at: str  # ISO timestamp this report was ingested (not published)
    char_count: int


class InstitutionalMention(BaseModel):
    """One extracted institutional theme/exposure row -- an input to research, not a verdict."""

    institution: str
    report_title: str
    publication_date: Optional[str] = None
    report_url: Optional[str] = None
    report_type: Optional[str] = None

    investment_horizon: Optional[str] = None  # institution's own stated horizon, free text (e.g. "6-12 months")
    theme: Optional[str] = None
    region: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    asset_class: Optional[str] = None

    company_name: Optional[str] = None
    ticker: Optional[str] = None
    isin: Optional[str] = None

    institutional_view: str  # the institution's own words/close paraphrase, e.g. "we favor European financials"
    view_direction: ViewDirection
    confidence: Confidence
    evidence: str  # short excerpt from the report supporting this row

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: Optional[str]) -> Optional[str]:
        return value.strip().upper() if value else None

    @field_validator("isin")
    @classmethod
    def normalize_isin(cls, value: Optional[str]) -> Optional[str]:
        return value.strip().upper() if value else None


class MentionExtraction(BaseModel):
    """Container returned by the structured LLM pipeline for one report chunk."""

    mentions: list[InstitutionalMention] = Field(default_factory=list)


OUTPUT_COLUMNS = [
    "institution", "report_title", "publication_date", "report_url", "report_type",
    "investment_horizon", "theme", "region", "sector", "industry", "asset_class",
    "company_name", "ticker", "isin", "institutional_view", "view_direction",
    "confidence", "evidence", "source_path", "chunk_id",
]
