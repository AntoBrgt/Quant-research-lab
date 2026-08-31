"""Extract structured financial signals from processed SEC filing chunks.

The script intentionally reads only the processed document dataset and never uses
future stock-price data or any raw SEC download logic. It is designed to be a
minimal, reproducible prototype for later quantitative research work.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover - optional dependency for local usage
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency for OpenAI usage
    ChatOpenAI = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "signals.parquet"

MODEL_NAME = os.getenv("OLLAMA_MODEL", os.getenv("OPENAI_MODEL", "llama3.2"))
TEMPERATURE = 0
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

SYSTEM_PROMPT = """
You are extracting observable financial signals from SEC filing text for a
quantitative research pipeline.

This is a signal-extraction step only. It must not generate investment advice,
portfolio recommendations, or action labels such as BUY, SELL, HOLD, or WATCH.
The later strategy and recommendation layers are separate.

Rules:
- Use only information explicitly present in the supplied document chunk.
- Do not use future market data or subsequent stock-price information.
- Do not provide investment advice, stock ratings, or buy/sell recommendations.
- Use only the allowed signal types: revenue_growth, earnings, margins,
  guidance, demand, pricing, costs, capital_expenditure, cash_flow, debt,
  liquidity, competition, regulation, management_confidence, risk.
- Ignore information that does not fit one of those types.
- Do not invent numerical values. If a metric is not clearly stated, use null.
- Strength must reflect the clarity and importance of the signal in the document,
  not the expected return of the stock.
- Evidence must be a short excerpt or close paraphrase that directly supports the
  signal.
- If the chunk contains no meaningful signal, return an empty list.
- Keep output structured and deterministic.
"""

ALLOWED_SIGNAL_TYPES = (
    "revenue_growth",
    "earnings",
    "margins",
    "guidance",
    "demand",
    "pricing",
    "costs",
    "capital_expenditure",
    "cash_flow",
    "debt",
    "liquidity",
    "competition",
    "regulation",
    "management_confidence",
    "risk",
)

PRIORITY_SECTIONS = (
    "MD&A",
    "Management's Discussion and Analysis",
    "Risk Factors",
    "Business",
    "Financial Statements",
    "Notes to Financial Statements",
    "Market Risk",
    "Quantitative and Qualitative Disclosures About Market Risk",
    "Legal Proceedings",
)

LOGGER_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)
logger = logging.getLogger(__name__)

if load_dotenv is not None:
    load_dotenv()


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class Signal(BaseModel):
    """One structured financial signal extracted from a document chunk."""

    signal_type: Literal[
        "revenue_growth",
        "earnings",
        "margins",
        "guidance",
        "demand",
        "pricing",
        "costs",
        "capital_expenditure",
        "cash_flow",
        "debt",
        "liquidity",
        "competition",
        "regulation",
        "management_confidence",
        "risk",
    ]
    direction: Literal["positive", "negative", "neutral"]
    strength: float = Field(..., ge=0.0, le=1.0)
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    metric_unit: Optional[str] = None
    growth_rate: Optional[float] = None
    evidence: str

    @field_validator("strength")
    @classmethod
    def validate_strength(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("strength must be between 0.0 and 1.0")
        return value

    @field_validator("metric_value", "growth_rate", mode="before")
    @classmethod
    def sanitize_numeric_values(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return value


class SignalExtraction(BaseModel):
    """Container returned by the structured LLM pipeline."""

    signals: list[Signal]


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def build_llm():
    """Build the configured local or cloud LLM client.

    The default path is a local Ollama model to avoid token costs during the
    research prototype phase.
    """
    if LLM_PROVIDER == "ollama":
        if ChatOllama is None:
            raise ImportError(
                "langchain-ollama is required for Ollama usage. Install it with pip."
            )

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=MODEL_NAME,
            base_url=base_url,
            temperature=TEMPERATURE,
        )

    if LLM_PROVIDER == "openai":
        if ChatOpenAI is None:
            raise ImportError(
                "langchain-openai is required for OpenAI usage. Install it with pip."
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your environment before running this script."
            )

        return ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            api_key=api_key,
        )

    raise ValueError(
        "Unsupported LLM_PROVIDER. Use 'ollama' or 'openai'."
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_documents(documents_path: Path = DOCUMENTS_PATH) -> pd.DataFrame:
    """Load processed SEC chunks from parquet."""
    if not documents_path.exists():
        raise FileNotFoundError(f"Processed documents file not found: {documents_path}")

    df = pd.read_parquet(documents_path)
    required_columns = {
        "ticker",
        "filing_type",
        "filing_date",
        "section",
        "chunk_id",
        "chunk_index",
        "text",
        "source_file",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"documents.parquet is missing required columns: {sorted(missing)}")

    return df


def filter_documents(
    df: pd.DataFrame,
    tickers: Optional[Iterable[str]] = None,
    max_chunks: Optional[int] = None,
) -> pd.DataFrame:
    """Filter inputs while prioritizing likely relevant filing sections."""
    filtered = df.copy()

    if tickers:
        allowed = {ticker.upper() for ticker in tickers}
        filtered = filtered[filtered["ticker"].astype(str).str.upper().isin(allowed)].copy()

    filtered["section_norm"] = filtered["section"].fillna("").astype(str).str.strip()

    priority_rank = {section: idx for idx, section in enumerate(PRIORITY_SECTIONS)}
    filtered["priority_rank"] = filtered["section_norm"].map(priority_rank).fillna(len(priority_rank))

    priority_mask = filtered["section_norm"].isin(PRIORITY_SECTIONS) | filtered["section_norm"].str.contains(
        "md&a|risk factors|business|financial statements|notes to financial statements|market risk",
        case=False,
        regex=True,
    )

    prioritized = filtered[priority_mask].copy()
    rest = filtered[~priority_mask].copy()

    ordered = pd.concat([prioritized, rest], ignore_index=True)
    ordered = ordered.sort_values(
        ["ticker", "priority_rank", "filing_date", "chunk_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if max_chunks is not None:
        ordered = ordered.head(max_chunks).copy()

    return ordered.drop(columns=["priority_rank"], errors="ignore")


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def build_extraction_prompt(row: dict) -> str:
    """Build the user prompt for a single chunk."""
    text = row.get("text", "")
    metadata_text = f"""
Ticker: {row.get('ticker')}
Filing type: {row.get('filing_type')}
Filing date: {row.get('filing_date')}
Section: {row.get('section')}
Chunk ID: {row.get('chunk_id')}
Source file: {row.get('source_file')}

Chunk text:
{text}
""".strip()

    return metadata_text


# ---------------------------------------------------------------------------
# Signal extraction pipeline
# ---------------------------------------------------------------------------

def extract_signals_from_chunk(llm, row: dict) -> list[Signal]:
    """Use the LLM to extract one or more structured signals from a chunk."""
    prompt = build_extraction_prompt(row)

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", "{input}"),
        ]
    )

    structured_llm = llm.with_structured_output(SignalExtraction)

    response = structured_llm.invoke(
        extraction_prompt.format(input=prompt)
    )

    if response is None:
        return []

    if isinstance(response, SignalExtraction):
        return response.signals

    if isinstance(response, dict):
        signals = response.get("signals", [])
        return [Signal(**signal) for signal in signals]

    return []


def process_chunk(llm, row: dict) -> list[dict]:
    """Process a single chunk and return rows for the final signals dataset."""
    try:
        signals = extract_signals_from_chunk(llm, row)
    except Exception:
        logger.exception(
            "LLM extraction failed for ticker=%s chunk_id=%s section=%s",
            row.get("ticker"),
            row.get("chunk_id"),
            row.get("section"),
        )
        return []

    if not signals:
        logger.info(
            "No signals found for %s | %s | chunk %s",
            row.get("ticker"),
            row.get("section"),
            row.get("chunk_id"),
        )
        return []

    signal_rows: list[dict] = []
    for signal in signals:
        signal_rows.append(
            {
                "ticker": row.get("ticker"),
                "filing_type": row.get("filing_type"),
                "filing_date": row.get("filing_date"),
                "section": row.get("section"),
                "chunk_id": row.get("chunk_id"),
                "chunk_index": row.get("chunk_index"),
                "signal_type": signal.signal_type,
                "direction": signal.direction,
                "strength": float(signal.strength),
                "metric_name": signal.metric_name,
                "metric_value": signal.metric_value,
                "metric_unit": signal.metric_unit,
                "growth_rate": signal.growth_rate,
                "evidence": signal.evidence,
                "source_file": row.get("source_file"),
            }
        )

    logger.info(
        "Extracted %d signals for %s | %s | chunk %s",
        len(signals),
        row.get("ticker"),
        row.get("section"),
        row.get("chunk_id"),
    )
    return signal_rows


def process_documents(
    documents: pd.DataFrame,
    tickers: Optional[Iterable[str]] = None,
    max_chunks: Optional[int] = None,
) -> pd.DataFrame:
    """Convert processed document chunks into a flattened signals dataset."""
    filtered = filter_documents(documents, tickers=tickers, max_chunks=max_chunks)

    llm = build_llm()
    rows: list[dict] = []
    chunks_processed = 0
    failed_chunks = 0

    for _, row in filtered.iterrows():
        chunks_processed += 1
        logger.info(
            "Processing %s | %s | chunk %s",
            row["ticker"],
            row["section"],
            row["chunk_id"],
        )

        try:
            chunk_rows = process_chunk(llm, row.to_dict())
            rows.extend(chunk_rows)
        except Exception:
            failed_chunks += 1
            logger.exception(
                "Unexpected processing error for %s chunk %s",
                row.get("ticker"),
                row.get("chunk_id"),
            )

    output = pd.DataFrame(rows)

    if output.empty:
        logger.warning("No signals were extracted from the selected documents")

    required_columns = [
        "ticker",
        "filing_type",
        "filing_date",
        "section",
        "chunk_id",
        "chunk_index",
        "signal_type",
        "direction",
        "strength",
        "metric_name",
        "metric_value",
        "metric_unit",
        "growth_rate",
        "evidence",
        "source_file",
    ]

    if not output.empty:
        output = output[required_columns]

    logger.info(
        "Files processed: %s | Chunks processed: %d | Signals extracted: %d | Failed chunks: %d",
        filtered["ticker"].nunique() if not filtered.empty else 0,
        chunks_processed,
        len(output),
        failed_chunks,
    )
    return output


def save_signals(signals: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Persist the signal rows to parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(output_path, index=False)
    logger.info("Saved %d signals to %s", len(signals), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured financial signals from processed SEC document chunks."
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        default=None,
        help="Optional ticker filter, e.g. --ticker AAPL MSFT JPM",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional cap for testing the pipeline on a small subset of chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    docs = load_documents()
    signals = process_documents(
        docs,
        tickers=args.ticker,
        max_chunks=args.max_chunks,
    )
    save_signals(signals)


if __name__ == "__main__":
    main()
