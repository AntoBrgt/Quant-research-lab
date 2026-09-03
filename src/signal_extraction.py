"""Cache-first LLM signal extraction -- the reusable core of the pipeline.

This is the one place LLM calls happen. Every call goes:

    input -> normalize -> hash -> cache key -> cache lookup
        HIT  -> return cached result (no LLM call)
        MISS -> call LLM (with a hang guard) -> validate -> cap runaway output
                -> cache result -> record usage -> return result

`extract_signals.py` is a thin CLI over `run_extraction`/`estimate_run` here;
this module has no dependency on it, so there is no import cycle. The
Signal/SignalExtraction schema lives here because it is what the cache/
validation step actually produces and checks, not because it is CLI-facing.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Iterable, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover - optional dependency for local usage
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency for OpenAI usage
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover - optional dependency for Anthropic usage
    ChatAnthropic = None

import cache
import config
import llm_usage

logger = logging.getLogger(__name__)

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
    "News",
)

SYSTEM_PROMPT = """
You are extracting observable financial signals from text (SEC filings or news
headlines) for a quantitative research pipeline.

This is a signal-extraction step only. It must not generate investment advice,
portfolio recommendations, or action labels such as BUY, SELL, HOLD, or WATCH.
The later strategy and recommendation layers are separate.

Rules:
- Use only information explicitly present in the supplied text.
- Do not use future market data or subsequent stock-price information.
- Do not provide investment advice, stock ratings, or buy/sell recommendations.
- Use only the allowed signal types: revenue_growth, earnings, margins,
  guidance, demand, pricing, costs, capital_expenditure, cash_flow, debt,
  liquidity, competition, regulation, management_confidence, risk.
- Ignore information that does not fit one of those types.
- Do not invent numerical values. If a metric is not clearly stated, use null.
- Strength must reflect the clarity and importance of the signal in the text,
  not the expected return of the stock.
- Evidence must be a short excerpt or close paraphrase that directly supports the
  signal.
- If the text contains no meaningful signal, return an empty list.
- Report at most a handful of the clearest, most important signals -- do not
  restate the same point multiple times or enumerate every sentence.
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


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class Signal(BaseModel):
    """One structured financial signal extracted from a document chunk or headline."""

    signal_type: Literal[
        "revenue_growth", "earnings", "margins", "guidance", "demand", "pricing",
        "costs", "capital_expenditure", "cash_flow", "debt", "liquidity",
        "competition", "regulation", "management_confidence", "risk",
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
# LLM provider (this is the abstraction point -- swap provider via config)
# ---------------------------------------------------------------------------

def build_llm():
    """Build the configured LLM client (Ollama, OpenAI, or Anthropic)."""
    if config.LLM_PROVIDER == "ollama":
        if ChatOllama is None:
            raise ImportError("langchain-ollama is required for Ollama usage. Install it with pip.")
        return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)

    if config.LLM_PROVIDER == "openai":
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is required for OpenAI usage. Install it with pip.")
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your environment before running this script.")
        return ChatOpenAI(model=config.LLM_MODEL, temperature=0, api_key=api_key)

    if config.LLM_PROVIDER == "anthropic":
        if ChatAnthropic is None:
            raise ImportError("langchain-anthropic is required for Anthropic usage. Install it with pip.")
        import os

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your environment before running this script.")
        return ChatAnthropic(model=config.LLM_MODEL, temperature=0, api_key=api_key)

    raise ValueError("Unsupported LLM_PROVIDER. Use 'ollama', 'openai', or 'anthropic'.")


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def build_extraction_prompt(item: dict) -> str:
    text = (item.get("text") or "")[: config.MAX_LLM_INPUT_CHARS]
    return f"""
Ticker: {item.get('ticker')}
Filing type: {item.get('filing_type')}
Filing date: {item.get('filing_date')}
Section: {item.get('section')}
Chunk ID: {item.get('chunk_id')}
Source file: {item.get('source_file')}

Text:
{text}
""".strip()


_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("user", "{input}")]
)


def _invoke_llm(llm, prompt_text: str):
    """Call the LLM with structured output, returning (parsed, raw_message)."""
    structured_llm = llm.with_structured_output(SignalExtraction, include_raw=True)
    result = structured_llm.invoke(_EXTRACTION_PROMPT.format(input=prompt_text))
    return result.get("parsed"), result.get("raw"), result.get("parsing_error")


def _extract_tokens(raw_message) -> tuple[Optional[int], Optional[int]]:
    """Pull real token counts from provider response metadata, else (None, None)."""
    usage = getattr(raw_message, "usage_metadata", None) if raw_message is not None else None
    if not usage:
        return None, None
    return usage.get("input_tokens"), usage.get("output_tokens")


# ---------------------------------------------------------------------------
# Cache-first extraction for one item
# ---------------------------------------------------------------------------

def extract_signal_for_item(item: dict, source: str, llm, skip_llm_call: bool = False) -> tuple[list[dict], bool]:
    """Cache-first extraction for one chunk/headline. Never raises on LLM failure.

    Returns (signals, was_cache_hit). If `skip_llm_call` is True and the item is
    not already cached, no LLM call is made and ([], False) is returned -- this
    is how `run_extraction` enforces `MAX_LLM_CALLS_PER_RUN` without blocking
    already-cached items.
    """
    text = item.get("text") or ""
    operation = f"signal_extraction:{source}"
    key = cache.build_cache_key(operation, text, config.LLM_MODEL)

    cached = cache.get("signals", key)
    if cached is not None:
        llm_usage.record_call(
            provider=config.LLM_PROVIDER, model=config.LLM_MODEL, operation=operation,
            cache_hit=True, cache_key=key,
        )
        return cached["signals"], True

    if skip_llm_call:
        return [], False

    prompt_text = build_extraction_prompt(item)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke_llm, llm, prompt_text)
            parsed, raw, parsing_error = future.result(timeout=config.LLM_REQUEST_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        logger.error(
            "LLM call timed out after %ss for %s | %s | chunk %s -- skipping, not cached",
            config.LLM_REQUEST_TIMEOUT_SECONDS, item.get("ticker"), item.get("section"), item.get("chunk_id"),
        )
        return [], False
    except Exception:
        logger.exception(
            "LLM extraction failed for ticker=%s chunk_id=%s section=%s",
            item.get("ticker"), item.get("chunk_id"), item.get("section"),
        )
        return [], False

    input_tokens, output_tokens = _extract_tokens(raw)

    if parsed is None or parsing_error:
        logger.warning(
            "Structured output validation failed for %s | %s | chunk %s -- not cached, will retry next run",
            item.get("ticker"), item.get("section"), item.get("chunk_id"),
        )
        llm_usage.record_call(
            provider=config.LLM_PROVIDER, model=config.LLM_MODEL, operation=operation,
            cache_hit=False, cache_key=key, input_tokens=input_tokens, output_tokens=output_tokens,
        )
        return [], False

    signals = parsed.signals
    if len(signals) > config.MAX_SIGNALS_PER_CHUNK:
        logger.warning(
            "Runaway generation detected for %s | %s | chunk %s: %d signals, truncating to %d",
            item.get("ticker"), item.get("section"), item.get("chunk_id"),
            len(signals), config.MAX_SIGNALS_PER_CHUNK,
        )
        signals = signals[: config.MAX_SIGNALS_PER_CHUNK]

    signal_dicts = [signal.model_dump() for signal in signals]
    cache.set("signals", key, {"signals": signal_dicts})

    llm_usage.record_call(
        provider=config.LLM_PROVIDER, model=config.LLM_MODEL, operation=operation,
        cache_hit=False, cache_key=key, input_tokens=input_tokens, output_tokens=output_tokens,
    )

    return signal_dicts, False


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def filter_items(
    df: pd.DataFrame,
    tickers: Optional[Iterable[str]] = None,
    max_chunks: Optional[int] = None,
) -> pd.DataFrame:
    """Filter inputs while prioritizing likely relevant sections."""
    filtered = df.copy()

    if tickers:
        allowed = {ticker.upper() for ticker in tickers}
        filtered = filtered[filtered["ticker"].astype(str).str.upper().isin(allowed)].copy()

    filtered["section_norm"] = filtered["section"].fillna("").astype(str).str.strip()
    priority_rank = {section: idx for idx, section in enumerate(PRIORITY_SECTIONS)}
    filtered["priority_rank"] = filtered["section_norm"].map(priority_rank).fillna(len(priority_rank))

    ordered = filtered.sort_values(
        ["ticker", "priority_rank", "filing_date", "chunk_id"], kind="mergesort"
    ).reset_index(drop=True)

    if max_chunks is not None:
        ordered = ordered.head(max_chunks).copy()

    return ordered.drop(columns=["priority_rank", "section_norm"], errors="ignore")


def estimate_run(df: pd.DataFrame, source: str) -> dict:
    """Dry-run cost estimate. Makes NO LLM calls."""
    total = len(df)
    if total == 0:
        return {
            "documents": 0, "unique_chunks": 0, "cache_hits": 0, "cache_misses": 0,
            "llm_calls_required": 0, "estimated_input_tokens": None,
        }

    operation = f"signal_extraction:{source}"
    keyed_texts: dict[str, str] = {}
    for row in df.to_dict("records"):
        text = row.get("text") or ""
        key = cache.build_cache_key(operation, text, config.LLM_MODEL)
        keyed_texts.setdefault(key, text)  # first text wins; identical content shares a key anyway

    hit_keys = {key for key in keyed_texts if cache.exists("signals", key)}
    cache_hits = len(hit_keys)
    cache_misses = len(keyed_texts) - cache_hits

    # Rough, clearly-labeled approximation -- never presented as real usage.
    miss_chars = sum(len(text) for key, text in keyed_texts.items() if key not in hit_keys)
    estimated_input_tokens = miss_chars // 4

    return {
        "documents": total,
        "unique_chunks": len(keyed_texts),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "llm_calls_required": cache_misses,
        "estimated_input_tokens": f"~{estimated_input_tokens} (approx, 4 chars/token)",
    }


OUTPUT_COLUMNS = [
    "ticker", "filing_type", "filing_date", "section", "chunk_id", "chunk_index",
    "signal_type", "direction", "strength", "metric_name", "metric_value",
    "metric_unit", "growth_rate", "evidence", "source_file",
]


def run_extraction(
    df: pd.DataFrame,
    source: str = "sec",
    tickers: Optional[Iterable[str]] = None,
    max_chunks: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """Cache-first extraction over a batch of chunks/headlines. Real LLM calls.

    `df` must have columns: ticker, filing_type, filing_date, section, chunk_id,
    chunk_index, text, source_file (news items adapt to this same shape).
    `MAX_LLM_CALLS_PER_RUN` caps actual LLM calls, not cached items -- once the
    cap is hit, already-cached items still resolve for free; only new items
    that would require a real call are skipped (and logged) for this run.
    """
    filtered = filter_items(df, tickers=tickers, max_chunks=max_chunks)

    llm = build_llm()
    rows: list[dict] = []
    chunks_processed = 0
    llm_calls_made = 0
    llm_calls_skipped_over_limit = 0

    for _, row in filtered.iterrows():
        chunks_processed += 1
        item = row.to_dict()
        logger.info("Processing %s | %s | chunk %s", item.get("ticker"), item.get("section"), item.get("chunk_id"))

        skip_llm_call = llm_calls_made >= config.MAX_LLM_CALLS_PER_RUN
        signals, was_cache_hit = extract_signal_for_item(item, source=source, llm=llm, skip_llm_call=skip_llm_call)

        if skip_llm_call and not was_cache_hit:
            llm_calls_skipped_over_limit += 1
            continue
        if not was_cache_hit:
            llm_calls_made += 1

        if not signals:
            logger.info("No signals found for %s | %s | chunk %s", item.get("ticker"), item.get("section"), item.get("chunk_id"))
            continue

        for signal in signals:
            rows.append({**{col: item.get(col) for col in OUTPUT_COLUMNS if col not in signal}, **signal})

    if llm_calls_skipped_over_limit:
        logger.warning(
            "MAX_LLM_CALLS_PER_RUN=%d reached; %d uncached item(s) skipped this run (re-run to continue them)",
            config.MAX_LLM_CALLS_PER_RUN, llm_calls_skipped_over_limit,
        )

    output = pd.DataFrame(rows)
    if not output.empty:
        for col in OUTPUT_COLUMNS:
            if col not in output.columns:
                output[col] = None
        output = output[OUTPUT_COLUMNS]

    summary = {
        "tickers": filtered["ticker"].nunique() if not filtered.empty else 0,
        "chunks_processed": chunks_processed,
        "llm_calls_made": llm_calls_made,
        "llm_calls_skipped_over_limit": llm_calls_skipped_over_limit,
        "signals_extracted": len(output),
    }
    logger.info(
        "Chunks processed: %d | LLM calls: %d | Signals extracted: %d",
        chunks_processed, llm_calls_made, len(output),
    )
    return output, summary
