"""Cache-first LLM extraction of institutional mentions from report text.

Deliberately mirrors `signal_extraction.py`'s shape (same cache-key
derivation, same hang/runaway guards, same `build_llm()`) rather than
reinventing it -- the two pipelines share `cache.py`, `config.py`,
`llm_usage.py`, and `signal_extraction.build_llm()` directly. They use
separate cache namespaces ("institutional_mentions" vs "signals") and
separate pydantic schemas, because what's being extracted (an institution's
stated theme/exposure) is conceptually different from a SEC-filing signal --
in particular a mention never has a strength/direction on the *security*
itself, only the institution's own stated view.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

import cache
import config
import llm_usage
import process_documents  # reused for split_into_chunks -- generic text chunking, not SEC-specific
import signal_extraction  # reused for build_llm() -- same provider abstraction, same config-driven model choice
from institutional_research.schemas import MentionExtraction, OUTPUT_COLUMNS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are extracting institutional research context from investment research
text published by asset managers/institutions (e.g. BlackRock Investment
Institute, J.P. Morgan Asset Management, Goldman Sachs, Morgan Stanley, UBS,
Fidelity, Vanguard).

This is a context-extraction step only. It must NEVER produce investment
advice, a rating, or an action label such as BUY, SELL, HOLD, or WATCH, and it
must NEVER convert an institution's stated view into one. For example,
"we favor European financials" becomes:
    region=Europe, sector=Financials, institutional_view="we favor European
    financials", view_direction=POSITIVE
It must NOT become "BUY BNP" or any other specific trade instruction, even if
a company in that sector is mentioned elsewhere in the text.

Rules:
- Use only information explicitly present in the supplied excerpt.
- A mention does NOT need a specific company: theme/region/sector/asset-class
  level views (company_name, ticker, isin left null) are valid and expected,
  and are usually the majority of what a macro/thematic report contains.
- Only set ticker/isin when the text itself states or unambiguously implies a
  specific, identifiable company -- never guess a ticker for a named company.
- view_direction is POSITIVE, NEUTRAL, NEGATIVE, or MENTIONED (MENTIONED = the
  theme/company is referenced without a clear directional view).
- confidence (High/Medium/Low) reflects how explicitly/clearly the text
  states this view, NOT a probability of any market outcome.
- institutional_view must be the institution's own words or a close, faithful
  paraphrase -- do not editorialize or add a view the text doesn't state.
- evidence must be a short excerpt that directly supports the row.
- If the excerpt contains no identifiable institutional theme or view, return
  an empty list.
- Report at most a handful of the clearest, most important mentions -- do not
  enumerate every sentence or restate the same theme multiple times.
"""

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", "{input}")])


def build_extraction_prompt(item: dict) -> str:
    text = (item.get("text") or "")[: config.MAX_LLM_INPUT_CHARS]
    return f"""
Institution: {item.get('institution')}
Report title: {item.get('report_title')}
Publication date: {item.get('publication_date')}
Report type: {item.get('report_type')}
Chunk ID: {item.get('chunk_id')}

Text:
{text}
""".strip()


def _invoke_llm(llm, prompt_text: str):
    structured_llm = llm.with_structured_output(MentionExtraction, include_raw=True)
    result = structured_llm.invoke(_EXTRACTION_PROMPT.format(input=prompt_text))
    return result.get("parsed"), result.get("raw"), result.get("parsing_error")


def chunk_report(report_row: dict) -> list[dict]:
    """Split one loaded report ({"report": InstitutionalReport, "text": str}) into chunk items.

    Reuses `process_documents.split_into_chunks` -- generic paragraph-aware
    text chunking, nothing SEC-specific about it.
    """
    report = report_row["report"]
    text = report_row["text"]
    chunks = process_documents.split_into_chunks(text)

    institution_slug = cache.normalize_text(report.institution).replace(" ", "_")[:40]
    report_slug = cache.content_hash(report.source_path)[:10]

    items = []
    for index, chunk_text in enumerate(chunks):
        items.append(
            {
                "institution": report.institution,
                "report_title": report.report_title,
                "publication_date": report.publication_date,
                "report_url": report.report_url,
                "report_type": report.report_type,
                "source_path": report.source_path,
                "chunk_id": f"{institution_slug}_{report_slug}_{index:03d}",
                "chunk_index": index,
                "text": chunk_text,
            }
        )
    return items


def extract_mentions_for_chunk(item: dict, llm, skip_llm_call: bool = False) -> tuple[list[dict], bool]:
    """Cache-first extraction for one report chunk. Never raises on LLM failure.

    Same contract as `signal_extraction.extract_signal_for_item`: returns
    (mentions, was_cache_hit); if `skip_llm_call` and not cached, returns
    ([], False) without calling the LLM, so `run_extraction` can enforce
    `MAX_LLM_CALLS_PER_RUN`.
    """
    text = item.get("text") or ""
    operation = "institutional_mention_extraction"
    key = cache.build_cache_key(operation, text, config.LLM_MODEL)

    cached = cache.get("institutional_mentions", key)
    if cached is not None:
        llm_usage.record_call(provider=config.LLM_PROVIDER, model=config.LLM_MODEL, operation=operation, cache_hit=True, cache_key=key)
        return cached["mentions"], True

    if skip_llm_call:
        return [], False

    prompt_text = build_extraction_prompt(item)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke_llm, llm, prompt_text)
            parsed, raw, parsing_error = future.result(timeout=config.LLM_REQUEST_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        logger.error("LLM call timed out after %ss for %s | chunk %s -- skipping, not cached", config.LLM_REQUEST_TIMEOUT_SECONDS, item.get("institution"), item.get("chunk_id"))
        return [], False
    except Exception:
        logger.exception("Institutional mention extraction failed for %s | chunk %s", item.get("institution"), item.get("chunk_id"))
        return [], False

    input_tokens, output_tokens = (None, None)
    usage = getattr(raw, "usage_metadata", None) if raw is not None else None
    if usage:
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")

    if parsed is None or parsing_error:
        logger.warning("Structured output validation failed for %s | chunk %s -- not cached, will retry next run", item.get("institution"), item.get("chunk_id"))
        llm_usage.record_call(provider=config.LLM_PROVIDER, model=config.LLM_MODEL, operation=operation, cache_hit=False, cache_key=key, input_tokens=input_tokens, output_tokens=output_tokens)
        return [], False

    mentions = parsed.mentions
    if len(mentions) > config.MAX_MENTIONS_PER_CHUNK:
        logger.warning("Runaway generation detected for %s | chunk %s: %d mentions, truncating to %d", item.get("institution"), item.get("chunk_id"), len(mentions), config.MAX_MENTIONS_PER_CHUNK)
        mentions = mentions[: config.MAX_MENTIONS_PER_CHUNK]

    mention_dicts = [m.model_dump() for m in mentions]
    cache.set("institutional_mentions", key, {"mentions": mention_dicts})
    llm_usage.record_call(provider=config.LLM_PROVIDER, model=config.LLM_MODEL, operation=operation, cache_hit=False, cache_key=key, input_tokens=input_tokens, output_tokens=output_tokens)

    return mention_dicts, False


def run_extraction(loaded_reports: list[dict], max_chunks: Optional[int] = None) -> tuple[pd.DataFrame, dict]:
    """Cache-first extraction over a batch of loaded reports. Real LLM calls.

    `loaded_reports` is `documents.load_all_reports()`'s output. Mirrors
    `signal_extraction.run_extraction`'s loop/guards exactly.
    """
    all_items: list[dict] = []
    for report_row in loaded_reports:
        all_items.extend(chunk_report(report_row))
    if max_chunks is not None:
        all_items = all_items[:max_chunks]

    llm = signal_extraction.build_llm()
    rows: list[dict] = []
    chunks_processed = 0
    llm_calls_made = 0
    llm_calls_skipped_over_limit = 0

    for item in all_items:
        chunks_processed += 1
        logger.info("Processing %s | %s | chunk %s", item.get("institution"), item.get("report_title"), item.get("chunk_id"))

        skip_llm_call = llm_calls_made >= config.MAX_LLM_CALLS_PER_RUN
        mentions, was_cache_hit = extract_mentions_for_chunk(item, llm=llm, skip_llm_call=skip_llm_call)

        if skip_llm_call and not was_cache_hit:
            llm_calls_skipped_over_limit += 1
            continue
        if not was_cache_hit:
            llm_calls_made += 1

        if not mentions:
            continue

        for mention in mentions:
            rows.append({**{col: item.get(col) for col in OUTPUT_COLUMNS if col not in mention}, **mention})

    if llm_calls_skipped_over_limit:
        logger.warning("MAX_LLM_CALLS_PER_RUN=%d reached; %d uncached chunk(s) skipped this run", config.MAX_LLM_CALLS_PER_RUN, llm_calls_skipped_over_limit)

    output = pd.DataFrame(rows)
    if not output.empty:
        for col in OUTPUT_COLUMNS:
            if col not in output.columns:
                output[col] = None
        output = output[OUTPUT_COLUMNS]

    summary = {
        "reports": len(loaded_reports),
        "chunks_processed": chunks_processed,
        "llm_calls_made": llm_calls_made,
        "llm_calls_skipped_over_limit": llm_calls_skipped_over_limit,
        "mentions_extracted": len(output),
    }
    logger.info("Chunks processed: %d | LLM calls: %d | Mentions extracted: %d", chunks_processed, llm_calls_made, len(output))
    return output, summary


def save_mentions(new_mentions: pd.DataFrame, output_path=config.INSTITUTIONAL_MENTIONS_PATH) -> pd.DataFrame:
    """Merge new mention rows into whatever's already saved, deduped, and persist."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        combined = pd.concat([existing, new_mentions], ignore_index=True) if not new_mentions.empty else existing
        dedup_columns = [c for c in ["chunk_id", "company_name", "theme", "evidence"] if c in combined.columns]
        if dedup_columns:
            combined = combined.drop_duplicates(subset=dedup_columns, keep="last")
    else:
        combined = new_mentions

    combined.to_parquet(output_path, index=False)
    return combined
