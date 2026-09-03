"""Cache-first behavior: first run calls the LLM, identical second run doesn't.

No Ollama/network required -- the LLM is a FakeLLM double from conftest.
"""

import config
import signal_extraction
from conftest import FakeLLM


ITEM = {
    "ticker": "AAPL", "filing_type": "10-K", "filing_date": "2026-01-01",
    "section": "Risk Factors", "chunk_id": "AAPL_10K_RISK_FACTORS_000",
    "chunk_index": 0, "text": "Competition in the smartphone market is intense.",
    "source_file": "AAPL.txt",
}

CANNED_SIGNAL = {
    "signal_type": "competition", "direction": "negative", "strength": 0.7,
    "metric_name": None, "metric_value": None, "metric_unit": None,
    "growth_rate": None, "evidence": "Competition in the smartphone market is intense.",
}


def test_first_run_is_cache_miss_and_calls_llm():
    llm = FakeLLM([CANNED_SIGNAL])
    signals, was_hit = signal_extraction.extract_signal_for_item(ITEM, source="sec", llm=llm)

    assert was_hit is False
    assert llm.call_count == 1
    assert len(signals) == 1
    assert signals[0]["signal_type"] == "competition"


def test_second_identical_run_is_cache_hit_and_llm_not_called():
    llm = FakeLLM([CANNED_SIGNAL])

    first_signals, first_hit = signal_extraction.extract_signal_for_item(ITEM, source="sec", llm=llm)
    second_signals, second_hit = signal_extraction.extract_signal_for_item(ITEM, source="sec", llm=llm)

    assert first_hit is False
    assert second_hit is True
    assert llm.call_count == 1  # NOT called a second time
    assert second_signals == first_signals


def test_different_text_is_a_separate_cache_entry():
    llm = FakeLLM([CANNED_SIGNAL])
    other_item = {**ITEM, "text": "Completely different content about margins."}

    signal_extraction.extract_signal_for_item(ITEM, source="sec", llm=llm)
    signal_extraction.extract_signal_for_item(other_item, source="sec", llm=llm)

    assert llm.call_count == 2


def test_runaway_output_is_capped(monkeypatch):
    monkeypatch.setattr(config, "MAX_SIGNALS_PER_CHUNK", 3)
    many_signals = [CANNED_SIGNAL for _ in range(50)]
    llm = FakeLLM(many_signals)

    signals, _ = signal_extraction.extract_signal_for_item(ITEM, source="sec", llm=llm)

    assert len(signals) == 3


def test_usage_is_logged_for_both_hit_and_miss():
    import llm_usage

    llm = FakeLLM([CANNED_SIGNAL])
    signal_extraction.extract_signal_for_item(ITEM, source="sec", llm=llm)
    signal_extraction.extract_signal_for_item(ITEM, source="sec", llm=llm)

    usage = llm_usage.load_usage()
    assert len(usage) == 2
    assert usage["cache_hit"].tolist() == [False, True]


def test_run_extraction_reuses_cache_across_a_full_batch(synthetic_documents):
    llm = FakeLLM([CANNED_SIGNAL])
    original_build_llm = signal_extraction.build_llm
    signal_extraction.build_llm = lambda: llm
    try:
        signals_1, summary_1 = signal_extraction.run_extraction(synthetic_documents, source="sec")
        signals_2, summary_2 = signal_extraction.run_extraction(synthetic_documents, source="sec")
    finally:
        signal_extraction.build_llm = original_build_llm

    assert summary_1["llm_calls_made"] == 2  # one per distinct chunk (AAPL + MSFT)
    assert summary_2["llm_calls_made"] == 0  # second run: everything cached
    assert len(signals_1) == len(signals_2) == 2
