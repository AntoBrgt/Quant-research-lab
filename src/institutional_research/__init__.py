"""Institutional research ingestion -> universe extraction.

    raw report files (local, or fetched from a user-supplied URL list)
        -> documents.py   (load + extract plain text + report-level metadata)
        -> parser.py      (cache-first LLM extraction -> InstitutionalMention rows)
        -> universe.py    (aggregate mentions -> per-company investable universe)

This package answers "why is this company in the universe" (institutional
context) only. It never produces a BUY/SELL/HOLD action and never decides
whether a security is a good investment -- that stays the job of
`research_engine.py` / `horizon.py` / `recommendations.py`, which treat
institutional context as one input among several (fundamentals, technicals,
historical evidence), not the final word. See `universe.py`'s docstring for
the explicit separation between "institutional universe" and "institutional
recommendation".
"""
