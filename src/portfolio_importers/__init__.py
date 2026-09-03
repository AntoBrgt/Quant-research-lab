"""Broker-agnostic portfolio input normalization.

    raw broker CSV
        -> detect.detect_format()
        -> a per-broker adapter (e.g. trade_republic.parse())     -- normalization
        -> schema.CanonicalTransaction rows
        -> schema.reconstruct_positions()                          -- reconstruction
        -> schema.CanonicalPosition rows
        -> (separate step, not this package) market enrichment
        -> src/portfolio.py 's existing validate_portfolio/enrich_positions

Normalization and reconstruction never touch live prices or the network --
see schema.py's module docstring. Enrichment is `src/portfolio.py`'s job.
"""
