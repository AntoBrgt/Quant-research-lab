"""Point-in-time backtest of the ranking (STEP 11).

The question this package answers before any ML model is allowed near the
app: does today's deterministic `rank_score` actually order stocks by their
forward return, out of sample? And does it beat trivially simple references
(12-1 momentum, low volatility)?

    data.py          long price history + raw fundamentals (own caches, never the app's)
    pit_features.py  features exactly as knowable on a past date (as_of + reporting lag)
    labels.py        forward excess return vs SPY, entered the day AFTER the signal
    dataset.py       (date, ticker) panel: features + baseline scores + labels,
                     over a fixed list or the point-in-time 13F universe (STEP 11b)
    evaluate.py      IC / rank IC / quantile spread / label buckets / walk-forward splits

Nothing here writes to the app's price/fundamentals caches or rankings, and
nothing here calls an LLM.
"""
