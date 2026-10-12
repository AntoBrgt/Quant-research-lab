# Quant-research-lab

**Question.** Can machine learning, or any systematic rule built on public data, pick US stocks that beat simply holding the index, once the test is honest?

**Answer: not so far.** Over three research phases (2013-2026 data), I tested:
- ranking models (hand-weighted scores, LightGBM, ridge, lambdarank, ensembles);
- short-term trading loops;
- value + survival models;
- a small-cap model;
- filing-text features ("Lazy Prices");
- three event rules (insider buying, earnings surprise, cheap quality + trend).

**None beat SPY or IWM after costs.** Several beat most *random* stock pickers, and a few had good stretches, but none cleared a pre-registered bar on data it had never seen. The repository is built to make that answer trustworthy: every rule was fixed in a commit before its test ran, and every holdout was opened once.

Live site: *link added once deployed* ([how it's deployed](docs/DEPLOY.md)) · Full step-by-step log: [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) · Pre-registrations: [Phase 2](PREREGISTRATION_PHASE2.md)

## Verdicts

| Phase | Test | Result | Annualized after costs | Benchmark, same period | Verdict |
|---|---|---|---|---|---|
| 1 (11b, 12, 13) | Ranking models: app score, LightGBM, + insider/13F features | best OOS rank IC 0.020, t 0.69 | -- (ranking test) | -- | No edge / not promoted |
| 1 (13b) | Short-term loop, app picks, hold 10 days | expectancy +0.005R, CI [-0.055, +0.068] | +1.9% (2019-26) | SPY +16.1% | NOT PROVEN |
| 1 (13c) | Same loop, best of 3 pre-registered exits (trailing 3 ATR) | expectancy +0.053R, CI [-0.012, +0.120] | +6.7% | SPY at same exposure +11.8% | NOT PROVEN |
| 1 (15) | Cheap quality + XGBoost survival model, +30% target | concordance 0.54 | +8.1% (2017-26) | SPY +15.3% | DOES NOT BEAT MARKET |
| 1 (14a) | Small-cap LightGBM, top decile | OOS rank IC -0.001, t -0.05 | +6.4% as priced | IWM +9.5%, SPY +15.2% | DOES NOT BEAT MARKET |
| 1b | 11 registered ML variants, top 500, one holdout look (2024-01 -> 2026-07) | best holdout rank IC t 0.93 (Holm over 11) | top decile +4.7% to +37.2% | SPY +21.0% | NO EDGE (all 11) |
| 1b | Lazy Prices text features | no text variant passes | -- | -- | FAILED |
| 2 | H1 insider cluster buying (small caps), holdout | t 1.25, random pct 78 | +11.0% (2024-09 -> 2026-09) | IWM +12.9% | NOT PROVEN |
| 2 | H2 top-decile earnings surprise (small caps), holdout | t -2.22, random pct 8 | +3.7% | IWM +14.7% | NOT PROVEN |
| 2 | H3 cheap quality + 200-day trend (top 500), holdout | t 0.32 on 44 trades; failed all checks in development | +17.3% | SPY +6.9% at 38% exposure | NOT PROVEN |
| 3 | Own text models + LLM agent + forward-tested pick ledger | pre-registration drafted; forward test not started | -- | -- | -- |

Details and every intermediate result: [PHASE 1 VERDICT](docs/RESEARCH_LOG.md#phase-1-verdict) · [PHASE 1b addendum result](docs/RESEARCH_LOG.md#phase-1b-addendum----result-holdout-opened-once-2026-10-10-1851-utc-commit-cb5e8eb) · [PHASE 2](docs/RESEARCH_LOG.md#phase-2----three-pre-registered-event-rules-python-srcrun_phase2py).

## Methodology

- **Point-in-time universes.** Each month uses only stocks that were knowable then, reconstructed from SEC 13F filings by filing date. Today's index members would bake in survivorship bias.
- **Point-in-time features.** Fundamentals come from SEC XBRL, dated by filing date with the earliest-filed value. Insider data comes from Form 4 filing dates. Nothing is dated by the period it describes.
- **Execution.** Entry at the next trading day's open after the signal; 0.15%/side costs for large caps, 0.30%/side for small caps.
- **Out of sample only.** Walk-forward training with an embargo at least as long as the label, so no training label overlaps a test date.
- **Baselines.** Every result is compared with hundreds of random pickers on the same dates and exits, with momentum, and with the benchmark at the same average exposure.
- **Statistics.** Newey-West t-stats on per-date ICs and on monthly excess returns, and bootstrap confidence intervals resampled by month. Holm correction when several variants are judged together.
- **Pre-registration and one-time holdouts.** The verdict rules are committed before the test runs. Each holdout is opened once, enforced in code by a lock file or logged gate, and any bug fix found before the holdout is documented with its effect.

## How to reproduce

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows; source .venv/bin/activate elsewhere
pip install -r requirements-lock.txt
cp .env.example .env                                      # optional: OPENFIGI_API_KEY speeds up CUSIP lookups
pytest                                                    # tests marked needs_data are skipped without local caches

python src/prepare_phase2_data.py --build                 # rebuilds the PIT data from SEC / OpenFIGI / Yahoo (hours on a first run)
python src/run_value_survival.py                          # STEP 15
python src/run_smallcap.py                                # STEP 14a
python src/run_strategy_backtest.py --step13c             # STEP 13b / 13c
python src/run_phase2.py                                  # PHASE 2, development period (the holdout is locked: --holdout refuses a re-run)
```

`data/` is never committed: everything is rebuilt from public sources and cached locally. The one-shot holdout runners refuse a second run (`run_phase2.py --holdout`, `run_phase1b_addendum.py`), so a re-run on another machine is flagged as a reproduction.

The original Streamlit research app is kept as a local tool (`streamlit run app.py`). Its Quick Picks page is a watchlist, not trade instructions.

## Repository layout

| Path | What |
|---|---|
| `src/backtest/` | PIT universes, features, walk-forward, labels, barriers, event engine, protocol (registry, DSR, PBO, holdout gates) |
| `src/run_*.py` | one runner per research step |
| `src/sec_xbrl.py`, `src/insider_form4.py`, `src/edgar_sections.py`, `src/institutional_research/` | SEC data: XBRL fundamentals, Form 4, 10-K/10-Q sections, 13F holdings |
| `app.py`, `pages/` | Streamlit research app (local) |
| `tests/` | 370+ tests: look-ahead, embargo, costs, locks, verdict rules |
| `docs/RESEARCH_LOG.md` | the full research log, STEP by STEP |
| `src/export_site_data.py`, `site/` | read-only export of the results to JSON, and the static site built from it ([docs/DEPLOY.md](docs/DEPLOY.md)) |

## What I learned

*(draft from the research log -- to be edited)*
- Most of the work in a backtest is making it honest. Point-in-time universes, filing-date features and next-open entries removed edges that looked real at first.
- A good-looking AUC can be mechanical. The 0.88 meta-label AUC came from target distance alone, and predicting "hit" was not predicting profit.
- Beating random pickers is not beating the market. Several strategies beat 74-98% of random pickers and still trailed SPY: the universe, not the selection, drove their returns.
- Development results reverse. H2's t of 2.5 in development became -2.2 in the holdout.
- Pre-registration changes behaviour. With the rule fixed in a commit, a failed test ends the idea instead of starting a rescue variant.
- Count your trials. The Phase 1b registry logs every variant run, so selection bias can be measured (deflated Sharpe, PBO) instead of ignored.
- A bug found in development must be fixed *and* reported with its effect. The H2 fix moved its t across the bar, which is exactly why it had to be documented before the holdout.
- Data coverage is a result too. In the small-cap band, many old CUSIPs can't be mapped to a priced ticker (69% of the band in 2013). The survivorship bounds then become so wide (-66%/yr to -100%/yr) that only a strategy already beating the benchmark as priced could be judged.
- A short holdout is weak evidence either way. 44 trades and 31 months can't confirm or rule out a small edge, so the bar is the forward test.

## License

MIT, see [LICENSE](LICENSE). Research project. Not investment advice. No strategy shown here has beaten the market.
