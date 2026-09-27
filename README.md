# Quant-research-lab
Can machine learning produce robust, out-of-sample investment signals?

## STEP 2 — Document processing

This step converts raw SEC filing text into a reproducible, chunked dataset suitable for later LLM-based signal extraction.

### Input
The processor reads raw filing text files from:

`data/raw/edgar/*.txt`

Each file is expected to correspond to a ticker, such as `AAPL.txt` or `MSFT.txt`.

### Processing
The pipeline performs the following:

1. Reads each filing without modifying the original raw source file.
2. Extracts available metadata such as ticker, filing type, filing date, and accession number when present in the text.
3. Detects common SEC filing sections using robust heading heuristics.
4. Splits large sections into smaller chunks of roughly 3,000–6,000 characters with a small overlap.
5. Stores the output as a Parquet dataset.

### Output
The resulting dataset is saved to:

`data/processed/documents.parquet`

### Output schema
Each row represents one chunk from one section of one filing.

Columns:

- `ticker`
- `filing_type`
- `filing_date`
- `section`
- `chunk_id`
- `chunk_index`
- `text`
- `source_file`
- `accession_number` (when available)
- `document_length`

### Example command

```bash
python src/process_documents.py
```

Optional ticker filtering is also supported:

```bash
python src/process_documents.py --ticker AAPL MSFT JPM
```

## STEP 3 — Signal extraction

This step reads processed SEC filing chunks and extracts structured financial signals for later quantitative research.

### Input
The processor reads the chunked dataset from:

`data/processed/documents.parquet`

### Output
The resulting signal dataset is saved to:

`data/processed/signals.parquet`

### Signal taxonomy
The extraction model is restricted to the following signal types:

- `revenue_growth`
- `earnings`
- `margins`
- `guidance`
- `demand`
- `pricing`
- `costs`
- `capital_expenditure`
- `cash_flow`
- `debt`
- `liquidity`
- `competition`
- `regulation`
- `management_confidence`
- `risk`

Each signal has one direction:

- `positive`
- `negative`
- `neutral`

Each signal also includes a strength value from `0.0` to `1.0` and evidence copied or lightly normalized from the filing text.

### Structured schema
The resulting DataFrame contains rows with:

- `ticker`
- `filing_type`
- `filing_date`
- `section`
- `chunk_id`
- `chunk_index`
- `signal_type`
- `direction`
- `strength`
- `metric_name`
- `metric_value`
- `metric_unit`
- `growth_rate`
- `evidence`
- `source_file`

### LLM configuration
The extraction layer uses LangChain and supports three interchangeable providers, selected by `LLM_PROVIDER` in `.env`:

| Provider | `LLM_PROVIDER` | Model env var | Notes |
|---|---|---|---|
| Ollama (local, free) | `ollama` | `OLLAMA_MODEL` (default `llama3.2`) | No token cost, but quality/reliability varies with the local model. |
| OpenAI | `openai` | `OPENAI_MODEL` (default `gpt-4o-mini`) | Requires `OPENAI_API_KEY`. |
| Anthropic | `anthropic` | `ANTHROPIC_MODEL` (default `claude-haiku-4-5`) | Requires `ANTHROPIC_API_KEY`. |

Other relevant `.env` variables are documented in `src/config.py` (paths, cache toggle, prompt/schema versions, cost/safety guards).

### Example commands

```bash
python src/extract_signals.py --dry-run          # cost/cache estimate, zero LLM calls
python src/extract_signals.py --ticker AAPL --max-chunks 10
python src/extract_signals.py --ticker AAPL MSFT JPM
python src/extract_signals.py --source news --ticker AAPL   # yfinance headlines instead of SEC chunks
```

## STEP 4 — Cache-first, multi-portfolio architecture

The goal beyond STEP 3 is: **run this on a real portfolio, then on several people's portfolios, without multiplying LLM cost per user.** The system is split into a shared, cached research layer and a per-user portfolio layer:

```text
SHARED (company-level, cached, computed once)          USER-SPECIFIC (per portfolio, pure Python)
─────────────────────────────────────────────          ──────────────────────────────────────────
SEC chunks ──┐                                          portfolio.py
News items ──┼─ signal_extraction.py (cache-first) ──┐     - load/validate CSV
Prices ──────┼─ market_features.py (pure Python) ────┤     - price/weight/P&L/concentration
             │                                        ▼
             └──────────────► research_engine.py (aggregate, no LLM)
                                       │
                                       ▼
                              strategy.py (score per horizon)
                                       │
                                       ▼
                          recommendations.py (asset/strategy/portfolio/risk, kept separate)
                                       │
                                       ▼
                                    app.py (Streamlit, orchestration only)
```

`research_engine.py` only reads what's already cached/saved -- it never triggers an LLM call itself. Every LLM call happens exclusively inside `signal_extraction.py`, gated by the cache. That is what guarantees 100 portfolios holding AAPL costs the same as 1 portfolio holding AAPL.

### Cache

`src/cache.py` is a deterministic, content-addressed filesystem cache under `data/cache/<namespace>/<key>.json`. The key is:

```text
SHA256(operation + "|" + SHA256(normalize(input_text)) + "|" + model + "|" + prompt_version + "|" + schema_version)
```

A ticker alone is never a valid key input -- the key is derived from the actual chunk/headline text plus everything that could change the output (model, `PROMPT_VERSION`, `SCHEMA_VERSION` in `src/config.py`). Bumping `PROMPT_VERSION` invalidates exactly the affected cached results, without deleting the cache.

### Cost/reliability guards

`src/config.py` also defines: `MAX_LLM_CALLS_PER_RUN` (caps real calls per run; cached items still resolve for free once the cap is hit), `MAX_SIGNALS_PER_CHUNK` (caps output size -- a well-formed 4500-character chunk should produce a handful of signals, not hundreds), and `LLM_REQUEST_TIMEOUT_SECONDS` (a stuck LLM call fails fast instead of hanging the whole run). The latter two exist because an earlier unscoped local-model run hung for hours and returned 900+ degenerate "signals" from single chunks.

### Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pick a holding period on the front page; the ranked opportunities appear automatically (see STEP 8).

## STEP 5 — Portfolio input (removed)

The portfolio upload flow (broker CSV importers, position reconstruction, per-holding recommendations) was removed in STEP 9 -- the app is now universe-first. It remains in git history before that commit.

## STEP 6 -- Institutional universe + horizon-aware research engine

A second, independent product surface alongside the portfolio flow above (portfolio upload is untouched and out of scope for this step). The direction:

```text
Institutional research reports -> institutional universe -> Streamlit
    -> user picks a horizon (1 day .. 20 years)
    -> fundamental + technical + institutional + historical evidence
    -> risk/reward framework -> research view
```

Institutional reports are treated as **one input for discovering themes, sectors, and companies** -- never as a source of BUY/SELL instructions, and never blindly copied. The project's own fundamental/technical/historical evidence remains the independent evaluation layer.

### Institutional research ingestion (`src/institutional_research/`)

```text
data/raw/institutional/<institution>/<report>.{txt,md,pdf,html}   (local files, or fetched via providers.py)
    -> documents.py   load + extract plain text + report metadata (with an optional <file>.meta.json sidecar override)
    -> parser.py      cache-first LLM extraction -> InstitutionalMention rows (institution, theme, region,
                       sector, company/ticker, institutional_view, view_direction, confidence, evidence)
    -> universe.py    aggregate mentions -> per-ticker universe (institution_count, themes, institutional_direction, ...)
```

Two ways to get reports in, both explicit and opt-in:
1. **Local files** (recommended default) -- drop files into `data/raw/institutional/<institution>/`, same layout as `data/raw/edgar/`.
2. **A user-supplied URL list** (`data/raw/institutional_urls.csv`, columns `institution, report_title, url[, report_type, publication_date]`) -- fetched only via an explicit action (`--fetch-urls` on the CLI, or the checkbox on the Universe page), checking `robots.txt` and rate-limiting per host. This is **not** a crawler -- it only ever fetches exact URLs a human listed.

`parser.py` reuses `signal_extraction.build_llm()`, `cache.py`, `llm_usage.py`, and `process_documents.split_into_chunks()` directly rather than reimplementing the cache-first/hang-guard/runaway-cap machinery a second time -- same cache keying, same `LLM_REQUEST_TIMEOUT_SECONDS`, same idea as `MAX_SIGNALS_PER_CHUNK` (here `MAX_MENTIONS_PER_CHUNK`), separate cache namespace (`institutional_mentions`) and schema.

**A mention is never a recommendation.** "We favor European financials" becomes `region=Europe, sector=Financials, view_direction=POSITIVE` -- not "BUY BNP" -- enforced both in the extraction prompt and structurally (`InstitutionalMention` has no action/rating field at all). `view_direction` is `POSITIVE | NEUTRAL | NEGATIVE | MENTIONED`; `confidence` is categorical (`High/Medium/Low`), not a float, for the same reason `Recommendation.confidence` elsewhere in this project is never presented as a probability.

**The universe is built only from ingested institutional reports** -- it does not seed itself from any other ticker already known to the project (e.g. the 3 tickers used to test the SEC pipeline). No reports ingested yet means an empty universe, by design, not a bug.

Run it:
```bash
python src/ingest_institutional_research.py --dry-run              # what would be processed, no LLM calls
python src/ingest_institutional_research.py --max-chunks 20        # cost-controlled test run
python src/ingest_institutional_research.py --fetch-urls           # opt-in: also download the URL list first
```

`universe.build_universe()` only includes rows that resolved to a `ticker` -- a theme/sector-only mention (no identifiable company) is real context, surfaced separately via `universe.theme_summary()`, not a security this project can price or run fundamentals on. **Separation is structural, not just documentation**: `institutional_research/` only ever answers "why is this company in the universe"; it has no code path that produces a research verdict.

### Fundamentals (`src/fundamentals.py`)

No LLM anywhere in this module; deterministic Python throughout. Same provider shape as `price_provider.py`/`news_provider.py` (a `Protocol`, a free `yfinance`-backed implementation, a disk cache at `data/raw/fundamentals/<ticker>.json`, 20h TTL), with three layers on top:

1. **`MetricObservation`** -- every metric (revenue, margins, EPS, FCF, ROE, ROIC, leverage, ...) is a chronological list of period-by-period observations, each carrying `period` (e.g. `"FY2025"`), `period_type` (`annual`/`ttm`/`forward`), `is_estimate`, `source`, and `retrieved_at` -- not a single latest scalar, and never mixing a TTM figure into an annual series. `total_debt`/`net_debt`/`leverage` are explicitly current-snapshot-only (yfinance has no reliable historical `total_debt` line item) rather than fabricating a fake historical series by pairing today's debt with past years' cash/equity.
2. **Data-quality validation** (`validate_observations`) -- flags (never deletes) impossible margins, implausible ratios, implausible coverage, extreme growth, invalid/future dates, duplicate periods, and statistical outliers, each with its own calibrated bound (a >150% ROE from a buyback-heavy balance sheet, or 30x interest coverage from a well-capitalized large-cap, are real and must not be flagged the same way a >150% *margin* would be -- validated directly against 8 real companies across 7 sectors, see below).
3. **`classify_trend()`** -- a deterministic `IMPROVING`/`STABLE`/`DETERIORATING`/`INSUFFICIENT_DATA` state per factor, from the real multi-period series (not a single-point comparison). Descriptive only -- there is no fundamental score and no BUY/SELL output anywhere in this module.

`compute_fundamentals(ticker)` returns the flat output schema (`CompanyFundamentals.model_dump()`): growth, profitability, balance sheet, cash generation, and valuation fields plus their trend states, `observation_date` (the underlying reporting period) separate from `data_freshness` (cache retrieval time), `data_quality` (aggregated flags), `sources`, and the full `history` every derived field is read off of. Different sectors don't get every field forced onto them -- a bank correctly has no `gross_margin`/`operating_margin`/`ev_ebitda` (no comparable statement line items via this data source), rather than a guessed value.

### Technical analysis (`src/market_features.py` + `src/technicals.py`)

`market_features.py` stays the low-level, `as_of`-gated, no-look-ahead-tested layer (returns, volatility, RSI, moving averages, ATR, volume ratio, swing support/resistance) -- unchanged in this pass, and still what `strategy.py`'s three horizon buckets read directly. `technicals.py` is a new layer on top, built to the same standard as `fundamentals.py`, reusing `market_features.py`'s functions rather than reimplementing them:

- **Historical series with provenance**: `TechnicalObservation` -- close, SMA20/50, RSI14, ATR14, and 20-day volatility as chronological, dated series (~3 months of trading days), not just the latest value, each carrying `source`/`retrieved_at`.
- **Returns**: 1D/5D/20D/60D/120D/252D, each backed by a `ReturnObservation` (start date, end date, value, and whether the horizon actually had enough history -- an incomplete horizon is `None`, never silently substituted with another window).
- **Moving averages/trend**: SMA20/50/100/200, `price_vs_ma*`, `ma20_vs_ma50`, `ma50_vs_ma200`, and a `trend` state (`BULLISH`/`BEARISH`/`MIXED`/`INSUFFICIENT_DATA`) from price-vs-SMA50-vs-SMA200 relationships -- descriptive only, never a recommendation (a `BULLISH` trend is not a BUY signal).
- **Momentum**: RSI14 plus `rsi_state` (`OVERBOUGHT`/`NEUTRAL`/`OVERSOLD`), and a separate `momentum_state` from the direction of the 20D/60D/252D returns.
- **Volatility**: 20-day and 60-day annualized volatility, ATR14 and ATR14-as-%-of-price, and a `volatility_regime` (`LOW`/`NORMAL`/`HIGH`) computed relative to *that security's own* trailing volatility distribution (tercile split) rather than one fixed universal threshold -- "high volatility" means something different for a utility than a semiconductor, and this project has no cross-sectional universe yet to derive a shared threshold from.
- **Volume**: 20-day average volume, volume-vs-average, and a `volume_trend` (`INCREASING`/`STABLE`/`DECREASING`) that directly reuses `fundamentals.classify_trend()`'s deterministic delta logic (relabeled, not reimplemented).
- **Relative strength**: 20D/60D/252D stock return minus benchmark return, both kept individually visible. **Benchmark: SPY** (SPDR S&P 500 ETF) -- no benchmark convention existed anywhere in this project before this; SPY was chosen as a broad, liquid, freely-available-via-`yfinance` US-equity proxy appropriate for this project's largely US-large-cap universe today, fetched through the same cached `price_provider`. A failed benchmark fetch means missing relative-strength fields, never a silently skipped calculation.
- **Support/resistance**: the same 60-day swing-high/low methodology as `market_features.compute_swing_levels` (kept under both names -- `support_60d`/`resistance_60d` for `risk_reward.py`'s existing usage, `nearest_support`/`nearest_resistance`/`distance_to_*_pct` as this layer's own names for the identical computation).
- **Data quality**: OHLCV validation flags structural impossibilities (`high < low`, close outside the high/low range, negative price/volume, duplicate/unsorted dates, insufficient history) as `INVALID_DATA`, and distinguishes a large single-day move that's corroborated by a volume spike (`VALID_EXTREME_MARKET_MOVE`) from one that isn't (`SUSPICIOUS_DATA`) -- a real crash or rally is not treated as a data error. Nothing is ever silently dropped.
- **Freshness/provenance**: `data_source`, `retrieved_at`, `history_start`/`history_end`, `as_of`.

**Price convention**: every indicator uses `adj_close`. `price_provider.py` now downloads with `auto_adjust=True` so `open`/`high`/`low`/`adj_close` are all the *same* fully split/dividend-adjusted series -- this replaced an earlier `auto_adjust=False` approach that paired an adjusted Close with raw High/Low, which real-data validation on JPM caught silently producing `close_below_low` on every row following a dividend adjustment (adjusted Close drifting below the still-raw Low). There is no separate raw/unadjusted series kept anywhere in this project now, so there is nothing left to mix.

`research_engine.load_extended_research()` merges `technicals.compute_technicals()`'s flat output into `research["market_features"]` (an additive superset -- every key `horizon.py`/`risk_reward.py`/`strategy.py`/`component_views` already read keeps its exact name and value), so the new fields are available to the horizon-aware research page with zero changes required to those modules.

### Numeric horizon + horizon-aware scoring (`src/horizon.py`)

`strategy.py`'s three buckets (`short_term`/`medium_term`/`long_term`) are untouched and still serve the portfolio flow. `horizon.py` has two complementary layers, both kept:

- **`score_horizon_fit()`** (unchanged) -- the continuous generalization: `horizon_days` blends ~15 individually-weighted raw signal/technical/fundamental fields from 4 log-interpolated anchor profiles (1 day / 30 days / 1 year / 10 years).
- **`compute_horizon_weighted_view()`** (new) -- a coarser, more directly explainable layer for "why does this horizon differ from that one": **5 transparent profiles** (`VERY_SHORT`/`SHORT`/`MEDIUM`/`LONG`/`VERY_LONG`, covering every named preset from "1 day" to "20 years") each weighting **10 named component groups** (5 technical: trend, momentum, volatility, volume, relative strength; 5 fundamental: growth, profitability, cash flow, balance sheet, valuation). Each profile's weights sum to 1.0 (`test_horizon.py`); technical weight runs 0.85 → 0.70 → 0.50 → 0.25 → 0.10 from `VERY_SHORT` to `VERY_LONG` (fundamental weight is the mirror image) -- validated directly on AAPL/JPM/NEE: technical factors (momentum, trend, volume) dominate every ticker's `dominant_factors` at a 1-day horizon, fundamental factors (profitability, growth, cash flow) dominate at 10 years, for all three regardless of company. A missing group (e.g. a company with no resolvable technicals) is **excluded and the remaining weights renormalize** -- it is never treated as a negative signal or padded with a zero (`test_missing_group_is_excluded_not_treated_as_negative`). The output stays a set of named, inspectable components (`components`, `technical_contribution`, `fundamental_contribution`, `dominant_factors`) -- never one opaque score -- and, like every other layer in this project, it is descriptive research weighting, not a BUY/SELL recommendation or a return/probability prediction. `research_engine.load_extended_research()` exposes it as `horizon_weighted_view`, alongside the untouched `horizon_fit`.

Either way, the same company can score positively at one horizon and negatively at another if the underlying evidence actually points that way -- verified directly in `test_horizon.py`. Every weight is a plain, readable module-level dict; there is no single hardcoded scoring formula.

### Risk/reward framework (`src/risk_reward.py`)

Horizon decides the *methodology*, not just the numbers:
- **Short/medium horizons**: an ATR-based (or recent-swing-low) technical stop, a swing-high/risk-multiple technical target, a numeric `risk_reward_ratio`.
- **Long horizons (>~2 years)**: no tight technical stop -- instead a **thesis invalidation** description (referencing the company's actual current revenue growth/margin, not a template) and an explicit **"No fixed take-profit -- thesis/valuation based exit"**, consistent with this project's existing rule that it never fabricates a price target.

`stop_loss`/`take_profit` always carry a `type` (`technical` vs `thesis_invalidation`/`none`) so the two are never confused with each other.

### Data freshness (`research_engine.assess_data_freshness`)

Per-source staleness (price/fundamentals/institutional) against **horizon-dependent** thresholds -- a 5-day-old price is disqualifying for a 1-day horizon and irrelevant for a 10-year one; a 300-day-old institutional report is the reverse. One overall `OK`/`STALE` flag per horizon, not one global threshold.

### Company research composition (`research_engine.load_extended_research` / `component_views` / `research_confidence`)

`load_extended_research(ticker, horizon_days)` composes (never recomputes) `load_company_research`, `fundamentals.py`, `institutional_research.universe.why_in_universe`, `horizon.py`, and `risk_reward.py` into one object for the company-research page -- read-only and cache-first throughout, same "no new LLM extraction just from viewing a company" contract as the rest of the app. `component_views()` derives independent `POSITIVE/NEUTRAL/NEGATIVE/INSUFFICIENT_EVIDENCE` labels for fundamental, technical, and institutional evidence -- **kept separate on purpose** (section 21's "no opaque AI score"); they are allowed to disagree, and the UI shows that disagreement rather than averaging it away. `research_confidence()` is evidence quality/agreement (`High/Medium/Low/Insufficient evidence`), explicitly **not** a probability of return.

### Streamlit pages

`app.py` is the front page, **Opportunities** (see STEP 8). `pages/1_Company_Research.py` (evidence verdict, live price chart, institutional context, fundamentals, technicals, historical evidence, risk/reward, research conclusion) follows in the sidebar.

### Is ML justified yet?

**No.** The universe today has as many companies as there are institutional mentions ingested (potentially zero until reports are supplied), there is no historical label set free of look-ahead bias, and no backtesting harness yet exists to check a model against a baseline out-of-sample. The deterministic baseline above (fundamentals + technicals + institutional context + historical signal evidence, horizon-weighted) is the whole system for now, by design -- see brief section 10's explicit conditions, none of which are close to satisfied yet.

### Known gaps in this pass

- No live webpage rendering/scraping for institutional reports -- PDF/HTML/text files only (a user saves a webpage first, same as `data/raw/edgar/` is pre-fetched rather than scraped live).
- Region is derived from `yfinance`'s `country` field via a small hand-written lookup table, not an authoritative geography source.
- `roic` and `interest_coverage` are always `None` -- no reliable free-data field/derivation found yet.
- No backtesting/evaluation harness yet (brief section 20) -- the risk/reward and horizon-fit methodology is not yet validated against historical forward returns.
- Support/resistance is a simple 60-day swing high/low, not a statistically fitted level.


## STEP 7 -- Automatic institutional input: SEC 13F holdings

Research reports have to be found, saved, and LLM-parsed by hand, and they mostly name themes rather than companies. `src/institutional_research/holdings_13f.py` adds a fully automatic second source: the quarterly **Form 13F-HR** every US manager with >$100M in US equities files with the SEC -- actual positions, structured XML, free, no LLM.

```text
SEC submissions JSON (per filer CIK) -> latest two 13F-HR filings (current + prior quarter)
    -> information table XML (cached at data/raw/13f/<cik>/<accession>.parquet -- a filed 13F never changes)
    -> one row per CUSIP (options and bond/PRN rows dropped, sub-manager rows summed)
    -> quarter-over-quarter change, flow-adjusted + split-adjusted
    -> top 30 holdings by value + 10 biggest adds/new + 10 biggest cuts/exits per filer
    -> CUSIP -> ticker via security_master.resolve_cusips (OpenFIGI, cached; ETFs/funds dropped)
    -> InstitutionalMention rows (report_type="13F") merged into institutional_mentions.parquet
```

**Default filers** (the five largest 13F filers): BlackRock (CIK 2012383), Vanguard (102909), State Street (93751), Fidelity/FMR (315066), JPMorgan Chase (19617). Override with `data/raw/13f_filers.csv` (`institution,cik`). Index giants hold essentially the whole market, so their *top holdings* are just the biggest companies -- the informative part is the adds/cuts. Adding a concentrated active manager (e.g. Berkshire Hathaway, CIK 1067983) makes the universe more opinionated.

**Direction mapping** -- what the filer *did*, never "BUY": `NEW` position -> `POSITIVE`/High, flow-adjusted add >10% -> `POSITIVE`/Medium, cut >10% -> `NEGATIVE`/Medium, full exit -> `NEGATIVE`/High, otherwise `MENTIONED`/Low.

- **Flow adjustment**: a position's share change is divided by the filer's *median* share change. An index manager whose every position grew +5% from inflows shows 0% on each, not +5% "buying".
- **Split adjustment**: a 4:1 split shows as shares x4 with the filing-implied price (value/shares) /4 in the same quarter; that's detected and divided out instead of reported as a +300% add.
- Refreshing replaces the previous 13F rows for those filers (latest quarter only); LLM report mentions are left untouched. One failing filer never stops the others.

Run it:
```bash
python src/ingest_institutional_research.py --13f-only   # 13F only, rebuild universe
python src/ingest_institutional_research.py --13f        # 13F + parse any local reports
```
or click **Refresh 13F holdings from SEC** on the front page. New 13Fs appear quarterly (mid-Feb/May/Aug/Nov), so a refresh a few times per quarter is plenty.

**Limits**: filed up to 45 days after quarter end; long US equity positions only (no shorts, non-US holdings, or intra-quarter trades); amendments (13F-HR/A) ignored; a CUSIP change from a merger/reorganization looks like an exit + new position; split detection is a heuristic on filing-implied prices.

## STEP 8 -- Automatic horizon ranking + live charts

`streamlit run app.py`, pick a holding period, and the list is there -- no manual step:

1. **13F auto-refresh**: if holdings were never fetched or are older than 7 days (`holdings_13f.needs_refresh`), the front page refreshes them and rebuilds the universe on load (once per session, so an SEC outage can't loop).
2. **Ranking** (`src/screener.py`): every universe ticker goes through the *same* `research_engine.load_extended_research` the Company Research page uses, and is sorted by
   `rank_score = horizon_weighted_view.score + 0.10 × institutional_direction_score`
   -- no new scoring model, just the existing horizon-weighted fundamentals/technicals with a small, visible institutional tilt. Labels: `Strong` (≥0.30), `Favorable` (≥0.10), `Neutral`, `Unfavorable` (≤−0.10). **Shortlist** = Strong/Favorable + fresh data + not "Insufficient evidence". Rankings are cached per horizon per day in `data/processed/rankings/` (prices/fundamentals already cache ~20h), so switching horizons back and forth is instant after the first pass. Prices are pre-fetched sequentially (yfinance bulk download isn't thread-safe), the rest runs on 6 threads; a ticker that fails is ranked last with its error, never dropped silently.
3. **Click a row** -> Company Research opens on that ticker with the horizon carried over.

**Live chart** (`src/live_chart.py`): candles + SMA20/50/200, volume, RSI(14) (same definition as `market_features.compute_rsi`), with stop-loss/take-profit/support/resistance lines from `risk_reward.py`. The window follows the horizon -- ≤3 days: 5-minute bars over 5 days, auto-refreshing every 60s; ≤2 weeks: 30-minute bars; ≤3 months: 6 months daily; ≤2 years: 2 years daily; longer: 10 years weekly -- and can be switched manually. Bars come straight from Yahoo (typically ~15 min delayed intraday), are never written to the research price cache, and never feed the scoring.

**Still true**: the ranking is a sort of the current evidence, not a return forecast. Nothing here has been validated against forward returns yet -- that backtest is the next step before trusting the labels with real size.

## STEP 9 -- Cleanup after the first real run

- **Portfolio upload removed**: `pages/2_Portfolio_Upload.py`, `portfolio.py`, `portfolio_importers/`, `recommendations.py`, `strategy.py` and their tests.
- **CUSIP -> ticker fixes** (seen on real 13F data): OpenFIGI writes share classes as `BRK/B`, Yahoo needs `BRK-B`; and a CUSIP lookup could land on a foreign line (e.g. a London GBP listing) that Yahoo can't price. CUSIP jobs now ask OpenFIGI for the US composite (`exchCode=US`) only and return nothing rather than a foreign guess. Cached CUSIP resolutions from the first version are ignored (new cache prefix), and `holdings_13f.REFRESH_VERSION` forces one automatic 13F/universe rebuild on next app start.
- yfinance's per-ticker ERROR tracebacks are silenced in the Streamlit pages (failures are still listed in the UI).

## STEP 10 -- Quick Picks (the "lazy" page)

`pages/0_Quick_Picks.py`: pick a horizon, a budget, the max loss per trade, and how many picks -> an order list.

- **BUY**: top of the shortlist (Strong/Favorable, fresh data, enough evidence) with risk/reward ≥ 1 (a strong name whose target is closer than its stop is a bad entry today). Each line: limit price (current), stop, target, % to each, **shares and amount**.
- **Sizing** (`screener.quick_picks`): lose at most *max loss %* of the budget if the stop is hit (`shares = budget × risk% / (price − stop)`), never more than 20% of the budget in one line; long horizons with no price stop fall back to an equal split.
- **SELL / avoid**: the most Unfavorable names -- sell if held (or keep the stop shown), don't buy.
- Same cached ranking as the Opportunities page (`screener.get_ranking`), so it costs nothing extra once a horizon has been scored today. Click any row for the full Company Research view.

## STEP 11 -- Backtest the ranking before any ML model

"Is ML justified yet?" (STEP 6) listed the missing preconditions: a look-ahead-free label set and a harness that checks a signal against a baseline out of sample. `src/backtest/` builds both, and scores the **current** ranking with them first -- an ML model only earns a place in `screener.py` if it beats this baseline out of sample.

```bash
python src/run_backtest.py                              # current institutional universe, 10y of prices
python src/run_backtest.py --tickers AAPL MSFT JPM ...  # any list
python src/run_backtest.py --no-fundamentals            # technicals only, no yfinance fundamentals calls
python src/run_backtest.py --reuse-panel                # re-evaluate the saved panel, no downloads
```

Outputs in `data/processed/backtest/`: `panel.parquet` (the future ML training set), `per_date.csv`, `report.md`.

**Design**
- **Point-in-time features** (`pit_features.py`): technicals through the same `as_of`-gated `market_features.py` functions; fundamentals only from `period_end + 90 days` (`--fundamental-lag-days`), because a statement can't be traded on before it's filed. Stale/delisted tickers get no features rather than a frozen snapshot.
- **Scores = production code**: `score_full`/`score_technical` are `horizon.compute_horizon_weighted_view` itself (3-month horizon -> MEDIUM profile by default), so the backtest tests what the app ranks by -- not a re-implementation.
- **Labels** (`labels.py`): 63-trading-day return minus SPY, entered at the close of the day *after* the signal.
- **Evaluation** (`evaluate.py`): per-date rank IC, top-minus-bottom quintile spread, top-quintile hit rate; Newey-West t-stats (monthly rebalance + 3-month labels overlap); the app's Strong/Favorable/Neutral/Unfavorable buckets vs forward excess return; IC by year. Reference signals: 12-1 momentum and low volatility -- if the score can't beat one-line momentum, its complexity isn't paying.
- `walk_forward_splits` (expanding window + embargo = label length) is ready for the ML stage.
- Tests (`tests/test_backtest.py`) check that removing all data after a date leaves every feature on or before it unchanged, that fundamentals stay invisible until the lag passes, that labels enter the next day, and that a planted signal is detected while noise isn't.

**Limits** (also printed in every report): with `--universe current` the universe is today's 13F holdings (survivorship bias -- fixed by the point-in-time universe, STEP 11b); yfinance fundamentals cover only ~4 fiscal years and are as-restated, so the fundamentals window starts ~3 years back while technicals cover the full history; forward P/E can't be reconstructed (valuation = FCF yield only); historical market cap is today's scaled by the price ratio; the 0.10 13F tilt is only tested in `pit` mode (STEP 11b); spreads are gross of costs.

**Reading the result**: rank IC > ~0.03 with a NW t-stat > 2 and Strong > Unfavorable in section 3 means the ranking carries real information; otherwise the Quick Picks labels are noise and the sizing on that page shouldn't be trusted with real money.

## STEP 11b -- Point-in-time 13F universe for the backtest

**Why.** The first real STEP 11 run (87 tickers, 2017-2026) looked wrong in a specific way: *every* score bucket beat SPY by ~5% per quarter, the labels were inverted (Unfavorable +12.6% vs Strong +4.9% forward excess), and low volatility had a rank-IC t-stat of -3.3. That is the signature of a biased universe, not of a signal: the backtest applied **today's** 13F holdings to every past date. Today's holdings are, by construction, companies that survived and grew into large positions -- so in 2017 the "universe" already knew who would win, and the most volatile (riskiest-looking) names in it were the ones that later went up the most. No model can be judged on that universe, so this is fixed before any ML.

**What changed.**

```text
SEC submissions JSON (filings.recent + every filings.files[] page back to the start year)
    -> every 13F-HR since --history-start-year (default 2016), one per quarter
    -> same per-accession parquet cache, compare_quarters, select_positions, direction mapping as STEP 7
    -> CUSIP -> ticker (security_master, OpenFIGI, cached; paced for the free 25 req/min tier)
    -> data/processed/backtest/universe_history.parquet
       one row per (filing_date, filer, CUSIP): report_period, ticker (None if unresolved),
       status, view_direction, confidence, value, weight
```

- `holdings_13f.build_universe_history()` / `build_filer_history()`. The live path (`refresh_13f_mentions`, latest two quarters, `institutional_mentions.parquet`) is unchanged and never written by history mode.
- **Availability date = SEC filing date**, never the quarter end. A Q1 13F (quarter end 31 March) filed on 15 May enters the universe on 15 May, not before.
- `dataset.build_panel(universe="pit", universe_history=...)`: on each rebalance date, a ticker is a member only if it is in the **latest filing of at least one filer with `filing_date <= date`**. Labels enter at the next day's close, so a filing made after that day's close is still tradable. A filer's latest filing stops counting after 200 days (`PIT_MAX_FILING_AGE_DAYS`), so a filer that stops filing doesn't freeze its last holdings into the universe.
- **Point-in-time `institutional_direction_score`** per (date, ticker), with the same aggregation as `universe.py` (`aggregate_direction` over the filers' rows), and **`score_rank = score_full + 0.10 x institutional_direction_score`**: the app's full `rank_score`, now testable. In `pit` mode the Strong/Favorable/... labels (report section 3) are cut from `score_rank`, as in the app.
- **Coverage is counted, not dropped**: members whose CUSIP didn't resolve, or whose ticker yfinance can't price (delisted, renamed), stay in `universe_membership.parquet`. The report gives per year the share with a ticker, the share priced, and the number of tickers never priced (`universe_coverage.csv`).

**Run it.**

```bash
python src/run_backtest.py --build-universe-history                       # all filers in 13f_filers.csv, 2016+
python src/run_backtest.py --build-universe-history --history-start-year 2019 \
    --filers Berkshire=1067983 "State Street" JPMorgan Fidelity --start 2019-03-01
python src/run_backtest.py                      # pit is the default once universe_history.parquet exists
python src/run_backtest.py --universe current   # the old, survivorship-biased universe, for comparison
```

The first history build is slow: a few thousand SEC requests (<= ~7 req/s), plus OpenFIGI for every new CUSIP at 25 requests/min without `OPENFIGI_API_KEY`. After that, filings and CUSIP resolutions are cached, and prices go to `data/raw/prices_long/` as in STEP 11.

**New in the report**: section 0 (universe size and price coverage by year), `score_rank` next to the other signals, and section 5: the mean forward excess return of the *whole* universe per year. On an unbiased universe this should be close to 0. If it is more than 1.5% per label period, the report says so explicitly. An equal-weighted universe can still legitimately trail a cap-weighted SPY for a while (e.g. mega-cap-led years).

**Remaining limits.**
- **Delisted names can't be priced by yfinance**: they are in the universe but outside every metric. Coverage (section 0) shows how big that hole is, and it is largest in the earliest years. Some survivorship bias remains until a price source with delisted history is added.
- **13F = long US equity positions only**: no shorts, options, non-US holdings or intra-quarter trades; amendments (13F-HR/A) are ignored.
- CUSIP -> ticker uses today's OpenFIGI mapping: a CUSIP retired after a merger may not resolve (counted as "no ticker"), and a ticker later reused by another company would price the wrong stock.
- Filer CIKs change (BlackRock files under CIK 2012383 only since 2024; its earlier 13Fs are under 1364742), so a default filer list can have a shorter history than the start year suggests. Check the per-filer `first_filed` printed by `--build-universe-history`.
