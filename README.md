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

## STEP 12 -- Learned model vs the baseline, out of sample only

**Why.** The point-in-time backtest (STEP 11b) found no robust edge in the hand-weighted score. Over the full history, `score_rank` has a rank IC of 0.020 (NW t 0.7) and its labels are non-monotonic. 12-1 momentum reaches t 2.1 only in the 2023-26 window. Before re-weighting by hand, this step asks whether a model can **learn** a better cross-sectional ordering from the same point-in-time features, judged only on dates it never trained on.

```bash
python src/run_backtest.py --reuse-panel --model     # needs the --universe pit panel
```

**`src/backtest/model.py`**
- **PIT panel only**: `require_pit_panel` refuses the current-universe panel, which is survivorship-biased.
- **Features** are an explicit whitelist (`model.FEATURES`):
  - technicals: returns, relative returns, volatility, volume ratio, price vs MA50/200, 12-1 momentum
  - fundamentals: revenue/EPS/FCF growth, net and FCF margin, ROE, FCF yield, net debt
  - point-in-time institutional: `institutional_direction_score`, `institution_count`

  `fwd_*`, `score_*`, `label`, `date` and `ticker` can never enter the feature matrix. This is checked at import and on every call, and is tested.
- **Per-date cross-sectional ranks** in [0, 1] for every feature and for the target (`fwd_excess`). Only the ordering within a date matters to a ranking, and ranks don't drift with market regimes. NaN stays NaN; fundamentals only exist from ~2022.
- **`score_ml`**: LightGBM regressor with fixed, deliberately conservative parameters (15 leaves, >= 100 rows per leaf, learning rate 0.03, 300 trees, feature/bagging fraction 0.8, L2 1.0). There is no tuning on test folds.
- **`score_linear`**: ridge regression on the same rank features (NaN -> 0.5). If LightGBM can't beat a linear model, its non-linearity isn't paying.
- **Walk-forward** (`evaluate.walk_forward_splits`):
  - expanding window, first 36 dates train-only, test blocks of 6 dates
  - embargo = overlapping-label dates + 1 = 3 dates (63 trading days), so no training label's return window reaches a test date
  - trained on labelled rows only; rows never in a test fold keep NaN
- Outputs in `data/processed/backtest/`: `predictions.parquet` (date, ticker, fold, score_ml, score_linear) and `feature_importance.csv` (mean LightGBM gain across folds).

**Report section 6.** It compares `score_ml`, `score_linear`, `score_rank`, `score_full`, `momentum_12_1` and `low_volatility` on **exactly the same rows** (those with an OOS `score_ml`). It also shows score_ml vs score_rank IC by year, the top-10 features, forward excess by `score_ml` quintile, and the promotion check below.

**Promotion rule.** `score_ml` may replace `rank_score` in `screener.py` **only if** all of the following hold:
1. against `score_rank` **and** against `momentum_12_1`: the Newey-West t-stat of the **per-date rank-IC difference** (score_ml minus that baseline, on the same rows and dates) is > 2. A higher mean alone isn't enough, because two noisy IC series can differ in mean by chance. score_ml's own t-stat is still printed, for information only;
2. there are at least 4 OOS calendar years, and its mean rank IC is positive in at least 4 of them.

(Revised in STEP 13. The first version compared means, required score_ml's own t > 2, and required 5 positive years, which with 5 OOS years meant every year.)

The report evaluates this automatically (`model.promotion_check`). The model is **not** wired into the app in this step, whatever the verdict.

**First result** (7-filer PIT panel, 49 OOS dates 2022-05 -> 2026-06, ~107 names/date): **score_ml does not qualify.**

| Signal | OOS rank IC | NW t |
|---|---|---|
| score_ml | 0.020 | 0.66 |
| score_rank | -0.012 | -0.36 |
| momentum_12_1 | 0.045 | 1.07 |
| score_linear | -0.069 | -2.54 |

- score_ml's IC is positive in only 2 of 5 years, and its quintiles are flat except the bottom one.
- The ridge baseline is significantly *negative*: relationships fitted in-sample reversed out of sample. That is a warning against hand re-weighting on this data too.
- The Quick Picks page now carries a visible warning that the ranking has no demonstrated edge.

## STEP 13 -- More power, new information

**Why.** STEP 12 found no learnable edge, but the test was weak: ~107 names per date is a narrow cross-section, and every feature was a transform of prices, fundamentals everyone has, or 13F data that is 45+ days old. This step widens the universe and adds two sources of information that could plausibly carry a signal.

```bash
python src/run_backtest.py --universe pit_broad --start 2019-03-01 --model
python src/run_backtest.py --reuse-panel --model          # re-evaluate; the saved panel remembers its universe
```

**1. Broad point-in-time universe (`--universe pit_broad`, `dataset.pit_broad_universe`).**
- On each date, the universe is the **top 500 US stocks by total value** across the filers' latest **full** 13F holdings. Availability follows the same rules as STEP 11b: SEC filing date <= date, and a filing stops counting after 200 days.
- ETFs and funds are excluded (by the CUSIP resolution's security type). Unresolved CUSIPs stay in as members with no ticker, and are counted by coverage (report section 0).
- Holdings come from the per-accession parquet cache, so nothing already cached is downloaded again.
- **13F value units**: filings made before 3 Jan 2023 report values in thousands of dollars, later ones in dollars. They are normalized before summing across filers, otherwise the ranking breaks around the switch.
- `institution_count` is the number of filers holding the stock. `institutional_direction_score` uses the live aggregation over the filers' selected rows, and is 0 when none of them selected the stock.

**2. Insider buying: SEC Form 4 (`src/insider_form4.py`).**
- **What counts**: officer/director open-market purchases (`P`) and sales (`S`) only. Grants, option exercises, tax withholding and gifts are compensation plumbing, not decisions. 10% owners who are not officers or directors are dropped.
- **Point-in-time** by SEC **filing** date, never the trade date.
- **Features per (date, ticker)**, over a 90-day window:
  - `insider_buyers_90d`: distinct buyers;
  - `insider_net_buy_value_90d_mcap`: (purchase value - sale value) / market cap;
  - `insider_cluster_buy`: 1 if there are >= 3 distinct buyers.
- **Missing vs zero**: no filings means 0. A ticker without an issuer CIK, or a date past the data's coverage, gets NaN ("unknown" is not "nobody bought").
- **Two sources**, same transaction table:
  - `per-issuer` (as specified): ticker -> CIK via SEC `company_tickers.json`, the issuer's submissions JSON, then each Form 4's XML, cached per accession. That is one request per filing: ~300k requests, ~13 hours at SEC fair-access pacing for this backtest.
  - `bulk` (default for the backtest): the SEC's quarterly Insider Transactions Data Sets. Same fields, one ~13 MB zip per quarter, parsed once and cached as parquet in `data/raw/form4/bulk/`.

  Select with `--insider-source`.

**3. Best-ideas 13F features.**
- Concentrated active managers are listed in `data/raw/13f_active_managers.csv` (`institution,cik`; default Berkshire 1067983).
- Their **full** books per quarter (`holdings_13f.build_position_history`, cached in `active_positions.parquet`) give:
  - `active_new_or_add`: 1 if any listed manager opened the position, or added > 10% flow-adjusted, in its latest filing;
  - `active_weight_max`: the largest portfolio weight among them.
- NaN when no listed manager has a filing public yet.

**4. Explicit size.** `log_market_cap` is a feature, so the model can't use `institution_count` as a hidden size proxy.

**Report.** Section 6 now also lists each new feature as a **standalone signal** (rank IC and t-stat on the same OOS rows), next to score_ml, score_rank and momentum_12_1. The promotion rule was tightened (see STEP 12): it now uses the NW t of the per-date IC **difference**, and needs >= 4 OOS years with >= 4 of them positive.

**Limits.**
- The broad universe is large-cap by construction: top 500 by institutional value.
- Form 4 bulk data sets are published after each quarter, so the latest weeks have no insider features.
- `company_tickers.json` is today's ticker map, so renamed or delisted issuers are missed.
- Best ideas start with a single manager.

**First result** (top-500 PIT universe, 690 tickers, 90 monthly dates 2019-03 -> 2026-09; OOS: 51 dates from 2022-06, ~450 names/date): **no edge.**

| Signal (same OOS rows) | Rank IC | NW t |
|---|---|---|
| score_ml | 0.011 | 0.36 |
| score_rank | -0.009 | -0.40 |
| momentum_12_1 | 0.026 | 0.82 |
| log_market_cap | 0.035 | 2.36 |
| insider_buyers_90d / net buy / cluster | 0.007 / 0.013 / 0.000 | 0.73 / 0.60 / 0.00 |
| active_new_or_add / active_weight_max | -0.011 / -0.003 | -1.04 / -0.26 |

- The insider and best-ideas features carry no standalone signal. LightGBM still ranks them at the top by gain, which is noise-fitting.
- Only size is significant (large caps beat smaller ones in 2022-26, a known regime effect).
- Promotion rule: fails all three checks (IC difference t 0.52 vs score_rank and -0.43 vs momentum; positive in 2 of 5 years).

## STEP 13b -- Trade-level backtest of the short-term loop + triple-barrier meta-labeling

**Why.** The app's short-horizon use is a trading loop, not a 3-month ranking: buy the picks, exit at the stop or target the app shows, or after N days, then repeat. A loop can make or lose money independently of ranking IC (costs, stop placement, position limits), so it is tested as a loop, on the point-in-time universe (`--universe pit`).

```bash
python src/run_strategy_backtest.py                  # weekly PIT panel, all variants, grid, meta-labeling, report
python src/run_strategy_backtest.py --reuse-panel    # skip the price/fundamentals panel build
python src/run_strategy_backtest.py --seeds 20       # fewer random seeds (faster, coarser percentiles)
```

Outputs go to `data/processed/backtest/strategy/`: `report.md`, `grid.csv`, `trades_*.parquet`, `candidates.parquet`, `ml_probabilities.parquet`.

**Labels (`src/backtest/barriers.py`).**
- **Entry** at the **next day's open**. An entry that gaps through its stop or target is not taken.
- **Levels** from the signal close via `risk_reward.compute_risk_reward` (the app's own levels at a 14-day horizon: 2.5x ATR stop or a tighter 60-day swing low; target = 60-day swing high, else 2R), or grid levels.
- **Daily bars** are walked until the first barrier:
  - stop and target in the same bar: **the stop wins**;
  - a gap through the stop fills at the (worse) open;
  - a gap through the target gets no extra credit;
  - time exit at the close of the last hold day.
- **Costs**: 0.10% + 0.05% slippage, **per side**.
- Stored per trade: outcome, net return, R-multiple, days held, and y (1 if the target came first).

**Strategy (`src/backtest/strategy.py`).**
- **Rules**: each week, top 5 picks, max 10 open positions, 1% of equity at risk per trade (cap 20% per position).
- **Variants**:
  - `score_rank`: the app's rank_score at the 14-day horizon + the Quick Picks filter (Strong/Favorable, risk/reward >= 1);
  - `momentum_12_1`;
  - `breakout_20d`: close above the prior 20-day high, ranked by 20-day return.
- Picks are fixed at the signal close. A pick that gaps is skipped, never replaced from further down the list.
- **Metrics**: return, CAGR, max drawdown, Sharpe, win rate, average win/loss, expectancy (R), profit factor, trades, by year. Compared with SPY buy-and-hold and an equal-weight PIT-universe portfolio.
- **Random-entry baseline**: the same exits, sizing and limits with random weekly picks from the same universe, 100 seeds (reproducible). The report gives the variant's percentile in that distribution.
- **Grid**: stop 1.5/2/3 ATR x target 1.5/2/3 R x max hold 5/10/20. The full 27-cell expectancy grid is shown for each signal.

**Meta-labeling (`src/backtest/barrier_model.py`).**
- **Model**: a LightGBM classifier for P(target first). Features: the `model.FEATURES` whitelist as per-date ranks, plus trade geometry (stop distance %, target distance %, ATR %). Days to earnings is left out: there is no point-in-time earnings calendar.
- **Training data**: every PIT member's barrier outcome, not only past picks (~5 picks a week is too few). Class imbalance is handled with `scale_pos_weight`.
- **Walk-forward**: 104 weekly dates of training, then 26-date test blocks. The embargo is ceil(max hold / 5) + 1 weeks, and a training trade must have exited before the first test signal.
- **Probabilities** are produced for every candidate on a test date. Whether an entry gaps is next-day information, so it never decides which names get a probability.
- **Threshold** chosen on **training data only**: an inner split of the training window picks the probability cut-off that maximizes mean R.
- **Report**: OOS AUC, a calibration table by decile, and top-decile precision vs the base rate. The filtered strategy is compared with the unfiltered one on the same OOS dates, with its own random baseline.

**"Is it luck?"** Each variant shows its trade count, expectancy with a 95% CI (bootstrapped by week), random-entry percentile, and share of grid cells with positive expectancy. The verdict is **TRADEABLE** only if all three hold:
- the expectancy CI is > 0 after costs;
- the variant beats >= 95% of random pickers;
- >= 2/3 of its grid cells are profitable.

Otherwise it is **NOT PROVEN**. The ML-filtered variant has no grid (its threshold is tied to one geometry), so it cannot be TRADEABLE by this rule.

**Tests** (`tests/test_strategy_backtest.py`) cover:
- same-bar stop first, next-day-open entry, gap fills, gap-at-entry skip, and costs on both sides;
- vectorized ATR/swing levels matching the app's `as_of` functions, and no candidate feature changing when post-signal prices change;
- position limits and sizing, and a random baseline reproducible by seed;
- the embargo, test-fold labels never affecting predictions or thresholds, and no label columns in the meta-features.

**Limits.**
- 13F-selected large caps only.
- Daily bars can't order the stop and target within a day (the stop is assumed first).
- No partial fills, borrow or taxes.
- Fundamentals are as restated by yfinance.

**First result** (381 weekly signals 2019-03 -> 2026-09, ~106 members/week): **every variant NOT PROVEN.**

| Hold 10, app levels | Total return | Max DD | Expectancy (R) [95% CI] | Beats random | Grid cells > 0 |
|---|---|---|---|---|---|
| score_rank (Quick Picks) | +15.1% | -29.2% | +0.005 [-0.055, +0.068] | 93% | 44% |
| momentum_12_1 | -14.8% | -34.2% | -0.063 [-0.142, +0.014] | 69% | 85% |
| breakout_20d | -40.7% | -41.9% | -0.048 [-0.090, -0.009] | 5% | 44% |
| score_rank + ML filter (OOS dates) | -8.5% | -21.8% | -0.033 [-0.121, +0.058] | 90% | n/a |
| SPY buy & hold (same period) | +209.8% | -33.7% | | | |
| Equal-weight PIT universe | +162.9% | -35.5% | | | |

- The best loop (score_rank, hold 20) made +27% in 7.5 years, against +210% for SPY.
- score_rank is consistently above most random pickers (89-98th percentile across holds), but its expectancy CI always includes 0.
- The grid's positive cells cluster at long holds and wide stops, i.e. closer to buy-and-hold in a rising market. The grid has no random baseline, so this is likely market drift rather than skill.
- **The meta-label AUC of 0.88 is mechanical, not skill.** Target distance alone gives an AUC of 0.878: the app's target is the 60-day high, often < 1% away, so it gets "hit". In the top decile of P, the median target is 0.75% away and the mean R is -0.07 after costs. Predicting *hit* is not predicting *profit*, so the filter lowered expectancy. The next meta-label should predict R (or hit with a minimum reward), not the raw hit.
- (The first generated report shows an equal-weight Sharpe of 1.73. That was a weekly series annualized as daily; the correct value is 0.77, fixed in `strategy.buy_and_hold`.)

## STEP 13c -- Final, pre-registered exit test (FINAL RESULT)

**Pre-registered before running.**
- Entries: score_rank only (the app's rank_score at 14 days, Strong/Favorable), weekly top 5, max 10 positions, 1% equity risk per trade.
- Execution: next-day-open entry, **hold 20**, costs 0.15%/side.
- Three exits replacing the app's 60-day-high target, and no others: (a) chandelier trailing stop 2 ATR, (b) trailing stop 3 ATR, (c) fixed target 2R with a 1.5 ATR stop.
- The random-entry baseline uses the **same** exits.
- Verdict rule: unchanged from STEP 13b. With no grid, its robustness leg is applied to the three exits (>= 2 of 3 with positive expectancy).
- The Quick Picks risk/reward filter is not applied, because it is defined on the target being replaced.

```bash
python src/run_strategy_backtest.py --reuse-panel --step13c     # -> data/processed/backtest/strategy/report_13c.md
```

Trailing stop = highest high since entry (completed bars only) - k x ATR(14) at the signal. It is never lowered, and a gap below it fills at the open. The R-multiple uses the initial risk.

| Exit (hold 20) | Trades | Expectancy R [95% CI] | Beats random (same exits) | Return | Max DD | CAGR vs SPY at same avg exposure |
|---|---|---|---|---|---|---|
| (a) trailing 2 ATR | 1,110 | -0.025 [-0.101, +0.046] | 30% | -4.3% | -26.7% | -0.6% vs +11.1% |
| (b) trailing 3 ATR | 959 | +0.053 [-0.012, +0.120] | 78% | +63.5% | -19.3% | +6.7% vs +11.8% |
| (c) 2R / 1.5 ATR | 1,022 | -0.035 [-0.132, +0.059] | 52% | -6.2% | -34.1% | -0.8% vs +11.6% |

Only 1 of the 3 exits has positive expectancy, and it lies within the random pickers' range (78th percentile). Its CI includes 0.

**Final result: none of the pre-registered exits is TRADEABLE.** On a point-in-time, survivorship-aware universe, after costs, the app's short-term trading loop (buy score_rank picks, exit at stop, target or time) has **no demonstrated edge**. At the same average exposure (67-72% invested), every variant did worse than simply holding SPY: CAGR gap -5 to -12.5 points a year.

**Per the pre-registration, no further strategy variants are added.** The Quick Picks warning stays: the ranking is a watchlist, not trade instructions.

## PHASE 1 -- STEP 15 and STEP 14a (final research push)

Both use the same machinery as before: point-in-time 13F universes (history rebuilt back to 2013-Q2, when 13F XML starts), next-open entries, costs, walk-forward with embargo, random baselines and NW t-stats. They also add two new free, point-in-time sources:
- **SEC XBRL fundamentals** (`src/sec_xbrl.py`, from the bulk `companyfacts.zip`). Each value is usable from its filing date, and the earliest-filed value wins across tag switches. Year-to-date cash flows are differenced into quarters. This replaces yfinance statements, which only cover ~4-5 years and are as-restated.
- **A 14-year price cache with the as-quoted close** (`backtest/data.load_ext_prices`), so market cap = quoted price x XBRL shares stays correct across splits.

The verdict rules were fixed before running (see `value_survival.portfolio_verdict` and `run_smallcap.py`).

### STEP 15 -- "Buy cheap quality, sell at target", survival models (`python src/run_value_survival.py`)
- **Candidates**: the top 500 by 13F value, monthly 2013-08 -> 2026-09. A stock qualifies if it is >= 25% below its 52-week high, its P/S is below its own 5-year median (price-only when there are < 36 months of P/S history, flagged), and net margin > 0, FCF margin > 0 and net debt / market cap < 1. That gives 4,197 candidate-months across 401 tickers.
- **Exits**: target +20/+30/+50%; a thesis break (2 consecutive quarters with negative FCF and revenue down year on year, known on the 2nd filing date); a 10-year cap. Training labels are censored at each fold's start: an open position is never a failure.
- **Models** (walk-forward by entry year, 12-month embargo): Kaplan-Meier, Cox PH and XGBoost AFT. The pre-registered primary is XGBoost AFT at +30%.
- **The models have almost no skill.** Out of sample, concordance is 0.52-0.54 (0.50 at +50%), and P(hit <= 12m) is badly overconfident: predicted 20-90%, actual 49-63%.
- **The screen itself is where the returns come from, not the model.** Out of sample, candidates hit +20/+30/+50% in 76/67/55% of cases. Kaplan-Meier portfolios, which use no stock-level model, beat both ML models at every target.
- **Primary portfolio (2017-2026)**: +8.1%/yr vs SPY +15.3% (first half 12.8% vs 18.3%, second half 3.6% vs 12.4%). It beats 76% of random-candidate portfolios. Sharpe is 0.53 vs 0.88. **DOES NOT BEAT MARKET** (fails all 5 checks).
- **Coverage caveat**: in 2013-15 only ~63% of the top-500 CUSIPs resolve to a ticker (old CUSIPs of acquired or renamed companies), rising to 92% by 2025.

### STEP 14a -- Small caps where institutions can't trade (`python src/run_smallcap.py`)
- **Universe**: every common stock in the filers' full 13F books, with ETFs, funds and ADRs excluded. Market cap $200M-$2B (quoted price x XBRL shares), 20-day dollar volume > $1M, price > $3. That is 93,464 stock-months across 1,958 tickers (~400 names/month in 2013, ~860 in 2026). A rough 13F-implied cap (13F value / 20%) only decides which CUSIPs get looked up.
- **Features and model**: the whitelist, plus SUE from XBRL EPS (dated by filing date), days since the last filing, and the 3-month change in 13F ownership breadth. LightGBM on per-date ranks, target = 63-day return minus IWM. Walk-forward with 36 training dates, test blocks of 6, embargo 4. OOS: 114 dates, 2016-12 -> 2026-06, ~636 names/date.
- **Signal results**:
  - score_ml rank IC is -0.001 (t -0.05), and its top-minus-bottom quintile spread is *negative* (t -2.4).
  - Momentum is -0.007, SUE alone 0.004, linear 0.008.
  - The only standalone signal with t > 2 is `insider_buyers_90d` (IC 0.013, t 2.09), borderline once 6 signals were tested.
- **Portfolio** (top decile, 0.30%/side): **+6.4%/yr as priced, vs IWM +9.5% and SPY +15.2%** (first half 12.1 vs 12.7 / 18.0; second half 0.9 vs 6.4 / 12.5).
- **Survivorship cases.** With unpriced members assumed to lose 50% or 100% per label period, the result is -66%/yr and -100%/yr.
  - These bounds are extreme because "unpriced" here is mostly CUSIPs OpenFIGI can't map at all: 69% of the band in 2013, 17% in 2026. That includes some non-common shares. Only 27 of 2,898 tickers failed to download.
  - The verdict does not depend on them: the strategy already trails IWM as priced.
- **Verdict: DOES NOT BEAT MARKET** (fails all 3 checks: returns, IC difference vs momentum t 0.37, IC > 0 in 5 of 11 years).

## PHASE 1 VERDICT

Every strategy tested from STEP 11b to STEP 15, on point-in-time, survivorship-aware universes, after costs. "Annualized" is the strategy's own period.

| Step | Strategy | Test | Annualized after costs | SPY / IWM, same period | Verdict |
|---|---|---|---|---|---|
| 11b | App rank_score, 3-month ranking (13F selection universe) | rank IC 0.020, t 0.69 | -- (ranking test, no portfolio) | -- | No edge |
| 12 | LightGBM ranking (score_ml) | OOS rank IC 0.020, t 0.66 | -- (ranking test) | -- | Not promoted |
| 13 | LightGBM + insider + best-ideas, top-500 universe | OOS rank IC 0.011, t 0.36 | -- (ranking test) | -- | Not promoted |
| 13b | Short-term loop, app picks, hold 10 | expectancy +0.005R, CI [-0.055, +0.068] | +1.9% (2019-26) | SPY +16.1% | NOT PROVEN |
| 13b | Short-term loop, 12-1 momentum, hold 10 | expectancy -0.063R | -2.1% | SPY +16.1% | NOT PROVEN |
| 13b | Short-term loop, 20-day breakout, hold 10 | expectancy -0.048R | -6.7% | SPY +16.1% | NOT PROVEN |
| 13b | App picks + meta-label ML filter | expectancy -0.033R | -1.6% (OOS dates) | SPY higher | NOT PROVEN |
| 13c | App picks, trailing stop 2 ATR, hold 20 | expectancy -0.025R | -0.6% | SPY at same exposure +11.1% | NOT PROVEN |
| 13c | App picks, trailing stop 3 ATR, hold 20 | expectancy +0.053R, CI [-0.012, +0.120] | +6.7% | SPY at same exposure +11.8% | NOT PROVEN |
| 13c | App picks, 2R target / 1.5 ATR stop, hold 20 | expectancy -0.035R | -0.8% | SPY at same exposure +11.6% | NOT PROVEN |
| 15 | Cheap quality + XGBoost survival, +30% (primary) | concordance 0.54 | +8.1% (2017-26) | SPY +15.3% | DOES NOT BEAT MARKET |
| 15 | Same, best variant (no model, +50%) | -- | +10.1% | SPY +15.3% | DOES NOT BEAT MARKET |
| 14a | Small-cap LightGBM, top decile | OOS rank IC -0.001, t -0.05 | +6.4% as priced; -66% in the -50% survivorship case | IWM +9.5%, SPY +15.2% | DOES NOT BEAT MARKET |

**Conclusion.** Tested with point-in-time data, embargoed walk-forward validation, costs, random baselines and pre-registered verdict rules, none of the ranking, trading-loop, value/survival or small-cap strategies beat simply holding SPY. Several beat most random pickers (the STEP 15 screen and the app's short-term picks), but not the market. **Per the pre-registration, no further strategy variants are added.** The app stays a research and watchlist tool, and the Quick Picks warning stays.
