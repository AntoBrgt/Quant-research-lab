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

### Portfolio & recommendations

`src/portfolio.py` validates a CSV (`ticker, quantity, average_cost[, currency]`) and computes market value/weight/P&L/concentration/sector exposure in plain pandas -- no LLM. `src/strategy.py` scores a company's cached research against a chosen horizon (`short_term`/`medium_term`/`long_term`) using one configurable weights dict. `src/recommendations.py` combines asset signal, strategy fit, portfolio fit, and risk into a `Recommendation` -- kept as separate, inspectable fields, never collapsed into one score -- with `INSUFFICIENT_EVIDENCE` returned rather than a forced call when signal coverage is thin.

### Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a portfolio CSV -- either the plain `ticker, quantity, average_cost` format or a supported broker export (see STEP 5) -- pick a risk profile and strategy, and run analysis. Only tickers with no cached signals yet trigger new (cache-first) extraction -- everything else reuses shared research.

## STEP 5 — Portfolio input normalization

Real users upload broker transaction exports (every BUY/SELL/DIVIDEND/CASH/corporate-action event), not the clean `ticker, quantity, average_cost` snapshot `portfolio.py` was originally built for. `src/portfolio_importers/` bridges the two, broker-agnostically:

```text
raw broker CSV
    -> detect.detect_format()                        (column-based, not filename-based)
    -> a per-broker adapter (e.g. trade_republic.py)  -- NORMALIZATION
    -> schema.CanonicalTransaction rows
    -> schema.reconstruct_positions()                 -- RECONSTRUCTION
    -> schema.CanonicalPosition rows
    -> schema.to_simple_portfolio()                   -- bridges to portfolio.py's existing schema
    -> portfolio.py (validate_portfolio / enrich_positions / recommendations, unchanged)
```

Normalization and reconstruction never touch the network or live prices -- an imported portfolio can be built and tested with zero network access. Market-price enrichment stays `portfolio.py`'s separate job, applied after reconstruction.

### Canonical transaction schema (`portfolio_importers/schema.py`)

| Field | Notes |
|---|---|
| `transaction_id` | Preserved from the broker; must be unique |
| `date` | Full timestamp (not just a calendar date) -- ordering matters for cost-basis accounting |
| `instrument_id` | Broker-agnostic identity -- see below |
| `identity_type` | `ISIN` \| `SYMBOL` \| `NAME` \| `BROKER_ID` -- how trustworthy `instrument_id` is |
| `name`, `symbol`, `asset_class` | Preserved as given by the broker |
| `side` | `BUY` \| `SELL` \| `DIVIDEND` \| `INTEREST` \| `CASH_IN` \| `CASH_OUT` \| `OTHER` -- only `BUY`/`SELL` affect reconstruction |
| `quantity`, `price`, `fees`, `tax` | Always non-negative magnitudes; direction lives in `side`, never in a sign |
| `currency`, `amount` | `amount` keeps the broker's own signed net cash flow |
| `broker`, `raw_type` | Traceability back to the source row |

### Canonical position schema

`instrument_id, name, symbol, asset_class, quantity, average_cost, total_invested, total_fees, currency`.

**Cost-basis methodology: moving-average cost**, not FIFO/LIFO lot tracking, and not a tax-accounting method -- for portfolio analytics only, no tax/accounting claim is made. A BUY updates `average_cost` as a quantity-weighted average; a SELL reduces `quantity` only, leaving `average_cost` of the remaining shares unchanged. `total_invested = quantity * average_cost` (the cost basis of what's currently held). `total_fees` sums every fee/tax paid on the instrument, ever -- not reduced by sells. A position that nets to (approximately) zero quantity is not returned as a current holding.

### Instrument identity

`symbol` is not assumed to be a ticker. `instrument_id` is chosen in priority order -- ISIN (detected by shape: `^[A-Z]{2}[A-Z0-9]{9}[0-9]$`) > symbol/ticker > name -- and `identity_type` records which tier was used. This layer does **not** resolve an ISIN to a tradeable ticker (no security-master lookup); `schema.to_simple_portfolio()` uses `symbol` as-is, so a fund ISIN with no ticker mapping simply fails `portfolio.py`'s existing ticker validation, honestly, rather than being silently guessed.

### How broker adapters work

Each adapter is one module in `src/portfolio_importers/` (e.g. `trade_republic.py`) that owns:
- `REQUIRED_COLUMNS`: the raw columns it needs (used by `detect.py` for format detection)
- `parse(df) -> (list[CanonicalTransaction], list[rejected_row])`: maps the broker's own transaction-type vocabulary to `TransactionSide`, resolves instrument identity, and validates each row -- a malformed row (bad date/number, a trade missing its instrument, a duplicate `transaction_id`) is collected with a reason in the rejected list rather than raising and aborting the whole import, or being silently dropped.

**To add a new broker:** write `src/portfolio_importers/<broker>.py` with its own `REQUIRED_COLUMNS` and a `parse()` matching the shape above; add one `elif`-equivalent branch to `detect.detect_format()`; add the module to `app.py`'s `IMPORTERS` dict. No other file needs to change -- `schema.reconstruct_positions()` is broker-agnostic and already handles whatever `CanonicalTransaction` rows the new adapter produces.

### Supported input formats today

- **Trade Republic transaction export** (`src/portfolio_importers/trade_republic.py`) -- currently the *only* broker adapter
- **Plain portfolio CSV** (`ticker, quantity, average_cost[, currency]`) -- `portfolio.py`'s original format, still fully supported, detected as `"canonical"`

### The full input-to-recommendation flow

`app.py`'s single upload widget drives both formats through one path (`import_and_prepare_portfolio()`), which stays separate from, but feeds directly into, the unchanged recommendation pipeline:

```text
Broker export ──┐
                ├→ detect_format() → adapter.parse() → reconstruct_positions() → to_simple_portfolio() ──┐
Simple CSV ─────┘                                                                                        │
                                                                                                            ▼
                                                                          portfolio.validate_portfolio()  (splits: analyzable / unmapped)
                                                                                                            │
                                                                                                            ▼
                                                              portfolio.enrich_positions() → research_engine (cached, no LLM)
                                                                                                            │
                                                                                                            ▼
                                                                              strategy.score_strategy_fit() → recommendations.generate_recommendation()
```

Positions that fail `validate_portfolio()` (most commonly: a fund/stock ISIN with no ticker mapping -- ISIN→ticker resolution is deliberately not implemented) are **never silently dropped**. They're shown in a separate "unmapped" table with their name and asset class (not just the raw ISIN), so the reason no recommendation exists for them is visible, not mysterious. Importantly: a real Trade Republic export's `symbol` column is an ISIN for every stock and fund position -- only crypto happens to already be ticker-shaped -- so **today, uploading a real Trade Republic export analyzes only crypto holdings**; every stock/fund position lands in the unmapped table until a ticker-mapping step is added (see limitations).

The importer never triggers extra LLM calls: it only ever produces a `ticker, quantity, average_cost, currency` frame, and it's `_ensure_research_available()` (unchanged) that decides whether a ticker needs new research -- exactly the same cache-first check regardless of whether the ticker came from a plain CSV or a broker export.

### Architecture boundary

Recommendations are model outputs with stated evidence, confidence, and data freshness -- not personalized financial advice, not a promise of performance, and never a fabricated price target.

### Prototype limitations

- The filesystem JSON cache and parquet usage log are fine at today's scale (a handful of tickers, one process) but aren't safe for concurrent writers or a large key count -- a real key-value store is the natural next step before many simultaneous users.
- There is no auth or multi-tenant storage; portfolios are files passed in per run, not persisted per user.
- JPM's filing text doesn't match the current section-heading regexes for Business/Risk Factors (only "Notes to Financial Statements" is detected) -- the heading heuristics in `process_documents.py` need broadening before JPM-style filings are fully covered.
- The news provider (`yfinance` headlines) is free and best-effort, not a real news feed -- coverage and quality will be thin for less-followed tickers.
- Sector exposure depends on `yfinance`'s `info` payload, which is itself best-effort and can be missing per ticker.
- Corporate actions (spin-offs, splits, mergers) are not modeled as position-changing events -- a `SPIN_OFF` row, for example, is preserved but never turns into a new position, and is rejected outright if it lacks a currency (observed on a real export).
- Only one broker adapter exists today (Trade Republic); `detect.py`'s precedence logic for an ambiguous/overlapping schema between two future adapters is untested against a real second broker.
- **No ISIN→ticker mapping means most of a real broker export is currently unanalyzable.** Validated against a real Trade Republic export: 15 reconstructed positions, only 1 (Bitcoin) was ticker-shaped enough to reach recommendations; the other 14 (every individual stock and fund) are correctly shown as unmapped rather than silently guessed, but that means zero actionable recommendations today for the stock/fund side of a typical real portfolio. This is the deliberate, documented tradeoff of not doing security-master resolution -- worth revisiting once there's an intentional decision on how to source ISIN→ticker mappings.

