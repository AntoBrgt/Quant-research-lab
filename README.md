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

Upload a portfolio CSV, pick a risk profile and strategy, and run analysis. Only tickers with no cached signals yet trigger new (cache-first) extraction -- everything else reuses shared research.

### Architecture boundary

Recommendations are model outputs with stated evidence, confidence, and data freshness -- not personalized financial advice, not a promise of performance, and never a fabricated price target.

### Prototype limitations

- The filesystem JSON cache and parquet usage log are fine at today's scale (a handful of tickers, one process) but aren't safe for concurrent writers or a large key count -- a real key-value store is the natural next step before many simultaneous users.
- There is no auth or multi-tenant storage; portfolios are files passed in per run, not persisted per user.
- JPM's filing text doesn't match the current section-heading regexes for Business/Risk Factors (only "Notes to Financial Statements" is detected) -- the heading heuristics in `process_documents.py` need broadening before JPM-style filings are fully covered.
- The news provider (`yfinance` headlines) is free and best-effort, not a real news feed -- coverage and quality will be thin for less-followed tickers.
- Sector exposure depends on `yfinance`'s `info` payload, which is itself best-effort and can be missing per ticker.

