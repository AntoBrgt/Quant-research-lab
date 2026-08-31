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
The extraction layer uses LangChain with a local Ollama model by default, so it does not require paying for API tokens during the prototype phase.

For this project, Qwen is the recommended default local model because it tends to handle structured extraction and instruction-following well on SEC text.

Required environment variables for Ollama:

- `OLLAMA_BASE_URL` (optional, defaults to `http://localhost:11434`)
- `OLLAMA_MODEL` (optional, defaults to `qwen3:8b`)

Optional fallback for OpenAI-compatible usage:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `LLM_PROVIDER` can be set to `ollama` or `openai`

### Example commands

```bash
python src/extract_signals.py
python src/extract_signals.py --ticker AAPL
python src/extract_signals.py --ticker AAPL MSFT JPM
python src/extract_signals.py --max-chunks 20
```

For a local Ollama setup, the model should already be pulled and running on your machine, for example:

```bash
ollama pull qwen3:8b
```

### Architecture boundary

This step is intentionally limited to the signal-extraction layer. It does not produce BUY/SELL/HOLD recommendations and does not use future market data.

The future architecture is:

```text
Financial documents
    ↓
Structured signals
    ↓
News signals
    ↓
Market data
    ↓
Historical validation
    ↓
Strategy engine
    ↓
Portfolio context
    ↓
BUY / HOLD / SELL / WATCH
```

This separation is required so the same signal can support different strategies over different horizons.

### Prototype limitations

- The model receives only the filing text and metadata, never future stock-price data.
- We do not yet perform backtesting, portfolio analysis, or recommendations.
- The LLM is restricted to a fixed signal taxonomy and is designed for research use, not investment advice.
- Local model quality can vary with the Ollama model choice and runtime environment.
- The extraction layer is intentionally minimal and will need refinement as the project grows.

