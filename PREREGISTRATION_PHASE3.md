# PHASE 3 pre-registration: new information, our own text models, an LLM agent, a forward-tested ledger

Written 2026-10-12 and committed **before any Phase 3 test touched data**. Nothing below may change after
that commit. A wrong rule is fixed by a new, dated pre-registration, and this one is reported as it stands.

## Prerequisites (done)

| Prerequisite | Result | Where |
|---|---|---|
| Phase 2 (H1 insider cluster, H2 top-decile SUE, H3 cheap quality + trend), development then the one-time holdout | all three **NOT PROVEN** (holdout 2026-10-10, commit `2aea837`) | README, PHASE 2 |
| Phase 1b: the 11 registered variants, one holdout look under the 2026-10-10 addendum | all 11 **NO EDGE**; **Lazy Prices FAILED** (commit `09ca96c`) | README, PHASE 1b addendum |
| Quick Picks: the Phase 2 "No pass" clause | applied: the page is a watchlist, with no BUY/SELL (commit `0029055`) | README, PHASE 2 |

Both phases are closed and merged into `main` (`5747704`). Phase 3 works on branch `phase3`, from `main`.

## Honest caveat: there is no untouched history left

Every month from 2013 to 2026-09 has now been looked at:
- Phase 1 used the full period;
- Phase 1b opened 2024-01 onward;
- Phase 2 opened 2024-09 onward.

Two of the funnel triggers below (insider buying, SUE) are the H1/H2 signals that just failed. Phase 3 backtests are therefore **research filters on reused data**, not clean tests. Consequences, fixed here:
1. A backtested PROMOTED (STEPs 18-19) only admits a model to the **live** pipeline as an arm of the ledger. It never makes anything tradeable.
2. **Only the forward test (STEP 22) can make an arm TRADEABLE.** It is the first truly fresh data.

## Lazy Prices: STEP 16 is dropped

Phase 1b tested Lazy Prices text-change features on the top-500 universe:
- TF-IDF and MiniLM similarity of Risk Factors and MD&A, year over year;
- development 2013-2023, holdout 2024-01 onward.

It **failed**: no text variant passed, and text added holdout IC in only 2 of 3 models, by noise-level amounts.
- The addendum's rule: if Lazy Prices failed, STEP 16 is dropped. **STEP 16 is not run.**
- The STEP 17 funnel has **no** text-change trigger.
- The STEP 19 ranker has **no** STEP 16 features.
- Phase 1b never tested the small-cap band. That gap is recorded here; it is not a reason to re-test a signal family that has just failed.

## Non-negotiable rules

1. **No large LLM in any backtest.** Backtested features may come only from: TF-IDF; models we train on PIT data; the pretrained models in the table below, under rule 1b.
   - `tests/test_phase3_no_llm.py` imports every backtest module in `src/phase3/` and fails if any of them imports, directly or transitively, `signal_extraction`, `llm_usage`, `src/phase3/agent.py`, `anthropic` or `openai`.
2. **Rule 1b, one cutoff rule for every pretrained model.** Each model's cutoff is the **public release date** of the version used. That is a conservative upper bound on its training data.
   - The **primary** test uses only signal dates strictly after the cutoff of every pretrained model behind any feature in that test.
   - Earlier dates are reported separately and never count toward a verdict.
   - `phase3/cutoffs.py` holds the table below, and the evaluation code takes its primary start date from it.
   - A test fails if a primary metric includes an earlier date.
   - TF-IDF vocabularies, IDF weights, scalers and any fitted preprocessing are fit **inside each training fold only**. A test checks this by changing test-fold text and confirming the fitted state is unchanged.
3. **Point in time.**
   - SEC text is dated by its **acceptance datetime** (EDGAR `acceptanceDateTime`; the filing date when missing, flagged).
   - Entry is at the open of the first trading day **strictly after** the acceptance date, even for pre-market filings.
   - Costs: 0.30%/side for small/mid caps, 0.15%/side for the top 500.
   - Benchmarks: IWM for small/mid caps, SPY for the top 500.
4. **Every verdict rule below is fixed in code and README before its test runs.** There is no tuning on test folds; hyperparameters come from an inner validation split of each training window only. A failed test is recorded, with no rescue variants.
5. **A model is kept only if it beats, out of sample: random ordering, then momentum_12_1, then the previous complexity level.**
6. **Paper only.** No broker integration or order placement, anywhere.
7. One STEP at a time. After each STEP: tests, run, README section, then stop for review.

## Pretrained models and their cutoffs (rule 1b)

| Model (version used) | Role | Public release (cutoff) | Training data | Primary test dates |
|---|---|---|---|---|
| TF-IDF (fit per fold) | STEP 18 tier 1 | none | our training fold only | all |
| `ProsusAI/finbert` (Araci 2019, arXiv 1908.10063) | STEP 17 8-K tone; STEP 18 tier 3 base | **2019-08-27** | BERT-base (Wikipedia + BookCorpus, 2018), Reuters TRC2 2008-10, Financial PhraseBank | **>= 2019-09-01** |
| `sentence-transformers/bert-base-nli-mean-tokens` (Reimers & Gurevych 2019, arXiv 1908.10084) | STEP 18 tier 2 | **2019-08-27** | BERT-base (2018) fine-tuned on SNLI (2015) and MultiNLI (2017) | **>= 2019-09-01** |
| *Not used:* `sentence-transformers/all-MiniLM-L6-v2` | -- | 2021-08 | 1.17B sentence pairs (Reddit, S2ORC, StackExchange, ...) | -- |
| *Not used:* `yiyanghkust/finbert` (Yang, Uy & Huang 2020) | -- | 2020-06-15 | **10-K/10-Q text and earnings calls**: the documents we test on | -- |

**Why the 2019 SBERT model for tier 2, not MiniLM.**
- MiniLM's 2021-08 cutoff would leave about 5 primary test years. The ">= 4 OOS years with IC > 0" rule would then require 80% positive years, a bar a real but small effect would often miss.
- With the 2019-08 cutoff, there are about 7 test years (2019-09 -> 2026-06 labels), so the same rule is about 57%.
- Statistical power matters more here than embedding quality. A weaker embedding that can be judged fairly is worth more than a better one that can't.
- The model card marks this model **deprecated** ("produces sentence embeddings of low quality"). That is accepted knowingly: tier 2 may fail because of it, and a failure is recorded as a failure, not re-run with a newer model. Its training data (SNLI + MultiNLI) comes from the paper; the card itself does not list it.
- Both pretrained models now share the 2019-08-27 cutoff, so **every Phase 3 primary test starts on 2019-09-01**. TF-IDF-only results before that date are reported separately.

## Data

- **Universe.** The STEP 14a point-in-time construction (13F full books, US common stock, $1M 20-day dollar volume, price > $3), with the cap band widened from $200M-$2B to **$200M-$10B**.
  - STEP 14a looked CUSIPs up only around the small-cap band, so $2B-$10B coverage is partial.
  - STEP 17 first reports coverage by year and cap bucket. If any year has < 60% of the $2B-$10B band priced, the band falls back to $200M-$2B for all of Phase 3, and the README says so.
- **8-K text.** All 8-Ks of universe members since 2013:
  - EDGAR primary document plus EX-99.1;
  - item codes from the filing header;
  - main-item text with the `fetch_edgar.html_to_text` / `process_documents` section heuristics.
  - Cached per accession in `data/raw/phase3/8k/`, as parquet grouped per CIK.
  - SEC fair access: <= 8 requests/s, shared limiter, project User-Agent, resumable.
- **10-K/10-Q Risk Factors and MD&A.** The Phase 1b extractor (`src/edgar_sections.py`) and its cache, extended to the small/mid band.
- **News.** `news_provider.py` returns only the ~10 most recent yfinance headlines: there is **no point-in-time historical news**. News is therefore **excluded from every backtest** (STEP 18 has no news documents) and used live only (STEP 20).

## STEP 17: 8-K event features + rules-only candidate funnel (descriptive baseline, no verdict)

- **8-K features**, from the acceptance datetime:
  - one-hot item codes (1.01, 1.02, 2.01, 2.02, 2.03, 3.01, 4.02, 5.02, 7.01, 8.01; other items grouped);
  - FinBERT tone of the main item text: (P(positive) - P(negative)), mean over 512-token chunks of the first 4,096 tokens.
  - Tone counts only from 2019-09 (rule 1b).
- **Funnel** (`src/phase3/funnel.py`, rules only). Each Friday close, a member is a candidate if it has at least one trigger in the 7 calendar days ending that Friday:
  - `8k`: any 8-K except those with only items 9.01 and/or 5.07;
  - `sue`: a 10-Q/10-K whose SUE (latest period in the filing, the Phase 2 fix) is >= the 90th percentile of the universe's previous 365 days;
  - `insider`: >= 1 officer/director open-market purchase (code P), Form 4 filed in the window;
  - `volume`: 5-day average volume >= 2.5x the previous 60-day average.

  All triggers that fired are recorded.
- **These thresholds were chosen in this document before any Phase 3 data was examined**: no 8-K text, no funnel candidate list, no candidate returns. They will not be tuned. (The `sue` trigger reuses the H2 definition, whose Phase 2 results are known; see the caveat above.)
- **Funnel backtest.** Each week, all candidates are entered equal-weight at Monday's open and held 63 trading days, as overlapping weekly cohorts, at 0.30%/side, vs IWM and SPY over the full PIT history.
  - Also reported: the share of candidates with 63-day excess > +10% / > +30%, vs the universe base rate on the same dates; and results by trigger.
  - **No verdict.** This is the baseline later layers must beat.

## STEP 18: our own text model (walk-forward)

- **Documents.**
  - 8-K main-item text;
  - 10-Q/10-K Risk Factors and MD&A sections;
  - **no news** (see Data);
  - each dated by acceptance datetime.
- **Labels**, entry at the next open after acceptance:
  - regression: 63-trading-day excess return vs IWM (or vs SPY for top-500 names);
  - **classification, the primary target**: 63-day excess > +10%, measured **at the end of the 63 trading days**. This matches a fixed-hold trade. "At any point within 63 days" would be easier to hit and harder to trade, so it is not used.
- **Tiers**, each kept only if it beats the previous one:
  1. TF-IDF (uni+bigrams, vocabulary fit per fold) + logistic regression (classification) / ridge (regression);
  2. frozen `bert-base-nli-mean-tokens` embeddings (mean of 512-token chunk embeddings) + LightGBM (STEP 12 conservative parameters);
  3. `ProsusAI/finbert` fine-tuned per fold: 2 epochs, lr 2e-5, max 512 tokens, early stopping on the inner validation split.
     - If CPU fine-tuning of one fold takes > 6 hours, tier 3 uses 8-Ks + Risk Factors only, and the subset is reported.
- **Walk-forward.** Expanding window by acceptance date; test blocks of 6 months; embargo 63 trading days (= label length).
  - First test block: 2016-01. Tier 1 results before 2019-09 are reported separately.
  - Every tier is retrained in every fold; hyperparameters are set on the last 20% of each training window.
  - A test checks that no test-fold document, and no document whose label window overlaps the test block, is in training.
- **Aggregation.** Per (week, ticker): the maximum score among that ticker's documents accepted in the week.
- **Evaluation**, per document and per (week, ticker):
  - weekly rank IC with NW t (lags 13 = 63 days / 5);
  - top-decile hit rate vs base rate;
  - calibration table of P(> +10%) by decile;
  - top-decile portfolio after costs vs IWM/SPY and vs random funnel picks.
- **PROMOTED** only if all of these hold on the primary dates (>= 2019-09-01):
  1. vs the previous tier **and** vs momentum_12_1, the NW t of the per-week IC difference is > 2;
  2. weekly IC > 0 in >= 4 OOS calendar years;
  3. the top-decile portfolio beats the benchmark after costs.
  - Tier 1 is compared with random ordering and momentum_12_1.
  - **Multiple testing**: 3 tiers x 1 primary target. The regression target is reported, not judged.

## STEP 19: event-conditional ranker on funnel candidates (walk-forward)

- **Model.** LightGBM on per-date ranks (STEP 12 parameters), trained only on historical funnel candidates.
  - Features: `model.FEATURES` whitelist + trigger flags + STEP 17 8-K features + the STEP 18 score (only if promoted, out-of-fold only). **No STEP 16 features.**
  - **Primary target**: P(63-day excess > +10%), end of horizon. The regression target is reported.
  - Walk-forward as STEP 12; embargo 63 trading days.
  - Primary dates >= 2019-09-01 (rule 1b: FinBERT tone, and the STEP 18 score if used).
- **PROMOTED** only if all hold:
  1. the NW t of the per-week IC difference is > 2 vs a random ordering of the funnel **and** vs momentum_12_1;
  2. IC > 0 in >= 4 OOS years;
  3. its top-K portfolio (K = 10/week, 63-day hold, overlapping cohorts, 0.30%/side) beats the STEP 17 funnel baseline after costs.

## STEP 20: LLM research agent (live only, never backtested)

- `src/phase3/agent.py` reuses the provider configuration, cache and cost guards of `signal_extraction.py`.
- Input: only information timestamped before the run.
- Output: Pydantic-validated, with up to 2 retries on invalid output; an invalid result after retries is logged as `invalid` and the candidate is skipped. Fields:
  - decision {buy, pass};
  - thesis (<= 120 words);
  - catalyst from a fixed taxonomy {earnings, guidance, M&A, contract/customer, product/regulatory, management, capital return, restructuring, insider/ownership, other};
  - horizon in trading days;
  - conviction 1-5;
  - invalidation price;
  - key risks;
  - source documents.
- Every call records model, prompt version (SHA-256 of the prompt file), input hash and cost. Results are reported per prompt version.
- Inputs used by the colleague's agent: **open TODO**. Added before the forward start date, or afterwards as a new prompt version, reported separately.
- Scope: top 15 candidates/week (ranker order if STEP 19 is promoted, otherwise funnel order). `--dry-run` prints the estimated weekly cost and makes zero LLM calls.

## STEP 21: append-only ledger + weekly runner

- **Ledger**: `src/phase3/ledger.py`, hash-chained JSONL in `data/ledger/`; `verify_ledger()`; never rewritten or deleted; corrections are new records referencing the original.
- **Arms**, every week, on the same funnel:
  - `random`: K = 10, seed = ISO year*100 + ISO week, stored;
  - `ml_top`: only if STEP 19 is promoted; otherwise skipped and logged;
  - `agent`: buys **and** passes;
  - `manual`: CLI entries, timestamped.
- **Entry and exits**: entry at the next trading day's open, filled by a later job, never the signal-day close. Exits are fixed at entry: 63 trading days, or the invalidation price as a stop (2.5 ATR(14) for arms without one).
- **Runner**: `src/run_phase3_weekly.py`, idempotent per ISO week.

## STEP 22: forward evaluation, pre-registered forward verdict

**FORWARD TEST START: ____-__-__** (the date of the first live weekly run, written into this file and the README in the commit made just before that run).

An arm is **TRADEABLE** only if, after >= 26 weeks **and** >= 60 closed picks:
1. its mean 63-day excess return vs IWM after costs (0.30%/side) has a 95% bootstrap CI (resampled by entry week, 10,000 draws) entirely > 0;
2. its mean excess is >= the 95th percentile of the random arm's distribution. That distribution is the mean excess of 1,000 random K-subsets of the same weeks' funnel candidates, drawn from the stored candidate lists;
3. its mean excess stays > 0 after excluding its 3 best picks.

Otherwise **NOT PROVEN**. The same rule applies to the agent, ml_top and manual arms, each judged on its own.
- The agent arm is judged on its **buys**; its passes are reported as a selection check (buys minus passes).
- Results by prompt version, conviction and catalyst are reported, not judged.
- A post-verdict calibration model on the agent's outputs is allowed only after >= 200 closed agent picks, walk-forward on the ledger, under STEP 18's promotion rule.

## STEP 23: app

One Streamlit page, "Phase 3 -- Agent picks", showing:
- this week's agent picks with thesis, catalyst, invalidation price and sources;
- arm-by-arm forward results;
- ledger-integrity status.

It carries the Quick Picks "No validated strategy" warning until an arm is TRADEABLE. No other page changes beyond a link.

## End of phase

A "PHASE 3 VERDICT" table in the README, in the Phase 1 format, covering every STEP 17-22 test, its result, annualized after costs, the benchmark over the same period, and its verdict.
