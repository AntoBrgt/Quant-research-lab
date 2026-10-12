# PHASE 2 pre-registration: can the app's BUY list be trusted?

Written 2026-09-29, committed **before** any Phase 2 code touched data.
Nothing below may change after the first run. If a rule turns out to be
wrong, the fix is a new, dated pre-registration (Phase 3), and this one is
reported as it stands.

## Why

Phase 1 (README, PHASE 1 VERDICT) showed that nothing the app ranks or trades
beats holding SPY after costs. The app therefore cannot show BUY/SELL orders
anyone should follow. Phase 2 tests three narrow, event-driven rules built on
the only leads Phase 1 found. A BUY in the app will only ever come from a rule
that passes this test **and** paper trading afterwards.

## Honest caveat about the holdout

Phase 1 already looked at insider buying and SUE over 2016-2026 in *ranking*
form, and H1/H2 were chosen partly because of those results. The holdout below
is therefore not pristine. Mitigations: (1) a rule must pass in **both** the
development period and the holdout, (2) the significance bar is raised for three
tests, (3) a pass only unlocks paper trading, which is the first truly fresh
test.

## Periods

| Period | Signal dates | Rule |
|---|---|---|
| Development | 2013-09-01 -> 2024-08-31 | A trade counts only if it **exits** by 2024-08-31 |
| Holdout | 2024-09-01 -> data end | Run once. `run_phase2.py --holdout` writes `holdout.lock` (commit, time); a second run refuses without `--force`, and a forced run is flagged in its own report |

## Common rules (all hypotheses)

- **Signal date** = the day the information became public: the SEC **filing
  date** (Form 4, 10-Q/10-K), or a month-end for H3.
- **Entry** at the **next trading day's open** after the signal date.
- **Costs**: 0.30% per side for small caps (H1, H2); 0.15% per side for the
  top-500 universe (H3). Same as Phase 1.
- **Universe membership is point-in-time**: a ticker is eligible on date d only
  if it is a member at the latest month-end <= d of the Phase 1 universe:
  STEP 14a small-cap members (`in_universe`) for H1/H2, STEP 15 candidates for H3.
- **Excess return per trade** = the trade's net return minus the benchmark's
  return from the same entry open to the same exit.
- **Portfolio**: at most 20 open positions, each opened with 1/20 of equity at its
  entry; signals on the same day are taken in order of signal strength (H1: buyers,
  then value; H2: SUE; H3: drawdown), ties by ticker; signals when full are skipped.
  Idle cash earns 0%. Compared with the benchmark held at the portfolio's average
  exposure.
- **Random baseline**: 200 seeds. Each real trade is replaced by a random ticker
  from the same month's eligible universe (H1/H2: STEP 14a members; H3: STEP 15
  top-500 members), same entry date, same exit rules.

## Hypotheses

| # | Rule | Exit | Benchmark |
|---|---|---|---|
| **H1** insider cluster | On Form 4 filing date D, the number of distinct officer/director open-market buyers (code P) with filing dates in (D-30 days, D] reaches **3 or more**, and the issuer had no H1 signal in the previous 90 days | 63 trading days, or a stop at signal close - 2.5 x ATR(14) (gap fills at the open; stop wins within a bar) | IWM |
| **H2** earnings drift | A 10-Q/10-K filing whose SUE (STEP 14a definition, `sec_xbrl.sue_series`) is **>= the 90th percentile of all SUE values filed by universe members in the 365 days before D**, with at least 100 such values | 60 trading days, no stop | IWM |
| **H3** cheap quality + trend | A STEP 15 candidate month (same screen) whose adjusted close is **above its 200-day moving average** at the month-end | +30% target, thesis break, or 252 trading days, as `value_survival.trade_outcome` | SPY |

## Pass rule (fixed)

A hypothesis **PASSES** only if **all** hold, in **both** periods:

1. Mean excess return per trade > 0.
2. Newey-West t-stat of the monthly mean excess (by entry month, lags = ceil(hold / 21)) **>= 2.4** (Bonferroni for 3 tests at 5%).
3. The 95% bootstrap CI of the mean excess (resampling entry months) is above 0.
4. Mean excess beats **>= 95%** of the random baselines.
5. Portfolio CAGR > the benchmark's CAGR at the same average exposure.

Anything else is **NOT PROVEN**. No parameter (windows, thresholds, stops,
holds) is tuned; the development run exists to catch bugs and data problems,
not to choose settings.

## What the app does with the result

- **No pass**: the Quick Picks page stops showing BUY/SELL orders and becomes a
  watchlist labelled "no validated strategy".
- **Pass**: only the passing rule's live signals appear as BUY candidates, each
  showing the rule's tested numbers, and they are marked "paper trading" until
  3 months of paper results track the backtest.
