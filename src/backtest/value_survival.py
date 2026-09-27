"""STEP 15 -- "buy cheap quality, sell at target", with survival models.

Idea under test: large/mid caps that fell hard (>= 25% below their 52-week
high), are cheap versus their OWN history, and are still profitable and
cash-generative, tend to recover -- and a survival model can tell which will
reach the target sooner. Holding periods are long and uneven, so the natural
label is *time to target*, with every position that hasn't hit (yet) treated
as censored, never as a failure.

Point-in-time everywhere:
- universe: `dataset.pit_broad_universe` (top 500 by 13F value, SEC filing dates);
- fundamentals: SEC XBRL, usable from each value's FILING date (`sec_xbrl.py`);
- prices: entry at the next day's open; targets on adjusted highs;
- thesis break: known on the filing date of the 2nd bad quarter, exit at the next open;
- training labels are censored at the training cutoff (an open position is
  "not yet", not "failed").

PRE-REGISTERED (fixed before the first run):
- primary model for the portfolio ranking = XGBoost survival:aft (`PRIMARY_MODEL`);
  Cox PH and Kaplan-Meier are reported alongside, never used to pick the verdict;
- primary target = +30% (`PRIMARY_TARGET`); +20% / +50% reported;
- verdict rule = `portfolio_verdict`.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

import sec_xbrl
from backtest import barriers, pit_features

logger = logging.getLogger(__name__)

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz

DRAWDOWN_MIN = 0.25
VALUATION_HISTORY_MONTHS = 60
VALUATION_MIN_HISTORY = 36  # fewer P/S points than this -> price-based only (flagged)
NET_DEBT_TO_MCAP_MAX = 1.0
TARGETS = (0.20, 0.30, 0.50)
PRIMARY_TARGET = 0.30
PRIMARY_MODEL = "xgb_aft"
CAP_TRADING_DAYS = 2520  # 10 years
TRADING_DAYS_PER_MONTH = 21
COST_PER_SIDE = 0.0015  # 0.10% + 0.05% slippage
BUYS_PER_MONTH = 3
MAX_OPEN = 15
EMBARGO_MONTHS = 12
MIN_TRAIN_YEARS = 3
HORIZONS_MONTHS = (6, 12, 24)
SECTORS = ["manufacturing", "services", "finance", "retail", "transport_utilities", "mining", "wholesale",
           "construction", "agriculture", "public_admin", "unknown"]


# ----------------------------------------------------------------------------
# Point-in-time fundamentals on a monthly grid
# ----------------------------------------------------------------------------


def monthly_fundamentals(table: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """`sec_xbrl.snapshot` at every filing date, carried forward to each monthly date (as-of join).

    Equivalent to calling snapshot(table, d) for each d -- a snapshot only
    changes on a filing date -- but ~20x cheaper.
    """
    if table.empty:
        return pd.DataFrame(index=pd.DatetimeIndex(dates))
    filed = sorted(pd.to_datetime(table["filed"].unique()))
    states = pd.DataFrame([{**sec_xbrl.snapshot(table, f), "state_date": f} for f in filed])
    states = states.drop(columns=["last_filed"]).sort_values("state_date")
    grid = pd.DataFrame({"date": pd.DatetimeIndex(sorted(dates))})
    merged = pd.merge_asof(grid, states, left_on="date", right_on="state_date", direction="backward")
    return merged.set_index("date").drop(columns=["state_date"])


def valuation_percentile(ps: pd.Series, date: pd.Timestamp) -> tuple[Optional[float], Optional[float], int]:
    """(percentile of today's P/S within the prior 60 monthly points, their median, count) -- strictly before `date`."""
    history = ps[(ps.index < date) & (ps.index >= date - pd.DateOffset(months=VALUATION_HISTORY_MONTHS))].dropna()
    today = ps.get(date)
    if today is None or pd.isna(today) or history.empty:
        return None, None, len(history)
    return float((history < today).mean()), float(history.median()), len(history)


# ----------------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------------


def build_candidates(
    membership: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    xbrl_tables: dict[str, pd.DataFrame],
    raw_closes: dict[str, pd.Series],
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Every (month, member) with its screen inputs; `is_candidate` marks the ones that pass.

    Screen: drawdown from the 52-week high >= 25% AND P/S below its own
    5-year median (or, with < 36 months of P/S history, price-based only,
    flagged) AND net margin > 0, FCF margin > 0, net debt / market cap < 1.
    Missing quality data = excluded.
    """
    dates = sorted(pd.to_datetime(membership["date"].unique()))
    rows = []
    for ticker, members in membership.dropna(subset=["ticker"]).groupby("ticker"):
        if ticker not in prices:
            continue
        px = prices[ticker].sort_index()
        member_dates = sorted(pd.to_datetime(members["date"].unique()))
        table = xbrl_tables.get(ticker)
        fund = monthly_fundamentals(table, dates) if table is not None else pd.DataFrame(index=pd.DatetimeIndex(dates))
        raw = raw_closes[ticker]
        high52 = px["high"].rolling(252, min_periods=252).max()
        # market cap and P/S on the whole monthly grid (valuation history needs non-member months too)
        raw_on = raw.reindex(pd.DatetimeIndex(dates), method="ffill", tolerance=pd.Timedelta(days=7))
        mcap = raw_on * fund["shares"] if "shares" in fund else pd.Series(np.nan, index=fund.index)
        ps = mcap / fund["revenue_ttm"] if "revenue_ttm" in fund else pd.Series(np.nan, index=fund.index)
        ps = ps.where(ps > 0)
        info = members.set_index(pd.to_datetime(members["date"]))
        for date in member_dates:
            tech = pit_features.pit_technicals(px, date, benchmark)
            if not tech:
                continue  # no price within MAX_STALE_DAYS of the date
            h = high52.asof(date) if len(high52) else np.nan
            drawdown = 1 - tech["adj_close"] / h if h and h > 0 else np.nan
            f = fund.loc[date] if date in fund.index else pd.Series(dtype=float)
            get = lambda k: (None if k not in f or pd.isna(f[k]) else float(f[k]))  # noqa: E731
            rev, ni, fcf = get("revenue_ttm"), get("net_income_ttm"), get("fcf_ttm")
            cap = None if pd.isna(mcap.get(date, np.nan)) else float(mcap[date])
            net_debt = get("net_debt")
            pct, median, n_hist = valuation_percentile(ps, date)
            row = {
                "date": date, "ticker": ticker,
                **{k: v for k, v in tech.items() if k != "adj_close"},
                "close": tech["adj_close"], "drawdown": drawdown,
                "market_cap": cap, "log_market_cap": math.log(cap) if cap and cap > 0 else None,
                "revenue_ttm": rev, "net_margin": ni / rev if rev and ni is not None and rev > 0 else None,
                "fcf_margin": fcf / rev if rev and fcf is not None and rev > 0 else None,
                "net_debt": net_debt, "net_debt_to_mcap": net_debt / cap if net_debt is not None and cap else None,
                "fcf_yield": fcf / cap if fcf is not None and cap else None,
                "roe": ni / get("equity") if ni is not None and get("equity") and get("equity") > 0 else None,
                "revenue_growth": rev / get("revenue_ttm_prior") - 1 if rev and get("revenue_ttm_prior") else None,
                "ps": ps.get(date), "valuation_pct": pct, "valuation_median": median, "valuation_history": n_hist,
                "institution_count": info.loc[date, "institution_count"] if date in info.index else None,
                "institutional_direction_score": info.loc[date, "institutional_direction_score"] if date in info.index else None,
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["price_only"] = frame["valuation_history"] < VALUATION_MIN_HISTORY
    cheap = frame["price_only"] | (frame["ps"] < frame["valuation_median"])
    quality = (frame["net_margin"] > 0) & (frame["fcf_margin"] > 0) & (frame["net_debt_to_mcap"] < NET_DEBT_TO_MCAP_MAX)
    frame["is_candidate"] = (frame["drawdown"] >= DRAWDOWN_MIN) & cheap & quality.fillna(False)
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Outcomes: target / thesis break / cap / data end
# ----------------------------------------------------------------------------


def trade_outcome(
    ohlc: pd.DataFrame, signal_date: pd.Timestamp, targets=TARGETS, break_dates=(), cap: int = CAP_TRADING_DAYS,
    cost: float = COST_PER_SIDE,
) -> dict:
    """Entry at the next open after `signal_date`; for each target, the first day's high >= entry x (1 + target).

    Thesis break: first break filing date AFTER the signal date -> exit at the
    next trading day's open (the filing may come after the close). Cap: close of
    day `cap`. Data end: prices stop (delisted) or the data ends -- the position is
    censored there, marked at the last close.
    Returns per target: event (1 = target first), duration (trading days from entry),
    exit reason, exit price, net return; plus entry info.
    """
    idx = ohlc.index
    after = idx[idx > signal_date]
    out = {"entry_date": None}
    if len(after) == 0 or pd.isna(ohlc.loc[after[0], "open"]):
        return out
    entry_pos = idx.get_loc(after[0])
    entry = float(ohlc["open"].iloc[entry_pos])
    window = ohlc.iloc[entry_pos: entry_pos + cap]
    closes = window["close"]
    last_valid = closes.last_valid_index()
    if last_valid is None:
        return {"entry_date": None}
    # Last day with a price. A short halt (NaN days followed by prices) is not an end;
    # a ticker whose prices stop (delisted) or the end of the data is.
    end_pos = window.index.get_loc(last_valid)

    brk = [pd.Timestamp(b) for b in break_dates if pd.Timestamp(b) >= after[0]]
    brk_pos = None
    if brk:
        later = window.index[window.index > brk[0]]
        if len(later):
            brk_pos = window.index.get_loc(later[0])  # exit at that day's open
    out.update({"entry_date": after[0], "entry_price": entry})
    highs = window["high"].to_numpy(dtype=float)
    for target in targets:
        level = entry * (1 + target)
        hit = np.where(highs[: end_pos + 1] >= level)[0]
        hit_pos = int(hit[0]) if len(hit) else None
        key = f"t{int(round(target * 100))}"
        if hit_pos is not None and (brk_pos is None or hit_pos < brk_pos):
            reason, pos, price, event = "target", hit_pos, level, 1
        elif brk_pos is not None and brk_pos <= end_pos:
            reason, pos, price, event = "thesis_break", brk_pos, float(window["open"].iloc[brk_pos]), 0
        elif end_pos >= cap - 1:
            reason, pos, price, event = "cap", cap - 1, float(closes.iloc[cap - 1]), 0
        else:
            reason, pos, price, event = "data_end", end_pos, float(closes.iloc[end_pos]), 0
        out[f"{key}_event"] = event
        out[f"{key}_duration"] = pos + 1
        out[f"{key}_reason"] = reason
        out[f"{key}_exit_date"] = window.index[pos]
        out[f"{key}_exit_price"] = price
        out[f"{key}_return"] = price * (1 - cost) / (entry * (1 + cost)) - 1
    out["last_price_date"] = last_valid
    return out


def censor_at(frame: pd.DataFrame, key: str, cutoff: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Labels as they were KNOWN at `cutoff`: any exit after it becomes 'still open' (event 0, duration to cutoff).

    This is what keeps training honest: a position opened in 2016 that hit its
    target in 2019 is, for a model trained at the start of 2018, an open position
    -- censored, never a failure.
    """
    f = frame.copy()
    exits = pd.to_datetime(f[f"{key}_exit_date"])
    late = exits > cutoff
    entry_pos = calendar.searchsorted(pd.to_datetime(f["entry_date"]))
    cutoff_pos = calendar.searchsorted(cutoff, side="right") - 1
    f.loc[late, f"{key}_event"] = 0
    f.loc[late, f"{key}_duration"] = np.maximum(cutoff_pos - entry_pos[late.to_numpy()] + 1, 1)
    f.loc[late, f"{key}_reason"] = "open_at_cutoff"
    return f[pd.to_datetime(f["entry_date"]) <= cutoff]


# ----------------------------------------------------------------------------
# Survival models
# ----------------------------------------------------------------------------


def survival_design(frame: pd.DataFrame, rank_features: list[str]) -> pd.DataFrame:
    """Per-date ranks of the whitelist + drawdown + valuation percentile + sector dummies. NaN rank -> 0.5."""
    from backtest import model as ranking_model

    x = ranking_model.cross_sectional_rank(frame[["date"] + rank_features].apply(
        lambda c: pd.to_numeric(c, errors="coerce") if c.name != "date" else c), rank_features).fillna(0.5)
    x["drawdown_depth"] = pd.to_numeric(frame["drawdown"], errors="coerce").fillna(0.25)
    x["valuation_pct"] = pd.to_numeric(frame["valuation_pct"], errors="coerce").fillna(0.5)
    x["price_only"] = frame["price_only"].astype(float)
    for sector in SECTORS[:-1]:  # "unknown" is the reference level
        x[f"sector_{sector}"] = (frame["sector"] == sector).astype(float)
    return x


def fit_predict(model: str, x_train, duration, event, x_test) -> dict:
    """Fit one survival model; return a function S(t) per test row (t in trading days) and a risk score.

    kaplan_meier: population curve, same for every row (the no-skill baseline).
    cox: lifelines CoxPHFitter, L2 penalizer 0.1.
    xgb_aft: XGBoost survival:aft (log-normal), fixed conservative params.
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    if model == "kaplan_meier":
        from lifelines import KaplanMeierFitter

        km = KaplanMeierFitter().fit(duration, event)
        sf = km.survival_function_["KM_estimate"]
        def surv(t):
            return np.tile(float(sf.asof(t) if t >= sf.index[0] else 1.0), len(x_test))
        return {"surv": surv, "risk": np.zeros(len(x_test))}
    if model == "cox":
        from lifelines import CoxPHFitter

        data = x_train.copy()
        keep = [c for c in data.columns if data[c].std() > 0]
        data = data[keep].assign(_d=duration, _e=event)
        cph = CoxPHFitter(penalizer=0.1).fit(data, "_d", "_e")
        sf = cph.predict_survival_function(x_test[keep])
        def surv(t):
            ix = sf.index.searchsorted(t, side="right") - 1
            return np.ones(len(x_test)) if ix < 0 else sf.iloc[ix].to_numpy()
        return {"surv": surv, "risk": cph.predict_partial_hazard(x_test[keep]).to_numpy()}
    if model == "xgb_aft":
        import xgboost as xgb
        from scipy.stats import norm

        dtrain = xgb.DMatrix(x_train.to_numpy(dtype=float))
        dtrain.set_float_info("label_lower_bound", duration)
        dtrain.set_float_info("label_upper_bound", np.where(event == 1, duration, np.inf))
        params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
                  "aft_loss_distribution_scale": 1.0, "eta": 0.03, "max_depth": 3, "min_child_weight": 20,
                  "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0, "seed": 0, "nthread": 1}
        booster = xgb.train(params, dtrain, num_boost_round=300)
        mu = np.log(np.maximum(booster.predict(xgb.DMatrix(x_test.to_numpy(dtype=float))), 1e-6))
        def surv(t):
            return 1 - norm.cdf((math.log(max(t, 1e-6)) - mu) / params["aft_loss_distribution_scale"])
        return {"surv": surv, "risk": -mu}
    raise ValueError(model)


def expected_annualized_return(surv: Callable, target: float, fail_return: float, cap: int = CAP_TRADING_DAYS) -> dict:
    """P(hit in 6/12/24 months), median time, expected return and its annualized rate, per test row.

    E[R] = P(hit by cap) x target + (1 - P(hit by cap)) x fail_return, where
    fail_return = mean net return of TRAINING positions that ended without the
    target (thesis breaks and caps -- the competing risk is kept, not dropped).
    E[T] = restricted mean time to target within the cap, integral of S(t).
    Annualized = (1 + E[R]) ** (1 / E[T in years]) - 1. A ranking heuristic, not a forecast.
    """
    grid = np.arange(0, cap + 1, TRADING_DAYS_PER_MONTH)
    curves = np.vstack([surv(t) for t in grid])  # (len(grid), n)
    p_cap = 1 - curves[-1]
    expected_t = _trapezoid(curves, grid, axis=0)
    expected_r = p_cap * target + (1 - p_cap) * fail_return
    years = np.maximum(expected_t / 252, 1 / 12)
    out = {f"p_hit_{m}m": 1 - surv(m * TRADING_DAYS_PER_MONTH) for m in HORIZONS_MONTHS}
    below_half = curves <= 0.5
    median_idx = np.where(below_half.any(axis=0), below_half.argmax(axis=0), -1)
    out["median_months"] = np.where(median_idx >= 0, grid[np.maximum(median_idx, 0)] / TRADING_DAYS_PER_MONTH, np.inf)
    out["expected_return"] = expected_r
    out["expected_annualized_return"] = np.sign(1 + expected_r) * np.abs(1 + expected_r) ** (1 / years) - 1
    return out


def concordance(duration, event, predicted_time) -> float:
    from lifelines.utils import concordance_index

    return float(concordance_index(duration, predicted_time, event))


def integrated_brier(duration, event, surv: Callable, grid, train_duration, train_event) -> float:
    """IPCW integrated Brier score (Graf et al.) over `grid`, censoring weights from the TRAINING data."""
    from lifelines import KaplanMeierFitter

    duration, event = np.asarray(duration, float), np.asarray(event, int)
    cens = KaplanMeierFitter().fit(np.asarray(train_duration, float), 1 - np.asarray(train_event, int))
    g = lambda t: max(float(cens.survival_function_.iloc[:, 0].asof(t)) if t >= cens.survival_function_.index[0] else 1.0, 1e-3)  # noqa: E731
    scores = []
    for t in grid:
        s = surv(t)
        died = (duration <= t) & (event == 1)
        alive = duration > t
        w_died = np.array([1 / g(d) for d in duration]) * died
        scores.append(np.mean(w_died * s ** 2 + alive * (1 - s) ** 2 / g(t)))
    return float(_trapezoid(scores, grid) / (grid[-1] - grid[0]))


def calibration_12m(duration, event, p_hit_12m) -> pd.DataFrame:
    """Deciles of predicted P(hit <= 12m) vs actual, on rows whose 12-month outcome is known."""
    h = 12 * TRADING_DAYS_PER_MONTH
    duration, event = np.asarray(duration, float), np.asarray(event, int)
    known = (event == 1) | (duration >= h)
    frame = pd.DataFrame({"p": np.asarray(p_hit_12m)[known], "hit": ((event == 1) & (duration <= h))[known].astype(int)})
    if len(frame) < 20:
        return pd.DataFrame()
    frame["decile"] = pd.qcut(frame["p"].rank(method="first"), 10, labels=False) + 1
    table = frame.groupby("decile").agg(rows=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean"))
    table.index.name = "decile (10 = highest P)"
    return table


# ----------------------------------------------------------------------------
# Portfolio
# ----------------------------------------------------------------------------


def simulate_portfolio(
    picks_by_month: dict[pd.Timestamp, list[int]],
    outcomes: pd.DataFrame,
    key: str,
    closes: dict[str, np.ndarray],
    calendar: pd.DatetimeIndex,
    cash_returns: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_open: int = MAX_OPEN,
    cost: float = COST_PER_SIDE,
) -> dict:
    """Monthly buys (in rank order), equal size = equity / max_open, exits from `outcomes`, cash earns BIL.

    A position whose exit is after `end` (or censored at data end) stays open and
    is marked to market at the close; nothing is ever closed with future information.
    """
    pos_of = {d: i for i, d in enumerate(calendar)}
    entries: dict[int, list[int]] = {}
    for month, idxs in picks_by_month.items():
        for i in idxs:
            ed = outcomes.at[i, "entry_date"]
            if ed is not None and not pd.isna(ed) and ed in pos_of:
                entries.setdefault(pos_of[ed], []).append(i)
    s, e = calendar.searchsorted(start), calendar.searchsorted(end, side="right") - 1
    cash, equity_prev = 1.0, 1.0
    open_pos, equity, trades = [], [], []
    cash_ret = cash_returns.reindex(calendar).fillna(0.0).to_numpy()
    for t in range(s, e + 1):
        cash *= 1 + cash_ret[t]
        for i in entries.get(t, []):
            ticker = outcomes.at[i, "ticker"]
            if len(open_pos) >= max_open or any(p["ticker"] == ticker for p in open_pos):
                continue
            value = min(equity_prev / max_open, cash / (1 + cost))
            if value <= 0:
                continue
            entry = float(outcomes.at[i, "entry_price"])
            shares = value / entry
            cash -= value * (1 + cost)
            exit_date = outcomes.at[i, f"{key}_exit_date"]
            reason = outcomes.at[i, f"{key}_reason"]
            exit_pos = pos_of.get(exit_date, None) if reason != "data_end" or exit_date < calendar[-1] else None
            open_pos.append({"i": i, "ticker": ticker, "shares": shares, "exit_pos": exit_pos, "reason": reason,
                             "exit_price": float(outcomes.at[i, f"{key}_exit_price"]), "cost_basis": value * (1 + cost)})
        still = []
        for p in open_pos:
            if p["exit_pos"] is not None and p["exit_pos"] == t:
                proceeds = p["shares"] * p["exit_price"] * (1 - cost)
                cash += proceeds
                trades.append({"i": p["i"], "ticker": p["ticker"], "reason": p["reason"], "exit_date": calendar[t],
                               "return": proceeds / p["cost_basis"] - 1})
            else:
                still.append(p)
        open_pos = still
        marked = cash + sum(p["shares"] * (closes[p["ticker"]][t] if np.isfinite(closes[p["ticker"]][t]) else p["exit_price"])
                            for p in open_pos)
        equity.append(marked)
        equity_prev = marked
    return {"equity": pd.Series(equity, index=calendar[s: e + 1]), "trades": pd.DataFrame(trades), "open_at_end": len(open_pos)}


def perf(equity: pd.Series) -> dict:
    if len(equity) < 2:
        return {}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    daily = equity.pct_change().dropna()
    return {"total_return": total, "annualized": (1 + total) ** (1 / years) - 1,
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
            "sharpe": float(daily.mean() / daily.std() * math.sqrt(252)) if daily.std() > 0 else float("nan")}


def portfolio_verdict(strategy_equity: pd.Series, spy_close: pd.Series, random_annualized: list[float]) -> tuple[str, list[str], pd.DataFrame]:
    """Pre-registered STEP 15 rule: BEATS MARKET only if, after costs,
    annualized return > SPY over the full period AND each half, >= 95% of random
    portfolios, and Sharpe >= SPY's. Otherwise DOES NOT BEAT MARKET.
    """
    start, end = strategy_equity.index[0], strategy_equity.index[-1]
    mid = start + (end - start) / 2
    spy = spy_close.loc[start:end]
    rows = {}
    for label, a, b in (("full", start, end), ("first half", start, mid), ("second half", mid, end)):
        rows[label] = {"strategy_annualized": perf(strategy_equity.loc[a:b]).get("annualized"),
                       "spy_annualized": perf(spy.loc[a:b]).get("annualized")}
    table = pd.DataFrame(rows).T
    s_perf, spy_perf = perf(strategy_equity), perf(spy)
    pct = float((np.asarray([r for r in random_annualized if np.isfinite(r)]) < s_perf["annualized"]).mean() * 100)
    checks = [
        ("annualized > SPY, full period", table.loc["full", "strategy_annualized"] > table.loc["full", "spy_annualized"]),
        ("annualized > SPY, first half", table.loc["first half", "strategy_annualized"] > table.loc["first half", "spy_annualized"]),
        ("annualized > SPY, second half", table.loc["second half", "strategy_annualized"] > table.loc["second half", "spy_annualized"]),
        (f">= 95% of random portfolios ({pct:.0f}%)", pct >= 95),
        (f"Sharpe >= SPY ({s_perf['sharpe']:.2f} vs {spy_perf['sharpe']:.2f})", s_perf["sharpe"] >= spy_perf["sharpe"]),
    ]
    lines = [f"- {'PASS' if ok else 'FAIL'} -- {name}" for name, ok in checks]
    verdict = "BEATS MARKET" if all(ok for _, ok in checks) else "DOES NOT BEAT MARKET"
    return verdict, lines, table
