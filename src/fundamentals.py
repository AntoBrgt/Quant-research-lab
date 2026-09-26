"""Fundamental data access AND analysis. No LLM anywhere in this file.

Three layers, in order:

1. **Provider** (`FundamentalsProvider`/`YFinanceFundamentalsProvider`) -- raw
   data access + a disk cache, same shape as `price_provider.py`/
   `news_provider.py`. This never interprets anything.
2. **Historical series** (`MetricObservation`) -- every metric is built as a
   chronological list of period-by-period observations with full provenance
   (source, period, period_type, whether it's reported or forward/estimated,
   retrieval time), not a single latest scalar. A 2023/2024/2025/2026 revenue
   growth series is more useful than "27%" on its own (brief section 4), and
   provenance stops a TTM figure from silently being compared against an
   annual one (section 6).
3. **Derived output** (`CompanyFundamentals`) -- the flat, "at least these
   fields" schema from the brief's section 10, plus a deterministic
   `classify_trend()` state (IMPROVING/STABLE/DETERIORATING/INSUFFICIENT_DATA)
   per factor. This is a *descriptive classification*, never a recommendation
   or a score (section 9) -- there is no BUY/SELL output anywhere here, and no
   combined "fundamental score."

Honesty rules enforced throughout: a missing metric is `None`, never a
fabricated or default-filled value (section 7); a suspicious value is
flagged in `quality_flags`/`data_quality`, never silently dropped (section 8);
different sectors don't all have the same statement line items, and metrics
that don't apply to a given company are simply `None`, not forced (section 5)
-- this module does not yet build sector-specific metric sets on top of that
(explicitly out of scope for this pass), but `CompanyFundamentals.sources`/
`history` keep the door open for that later without a schema change.
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Protocol

from pydantic import BaseModel, Field

import config

logger = logging.getLogger(__name__)

# Fundamentals move at most quarterly; a research prototype doesn't need to
# re-hit yfinance more than once a day for the same ticker. This caches the
# raw provider payload -- the derived CompanyFundamentals is recomputed from
# it each call (cheap, pure Python), so a `PROMPT_VERSION`-style bump is never
# needed here the way it is for LLM caches.
CACHE_MAX_AGE_HOURS = 20

PeriodType = Literal["annual", "quarterly", "ttm", "forward"]
TrendState = Literal["IMPROVING", "STABLE", "DETERIORATING", "INSUFFICIENT_DATA"]

INCOME_STMT_SOURCE = "yfinance:financials"
CASHFLOW_SOURCE = "yfinance:cashflow"
BALANCE_SHEET_SOURCE = "yfinance:balance_sheet"
INFO_SOURCE = "yfinance:info"


# ---------------------------------------------------------------------------
# Provider (raw data access + disk cache) -- unchanged in spirit from before
# ---------------------------------------------------------------------------

class FundamentalsProvider(Protocol):
    def get_raw_fundamentals(self, ticker: str) -> dict:
        """Return the provider's raw payload (implementation-specific shape)."""
        ...


def _cache_path(ticker: str) -> Path:
    return config.FUNDAMENTALS_DIR / f"{ticker.upper()}.json"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_MAX_AGE_HOURS


def _read_cache(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str)
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def pd_isna(value) -> bool:
    try:
        import math as _math

        return value is None or (isinstance(value, float) and _math.isnan(value))
    except Exception:
        return value is None


def _safe_to_dict(frame) -> dict:
    """yfinance statement frames are columns=period timestamps, index=line items.

    Converted to {line_item: {period_iso: value}} so it's JSON-cacheable and
    doesn't require pandas to read back.
    """
    if frame is None or getattr(frame, "empty", True):
        return {}
    try:
        return {
            str(idx): {
                str(col.date() if hasattr(col, "date") else col): (None if pd_isna(v) else float(v))
                for col, v in row.items()
            }
            for idx, row in frame.iterrows()
        }
    except Exception:
        return {}


class YFinanceFundamentalsProvider:
    """Free, no-API-key fundamentals provider backed by yfinance, with a disk cache."""

    def get_raw_fundamentals(self, ticker: str) -> dict:
        ticker = ticker.upper()
        path = _cache_path(ticker)

        if _cache_is_fresh(path):
            cached = _read_cache(path)
            if cached is not None:
                return cached

        try:
            import yfinance as yf  # imported lazily so tests never need it installed

            handle = yf.Ticker(ticker)
            info = handle.info or {}
            raw = {
                "info": info,
                "income_stmt": _safe_to_dict(getattr(handle, "financials", None)),
                "cashflow": _safe_to_dict(getattr(handle, "cashflow", None)),
                "balance_sheet": _safe_to_dict(getattr(handle, "balance_sheet", None)),
            }
        except Exception:
            logger.exception("Failed to fetch fundamentals for %s", ticker)
            stale = _read_cache(path)
            if stale is not None:
                logger.warning("Falling back to stale cached fundamentals for %s", ticker)
                return stale
            return {"info": {}, "income_stmt": {}, "cashflow": {}, "balance_sheet": {}}

        _write_cache(path, raw)
        return raw


def last_updated(ticker: str) -> Optional[str]:
    """ISO date the fundamentals cache for this ticker was last written, or None."""
    path = _cache_path(ticker)
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()


# ---------------------------------------------------------------------------
# Historical series with provenance
# ---------------------------------------------------------------------------

class MetricObservation(BaseModel):
    """One value, for one metric, for one reporting period -- with enough
    metadata (section 6) to never be silently mixed with a different period
    type or an estimate.
    """

    metric: str
    value: Optional[float]
    period: str  # e.g. "FY2025", or the raw period_end when not annual
    period_type: PeriodType
    period_end: Optional[str] = None  # ISO date the reporting period covers
    is_estimate: bool = False
    source: str
    retrieved_at: str
    quality_flags: list[str] = Field(default_factory=list)


def _first_present(statement: dict, *row_names: str) -> dict:
    for name in row_names:
        if name in statement:
            return statement[name]
    return {}


def _sorted_periods(raw: dict) -> list[str]:
    """ISO period-end date strings with a non-null value, oldest first."""
    def _key(period: str):
        try:
            return datetime.fromisoformat(period)
        except ValueError:
            return datetime.min

    return sorted((p for p, v in raw.items() if v is not None), key=_key)


def _period_label(period_end: str, period_type: PeriodType) -> str:
    if period_type == "annual":
        try:
            return f"FY{datetime.fromisoformat(period_end).year}"
        except ValueError:
            return period_end
    return period_end


def _raw_ratio(numerator: dict, denominator: dict) -> dict:
    result = {}
    for period, num_value in numerator.items():
        den_value = denominator.get(period)
        if num_value is None or den_value in (None, 0):
            continue
        result[period] = num_value / den_value
    return result


def _raw_sum(a: dict, b: dict) -> dict:
    result = {}
    for period in set(a) | set(b):
        av, bv = a.get(period), b.get(period)
        if av is None or bv is None:
            continue
        result[period] = av + bv
    return result


def _raw_growth(raw: dict) -> dict:
    """Period-over-period growth, keyed by the LATER period's date."""
    periods = _sorted_periods(raw)
    result = {}
    for prev_period, curr_period in zip(periods, periods[1:]):
        prev_value, curr_value = raw[prev_period], raw[curr_period]
        if prev_value in (None, 0) or curr_value is None:
            continue
        result[curr_period] = (curr_value - prev_value) / abs(prev_value)
    return result


# ---------------------------------------------------------------------------
# Data quality validation -- flags, never deletes or invents (section 8)
# ---------------------------------------------------------------------------

# Documented, tunable heuristics -- not statistically fit. A true margin
# (revenue-based: gross/operating/net/FCF margin) above 150% or below -500%
# is treated as "worth a human look". Return/coverage ratios (ROE, ROIC,
# leverage, interest coverage) have a much wider legitimate range -- e.g.
# Apple's actual ROE is routinely >150% because aggressive buybacks shrink
# its equity base, which is real, not an error -- so they get a separate,
# wider bound and are only flagged when clearly implausible (>1000%).
_IMPOSSIBLE_MARGIN_HIGH = 1.5
_IMPOSSIBLE_MARGIN_LOW = -5.0
_IMPOSSIBLE_RATIO_HIGH = 10.0
_IMPOSSIBLE_RATIO_LOW = -10.0
# Interest coverage (EBIT/interest expense) lives on a much larger natural
# scale than ROE/ROIC/leverage -- a well-capitalized large-cap routinely
# covers interest 20-80x over, which is healthy, not suspicious. Only an
# extreme value (near-zero interest expense relative to EBIT, or negative
# EBIT) is worth flagging here.
_IMPLAUSIBLE_COVERAGE_ABS = 200.0
_EXTREME_GROWTH_ABS = 3.0  # +/-300% period-over-period growth
_FUTURE_DATE_TOLERANCE_DAYS = 3
_OUTLIER_MEDIAN_MULTIPLE = 20


def _validate_dates(observations: list[MetricObservation]) -> None:
    now = datetime.now(timezone.utc)
    for obs in observations:
        if not obs.period_end:
            continue
        try:
            period_dt = datetime.fromisoformat(obs.period_end).replace(tzinfo=timezone.utc)
        except ValueError:
            obs.quality_flags.append("invalid_date")
            continue
        if (period_dt - now).days > _FUTURE_DATE_TOLERANCE_DAYS:
            obs.quality_flags.append("future_dated_period")


def _validate_duplicates(observations: list[MetricObservation]) -> None:
    seen: dict[str, MetricObservation] = {}
    for obs in observations:
        if obs.period in seen:
            obs.quality_flags.append("duplicate_period")
            seen[obs.period].quality_flags.append("duplicate_period")
        else:
            seen[obs.period] = obs


def _validate_outliers(observations: list[MetricObservation]) -> None:
    values = [o.value for o in observations if o.value is not None]
    if len(values) < 3:
        return
    magnitudes = sorted(abs(v) for v in values)
    median = magnitudes[len(magnitudes) // 2]
    if median < 1e-9:
        return
    for obs in observations:
        if obs.value is not None and abs(obs.value) > median * _OUTLIER_MEDIAN_MULTIPLE:
            obs.quality_flags.append(f"outlier_vs_series_median:{obs.value:.4g}")


def _validate_margin_bounds(observations: list[MetricObservation]) -> None:
    for obs in observations:
        if obs.value is not None and (obs.value > _IMPOSSIBLE_MARGIN_HIGH or obs.value < _IMPOSSIBLE_MARGIN_LOW):
            obs.quality_flags.append(f"impossible_margin:{obs.value:.2%}")


def _validate_ratio_bounds(observations: list[MetricObservation]) -> None:
    for obs in observations:
        if obs.value is not None and (obs.value > _IMPOSSIBLE_RATIO_HIGH or obs.value < _IMPOSSIBLE_RATIO_LOW):
            obs.quality_flags.append(f"implausible_ratio:{obs.value:.2%}")


def _validate_coverage_bounds(observations: list[MetricObservation]) -> None:
    for obs in observations:
        if obs.value is not None and abs(obs.value) > _IMPLAUSIBLE_COVERAGE_ABS:
            obs.quality_flags.append(f"implausible_coverage:{obs.value:.1f}x")


def _validate_growth_bounds(observations: list[MetricObservation]) -> None:
    for obs in observations:
        if obs.value is not None and abs(obs.value) > _EXTREME_GROWTH_ABS:
            obs.quality_flags.append(f"extreme_growth_value:{obs.value:.2%}")


def validate_observations(observations: list[MetricObservation], kind: Literal["raw", "margin", "ratio", "coverage", "growth"] = "raw") -> list[MetricObservation]:
    """Attach quality flags in place. Never removes or alters a value -- a
    flagged observation stays in the series for a human to review (section 8).
    """
    _validate_dates(observations)
    _validate_duplicates(observations)
    if kind == "margin":
        _validate_margin_bounds(observations)
    elif kind == "ratio":
        _validate_ratio_bounds(observations)
    elif kind == "coverage":
        _validate_coverage_bounds(observations)
    elif kind == "growth":
        _validate_growth_bounds(observations)
    _validate_outliers(observations)
    return observations


def _to_observations(raw: dict, metric: str, source: str, retrieved_at: str, kind: Literal["raw", "margin", "ratio", "coverage", "growth"] = "raw", period_type: PeriodType = "annual") -> list[MetricObservation]:
    periods = _sorted_periods(raw)
    observations = [
        MetricObservation(
            metric=metric, value=raw[p], period=_period_label(p, period_type),
            period_type=period_type, period_end=p, is_estimate=False,
            source=source, retrieved_at=retrieved_at,
        )
        for p in periods
    ]
    return validate_observations(observations, kind=kind)


# ---------------------------------------------------------------------------
# Trend classification -- descriptive only, never a recommendation (section 9)
# ---------------------------------------------------------------------------

def classify_trend(observations: list[MetricObservation], higher_is_better: bool = True, tolerance: float = 0.02) -> TrendState:
    """Deterministic trend state from a chronological (oldest->newest) series.

    Needs at least 2 non-null values. Uses both the average period-over-period
    move (the multi-period trend) and the most recent move (so a single sharp
    reversal after a long run doesn't get mislabeled) -- a documented,
    tunable heuristic, not a statistical fit, consistent with the rest of this
    project's normalization functions (`horizon.py`, `strategy.py`).
    """
    values = [o.value for o in observations if o.value is not None]
    if len(values) < 2:
        return "INSUFFICIENT_DATA"

    deltas = [b - a for a, b in zip(values, values[1:])]
    if not higher_is_better:
        deltas = [-d for d in deltas]

    avg_delta = sum(deltas) / len(deltas)
    last_delta = deltas[-1]

    if avg_delta > tolerance and last_delta >= -tolerance:
        return "IMPROVING"
    if avg_delta < -tolerance and last_delta <= tolerance:
        return "DETERIORATING"
    return "STABLE"


def _valuation_context(pe: Optional[float], forward_pe: Optional[float]) -> TrendState:
    """Forward-vs-trailing P/E compression as a simple valuation-direction
    proxy. NOT a peer or historical-multiple comparison (this pass has no
    sector-peer framework -- see module docstring); a forward multiple well
    below trailing suggests the market expects growth to compress the
    multiple, and vice versa.
    """
    if pe is None or forward_pe is None or pe <= 0:
        return "INSUFFICIENT_DATA"
    change = (forward_pe - pe) / pe
    if change <= -0.05:
        return "IMPROVING"
    if change >= 0.05:
        return "DETERIORATING"
    return "STABLE"


# ---------------------------------------------------------------------------
# Output schema (brief section 10)
# ---------------------------------------------------------------------------

class CompanyFundamentals(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    observation_date: Optional[str] = None  # most recent underlying reporting period end

    revenue_growth: Optional[float] = None
    revenue_growth_trend: TrendState = "INSUFFICIENT_DATA"

    eps_growth: Optional[float] = None
    eps_growth_trend: TrendState = "INSUFFICIENT_DATA"

    gross_margin: Optional[float] = None  # extra, beyond the minimum schema -- requested in section 3

    operating_margin: Optional[float] = None
    operating_margin_trend: TrendState = "INSUFFICIENT_DATA"
    operating_income_growth: Optional[float] = None  # extra -- requested in section 3's Growth list

    net_margin: Optional[float] = None
    net_margin_trend: TrendState = "INSUFFICIENT_DATA"

    roe: Optional[float] = None
    roic: Optional[float] = None  # only when NOPAT/invested-capital inputs are all real, never assumed

    free_cash_flow: Optional[float] = None
    fcf_margin: Optional[float] = None
    fcf_growth: Optional[float] = None
    fcf_trend: TrendState = "INSUFFICIENT_DATA"

    cash: Optional[float] = None
    total_debt: Optional[float] = None  # current snapshot only -- see compute_fundamentals note
    net_debt: Optional[float] = None  # current snapshot only (total_debt has no reliable per-period history)
    leverage: Optional[float] = None  # total_debt / total_equity, current snapshot only, same reason
    interest_coverage: Optional[float] = None  # extra -- only when an interest-expense line is actually present
    balance_sheet_trend: TrendState = "INSUFFICIENT_DATA"  # derived from the cash trend (the one genuinely historical balance-sheet series available); rising cash = IMPROVING

    market_cap: Optional[float] = None  # extra -- needed for fcf_yield/net_debt-to-cap and shown in the UI
    pe: Optional[float] = None
    forward_pe: Optional[float] = None
    ev_ebitda: Optional[float] = None
    price_sales: Optional[float] = None
    price_book: Optional[float] = None
    fcf_yield: Optional[float] = None
    peg_ratio: Optional[float] = None  # extra
    valuation_context: TrendState = "INSUFFICIENT_DATA"  # see `_valuation_context`

    data_freshness: Optional[str] = None  # when this snapshot was retrieved (cache write time)
    data_quality: list[str] = Field(default_factory=list)  # aggregated quality_flags across every series, for review
    sources: list[str] = Field(default_factory=list)

    # Full per-metric historical series (section 4) -- the derived fields
    # above are read off the *latest* observation in these series, not a
    # second, independently-fetched value, so there is exactly one pipeline
    # per metric, not a duplicate one for "latest" vs "history".
    history: dict[str, list[MetricObservation]] = Field(default_factory=dict)


def _latest(observations: list[MetricObservation]) -> Optional[float]:
    for obs in reversed(observations):
        if obs.value is not None:
            return obs.value
    return None


def compute_fundamentals(ticker: str, provider: Optional[FundamentalsProvider] = None) -> dict:
    """Build the full `CompanyFundamentals` object for one ticker, as a plain dict.

    Every leaf value is `Optional[float]`/`TrendState`, never fabricated --
    see module docstring. Returns `.model_dump()` (a plain dict) rather than
    the pydantic object directly, matching how `Signal`/`InstitutionalMention`
    are handled elsewhere in this project once validated.
    """
    ticker = ticker.upper()
    provider = provider or YFinanceFundamentalsProvider()
    try:
        raw = provider.get_raw_fundamentals(ticker)
    except Exception:
        # The built-in YFinanceFundamentalsProvider already falls back to a
        # stale cache internally; this is the defense-in-depth layer for any
        # other provider (e.g. a caller-supplied one in tests, or a future
        # paid provider) that raises instead -- one ticker's fundamentals
        # failure must never crash a batch/UI render (same principle as
        # research_engine.py's price-fetch try/except).
        logger.exception("Fundamentals provider failed for %s -- returning an empty snapshot", ticker)
        raw = {}
    info = raw.get("info") or {}
    income_stmt = raw.get("income_stmt") or {}
    cashflow = raw.get("cashflow") or {}
    balance_sheet = raw.get("balance_sheet") or {}

    retrieved_at = datetime.now(timezone.utc).isoformat()
    history: dict[str, list[MetricObservation]] = {}
    sources: set[str] = set()

    def track(metric: str, raw_series: dict, source: str, kind: Literal["raw", "margin", "growth"] = "raw") -> list[MetricObservation]:
        if raw_series:
            sources.add(source)
        observations = _to_observations(raw_series, metric, source, retrieved_at, kind=kind)
        history[metric] = observations
        return observations

    # --- Income statement series -------------------------------------------------
    revenue_raw = _first_present(income_stmt, "Total Revenue", "TotalRevenue")
    gross_profit_raw = _first_present(income_stmt, "Gross Profit", "GrossProfit")
    operating_income_raw = _first_present(income_stmt, "Operating Income", "OperatingIncome")
    net_income_raw = _first_present(income_stmt, "Net Income", "NetIncome")
    eps_raw = _first_present(income_stmt, "Diluted EPS", "DilutedEPS", "Basic EPS")
    interest_expense_raw = _first_present(income_stmt, "Interest Expense", "InterestExpense")
    tax_provision_raw = _first_present(income_stmt, "Tax Provision", "TaxProvision")
    pretax_income_raw = _first_present(income_stmt, "Pretax Income", "PretaxIncome")

    revenue_obs = track("revenue", revenue_raw, INCOME_STMT_SOURCE)
    track("revenue_growth", _raw_growth(revenue_raw), INCOME_STMT_SOURCE, kind="growth")
    track("gross_margin", _raw_ratio(gross_profit_raw, revenue_raw), INCOME_STMT_SOURCE, kind="margin")
    track("operating_margin", _raw_ratio(operating_income_raw, revenue_raw), INCOME_STMT_SOURCE, kind="margin")
    track("operating_income_growth", _raw_growth(operating_income_raw), INCOME_STMT_SOURCE, kind="growth")
    track("net_margin", _raw_ratio(net_income_raw, revenue_raw), INCOME_STMT_SOURCE, kind="margin")
    track("eps_growth", _raw_growth(eps_raw), INCOME_STMT_SOURCE, kind="growth")

    # --- Cash flow series ----------------------------------------------------------
    operating_cf_raw = _first_present(cashflow, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex_raw = _first_present(cashflow, "Capital Expenditure", "CapitalExpenditure")
    direct_fcf_raw = _first_present(cashflow, "Free Cash Flow", "FreeCashFlow")
    fcf_raw = direct_fcf_raw or _raw_sum(operating_cf_raw, capex_raw)  # capex is reported negative

    fcf_source = CASHFLOW_SOURCE
    fcf_obs = track("free_cash_flow", fcf_raw, fcf_source)
    track("fcf_growth", _raw_growth(fcf_raw), fcf_source, kind="growth")
    track("fcf_margin", _raw_ratio(fcf_raw, revenue_raw), fcf_source, kind="margin")

    # --- Balance sheet series --------------------------------------------------------
    cash_raw = _first_present(balance_sheet, "Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash")
    equity_raw = _first_present(balance_sheet, "Stockholders Equity", "Total Stockholder Equity", "TotalStockholderEquity")

    cash_obs = track("cash", cash_raw, BALANCE_SHEET_SOURCE)
    equity_obs = track("total_equity", equity_raw, BALANCE_SHEET_SOURCE)

    # total_debt is only reliably available as a current snapshot from `.info`
    # for most tickers (balance-sheet debt line items are split/renamed too
    # inconsistently across issuers to trust a generic row lookup); it is
    # still tracked with full provenance as a single-observation series.
    total_debt_value = info.get("totalDebt")
    total_debt_raw = {datetime.now(timezone.utc).date().isoformat(): total_debt_value} if total_debt_value is not None else {}
    total_debt_obs = track("total_debt", total_debt_raw, INFO_SOURCE, kind="raw")
    if total_debt_value is not None:
        for obs in total_debt_obs:
            obs.period_type = "ttm"

    # net_debt/leverage use `total_debt_value`, which is only a *current*
    # snapshot (see above) -- computing it against each historical cash/equity
    # period would silently mix a TTM figure into an annual series (exactly
    # what section 6 says never to do). So net_debt/leverage are single
    # current-period values, not fabricated historical series; only `cash`
    # and `total_equity` (both genuinely historical) get a real trend.
    latest_cash = _latest(cash_obs)
    latest_equity = _latest(equity_obs)
    latest_cash_period = cash_obs[-1].period_end if cash_obs else None

    net_debt_value = (total_debt_value - latest_cash) if (total_debt_value is not None and latest_cash is not None) else None
    net_debt_raw = {latest_cash_period: net_debt_value} if (net_debt_value is not None and latest_cash_period) else {}
    net_debt_obs = track("net_debt", net_debt_raw, f"{BALANCE_SHEET_SOURCE}+{INFO_SOURCE}")
    for obs in net_debt_obs:
        obs.period_type = "ttm"

    leverage_value = (total_debt_value / latest_equity) if (total_debt_value is not None and latest_equity) else None
    leverage_raw = {latest_cash_period: leverage_value} if (leverage_value is not None and latest_cash_period) else {}
    track("leverage", leverage_raw, f"{BALANCE_SHEET_SOURCE}+{INFO_SOURCE}", kind="ratio")
    for obs in history["leverage"]:
        obs.period_type = "ttm"

    interest_coverage_obs = track("interest_coverage", _raw_ratio(operating_income_raw, {p: abs(v) if v is not None else None for p, v in interest_expense_raw.items()}), f"{INCOME_STMT_SOURCE}", kind="coverage")

    # ROE/ROIC: prefer statement-derived per-period series over `.info`'s
    # single snapshot when the required rows are all present.
    roe_by_period = _raw_ratio(net_income_raw, equity_raw)
    roe_obs = track("roe", roe_by_period, f"{INCOME_STMT_SOURCE}+{BALANCE_SHEET_SOURCE}", kind="ratio")

    tax_rate_raw = _raw_ratio(tax_provision_raw, pretax_income_raw)
    roic_by_period: dict[str, float] = {}
    for period, op_income in operating_income_raw.items():
        tax_rate = tax_rate_raw.get(period)
        equity_value = equity_raw.get(period)
        cash_value = cash_raw.get(period)
        if op_income is None or tax_rate is None or equity_value is None or total_debt_value is None or cash_value is None:
            continue
        invested_capital = total_debt_value + equity_value - cash_value
        if invested_capital <= 0:
            continue
        nopat = op_income * (1 - tax_rate)
        roic_by_period[period] = nopat / invested_capital
    roic_obs = track("roic", roic_by_period, f"{INCOME_STMT_SOURCE}+{BALANCE_SHEET_SOURCE}", kind="ratio")

    # --- Info-only current values (no reliable multi-period history via yfinance) ---
    market_cap = info.get("marketCap")
    pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    ev_ebitda = info.get("enterpriseToEbitda")
    price_sales = info.get("priceToSalesTrailing12Months")
    price_book = info.get("priceToBook")
    peg_ratio = info.get("trailingPegRatio") or info.get("pegRatio")
    if info:
        sources.add(INFO_SOURCE)

    latest_fcf = _latest(fcf_obs)
    fcf_yield = (latest_fcf / market_cap) if (latest_fcf and market_cap) else None

    # --- observation_date: the most recent real reporting period we have ----------
    candidate_dates = [
        obs.period_end for series in (revenue_obs, fcf_obs, cash_obs) for obs in series if obs.period_end
    ]
    observation_date = max(candidate_dates) if candidate_dates else None

    data_quality = sorted(
        f"{metric}:{obs.period}:{flag}"
        for metric, observations in history.items()
        for obs in observations
        for flag in obs.quality_flags
    )

    result = CompanyFundamentals(
        ticker=ticker,
        company_name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        observation_date=observation_date,
        revenue_growth=_latest(history["revenue_growth"]),
        revenue_growth_trend=classify_trend(history["revenue_growth"]),
        eps_growth=_latest(history["eps_growth"]),
        eps_growth_trend=classify_trend(history["eps_growth"]),
        gross_margin=_latest(history["gross_margin"]),
        operating_margin=_latest(history["operating_margin"]),
        operating_margin_trend=classify_trend(history["operating_margin"]),
        operating_income_growth=_latest(history["operating_income_growth"]),
        net_margin=_latest(history["net_margin"]),
        net_margin_trend=classify_trend(history["net_margin"]),
        roe=_latest(roe_obs) if roe_obs else info.get("returnOnEquity"),
        roic=_latest(roic_obs),
        free_cash_flow=latest_fcf,
        fcf_margin=_latest(history["fcf_margin"]),
        fcf_growth=_latest(history["fcf_growth"]),
        fcf_trend=classify_trend(history["fcf_growth"]),
        cash=_latest(cash_obs),
        total_debt=total_debt_value,
        net_debt=_latest(net_debt_obs),
        leverage=_latest(history["leverage"]),
        interest_coverage=_latest(interest_coverage_obs),
        # net_debt/total_debt are current-only (see above), so the only
        # genuinely historical balance-sheet series available is cash --
        # rising cash reads as an improving balance sheet.
        balance_sheet_trend=classify_trend(cash_obs, higher_is_better=True),
        market_cap=market_cap,
        pe=pe,
        forward_pe=forward_pe,
        ev_ebitda=ev_ebitda,
        price_sales=price_sales,
        price_book=price_book,
        fcf_yield=fcf_yield,
        peg_ratio=peg_ratio,
        valuation_context=_valuation_context(pe, forward_pe),
        data_freshness=last_updated(ticker),
        data_quality=data_quality,
        sources=sorted(sources),
        history=history,
    )
    return result.model_dump()
