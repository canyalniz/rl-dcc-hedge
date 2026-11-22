import inspect
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import quantstats as qs
from scipy.stats import norm
from scipy.stats import t as student_t

def annual_returns(daily_returns: pd.Series, periods: int = 52):
    compounded_growth = (1 + daily_returns).prod()
    n_periods = daily_returns.shape[0] / periods  # Assuming 252 trading days in a year
    return compounded_growth ** (1 / n_periods) - 1

def cumulative_returns(daily_returns: pd.Series):
    return (1 + daily_returns).prod() - 1

def empirical_cvar(returns, sigma=1, confidence=0.95, prepare_returns=True):
    """
    Empirical (historical) Conditional Value at Risk, a drop-in for qs.stats.cvar.

    Same signature and return convention as qs.stats.cvar (a negative number for
    a loss; a float for a Series, a per-column Series for a DataFrame), but the
    VaR threshold is the empirical ``1 - confidence`` quantile of ``returns``
    (np.quantile, linear interpolation) instead of the normal-distribution
    quantile, and the CVaR is the mean of the returns at or below it. The tail
    always holds at least the minimum return, so no VaR fallback is needed.

    Args:
        returns (pd.Series | pd.DataFrame): return series to analyze
        sigma (float): accepted for interface parity with qs.stats.cvar; it has
            no effect on the empirical estimate
        confidence (float): confidence level (0.95 = 95%; values > 1 are read
            as percentages, as in quantstats)
        prepare_returns (bool): run quantstats' _prepare_returns first

    Returns:
        float | pd.Series: empirical CVaR (expected return in the tail)
    """
    if prepare_returns:
        returns = qs.utils._prepare_returns(returns)
    if confidence > 1:
        confidence = confidence / 100

    def _series_cvar(r: pd.Series) -> float:
        r = r.dropna().to_numpy(dtype=np.float64)
        if r.size == 0:
            return np.nan
        var = np.quantile(r, 1 - confidence)
        return r[r <= var].mean()

    if isinstance(returns, pd.DataFrame):
        return pd.Series({col: _series_cvar(returns[col]) for col in returns.columns})
    return _series_cvar(pd.Series(returns))

def _t_es_factor(nu, alpha: float):
    """Lower-tail expected shortfall of a standard (raw) Student-t with ``nu`` dof at
    tail probability ``alpha``, in units of the t scale (a negative number):
    ES = -t.pdf(q) (nu + q^2) / ((nu - 1) alpha), q = t.ppf(alpha); NaN for nu <= 1
    (infinite mean). Tends to the Gaussian factor -norm.pdf(norm.ppf(alpha)) / alpha
    as nu -> inf."""
    nu = np.asarray(nu, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = student_t.ppf(alpha, nu)
        es = -student_t.pdf(q, nu) * (nu + q ** 2) / ((nu - 1) * alpha)
    return np.where(nu > 1, es, np.nan)

def student_t_cvar(returns, sigma=1, confidence=0.95, prepare_returns=True, df=None):
    """
    Parametric Student-t Conditional Value at Risk, a drop-in for qs.stats.cvar.

    Same signature and return convention as qs.stats.cvar (a negative number for
    a loss; a float for a Series, a per-column Series for a DataFrame), but the
    returns are modeled as loc + scale * T_nu (T_nu a raw Student-t) instead of a
    normal, and the closed-form expected shortfall of the ``1 - confidence``
    lower tail is returned:

        CVaR = loc + sigma * scale * ES_nu,
        ES_nu = -t.pdf(q, nu) (nu + q^2) / ((nu - 1) (1 - confidence)),  q = t.ppf(1 - confidence, nu)

    Parameters are estimated as follows:
      - ``df=None``: (nu, loc, scale) by maximum likelihood (scipy.stats.t.fit;
        ~40 ms per 520-observation series);
      - ``df`` given: nu = df is fixed and (loc, scale) are moment-matched,
        loc = mean, scale = std * sqrt((nu - 2) / nu) (requires df > 2), the
        convention of the cvar_95 reward component (cf. RLConfig.dist_df).
    A (fitted) nu <= 1 has no finite expected shortfall and yields NaN.

    Args:
        returns (pd.Series | pd.DataFrame): return series to analyze
        sigma (float): multiplier of the fitted scale (default: 1), as quantstats
            multiplies the std
        confidence (float): confidence level (0.95 = 95%; values > 1 are read
            as percentages, as in quantstats)
        prepare_returns (bool): run quantstats' _prepare_returns first
        df (float | None): fixed degrees of freedom; None fits them

    Returns:
        float | pd.Series: Student-t CVaR (expected return in the tail)
    """
    if prepare_returns:
        returns = qs.utils._prepare_returns(returns)
    if confidence > 1:
        confidence = confidence / 100
    alpha = 1 - confidence

    def _series_cvar(r: pd.Series) -> float:
        r = r.dropna().to_numpy(dtype=np.float64)
        if r.size < 2:
            return np.nan
        if df is None:
            nu, loc, scale = student_t.fit(r)
        else:
            if df <= 2:
                raise ValueError(f"student_t_cvar: a fixed df must exceed 2 for moment matching, got {df}")
            nu, loc, scale = df, r.mean(), r.std(ddof=1) * np.sqrt((df - 2) / df)
        return float(loc + sigma * scale * _t_es_factor(nu, alpha))

    if isinstance(returns, pd.DataFrame):
        return pd.Series({col: _series_cvar(returns[col]) for col in returns.columns})
    return _series_cvar(pd.Series(returns))

def periods_modifier(f:callable, periods:int=52):
    def _f(*args,**kwargs):
        return f(*args, periods=periods, **kwargs)
    return _f

STAT_METRICS = {
    "Annual Return": annual_returns,
    "Cumulative Returns": cumulative_returns,
    "Annual Volatility":qs.stats.volatility,
    "Sharpe Ratio": qs.stats.sharpe,
    "Calmar Ratio": qs.stats.calmar,
    "Max Drawdown": qs.stats.max_drawdown,
    "Sortino Ratio": qs.stats.sortino,
    "Skew": qs.stats.skew,
    "Tail Ratio": qs.stats.tail_ratio,
    "Daily VaR (95%)": qs.stats.value_at_risk,
    "CVaR (95%)": qs.stats.cvar,
}

mod_list = [
    annual_returns,
    qs.stats.volatility,
    qs.stats.sharpe,
    qs.stats.calmar,
    qs.stats.sortino,
]

def compute_portfolio_metrics(daily_returns: pd.Series, metrics: dict = STAT_METRICS, periods: int = 52):
    results = pd.Series()
    for name, func in metrics.items():
        func = periods_modifier(func, periods) if getattr(func, "func", func) in mod_list else func
        try:
            results[name] = func(daily_returns)
        except Exception as e:
            print(f"Error computing {name}: {e}")
            results[name] = np.nan
    return results

def portfolio_metrics_from_traj_df_tuple(tuple_of_trajectory_dfs: Tuple[pd.DataFrame], metrics: dict = STAT_METRICS, df_obs_freq:str="W", periods:int=52):
    tuple_perf_stats = []
    for trajectory_df in tuple_of_trajectory_dfs:
        portfolio_simple_returns = trajectory_df["portfolio_simple_returns"]
        portfolio_simple_returns.index = pd.date_range(
            start="2000-01-01",
            periods=len(portfolio_simple_returns),
            freq=df_obs_freq,
        )
        tuple_perf_stats.append(
            compute_portfolio_metrics(
                portfolio_simple_returns,
                metrics = metrics,
                periods = periods
            )
        )

    agg_stats = {}
    for metric in metrics:
        agg_stats[metric] = np.array(
            [perf_stats[metric] for perf_stats in tuple_perf_stats]
        ).mean()

    return agg_stats

# --------------------------------------------------------------------------- #
# batched (row-wise) metrics
# --------------------------------------------------------------------------- #
# NumPy re-implementations of the metrics above, mirroring the quantstats
# (0.0.81) / pandas formulas they call, computed over a (n_runs, T) matrix in
# one pass instead of one compute_portfolio_metrics call per run (each of which
# pays quantstats' per-call pandas overhead, ~13 ms per run). Rows whose values
# would hit a quantstats preprocessing heuristic or a degenerate edge case are
# routed through compute_portfolio_metrics instead (see _batch_fallback_mask).

def _batch_annual_returns(r: np.ndarray, periods: int = 52) -> np.ndarray:
    n_periods = r.shape[1] / periods
    return np.prod(1 + r, axis=1) ** (1 / n_periods) - 1

def _batch_cumulative_returns(r: np.ndarray) -> np.ndarray:
    return np.prod(1 + r, axis=1) - 1

def _batch_std(r: np.ndarray) -> np.ndarray:
    return np.std(r, axis=1, ddof=1)

def _batch_excess(r: np.ndarray, rf: float, periods: Optional[int]) -> np.ndarray:
    # quantstats _prepare_returns -> to_excess_returns (only for rf > 0)
    if rf > 0:
        rf_period = np.power(1 + rf, 1.0 / periods) - 1.0 if periods is not None else rf
        return r - rf_period
    return r

def _batch_volatility(r: np.ndarray, periods: int = 252) -> np.ndarray:
    return _batch_std(r) * np.sqrt(periods)

def _batch_sharpe(r: np.ndarray, rf: float = 0.0, periods: int = 252) -> np.ndarray:
    ex = _batch_excess(r, rf, periods)
    return ex.mean(axis=1) / _batch_std(ex) * np.sqrt(1 if periods is None else periods)

def _batch_sortino(r: np.ndarray, rf: float = 0.0, periods: int = 252) -> np.ndarray:
    ex = _batch_excess(r, rf, periods)
    downside = np.sqrt(np.where(ex < 0, ex ** 2, 0.0).sum(axis=1) / ex.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(downside == 0, np.nan, ex.mean(axis=1) / downside)
    return res * np.sqrt(1 if periods is None else periods)

def _batch_max_drawdown(r: np.ndarray) -> np.ndarray:
    # _prepare_prices -> to_prices(base=1): 1 + compsum, plus the phantom 1.0 baseline row
    prices = 1.0 + (np.cumprod(1 + r, axis=1) - 1)
    extended = np.concatenate([np.ones((r.shape[0], 1)), prices], axis=1)
    return (extended / np.maximum.accumulate(extended, axis=1)).min(axis=1) - 1

def _batch_calmar(r: np.ndarray, periods: int = 252) -> np.ndarray:
    years = r.shape[1] / periods
    cagr = np.abs(_batch_cumulative_returns(r) + 1.0) ** (1.0 / years) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        return cagr / np.abs(_batch_max_drawdown(r))

def _batch_skew(r: np.ndarray) -> np.ndarray:
    # pandas nanops.nanskew (bias-corrected sample skewness)
    count = r.shape[1]
    adjusted = r - (r.sum(axis=1, dtype=np.float64) / count)[:, None]
    adjusted2 = adjusted ** 2
    m2 = adjusted2.sum(axis=1, dtype=np.float64)
    m3 = (adjusted2 * adjusted).sum(axis=1, dtype=np.float64)
    max_abs = np.abs(r).max(axis=1, initial=0.0)
    eps = np.finfo(np.float64).eps
    m2 = np.where(np.abs(m2) < ((eps * max_abs) ** 2) * count, 0, m2)
    m3 = np.where(np.abs(m3) < ((eps * max_abs) ** 3) * count, 0, m3)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = (count * (count - 1) ** 0.5 / (count - 2)) * (m3 / m2 ** 1.5)
    return np.where(m2 == 0, 0, res)

def _batch_tail_ratio(r: np.ndarray) -> np.ndarray:
    upper = np.quantile(r, 0.95, axis=1)
    lower = np.quantile(r, 0.05, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(lower == 0, np.nan, np.abs(upper / lower))

def _batch_value_at_risk(r: np.ndarray) -> np.ndarray:
    return norm.ppf(0.05, r.mean(axis=1), _batch_std(r))

def _batch_cvar(r: np.ndarray) -> np.ndarray:
    var = _batch_value_at_risk(r)
    below = r < var[:, None]
    n_below = below.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_var = np.where(below, r, 0.0).sum(axis=1) / n_below
    return np.where(n_below > 0, c_var, var)

def _batch_empirical_cvar(r: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    if confidence > 1:
        confidence = confidence / 100
    var = np.quantile(r, 1 - confidence, axis=1)
    below = r <= var[:, None]
    return np.where(below, r, 0.0).sum(axis=1) / below.sum(axis=1)

def _batch_student_t_cvar(r: np.ndarray, sigma: float = 1, confidence: float = 0.95, df: Optional[float] = None) -> np.ndarray:
    if confidence > 1:
        confidence = confidence / 100
    alpha = 1 - confidence
    if df is None:  # no vectorized t MLE: fit row by row
        nu, loc, scale = np.array([student_t.fit(row) for row in r]).T
    else:
        if df <= 2:
            raise ValueError(f"student_t_cvar: a fixed df must exceed 2 for moment matching, got {df}")
        nu, loc, scale = df, r.mean(axis=1), _batch_std(r) * np.sqrt((df - 2) / df)
    return loc + sigma * scale * _t_es_factor(nu, alpha)

_BATCH_IMPLS = {
    annual_returns: _batch_annual_returns,
    cumulative_returns: _batch_cumulative_returns,
    qs.stats.volatility: _batch_volatility,
    qs.stats.sharpe: _batch_sharpe,
    qs.stats.calmar: _batch_calmar,
    qs.stats.max_drawdown: _batch_max_drawdown,
    qs.stats.sortino: _batch_sortino,
    qs.stats.skew: _batch_skew,
    qs.stats.tail_ratio: _batch_tail_ratio,
    qs.stats.value_at_risk: _batch_value_at_risk,
    qs.stats.cvar: _batch_cvar,
    empirical_cvar: _batch_empirical_cvar,
    student_t_cvar: _batch_student_t_cvar,
}
"""metric function -> batched implementation"""

_BATCH_KWARGS = {
    annual_returns: ("periods",),
    qs.stats.volatility: ("periods",),
    qs.stats.sharpe: ("rf", "periods"),
    qs.stats.calmar: ("periods",),
    qs.stats.sortino: ("rf", "periods"),
    empirical_cvar: ("confidence",),
    student_t_cvar: ("sigma", "confidence", "df"),
}
"""keyword arguments each batched implementation supports (all others take none)"""

def _resolve_batch_metrics(metrics: dict, periods: int) -> Optional[Dict[str, Tuple[callable, dict]]]:
    """name -> (batched impl, kwargs) with the kwargs compute_portfolio_metrics would
    pass (partial keywords, function defaults, and the periods override of
    mod_list functions); None when any metric has no faithful batched version."""
    resolved = {}
    for name, func in metrics.items():
        base = getattr(func, "func", func)
        if base not in _BATCH_IMPLS or getattr(func, "args", ()):
            return None
        bound = dict(getattr(func, "keywords", None) or {})
        supported = _BATCH_KWARGS.get(base, ())
        if set(bound) - set(supported):
            return None
        signature = inspect.signature(base).parameters
        kwargs = {key: bound.get(key, signature[key].default) for key in supported}
        if base in mod_list:
            kwargs["periods"] = periods
        resolved[name] = (_BATCH_IMPLS[base], kwargs)
    return resolved

def _batch_fallback_mask(r: np.ndarray) -> np.ndarray:
    """Rows the batched formulas would not reproduce: non-finite values (quantstats
    fills / drops them), rows quantstats would read as prices (all >= 0 with
    values > 1 -> pct_change / no to_prices), a first compounded price > 10
    (other drawdown baseline), fewer than 3 returns or zero dispersion
    (NaN skew / degenerate VaR)."""
    if r.shape[1] < 3:
        return np.ones(r.shape[0], dtype=bool)
    nonneg = r.min(axis=1) >= 0
    return (
        ~np.isfinite(r).all(axis=1)
        | (nonneg & (r.max(axis=1) >= 1))
        | (1 + r[:, 0] > 10)
        | (np.std(r, axis=1) == 0)
    )

def compute_portfolio_metrics_batch(
    returns_list: List[np.ndarray],
    metrics: dict = STAT_METRICS,
    periods: int = 52,
    index_list: Optional[List[Optional[pd.Index]]] = None,
    freq: str = "W",
) -> pd.DataFrame:
    """
    compute_portfolio_metrics for many return series at once.

    Returns a DataFrame with one row per entry of ``returns_list`` (in order) and
    one column per metric. Series of equal length are stacked and evaluated with
    the batched implementations; series outside their validity domain (see
    _batch_fallback_mask), or every series when a metric has no batched version,
    go through compute_portfolio_metrics, using ``index_list[i]`` as the index
    (a dummy ``freq`` date range when it, or index_list, is None).
    """
    names = list(metrics)
    out = np.full((len(returns_list), len(names)), np.nan)
    resolved = _resolve_batch_metrics(metrics, periods)

    fallback_rows: List[int] = []
    by_length: Dict[int, List[int]] = {}
    for i, returns in enumerate(returns_list):
        if resolved is None:
            fallback_rows.append(i)
        else:
            by_length.setdefault(len(returns), []).append(i)

    for rows in by_length.values():
        r = np.vstack([np.asarray(returns_list[i], dtype=np.float64) for i in rows])
        fallback = _batch_fallback_mask(r)
        fallback_rows.extend(np.asarray(rows)[fallback].tolist())
        if fallback.all():
            continue
        batch_rows = np.asarray(rows)[~fallback]
        r = r[~fallback]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for j, name in enumerate(names):
                impl, kwargs = resolved[name]
                out[batch_rows, j] = impl(r, **kwargs)

    for i in fallback_rows:
        returns = pd.Series(np.asarray(returns_list[i], dtype=np.float64))
        index = index_list[i] if index_list is not None else None
        returns.index = index if index is not None else pd.date_range(start="2000-01-01", periods=len(returns), freq=freq)
        perf = compute_portfolio_metrics(returns, metrics=metrics, periods=periods)
        out[i] = [perf[name] for name in names]

    return pd.DataFrame(out, columns=names)
