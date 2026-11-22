"""Reproduce Table tab:data:simstats (3-Methodology.tex): weekly log-return
statistics of the simulated training trajectories vs. the historical
modelling sample.

Simulated statistics are pooled over all trajectories; values in parentheses
are medians of the per-trajectory statistics.

Each trajectory file has n_hold + 2 rows (0..n_hold+1). Row 0 is the last
historical return (shared by all trajectories) and is dropped; the last row
is dropped too, so each trajectory contributes `--n-steps` (default 208)
simulated weekly returns (rows 1..208).
"""

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

ASSETS = ("SPY", "XLU")
WEEKS_PER_YEAR = 52


def _stats(x: np.ndarray, axis: int = 0) -> dict[str, np.ndarray]:
    """Statistics along `axis` (x may be 1-D or 2-D)."""
    return {
        "mean_ann": np.mean(x, axis=axis) * WEEKS_PER_YEAR,
        "vol_ann": np.std(x, axis=axis, ddof=1) * np.sqrt(WEEKS_PER_YEAR),
        "skew": stats.skew(x, axis=axis, bias=False),
        "exkurt": stats.kurtosis(x, axis=axis, fisher=True, bias=False),
        "q01": np.quantile(x, 0.01, axis=axis),
        "q99": np.quantile(x, 0.99, axis=axis),
    }


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of a[:, j] with b[:, j] for every column j."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return (a * b).sum(axis=0) / np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))


def load_simulated(path: str, n_steps: int) -> dict[str, np.ndarray]:
    """Return {asset: array of shape (n_steps, n_traj)}."""
    cols = pq.read_schema(path).names
    out = {}
    for i, asset in enumerate(ASSETS):
        asset_cols = [c for c in cols if c.endswith(f"_{i}_returns")]
        asset_cols.sort(key=lambda c: int(c.split("_")[0]))
        df = pd.read_parquet(path, columns=asset_cols)
        r = df.to_numpy()[1 : 1 + n_steps]  # drop shared row 0 and trailing row(s)
        assert r.shape[0] == n_steps, r.shape
        out[asset] = r
    return out


def load_historical(path: str, cutoff: str) -> dict[str, np.ndarray]:
    seed = pd.read_parquet(path)
    seed.index = pd.to_datetime(seed.index)
    seed = seed[seed.index < cutoff]
    return {a: seed[a].to_numpy() for a in ASSETS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", default="datasets/SPY_XLU/train_simulation_60_209_10000_unconditional.parquet")
    p.add_argument("--seed-df", default="datasets/SPY_XLU/seed_df.parquet")
    p.add_argument("--cutoff", default="2022-01-01", help="modelling sample = dates strictly before this")
    p.add_argument("--n-steps", type=int, default=208, help="simulated returns per trajectory")
    args = p.parse_args()

    sim = load_simulated(args.sim, args.n_steps)
    hist = load_historical(args.seed_df, args.cutoff)
    n_traj = sim[ASSETS[0]].shape[1]
    print(f"Simulated: {n_traj} trajectories x {args.n_steps} weeks; "
          f"historical: {len(hist[ASSETS[0]])} weeks before {args.cutoff}\n")

    pooled = {a: _stats(sim[a].ravel()) for a in ASSETS}
    per_traj_med = {a: {k: np.median(v) for k, v in _stats(sim[a], axis=0).items()} for a in ASSETS}
    hist_s = {a: _stats(hist[a]) for a in ASSETS}

    rho_pooled = np.corrcoef(sim["SPY"].ravel(), sim["XLU"].ravel())[0, 1]
    rho_med = np.median(_corr_rows(sim["SPY"], sim["XLU"]))
    rho_hist = np.corrcoef(hist["SPY"], hist["XLU"])[0, 1]

    rows = [
        ("Mean (annualised)", "mean_ann", "{:.1%}", False),
        ("Volatility (annualised)", "vol_ann", "{:.1%}", True),
        ("Skewness", "skew", "{:.2f}", True),
        ("Excess kurtosis", "exkurt", "{:.2f}", True),
        ("1% quantile", "q01", "{:.3f}", False),
        ("99% quantile", "q99", "{:.3f}", False),
    ]
    header = f"{'Statistic':<26}{'Sim SPY':>20}{'Sim XLU':>20}{'Hist SPY':>12}{'Hist XLU':>12}"
    print(header)
    print("-" * len(header))
    for label, key, fmt, show_med in rows:
        cells = []
        for a in ASSETS:
            c = fmt.format(pooled[a][key])
            if show_med:
                c += f" ({fmt.format(per_traj_med[a][key])})"
            cells.append(c)
        print(f"{label:<26}{cells[0]:>20}{cells[1]:>20}"
              f"{fmt.format(hist_s['SPY'][key]):>12}{fmt.format(hist_s['XLU'][key]):>12}")
    sim_rho = f"{rho_pooled:.2f} ({rho_med:.2f})"
    print(f"{'rho(SPY, XLU)':<26}{sim_rho:>40}{rho_hist:>24.2f}")


if __name__ == "__main__":
    main()
