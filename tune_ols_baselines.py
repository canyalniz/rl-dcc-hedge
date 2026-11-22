"""Grid search of the rolling-window OLS hedge baseline (rl_dcc_hedge.baseline_strategies.RollingOLSHedge)
on the synthetic evaluation trajectories.

Every (window, rebalance_period) configuration of the grid is run on every
trajectory of the evaluation simulation parquet (one raw MT_HedgeEnv per
trajectory, exactly as run_experiment_tests.py runs its baselines), next to
the unhedged anchor reference of the same trajectories. Per trajectory the
TEST_STAT_METRICS (annualized with args.periods) and the two hedge-
effectiveness metrics against the anchor are recorded; the configurations are
then ranked by the across-trajectory mean of ``rank_metric`` (by default
ascending Annual Volatility, i.e. the minimum-volatility configuration first).

Outputs, under <experiments_root>/<experiment_run_name or timestamp>/:

  metrics.csv   long format [id_tag, window, rebalance_period, data, metric_name, metric_value]
  summary.csv   one row per configuration: across-trajectory mean/std of every metric, ranked

Usage
-----
  ~/.local/share/venvs/rl-dcc-hedge/bin/python tune_ols_baselines.py
  ~/.local/share/venvs/rl-dcc-hedge/bin/python tune_ols_baselines.py --windows 208 416 --rebalance-periods 1 26
"""
from __future__ import annotations

import itertools
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tyro

from rl_dcc_hedge.baseline_strategies import BaselineStrategy, RollingOLSHedge, UnhedgedAnchor
from rl_dcc_hedge.environments import MT_HedgeEnv
from rl_dcc_hedge.rl_experiments import RLExperiment
from rl_dcc_hedge.utils.portfolio_metrics import compute_portfolio_metrics

from configs.experiment_tests_config import (
    FITTING_LOG_RETURNS,
    ExpTestsConfig,
    make_test_stat_metrics,
    timestamp,
)
from run_experiment_tests import (
    ANCHOR_ID,
    HE_EDERINGTON_METRIC,
    HE_VARIANCE_REDUCTION_METRIC,
    hedge_effectiveness,
)
from run_rl import _normalize_wandb_bool_args


@dataclass
class OLSGridConfig(ExpTestsConfig):
    """Configuration of the RollingOLSHedge grid search on the evaluation trajectories."""

    eval_data_path: str = "datasets/SPY_XLU/eval_simulation_10065_209_64_unconditional.parquet"
    """evaluation simulation parquet (MT_HedgeEnv simulation matrix layout)"""

    windows: Tuple[int, ...] = (208, 416, 624, 834)
    """RollingOLSHedge window grid (number of per-step returns in the OLS window)"""

    rebalance_periods: Tuple[int, ...] = (1, 4, 26, 52, 104)
    """RollingOLSHedge rebalance_period grid (environment steps between re-estimations)"""

    rank_metric: str = "Annual Volatility"
    """metric whose across-trajectory mean ranks the configurations"""

    rank_ascending: bool = True
    """rank ascending (lower is better, e.g. Annual Volatility); set False for higher-is-better metrics such as hedge effectiveness"""

    experiments_root: str = "./results/ols_grid_search"
    """root directory of the grid search outputs"""


def main() -> None:
    args: OLSGridConfig = tyro.cli(
        OLSGridConfig,
        args=_normalize_wandb_bool_args(sys.argv[1:]),
        description="Grid search of RollingOLSHedge (window x rebalance_period) on the evaluation trajectories",
    )

    ### 1. grid ###############################################################
    n_fit = len(FITTING_LOG_RETURNS)
    bad_windows = [w for w in args.windows if not 3 <= w <= n_fit]
    if bad_windows:
        raise ValueError(f"Windows {bad_windows} outside [3, {n_fit}] (the fitting sample length)")
    grid: List[Tuple[str, int, int]] = []  # (id_tag, window, rebalance_period)
    for window, period in itertools.product(args.windows, args.rebalance_periods):
        grid.append((str(RollingOLSHedge(FITTING_LOG_RETURNS, window=window, rebalance_period=period)), window, period))

    ### 2. output directory ###################################################
    run_dir = os.path.join(args.experiments_root, args.experiment_run_name or timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Grid search run directory: {run_dir}")
    print(f"{len(grid)} configurations: windows={list(args.windows)} x rebalance_periods={list(args.rebalance_periods)}")

    ### 3. evaluation data (mirrors run_experiment_tests.py's synthetic data prep) ###
    print(f"Loading evaluation data: {args.eval_data_path}")
    eval_df = pd.read_parquet(args.eval_data_path)
    traj_ids = sorted(int(col.rsplit("_", 2)[0]) for col in eval_df.columns if col.endswith("_0_prices"))
    if not traj_ids or traj_ids != list(range(len(traj_ids))):
        raise ValueError("Evaluation data must contain contiguous '{i}_0_prices' trajectory columns 0..N-1")
    n_traj = len(traj_ids)
    args.eval_m_sim = args.eval_num_envs = n_traj
    args.eval_sim_steps = len(eval_df) - args.eval_n_start

    try:
        MT_HedgeEnv.verify_data_matrix_alignment(eval_df, mu=args.verify_mu)  # only checks traj_idx=0
    except ValueError as err:
        print(f"WARNING: evaluation data alignment verification failed ({err}); continuing with a warning only")

    innov_cols = [col for col in eval_df.columns if "_innov" in col]
    eval_df[innov_cols] = eval_df[innov_cols].clip(-4.5, 4.5)

    starting_prices = {
        "anchor": float(eval_df["0_0_prices"].iloc[0]),
        "secondary": float(eval_df["0_1_prices"].iloc[0]),
    }
    print(f"Evaluation data: {n_traj} trajectories x {args.eval_sim_steps} simulated steps, starting prices {starting_prices}")

    def run_on_eval(strategy: BaselineStrategy) -> Dict[int, np.ndarray]:
        """Portfolio simple returns of ``strategy`` on every evaluation trajectory."""
        returns = {}
        for i in traj_ids:
            env = MT_HedgeEnv(**RLExperiment._eval_env_params(
                args=args, data=eval_df, traj_indices=[i], starting_prices=starting_prices,
            ))
            returns[i] = strategy.run(env)["portfolio_simple_returns"].to_numpy(dtype=float)
        return returns

    ### 4. runs ###############################################################
    print("Running the unhedged anchor...")
    anchor_returns = run_on_eval(UnhedgedAnchor())

    test_stat_metrics = make_test_stat_metrics(periods=args.periods, rf=args.annual_financing_rate)
    expected_steps = args.eval_sim_steps - 1
    metric_rows: List[Dict[str, object]] = []

    for k, (id_tag, window, period) in enumerate(grid, start=1):
        print(f"[{k}/{len(grid)}] {id_tag}")
        strategy = RollingOLSHedge(FITTING_LOG_RETURNS, window=window, rebalance_period=period)
        for i, returns in run_on_eval(strategy).items():
            data_label = f"eval_{i}"
            if len(returns) < expected_steps:
                print(f"WARNING: {data_label} of {id_tag} terminated after {len(returns)}/{expected_steps} steps")
            series = pd.Series(returns, index=pd.date_range("2000-01-01", periods=len(returns), freq=args.df_obs_freq))
            perf = dict(compute_portfolio_metrics(series, metrics=test_stat_metrics, periods=args.periods))
            perf[HE_EDERINGTON_METRIC], perf[HE_VARIANCE_REDUCTION_METRIC] = hedge_effectiveness(anchor_returns[i], returns)
            for metric_name, metric_value in perf.items():
                metric_rows.append({
                    "id_tag": id_tag,
                    "window": window,
                    "rebalance_period": period,
                    "data": data_label,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                })

    ### 5. outputs ############################################################
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    agg = (
        metrics_df.groupby(["id_tag", "window", "rebalance_period", "metric_name"])["metric_value"]
        .agg(["mean", "std"])
        .unstack("metric_name")
    )
    agg.columns = [f"{metric} ({stat})" for stat, metric in agg.columns]
    rank_col = f"{args.rank_metric} (mean)"
    if rank_col not in agg.columns:
        raise ValueError(f"Unknown rank_metric {args.rank_metric!r}; available: {sorted(metrics_df['metric_name'].unique())}")
    summary = agg.sort_values(rank_col, ascending=args.rank_ascending).reset_index()
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary_path = os.path.join(run_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    show_cols = ["rank", "id_tag", "window", "rebalance_period"] + [
        f"{m} (mean)" for m in ("Annual Volatility", HE_VARIANCE_REDUCTION_METRIC, HE_EDERINGTON_METRIC, "Annual Return", "Sharpe Ratio", "CVaR (95%)")
    ]
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print("\n" + "=" * 100)
        order = "ascending" if args.rank_ascending else "descending"
        print(f"Configurations ranked by mean {args.rank_metric} ({order}) across {n_traj} evaluation trajectories")
        print(summary[show_cols].to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print("\nArtifacts:")
    print(f"  {metrics_path} ({len(metrics_df)} rows)")
    print(f"  {summary_path} ({len(summary)} configurations)")


if __name__ == "__main__":
    main()
