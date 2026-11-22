"""Paired, condition-level comparison of one RL reward-weight tag against one baseline
(default: RL w=0.5 vs MV) over the synthetic test trajectories of a run_experiment_tests.py run.

Asks *when* one strategy beats the other: every synthetic trajectory is a paired
experiment (same market path, both strategies), so per-trajectory advantages are
related to the market conditions of that path, and per-step hedge ratios / return
contributions are related to the conditional state at each step.

Inputs (existing experiment data only, nothing is re-simulated or re-evaluated):

  <run_dir>/metrics.csv           per-trajectory metrics (outcomes)
  <run_dir>/trajectories/*.csv    per-step hedge ratios, returns, costs
  synthetic test parquet          the simulated market paths the run evaluated on
                                  (default: the DataHandler.test_sim_save_path of the
                                  configs/data_config.py defaults; verified against the
                                  unhedged anchor trajectories)
  test partition parquet          the historical test path, for reference markers
                                  (default: DataHandler.test_partition_save_path)

Alignment: trajectory step t holds the position set at decision time t
(post_tx_alpha, using features of data row t: condstd, condcorr) and the return
realized over (t, t+1] (data row t+1: returns, innov).

Condition kinds:
  ex-ante    known at decision time (model forecasts: conditional vols/correlation,
             the theoretical MV ratio, time in the episode)
  ex-post    realized along the path (realized vols/correlation, forecast surprises,
             shocks, drift, drawdown)
  behavior   what the strategies did (hedge-ratio level gap, turnover, costs)

Outputs (default <run_dir>/rl_vs_<baseline>_w<weight>/):

  report.md                       all tables below plus the run's identifying facts
  trajectory_table.csv            one row per trajectory: conditions + outcomes + advantages
  paired_outcomes.csv             per metric: mean/median advantage, bootstrap CI, win rate, Wilcoxon p
  paired_outcomes.tex             the paired-outcomes table as a LaTeX snippet (grid rules;
                                  makecell / adjustbox, cf. docs/packages.tex)
  condition_associations.csv      Spearman rho (+ BH q) of every condition vs every advantage
  condition_quartiles.csv         advantage / win rate by condition quartile
  step_bins.csv                   per-step bins of ex-ante/ex-post state: hedge ratios, pooled VR
  trajectory_bins.csv             trajectory bins of ex-post path measures: hedge ratios, pooled VR,
                                  with the historical test partition's value of each measure
  fig_paired_outcomes.png         baseline vs RL per trajectory, per metric (y=x line)
  fig_conditions_vr.png           VR advantage vs the most associated trajectory conditions
  fig_step_bins.png               hedge ratios and pooled VR by trajectory bins of the ex-post
                                  measures (orange line: historical test partition value)
  fig_vs_alphas.png               RL vs baseline hedge ratio over all steps
  fig_variance_decomposition.png  VR advantage split into var(d) and 2cov(r_base, d)
  fig_case_studies.png            the trajectories with the largest wins either way

"Advantage" is direction-adjusted so that positive always means the RL tag is better
(e.g. for Annual Volatility advantage = vol_baseline - vol_RL). RL per-trajectory
values are the mean over seeds; seed consistency is reported alongside.

Usage:
  python compare_rl_vs_baseline.py --run-dir <run_dir> [--rl-weight 0.5] [--baseline MV]
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from scipy import stats

from configs.data_config import SeedDataConfig, SimDataConfig
from summarize_experiment_tests import (
    ANCHOR_ID,
    ANNRET_METRIC,
    CVAR_METRIC,
    METRIC_LABEL,
    SHARPE_METRIC,
    SYNTH_PREFIX,
    VOL_METRIC,
    VR_METRIC,
    _latex_escape,
    _tag_weight,
)

MAXDD_METRIC = "Max Drawdown"
# (metric, direction): direction * (RL - baseline) > 0 means RL is better
OUTCOMES = [(VR_METRIC, +1), (VOL_METRIC, -1), (CVAR_METRIC, +1), (ANNRET_METRIC, +1),
            (SHARPE_METRIC, +1), (MAXDD_METRIC, +1)]
OUTCOME_KEY = {VR_METRIC: "vr", VOL_METRIC: "vol", CVAR_METRIC: "cvar", ANNRET_METRIC: "annret",
               SHARPE_METRIC: "sharpe", MAXDD_METRIC: "maxdd"}

# trajectory-level conditions: column -> (label, kind)
CONDITIONS = {
    "mean_condcorr": ("mean cond. corr", "ex-ante"),
    "std_condcorr": ("std cond. corr", "ex-ante"),
    "mean_condvol_a": ("mean cond. vol anchor (ann.)", "ex-ante"),
    "mean_condvol_s": ("mean cond. vol secondary (ann.)", "ex-ante"),
    "mean_sigma_ratio": ("mean cond. vol ratio sa/ss", "ex-ante"),
    "mean_alpha_star": ("mean MV ratio a*", "ex-ante"),
    "std_alpha_star": ("std MV ratio a*", "ex-ante"),
    "real_vol_a": ("realized vol anchor (ann.)", "ex-post"),
    "real_vol_s": ("realized vol secondary (ann.)", "ex-post"),
    "real_corr": ("realized corr", "ex-post"),
    "corr_surprise": ("realized - mean cond. corr", "ex-post"),
    "vol_surprise_a": ("realized / cond. vol anchor", "ex-post"),
    "vol_surprise_s": ("realized / cond. vol secondary", "ex-post"),
    "expost_alpha": ("ex-post optimal ratio cov/var_s", "ex-post"),
    "anchor_ann_return": ("anchor annual return", "ex-post"),
    "anchor_max_dd": ("anchor max drawdown", "ex-post"),
    "secondary_ann_return": ("secondary annual return", "ex-post"),
    "max_abs_innov_s": ("max |shock| secondary", "ex-post"),
    "excess_kurt_s": ("excess kurtosis secondary", "ex-post"),
    "base_alpha_gap_expost": ("baseline mean ratio - ex-post ratio", "ex-post"),
    "alpha_gap_rl_base": ("RL - baseline mean ratio", "behavior"),
    "base_turnover": ("baseline turnover mean|da|", "behavior"),
    "rl_turnover": ("RL turnover mean|da|", "behavior"),
    "tc_gap_rl_base": ("RL - baseline total cost (% equity)", "behavior"),
}

# step-level binning variables: column -> (label, kind)
STEP_BIN_VARS = {
    "condcorr": ("cond. corr rho_t", "ex-ante"),
    "sigma_ratio": ("cond. vol ratio sa/ss", "ex-ante"),
    "condvol_a": ("cond. vol anchor (ann.)", "ex-ante"),
    "alpha_star": ("MV ratio a*_t", "ex-ante"),
    "step": ("episode step t", "ex-ante"),
    "abs_innov_s": ("|shock secondary| (t+1)", "ex-post"),
}

# trajectory-level ex-post path measures binned in fig_step_bins.png (keys of CONDITIONS)
TRAJ_BIN_VARS = ["real_corr", "corr_surprise", "real_vol_a", "secondary_ann_return"]

TRAJ_COLUMNS = ["action", "post_tx_alpha", "post_tx_equity", "portfolio_simple_returns", "transaction_cost"]
TEST_LABEL = "test_partition"


@dataclass
class CompareConfig:
    """Configuration of the RL-tag vs baseline comparison."""

    run_dir: str
    """experiment run directory (contains metrics.csv and trajectories/)"""

    rl_weight: float = 0.4
    """log_return reward weight selecting the RL tag (must identify exactly one tag)"""

    rl_tag: Optional[str] = None
    """explicit RL tag (overrides rl_weight), e.g. 'log_return:0.5,vol:0.22'"""

    baseline: str = "MRR-C_100"
    """baseline id_tag to compare against"""

    sim_data_path: Optional[str] = None
    """synthetic test parquet the run evaluated on; None = configs/data_config.py defaults"""

    test_data_path: Optional[str] = None
    """historical test partition parquet (reference markers); None = configs/data_config.py defaults"""

    output_dir: Optional[str] = None
    """where to write the artifacts; None = <run_dir>/rl_vs_<baseline>_w<weight>"""

    periods: int = SeedDataConfig.periods
    """periods per year for annualization"""

    n_bins: int = 4
    """number of quantile bins for the step-level analysis"""

    n_bootstrap: int = 10000
    """bootstrap resamples (over trajectories) for the mean-advantage CIs"""

    n_case_studies: int = 2
    """trajectories shown per side (largest RL wins, largest baseline wins)"""

    top_conditions: int = 6
    """conditions shown in the figure / quartile tables of report.md"""

    dpi: int = 150
    """figure resolution"""


def _sanitize(name: str) -> str:
    """Trajectory file-name sanitization of run_experiment_tests._sanitize (copied to avoid importing torch)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _default_sim_data_path() -> str:
    """DataHandler.test_sim_save_path under the configs/data_config.py defaults."""
    return os.path.join(
        SeedDataConfig.datasets_root,
        SeedDataConfig.dataset_name,
        f"eval_simulation_{SimDataConfig.test_sim_seed}_{SimDataConfig.test_sim_steps}_"
        f"{SimDataConfig.test_m_sim}_{SimDataConfig.startMethod}.parquet",
    )


def _default_test_data_path() -> str:
    """DataHandler.test_partition_save_path under the configs/data_config.py defaults."""
    return os.path.join(SeedDataConfig.datasets_root, SeedDataConfig.dataset_name, "test_partition.parquet")


def _market_conditions(condstd: np.ndarray, condcorr: np.ndarray, ret_a: np.ndarray, ret_s: np.ndarray,
                       innov_s: np.ndarray, periods: int) -> Dict[str, float]:
    """Ex-ante / ex-post market conditions of one path (the strategy-independent CONDITIONS).

    condstd (n, 2) and condcorr (n,) are the decision-time features of steps 0..n-1; ret_a, ret_s
    (simple returns) and innov_s the realizations over the same steps (data rows 1..n).
    """
    sqrt_p = np.sqrt(periods)
    sigma_ratio = condstd[:, 0] / condstd[:, 1]
    alpha_star = condcorr * sigma_ratio
    real_vol_a, real_vol_s = np.std(ret_a, ddof=1) * sqrt_p, np.std(ret_s, ddof=1) * sqrt_p
    real_corr = np.corrcoef(ret_a, ret_s)[0, 1]
    return {
        "mean_condcorr": condcorr.mean(), "std_condcorr": condcorr.std(ddof=1),
        "mean_condvol_a": condstd[:, 0].mean() * sqrt_p, "mean_condvol_s": condstd[:, 1].mean() * sqrt_p,
        "mean_sigma_ratio": sigma_ratio.mean(),
        "mean_alpha_star": alpha_star.mean(), "std_alpha_star": alpha_star.std(ddof=1),
        "real_vol_a": real_vol_a, "real_vol_s": real_vol_s,
        "real_corr": real_corr, "corr_surprise": real_corr - condcorr.mean(),
        "vol_surprise_a": real_vol_a / (condstd[:, 0].mean() * sqrt_p),
        "vol_surprise_s": real_vol_s / (condstd[:, 1].mean() * sqrt_p),
        "expost_alpha": np.cov(ret_a, ret_s, ddof=1)[0, 1] / np.var(ret_s, ddof=1),
        "anchor_ann_return": _annual_return(ret_a, periods), "anchor_max_dd": _max_drawdown(ret_a),
        "secondary_ann_return": _annual_return(ret_s, periods),
        "max_abs_innov_s": np.abs(innov_s).max(), "excess_kurt_s": stats.kurtosis(ret_s),
    }


def _load_trajectory(traj_dir: str, data_label: str, id_tag: str) -> pd.DataFrame:
    path = os.path.join(traj_dir, f"{_sanitize(data_label)}_{_sanitize(id_tag)}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    return pd.read_csv(path, usecols=TRAJ_COLUMNS)


def _max_drawdown(simple_returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + simple_returns)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float((wealth / peak - 1.0).min())


def _annual_return(simple_returns: np.ndarray, periods: int) -> float:
    return float(np.prod(1.0 + simple_returns) ** (periods / len(simple_returns)) - 1.0)


def _bootstrap_mean_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _latex_p_value(p: float) -> str:
    """Math-mode p-value: 3 decimals, or mantissa x 10^exponent below 1e-3; '--' when undefined."""
    if pd.isna(p):
        return "--"
    if p >= 1e-3:
        return f"${p:.3f}$"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"${mantissa}\\times 10^{{{int(exponent)}}}$"


def _resolve_rl_tag(agent_ids: List[str], args: CompareConfig) -> str:
    tags = sorted({i.rsplit("_", 1)[0] for i in agent_ids})
    if args.rl_tag is not None:
        if args.rl_tag not in tags:
            raise ValueError(f"RL tag {args.rl_tag!r} not in the run; available: {tags}")
        return args.rl_tag
    matches = [t for t in tags if _tag_weight(t) is not None and np.isclose(_tag_weight(t), args.rl_weight)]
    if len(matches) != 1:
        raise ValueError(f"rl_weight={args.rl_weight:g} matches tags {matches} (need exactly one); pass --rl-tag. "
                         f"Available: {tags}")
    return matches[0]


def main() -> None:
    args: CompareConfig = tyro.cli(CompareConfig, description=__doc__.split("\n")[0])

    metrics_path = os.path.join(args.run_dir, "metrics.csv")
    traj_dir = os.path.join(args.run_dir, "trajectories")
    if not os.path.isfile(metrics_path) or not os.path.isdir(traj_dir):
        raise FileNotFoundError(f"{args.run_dir} needs metrics.csv and trajectories/ (a run_experiment_tests.py output)")
    sim_data_path = args.sim_data_path or _default_sim_data_path()
    if not os.path.isfile(sim_data_path):
        raise FileNotFoundError(f"Synthetic test data not found: {sim_data_path}; pass --sim-data-path")

    ### 1. structure: RL tag, seeds, trajectories ###########################
    m = pd.read_csv(metrics_path)
    m = m[m["data"].str.startswith(SYNTH_PREFIX)]
    ids = sorted(m["id_tag"].unique())
    if args.baseline not in ids:
        raise ValueError(f"Baseline {args.baseline!r} not in the run; id tags: {[i for i in ids if ':' not in i]}")
    rl_tag = _resolve_rl_tag([i for i in ids if ":" in i], args)
    rl_ids = sorted((i for i in ids if i.rsplit("_", 1)[0] == rl_tag), key=lambda i: int(i.rsplit("_", 1)[1]))
    seeds = [int(i.rsplit("_", 1)[1]) for i in rl_ids]
    traj_ids = sorted(int(d.rsplit("_", 1)[1]) for d in m["data"].unique())
    rl_label = f"RL w={_tag_weight(rl_tag):g}" if _tag_weight(rl_tag) is not None else f"RL {rl_tag}"
    base = args.baseline

    w_str = f"{_tag_weight(rl_tag):g}" if _tag_weight(rl_tag) is not None else _sanitize(rl_tag)
    output_dir = args.output_dir or os.path.join(args.run_dir, f"rl_vs_{_sanitize(base)}_w{w_str}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Comparing {rl_label} ({rl_tag}, seeds {seeds}) vs {base} on {len(traj_ids)} synthetic trajectories")
    print(f"Synthetic test data: {sim_data_path}")

    sim = pd.read_parquet(sim_data_path)
    sqrt_p = np.sqrt(args.periods)

    ### 2. per-step frame + per-trajectory conditions ########################
    step_frames: List[pd.DataFrame] = []
    traj_rows: List[Dict[str, float]] = []
    for i in traj_ids:
        label = f"{SYNTH_PREFIX}_{i}"
        anchor_traj = _load_trajectory(traj_dir, label, ANCHOR_ID)
        base_traj = _load_trajectory(traj_dir, label, base)
        rl_trajs = [_load_trajectory(traj_dir, label, rid) for rid in rl_ids]
        n_steps = min(len(anchor_traj), len(base_traj), *(len(t) for t in rl_trajs))
        if any(len(t) != n_steps for t in [anchor_traj, base_traj, *rl_trajs]):
            print(f"WARNING: {label}: unequal trajectory lengths (early termination); using the common {n_steps} steps")

        # market path: decision-time features at rows 0..n-1, realized returns/shocks at rows 1..n
        condstd = sim[[f"{i}_0_condstd", f"{i}_1_condstd"]].to_numpy()[:n_steps]
        condcorr = sim[f"{i}_condcorr"].to_numpy()[:n_steps]
        ret_a = np.expm1(sim[f"{i}_0_returns"].to_numpy()[1:n_steps + 1])
        ret_s = np.expm1(sim[f"{i}_1_returns"].to_numpy()[1:n_steps + 1])
        innov_s = sim[f"{i}_1_innov"].to_numpy()[1:n_steps + 1]

        r_u = anchor_traj["portfolio_simple_returns"].to_numpy()[:n_steps]
        if np.corrcoef(r_u, ret_a)[0, 1] < 0.99:
            raise ValueError(f"{label}: the unhedged anchor returns do not track {sim_data_path}'s anchor returns; "
                             "wrong --sim-data-path for this run?")

        r_b = base_traj["portfolio_simple_returns"].to_numpy()[:n_steps]
        a_b = base_traj["post_tx_alpha"].to_numpy()[:n_steps]
        act_b = base_traj["action"].to_numpy()[:n_steps]  # requested hedge ratio, before the env's repair
        tc_b = (base_traj["transaction_cost"] / base_traj["post_tx_equity"]).to_numpy()[:n_steps]
        r_rl = np.stack([t["portfolio_simple_returns"].to_numpy()[:n_steps] for t in rl_trajs])  # (seeds, steps)
        a_rl = np.stack([t["post_tx_alpha"].to_numpy()[:n_steps] for t in rl_trajs])
        # requested hedge ratio (policy output after ClipAction, before the env's alpha_LB/UB repair)
        act_rl = np.stack([t["action"].to_numpy()[:n_steps] for t in rl_trajs])
        tc_rl = np.stack([(t["transaction_cost"] / t["post_tx_equity"]).to_numpy()[:n_steps] for t in rl_trajs])

        sigma_ratio = condstd[:, 0] / condstd[:, 1]
        alpha_star = condcorr * sigma_ratio

        step = pd.DataFrame({
            "traj": i, "step": np.arange(n_steps),
            "condcorr": condcorr, "condvol_a": condstd[:, 0] * sqrt_p, "condvol_s": condstd[:, 1] * sqrt_p,
            "sigma_ratio": sigma_ratio, "alpha_star": alpha_star,
            "ret_a": ret_a, "ret_s": ret_s, "abs_innov_s": np.abs(innov_s),
            "alpha_base": a_b, "alpha_rl": a_rl.mean(axis=0), "action_base": act_b, "action_rl": act_rl.mean(axis=0),
            # squared deviations from each strategy's own trajectory mean: per-step variance contributions
            "sq_u": (r_u - r_u.mean()) ** 2,
            "sq_base": (r_b - r_b.mean()) ** 2,
            "sq_rl": ((r_rl - r_rl.mean(axis=1, keepdims=True)) ** 2).mean(axis=0),
            "r_u": r_u, "r_base": r_b, "r_rl": r_rl.mean(axis=0),
            "tc_base": tc_b, "tc_rl": tc_rl.mean(axis=0),
        })
        for k, s in enumerate(seeds):
            step[f"alpha_rl_{s}"] = a_rl[k]
            step[f"r_rl_{s}"] = r_rl[k]
        step_frames.append(step)

        # VR advantage decomposition per seed: VR_rl - VR_base = -(var(d) + 2 cov(r_base, d)) / var(r_u), d = r_rl - r_base
        var_u = np.var(r_u, ddof=1)
        dec_var, dec_cov = [], []
        for k in range(len(seeds)):
            d = r_rl[k] - r_b
            dec_var.append(np.var(d, ddof=1) / var_u)
            dec_cov.append(2.0 * np.cov(r_b, d, ddof=1)[0, 1] / var_u)

        market = _market_conditions(condstd, condcorr, ret_a, ret_s, innov_s, args.periods)
        traj_rows.append({
            "traj": i,
            **market,
            "base_alpha_gap_expost": a_b.mean() - market["expost_alpha"],
            "alpha_gap_rl_base": a_rl.mean() - a_b.mean(),
            "base_turnover": np.abs(np.diff(a_b)).mean(),
            "rl_turnover": np.abs(np.diff(a_rl, axis=1)).mean(),
            "tc_gap_rl_base": 100.0 * (tc_rl.sum(axis=1).mean() - tc_b.sum()),
            "mean_alpha_base": a_b.mean(), "mean_alpha_rl": a_rl.mean(),
            "dec_var_d": float(np.mean(dec_var)), "dec_2cov": float(np.mean(dec_cov)),
            "n_steps": n_steps,
        })

    steps = pd.concat(step_frames, ignore_index=True)
    traj = pd.DataFrame(traj_rows).set_index("traj")

    # historical test partition: the same market measures, as reference markers
    test_conditions: Optional[Dict[str, float]] = None
    test_data_path = args.test_data_path or _default_test_data_path()
    test_anchor_path = os.path.join(traj_dir, f"{TEST_LABEL}_{_sanitize(ANCHOR_ID)}.csv")
    if os.path.isfile(test_data_path) and os.path.isfile(test_anchor_path):
        test_df = pd.read_parquet(test_data_path)
        tics = [t.replace("^", "") for t in SeedDataConfig.tics]
        anchor_tic = SeedDataConfig.anchor_asset.replace("^", "")
        secondary_tic = next(t for t in tics if t != anchor_tic)
        r_u_test = pd.read_csv(test_anchor_path, usecols=["portfolio_simple_returns"])["portfolio_simple_returns"].to_numpy()
        n_test = len(r_u_test)
        ret_a_test = np.expm1(test_df[f"{anchor_tic}_returns"].to_numpy()[1:n_test + 1])
        if np.corrcoef(r_u_test, ret_a_test)[0, 1] < 0.99:
            raise ValueError(f"The unhedged anchor test-partition returns do not track {test_data_path}; "
                             "wrong --test-data-path for this run?")
        test_conditions = _market_conditions(
            condstd=test_df[[f"{anchor_tic}_condstd", f"{secondary_tic}_condstd"]].to_numpy()[:n_test],
            condcorr=test_df["condcorr"].to_numpy()[:n_test],
            ret_a=ret_a_test,
            ret_s=np.expm1(test_df[f"{secondary_tic}_returns"].to_numpy()[1:n_test + 1]),
            innov_s=test_df[f"{secondary_tic}_innov"].to_numpy()[1:n_test + 1],
            periods=args.periods,
        )
        print(f"Historical test partition: {test_data_path} ({n_test} steps)")
    else:
        print(f"WARNING: no historical test partition reference ({test_data_path} / {test_anchor_path} missing); "
              "fig_step_bins.png is drawn without the test-partition markers")

    ### 3. outcomes and advantages (metrics.csv) ############################
    wide = m.pivot_table(index=["id_tag", "data"], columns="metric_name", values="metric_value")
    for metric, direction in OUTCOMES:
        key = OUTCOME_KEY[metric]
        per_traj = lambda ident: np.array([wide.loc[(ident, f"{SYNTH_PREFIX}_{i}"), metric]
                                           if metric in wide.columns and (ident, f"{SYNTH_PREFIX}_{i}") in wide.index
                                           else np.nan for i in traj_ids])
        if metric != VR_METRIC:
            traj[f"anchor_{key}"] = per_traj(ANCHOR_ID)
        traj[f"base_{key}"] = per_traj(base)
        rl_vals = np.stack([per_traj(rid) for rid in rl_ids])  # (seeds, traj)
        traj[f"rl_{key}"] = rl_vals.mean(axis=0)
        traj[f"rl_{key}_min"], traj[f"rl_{key}_max"] = rl_vals.min(axis=0), rl_vals.max(axis=0)
        traj[f"adv_{key}"] = direction * (traj[f"rl_{key}"] - traj[f"base_{key}"])
        traj[f"seeds_better_{key}"] = (direction * (rl_vals - traj[f"base_{key}"].to_numpy()) > 0).sum(axis=0)

    # consistency check of the decomposition against the recorded VR metrics
    dec_err = np.nanmax(np.abs(traj["adv_vr"] + traj["dec_var_d"] + traj["dec_2cov"]))
    if dec_err > 1e-3:
        print(f"WARNING: VR decomposition deviates from metrics.csv by up to {dec_err:.2e} (realized-prefix alignment?)")
    traj.to_csv(os.path.join(output_dir, "trajectory_table.csv"))

    ### 4. paired outcome statistics #########################################
    rng = np.random.default_rng(0)
    paired_rows = []
    for metric, direction in OUTCOMES:
        key = OUTCOME_KEY[metric]
        adv = traj[f"adv_{key}"].dropna().to_numpy()
        lo, hi = _bootstrap_mean_ci(adv, args.n_bootstrap, rng)
        wilcoxon_p = stats.wilcoxon(adv).pvalue if np.any(adv != 0) else np.nan
        paired_rows.append({
            "metric": METRIC_LABEL.get(metric, metric), "direction": "higher" if direction > 0 else "lower",
            "baseline_mean": traj[f"base_{key}"].mean(), "rl_mean": traj[f"rl_{key}"].mean(),
            "baseline_median": traj[f"base_{key}"].median(), "rl_median": traj[f"rl_{key}"].median(),
            "adv_mean": adv.mean(), "adv_ci_low": lo, "adv_ci_high": hi, "adv_median": np.median(adv),
            "rl_win_rate": (adv > 0).mean(),
            "all_seeds_better": (traj[f"seeds_better_{key}"] == len(seeds)).mean(),
            "no_seed_better": (traj[f"seeds_better_{key}"] == 0).mean(),
            "wilcoxon_p": wilcoxon_p, "n_traj": len(adv),
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(os.path.join(output_dir, "paired_outcomes.csv"), index=False)

    # LaTeX version of the paired-outcomes table, in the table_primary.tex style of
    # summarize_experiment_tests.py (grid rules, math-mode numbers, no bold)
    rl_header = f"RL $w{{=}}{_tag_weight(rl_tag):g}$" if _tag_weight(rl_tag) is not None else _latex_escape(rl_label)
    base_header = _latex_escape(base)
    tex_columns = [
        (f"{base_header} mean", "baseline_mean"), ("RL mean", "rl_mean"),
        (f"{base_header} median", "baseline_median"), ("RL median", "rl_median"),
        ("Adv.\\ mean", "adv_mean"), ("Adv.\\ median", "adv_median"),
        ("RL win rate", "rl_win_rate"), ("Wilcoxon $p$", "wilcoxon_p"),
    ]
    lower_better = [_latex_escape(r["metric"]).lower() for _, r in paired.iterrows() if r["direction"] == "lower"]
    tex_caption = (
        f"Paired outcomes of {rl_header} against {base_header} over the {len(traj_ids)} synthetic test "
        f"trajectories. RL values are per-trajectory means over seeds {', '.join(str(s) for s in seeds)}. "
        f"The advantage (Adv.) is the direction-adjusted per-trajectory difference, positive when RL is better: "
        f"RL $-$ {base_header} for higher-is-better metrics"
        + (f" and {base_header} $-$ RL for {', '.join(lower_better)}" if lower_better else "")
        + ". RL win rate: share of trajectories with a positive advantage. Wilcoxon $p$: two-sided "
        "Wilcoxon signed-rank test of the per-trajectory advantages."
    )
    tex_label = f"tab:paired-outcomes-{_sanitize(base).lower()}-w{w_str}".replace("_", "-")
    tex_lines = [
        "% LaTeX version of the paired outcomes table (synthetic test trajectories), generated by",
        f"% compare_rl_vs_baseline.py from {metrics_path}.",
        "% Requires the packages loaded in packages.tex (adjustbox).",
        "",
        "\\begin{table}[tb]",
        "  \\centering",
        f"  \\caption{{{tex_caption}}}",
        f"  \\label{{{tex_label}}}",
        "  \\begin{adjustbox}{max width=\\linewidth}",
        "    \\begin{tabular}{|l|" + "r|" * len(tex_columns) + "}",
        "      \\hline",
        "      Metric & " + " & ".join(header for header, _ in tex_columns) + " \\\\",
        "      \\hline",
    ]
    for _, r in paired.iterrows():
        cells = [_latex_p_value(r[col]) if col == "wilcoxon_p" else
                 ("--" if pd.isna(r[col]) else f"${r[col]:.3f}$") for _, col in tex_columns]
        tex_lines.append("      " + _latex_escape(r["metric"]) + " & " + " & ".join(cells) + " \\\\")
        tex_lines.append("      \\hline")  # full grid: a rule beneath every body row
    tex_lines += ["    \\end{tabular}", "  \\end{adjustbox}", "\\end{table}"]
    latex = "\n".join(tex_lines) + "\n"
    latex.encode("ascii")  # guard: snippets must be plain ASCII (no raw unicode symbols)
    with open(os.path.join(output_dir, "paired_outcomes.tex"), "w") as f:
        f.write(latex)

    ### 5. condition associations (trajectory level) #########################
    assoc_rows = []
    for metric, _ in OUTCOMES:
        key = OUTCOME_KEY[metric]
        rows = []
        for col, (lab, kind) in CONDITIONS.items():
            ok = traj[[col, f"adv_{key}"]].dropna()
            res = stats.spearmanr(ok[col], ok[f"adv_{key}"])
            rows.append({"outcome": METRIC_LABEL.get(metric, metric), "condition": col, "label": lab, "kind": kind,
                         "spearman_rho": res.statistic, "p_value": res.pvalue, "n": len(ok)})
        q = stats.false_discovery_control(np.array([r["p_value"] for r in rows]))  # BH within each outcome
        for r, qv in zip(rows, q):
            r["bh_q"] = qv
        assoc_rows += rows
    assoc = pd.DataFrame(assoc_rows)
    assoc.to_csv(os.path.join(output_dir, "condition_associations.csv"), index=False)

    vr_assoc = assoc[assoc["outcome"] == METRIC_LABEL[VR_METRIC]].reindex(
        assoc[assoc["outcome"] == METRIC_LABEL[VR_METRIC]]["spearman_rho"].abs().sort_values(ascending=False).index)
    top_conditions = vr_assoc["condition"].head(args.top_conditions).tolist()

    quart_rows = []
    for col, (lab, kind) in CONDITIONS.items():
        quartile = pd.qcut(traj[col], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop")
        for q_label, grp in traj.groupby(quartile, observed=True):
            quart_rows.append({
                "condition": col, "label": lab, "kind": kind, "quartile": q_label,
                "condition_mean": grp[col].mean(), "n": len(grp),
                "base_vr": grp["base_vr"].mean(), "rl_vr": grp["rl_vr"].mean(),
                "adv_vr": grp["adv_vr"].mean(), "rl_win_rate_vr": (grp["adv_vr"] > 0).mean(),
                "adv_vol": grp["adv_vol"].mean(), "adv_annret": grp["adv_annret"].mean(),
                "adv_sharpe": grp["adv_sharpe"].mean(),
            })
    quartiles = pd.DataFrame(quart_rows)
    quartiles.to_csv(os.path.join(output_dir, "condition_quartiles.csv"), index=False)

    ### 6. step-level bins ###################################################
    bin_rows = []
    for col, (lab, kind) in STEP_BIN_VARS.items():
        bins = pd.qcut(steps[col], args.n_bins, duplicates="drop")
        for b, grp in steps.groupby(bins, observed=True):
            bin_rows.append({
                "variable": col, "label": lab, "kind": kind, "bin": str(b),
                "bin_mid": grp[col].median(), "n_steps": len(grp),
                "alpha_star": grp["alpha_star"].mean(),
                "alpha_base": grp["alpha_base"].mean(), "alpha_rl": grp["alpha_rl"].mean(),
                "alpha_gap_rl_base": (grp["alpha_rl"] - grp["alpha_base"]).mean(),
                # pooled VR of the bin: share of the anchor's variance contribution removed
                "pooled_vr_base": 1.0 - grp["sq_base"].sum() / grp["sq_u"].sum(),
                "pooled_vr_rl": 1.0 - grp["sq_rl"].sum() / grp["sq_u"].sum(),
                "share_of_anchor_var": grp["sq_u"].sum() / steps["sq_u"].sum(),
                "tc_base_bp": 1e4 * grp["tc_base"].mean(), "tc_rl_bp": 1e4 * grp["tc_rl"].mean(),
            })
    step_bins = pd.DataFrame(bin_rows)
    step_bins["pooled_vr_adv"] = step_bins["pooled_vr_rl"] - step_bins["pooled_vr_base"]
    step_bins.to_csv(os.path.join(output_dir, "step_bins.csv"), index=False)

    # trajectory bins of the ex-post path measures (steps pooled within each bin)
    traj_bin_rows = []
    for col in TRAJ_BIN_VARS:
        bins = pd.qcut(traj[col], args.n_bins, duplicates="drop")
        for b, grp in traj.groupby(bins, observed=True):
            st = steps[steps["traj"].isin(grp.index)]
            traj_bin_rows.append({
                "variable": col, "label": CONDITIONS[col][0], "bin": str(b),
                "bin_mean": grp[col].mean(), "n_traj": len(grp),
                "test_partition_value": test_conditions[col] if test_conditions is not None else np.nan,
                "alpha_star": st["alpha_star"].mean(),
                "alpha_base": st["alpha_base"].mean(), "alpha_rl": st["alpha_rl"].mean(),
                "action_base": st["action_base"].mean(), "action_rl": st["action_rl"].mean(),
                "pooled_vr_base": 1.0 - st["sq_base"].sum() / st["sq_u"].sum(),
                "pooled_vr_rl": 1.0 - st["sq_rl"].sum() / st["sq_u"].sum(),
                "rl_win_rate_vr": (grp["adv_vr"] > 0).mean(),
            })
    traj_bins = pd.DataFrame(traj_bin_rows)
    traj_bins["pooled_vr_adv"] = traj_bins["pooled_vr_rl"] - traj_bins["pooled_vr_base"]
    traj_bins.to_csv(os.path.join(output_dir, "trajectory_bins.csv"), index=False)

    # hedge-ratio policy: RL ratio as a linear function of the baseline ratio (pooled steps)
    policy = stats.linregress(steps["alpha_base"], steps["alpha_rl"])
    policy_star = stats.linregress(steps["alpha_star"], steps["alpha_rl"])

    ### 7. report ############################################################
    fmt = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    fmt_p = lambda v: "" if pd.isna(v) else (f"{v:.3f}" if v >= 1e-3 else f"{v:.1e}")
    n_var_win = int(((traj["adv_vr"] > 0) & (traj["dec_2cov"] < 0)).sum())
    report = [
        f"# {rl_label} vs {base} on the synthetic test trajectories", "",
        f"- run: `{args.run_dir}`",
        f"- RL tag: `{rl_tag}`, seeds {seeds} (per-trajectory RL value = mean over seeds)",
        f"- synthetic data: `{sim_data_path}` ({len(traj_ids)} trajectories x {int(traj['n_steps'].median())} steps)",
        "- advantage = direction-adjusted RL - baseline: **positive means RL is better**", "",
        "## Paired outcomes (per trajectory)", "",
        paired.assign(**{c: paired[c].map(fmt) for c in ["baseline_mean", "rl_mean", "baseline_median", "rl_median",
                                                           "adv_mean", "adv_ci_low", "adv_ci_high", "adv_median",
                                                           "rl_win_rate", "all_seeds_better", "no_seed_better"]},
                      wilcoxon_p=paired["wilcoxon_p"].map(fmt_p)).to_markdown(index=False), "",
        "`adv_ci_*`: 95% bootstrap CI of the mean advantage over trajectories; `all_seeds_better` / "
        "`no_seed_better`: share of trajectories where every / no RL seed beats the baseline.", "",
        "## Hedge-ratio policy (pooled steps)", "",
        f"- mean ratio: {base} {steps['alpha_base'].mean():.3f}, {rl_label} {steps['alpha_rl'].mean():.3f}, "
        f"MV theory a* {steps['alpha_star'].mean():.3f}",
        f"- alpha_RL = {policy.intercept:.3f} + {policy.slope:.3f} * alpha_{base} (R2 {policy.rvalue ** 2:.3f})",
        f"- alpha_RL = {policy_star.intercept:.3f} + {policy_star.slope:.3f} * a* (R2 {policy_star.rvalue ** 2:.3f})",
        f"- turnover mean|da|: {base} {traj['base_turnover'].mean():.4f}, {rl_label} {traj['rl_turnover'].mean():.4f}",
        f"- total transaction cost, RL - {base}: {traj['tc_gap_rl_base'].mean():+.3f}% of equity per trajectory", "",
        "## VR advantage decomposition", "",
        "VR_RL - VR_base = -(var(d) + 2 cov(r_base, d)) / var(r_anchor), d = r_RL - r_base. var(d) >= 0 always "
        "costs RL; RL can only win when its deviation from the baseline offsets the baseline's residual risk "
        "(2cov < 0).", "",
        f"- mean var(d)/var_u {traj['dec_var_d'].mean():.3f}, mean 2cov/var_u {traj['dec_2cov'].mean():.3f}",
        f"- RL wins VR on {int((traj['adv_vr'] > 0).sum())}/{len(traj)} trajectories ({n_var_win} with 2cov < 0)", "",
        "## Trajectory conditions most associated with the VR advantage", "",
        "Spearman rho of condition vs VR advantage (positive rho: RL gains on the baseline as the condition grows); "
        "`bh_q`: Benjamini-Hochberg q over all conditions of this outcome.", "",
        vr_assoc[["label", "kind", "spearman_rho", "p_value", "bh_q"]].assign(
            spearman_rho=vr_assoc["spearman_rho"].map(fmt), p_value=vr_assoc["p_value"].map(fmt_p),
            bh_q=vr_assoc["bh_q"].map(fmt_p)).to_markdown(index=False), "",
        "Spearman rho for every outcome (rows: condition, columns: outcome advantage):", "",
        assoc.pivot(index="label", columns="outcome", values="spearman_rho")
        .reindex([CONDITIONS[c][0] for c in CONDITIONS]).map(fmt).to_markdown(), "",
        "## Quartiles of the most associated conditions", "",
    ]
    for col in top_conditions:
        q = quartiles[quartiles["condition"] == col]
        report += [f"### {CONDITIONS[col][0]} ({CONDITIONS[col][1]})", "",
                   q[["quartile", "condition_mean", "n", "base_vr", "rl_vr", "adv_vr", "rl_win_rate_vr",
                      "adv_vol", "adv_annret", "adv_sharpe"]]
                   .assign(**{c: q[c].map(fmt) for c in ["condition_mean", "base_vr", "rl_vr", "adv_vr",
                                                          "rl_win_rate_vr", "adv_vol", "adv_annret", "adv_sharpe"]})
                   .to_markdown(index=False), ""]
    report += ["## Step-level state bins", "",
               "Steps pooled over trajectories, binned by quantiles of the state; pooled VR = 1 - sum of squared "
               "(trajectory-demeaned) hedged returns / the anchor's in the bin. `share_of_anchor_var`: how much of "
               "the total anchor variance falls in the bin.", ""]
    for col, (lab, kind) in STEP_BIN_VARS.items():
        sb = step_bins[step_bins["variable"] == col]
        report += [f"### {lab} ({kind})", "",
                   sb[["bin", "n_steps", "alpha_star", "alpha_base", "alpha_rl", "pooled_vr_base", "pooled_vr_rl",
                       "pooled_vr_adv", "share_of_anchor_var", "tc_base_bp", "tc_rl_bp"]]
                   .assign(**{c: sb[c].map(fmt) for c in ["alpha_star", "alpha_base", "alpha_rl", "pooled_vr_base",
                                                           "pooled_vr_rl", "pooled_vr_adv", "share_of_anchor_var",
                                                           "tc_base_bp", "tc_rl_bp"]})
                   .to_markdown(index=False), ""]
    report += ["## Trajectory bins of ex-post path measures", "",
               f"Trajectories binned into {args.n_bins} quantile bins of each measure; hedge ratios and pooled VR over "
               "the steps of the bin's trajectories. `test_partition_value`: the measure on the historical test "
               "partition (orange line in fig_step_bins.png).", ""]
    for col in TRAJ_BIN_VARS:
        tb = traj_bins[traj_bins["variable"] == col]
        report += [f"### {CONDITIONS[col][0]} (historical test partition: {fmt(tb['test_partition_value'].iloc[0])})", "",
                   tb[["bin", "n_traj", "bin_mean", "alpha_base", "alpha_rl", "pooled_vr_base", "pooled_vr_rl",
                       "pooled_vr_adv", "rl_win_rate_vr"]]
                   .assign(**{c: tb[c].map(fmt) for c in ["bin_mean", "alpha_base", "alpha_rl", "pooled_vr_base",
                                                           "pooled_vr_rl", "pooled_vr_adv", "rl_win_rate_vr"]})
                   .to_markdown(index=False), ""]

    ### 8. figures ###########################################################
    plt.rcParams.update({"figure.dpi": args.dpi, "savefig.dpi": args.dpi, "font.size": 9})
    c_base, c_rl, c_u = "dimgray", "steelblue", "black"

    # 8a. paired outcomes: baseline (x) vs RL (y), colored by realized anchor vol
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for ax, (metric, direction) in zip(axes.flat, OUTCOMES):
        key = OUTCOME_KEY[metric]
        x, y = traj[f"base_{key}"], traj[f"rl_{key}"]
        ax.vlines(x, traj[f"rl_{key}_min"], traj[f"rl_{key}_max"], color="lightgray", lw=0.8, zorder=1)
        sc = ax.scatter(x, y, c=traj["real_vol_a"], cmap="viridis", s=18, zorder=2)
        lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
        ax.plot(lims, lims, "--", color="gray", lw=0.8)
        win = (traj[f"adv_{key}"] > 0).mean()
        ax.set_title(f"{METRIC_LABEL.get(metric, metric)} ({'higher' if direction > 0 else 'lower'} better)\n"
                     f"RL better on {win:.0%} of trajectories", fontsize=9)
        ax.set_xlabel(base)
        ax.set_ylabel(f"{rl_label} (seed mean, whiskers: seed range)")
    fig.colorbar(sc, ax=axes, label="realized anchor vol (ann.)", shrink=0.6)
    fig.suptitle(f"{rl_label} vs {base}: per-trajectory outcomes (synthetic test)", fontsize=11)
    fig.savefig(os.path.join(output_dir, "fig_paired_outcomes.png"))
    plt.close(fig)

    # 8b. VR advantage vs the most associated conditions, with quartile means
    n_top = len(top_conditions)
    ncols = 3
    nrows = int(np.ceil(n_top / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows), constrained_layout=True, squeeze=False)
    for ax, col in zip(axes.flat, top_conditions):
        colors = np.where(traj["adv_vr"] > 0, c_rl, "indianred")
        ax.scatter(traj[col], traj["adv_vr"], c=colors, s=16, alpha=0.8)
        q = quartiles[quartiles["condition"] == col]
        ax.plot(q["condition_mean"], q["adv_vr"], "o-", color="black", ms=4, lw=1, label="quartile mean")
        ax.axhline(0, color="gray", lw=0.8)
        r = vr_assoc.set_index("condition").loc[col]
        ax.set_title(f"{CONDITIONS[col][0]} [{CONDITIONS[col][1]}]\nrho={r['spearman_rho']:.2f}, q={r['bh_q']:.3f}",
                     fontsize=9)
        ax.set_ylabel(f"VR advantage ({rl_label} - {base})")
    for ax in list(axes.flat)[n_top:]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Hedge-effectiveness advantage vs trajectory conditions (blue: RL better, red: baseline better)",
                 fontsize=11)
    fig.savefig(os.path.join(output_dir, "fig_conditions_vr.png"))
    plt.close(fig)

    # 8c. trajectory bins of the ex-post measures: hedge ratios (top) and pooled VR (bottom),
    # orange line: the historical test partition's value of the measure
    nvar = len(TRAJ_BIN_VARS)
    fig, axes = plt.subplots(2, nvar, figsize=(3.6 * nvar, 6.4), constrained_layout=True, sharey="row")
    for j, col in enumerate(TRAJ_BIN_VARS):
        tb = traj_bins[traj_bins["variable"] == col]
        test_value = test_conditions[col] if test_conditions is not None else None
        title = CONDITIONS[col][0] + (f"\nhistorical test = {test_value:.3f}" if test_value is not None else "")
        axes[0, j].plot(tb["bin_mean"], tb["alpha_base"], "s-", color=c_base, ms=4, label=base)
        axes[0, j].plot(tb["bin_mean"], tb["alpha_rl"], "o-", color=c_rl, ms=4, label=rl_label)
        axes[0, j].plot(tb["bin_mean"], tb["action_base"], ":", color=c_base, lw=1.2, label=rf"{base} $\alpha^{{\mathrm{{req}}}}$")
        axes[0, j].plot(tb["bin_mean"], tb["action_rl"], ":", color=c_rl, lw=1.2, label=rf"{rl_label} $\alpha^{{\mathrm{{req}}}}$")
        axes[0, j].set_title(title, fontsize=9)
        axes[1, j].plot(tb["bin_mean"], tb["pooled_vr_base"], "s-", color=c_base, ms=4, label=base)
        axes[1, j].plot(tb["bin_mean"], tb["pooled_vr_rl"], "o-", color=c_rl, ms=4, label=rl_label)
        axes[1, j].axhline(0, color="gray", lw=0.8)
        axes[1, j].set_xlabel(f"bin mean ({args.n_bins} quantile bins of trajectories)", fontsize=8)
        if test_value is not None:
            for ax in axes[:, j]:
                ax.axvline(test_value, color="darkorange", ls="--", lw=1.2, label="hist. test stats")
                # keep the marker in view when the test value lies outside the synthetic bins
                lo, hi = min(tb["bin_mean"].min(), test_value), max(tb["bin_mean"].max(), test_value)
                pad = 0.08 * (hi - lo)
                ax.set_xlim(lo - pad, hi + pad)
    axes[0, 0].set_ylabel(r"mean hedge ratio $\alpha^{\mathrm{exe}}$")
    axes[1, 0].set_ylabel("pooled VR in bin")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Hedge ratios and pooled VR by ex-post path measures (synthetic trajectories)", fontsize=11)
    fig.savefig(os.path.join(output_dir, "fig_step_bins.png"))
    plt.close(fig)

    # 8d. RL vs baseline hedge ratio over all steps
    fig, ax = plt.subplots(figsize=(5.6, 4.5), constrained_layout=True)
    hb = ax.hexbin(steps["alpha_base"], steps["alpha_rl"], gridsize=45, cmap="Blues", mincnt=1, bins="log")
    lims = [min(steps["alpha_base"].min(), steps["alpha_rl"].min()), max(steps["alpha_base"].max(), steps["alpha_rl"].max())]
    ax.plot(lims, lims, "--", color="gray", lw=0.8, label="y = x")
    ax.plot(lims, policy.intercept + policy.slope * np.array(lims), "-", color="darkorange", lw=1,
            label=f"OLS slope {policy.slope:.2f}")
    ax.set_xlabel(f"{base} hedge ratio")
    ax.set_ylabel(f"{rl_label} hedge ratio (seed mean)")
    ax.legend(fontsize=7)
    fig.colorbar(hb, ax=ax, label="steps (log)")
    ax.set_title(f"{base} vs. RL Ratios over all synth. steps", fontsize=11)
    fig.savefig(os.path.join(output_dir, "fig_vs_alphas.png"))
    plt.close(fig)

    # 8e. VR advantage decomposition per trajectory, sorted by advantage
    order = traj["adv_vr"].sort_values().index
    xs = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.bar(xs, -traj.loc[order, "dec_var_d"], color="indianred", label="-var(d)/var_u (always <= 0)")
    ax.bar(xs, -traj.loc[order, "dec_2cov"], bottom=np.where(-traj.loc[order, "dec_2cov"] > 0, 0,
                                                             -traj.loc[order, "dec_var_d"]),
           color="steelblue", alpha=0.8, label="-2cov(r_base, d)/var_u")
    ax.plot(xs, traj.loc[order, "adv_vr"], "k.", ms=5, label="VR advantage (sum)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(xs, [str(i) for i in order], fontsize=6, rotation=90)
    ax.set_xlabel("synthetic trajectory (sorted by VR advantage)")
    ax.set_ylabel(f"VR_RL - VR_{base}")
    ax.legend(fontsize=7)
    ax.set_title(f"VR advantage decomposition, d = r_RL - r_{base}", fontsize=10)
    fig.savefig(os.path.join(output_dir, "fig_variance_decomposition.png"))
    plt.close(fig)

    # 8f. case studies: largest wins either way
    k = args.n_case_studies
    cases = list(traj["adv_vr"].sort_values(ascending=False).index[:k]) + list(traj["adv_vr"].sort_values().index[:k])
    fig, axes = plt.subplots(3, len(cases), figsize=(3.4 * len(cases), 8), constrained_layout=True,
                             sharex=True, squeeze=False)
    for j, i in enumerate(cases):
        st = steps[steps["traj"] == i]
        ax = axes[0, j]
        ax.plot(st["step"], np.cumprod(1 + st["ret_a"]), color=c_u, lw=1, label="anchor")
        ax.plot(st["step"], np.cumprod(1 + st["ret_s"]), color="darkorange", lw=1, label="secondary")
        winner = rl_label if traj.loc[i, "adv_vr"] > 0 else base
        ax.set_title(f"traj {i}: {winner} better\nVR {base} {traj.loc[i, 'base_vr']:.2f} / RL {traj.loc[i, 'rl_vr']:.2f}",
                     fontsize=9)
        ax = axes[1, j]
        rl_seed_alpha = st[[f"alpha_rl_{s}" for s in seeds]].to_numpy()
        ax.fill_between(st["step"], rl_seed_alpha.min(axis=1), rl_seed_alpha.max(axis=1), color=c_rl, alpha=0.25)
        ax.plot(st["step"], st["alpha_rl"], color=c_rl, lw=1, label=rl_label)
        ax.plot(st["step"], st["alpha_base"], color=c_base, lw=1, label=base)
        ax.axhline(traj.loc[i, "expost_alpha"], color="gray", ls=":", lw=1, label="ex-post optimal")
        ax = axes[2, j]
        ax.plot(st["step"], st["sq_u"].cumsum(), color=c_u, lw=1, label="anchor")
        ax.plot(st["step"], st["sq_base"].cumsum(), color=c_base, lw=1, label=base)
        ax.plot(st["step"], st["sq_rl"].cumsum(), color=c_rl, lw=1, label=rl_label)
        ax.set_xlabel("step")
    axes[0, 0].set_ylabel("price (rebased)")
    axes[1, 0].set_ylabel("hedge ratio alpha")
    axes[2, 0].set_ylabel("cumulative squared deviation")
    for r in range(3):
        axes[r, 0].legend(fontsize=7)
    fig.suptitle(f"Case studies: {k} largest RL wins (left), {k} largest {base} wins (right), by VR", fontsize=11)
    fig.savefig(os.path.join(output_dir, "fig_case_studies.png"))
    plt.close(fig)

    ### 9. write report / console summary ####################################
    report += ["## Artifacts", ""] + [f"- `{name}`" for name in sorted(set(os.listdir(output_dir)) | {"report.md"})]
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write("\n".join(report) + "\n")

    with pd.option_context("display.width", 240, "display.max_columns", None):
        print("\nPaired outcomes (advantage > 0: RL better)")
        print(paired[["metric", "baseline_mean", "rl_mean", "adv_mean", "adv_ci_low", "adv_ci_high", "adv_median",
                      "rl_win_rate", "wilcoxon_p"]].to_string(index=False, float_format=lambda v: f"{v: .4f}"))
        print("\nTop conditions vs VR advantage (Spearman)")
        print(vr_assoc[["label", "kind", "spearman_rho", "p_value", "bh_q"]].head(args.top_conditions)
              .to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    print(f"\nArtifacts in {output_dir}:")
    for name in sorted(os.listdir(output_dir)):
        print(f"  {os.path.join(output_dir, name)}")


if __name__ == "__main__":
    main()
