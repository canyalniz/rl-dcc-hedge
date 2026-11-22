"""Checkpoint / baseline experiment tests on the synthetic test trajectories and the historical test partition.

Evaluates

  * the unhedged anchor reference (rl_dcc_hedge.baseline_strategies.UnhedgedAnchor:
    the environment's no-trade book realized through the same mechanics and
    trajectory format as everything else),
  * the baseline strategies named in KNOWN_BASELINE_STRATEGIES
    (configs/experiment_tests_config.py), and
  * every (tag, seed) agent checkpoint of AGENT_CHECKPOINTS
    (configs/experiment_tests_config.py) with id_tag = f"{tag}_{seed}" and
    reward components f"{tag},alpha_penalty:0.05" reproducing the sweep's
    training mixes (the reward components only affect the recorded reward
    columns, never the deterministic policy actions),

on

  * ALL synthetic test trajectories (the test-simulation parquet exposed by
    data_handler.test_sim_save_path; one MT_HedgeEnv per trajectory for the
    anchor and the baselines, one shared SyncVectorEnv stack +
    RLExperiment.evaluate -- the deterministic tracked_evaluation parity
    path -- for the agents), and
  * the single historical test partition episode (ST_HedgeEnv, mirroring
    RLExperiment.test's env construction but without its rl_test_results.csv
    side effect and with this script's trajectories/ naming), with its last
    step discarded so it runs for as many steps as the synthetic episodes.

and produces, under <experiments_root>/<experiment_run_name or timestamp>/:

  trajectories/<data>_<id_tag>.csv   per-run trajectory frames
  metrics.csv                        long format [id_tag, data, metric_name, metric_value]

where data is one of synth_test_0..synth_test_{N-1}, test_partition and the
metric rows are TEST_STAT_METRICS (rebuilt for args.periods observations per
year: weekly returns annualized with 52 periods by default) plus, for every hedged run, the two
hedge-effectiveness (HE) metrics against the unhedged anchor's returns of the
SAME dataset (aligned on the common realized prefix):

  HE (Ederington R2)        squared Pearson correlation of hedged vs unhedged returns
  HE (Variance Reduction)   1 - Var(hedged)/Var(unhedged), ddof=1 variances

The anchor itself contributes portfolio-metric rows but no HE rows.

Notes
-----
* All inherited RLExpConfig fields are accepted for env/data-construction
  parity with run_rl.py; training-only fields (num_iterations, learning_rate,
  PPO hyperparameters, ...) are unused by this script.
* wandb is never initialized (the inherited wandb flag is overridden to False
  by ExpTestsConfig).
* The datasets are assumed materialized; DataHandler(args, init_test_only=True)
  would regenerate missing test files through the R pipeline when absent.
* Trajectory file names sanitize [^A-Za-z0-9._-] to '_' (agent id tags contain
  ':' and ','); the id_tag column of metrics.csv stays verbatim.
"""
from __future__ import annotations

import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import tyro

from rl_dcc_hedge.baseline_strategies import BaselineStrategy, UnhedgedAnchor
from rl_dcc_hedge.data_handlers import DataHandler
from rl_dcc_hedge.environments import MT_HedgeEnv, ST_HedgeEnv
from rl_dcc_hedge.rl_experiments import ENV_TYPES_DICT, RLExperiment, _make_env, _feature_scaler_kwargs
from rl_dcc_hedge.utils.portfolio_metrics import compute_portfolio_metrics

from configs.experiment_tests_config import (
    AGENT_CHECKPOINTS,
    ExpTestsConfig,
    KNOWN_BASELINE_STRATEGIES,
    make_test_stat_metrics,
    timestamp,
)
from run_rl import _normalize_wandb_bool_args

ANCHOR_ID = "unhedged_anchor"
"""id_tag of the unhedged anchor reference (str(UnhedgedAnchor()))"""

SYNTH_DATA_PREFIX = "synth_test"
"""data label prefix of the synthetic test trajectories: synth_test_{i}"""

TEST_DATA_LABEL = "test_partition"
"""data label of the historical test partition"""

HE_EDERINGTON_METRIC = "HE (Ederington R2)"
HE_VARIANCE_REDUCTION_METRIC = "HE (Variance Reduction)"
HE_METRICS = (HE_EDERINGTON_METRIC, HE_VARIANCE_REDUCTION_METRIC)


def _sanitize(name: str) -> str:
    """Filesystem-safe stand-in for ``name`` in trajectory file names.

    Agent id tags contain ':' and ',' (e.g. 'log_return:0.5,cvar_95:0.1_1');
    anything outside [A-Za-z0-9._-] becomes '_'. The id_tag column of
    metrics.csv keeps the verbatim tag.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def hedge_effectiveness(r_unhedged: np.ndarray, r_hedged: np.ndarray) -> Tuple[float, float]:
    """The two hedge-effectiveness variants of ``r_hedged`` against ``r_unhedged``.

    Both are computed on the common realized prefix of the two return series
    (early-terminated episodes are aligned on their overlap):

      HE (Ederington R2)        squared Pearson correlation of the two series
      HE (Variance Reduction)   1 - Var(r_hedged)/Var(r_unhedged), ddof=1

    Returns (he_ederington, he_variance_reduction); NaN values with a printed
    warning on degenerate cases (fewer than 3 common steps, zero unhedged
    variance, zero hedged variance for the correlation variant).
    """
    n = min(len(r_unhedged), len(r_hedged))
    r_u = np.asarray(r_unhedged[:n], dtype=np.float64)
    r_h = np.asarray(r_hedged[:n], dtype=np.float64)
    if n < 3:
        print(f"WARNING: only {n} common return steps; hedge effectiveness undefined -> NaN")
        return np.nan, np.nan
    var_u = float(np.var(r_u, ddof=1))
    var_h = float(np.var(r_h, ddof=1))
    if not var_u > 0:
        print("WARNING: unhedged return variance is zero/unstable; hedge effectiveness undefined -> NaN")
        return np.nan, np.nan
    he_variance_reduction = 1.0 - var_h / var_u
    corr = np.nan
    if var_h > 0:
        corr = float(np.cov(r_u, r_h, ddof=1)[0, 1]) / (np.sqrt(var_u) * np.sqrt(var_h))
    else:
        print("WARNING: hedged return variance is zero; HE (Ederington R2) undefined -> NaN")
    he_ederington = corr**2 if np.isfinite(corr) else np.nan
    return he_ederington, he_variance_reduction


def run_on_test_partition(
    args: ExpTestsConfig,
    data_handler: DataHandler,
    test_data: Dict[str, pd.DataFrame],
    device: torch.device,
    baseline_strategy: Optional[BaselineStrategy] = None,
    agent_checkpoint_path: Optional[str] = None,
) -> pd.DataFrame:
    """One episode on the historical test partition, mirroring RLExperiment.test.

    Either ``baseline_strategy`` (a raw ST_HedgeEnv, exactly RLExperiment.test's
    baseline path) or ``agent_checkpoint_path`` (a SyncVectorEnv +
    RLExperiment.evaluate, exactly RLExperiment.test's agent path) must be
    provided. Unlike RLExperiment.test this writes no rl_test_results.csv row
    and no trajectory of its own: the caller saves the returned frame.

    ``test_data`` must already be verified (ST_HedgeEnv.verify_data_alignment)
    and innov-clipped; the reward components of ``args`` at call time are
    baked into the constructed environment.
    """
    anchor_asset, secondary_asset = data_handler.anchor_asset, data_handler.secondary_asset

    test_env_params = {
        "features": test_data["features"],
        "prices": test_data["prices"],
        "sampling_times": test_data["sampling_times"],
        "num_features": test_data["features"].shape[1],
        "tics": [tic.replace("^", "") for tic in args.tics],
        "initial_amount": args.start_capital,
        "reward_scaling": args.reward_scaling,
        "transaction_cost_rate": args.commission_fee_pct,
        "annual_financing_rate": args.annual_financing_rate,
        "reward_components": args.reward_components,
        "float_dtype": np.float32,
        "abs_ratio_bound": args.abs_ratio_bound,
        "im": args.im,
        "mm_l": args.mm_long,
        "mm_s": args.mm_short,
        "starting_prices": data_handler.starting_prices,
    }
    if args.dist_df is not None:
        # cvar_95 support: fitted innovation distribution constants (ST
        # falls back to the repo-wide default feature indices)
        test_env_params["dist_params"] = {"df": args.dist_df, "mu": tuple(args.verify_mu)}

    env_type = ENV_TYPES_DICT[args.test_env_type]

    if baseline_strategy is not None:
        test_env = env_type["cls"](**test_env_params)
        return baseline_strategy.run(test_env)

    if agent_checkpoint_path is None:
        raise ValueError("run_on_test_partition needs a baseline_strategy or an agent_checkpoint_path")

    test_env = gym.vector.SyncVectorEnv(
        [_make_env(
            env_id=env_type["id"],
            norm_gamma=args.norm_gamma,
            reward_clip=args.reward_clip,
            env_kwargs=test_env_params,
            normalize_obs=args.normalize_obs,
            feature_scaler_kwargs=_feature_scaler_kwargs(args),
        )]
    )
    args.obs_dim = test_env.single_observation_space.shape[0]
    args.action_dim = test_env.single_action_space.shape[0]
    args.agent_checkpoint_path = agent_checkpoint_path
    agent = RLExperiment._make_agent(args)
    trajectory_df_tuple = RLExperiment.evaluate(agent=agent, env=test_env, device=device)
    return trajectory_df_tuple[0]  # there is a single env


def main() -> None:
    args: ExpTestsConfig = tyro.cli(
        ExpTestsConfig,
        args=_normalize_wandb_bool_args(sys.argv[1:]),
        description=(
            "Evaluate the unhedged anchor, the KNOWN_BASELINE_STRATEGIES baselines and the "
            "AGENT_CHECKPOINTS agent checkpoints on all synthetic test trajectories and the "
            "historical test partition, with hedge-effectiveness metrics against the anchor"
        ),
    )

    ### 1. input validation ###################################################
    baseline_names = [name.strip() for name in args.baselines]
    unknown_baselines = [name for name in baseline_names if name not in KNOWN_BASELINE_STRATEGIES]
    if unknown_baselines:
        raise ValueError(
            f"Unknown baseline strategy name(s) {unknown_baselines}; available: {list(KNOWN_BASELINE_STRATEGIES)}"
        )
    if len(set(baseline_names)) != len(baseline_names):
        raise ValueError(f"Duplicate baseline names in {baseline_names!r}")
    if ANCHOR_ID in baseline_names:
        raise ValueError(f"Baseline name {ANCHOR_ID!r} collides with the reserved anchor id_tag")

    # every (tag, seed) checkpoint combo is run as its own id_tag
    agent_runs: List[Tuple[str, int, str, str]] = []  # (tag, seed, checkpoint path, id_tag)
    seen_pairs = set()
    seen_id_tags = {ANCHOR_ID} | set(baseline_names)
    for tag, seed_path_pairs in AGENT_CHECKPOINTS.items():
        for seed, checkpoint_path in seed_path_pairs:
            pair = (tag, int(seed))
            if pair in seen_pairs:
                raise ValueError(f"Duplicate (tag, seed) checkpoint pair {pair!r} in AGENT_CHECKPOINTS")
            seen_pairs.add(pair)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Agent checkpoint not found: {checkpoint_path!r}")
            id_tag = f"{tag}_{seed}"
            if id_tag in seen_id_tags:
                raise ValueError(f"Duplicate id_tag {id_tag!r}; id tags must be unique")
            seen_id_tags.add(id_tag)
            agent_runs.append((tag, int(seed), checkpoint_path, id_tag))

    ### 2. output directory ###################################################
    run_name = args.experiment_run_name or timestamp
    run_dir = os.path.join(args.experiments_root, run_name)
    args.run_dir = run_dir
    trajectories_dir = os.path.join(run_dir, "trajectories")
    os.makedirs(trajectories_dir, exist_ok=True)
    print(f"Experiment run directory: {run_dir}")

    ### 3. determinism (mirrors RLExperiment.determinism) #####################
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    ### 4. data ###############################################################
    print(
        "Instantiating DataHandler (the dataset is assumed materialized; a missing test file "
        "may trigger the R-side data pipeline)..."
    )
    data_handler = DataHandler(args, init_test_only=True)

    # synthetic test data: no loader method exists, read the parquet directly
    print(f"Loading synthetic test data: {data_handler.test_sim_save_path}")
    synth_df = pd.read_parquet(data_handler.test_sim_save_path)

    # infer the simulation matrix layout: trajectory ids from the {i}_0_prices
    # columns, step count from the row count (rows = sim_steps + n_start)
    traj_ids = sorted(int(col.rsplit("_", 2)[0]) for col in synth_df.columns if col.endswith("_0_prices"))
    if not traj_ids:
        raise ValueError(f"No '{{i}}_0_prices' columns found in the synthetic test data; not an MT_HedgeEnv simulation matrix?")
    if traj_ids != list(range(len(traj_ids))):
        raise ValueError(
            f"Synthetic test data trajectory ids must be contiguous 0..N-1 for per-trajectory environments, got {traj_ids[:5]}...{traj_ids[-5:]}"
        )
    n_traj = len(traj_ids)
    n_rows = len(synth_df)
    args.eval_m_sim = args.eval_num_envs = n_traj
    args.eval_sim_steps = n_rows - args.test_n_start
    if args.eval_num_envs > 64:
        print(f"Eval number of SyncVectorEnv was set to eval_m_sim ({args.eval_m_sim}), this many evaluation environments will be created. Might have performance implications.")

    # warn-only alignment verification, on the raw frame (before the innov
    # clipping breaks the model identity on purpose)
    try:
        MT_HedgeEnv.verify_data_matrix_alignment(synth_df, mu=args.verify_mu)  # only checks traj_idx=0
    except ValueError as err:
        print(f"WARNING: synthetic test data alignment verification failed ({err}); continuing with a warning only")

    print("Normalizing innov by clipping")
    innov_cols = [col for col in synth_df.columns if "_innov" in col]
    synth_df[innov_cols] = synth_df[innov_cols].clip(-4.5, 4.5)

    # the test-simulation file's own row-0 prices anchor the initial book
    synth_starting_prices = {
        "anchor": float(synth_df["0_0_prices"].iloc[0]),
        "secondary": float(synth_df["0_1_prices"].iloc[0]),
    }
    print(
        f"Synthetic test data: {n_traj} trajectories x {args.eval_sim_steps} simulated steps "
        f"({n_rows} rows), starting prices {synth_starting_prices}"
    )

    # historical test partition (mirrors RLExperiment.test's preparation)
    test_data = data_handler.load_test_partition_data(get_in_parts=True)
    test_sampling_times = test_data["sampling_times"]
    anchor_asset, secondary_asset = data_handler.anchor_asset, data_handler.secondary_asset
    ST_HedgeEnv.verify_data_alignment(
        prices=test_data["prices"].values[:-1,],
        returns=test_data["log_returns"].values[:-1,],
        innov=test_data["features"][[f"{anchor_asset}_innov", f"{secondary_asset}_innov"]].values[:-1,],
        condstd=test_data["features"][[f"{anchor_asset}_condstd", f"{secondary_asset}_condstd"]].values[:-1,],
        condcorr=test_data["features"]["condcorr"].values[:-1,],
        mu=np.array(args.verify_mu),
    )
    # discard the last row (one step) of the test partition so the historical
    # episode runs for the same number of steps as the synthetic ones
    # (len(sampling_times) - 1 = 208 by default, see RLExperiment._eval_env_params);
    # verified above on the full partition, trimmed only afterwards
    test_data = {
        key: (value.iloc[:-1].copy() if isinstance(value, (pd.DataFrame, pd.Series)) else value[:-1])
        for key, value in test_data.items()
    }
    test_sampling_times = test_data["sampling_times"]
    print(
        f"Historical test partition trimmed by one step to {len(test_sampling_times) - 1} steps "
        f"(last date {test_sampling_times[-1]}), matching the synthetic episodes"
    )
    print("Normalizing test innov by clipping")
    test_innov_cols = [col for col in test_data["features"].columns if "_innov" in col]
    test_data["features"][test_innov_cols] = test_data["features"][test_innov_cols].clip(-4.5, 4.5)

    data_labels = [f"{SYNTH_DATA_PREFIX}_{i}" for i in traj_ids] + [TEST_DATA_LABEL]

    ### 5. run registry / helpers #############################################
    returns_by_run: Dict[Tuple[str, str], np.ndarray] = {}  # (id_tag, data label) -> portfolio simple returns

    def register_trajectory(traj_df: pd.DataFrame, id_tag: str, data_label: str) -> None:
        """Saves the trajectory frame under trajectories/ and registers its returns."""
        save_path = os.path.join(trajectories_dir, f"{_sanitize(data_label)}_{_sanitize(id_tag)}.csv")
        traj_df.to_csv(save_path, index=False)
        returns_by_run[(id_tag, data_label)] = traj_df["portfolio_simple_returns"].to_numpy(dtype=float)

    def run_baseline_on_synth(strategy: BaselineStrategy) -> List[pd.DataFrame]:
        """One raw MT_HedgeEnv per synthetic trajectory, mirroring RLExperiment.test's baseline path."""
        frames = []
        for i in traj_ids:
            env = MT_HedgeEnv(
                **RLExperiment._eval_env_params(
                    args=args,
                    data=synth_df,
                    traj_indices=[i],
                    starting_prices=synth_starting_prices,
                )
            )
            frames.append(strategy.run(env))
        return frames

    ### 6. phase A: the unhedged anchor reference #############################
    anchor = UnhedgedAnchor()
    print(f"[{SYNTH_DATA_PREFIX}] running the unhedged anchor on {n_traj} trajectories...")
    for i, traj_df in zip(traj_ids, run_baseline_on_synth(anchor)):
        register_trajectory(traj_df, ANCHOR_ID, f"{SYNTH_DATA_PREFIX}_{i}")
    print(f"[{TEST_DATA_LABEL}] running the unhedged anchor...")
    register_trajectory(
        run_on_test_partition(args, data_handler, test_data, device, baseline_strategy=anchor),
        ANCHOR_ID,
        TEST_DATA_LABEL,
    )

    ### 7. phase B: the baseline strategies ###################################
    # the KNOWN_BASELINE_STRATEGIES instances are stateless across runs and are reused
    for name in baseline_names:
        print(f"[{SYNTH_DATA_PREFIX}] running baseline {name!r}...")
        for i, traj_df in zip(traj_ids, run_baseline_on_synth(KNOWN_BASELINE_STRATEGIES[name])):
            register_trajectory(traj_df, name, f"{SYNTH_DATA_PREFIX}_{i}")
        print(f"[{TEST_DATA_LABEL}] running baseline {name!r}...")
        register_trajectory(
            run_on_test_partition(args, data_handler, test_data, device, baseline_strategy=KNOWN_BASELINE_STRATEGIES[name]),
            name,
            TEST_DATA_LABEL,
        )

    ### 8. phase C: the agent checkpoints #####################################
    for tag, seed, checkpoint_path, id_tag in agent_runs:
        print(f"[agents] evaluating {id_tag!r} ({checkpoint_path})...")
        # reproduces the sweep's training reward mix; only the recorded reward
        # columns of the trajectory frames depend on it
        args.reward_components = f"{tag},alpha_penalty:0.05"
        args.agent_checkpoint_path = checkpoint_path

        # synthetic trajectories: one shared env stack covering all of them
        # (evaluate() resets it; with eval_m_sim == eval_num_envs env i covers
        # exactly trajectory i, the i-th singleton shard), the deterministic
        # tracked_evaluation parity path
        synth_envs = RLExperiment._setup_envs(
            args=args,
            data_handler=data_handler,
            partition="eval",
            data=synth_df,
            strict_data_verification=False,
        )
        args.obs_dim = synth_envs.single_observation_space.shape[0]
        args.action_dim = synth_envs.single_action_space.shape[0]

        agent = RLExperiment._make_agent(args)
        traj_df_tuple = RLExperiment.evaluate(agent=agent, env=synth_envs, device=device)
        del agent
        del synth_envs  # drop the n_traj-env vector env

        for i, traj_df in zip(traj_ids, traj_df_tuple):
            register_trajectory(traj_df, id_tag, f"{SYNTH_DATA_PREFIX}_{i}")

        # historical test partition
        register_trajectory(
            run_on_test_partition(args, data_handler, test_data, device, agent_checkpoint_path=checkpoint_path),
            id_tag,
            TEST_DATA_LABEL,
        )

    ### 9. metrics ############################################################
    # metrics of the per-period (weekly by default) portfolio returns, annualized with args.periods
    test_stat_metrics = make_test_stat_metrics(periods=args.periods, rf=args.annual_financing_rate)
    print(f"\nComputing portfolio metrics ({args.df_obs_freq} observations, {args.periods} periods per year)...")

    # the historical test partition's own dates must match the annualization frequency
    test_spacing_days = pd.Series(pd.to_datetime(test_sampling_times)).diff().dt.days.median()
    expected_spacing_days = 365.25 / args.periods
    if not abs(test_spacing_days - expected_spacing_days) <= 0.25 * expected_spacing_days:
        print(
            f"WARNING: median test partition date spacing is {test_spacing_days} days but periods={args.periods} "
            f"implies ~{expected_spacing_days:.1f} days; annualized test metrics will be mis-scaled"
        )
    expected_synth_steps = args.eval_sim_steps - 1
    expected_test_steps = len(test_sampling_times) - 1

    def record_metric_rows(id_tag: str, data_label: str) -> None:
        """test_stat_metrics rows of one (id_tag, data) run, on its realized steps."""
        returns = returns_by_run[(id_tag, data_label)]
        if data_label == TEST_DATA_LABEL:
            sampling_index = test_sampling_times[1:]
            if len(returns) < len(sampling_index):
                print(
                    f"WARNING: test episode of {id_tag!r} terminated after {len(returns)}/{len(sampling_index)} "
                    f"steps; computing metrics on realized steps only"
                )
                sampling_index = sampling_index[: len(returns)]
        else:
            if len(returns) < expected_synth_steps:
                print(
                    f"WARNING: {data_label} episode of {id_tag!r} terminated after {len(returns)}/{expected_synth_steps} "
                    f"steps; computing metrics on realized steps only"
                )
            # dummy dates at the observation frequency (weekly by default), as portfolio_metrics_from_traj_df_tuple does
            sampling_index = pd.date_range(start="2000-01-01", periods=len(returns), freq=args.df_obs_freq)
        returns_series = pd.Series(returns)
        returns_series.index = sampling_index
        perf = compute_portfolio_metrics(returns_series, metrics=test_stat_metrics, periods=args.periods)
        for metric_name, metric_value in perf.items():
            metric_rows.append({
                "id_tag": id_tag,
                "data": data_label,
                "metric_name": metric_name,
                "metric_value": metric_value,
            })

    hedged_ids = baseline_names + [id_tag for _, _, _, id_tag in agent_runs]
    all_ids = [ANCHOR_ID] + hedged_ids

    metric_rows: List[Dict[str, object]] = []
    for id_tag in all_ids:
        for data_label in data_labels:
            record_metric_rows(id_tag, data_label)

    # hedge effectiveness of every hedged run against the anchor of the SAME
    # dataset, aligned on the common realized prefix (the anchor gets no HE rows)
    print("Computing hedge-effectiveness metrics against the unhedged anchor...")
    for id_tag in hedged_ids:
        for data_label in data_labels:
            he_ederington, he_variance_reduction = hedge_effectiveness(
                returns_by_run[(ANCHOR_ID, data_label)],
                returns_by_run[(id_tag, data_label)],
            )
            metric_rows.append({
                "id_tag": id_tag,
                "data": data_label,
                "metric_name": HE_EDERINGTON_METRIC,
                "metric_value": he_ederington,
            })
            metric_rows.append({
                "id_tag": id_tag,
                "data": data_label,
                "metric_name": HE_VARIANCE_REDUCTION_METRIC,
                "metric_value": he_variance_reduction,
            })

    metrics_df = pd.DataFrame(metric_rows, columns=["id_tag", "data", "metric_name", "metric_value"])
    metrics_path = os.path.join(run_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    ### 10. console summary ###################################################
    synth_he_summary = (
        metrics_df[
            (metrics_df["data"] != TEST_DATA_LABEL) & (metrics_df["metric_name"].isin(HE_METRICS))
        ]
        .groupby(["id_tag", "metric_name"])["metric_value"]
        .mean()
        .unstack(level="metric_name")[list(HE_METRICS)]
    )
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print("\n" + "=" * 100)
        print("Mean hedge effectiveness across the synthetic test trajectories")
        print(synth_he_summary.to_string(float_format=lambda x: f"{x: .4f}"))
    print("\n" + "=" * 100)
    print(f"Strategies evaluated: 1 unhedged anchor + {len(baseline_names)} baselines + {len(agent_runs)} agents")
    print(
        f"Datasets: {n_traj} synthetic test trajectories ({os.path.basename(data_handler.test_sim_save_path)}) "
        f"+ the historical test partition"
    )
    print("Artifacts:")
    print(f"  {metrics_path} ({len(metrics_df)} rows)")
    print(f"  {trajectories_dir}/ ({len(os.listdir(trajectories_dir))} trajectory csvs)")


if __name__ == "__main__":
    main()
