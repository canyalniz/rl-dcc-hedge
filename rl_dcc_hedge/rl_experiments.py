# The actor-critic network (layer_init, HedgingAgent layout) in this file is adapted from
# CleanRL (https://github.com/vwxyzjn/cleanrl), used under the MIT License:
#
# MIT License
#
# Copyright (c) 2019 CleanRL developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
import re
import time
import os

import pandas as pd
import torch
from tensordict.tensordict import TensorDict
import numpy as np
import random

from typing import Optional, Dict

import gymnasium as gym
from gymnasium.envs.registration import register

from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("agg") # headless-safe: figures are only saved to disk
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from rl_dcc_hedge.environments import MT_HedgeEnv, ST_HedgeEnv
from rl_dcc_hedge.data_handlers import DataHandler
from rl_dcc_hedge.cleanrl_agents import HedgingAgent
from rl_dcc_hedge.utils.portfolio_metrics import compute_portfolio_metrics, portfolio_metrics_from_traj_df_tuple, STAT_METRICS
from rl_dcc_hedge.utils.data import two_step_aggregation, global_aggregation
from rl_dcc_hedge.baseline_strategies import BaselineStrategy

from configs.rl_config import RLExpConfig

AGENT_TYPES_DICT = {
    "HedgingAgent": HedgingAgent
}

ENV_TYPES_DICT = {
    "MT_HedgeEnv": {
        "cls": MT_HedgeEnv,
        "id": "rl_dcc_hedge/MT_HedgeEnv-v1"
    },
    "ST_HedgeEnv": {
        "cls": ST_HedgeEnv,
        "id": "rl_dcc_hedge/ST_HedgeEnv-v1"
    },
}

for env_type in ENV_TYPES_DICT.values():
    register(
        id=env_type['id'],
        entry_point=env_type['cls']
    )

class FeatureScaler(gym.ObservationWrapper):
    """
    Fixed (frozen-constant) scaling of the market features in the observation.

    The observation is [portfolio vars, features]; feature indices below are
    relative to the feature block, which sits at the end of the observation.
    Only the observation is transformed: the environment's raw features (read
    by e.g. the cvar_95 reward) are left untouched.

    - log_feature_indices: (log(x) - log_mu) / log_sd, e.g. conditional std
    - z_feature_indices: (x - z_mu) / z_sd, e.g. conditional correlation
    - every feature is then clipped to [-clip, clip]
    """
    def __init__(self, env, log_feature_indices, log_mu, log_sd, z_feature_indices, z_mu, z_sd, clip=5.0):
        super().__init__(env)
        offset = env.observation_space.shape[0] - env.unwrapped.num_features
        self._feature_slice = slice(offset, None)
        self._log_idx = np.asarray(log_feature_indices, dtype=int) + offset
        self._log_mu = np.asarray(log_mu, dtype=np.float64)
        self._log_sd = np.asarray(log_sd, dtype=np.float64)
        self._z_idx = np.asarray(z_feature_indices, dtype=int) + offset
        self._z_mu = np.asarray(z_mu, dtype=np.float64)
        self._z_sd = np.asarray(z_sd, dtype=np.float64)
        self._clip = clip

    def observation(self, observation):
        obs = np.array(observation, dtype=np.float64)
        obs[self._log_idx] = (np.log(obs[self._log_idx]) - self._log_mu) / self._log_sd
        obs[self._z_idx] = (obs[self._z_idx] - self._z_mu) / self._z_sd
        obs[self._feature_slice] = np.clip(obs[self._feature_slice], -self._clip, self._clip)
        return obs.astype(observation.dtype)


def _feature_scaler_kwargs(args: RLExpConfig) -> Optional[Dict]:
    """FeatureScaler kwargs for the DataHandler feature layout, None when args.scale_features is off."""
    if not args.scale_features:
        return None
    per_tic_tags = DataHandler.PER_TIC_FEATURES_COLUMN_TAGS
    standalone_tags = DataHandler.STANDALONE_FEATURES_COLUMN_TAGS
    num_tics = len(args.tics)
    # per-tic-major layout: [tic0 tags..., tic1 tags..., standalone tags...]
    condstd_idx = [j * len(per_tic_tags) + per_tic_tags.index("condstd") for j in range(num_tics)]
    condcorr_idx = [num_tics * len(per_tic_tags) + standalone_tags.index("condcorr")]
    return {
        "log_feature_indices": condstd_idx,
        "log_mu": args.condstd_log_mu,
        "log_sd": args.condstd_log_sd,
        "z_feature_indices": condcorr_idx,
        "z_mu": (args.condcorr_mu,),
        "z_sd": (args.condcorr_sd,),
        "clip": args.feature_clip,
    }

def _make_env(env_id, norm_gamma=None, reward_clip=(None,None), env_kwargs=None, normalize_obs=True, feature_scaler_kwargs=None):
    def thunk():
        # the passive env checker only validates the first reset/step but stays in
        # the wrapper stack of every call (~10% of the eval step time)
        env = gym.make(env_id, disable_env_checker=True, **(env_kwargs or {}))
        # env = gym.wrappers.RecordEpisodeStatistics(env)

        env = gym.wrappers.ClipAction(env)
        if feature_scaler_kwargs is not None:
            env = FeatureScaler(env, **feature_scaler_kwargs)
        if normalize_obs:
            env = gym.wrappers.NormalizeObservation(env)

        if norm_gamma:
            env = gym.wrappers.NormalizeReward(env, gamma=norm_gamma)
        
        if reward_clip[0] is not None or reward_clip[1] is not None:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, reward_clip[0], reward_clip[1]))
        
        return env

    return thunk

class RLExperiment(object):
    def __init__(self, args:RLExpConfig):
        self.args = args
        
        # set determinism for reproducibility
        self.determinism()

        # instantiate the data handler
        self._data_handler = DataHandler(args)

        self.run_dir = os.path.join(
            args.cleanrl_root,
            args.run_name
        )

        if args.wandb:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=args.run_name,
                monitor_gym=True,
                save_code=True,
            )

        self.writer = SummaryWriter(args.run_dir)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        self.envs = self._setup_envs(args=self.args, data_handler=self._data_handler, partition='train')
        self.eval_envs = self._setup_envs(args=self.args, data_handler=self._data_handler, partition='eval')

        observation_space = self.envs.single_observation_space
        action_space = self.envs.single_action_space

        assert isinstance(action_space, gym.spaces.Box), "only continuous action space is supported"

        self.args.obs_dim = observation_space.shape[0]
        self.args.action_dim = action_space.shape[0]

        self.agent = self._make_agent(args)
        
        # if args.wandb:
        #     wandb.watch(self.agent, log="all", log_freq=10)

        self.eval_metrics = self._parse_eval_metrics(args.eval_metrics)
        unknown_metrics = [name for name in self.eval_metrics if name not in STAT_METRICS]
        if unknown_metrics:
            raise ValueError(
                f"Unknown evaluation metric(s) {unknown_metrics} in args.eval_metrics={args.eval_metrics!r}. "
                f"Available metrics: {list(STAT_METRICS.keys())}"
            )
        # best value seen so far and checkpoint path, per tracked metric
        self.best_eval_metric_values = {
            metric: -np.inf if order == "max" else np.inf
            for metric, order in self.eval_metrics.items()
        }
        self.best_checkpoint_paths: Dict[str, Optional[str]] = {metric: None for metric in self.eval_metrics}
        # path of the final-iteration checkpoint saved by train() when
        # args.also_final_iteration and args.save_model are both set
        self.final_checkpoint_path: Optional[str] = None

    @staticmethod
    def _parse_eval_metrics(eval_metrics: str) -> Dict[str, str]:
        """
        Parses a comma separated list of '<metric_name>:<order>' entries into an
        ordered {metric_name: order} mapping, order being one of 'max' or 'min'.

        rpartition is used so metric names containing ':' stay valid.
        """
        if eval_metrics is None or not eval_metrics.strip():
            raise ValueError("eval_metrics must contain at least one '<metric_name>:<max|min>' entry")

        metrics: Dict[str, str] = {}
        for entry in eval_metrics.split(","):
            name, sep, order = entry.rpartition(":")
            name, order = name.strip(), order.strip().lower()
            if not sep or not name or not order:
                raise ValueError(
                    f"Malformed eval metric entry {entry!r}; expected '<metric_name>:<max|min>'"
                )
            if order not in ("max", "min"):
                raise ValueError(f"Invalid eval metric order {order!r} for metric {name!r}; expected 'max' or 'min'")
            if name in metrics:
                raise ValueError(f"Duplicate eval metric {name!r} in eval_metrics")
            metrics[name] = order
        return metrics

    @staticmethod
    def metric_tag(metric_name: str, order: str) -> str:
        """Filesystem-safe tag identifying a tracked metric and its order, e.g. ('Sortino Ratio', 'max') -> 'Sortino_Ratio_max'"""
        return f"{re.sub(r'[^A-Za-z0-9]+', '_', metric_name).strip('_')}_{order}"

    @staticmethod
    def _make_agent(args: RLExpConfig):
        agent_cls = AGENT_TYPES_DICT.get(args.agent_type, None)
        if agent_cls is None:
            raise ValueError(f"Unknown agent type: {args.agent_type}. Available types: {list(AGENT_TYPES_DICT.keys())}")
        agent = agent_cls(args=args)
        # load from checkpoint if a checkpoint is provided
        if args.agent_checkpoint_path is not None:
            print(f"Loading agent weights from {args.agent_checkpoint_path}")
            device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
            checkpoint = torch.load(args.agent_checkpoint_path, map_location=device)
            agent.load_state_dict(checkpoint)
            print("Agent weights loaded successfully.")
        return agent

    @staticmethod
    def _eval_env_params(
        args: RLExpConfig,
        data: pd.DataFrame,
        traj_indices,
        starting_prices: Dict[str, float],
        record_trajectory: bool = True,
    ) -> dict:
        """
        kwargs dict constructing one MT_HedgeEnv evaluation environment covering
        ``traj_indices`` of ``data`` (evaluation semantics: silent trajectory
        cycling, trajectory recording on by default).

        The innovation clipping of ``data`` stays with the callers (as in
        _setup_envs): the env reads the columns lazily at reset(), so the
        clipping only has to happen before the first episode runs.
        """
        env_params = {
            "data": data,
            "traj_indices": traj_indices,
            "sampling_times": pd.date_range(start='2000-01-01', periods=int(args.eval_sim_steps), freq=args.df_obs_freq).to_numpy(),
            "tics": [tic.replace("^","") for tic in args.tics],
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
            "starting_prices": starting_prices,
            "silent_traj_idx_reset": True,
            "per_tic_features_column_tags": DataHandler.PER_TIC_FEATURES_COLUMN_TAGS,
            "standalone_features_column_tags": DataHandler.STANDALONE_FEATURES_COLUMN_TAGS,
            "record_trajectory": record_trajectory,
        }
        if args.dist_df is not None:
            # cvar_95 support: fitted innovation distribution constants (MT
            # derives the condstd/condcorr feature indices from its tags)
            env_params["dist_params"] = {"df": args.dist_df, "mu": tuple(args.verify_mu)}
        return env_params

    @staticmethod
    def _setup_envs(
        args: RLExpConfig,
        data_handler: DataHandler,
        partition:str='train',
        data: Optional[pd.DataFrame] = None,
        strict_data_verification: bool = True,
    ):
        """
        Sets up a gymnasium.vector.SyncVectorEnv instance following args

        Parameters
        ----------
        partition:str
        One of ['train', 'eval']

        data: Optional[pd.DataFrame]
        DataFrame overriding the EVALUATION simulation data (partition='eval'
        only); when None the data handler loads the partition data as usual.
        Since the override may come from a different simulation seed file, the
        env starting prices are then derived from the data's first price row
        (0_0_prices / 0_1_prices) instead of data_handler.starting_prices so
        the MT_HedgeEnv starting-price assertion holds for any such file.

        strict_data_verification: bool
        Governs verify_data_matrix_alignment of a PROVIDED data frame: with
        the default True the verification raises on failure (identical to the
        historical behavior); with False it is skipped with a printed note,
        letting callers evaluate data whose alignment they verified themselves
        (e.g. on the raw frame before the innov clipping, as run_experiments.py
        does). The default-loaded partitions are always verified strictly,
        preserving run_rl.py semantics.
        """
        if partition.lower() in ['train', 'training']:
            partition = 'train'
        elif partition.lower() in ['eval', 'evaluation']:
            partition = 'eval'
        else:
            raise ValueError(
                f"RLExperiment._setup_envs expects a partition tag representing either train or eval, got {partition!r}"
            )

        data_override = data is not None
        if data_override and partition != 'eval':
            raise ValueError(
                "The data override of RLExperiment._setup_envs is only supported for the eval partition"
            )

        m_sim = int(args.m_sim if partition=='train' else args.eval_m_sim)
        sim_steps = int(args.sim_steps if partition=='train' else args.eval_sim_steps)
        num_envs = int(args.num_envs if partition=='train' else args.eval_num_envs)

        if m_sim % num_envs != 0:
            print(f"WARNING: The {partition} m_sim{m_sim} is not divisible by num_envs{num_envs}. This will cause an unbalanced distribution of trajectories among the training environments.")

        traj_idx_iterables = np.array_split(np.arange(m_sim), num_envs)

        if any(len(shard) == 0 for shard in traj_idx_iterables):
            raise ValueError(
                f"The {partition} m_sim ({m_sim}) is smaller than num_envs ({num_envs}); "
                "every environment must receive at least one trajectory index"
            )
        
        dummy_dates = pd.date_range(start='2000-01-01', periods=sim_steps, freq=args.df_obs_freq).to_numpy()
        
        # load data into memory
        if not data_override:
            data = data_handler.load_rl_training_data() if partition=='train' else data_handler.load_rl_evaluation_data()
            MT_HedgeEnv.verify_data_matrix_alignment(data, mu=args.verify_mu) # only checks traj_idx=0
        elif strict_data_verification:
            MT_HedgeEnv.verify_data_matrix_alignment(data, mu=args.verify_mu) # only checks traj_idx=0
        else:
            print(
                "NOTE: skipping the alignment verification of the provided evaluation data "
                "(strict_data_verification=False); the caller is responsible for verifying it "
                "(e.g. on the raw frame before the innov clipping)"
            )

        print("Normalizing innov by clipping")
        innov_cols = [
            col for col in data.columns if "_innov" in col
        ]
        data[innov_cols] = data[innov_cols].clip(-4.5,4.5)

        if partition == 'eval':
            if data_override:
                # custom-seed file: its own row-0 prices anchor the initial book
                starting_prices = {
                    "anchor": float(data["0_0_prices"].iloc[0]),
                    "secondary": float(data["0_1_prices"].iloc[0]),
                }
            else:
                starting_prices = data_handler.starting_prices
            env_params = [
                RLExperiment._eval_env_params(
                    args=args,
                    data=data,
                    traj_indices=traj_idx_iterables[i],
                    starting_prices=starting_prices,
                    record_trajectory=True,
                )
                for i in range(num_envs)
            ]
        else:
            env_params = [{
                "data": data,
                "traj_indices": traj_idx_iterables[i],
                "sampling_times": dummy_dates,
                "tics": [tic.replace("^","") for tic in args.tics],
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
                "silent_traj_idx_reset": False,
                "per_tic_features_column_tags": data_handler.PER_TIC_FEATURES_COLUMN_TAGS,
                "standalone_features_column_tags": data_handler.STANDALONE_FEATURES_COLUMN_TAGS,
                "record_trajectory": False,
            } for i in range(num_envs)]
            if args.dist_df is not None:
                # cvar_95 support: fitted innovation distribution constants (MT
                # derives the condstd/condcorr feature indices from its tags)
                for params in env_params:
                    params["dist_params"] = {"df": args.dist_df, "mu": tuple(args.verify_mu)}

        env_name = args.train_env_type if partition == 'train' else args.eval_env_type
        env_id = ENV_TYPES_DICT[env_name]['id']

        # env setup
        return gym.vector.SyncVectorEnv(
            [_make_env(env_id=env_id, norm_gamma=args.norm_gamma, reward_clip=args.reward_clip, env_kwargs=env_params[i], normalize_obs=args.normalize_obs, feature_scaler_kwargs=_feature_scaler_kwargs(args)) for i in range(num_envs)]
        )

    @staticmethod
    def evaluate(
        agent: torch.nn.Module,
        env: gym.vector.SyncVectorEnv,
        device: str
    ):
        agent.eval()
        agent.to(device)

        next_obs, _ = env.reset()
        next_obs = torch.Tensor(next_obs).to(device=device)

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(next_obs, deterministic=True)
            num_envs = env.num_envs
            finished = np.zeros(num_envs, dtype=bool)
            trajectory_df_tuple = [None] * num_envs

            while not finished.all():
                next_obs, reward, terminations, truncations, infos = env.step(action.cpu().numpy())
                episode_end = np.logical_or(terminations, truncations)

                for i in np.where(episode_end & ~finished)[0]:
                    # harvest BEFORE the next step() triggers the autoreset memory wipe
                    trajectory_df_tuple[i] = env.envs[i].get_wrapper_attr("get_trajectory_df")()

                finished |= episode_end
                if not finished.all():
                    next_obs = torch.Tensor(next_obs).to(device)
                    action, _, _, _ = agent.get_action_and_value(next_obs, deterministic=True)

        return tuple(trajectory_df_tuple)

    @classmethod
    def tracked_evaluation(
        cls,
        args: RLExpConfig,
        data_handler: DataHandler,
        device: str,
        artifacts_suffix: Optional[str] = None,
    ):
        """
        Runs one deterministic evaluation episode per evaluation environment and
        saves the per-environment artifacts to args.run_dir:

        - eval_trajectory_env_<i>[_<artifacts_suffix>].csv: the trajectory DataFrame returned by evaluate
        - eval_trajectory_env_<i>[_<artifacts_suffix>].png: the equity of the actual portfolio together
          with two hypothetical buy-and-hold benchmarks (100% anchor, and an
          equal-weights anchor/secondary portfolio held from the start), all in a
          single line graph on a common equity axis

        The agent weights are loaded from args.agent_checkpoint_path (via
        cls._make_agent), so a checkpoint must be provided for a trained agent.

        Parameters
        ----------
        args: RLExpConfig
        Experiment configuration (eval_* fields drive the environments)

        data_handler: DataHandler
        Data handler providing the evaluation simulation data

        device: str
        Torch device the agent runs on

        artifacts_suffix: Optional[str]
        When provided (e.g. the metric tag of the checkpoint under evaluation),
        appended to the artifact file names so that tracked evaluations of
        different checkpoints do not overwrite each other

        Returns
        -------
        tuple of pd.DataFrame
        The trajectory DataFrame of each evaluation environment
        """
        os.makedirs(args.run_dir, exist_ok=True)
        artifact_tag = f"_{artifacts_suffix}" if artifacts_suffix else ""

        # evaluation envs record trajectories (record_trajectory=True in _setup_envs)
        eval_envs = cls._setup_envs(args=args, data_handler=data_handler, partition='eval')

        args.obs_dim = eval_envs.single_observation_space.shape[0]
        args.action_dim = eval_envs.single_action_space.shape[0]

        # creates the agent and loads the checkpoint
        agent = cls._make_agent(args)

        trajectory_df_tuple = cls.evaluate(agent=agent, env=eval_envs, device=device)

        # _setup_envs hands env i the i-th shard of
        # np.array_split(np.arange(eval_m_sim), eval_num_envs), and evaluate()
        # runs exactly one episode per env starting from the shard's first
        # trajectory; env i's harvested episode is therefore shard[i][0]. The
        # price paths are recovered from the simulation data because the env's
        # own `prices` attribute may already have advanced past the autoreset.
        traj_idx_shards = np.array_split(np.arange(int(args.eval_m_sim)), int(args.eval_num_envs))
        eval_data = data_handler.load_rl_evaluation_data()

        anchor_asset, secondary_asset = data_handler.anchor_asset, data_handler.secondary_asset

        for i, trajectory_df in enumerate(trajectory_df_tuple):
            # save the trajectory DataFrame
            traj_save_path = os.path.join(args.run_dir, f"eval_trajectory_env_{i}{artifact_tag}.csv")
            trajectory_df.to_csv(traj_save_path, index=False)

            # price paths of the evaluated trajectory (col 0 = anchor, col 1 = secondary,
            # matching the [anchor, secondary] subject ordering of the simulation data)
            traj_idx = int(traj_idx_shards[i][0])
            anchor_prices = eval_data[f"{traj_idx}_0_prices"].to_numpy(dtype=float)
            secondary_prices = eval_data[f"{traj_idx}_1_prices"].to_numpy(dtype=float)

            # end-of-step equity; prepend the initial equity (args.start_capital, the
            # all-anchor initial book) so all curves cover the full episode
            equity = np.concatenate(
                ([float(args.start_capital)], trajectory_df["post_evo_equity"].to_numpy(dtype=float))
            )

            n_points = len(equity)
            if len(anchor_prices) < n_points:
                print(
                    f"WARNING: evaluation env {i} recorded {n_points - 1} steps but the "
                    f"simulation data only holds {len(anchor_prices) - 1} evolution steps; "
                    "clipping the plot to the available data"
                )
                n_points = len(anchor_prices)
                equity = equity[:n_points]

            steps = np.arange(n_points)
            anchor_path = anchor_prices[:n_points]
            secondary_path = secondary_prices[:n_points]

            initial_equity = float(args.start_capital)

            # hypothetical (frictionless) buy-and-hold benchmarks:
            # 1) all equity in the anchor asset from the start
            anchor_only_equity = initial_equity * anchor_path / anchor_path[0]
            # 2) equal-weights portfolio (E0/2 per asset) formed at the start and held
            equal_weights_equity = (
                0.5 * initial_equity * (anchor_path / anchor_path[0]
                                         + secondary_path / secondary_path[0])
            )

            # single line graph on a single equity-scaled y axis common to all curves
            fig, ax = plt.subplots(figsize=(10, 5.5))

            ax.plot(steps, equity, color="#1f77b4", linewidth=1.8,
                    label="Agent portfolio (actual)")
            ax.plot(steps, anchor_only_equity, color="#ff7f0e", linewidth=1.8, linestyle="--",
                    label=f"100% {anchor_asset} buy-and-hold")
            ax.plot(steps, equal_weights_equity, color="#2ca02c", linewidth=1.8, linestyle="--",
                    label=f"50/50 {anchor_asset}/{secondary_asset} buy-and-hold")
            ax.set_xlabel("Episode step")
            ax.set_ylabel(f"Equity (initial = {initial_equity:,.0f})")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", frameon=True)

            ax.set_title(
                f"Evaluation environment {i} (trajectory {traj_idx}): "
                "actual vs hypothetical portfolio equity"
            )
            fig.tight_layout()

            plot_save_path = os.path.join(args.run_dir, f"eval_trajectory_env_{i}{artifact_tag}.png")
            fig.savefig(plot_save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            print(f"Evaluation env {i} (trajectory {traj_idx}): trajectory saved to {traj_save_path}, plot saved to {plot_save_path}")

        return trajectory_df_tuple

    @classmethod
    def test(
        cls,
        args: RLExpConfig,
        data_handler: DataHandler,
        device: str,
        baseline_strategy: Optional[BaselineStrategy] = None,
        strategy_name_suffix: Optional[str] = None,
    ):  
        # Load the RL test data
        test_data = data_handler.load_test_partition_data(get_in_parts=True)

        os.makedirs(args.records_dir, exist_ok=True)

        anchor_asset, secondary_asset = data_handler.anchor_asset, data_handler.secondary_asset
        ST_HedgeEnv.verify_data_alignment(
            prices=test_data["prices"].values[:-1,],
            returns=test_data["log_returns"].values[:-1,],
            innov=test_data["features"][[f"{anchor_asset}_innov", f"{secondary_asset}_innov"]].values[:-1,],
            condstd=test_data["features"][[f"{anchor_asset}_condstd", f"{secondary_asset}_condstd"]].values[:-1,],
            condcorr=test_data["features"]["condcorr"].values[:-1,],
            mu=np.array(args.verify_mu),
        )

        print("Normalizing test innov by clipping")
        innov_cols = [
            col for col in test_data["features"].columns if "_innov" in col
        ]
        test_data["features"][innov_cols] = test_data["features"][innov_cols].clip(-4.5,4.5)

        test_env_params = {
            "features": test_data['features'],
            "prices": test_data['prices'],
            "sampling_times": test_data['sampling_times'],
            "num_features": test_data['features'].shape[1],
            "tics": [tic.replace("^","") for tic in args.tics],
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

        env_id = ENV_TYPES_DICT[args.test_env_type]['id']

        if baseline_strategy is None:
            test_env = gym.vector.SyncVectorEnv([_make_env(env_id=env_id, norm_gamma=args.norm_gamma, reward_clip=args.reward_clip, env_kwargs=test_env_params, normalize_obs=args.normalize_obs, feature_scaler_kwargs=_feature_scaler_kwargs(args))])
            
            args.obs_dim = test_env.single_observation_space.shape[0]
            args.action_dim = test_env.single_action_space.shape[0]

            # creates agent and loads the checkpoint
            agent = cls._make_agent(args)
            trajectory_df_tuple = cls.evaluate(agent=agent, env=test_env, device=device)
            trajectory_df = trajectory_df_tuple[0] # there is a single env
            strategy_name = args.run_name if strategy_name_suffix is None else f"{args.run_name}_{strategy_name_suffix}"
        else:
            test_env = ENV_TYPES_DICT[args.test_env_type]['cls'](**test_env_params)
            trajectory_df = baseline_strategy.run(test_env)
            strategy_name = baseline_strategy.__str__()

        save_path = os.path.join(args.run_dir, f"test_trajectory_{strategy_name}.csv")
        trajectory_df.to_csv(save_path, index=False)
        portfolio_simple_returns = trajectory_df["portfolio_simple_returns"].values.copy()

        cls.record_metrics(
            portfolio_simple_returns=portfolio_simple_returns,
            args=args,
            sampling_times=test_data['sampling_times'],
            strategy_name=strategy_name
        )

        return trajectory_df

    @staticmethod
    def record_metrics(
        portfolio_simple_returns: np.typing.ArrayLike,
        args: RLExpConfig,
        sampling_times: np.ndarray,
        strategy_name: str = None
    ):
        testing_records_path = os.path.join(args.records_dir, "rl_test_results.csv")
        if os.path.exists(testing_records_path):
            testing_records_df = pd.read_csv(testing_records_path)
        else:
            testing_records_df = pd.DataFrame()

        sampling_times = sampling_times[1:]
        if len(portfolio_simple_returns) < len(sampling_times):
            print(f"WARNING: test episode terminated after {len(portfolio_simple_returns)}/{len(sampling_times)} steps; computing metrics on realized steps only")
            sampling_times = sampling_times[:len(portfolio_simple_returns)]
        portfolio_simple_returns = pd.Series(portfolio_simple_returns)
        portfolio_simple_returns.index = sampling_times

        perf_stats = compute_portfolio_metrics(portfolio_simple_returns, metrics=STAT_METRICS, periods=args.periods)
        perf_stats = perf_stats.to_frame().T

        perf_stats['Agent'] = strategy_name

        # Reorder columns to have 'Agent' first
        cols = ['Agent'] + [col for col in perf_stats.columns if col != 'Agent']
        perf_stats = perf_stats[cols]

        testing_records_df = pd.concat([testing_records_df, perf_stats])
        testing_records_df.to_csv(testing_records_path, index=False)
    
    def train(self) -> Dict[str, Optional[str]]:
        """
        Runs PPO training, saving a checkpoint per tracked evaluation metric
        (args.eval_metrics) whenever it improves.

        When args.save_model and args.also_final_iteration are both set, a
        checkpoint of the agent as of the end of training is additionally
        saved and its path stored in self.final_checkpoint_path.

        Returns
        -------
        dict of str -> str
        Mapping from tracked metric name to the path of its best checkpoint
        (metrics that never improved within the evaluation schedule map to None)
        """
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        # set agent to training mode
        self.agent.train()

        # move agent to device
        self.agent.to(self.device)

        # ALGO Logic: Storage setup
        obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        # M3: with gymnasium's NEXT_STEP autoreset (SyncVectorEnv default), the step taken
        # right after an episode ends discards the action, resets the env and returns
        # (reset obs, r=0.0, term=False, trunc=False). The sample stored at such a slot
        # pairs the terminal observation with a never-executed action; when
        # args.mask_autoreset_transitions is set we track the live slots and exclude
        # the phantom ones from the PPO update below.
        live_mask = torch.ones((self.args.num_steps, self.args.num_envs), dtype=torch.bool).to(self.device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)

        for iteration in range(1, self.args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.args.num_steps):
                global_step += self.args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                if self.args.mask_autoreset_transitions:
                    # this envs.step() call autoresets (and discards the action) exactly for
                    # the envs flagged in next_done, so those slots hold phantom samples
                    live_mask[step] = ~next_done.bool()

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

                # if "final_info" in infos:
                #     for info in infos["final_info"]:
                #         if info and "episode" in info:
                #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                #             self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                #             self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                # if "episode" in infos:
                #     for ep_return, ep_length, is_valid in zip(
                #         infos["episode"]["r"], infos["episode"]["l"], infos["episode"]["_r"]
                #     ):
                #         if is_valid:
                #             print(f"global_step={global_step}, episodic_return={ep_return:.2f}")
                #             self.writer.add_scalar("charts/episodic_return", ep_return, global_step)
                #             self.writer.add_scalar("charts/episodic_length", ep_length, global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch, keeping only live (non-phantom) samples when
            # mask_autoreset_transitions is toggled (M3 fix); slice(None) keeps every
            # slot, reproducing the vanilla CleanRL flatten
            b_live = live_mask.reshape(-1) if self.args.mask_autoreset_transitions else slice(None)
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)[b_live]
            b_logprobs = logprobs.reshape(-1)[b_live]
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)[b_live]
            b_advantages = advantages.reshape(-1)[b_live]
            b_returns = returns.reshape(-1)[b_live]
            b_values = values.reshape(-1)[b_live]

            # Optimizing the policy and value network
            b_inds = np.arange(b_obs.shape[0])
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_inds), self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if iteration > self.args.evaluation_burn_in and ((iteration - self.args.evaluation_burn_in) % self.args.evaluation_interval == 0):
                self.agent.eval() # Put agent in eval mode
                eval_trajectory_df_tuple = self.evaluate(agent=self.agent, env=self.eval_envs, device=self.device)
                self.agent.train()  # Switch back to training mode

                self.writer.add_scalar("eval/reward", two_step_aggregation(eval_trajectory_df_tuple, "reward"), global_step)
                
                self.writer.add_scalar("eval/action_max", two_step_aggregation(eval_trajectory_df_tuple, "action", agg_func_1=np.max), global_step)
                self.writer.add_scalar("eval/action_min", two_step_aggregation(eval_trajectory_df_tuple, "action", agg_func_1=np.min), global_step)

                self.writer.add_scalar("eval/leverage_init_max", two_step_aggregation(eval_trajectory_df_tuple, "post_tx_leverage", agg_func_1=np.max), global_step)
                self.writer.add_scalar("eval/leverage_fin_max", two_step_aggregation(eval_trajectory_df_tuple, "post_evo_leverage", agg_func_1=np.max), global_step)
                
                self.writer.add_scalar("eval/margin_calls", global_aggregation(eval_trajectory_df_tuple, "margin_calls", agg_func=np.sum) / self.args.eval_num_envs, global_step)

                self.writer.add_scalar("eval/total_transaction_cost", two_step_aggregation(eval_trajectory_df_tuple, "transaction_cost"), global_step)

                eval_perf_stats = portfolio_metrics_from_traj_df_tuple(
                    eval_trajectory_df_tuple,
                    metrics=STAT_METRICS,
                    df_obs_freq=self.args.df_obs_freq,
                    periods = self.args.periods
                )
                
                for metric_name, metric_value in eval_perf_stats.items():
                    self.writer.add_scalar(f"eval/{metric_name}", metric_value, global_step)

                # log all reward components (those columns with reward_component_{component_name})
                for col in eval_trajectory_df_tuple[0].columns:
                    if col.startswith("reward_component_"):
                        self.writer.add_scalar(f"eval/{col}", two_step_aggregation(eval_trajectory_df_tuple, col), global_step)

                if self.args.save_model:
                    # update the best checkpoint of every tracked metric
                    for metric_name, metric_order in self.eval_metrics.items():
                        current_eval_metric = eval_perf_stats[metric_name]
                        improved = (
                            current_eval_metric > self.best_eval_metric_values[metric_name]
                            if metric_order == "max"
                            else current_eval_metric < self.best_eval_metric_values[metric_name]
                        )
                        if improved:
                            self.best_eval_metric_values[metric_name] = current_eval_metric
                            # Save the model checkpoint for the best evaluation metric
                            model_path = os.path.join(
                                self.args.run_dir,
                                f"best_cleanrl_model_{self.metric_tag(metric_name, metric_order)}.pt"
                            )
                            self.best_checkpoint_paths[metric_name] = model_path
                            torch.save(self.agent.state_dict(), model_path)
                            print(f"New best {metric_name} ({metric_order}): {current_eval_metric:.4f}. Model saved to {model_path}")

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            self.writer.add_scalar("charts/training_iteration", iteration, global_step)
        
        print("FINISHED TRAINING...")

        if self.args.save_model and self.args.also_final_iteration:
            # checkpoint of the agent as of the end of training, saved in
            # addition to the metric based best checkpoints
            final_model_path = os.path.join(self.args.run_dir, "final_cleanrl_model.pt")
            torch.save(self.agent.state_dict(), final_model_path)
            self.final_checkpoint_path = final_model_path
            print(f"Final-iteration model saved to {final_model_path}")
        
        return self.best_checkpoint_paths

    def determinism(self):
        seed = self.args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
        torch.backends.cudnn.benchmark = False
    
