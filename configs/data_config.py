from dataclasses import dataclass
from typing import Callable, Optional
from rl_dcc_hedge.utils.data import resampled_weekly_returns_factory
import os

@dataclass(kw_only=True)
class SeedDataConfig:
    dataset_name: str = "SPY_XLU"
    """the name of the dataset to be used for training and evaluation"""
    datasets_root: str = "./datasets"
    """the root directory where datasets are stored"""
    tics: tuple | list = ("SPY", "XLU")
    """the list of stock tickers"""
    anchor_asset: str = "SPY"
    """Which one of the tics to use as the anchor asset"""
    df_transformation: Callable = resampled_weekly_returns_factory("W-WED")
    """transformation function s.t. func(df) -> transformed_df """
    df_obs_freq: str = "W"
    """the frequency of observations"""
    periods: int = 52
    """periods per year for annualization (e.g. 52 for W, 252 for B, 365 for D)"""

    dataset_start_date: str = "2006-01-01"
    """start date of the dataset, inclusive"""
    dataset_end_date: str = "2026-01-01"
    """end date of the dataset, exclusive"""

    modeling_end_date: str = "2022-01-01"
    """exclusive upper bound for statistical model fitting, weekly sampling may move the date"""

    testing_start_date: str = "2022-01-01"
    """inclusive lower bound for historical RL testing, weekly sampling may move the date"""
    testing_end_date: str = "2026-01-01"
    """exclusive upper bound for historical RL testing, weekly sampling may move the date"""

    def prime_for_r(self):
        self.tics = list(self.tics)
        return self

@dataclass(kw_only=True)
class SimDataConfig:
    sim_steps: int = 209
    """number of timesteps to simulate for each trajectory"""
    m_sim: int = int(1e4)
    """number of trajectories to simulate"""
    n_start: int = 1
    """number of burn-in steps to discard for each simulation"""
    train_sim_seed: int = 60
    """training simulation runs seed, make sure [train_sim_seed, train_sim_seed+eval_m_sim] and [eval_sim_seed, eval_sim_seed+eval_m_sim] do not intersect"""

    eval_sim_steps: int = 209
    """number of timesteps to simulate for each eval trajectory"""
    eval_m_sim: int = 64
    """number of eval trajectories to simulate, make sure (eval_m_sim + eval_seed) < seed to prevent shared simulation seeds between train and eval partitions"""
    eval_n_start: int = 1
    """number of burn-in steps to discard for each simulation"""
    eval_sim_seed: int = 10065
    """evaluation simulation runs seed, make sure [train_sim_seed, train_sim_seed+eval_m_sim] and [eval_sim_seed, eval_sim_seed+eval_m_sim] do not intersect"""

    test_sim_steps: int = 209
    """number of timesteps to simulate for each eval trajectory"""
    test_m_sim: int = 64
    """number of eval trajectories to simulate, make sure (eval_m_sim + eval_seed) < seed to prevent shared simulation seeds between train and eval partitions"""
    test_n_start: int = 1
    """number of burn-in steps to discard for each simulation"""
    test_sim_seed: int = 10265
    """evaluation simulation runs seed, make sure [train_sim_seed, train_sim_seed+eval_m_sim] and [eval_sim_seed, eval_sim_seed+eval_m_sim] do not intersect"""

    num_bootstrapped_models: int = 32
    """number of bootstrapped models for the domain randomized synthetic test data, bootstrap sample attempt k (1-based) uses seed test_sim_seed+test_m_sim+k-1"""
    b_m_sim: int = 64
    """number of synthetic test trajectories to simulate per bootstrapped model, accepted model i (1-based) uses seeds test_sim_seed+test_m_sim+i*b_m_sim+[0, b_m_sim), resimulated trajectories use seeds from test_sim_seed+test_m_sim+(num_bootstrapped_models+1)*b_m_sim on"""

    # bootstrap guardrails (r/fit.r generate_bootstrapped_synth_test_data), rejected bootstrap samples / refits are
    # replaced by the next attempt and logged to <bootstrap_traj_dir>/bootstrap_rejections.csv
    b_max_attempts: int = 64
    """bootstrap samples drawn at most to get num_bootstrapped_models accepted models, needs num_bootstrapped_models <= b_max_attempts <= b_m_sim"""
    b_max_traj_retries: int = 10
    """resimulations of a rejected synthetic test trajectory before its bootstrapped model is rejected"""
    b_max_abs_return_sds: float = 20.0
    """bootstrap sample / trajectory check: max |weekly log-return| in seed return standard deviations"""
    b_min_sd_ratio: float = 1/3
    """bootstrap sample check: lower bound of the sample return sd / seed return sd"""
    b_max_sd_ratio: float = 3.0
    """bootstrap sample check: upper bound of the sample return sd / seed return sd"""
    b_bound_tol: float = 1e-4
    """refit check: a parameter within b_bound_tol * max(1, |bound|) of an optimizer bound rejects the refit"""
    b_max_beta: float = 0.995
    """refit check: upper bound of the eGARCH |beta1|"""
    b_min_shape: float = 3.0
    """refit check: lower bound of the univariate sstd shape (the simulations draw from the joint mvt, see b_min_mshape)"""
    b_min_skew: float = 0.5
    """refit check: lower bound of the univariate sstd skew"""
    b_max_skew: float = 2.0
    """refit check: upper bound of the univariate sstd skew"""
    b_max_dcc_persistence: float = 0.995
    """refit check: upper bound of dcca1 + dccb1"""
    b_min_dcca: float = 1e-4
    """refit check: lower bound of dcca1"""
    b_min_mshape: float = 4.1
    """refit check: lower bound of the joint mvt shape"""
    b_max_unc_corr: float = 0.99
    """refit check: upper bound of the |unconditional correlation|"""
    b_min_uncvol_ratio: float = 0.5
    """refit check: lower bound of the unconditional vol / seed unconditional vol, per asset"""
    b_max_uncvol_ratio: float = 2.0
    """refit check: upper bound of the unconditional vol / seed unconditional vol, per asset"""
    b_max_unc_corr_diff: float = 0.4
    """refit check: upper bound of |unconditional correlation - seed unconditional correlation|"""
    b_resid_pval: float = 0.01
    """refit soft flag: residual Ljung-Box / Li-McLeod p-value below which the model is flagged (never rejected)"""
    b_max_condstd_ratio: float = 5.0
    """trajectory check: upper bound of the max condstd / max seed in-sample condstd"""
    b_max_cum_return_sds: float = 8.0
    """trajectory check: upper bound of |cumulative log-return - seed drift| in seed return sds * sqrt(steps)"""
    b_min_innov_sd: float = 1/3
    """trajectory check: lower bound of the sd of the innovations filtered through the seed model"""
    b_max_innov_sd: float = 3.0
    """trajectory check: upper bound of the sd of the innovations filtered through the seed model"""
    b_warn_rejection_rate: float = 0.25
    """warn when more than this share of the bootstrap attempts is rejected"""


    startMethod: str = "unconditional"
    """one of ['unconditional','sample'] corresponding to [start fresh, use seed data tail] when initializing simulations"""

    mc_cores: int = 8
    """CPU cores to use for the simulation"""

    def prime_for_r(self):
        return self

@dataclass(kw_only=True)
class DataConfig(SeedDataConfig, SimDataConfig):
    def prime_for_r(self):
        return super().prime_for_r()
