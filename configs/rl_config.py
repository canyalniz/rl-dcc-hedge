from dataclasses import dataclass, asdict, fields, make_dataclass
import datetime
from typing import Dict, Optional, Tuple

from .data_config import DataConfig

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

@dataclass
class RLConfig:
    num_iterations: int = 100
    """number of training iterations to step through, may be modified if all_traj is set to True"""
    learning_rate: float = 0.004
    """the learning rate of the optimizer"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.9
    """the discount factor gamma"""
    gae_lambda: float = 0.97
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    reward_clip: Tuple[Optional[float], Optional[float]] = (None, None)
    """the clipping range for rewards, specified as a tuple (min, max)"""
    mask_autoreset_transitions: bool = True
    """if toggled, rollout slots produced by the vector env's NEXT_STEP autoreset are excluded from the PPO update: the step taken right after an episode ends discards the action and returns (reset obs, r=0), so the stored sample pairs the terminal observation with a never-executed action and a zero reward; disabling restores the vanilla CleanRL behavior"""
    normalize_obs: bool = False
    """Whether to apply the gym.wrappers.NormalizeObservation wrapper"""
    scale_features: bool = True
    """Whether to apply the FeatureScaler wrapper: fixed log z-score of condstd and z-score of condcorr in the observation, then clipping of all features to +-feature_clip (raw features used by the rewards are untouched)"""
    condstd_log_mu: Tuple[float, float] = (-3.863, -3.904)
    """mean of log(condstd) [anchor, secondary], estimated on the SPY_XLU train_simulation_60_209_10000 data (rows 0-208); recompute when the model is re-fitted"""
    condstd_log_sd: Tuple[float, float] = (0.382, 0.243)
    """std of log(condstd) [anchor, secondary], same source as condstd_log_mu"""
    condcorr_mu: float = 0.492
    """mean of condcorr, same source as condstd_log_mu"""
    condcorr_sd: float = 0.149
    """std of condcorr, same source as condstd_log_mu"""
    feature_clip: float = 5.0
    """FeatureScaler clipping bound applied to every feature after scaling"""


    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    total_timesteps: int = 0
    """total number of timesteps to be taken over the training run (computed in runtime)"""

@dataclass
class AgentConfig:
    agent_type: str = "HedgingAgent"
    """the type of agent"""
    agent_checkpoint_path: Optional[str] = None
    """the path to the agent checkpoint for loading a pre-trained agent"""

@dataclass
class EnvConfig:
    obs_dim: int = 0
    """environment observation vector dimension"""
    action_dim: int = 0
    """environment action vector dimension"""
    num_envs: int = 8
    """the number of parallel training environments"""
    train_env_type: str = "MT_HedgeEnv"
    """rl training environment"""
    eval_env_type: str = "MT_HedgeEnv"
    """rl evaluation environment"""
    test_env_type: str = "ST_HedgeEnv"
    """rl testing environment"""
    start_capital: float = 1e4
    """initial investment capital"""
    reward_scaling: float = 20
    """coefficient with which to scale the reward"""
    reward_components: str = "log_return:0.4,vol:0.22,alpha_penalty:0.05"
    """a list of comma separated reward components each as <component_name>:<component_weight>"""
    commission_fee_pct: float = 0.001
    """commision fee percentage to be used when calculating transaction costs"""
    annual_financing_rate: float = 0.02
    """annual cash rate applied by the environments: positive cash accrues it as risk-free interest while negative (borrowed) cash pays it as financing cost (equal-rates idealization); 0.0 disables accrual. A nonzero rate requires datetime sampling_times (both ST and MT envs already use them)"""
    norm_gamma: Optional[float] = None
    """gamma for environment reward normalization"""
    abs_ratio_bound: float = 2.5
    """action absolute value bound"""
    im: float = 0.5
    """initial margin rate"""
    mm_long: float = 0.25
    """long maintenance margin rate"""
    mm_short: float = 0.30
    """short maintenance margin rate"""
    verify_mu: Tuple[float, float] = (0.001440, 0.001407)
    """fitted model drift constants [anchor, secondary]; used both for data-alignment verification and as the drift input of the cvar_95 reward component"""
    dist_df: Optional[float] = 5.348540
    """Student-t degrees of freedom of the fitted innovation distribution; feeds the cvar_95 reward component together with verify_mu (None: no distribution constants are passed to the environments)"""


@dataclass
class RLExpConfig(RLConfig, AgentConfig, EnvConfig, DataConfig):
    """A dataclass that combines RLArgs, EnvArgs, AgentArgs, and EncoderArgs for a comprehensive configuration of the RL experiment."""
    
    run_name: str = timestamp
    """the name of this experiment"""
    cleanrl_root: str = "./results/cleanrl_runs"
    """path to cleanrl root directory on disk"""
    run_dir: str = ""
    """path to run directory (computed at runtime)"""

    records_dir: str = "./results/test_records/"
    """the directory to save training records"""

    train_only: bool = False
    """if toggled, only training will be performed (no testing)"""
    test_only: bool = False
    """if toggled, only testing will be performed (no training)"""
    run_baselines: bool = False
    """if toggled, only baseline strategies will be run"""
    tracked_evaluation: bool = True
    """if toggled, a tracked evaluation run will be carried out on all evaluation environments with the best agent checkpoint before testing begins"""

    seed: int = 137
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""

    wandb: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rl-dcc-hedge"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    evaluation_interval: int = 4
    """the interval (in training iterations) for evaluating the agent's performance"""
    evaluation_burn_in: int = 2
    """the number of training iterations to wait before starting evaluating the agent"""
    eval_metrics: str = "CVaR (95%):max"
    """comma separated list of evaluation metrics to track, each as <metric_name>:<max|min>; a best checkpoint is saved for each tracked metric (metric names must be STAT_METRICS keys)"""

    all_traj: bool = False
    """if set to true sets num_iterations to the minimum s.t. num_iterations * num_steps >= n_sim * m_sim"""

    save_model: bool = True
    """whether to save the best checkpoints (one per tracked eval metric) during training"""

    also_final_iteration: bool = True
    """if toggled together with save_model, a checkpoint of the agent at the end of training is saved (and tested) in addition to the metric based checkpoints"""