from dataclasses import dataclass
import datetime
from functools import partial
from typing import Optional, Tuple

from configs.data_config import DataConfig
from configs.rl_config import RLExpConfig
from rl_dcc_hedge.baseline_strategies import TheoreticalMinVar, CostAwareMeanVarHedge, StaticOLSHedge, PeriodicOLSHedge, RollingOLSHedge
from rl_dcc_hedge.utils.portfolio_metrics import annual_returns, cumulative_returns, student_t_cvar
import os
import pandas as pd
import quantstats as qs

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _fitting_log_returns(cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """(anchor, secondary) weekly log returns of the modeling period (seed df rows before modeling_end_date)"""
    seed_df = pd.read_parquet(os.path.join(cfg.datasets_root, cfg.dataset_name, "seed_df.parquet"))
    tics = [tic.replace("^", "") for tic in cfg.tics] # seed df convention, see DataHandler
    anchor = cfg.anchor_asset.replace("^", "")
    secondary = tics[0] if tics[1] == anchor else tics[1]
    return seed_df.loc[seed_df.index < cfg.modeling_end_date, [anchor, secondary]]

FITTING_LOG_RETURNS = _fitting_log_returns()

KNOWN_BASELINE_STRATEGIES = {
	"MV": TheoreticalMinVar(),
	"MRR-C_10": CostAwareMeanVarHedge(lam=10),
    "MRR-C_100": CostAwareMeanVarHedge(lam=100),
    "MRR-C_1000": CostAwareMeanVarHedge(lam=1000),
    "OLS-static": StaticOLSHedge(FITTING_LOG_RETURNS),
    "OLS-R416-P52": RollingOLSHedge(FITTING_LOG_RETURNS, window=416, rebalance_period=52),
}

## SPY min vol checkpoints
AGENT_CHECKPOINTS = {
    "log_return:2.0,vol:0.22":[
        (1, "./results/cleanrl_runs/2026-10-27_01-51-18/best_cleanrl_model_Annual_Volatility_min.pt"),
        (10, "./results/cleanrl_runs/2026-10-27_02-04-44/best_cleanrl_model_Annual_Volatility_min.pt"),
        (42, "./results/cleanrl_runs/2026-10-27_02-17-50/best_cleanrl_model_Annual_Volatility_min.pt"),
        (137, "./results/cleanrl_runs/2026-10-27_02-30-52/best_cleanrl_model_Annual_Volatility_min.pt"),
    ],
    "log_return:1.2,vol:0.22":[
        (1, "./results/cleanrl_runs/2026-10-27_03-36-24/best_cleanrl_model_Annual_Volatility_min.pt"),
        (10, "./results/cleanrl_runs/2026-10-27_03-49-26/best_cleanrl_model_Annual_Volatility_min.pt"),
        (42, "./results/cleanrl_runs/2026-10-27_04-02-27/best_cleanrl_model_Annual_Volatility_min.pt"),
        (137, "./results/cleanrl_runs/2026-10-27_04-15-29/best_cleanrl_model_Annual_Volatility_min.pt"),
    ],
    "log_return:0.8,vol:0.22":[
        (1, "./results/cleanrl_runs/2026-10-27_05-20-01/best_cleanrl_model_Annual_Volatility_min.pt"),
        (10, "./results/cleanrl_runs/2026-10-27_05-32-57/best_cleanrl_model_Annual_Volatility_min.pt"),
        (42, "./results/cleanrl_runs/2026-10-27_05-45-49/best_cleanrl_model_Annual_Volatility_min.pt"),
        (137, "./results/cleanrl_runs/2026-10-27_05-58-45/best_cleanrl_model_Annual_Volatility_min.pt"),
    ],
    "log_return:0.4,vol:0.22":[
        (1, "./results/cleanrl_runs/2026-10-26_20-09-47/best_cleanrl_model_Annual_Volatility_min.pt"),
        (10, "./results/cleanrl_runs/2026-10-26_20-23-14/best_cleanrl_model_Annual_Volatility_min.pt"),
        (42, "./results/cleanrl_runs/2026-10-26_20-37-20/best_cleanrl_model_Annual_Volatility_min.pt"),
        (137, "./results/cleanrl_runs/2026-10-26_20-50-57/best_cleanrl_model_Annual_Volatility_min.pt"),
    ],
    "log_return:0.0,vol:0.22":[
        (1, "./results/cleanrl_runs/2026-10-26_21-59-15/best_cleanrl_model_Annual_Volatility_min.pt"),
        (10, "./results/cleanrl_runs/2026-10-26_22-12-42/best_cleanrl_model_Annual_Volatility_min.pt"),
        (42, "./results/cleanrl_runs/2026-10-26_22-26-18/best_cleanrl_model_Annual_Volatility_min.pt"),
        (137, "./results/cleanrl_runs/2026-10-26_22-40-05/best_cleanrl_model_Annual_Volatility_min.pt"),
    ],
}

OBS_FREQ = "W"
"""observation frequency of the evaluated trajectories (weekly)"""

PERIODS_PER_YEAR = 52
"""annualization factor matching OBS_FREQ (52 weekly periods per year)"""

PERIOD_LABELS = {52: "Weekly", 252: "Daily", 365: "Daily", 12: "Monthly"}


def make_test_stat_metrics(periods: int = PERIODS_PER_YEAR, rf: float = RLExpConfig.annual_financing_rate) -> dict:
    """TEST_STAT_METRICS for per-period returns observed ``periods`` times a year.

    The annualized metrics bind ``periods`` explicitly (quantstats defaults to
    252 daily periods): Annual Return / Calmar use years = n_obs / periods,
    Annual Volatility and the Sharpe / Sortino ratios scale by sqrt(periods),
    and the annual ``rf`` is de-annualized to the per-period rate by
    quantstats via the same ``periods``. VaR / CVaR / Skew / Tail Ratio /
    Max Drawdown are per-period (here weekly) quantities and are not annualized.
    """
    period_label = PERIOD_LABELS.get(periods, f"Per-period (1/{periods} yr)")
    return {
        "Annual Return": partial(annual_returns, periods=periods),
        "Cumulative Returns": cumulative_returns,
        "Annual Volatility": partial(qs.stats.volatility, periods=periods),
        "Sharpe Ratio": partial(qs.stats.sharpe, rf=rf, periods=periods),
        "Calmar Ratio": partial(qs.stats.calmar, periods=periods),
        "Max Drawdown": qs.stats.max_drawdown,
        "Sortino Ratio": partial(qs.stats.sortino, rf=rf, periods=periods),
        "Skew": qs.stats.skew,
        "Tail Ratio": qs.stats.tail_ratio,
        f"{period_label} VaR (95%)": qs.stats.value_at_risk,
        "CVaR (95%)": student_t_cvar,  # Student-t CVaR, (nu, loc, scale) MLE-fitted per trajectory
    }


TEST_STAT_METRICS = make_test_stat_metrics()
"""weekly-return metrics (periods = PERIODS_PER_YEAR); run_experiment_tests.py rebuilds them from args.periods"""



@dataclass
class ExpTestsConfig(RLExpConfig):
    """Configuration of the run_experiment_tests.py checkpoint/baseline evaluation.

    Inherits RLExpConfig for env/agent-construction parity with run_rl.py
    (start_capital, commission_fee_pct, reward_*, margin rates, verify_mu,
    dist_df, tics, agent_type, seed, cuda, ...); the training-only fields are
    unused by the evaluation harness.
    """

    experiments_root: str = "./results/experiments"
    """root directory of the experiment outputs; artifacts go to <experiments_root>/<experiment_run_name or the config timestamp>"""

    experiment_run_name: Optional[str] = None
    """name of this experiment run directory; None = the module-level timestamp of this config module"""

    baselines: Tuple[str, ...] = tuple(KNOWN_BASELINE_STRATEGIES.keys())
    """baseline strategy names to evaluate (keys of KNOWN_BASELINE_STRATEGIES)"""

    df_obs_freq: str = OBS_FREQ
    """observation frequency of the trajectories (weekly); sets the dummy dates of the synthetic trajectories"""

    periods: int = PERIODS_PER_YEAR
    """periods per year for annualizing the weekly-return metrics (52 for W)"""

    wandb: bool = False
    """never used: run_experiment_tests.py does not initialize wandb (override of the RLExpConfig default)"""