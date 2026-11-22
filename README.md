# rl-dcc-hedge: RL hedging under a DCC–eGARCH market model

rl-dcc-hedge trains PPO agents to hedge a long **SPY** position with **XLU** and compares
them against closed-form and regression-based hedging baselines. The agents train
on market paths simulated from a fitted multivariate GARCH model, not on resampled
history.

Here is how the pipeline fits together:

1. **Market model (R).** Weekly SPY/XLU log returns are modeled as
   DCC(1,1) with ARMA(0,0)–eGARCH(1,1) marginals, skewed-Student-t (`sstd`) marginal
   innovations and a multivariate-t (`mvt`) joint distribution (`rugarch`/`rmgarch`,
   called from Python via `rpy2`). The model is fitted on data before 2022-01-01.
   2022-01-01 – 2026-01-01 is held out as the historical test partition.
2. **Simulation.** The fitted model simulates training, evaluation and synthetic test
   trajectories. Each trajectory is filtered back through the
   model, so every step includes the conditional volatilities (`condstd`) and the
   conditional correlation (`condcorr`).
3. **Environment.** Gymnasium hedging environments (`ST_HedgeEnv`, `MT_HedgeEnv`)
   model a margin account. They include initial and maintenance margins, proportional
   transaction costs, and a single cash rate that serves as both interest and
   financing rate.
4. **Strategies.** These are PPO agents (CleanRL-style) with composite rewards
   (`log_return`, `vol`, `var`, `cvar_95`, `downside_risk_penalty`, `alpha_penalty`)
   and the baselines: minimum variance (MV), cost-aware mean-variance (MRR-C), and
   static or rolling OLS hedges.
5. **Evaluation.** Every strategy is scored on the synthetic test trajectories and the
   historical test partition. The metrics are return/risk metrics plus hedge
   effectiveness against the unhedged position.

## Setup

**Python.** The project needs Python 3.11+ (pandas 3 requires it).

```bash
python -m venv ~/.local/share/venvs/rl-dcc-hedge
~/.local/share/venvs/rl-dcc-hedge/bin/pip install -r requirements.txt
```

**Sweep script.** `run_rl_sweep.sh` adds a Python environment to `PATH` through the
placeholder `PYTHON_ENV_BIN="/path/to/your/python/env/bin"` near the top of the file.
Before running a sweep, replace that placeholder with the `bin/` directory of the
environment you installed the requirements into (for example
`$HOME/.local/share/venvs/rl-dcc-hedge/bin`).

**R.** R is needed only to generate datasets. Install R and these packages:

```r
install.packages(c("rugarch", "rmgarch", "WeightedPortTest", "FinTS", "xts",
                   "checkmate", "fs", "arrow", "ggplot2", "stringr"))
```

Run every command from the repository root, because all config paths are relative
to it.

## Data

Datasets are created automatically the first time `run_rl.py` runs. The
`DataHandler` downloads daily prices from Yahoo Finance, resamples them to weekly
(W-WED) log returns, fits the model in R and writes these files:

```
datasets/SPY_XLU/
  seed_df.parquet                                        weekly log returns (model input)
  train_simulation_60_209_10000_unconditional.parquet    PPO training pool (10 000 paths)
  eval_simulation_10065_209_64_unconditional.parquet     checkpoint selection (64 paths)
  test_simulation_10265_209_64_unconditional.parquet     synthetic test set (64 paths)
  test_partition.parquet                                 historical hold-out, DCC-filtered
r/fitted_model.RData                                     the fitted model
```

A file is regenerated only when it is missing. The file names encode the seed, the
number of steps, the number of paths and the start method, so changing any of those
settings in `configs/data_config.py` creates new files and leaves the old ones alone.

## Main entry points

### `run_rl.py`: train and test an agent

`run_rl.py` trains a PPO agent on the training simulations and evaluates it on the
eval simulations every `evaluation_interval` iterations. It keeps the best checkpoint
for each metric in `eval_metrics` (plus the final checkpoint) and then tests each kept
checkpoint on the historical test partition.

The settings are the fields of `RLExpConfig` (`configs/rl_config.py`), and each one is
a CLI flag. Run `python run_rl.py --help` to see all of them.

```bash
PY=~/.local/share/venvs/rl-dcc-hedge/bin/python

# Train and test with the defaults (logs to wandb unless --no-wandb is given)
$PY run_rl.py --no-wandb

# A custom reward mix, one pass over every training path, with the best checkpoint chosen by volatility
$PY run_rl.py --reward-components "log_return:0.4,vol:0.22,alpha_penalty:0.05" \
              --all-traj --eval-metrics "Annual Volatility:min" --seed 42

# Train only, or test an existing checkpoint
$PY run_rl.py --train-only
$PY run_rl.py --test-only --agent-checkpoint-path results/cleanrl_runs/<run>/best_cleanrl_model_<metric>.pt

# Run the baselines in HEDGE_BASELINE_STRATEGIES on the test partition, then exit
$PY run_rl.py --run-baselines
```

These are the outputs:

| Path | Contents |
|---|---|
| `results/cleanrl_runs/<run_name>/` | `best_cleanrl_model_<metric>.pt`, `final_cleanrl_model.pt`, eval trajectories (CSV/PNG), `test_trajectory_<strategy>.csv` |
| `results/test_records/rl_test_results.csv` | one row of test-partition metrics per tested strategy, added on each run |

`<run_name>` defaults to a timestamp.

Some notes on the settings:

- `--reward-components` is a comma-separated list of `name:weight` pairs.
- `--all-traj` sets `num_iterations` so that training goes through every simulated
  path once.
- `--cuda` is off by default because this PPO setup runs faster on the CPU.

### `run_experiment_tests.py`: evaluate everything on the test sets

`run_experiment_tests.py` runs a set of strategies through the same environment
mechanics on **every synthetic test trajectory** and on the **historical test
partition**. The strategies are:

- the unhedged anchor (the reference with no trades),
- every baseline in `KNOWN_BASELINE_STRATEGIES`,
- every agent checkpoint in `AGENT_CHECKPOINTS`.

The two registries are in `configs/experiment_tests_config.py`. Each hedged run gets
two hedge-effectiveness scores against the anchor on the same data:

- **HE (Variance Reduction)**: `1 − Var(hedged) / Var(unhedged)`
- **HE (Ederington R²)**: the squared correlation of hedged and unhedged returns

```bash
$PY run_experiment_tests.py --experiment-run-name my_run
$PY run_experiment_tests.py --baselines MV MRR-C_100     # only some of the baselines
```

The results go to `results/experiments/<experiment_run_name or timestamp>/`:

```
metrics.csv                        long format: id_tag, data, metric_name, metric_value
trajectories/<data>_<id_tag>.csv   per-step trajectory of every run
```

`data` is `synth_test_0` … `synth_test_63` or `test_partition`. An agent's `id_tag` is
`<reward tag>_<seed>`.

Before running it, check two things:

- **`AGENT_CHECKPOINTS` should be updated** Point
  the checkpoints at your own runs before starting. The script stops if any file is missing.
- **The configuration needs the data to exist already.** Importing
  `configs/experiment_tests_config.py` reads `datasets/SPY_XLU/seed_df.parquet`
  because the OLS baselines are fitted on the modeling-period returns. On a fresh
  clone, run `run_rl.py` once first to build the datasets.

## Typical workflow

```bash
# 1. Train the agent grid: 5 reward mixes × 4 seeds (configs/sweep_config.yaml)
./run_rl_sweep.sh                         # needs a wandb login

# 2. Put the best checkpoints in AGENT_CHECKPOINTS (configs/experiment_tests_config.py)

# 3. Evaluate the agents and the baselines
$PY run_experiment_tests.py --experiment-run-name final

# 4. Make the tables and figures
$PY summarize_experiment_tests.py --run-dir results/experiments/final
$PY compare_rl_vs_baseline.py --run-dir results/experiments/final --rl-weight 0.4 --baseline MRR-C_100
```

## Supporting scripts

| Script | Purpose |
|---|---|
| `run_rl_sweep.sh [CONFIG] [MAX_RUNS]` | Creates a wandb sweep (default `configs/sweep_config.yaml`) and starts an agent for it |
| `summarize_experiment_tests.py` | Turns a run's `metrics.csv` into summary tables (CSV/Markdown/LaTeX), dose-response and pairwise comparisons, and frontier figures, written to `<run_dir>/summary/` |
| `compare_rl_vs_baseline.py` | Compares one RL reward tag with one baseline, trajectory by trajectory. It relates each one's wins to market conditions and writes the results to `<run_dir>/rl_vs_<baseline>_w<weight>/` |
| `tune_ols_baselines.py` | Grid search over the rolling-OLS baseline's window and rebalance period on the eval simulations, written to `results/ols_grid_search/` |

## Repository layout

| Path | Role |
|---|---|
| `configs/data_config.py` | Tickers, date ranges, simulation sizes, seeds and bootstrap settings |
| `configs/rl_config.py` | `RLExpConfig`: PPO hyperparameters, environment and market frictions, run flags |
| `configs/experiment_tests_config.py` | `ExpTestsConfig`, the baseline and checkpoint registries, the test metric set |
| `configs/sweep_config.yaml` | wandb grid of reward mix × seed |
| `rl_dcc_hedge/data_handlers.py` | Yahoo Finance download, seed dataset, calls into the R pipeline, parquet caching |
| `rl_dcc_hedge/environments.py` | `HedgeEnv` base class, `ST_HedgeEnv` (single path) and `MT_HedgeEnv` (many paths) |
| `rl_dcc_hedge/baseline_strategies.py` | `UnhedgedAnchor`, `TheoreticalMinVar`, `CostAwareMeanVarHedge`, `StaticOLSHedge`, `PeriodicOLSHedge`, `RollingOLSHedge` |
| `rl_dcc_hedge/cleanrl_agents.py` | The PPO actor-critic network (`HedgingAgent`) |
| `rl_dcc_hedge/rl_experiments.py` | PPO training loop, evaluation, checkpointing, testing |
| `rl_dcc_hedge/utils/` | Margin-feasibility tools, portfolio metrics, weekly resampling, training helpers |
| `r/fit.r` | Model fitting, simulation, DCC filtering, test-partition creation, bootstrapped test data |
| `r/portfolio_modeling_functions.r` | R helpers: spec builders, I/O, diagnostics |
| `r/portfolio_modeling.ipynb` | Notebook used to develop and check the model |

The git-ignored outputs are `datasets/`, `results/`, `wandb/`, `runs/` and `*.RData`.
