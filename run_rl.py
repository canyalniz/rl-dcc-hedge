import math
import os
import sys
import torch
import tyro

from rl_dcc_hedge.data_handlers import DataHandler
from rl_dcc_hedge.rl_experiments import RLExperiment
from rl_dcc_hedge.utils.tools import dotdict
from rl_dcc_hedge.baseline_strategies import HEDGE_BASELINE_STRATEGIES
from rl_dcc_hedge.utils.portfolio_metrics import compute_portfolio_metrics, STAT_METRICS
from dataclasses import fields

from configs.rl_config import RLExpConfig

def _normalize_wandb_bool_args(argv: list[str]) -> list[str]:
    """Translate wandb sweep style '--flag=True/False' into tyro's '--flag'/'--no-flag'."""
    bool_names = {f.name for f in fields(RLExpConfig) if f.type is bool}
    out, i = [], 0
    while i < len(argv):
        arg = argv[i]
        name, eq, val = arg.partition("=")
        if name.startswith("--") and name[2:] in bool_names and eq:
            out.append(name if val.lower() == "true" else "--no-" + name[2:])
            i += 1
        elif (name.startswith("--") and name[2:] in bool_names and not eq
                and i + 1 < len(argv) and argv[i + 1].lower() in ("true", "false")):
            out.append(name if argv[i + 1].lower() == "true" else "--no-" + name[2:])
            i += 2
        else:
            out.append(arg)
            i += 1
    return out

def get_args():
    args = tyro.cli(RLExpConfig, args=_normalize_wandb_bool_args(sys.argv[1:]), description="Run RL experiment for portfolio optimization")

    args.batch_size = int(args.num_envs * args.num_steps)

    if args.all_traj:
        print("INFO: all_traj flag is set...")
        total_traj_steps = args.sim_steps  * args.m_sim
        args.num_iterations = math.ceil(total_traj_steps/args.batch_size)
        print(f"Total simulated steps: {total_traj_steps}")
        print(f"WARNING: Changing num_iterations to {args.num_iterations}")
        print(f"Total number of steps will be: {args.num_iterations * args.batch_size}")

    
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    if args.minibatch_size == 0:
        raise ValueError(f"minibatch_size is 0. Please ensure that num_envs * num_steps >= num_minibatches. Current values: num_envs={args.num_envs}, num_steps={args.num_steps}, num_minibatches={args.num_minibatches}")
    args.total_timesteps = args.num_iterations * args.batch_size
    args.run_dir = os.path.join(args.cleanrl_root, args.run_name)

    args.eval_num_envs = args.eval_m_sim
    if args.eval_num_envs > 64:
        print(f"Eval number of SyncVectorEnv was set to eval_m_sim ({args.eval_m_sim}), this many evaluation environments will be created. Might have performance implications.")
    
    os.makedirs(args.run_dir, exist_ok=True)
        
    return args


if __name__ == '__main__':
    args = get_args()
    data_handler = DataHandler(args)

    if args.run_baselines:
        print("Running baseline strategies...")

        test_data = data_handler.load_test_partition_data(get_in_parts=True)
        test_log_returns = test_data['log_returns']
        test_sampling_times = test_data['sampling_times']
        # for name, fn_baseline_strategy in CLASSIC_BASELINE_STRATEGIES.items():
        #     print(f"Running {name}")
        #     portfolio_simple_returns = fn_baseline_strategy(log_returns=test_log_returns[1:], transaction_cost_rate=args.commission_fee_pct)
        #     RLExperiment.record_metrics(portfolio_simple_returns=portfolio_simple_returns, args=args, sampling_times=test_sampling_times, strategy_name=name)

        for name, baseline_strategy in HEDGE_BASELINE_STRATEGIES.items():
            print(f"Running {name}")
            RLExperiment.test(args, data_handler, device="cpu", baseline_strategy=baseline_strategy)
        exit(0)  # Exit after running baselines to avoid running the RL experiment
    
    if args.test_only and args.train_only:
        raise ValueError("Only one of --test_only or --train_only can be set at a time.")

    if not args.test_only:
        print("Instantiating RLExperiment...")
        args.agent_checkpoint_path = None # ignore any given checkpoint path for training
        experiment = RLExperiment(args)
        print("Starting training...")
        try:
            best_checkpoint_paths = experiment.train()
        except KeyboardInterrupt:
            print("Keyboard interrupt received, halting training...")
            if any(path is not None for path in experiment.best_checkpoint_paths.values()):
                best_checkpoint_paths = experiment.best_checkpoint_paths
                print("Checkpoint(s) detected, running tests")
            else:
                print("No checkpoints so far, exiting...")
                raise KeyboardInterrupt
        
    
    if not args.train_only:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        if args.test_only:
            if args.agent_checkpoint_path is None:
                raise ValueError("In test_only mode, --agent_checkpoint_path must be provided to load the trained agent.")
            print(f"Running test from checkpoint: {args.agent_checkpoint_path}")
            # a single user-provided checkpoint: no metric tag is appended to the agent name
            eval_runs = [(None, args.agent_checkpoint_path)]
        else:
            final_checkpoint_path = experiment.final_checkpoint_path if args.also_final_iteration else None
            if not any(path is not None for path in best_checkpoint_paths.values()) and final_checkpoint_path is None:
                raise RuntimeError(
                    "Training produced no checkpoint (save_model disabled, run shorter "
                    "than evaluation_burn_in, or eval metrics NaN); refusing to test an "
                    "untrained agent."
                )
            # one (metric tag, checkpoint) pair per tracked metric that produced a checkpoint;
            # the tag carries the metric name and its max/min order
            eval_runs = [
                (RLExperiment.metric_tag(metric, experiment.eval_metrics[metric]), path)
                for metric, path in best_checkpoint_paths.items()
                if path is not None
            ]
            # the final-iteration checkpoint joins the metric based checkpoints when
            # also_final_iteration is toggled, so it is tracked-evaluated and tested too
            if final_checkpoint_path is not None:
                eval_runs.append(("final", final_checkpoint_path))

        for metric_tag, checkpoint_path in eval_runs:
            args.agent_checkpoint_path = checkpoint_path
            if metric_tag is not None:
                print(f"Using best checkpoint for {metric_tag}: {checkpoint_path}")

            if args.tracked_evaluation:
                try:
                    df_account_value = RLExperiment.tracked_evaluation(args, data_handler, device, artifacts_suffix=metric_tag)
                    print("Tracked evaluation finished successfully.")
                except Exception as e:
                    print(f"Error during tracked evaluation: {e}")
                    raise e
        
            try:
                df_account_value = RLExperiment.test(args, data_handler, device, strategy_name_suffix=metric_tag)
                print(f"Test finished successfully. Records saved in {args.records_dir}")
            except Exception as e:
                print(f"Error during testing: {e}")
                raise e