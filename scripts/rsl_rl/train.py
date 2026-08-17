# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from export_iteration_metrics import export_tensorboard_scalars  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--disable_iteration_metrics_csv",
    action="store_true",
    default=False,
    help="Disable automatic TensorBoard scalar export to per-iteration CSV files.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--warm_start_checkpoint",
    type=str,
    default=None,
    help="Checkpoint used only to initialize actor parameters.",
)
parser.add_argument(
    "--warm_start_mode",
    type=str,
    choices=(
        "actor_suspension",
        "actor_wheel",
        "actor_suspension_only",
        "actor_wheel_reset_suspension",
        "actor_wheel_reset_final",
        "actor_reset_heads",
        "actor_heads",
        "actor_full",
        "recurrent_v10_actor",
    ),
    default=None,
    help=(
        "Actor warm-start mode. "
        "'actor_suspension' loads the shared encoders, actor trunk, "
        "and suspension head while keeping the wheel head random. "
        "'actor_wheel' loads the complete actor and trains only the wheel head. "
        "'actor_suspension_only' loads the complete actor and trains only the suspension head. "
        "'actor_wheel_reset_suspension' loads the shared actor and wheel head, resets the suspension head, "
        "and trains only the suspension head. "
        "'actor_wheel_reset_final' loads the complete actor, resets only the wheel head's final Linear, "
        "and trains only that final Linear. "
        "'actor_reset_heads' loads only the shared encoders/trunk, resets both action heads, "
        "and trains both action heads. "
        "'actor_heads' loads the complete actor and trains both action heads while freezing encoders/trunk. "
        "'actor_full' loads the complete actor. "
        "'recurrent_v10_actor' is recurrent-only and copies only the V10-compatible map encoder, actor trunk, "
        "suspension head, and wheel head; recurrent representation/critic/std stay newly initialized."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.resume and args_cli.warm_start_checkpoint is not None:
    raise ValueError("--resume and --warm_start_checkpoint cannot be used together.")
if args_cli.warm_start_checkpoint is not None and args_cli.warm_start_mode is None:
    raise ValueError("--warm_start_mode is required when --warm_start_checkpoint is provided.")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

try:
    from isaaclab.utils.io import dump_pickle
except ImportError:
    try:
        import cloudpickle as pickle
    except ImportError:
        import pickle

    def dump_pickle(filename, data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Ranger.tasks  # noqa: F401
from Ranger.tasks.manager_based.ranger.agents import RangerTerrainActorCritic, RangerTerrainActorCriticRecurrent
from warm_start import warm_start_ranger_actor, warm_start_ranger_recurrent_from_feedforward

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

rsl_on_policy_runner.RangerTerrainActorCritic = RangerTerrainActorCritic
rsl_on_policy_runner.RangerTerrainActorCriticRecurrent = RangerTerrainActorCriticRecurrent


def _resolve_resume_path(log_root_path: str, load_run: str, load_checkpoint: str) -> str:
    """Resolve a checkpoint path from either an explicit file path or the standard run/checkpoint selectors."""

    expanded_checkpoint = os.path.abspath(os.path.expanduser(load_checkpoint))
    if os.path.isfile(expanded_checkpoint):
        return expanded_checkpoint
    return get_checkpoint_path(log_root_path, load_run, load_checkpoint)


def _reset_action_std_from_env(runner: OnPolicyRunner) -> None:
    """Optionally reset policy exploration std after checkpoint loading."""

    reset_std = os.environ.get("RANGER_RESET_ACTION_STD")
    if reset_std is None:
        return

    try:
        reset_std_value = float(reset_std)
    except ValueError as exc:
        raise ValueError(f"RANGER_RESET_ACTION_STD must be a positive float, got: {reset_std!r}") from exc
    if reset_std_value <= 0.0:
        raise ValueError(f"RANGER_RESET_ACTION_STD must be positive, got: {reset_std_value}")

    actor_critic = runner.alg.policy
    with torch.no_grad():
        if hasattr(actor_critic, "std"):
            actor_critic.std.fill_(reset_std_value)
        elif hasattr(actor_critic, "log_std"):
            log_std_value = torch.log(
                torch.tensor(reset_std_value, device=actor_critic.log_std.device, dtype=actor_critic.log_std.dtype)
            )
            actor_critic.log_std.fill_(log_std_value)
        else:
            print("[WARN] RANGER_RESET_ACTION_STD was set, but actor_critic has no std/log_std attribute.")
            return

    print(f"[INFO] Reset action std to {reset_std_value} from RANGER_RESET_ACTION_STD.")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = _resolve_resume_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        _reset_action_std_from_env(runner)
    elif args_cli.warm_start_checkpoint is not None:
        is_recurrent_policy = bool(getattr(runner.alg.policy, "is_recurrent", False))
        if is_recurrent_policy:
            if args_cli.warm_start_mode != "recurrent_v10_actor":
                raise ValueError(
                    "RangerTerrainActorCriticRecurrent requires --warm_start_mode recurrent_v10_actor when "
                    "initializing from a feedforward V10 checkpoint. Use --resume for recurrent checkpoints."
                )
            warm_start_ranger_recurrent_from_feedforward(
                runner=runner,
                checkpoint_path=args_cli.warm_start_checkpoint,
            )
        else:
            if args_cli.warm_start_mode == "recurrent_v10_actor":
                raise ValueError("--warm_start_mode recurrent_v10_actor is valid only for the recurrent Ranger task.")
            warm_start_ranger_actor(
                runner=runner,
                checkpoint_path=args_cli.warm_start_checkpoint,
                mode=args_cli.warm_start_mode,
            )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training. Always flush and export scalar history, including on KeyboardInterrupt.
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        writer = getattr(runner, "writer", None)
        if writer is not None and hasattr(writer, "flush"):
            writer.flush()
        if not args_cli.disable_iteration_metrics_csv:
            try:
                export_tensorboard_scalars(log_dir)
            except Exception as error:
                print(f"[WARN] Failed to export per-iteration CSV metrics: {error}")
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
