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
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    if getattr(args_cli, "width", None) is None:
        args_cli.width = 1920
    if getattr(args_cli, "height", None) is None:
        args_cli.height = 1080
    if getattr(args_cli, "window_width", None) is None:
        args_cli.window_width = 1920
    if getattr(args_cli, "window_height", None) is None:
        args_cli.window_height = 1080

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
from pxr import UsdGeom

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaacsim.core.utils.viewports import set_camera_view

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Ranger.tasks  # noqa: F401
from video_utils import HighQualityRecordVideo

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _configure_video_viewer(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Mirror the default viewer to the right-side diagonal during video capture."""

    if not args_cli.video:
        return
    if args_cli.task.split(":")[-1] == "Template-Ranger-SimpleTerrain-Visual-v0":
        env_cfg.viewer.eye = (-10.0, 20.0, 5.0)
        env_cfg.viewer.lookat = (-10.0, 5.0, 0.0)
        return
    eye_x, eye_y, eye_z = env_cfg.viewer.eye
    env_cfg.viewer.eye = (eye_x, abs(eye_y), eye_z)


def _configure_simpleterrain_visual_camera(env) -> None:
    """Apply the fixed recording view and narrower FOV only for the visual simple-terrain task."""

    if not args_cli.video or args_cli.task.split(":")[-1] != "Template-Ranger-SimpleTerrain-Visual-v0":
        return

    unwrapped_env = env.unwrapped
    eye = (-10.0, 20.0, 5.0)
    target = (-10.0, 5.0, 0.0)
    try:
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    except TypeError:
        set_camera_view(eye, target, camera_prim_path="/OmniverseKit_Persp")
    except Exception:
        unwrapped_env.sim.set_camera_view(eye, target)

    camera_prim = UsdGeom.Camera(unwrapped_env.sim.stage.GetPrimAtPath("/OmniverseKit_Persp"))
    if camera_prim:
        camera_prim.GetFocalLengthAttr().Set(30.5)


def _resolve_resume_path(log_root_path: str, load_run: str, load_checkpoint: str) -> str:
    """Resolve a checkpoint path from either an explicit file path or the standard run/checkpoint selectors."""

    expanded_checkpoint = os.path.abspath(os.path.expanduser(load_checkpoint))
    if os.path.isfile(expanded_checkpoint):
        return expanded_checkpoint
    return get_checkpoint_path(log_root_path, load_run, load_checkpoint)


def _is_forward_finetune_task(task_name: str) -> bool:
    return task_name.split(":")[-1] in {
        "Template-Ranger-Forward-v0",
        "Template-Ranger-Forward-Visual-v0",
        "Template-Ranger-SimpleTerrain-v0",
        "Template-Ranger-SimpleTerrain-Visual-v0",
        "Template-Ranger-MapPosture-v0",
        "Template-Ranger-MapPosture-Visual-v0",
    }


def _load_forward_finetune_weights(runner: OnPolicyRunner, checkpoint_path: str) -> None:
    """Load only actor/critic weights for forward-stage fine-tuning and reset policy std."""

    loaded_dict = torch.load(checkpoint_path, map_location=runner.device, weights_only=False)
    model_state_dict = loaded_dict["model_state_dict"]
    actor_critic = runner.alg.policy
    actor_critic_state = actor_critic.state_dict()

    actor_critic_keys = {
        key: value for key, value in model_state_dict.items() if key.startswith("actor.") or key.startswith("critic.")
    }
    missing_actor_critic_keys = {
        key for key in actor_critic_state.keys() if (key.startswith("actor.") or key.startswith("critic.")) and key not in actor_critic_keys
    }
    if missing_actor_critic_keys:
        missing_preview = sorted(missing_actor_critic_keys)
        raise KeyError(f"Checkpoint is missing actor/critic weights required for forward fine-tuning: {missing_preview}")

    actor_critic.load_state_dict(actor_critic_keys, strict=False)

    if hasattr(actor_critic, "std"):
        actor_critic.std.data.fill_(0.5)
    elif hasattr(actor_critic, "log_std"):
        actor_critic.log_std.data.fill_(torch.log(torch.tensor(0.5, device=actor_critic.log_std.device)))
    else:
        raise AttributeError("Actor-critic policy does not expose 'std' or 'log_std' for action noise reset.")

    runner.current_learning_iteration = 0
    print("Loaded stand policy weights for forward fine-tuning; reset action std to 0.5.")


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
    _configure_video_viewer(env_cfg)

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
    _configure_simpleterrain_visual_camera(env)

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
        env = HighQualityRecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if _is_forward_finetune_task(args_cli.task):
            _load_forward_finetune_weights(runner, resume_path)
        else:
            # load previously trained model
            runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
