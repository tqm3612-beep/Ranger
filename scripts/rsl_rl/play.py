# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

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
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--width", type=int, default=3840, help="Render width for video recording.")
parser.add_argument("--height", type=int, default=2160, help="Render height for video recording.")
parser.add_argument("--window_width", type=int, default=3840, help="Window width for rendering.")
parser.add_argument("--window_height", type=int, default=2160, help="Window height for rendering.")
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Recording camera position in world coordinates.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Recording camera look-at target in world coordinates.",
)
parser.add_argument(
    "--camera_focal_length",
    type=float,
    default=None,
    help="Recording camera focal length. Larger values zoom in.",
)
parser.add_argument(
    "--video_output_width",
    type=int,
    default=None,
    help="Output video width. Defaults to the render width.",
)
parser.add_argument(
    "--video_output_height",
    type=int,
    default=None,
    help="Output video height. Defaults to the render height.",
)
parser.add_argument("--video_bitrate", type=str, default="30000k", help="Output video bitrate.")
parser.add_argument("--video_crf", type=int, default=14, help="Output video CRF. Lower is higher quality.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    if getattr(args_cli, "width", None) is None:
        args_cli.width = 3840
    if getattr(args_cli, "height", None) is None:
        args_cli.height = 2160
    if getattr(args_cli, "window_width", None) is None:
        args_cli.window_width = 3840
    if getattr(args_cli, "window_height", None) is None:
        args_cli.window_height = 2160

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from pxr import UsdGeom

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaacsim.core.utils.viewports import set_camera_view

import Ranger.tasks  # noqa: F401
from video_utils import HighQualityRecordVideo


def _configure_video_viewer(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Mirror the default viewer to the right-side diagonal during video capture."""

    if not args_cli.video:
        return
    if args_cli.camera_eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.camera_eye)
    if args_cli.camera_lookat is not None:
        env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)
    if args_cli.camera_eye is not None or args_cli.camera_lookat is not None:
        return
    if args_cli.task.split(":")[-1] == "Template-Ranger-SimpleTerrain-Visual-v0":
        env_cfg.viewer.eye = (-10.0, 20.0, 5.0)
        env_cfg.viewer.lookat = (-10.0, 7.0, 0.0)
        return
    eye_x, eye_y, eye_z = env_cfg.viewer.eye
    env_cfg.viewer.eye = (eye_x, abs(eye_y), eye_z)


def _configure_recording_camera(env) -> None:
    """Apply the configured recording camera view and optional narrower FOV."""

    if not args_cli.video:
        return

    unwrapped_env = env.unwrapped
    eye = tuple(args_cli.camera_eye) if args_cli.camera_eye is not None else tuple(unwrapped_env.cfg.viewer.eye)
    target = (
        tuple(args_cli.camera_lookat) if args_cli.camera_lookat is not None else tuple(unwrapped_env.cfg.viewer.lookat)
    )
    try:
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    except TypeError:
        set_camera_view(eye, target, camera_prim_path="/OmniverseKit_Persp")
    except Exception:
        unwrapped_env.sim.set_camera_view(eye, target)

    camera_prim = UsdGeom.Camera(unwrapped_env.sim.stage.GetPrimAtPath("/OmniverseKit_Persp"))
    if camera_prim:
        if args_cli.camera_focal_length is not None:
            camera_prim.GetFocalLengthAttr().Set(args_cli.camera_focal_length)
        elif args_cli.task.split(":")[-1] == "Template-Ranger-SimpleTerrain-Visual-v0":
            camera_prim.GetFocalLengthAttr().Set(30.5)


def _get_policy_normalizer(ppo_runner: OnPolicyRunner, policy_nn: torch.nn.Module) -> torch.nn.Module | None:
    """Return the observation normalizer across supported RSL-RL versions."""

    if hasattr(ppo_runner, "obs_normalizer"):
        return ppo_runner.obs_normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        return policy_nn.actor_obs_normalizer
    return None


def _get_current_observations(env):
    """Return observations across Isaac Lab wrapper API versions."""

    obs = env.get_observations()
    if isinstance(obs, tuple):
        return obs[0]
    return obs


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    _configure_video_viewer(env_cfg)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        try:
            from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "This Isaac Lab installation does not provide published pretrained checkpoint lookup. "
                "Use --checkpoint with a local model path instead."
            ) from exc

        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _configure_recording_camera(env)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "video_bitrate": args_cli.video_bitrate,
            "video_crf": args_cli.video_crf,
            "output_size": (
                args_cli.video_output_width or args_cli.width,
                args_cli.video_output_height or args_cli.height,
            ),
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = HighQualityRecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    policy_normalizer = _get_policy_normalizer(ppo_runner, policy_nn)
    try:
        export_policy_as_jit(policy_nn, policy_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=policy_normalizer, path=export_model_dir, filename="policy.onnx")
    except Exception as exc:
        print(f"[WARNING] Failed to export policy, continuing play without exported files: {exc}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = _get_current_observations(env)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            step_out = env.step(actions)
            obs = step_out[0]
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
