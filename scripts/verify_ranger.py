# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test the Ranger environment and print the resolved robot joints."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the Ranger robot asset inside the Ranger task.")
parser.add_argument("--task", type=str, default="Template-Ranger-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--steps", type=int, default=5, help="Number of zero-action simulation steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Ranger.tasks  # noqa: F401
from Ranger.assets.ranger.ranger_cfg import RANGER_URDF_PATH, RANGER_USD_PATH


def main() -> None:
    """Create the environment, print robot metadata, and step zero actions."""
    if not RANGER_USD_PATH.exists():
        raise FileNotFoundError(
            f"Ranger USD not found: {RANGER_USD_PATH}\n"
            f"Place the URDF at {RANGER_URDF_PATH} and convert it to this USD path first."
        )

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        print(f"[INFO]: Robot prim path expression: {robot.cfg.prim_path}")
        print(f"[INFO]: Joint names: {robot.data.joint_names}")
        print(f"[INFO]: Body names: {robot.data.body_names}")
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
