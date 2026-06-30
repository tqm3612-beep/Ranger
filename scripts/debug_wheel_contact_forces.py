#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print Ranger wheel contact forces during runtime for debugging."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Ranger wheel-ground contact forces.")
parser.add_argument("--task", type=str, default="Template-Ranger-Stand-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--steps", type=int, default=120, help="Number of zero-action steps to simulate.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to print.")
parser.add_argument("--print_every", type=int, default=5, help="Print contact forces every N steps.")
parser.add_argument("--contact_threshold", type=float, default=1.0, help="Force threshold for contact detection.")
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


def _format_force_line(body_name: str, force: torch.Tensor, threshold: float) -> str:
    force_norm = float(torch.norm(force).item())
    in_contact = force_norm > threshold
    return (
        f"{body_name}: "
        f"Fx={float(force[0].item()): .3f} "
        f"Fy={float(force[1].item()): .3f} "
        f"Fz={float(force[2].item()): .3f} "
        f"|F|={force_norm: .3f} "
        f"contact={int(in_contact)}"
    )


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        env.reset()
        unwrapped = env.unwrapped
        sensor = unwrapped.scene.sensors["wheel_contact_forces"]
        body_ids, body_names = sensor.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)

        if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
            raise ValueError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}.")

        print(f"[INFO]: task = {args_cli.task}", flush=True)
        print(f"[INFO]: sensor bodies = {body_names}", flush=True)
        print(f"[INFO]: printing env_id = {args_cli.env_id}", flush=True)

        with torch.inference_mode():
            for step in range(args_cli.steps):
                actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
                env.step(actions)

                if step % args_cli.print_every != 0:
                    continue

                force_history = sensor.data.net_forces_w_history[args_cli.env_id, :, body_ids, :]
                # Use the largest magnitude sample in the short sensor history to avoid missing transient contacts.
                force_norm_history = torch.norm(force_history, dim=-1)
                peak_indices = torch.argmax(force_norm_history, dim=0)
                current_forces = force_history[peak_indices, torch.arange(len(body_ids), device=unwrapped.device)]

                print(f"[STEP {step:04d}]", flush=True)
                for body_name, force in zip(body_names, current_forces):
                    print("  " + _format_force_line(body_name, force, threshold=args_cli.contact_threshold), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
