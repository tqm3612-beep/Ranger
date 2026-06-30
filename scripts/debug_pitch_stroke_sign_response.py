#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug how fixed front/rear stroke splits affect base pitch."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Ranger front/rear stroke split versus base pitch.")
parser.add_argument("--task", type=str, default="Template-Ranger-SimpleTerrain-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to print.")
parser.add_argument("--command_duration", type=float, default=1.5, help="Duration in seconds for each command.")
parser.add_argument("--settle_steps", type=int, default=30, help="Number of settle steps before each test.")
parser.add_argument("--print_every", type=int, default=10, help="Print debug information every N steps.")
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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

import Ranger.tasks  # noqa: F401


def _get_action_slice(unwrapped_env, term_name: str) -> slice:
    start = 0
    for active_name, dim in zip(unwrapped_env.action_manager.active_terms, unwrapped_env.action_manager.action_term_dim):
        if active_name == term_name:
            return slice(start, start + dim)
        start += dim
    raise KeyError(f"Action term '{term_name}' not found in action manager.")


def _stroke_to_raw_action(leg_action_term, stroke_values: torch.Tensor) -> torch.Tensor:
    stroke_mid = 0.5 * (leg_action_term.cfg.stroke_min + leg_action_term.cfg.stroke_max)
    stroke_half_range = 0.5 * (leg_action_term.cfg.stroke_max - leg_action_term.cfg.stroke_min)
    raw = (stroke_values - stroke_mid) / max(stroke_half_range, 1.0e-6)
    return torch.clamp(raw, min=-1.0, max=1.0)


def _collect_state(unwrapped, env_id: int, wheel_body_ids: list[int]) -> dict[str, torch.Tensor | float]:
    robot = unwrapped.scene["robot"]
    leg_action_term = unwrapped.action_manager.get_term("leg_hydraulic")

    root_pos_w = robot.data.root_pos_w[env_id].detach().clone()
    root_quat_w = robot.data.root_quat_w[env_id].detach().clone().unsqueeze(0)
    wheel_pos_w = robot.data.body_pos_w[env_id, wheel_body_ids, :].detach().clone()
    wheel_pos_b = quat_apply_inverse(root_quat_w.repeat(wheel_pos_w.shape[0], 1), wheel_pos_w - root_pos_w)
    roll, pitch, _ = euler_xyz_from_quat(root_quat_w)

    stroke_command = leg_action_term.stroke_command[env_id].detach().clone()
    front_stroke_mean = stroke_command[[1, 2]].mean()
    rear_stroke_mean = stroke_command[[0, 3]].mean()
    front_rear_wheel_z_diff = wheel_pos_b[1:3, 2].mean() - wheel_pos_b[[0, 3], 2].mean()

    return {
        "pitch_deg": float(torch.rad2deg(pitch)[0].item()),
        "root_height_w": float(root_pos_w[2].item()),
        "front_rear_wheel_z_diff": float(front_rear_wheel_z_diff.item()),
        "front_stroke_mean": float(front_stroke_mean.item()),
        "rear_stroke_mean": float(rear_stroke_mean.item()),
    }


def _print_state(tag: str, state: dict[str, float]) -> None:
    print(
        f"[{tag}] pitch_deg={state['pitch_deg']:.4f} "
        f"root_height_w={state['root_height_w']:.4f} "
        f"front_rear_wheel_z_diff={state['front_rear_wheel_z_diff']:.4f} "
        f"front_stroke_mean={state['front_stroke_mean']:.4f} "
        f"rear_stroke_mean={state['rear_stroke_mean']:.4f}",
        flush=True,
    )


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        leg_action_term = unwrapped.action_manager.get_term("leg_hydraulic")
        leg_action_slice = _get_action_slice(unwrapped, "leg_hydraulic")
        wheel_action_slice = _get_action_slice(unwrapped, "wheel_motor_csv")
        wheel_body_ids, wheel_body_names = robot.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)

        if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
            raise ValueError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}.")

        print(f"[INFO]: task = {args_cli.task}", flush=True)
        print(f"[INFO]: wheel bodies = {wheel_body_names}", flush=True)

        steps_per_command = max(1, int(round(args_cli.command_duration / unwrapped.step_dt)))
        command_patterns = {
            "A_all_0p40": [0.40, 0.40, 0.40, 0.40],
            "B_front_0p50_rear_0p35": [0.35, 0.50, 0.50, 0.35],
            "C_front_0p35_rear_0p50": [0.50, 0.35, 0.35, 0.50],
        }

        with torch.inference_mode():
            zero_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
            summaries = {}

            for pattern_name, stroke_values in command_patterns.items():
                env.reset()
                for _ in range(args_cli.settle_steps):
                    step_out = env.step(zero_actions)
                    _ = step_out[0]

                stroke_tensor = torch.tensor(stroke_values, dtype=torch.float32, device=unwrapped.device)
                raw_leg_action = _stroke_to_raw_action(leg_action_term, stroke_tensor)
                command_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
                command_actions[:, leg_action_slice] = raw_leg_action.unsqueeze(0).repeat(command_actions.shape[0], 1)
                command_actions[:, wheel_action_slice] = 0.0

                print(f"\n[COMMAND] {pattern_name} stroke={stroke_values}", flush=True)
                running = {
                    "pitch_deg": 0.0,
                    "root_height_w": 0.0,
                    "front_rear_wheel_z_diff": 0.0,
                    "front_stroke_mean": 0.0,
                    "rear_stroke_mean": 0.0,
                }

                for step in range(steps_per_command):
                    step_out = env.step(command_actions)
                    _ = step_out[0]
                    state = _collect_state(unwrapped, args_cli.env_id, wheel_body_ids)
                    if step % args_cli.print_every == 0 or step == steps_per_command - 1:
                        _print_state(f"step_{step:04d}", state)
                    for key in running:
                        running[key] += state[key]

                summary = {key: value / steps_per_command for key, value in running.items()}
                summaries[pattern_name] = summary
                _print_state("mean", summary)

            print("\n[SUMMARY]", flush=True)
            for pattern_name, summary in summaries.items():
                _print_state(pattern_name, summary)

            pitch_b = summaries["B_front_0p50_rear_0p35"]["pitch_deg"]
            pitch_c = summaries["C_front_0p35_rear_0p50"]["pitch_deg"]
            if pitch_b > pitch_c:
                direction = "front_stroke > rear_stroke makes pitch larger"
                k_pitch_hint = "If positive pitch should be compensated, k_pitch is likely negative."
            elif pitch_b < pitch_c:
                direction = "front_stroke > rear_stroke makes pitch smaller"
                k_pitch_hint = "If positive pitch should be compensated, k_pitch is likely positive."
            else:
                direction = "front_stroke > rear_stroke has no clear pitch effect in this short rollout"
                k_pitch_hint = "Keep k_pitch sign unchanged until a stronger test result is available."
            print(f"\n[INTERPRETATION] {direction}", flush=True)
            print(f"[INTERPRETATION] {k_pitch_hint}", flush=True)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
