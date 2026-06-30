#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug the suspension stroke_des -> position_target command chain without policy involvement."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Ranger suspension stroke_des command response.")
parser.add_argument("--task", type=str, default="Template-Ranger-SimpleTerrain-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to print.")
parser.add_argument("--command_duration", type=float, default=1.5, help="Duration in seconds for each command.")
parser.add_argument("--settle_steps", type=int, default=30, help="Number of zero-action settle steps before tests.")
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


def _format_tensor(name: str, value: torch.Tensor) -> str:
    return f"{name}={[round(float(x), 4) for x in value.tolist()]}"


def _collect_state(unwrapped, env_id: int, leg_joint_ids: list[int], wheel_body_ids: list[int]) -> dict[str, torch.Tensor | float]:
    robot = unwrapped.scene["robot"]
    leg_action_term = unwrapped.action_manager.get_term("leg_hydraulic")

    stroke_des = leg_action_term.stroke_des[env_id].detach().clone()
    stroke_command = leg_action_term.stroke_command[env_id].detach().clone()
    position_target = leg_action_term.position_target[env_id].detach().clone()
    joint_position_sign = leg_action_term.joint_position_sign[0].detach().clone()
    effort_actual = leg_action_term.effort_actual[env_id].detach().clone()
    joint_pos = robot.data.joint_pos[env_id, leg_joint_ids].detach().clone()

    wheel_pos_w = robot.data.body_pos_w[env_id, wheel_body_ids, :].detach().clone()
    root_pos_w = robot.data.root_pos_w[env_id].detach().clone()
    root_quat_w = robot.data.root_quat_w[env_id].detach().clone().unsqueeze(0)
    wheel_pos_b = quat_apply_inverse(root_quat_w.repeat(wheel_pos_w.shape[0], 1), wheel_pos_w - root_pos_w)
    roll, pitch, _ = euler_xyz_from_quat(root_quat_w)

    return {
        "stroke_des": stroke_des,
        "stroke_command": stroke_command,
        "position_target": position_target,
        "joint_position_sign": joint_position_sign,
        "effort_actual": effort_actual,
        "joint_pos": joint_pos,
        "wheel_z_b": wheel_pos_b[:, 2].detach().clone(),
        "root_height": float(root_pos_w[2].item()),
        "roll_deg": float(torch.rad2deg(roll)[0].item()),
        "pitch_deg": float(torch.rad2deg(pitch)[0].item()),
    }


def _print_state(tag: str, raw_leg_action: torch.Tensor, state: dict[str, torch.Tensor | float]) -> None:
    print(f"[{tag}] {_format_tensor('raw_leg_action', raw_leg_action)}", flush=True)
    print(f"  {_format_tensor('joint_position_sign', state['joint_position_sign'])}", flush=True)
    print(f"  {_format_tensor('stroke_des', state['stroke_des'])}", flush=True)
    print(f"  {_format_tensor('stroke_command', state['stroke_command'])}", flush=True)
    print(f"  {_format_tensor('position_target', state['position_target'])}", flush=True)
    print(f"  {_format_tensor('effort_actual_not_used', state['effort_actual'])}", flush=True)
    print(f"  {_format_tensor('actual_joint_pos_g', state['joint_pos'])}", flush=True)
    print(f"  {_format_tensor('wheel_z_b', state['wheel_z_b'])}", flush=True)
    print(
        f"  root_height={state['root_height']:.4f} roll_deg={state['roll_deg']:.4f} pitch_deg={state['pitch_deg']:.4f}",
        flush=True,
    )


def _print_diagnosis(initial_state: dict[str, torch.Tensor | float], final_state: dict[str, torch.Tensor | float]) -> None:
    pos_target_delta = torch.max(torch.abs(final_state["position_target"] - initial_state["position_target"])).item()
    joint_pos_delta = torch.max(torch.abs(final_state["joint_pos"] - initial_state["joint_pos"])).item()
    wheel_z_delta = torch.max(torch.abs(final_state["wheel_z_b"] - initial_state["wheel_z_b"])).item()

    if pos_target_delta > 1.0e-4 and joint_pos_delta <= 1.0e-4:
        print(
            "  diagnosis: position_target changed, but actual joint_pos did not. "
            "The implicit actuator / position-target execution layer may not be taking effect.",
            flush=True,
        )
    elif joint_pos_delta > 1.0e-4 and wheel_z_delta <= 1.0e-4:
        print(
            "  diagnosis: joint_pos changed, but wheel_z_b did not. "
            "The g_* joints may not be the suspension joints that move wheel-center height.",
            flush=True,
        )
    elif pos_target_delta > 1.0e-4 and joint_pos_delta > 1.0e-4 and wheel_z_delta > 1.0e-4:
        print(
            "  diagnosis: position_target, joint_pos, and wheel_z_b all changed. "
            "The position-target active-suspension chain appears to be working.",
            flush=True,
        )
    else:
        print("  diagnosis: response is weak or mixed; inspect the printed values directly.", flush=True)


def _print_pattern_expectation(pattern_name: str, final_state: dict[str, torch.Tensor | float]) -> None:
    roll_abs = abs(float(final_state["roll_deg"]))
    pitch_abs = abs(float(final_state["pitch_deg"]))
    if pattern_name == "left_high_right_low":
        dominant = "roll" if roll_abs >= pitch_abs else "pitch"
        print(
            f"  expectation: left_high_right_low should mainly excite roll. dominant_axis={dominant} "
            f"(abs_roll_deg={roll_abs:.4f}, abs_pitch_deg={pitch_abs:.4f})",
            flush=True,
        )
    elif pattern_name == "front_high_rear_low":
        dominant = "pitch" if pitch_abs >= roll_abs else "roll"
        print(
            f"  expectation: front_high_rear_low should mainly excite pitch. dominant_axis={dominant} "
            f"(abs_roll_deg={roll_abs:.4f}, abs_pitch_deg={pitch_abs:.4f})",
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
        leg_action_slice = _get_action_slice(unwrapped, "leg_hydraulic")
        wheel_action_slice = _get_action_slice(unwrapped, "wheel_motor_csv")
        leg_joint_ids, leg_joint_names = robot.find_joints(["g_lb", "g_lf", "g_rf", "g_rb"], preserve_order=True)
        wheel_body_ids, wheel_body_names = robot.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)

        if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
            raise ValueError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}.")

        print(f"[INFO]: task = {args_cli.task}", flush=True)
        print(f"[INFO]: leg joints = {leg_joint_names}", flush=True)
        print(f"[INFO]: wheel bodies = {wheel_body_names}", flush=True)

        steps_per_command = max(1, int(round(args_cli.command_duration / unwrapped.step_dt)))
        command_patterns = {
            "all_stroke_low": [-1.0, -1.0, -1.0, -1.0],
            "all_stroke_middle": [0.0, 0.0, 0.0, 0.0],
            "all_stroke_high": [1.0, 1.0, 1.0, 1.0],
            "left_high_right_low": [1.0, 1.0, -1.0, -1.0],
            "front_high_rear_low": [-1.0, 1.0, 1.0, -1.0],
        }

        with torch.inference_mode():
            zero_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)

            for pattern_name, leg_values in command_patterns.items():
                env.reset()
                for _ in range(args_cli.settle_steps):
                    step_out = env.step(zero_actions)
                    _ = step_out[0]

                command_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
                raw_leg_action = torch.tensor(leg_values, dtype=torch.float32, device=unwrapped.device)
                command_actions[:, leg_action_slice] = raw_leg_action.unsqueeze(0).repeat(command_actions.shape[0], 1)
                command_actions[:, wheel_action_slice] = 0.0

                initial_state = _collect_state(unwrapped, args_cli.env_id, leg_joint_ids, wheel_body_ids)
                print(f"\n[COMMAND] {pattern_name}", flush=True)
                _print_state("initial", raw_leg_action, initial_state)

                final_state = initial_state
                for step in range(steps_per_command):
                    step_out = env.step(command_actions)
                    _ = step_out[0]
                    final_state = _collect_state(unwrapped, args_cli.env_id, leg_joint_ids, wheel_body_ids)
                    if step % args_cli.print_every == 0 or step == steps_per_command - 1:
                        _print_state(f"step_{step:04d}", raw_leg_action, final_state)

                _print_diagnosis(initial_state, final_state)
                _print_pattern_expectation(pattern_name, final_state)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
