#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate the effective direction of each Ranger suspension joint."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate per-joint g_* direction against wheel height response.")
parser.add_argument("--task", type=str, default="Template-Ranger-SimpleTerrain-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to print.")
parser.add_argument("--settle_steps", type=int, default=20, help="Number of zero-action settle steps before tests.")
parser.add_argument("--steps_per_target", type=int, default=60, help="Number of simulation steps to hold each target.")
parser.add_argument("--print_every", type=int, default=20, help="Print debug information every N steps.")
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


def _collect_state(unwrapped, env_id: int, leg_joint_ids: list[int], wheel_body_ids: list[int]) -> dict[str, torch.Tensor | float]:
    robot = unwrapped.scene["robot"]
    joint_pos = robot.data.joint_pos[env_id, leg_joint_ids].detach().clone()
    joint_target = robot.data.joint_pos_target[env_id, leg_joint_ids].detach().clone()
    wheel_pos_w = robot.data.body_pos_w[env_id, wheel_body_ids, :].detach().clone()
    root_pos_w = robot.data.root_pos_w[env_id].detach().clone()
    root_quat_w = robot.data.root_quat_w[env_id].detach().clone().unsqueeze(0)
    wheel_pos_b = quat_apply_inverse(root_quat_w.repeat(wheel_pos_w.shape[0], 1), wheel_pos_w - root_pos_w)
    roll, pitch, _ = euler_xyz_from_quat(root_quat_w)
    return {
        "joint_target": joint_target,
        "joint_pos": joint_pos,
        "wheel_z_b": wheel_pos_b[:, 2].detach().clone(),
        "root_height": float(root_pos_w[2].item()),
        "roll_deg": float(torch.rad2deg(roll)[0].item()),
        "pitch_deg": float(torch.rad2deg(pitch)[0].item()),
    }


def _format_tensor(name: str, value: torch.Tensor) -> str:
    return f"{name}={[round(float(x), 4) for x in value.tolist()]}"


def _print_state(tag: str, state: dict[str, torch.Tensor | float], wheel_delta: torch.Tensor) -> None:
    print(f"[{tag}] {_format_tensor('position_target_g', state['joint_target'])}", flush=True)
    print(f"  {_format_tensor('actual_joint_pos_g', state['joint_pos'])}", flush=True)
    print(f"  {_format_tensor('wheel_z_b', state['wheel_z_b'])}", flush=True)
    print(f"  {_format_tensor('wheel_z_b_delta', wheel_delta)}", flush=True)
    print(
        f"  root_height={state['root_height']:.4f} roll_deg={state['roll_deg']:.4f} pitch_deg={state['pitch_deg']:.4f}",
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
        leg_joint_ids, leg_joint_names = robot.find_joints(["g_lb", "g_lf", "g_rf", "g_rb"], preserve_order=True)
        wheel_body_ids, wheel_body_names = robot.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)

        if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
            raise ValueError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}.")

        print(f"[INFO]: task = {args_cli.task}", flush=True)
        print(f"[INFO]: leg joints = {leg_joint_names}", flush=True)
        print(f"[INFO]: wheel bodies = {wheel_body_names}", flush=True)

        joint_to_wheel = {
            "g_lb": "w_lb",
            "g_lf": "w_lf",
            "g_rf": "w_rf",
            "g_rb": "w_rb",
        }
        wheel_name_to_idx = {name: idx for idx, name in enumerate(wheel_body_names)}

        env_ids = torch.tensor([args_cli.env_id], dtype=torch.long, device=robot.device)
        joint_ids = torch.tensor(leg_joint_ids, dtype=torch.long, device=robot.device)
        zero_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)

        summary_rows: list[dict[str, object]] = []

        with torch.inference_mode():
            for joint_index, joint_name in enumerate(leg_joint_names):
                for target_value in (0.5, -0.5):
                    env.reset()
                    for _ in range(args_cli.settle_steps):
                        step_out = env.step(zero_actions)
                        _ = step_out[0]

                    target = torch.zeros((1, len(leg_joint_ids)), dtype=robot.data.joint_pos.dtype, device=robot.device)
                    target[0, joint_index] = target_value

                    initial_state = _collect_state(unwrapped, args_cli.env_id, leg_joint_ids, wheel_body_ids)
                    print(f"\n[TEST] only_{joint_name}_{'high' if target_value > 0 else 'low'}", flush=True)
                    _print_state("initial", initial_state, torch.zeros_like(initial_state["wheel_z_b"]))

                    final_state = initial_state
                    final_delta = torch.zeros_like(initial_state["wheel_z_b"])
                    for step in range(args_cli.steps_per_target):
                        robot.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)
                        unwrapped.scene.write_data_to_sim()
                        unwrapped.sim.step(render=False)
                        if unwrapped.sim.has_gui() or unwrapped.sim.has_rtx_sensors():
                            if step % unwrapped.cfg.sim.render_interval == 0:
                                unwrapped.sim.render()
                        unwrapped.scene.update(dt=unwrapped.physics_dt)

                        final_state = _collect_state(unwrapped, args_cli.env_id, leg_joint_ids, wheel_body_ids)
                        final_delta = final_state["wheel_z_b"] - initial_state["wheel_z_b"]
                        if step % args_cli.print_every == 0 or step == args_cli.steps_per_target - 1:
                            _print_state(f"step_{step:04d}", final_state, final_delta)

                    affected_wheel_name = joint_to_wheel[joint_name]
                    affected_wheel_idx = wheel_name_to_idx[affected_wheel_name]
                    affected_delta = float(final_delta[affected_wheel_idx].item())
                    summary_rows.append(
                        {
                            "joint_name": joint_name,
                            "target_value": target_value,
                            "affected_wheel_name": affected_wheel_name,
                            "affected_wheel_delta": affected_delta,
                            "all_wheel_delta": [float(x) for x in final_delta.tolist()],
                            "roll_deg": float(final_state["roll_deg"]),
                            "pitch_deg": float(final_state["pitch_deg"]),
                        }
                    )

        print("\n[SUMMARY] per-joint direction calibration", flush=True)
        for row in summary_rows:
            direction = "increase" if row["affected_wheel_delta"] > 0.0 else "decrease"
            print(
                f"  {row['joint_name']} target={row['target_value']:+.1f}: "
                f"{row['affected_wheel_name']} wheel_z_b tends to {direction} "
                f"({row['affected_wheel_delta']:+.4f} m), "
                f"all_wheel_delta={[round(x, 4) for x in row['all_wheel_delta']]}, "
                f"roll_deg={row['roll_deg']:.4f}, pitch_deg={row['pitch_deg']:.4f}",
                flush=True,
            )

        print("\n[CONCLUSION]", flush=True)
        per_joint_positive_direction = {}
        for joint_name in leg_joint_names:
            positive_row = next(row for row in summary_rows if row["joint_name"] == joint_name and row["target_value"] > 0)
            per_joint_positive_direction[joint_name] = positive_row["affected_wheel_delta"]
            direction = "increase" if positive_row["affected_wheel_delta"] > 0.0 else "decrease"
            print(f"  + direction on {joint_name} makes its paired wheel_z_b {direction}.", flush=True)

        signs = [1.0 if per_joint_positive_direction[name] > 0.0 else -1.0 for name in leg_joint_names]
        if len(set(signs)) == 1:
            print(
                "  All four g_* joints show the same effective sign on their paired wheel height. "
                "A shared stroke->joint sign is likely sufficient.",
                flush=True,
            )
        else:
            print(
                "  The four g_* joints do not share the same effective sign on paired wheel height. "
                "You likely need a per-joint sign in the stroke->joint mapping.",
                flush=True,
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
