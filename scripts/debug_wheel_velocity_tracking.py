#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug whether the Ranger wheel CSV action drives actual wheel motion."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Ranger wheel velocity tracking without policy involvement.")
parser.add_argument("--task", type=str, default="Template-Ranger-Forward-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to print.")
parser.add_argument("--settle_steps", type=int, default=30, help="Number of zero-action settle steps before tests.")
parser.add_argument("--steps_per_target", type=int, default=60, help="Number of steps to run for each wheel target.")
parser.add_argument("--print_every", type=int, default=5, help="Print debug information every N steps.")
parser.add_argument(
    "--target_speed",
    type=float,
    default=2.0,
    help="Absolute wheel velocity target in rad/s used for sign-combination tests.",
)
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


def _format_wheel_line(
    wheel_name: str,
    wheel_raw_action: float,
    velocity_target: float,
    wheel_joint_vel: float,
    wheel_torque: float,
    contact_force: torch.Tensor,
) -> str:
    force_norm = float(torch.norm(contact_force).item())
    return (
        f"{wheel_name}: "
        f"wheel_raw_action={wheel_raw_action: .4f} "
        f"wheel_velocity_target={velocity_target: .4f} "
        f"wheel_joint_vel={wheel_joint_vel: .4f} "
        f"wheel_torque={wheel_torque: .4f} "
        f"contact_force=({float(contact_force[0].item()): .3f},"
        f" {float(contact_force[1].item()): .3f},"
        f" {float(contact_force[2].item()): .3f}) "
        f"|F|={force_norm: .3f}"
    )


def _get_action_slice(unwrapped_env, term_name: str) -> slice:
    start = 0
    for active_name, dim in zip(unwrapped_env.action_manager.active_terms, unwrapped_env.action_manager.action_term_dim):
        if active_name == term_name:
            return slice(start, start + dim)
        start += dim
    raise KeyError(f"Action term '{term_name}' not found in action manager.")


def _build_actions(env, wheel_action_slice: slice, wheel_targets_rad_s: torch.Tensor) -> torch.Tensor:
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    wheel_action_values = wheel_targets_rad_s / 20.0
    actions[:, wheel_action_slice] = wheel_action_values.unsqueeze(0).repeat(actions.shape[0], 1)
    return actions


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        wheel_action_term = unwrapped.action_manager.get_term("wheel_motor_csv")
        contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
        wheel_action_slice = _get_action_slice(unwrapped, "wheel_motor_csv")

        wheel_joint_ids, wheel_joint_names = robot.find_joints(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
        wheel_body_ids, wheel_body_names = contact_sensor.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)

        if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
            raise ValueError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}.")

        print(f"[INFO]: task = {args_cli.task}", flush=True)
        print(f"[INFO]: testing env_id = {args_cli.env_id}", flush=True)
        print(f"[INFO]: wheel joints = {wheel_joint_names}", flush=True)
        print(f"[INFO]: wheel bodies = {wheel_body_names}", flush=True)
        print(f"[INFO]: wheel action slice = [{wheel_action_slice.start}:{wheel_action_slice.stop}]", flush=True)
        print("[INFO]: leg action is fixed to 0 for the whole script.", flush=True)

              
        with torch.inference_mode():
            zero_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)

            test_patterns = {
                "all_pos": [1.0, 1.0, 1.0, 1.0],
                "all_neg": [-1.0, -1.0, -1.0, -1.0],
                "left_pos_right_neg": [1.0, 1.0, -1.0, -1.0],
                "left_neg_right_pos": [-1.0, -1.0, 1.0, 1.0],
            }

            summary = []

            for target_index, (pattern_name, signs) in enumerate(test_patterns.items()):
                env.reset()

                for _ in range(args_cli.settle_steps):
                    env.step(zero_actions)

                wheel_targets = torch.tensor(
                    [s * args_cli.target_speed for s in signs],
                    device=unwrapped.device,
                    dtype=torch.float32,
                )

                print(
                    f"[TARGET {target_index}] pattern={pattern_name} "
                    f"wheel_targets={[float(x) for x in wheel_targets.tolist()]} rad/s",
                    flush=True,
                )

                test_actions = _build_actions(env, wheel_action_slice, wheel_targets)

                base_vx_values = []
                base_wz_values = []
                final_wheel_joint_vel = None
                final_velocity_target = None
                final_wheel_torque = None

                for step in range(args_cli.steps_per_target):
                    env.step(test_actions)

                    wheel_raw_action = wheel_action_term.raw_actions[args_cli.env_id]
                    velocity_target = wheel_action_term.velocity_target[args_cli.env_id]
                    wheel_joint_vel = robot.data.joint_vel[args_cli.env_id, wheel_joint_ids]
                    wheel_torque = wheel_action_term.torque_actual[args_cli.env_id]
                    contact_forces = contact_sensor.data.net_forces_w_history[
                        args_cli.env_id, 0, wheel_body_ids, :
                    ]

                    base_lin_vel_x = float(robot.data.root_lin_vel_b[args_cli.env_id, 0].item())
                    base_ang_vel_z = float(robot.data.root_ang_vel_b[args_cli.env_id, 2].item())

                    base_vx_values.append(base_lin_vel_x)
                    base_wz_values.append(base_ang_vel_z)

                    final_wheel_joint_vel = wheel_joint_vel.detach().clone()
                    final_velocity_target = velocity_target.detach().clone()
                    final_wheel_torque = wheel_torque.detach().clone()

                    if step % args_cli.print_every != 0:
                        continue

                    print(
                        f"[STEP {step:04d}] pattern={pattern_name} "
                        f"base_lin_vel_x={base_lin_vel_x: .4f} "
                        f"base_ang_vel_z={base_ang_vel_z: .4f}",
                        flush=True,
                    )

                    for wheel_name, raw_action, target_value, joint_vel, torque, force in zip(
                        wheel_joint_names,
                        wheel_raw_action,
                        velocity_target,
                        wheel_joint_vel,
                        wheel_torque,
                        contact_forces,
                    ):
                        print(
                            "  "
                            + _format_wheel_line(
                                wheel_name=wheel_name,
                                wheel_raw_action=float(raw_action.item()),
                                velocity_target=float(target_value.item()),
                                wheel_joint_vel=float(joint_vel.item()),
                                wheel_torque=float(torque.item()),
                                contact_force=force,
                            ),
                            flush=True,
                        )

                mean_vx = sum(base_vx_values) / max(len(base_vx_values), 1)
                mean_abs_wz = sum(abs(x) for x in base_wz_values) / max(len(base_wz_values), 1)

                summary.append(
                    {
                        "pattern": pattern_name,
                        "targets": [float(x) for x in wheel_targets.tolist()],
                        "mean_vx": mean_vx,
                        "mean_abs_wz": mean_abs_wz,
                        "final_joint_vel": [float(x) for x in final_wheel_joint_vel.tolist()],
                        "final_velocity_target": [float(x) for x in final_velocity_target.tolist()],
                        "final_torque": [float(x) for x in final_wheel_torque.tolist()],
                    }
                )

            print("\n[SUMMARY] wheel sign pattern test", flush=True)
            print(
                "pattern | targets[w_lb,w_lf,w_rf,w_rb] | mean_base_lin_vel_x | "
                "mean_abs_base_ang_vel_z | final_joint_vel[w_lb,w_lf,w_rf,w_rb] | "
                "final_velocity_target[w_lb,w_lf,w_rf,w_rb] | final_torque[w_lb,w_lf,w_rf,w_rb]",
                flush=True,
            )

            for item in summary:
                print(
                    f"{item['pattern']} | {item['targets']} | "
                    f"{item['mean_vx']: .5f} | {item['mean_abs_wz']: .5f} | "
                    f"{item['final_joint_vel']} | {item['final_velocity_target']} | "
                    f"{item['final_torque']}",
                    flush=True,
                )

            best = max(summary, key=lambda x: x["mean_vx"] - 0.2 * x["mean_abs_wz"])
            print(
                f"\n[BEST_GUESS] pattern={best['pattern']} "
                f"targets={best['targets']} "
                f"mean_base_lin_vel_x={best['mean_vx']: .5f} "
                f"mean_abs_base_ang_vel_z={best['mean_abs_wz']: .5f}",
                flush=True,
            )
            

    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
