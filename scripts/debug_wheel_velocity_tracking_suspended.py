#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug Ranger wheel velocity tracking with the robot suspended off the ground."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug suspended Ranger wheel velocity tracking without policy involvement.")
parser.add_argument("--task", type=str, default="Template-Ranger-Forward-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to print.")
parser.add_argument("--init_height", type=float, default=1.5, help="Initial base height for the suspended test.")
parser.add_argument("--settle_steps", type=int, default=10, help="Number of zero-action settle steps before tests.")
parser.add_argument("--steps_per_target", type=int, default=40, help="Number of steps to run for each wheel target.")
parser.add_argument("--print_every", type=int, default=5, help="Print debug information every N steps.")
parser.add_argument(
    "--wheel_targets",
    type=float,
    nargs=2,
    default=(2.0, -2.0),
    help="Two wheel velocity targets in rad/s to test sequentially.",
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


def _build_actions(env, wheel_action_slice: slice, wheel_target_rad_s: float) -> torch.Tensor:
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    wheel_action_value = wheel_target_rad_s / 20.0
    actions[:, wheel_action_slice] = wheel_action_value
    return actions


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.init_height)
    # Keep the suspended test alive long enough to inspect the wheel velocity loop itself.
    env_cfg.terminations.bad_orientation.params["limit_angle"] = 10.0
    env_cfg.terminations.root_height_low.params["minimum_height"] = -10.0

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
        print(f"[INFO]: suspended init height = {args_cli.init_height}", flush=True)
        print("[INFO]: sim gravity forced to (0, 0, 0) for this test.", flush=True)
        print(f"[INFO]: wheel joints = {wheel_joint_names}", flush=True)
        print(f"[INFO]: wheel bodies = {wheel_body_names}", flush=True)
        print(f"[INFO]: wheel action slice = [{wheel_action_slice.start}:{wheel_action_slice.stop}]", flush=True)
        print("[INFO]: leg action is fixed to 0 for the whole script.", flush=True)

        with torch.inference_mode():
            zero_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
            for _ in range(args_cli.settle_steps):
                env.step(zero_actions)

            for target_index, wheel_target in enumerate(args_cli.wheel_targets):
                print(f"[TARGET {target_index}] commanded wheel target = {wheel_target: .4f} rad/s", flush=True)
                test_actions = _build_actions(env, wheel_action_slice, wheel_target)

                for step in range(args_cli.steps_per_target):
                    env.step(test_actions)

                    if step % args_cli.print_every != 0:
                        continue

                    wheel_raw_action = wheel_action_term.raw_actions[args_cli.env_id]
                    velocity_target = wheel_action_term.velocity_target[args_cli.env_id]
                    wheel_joint_vel = robot.data.joint_vel[args_cli.env_id, wheel_joint_ids]
                    wheel_torque = wheel_action_term.torque_actual[args_cli.env_id]
                    contact_forces = contact_sensor.data.net_forces_w_history[args_cli.env_id, 0, wheel_body_ids, :]
                    base_lin_vel_x = float(robot.data.root_lin_vel_b[args_cli.env_id, 0].item())
                    base_ang_vel_z = float(robot.data.root_ang_vel_b[args_cli.env_id, 2].item())

                    print(
                        f"[STEP {step:04d}] base_lin_vel_x={base_lin_vel_x: .4f} "
                        f"base_ang_vel_z={base_ang_vel_z: .4f} "
                        f"target_command={wheel_target: .4f} rad/s",
                        flush=True,
                    )
                    for wheel_name, raw_action, target_value, joint_vel, torque, force in zip(
                        wheel_joint_names, wheel_raw_action, velocity_target, wheel_joint_vel, wheel_torque, contact_forces
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
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
