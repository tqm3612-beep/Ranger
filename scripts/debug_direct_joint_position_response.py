#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug Ranger suspension structure by directly commanding g_* joint position targets."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug direct g_* joint position response without HydraulicActuatorAction.")
parser.add_argument("--task", type=str, default="Template-Ranger-SimpleTerrain-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to print.")
parser.add_argument("--settle_steps", type=int, default=30, help="Number of zero-action settle steps before tests.")
parser.add_argument("--steps_per_target", type=int, default=90, help="Number of simulation steps to hold each target.")
parser.add_argument("--print_every", type=int, default=15, help="Print debug information every N steps.")
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
from Ranger.assets.ranger.ranger_cfg import RANGER_CFG, RANGER_URDF_PATH


def _parse_urdf_joint_limits(urdf_path: str, joint_names: list[str]) -> dict[str, dict[str, float | None]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    parsed: dict[str, dict[str, float | None]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in joint_names:
            continue
        limit = joint.find("limit")
        parsed[name] = {
            "lower": float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            "upper": float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
            "effort": float(limit.attrib["effort"]) if limit is not None and "effort" in limit.attrib else None,
            "velocity": float(limit.attrib["velocity"]) if limit is not None and "velocity" in limit.attrib else None,
        }
    return parsed


def _collect_pose_state(unwrapped, env_id: int, leg_joint_ids: list[int], wheel_body_ids: list[int]) -> dict[str, torch.Tensor | float]:
    robot = unwrapped.scene["robot"]
    joint_pos = robot.data.joint_pos[env_id, leg_joint_ids].detach().clone()
    wheel_pos_w = robot.data.body_pos_w[env_id, wheel_body_ids, :].detach().clone()
    root_pos_w = robot.data.root_pos_w[env_id].detach().clone()
    root_quat_w = robot.data.root_quat_w[env_id].detach().clone().unsqueeze(0)
    wheel_pos_b = quat_apply_inverse(root_quat_w.repeat(wheel_pos_w.shape[0], 1), wheel_pos_w - root_pos_w)
    roll, pitch, _ = euler_xyz_from_quat(root_quat_w)
    return {
        "joint_pos": joint_pos,
        "wheel_z_b": wheel_pos_b[:, 2].detach().clone(),
        "root_height": float(root_pos_w[2].item()),
        "roll_deg": float(torch.rad2deg(roll)[0].item()),
        "pitch_deg": float(torch.rad2deg(pitch)[0].item()),
    }


def _format_tensor(name: str, value: torch.Tensor) -> str:
    return f"{name}={[round(float(x), 4) for x in value.tolist()]}"


def _print_pose_state(tag: str, target: torch.Tensor, state: dict[str, torch.Tensor | float]) -> None:
    print(f"[{tag}] {_format_tensor('joint_target_g', target)}", flush=True)
    print(f"  {_format_tensor('actual_joint_pos_g', state['joint_pos'])}", flush=True)
    print(f"  {_format_tensor('wheel_z_b', state['wheel_z_b'])}", flush=True)
    print(
        f"  root_height={state['root_height']:.4f} roll_deg={state['roll_deg']:.4f} pitch_deg={state['pitch_deg']:.4f}",
        flush=True,
    )


def _print_joint_limit_report(robot, joint_ids: list[int], joint_names: list[str]) -> None:
    print("[JOINT LIMITS] runtime values from loaded articulation", flush=True)
    pos_limits = robot.data.joint_pos_limits[0, joint_ids]
    vel_limits = robot.data.joint_vel_limits[0, joint_ids]
    effort_limits = robot.data.joint_effort_limits[0, joint_ids]
    for name, pos_limit, vel_limit, effort_limit in zip(joint_names, pos_limits, vel_limits, effort_limits, strict=True):
        print(
            f"  {name}: lower={float(pos_limit[0].item()):.4f} upper={float(pos_limit[1].item()):.4f} "
            f"velocity_limit={float(vel_limit.item()):.4f} effort_limit={float(effort_limit.item()):.4f}",
            flush=True,
        )


def _print_urdf_limit_report(joint_names: list[str]) -> None:
    print("[JOINT LIMITS] URDF values", flush=True)
    urdf_limits = _parse_urdf_joint_limits(str(RANGER_URDF_PATH), joint_names)
    for name in joint_names:
        item = urdf_limits.get(name, {})
        print(
            f"  {name}: lower={item.get('lower')} upper={item.get('upper')} "
            f"velocity_limit={item.get('velocity')} effort_limit={item.get('effort')}",
            flush=True,
        )


def _print_implicit_actuator_report() -> None:
    print("[ASSET ACTUATORS] ranger_cfg.py", flush=True)
    leg_cfg = RANGER_CFG.actuators["leg_joints"]
    print(
        "  leg_joints: "
        f"joint_names_expr={leg_cfg.joint_names_expr} "
        f"effort_limit_sim={leg_cfg.effort_limit_sim} "
        f"velocity_limit_sim={leg_cfg.velocity_limit_sim} "
        f"stiffness={leg_cfg.stiffness} damping={leg_cfg.damping}",
        flush=True,
    )
    print(
        "  interpretation: this implicit actuator is still active on g_* joints. "
        "Its high stiffness/damping can hold the joints near default or compete with the custom hydraulic effort controller.",
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

        _print_urdf_limit_report(leg_joint_names)
        _print_joint_limit_report(robot, leg_joint_ids, leg_joint_names)
        _print_implicit_actuator_report()

        target_patterns = {
            "all_low": [-0.5, -0.5, -0.5, -0.5],
            "all_mid": [0.0, 0.0, 0.0, 0.0],
            "all_high": [0.5, 0.5, 0.5, 0.5],
            "left_high_right_low": [0.5, 0.5, -0.5, -0.5],
            "front_high_rear_low": [-0.5, 0.5, 0.5, -0.5],
        }

        env_ids = torch.tensor([args_cli.env_id], dtype=torch.long, device=robot.device)
        joint_ids = torch.tensor(leg_joint_ids, dtype=torch.long, device=robot.device)
        zero_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)

        with torch.inference_mode():
            for pattern_name, values in target_patterns.items():
                env.reset()
                for _ in range(args_cli.settle_steps):
                    step_out = env.step(zero_actions)
                    _ = step_out[0]

                target = torch.tensor(values, dtype=robot.data.joint_pos.dtype, device=robot.device).unsqueeze(0)
                initial_state = _collect_pose_state(unwrapped, args_cli.env_id, leg_joint_ids, wheel_body_ids)
                print(f"\n[DIRECT TARGET] {pattern_name}", flush=True)
                _print_pose_state("initial", target[0], initial_state)

                final_state = initial_state
                for step in range(args_cli.steps_per_target):
                    robot.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)
                    unwrapped.scene.write_data_to_sim()
                    unwrapped.sim.step(render=False)
                    if unwrapped.sim.has_gui() or unwrapped.sim.has_rtx_sensors():
                        if step % unwrapped.cfg.sim.render_interval == 0:
                            unwrapped.sim.render()
                    unwrapped.scene.update(dt=unwrapped.physics_dt)
                    final_state = _collect_pose_state(unwrapped, args_cli.env_id, leg_joint_ids, wheel_body_ids)
                    if step % args_cli.print_every == 0 or step == args_cli.steps_per_target - 1:
                        _print_pose_state(f"step_{step:04d}", target[0], final_state)

                joint_delta = torch.max(torch.abs(final_state["joint_pos"] - target[0])).item()
                wheel_z_delta = torch.max(torch.abs(final_state["wheel_z_b"] - initial_state["wheel_z_b"])).item()
                print(
                    f"  summary: max_abs_joint_tracking_error={joint_delta:.4f} "
                    f"max_abs_wheel_z_change={wheel_z_delta:.4f}",
                    flush=True,
                )
                if wheel_z_delta > 1.0e-2:
                    print(
                        "  diagnosis: direct g_* position targeting changes wheel_z_b noticeably. "
                        "The suspension structure is effective; the main problem is likely in the actuator/implicit-drive chain.",
                        flush=True,
                    )
                elif joint_delta <= 5.0e-2 and wheel_z_delta <= 1.0e-2:
                    print(
                        "  diagnosis: g_* can be moved near target, but wheel_z_b barely changes. "
                        "This suggests the g_* joints are not effective suspension-height joints or the geometry coupling is weak.",
                        flush=True,
                    )
                else:
                    print(
                        "  diagnosis: g_* does not reach the commanded target well. "
                        "This suggests joint limits or active drive/actuator configuration is restricting motion.",
                        flush=True,
                    )

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
