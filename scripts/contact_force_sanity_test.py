#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a single-env Ranger contact-force sanity test without PPO."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

faulthandler.enable()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a single-env Ranger contact-force sanity test.")
parser.add_argument("--task", type=str, default="Template-Ranger-Stand-v0", help="Task used for the sanity test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to report.")
parser.add_argument(
    "--heights",
    type=float,
    nargs="+",
    default=(0.68, 0.70, 0.72, 0.75),
    help="Root/base heights to test.",
)
parser.add_argument("--num_steps", type=int, default=200, help="Number of zero-action control steps per height.")
parser.add_argument(
    "--print_steps",
    type=int,
    nargs="+",
    default=(0, 1, 2, 10, 50, 100, 200),
    help="Step indices to print for each tested height.",
)
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/contact_force_sanity",
    help="Directory where summaries and time-series CSV files will be written.",
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
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils import parse_env_cfg

import Ranger.tasks  # noqa: F401
from Ranger.assets.ranger.ranger_cfg import RANGER_URDF_PATH


WHEEL_BODY_NAMES_LF_LR_RF_RR = ["w_lf", "w_lb", "w_rf", "w_rb"]
WHEEL_BODY_NAME_SET = {"w_lb", "w_lf", "w_rf", "w_rb"}
HYDRAULIC_JOINT_NAMES_LF_LR_RF_RR = ["g_lf", "g_lb", "g_rf", "g_rb"]
PRINT_KEYS = (
    "root_height",
    "root_lin_vel_z",
    "base_roll",
    "base_pitch",
    "terminated",
    "truncated",
    "hydraulic_stroke_lf",
    "hydraulic_stroke_lr",
    "hydraulic_stroke_rf",
    "hydraulic_stroke_rr",
    "wheel_link_world_z_lf",
    "wheel_link_world_z_lr",
    "wheel_link_world_z_rf",
    "wheel_link_world_z_rr",
    "wheel_contact_force_total_norm",
    "wheel_contact_force_total_z",
    "wheel_contact_force_total_z_over_weight",
    "all_body_contact_force_total_norm",
    "all_body_contact_force_total_z",
    "all_body_contact_force_total_z_over_weight",
    "non_wheel_contact_force_total_norm",
    "non_wheel_contact_force_total_z",
    "non_wheel_contact_force_total_z_over_weight",
    "contact_bool_lf",
    "contact_bool_lr",
    "contact_bool_rf",
    "contact_bool_rr",
    "min_contact_force",
)


def _parse_robot_total_mass(urdf_path: Path) -> float:
    root = ET.parse(urdf_path).getroot()
    total_mass = 0.0
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_node = inertial.find("mass")
        if mass_node is None:
            continue
        total_mass += float(mass_node.attrib.get("value", 0.0))
    return total_mass


def _build_zero_actions(env) -> torch.Tensor:
    unwrapped = env.unwrapped
    device = getattr(unwrapped, "device", getattr(env, "device", None))
    if device is None:
        raise RuntimeError("Unable to determine environment device for action tensor creation.")
    action_space_shape = tuple(int(v) for v in env.action_space.shape)
    if len(action_space_shape) == 2:
        action_shape = action_space_shape
    elif len(action_space_shape) == 1:
        action_shape = (int(unwrapped.num_envs), action_space_shape[0])
    else:
        raise RuntimeError(f"Unsupported action_space shape: {action_space_shape}")
    return torch.zeros(action_shape, device=device, dtype=torch.float32)


def _env_ids_tensor(env_id: int, num_envs: int, device: torch.device) -> torch.Tensor:
    if not 0 <= int(env_id) < int(num_envs):
        raise RuntimeError(f"env_id out of bounds: env_id={env_id}, num_envs={num_envs}")
    env_ids = torch.tensor([int(env_id)], device=device, dtype=torch.long)
    if env_ids.ndim != 1 or env_ids.numel() != 1 or int(env_ids.item()) != int(env_id):
        raise RuntimeError(f"invalid env_ids tensor: env_ids={env_ids}, expected tensor([{env_id}])")
    return env_ids


def _validate_ids(name: str, ids: torch.Tensor, upper_bound: int) -> None:
    if ids.ndim != 1:
        raise RuntimeError(f"{name} must be 1D, got shape={tuple(ids.shape)}")
    if ids.numel() == 0:
        return
    ids_min = int(ids.min().item())
    ids_max = int(ids.max().item())
    if ids_min < 0 or ids_max >= int(upper_bound):
        raise RuntimeError(
            f"{name} out of bounds: min={ids_min}, max={ids_max}, upper_bound={upper_bound}, ids={ids.detach().cpu().tolist()}"
        )


def _validate_env_and_tensor(name: str, tensor: torch.Tensor, env_id: int, num_envs: int) -> None:
    if tensor.ndim < 1:
        raise RuntimeError(f"{name} must have env dimension, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] != int(num_envs):
        raise RuntimeError(f"{name} env dimension mismatch: shape={tuple(tensor.shape)}, num_envs={num_envs}")
    if not 0 <= int(env_id) < int(tensor.shape[0]):
        raise RuntimeError(f"{name} env index out of bounds: env_id={env_id}, shape={tuple(tensor.shape)}")


def _sensor_local_body_ids(sensor, body_names: list[str], *, device: torch.device, sensor_name: str) -> torch.Tensor:
    sensor_body_ids, resolved_body_names = sensor.find_bodies(body_names, preserve_order=True)
    sensor_body_ids = torch.as_tensor(sensor_body_ids, device=device, dtype=torch.long)
    sensor_body_count = int(len(sensor.body_names))
    _validate_ids(f"{sensor_name}.body_ids", sensor_body_ids, sensor_body_count)
    if list(resolved_body_names) != list(body_names):
        raise RuntimeError(
            f"{sensor_name} resolved unexpected body order: requested={body_names}, resolved={list(resolved_body_names)}"
        )
    return sensor_body_ids


def _robot_body_ids(robot, body_names: list[str], *, device: torch.device) -> torch.Tensor:
    robot_body_ids, resolved_body_names = robot.find_bodies(body_names, preserve_order=True)
    robot_body_ids = torch.as_tensor(robot_body_ids, device=device, dtype=torch.long)
    _validate_ids("robot.body_ids", robot_body_ids, len(robot.body_names))
    if list(resolved_body_names) != list(body_names):
        raise RuntimeError(
            f"robot resolved unexpected body order: requested={body_names}, resolved={list(resolved_body_names)}"
        )
    return robot_body_ids


def _joint_ids(robot, joint_names: list[str], *, device: torch.device) -> torch.Tensor:
    joint_ids, resolved_joint_names = robot.find_joints(joint_names, preserve_order=True)
    joint_ids = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
    _validate_ids("robot.joint_ids", joint_ids, len(robot.joint_names))
    if list(resolved_joint_names) != list(joint_names):
        raise RuntimeError(
            f"robot resolved unexpected joint order: requested={joint_names}, resolved={list(resolved_joint_names)}"
        )
    return joint_ids


def _step_env(env, actions):
    step_result = env.step(actions)
    if not isinstance(step_result, tuple):
        raise RuntimeError(f"Unexpected env.step result type: {type(step_result)!r}")
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated, truncated, info
    if len(step_result) == 4:
        obs, reward, dones, info = step_result
        dones_tensor = dones if isinstance(dones, torch.Tensor) else torch.as_tensor(dones, device=actions.device)
        terminated = dones_tensor
        truncated = torch.zeros_like(dones_tensor, dtype=torch.bool)
        return obs, reward, terminated, truncated, info
    raise RuntimeError(f"Unexpected env.step tuple length: {len(step_result)}")


def _as_bool(value: Any, env_id: int) -> bool:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return bool(value.item())
        return bool(value[env_id].item())
    if isinstance(value, (list, tuple)):
        return bool(value[env_id])
    return bool(value)


def _write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: Any) -> str:
    return f"{float(value):+.4f}"


def _print_sensor_audit(
    wheel_contact_sensor,
    all_body_contact_sensor,
    non_wheel_body_names: list[str],
) -> None:
    print("[AUDIT] sensor coverage", flush=True)
    print(f"  wheel_contact_forces.body_names={list(wheel_contact_sensor.body_names)}", flush=True)
    print(f"  all_body_contact_forces.body_names={list(all_body_contact_sensor.body_names)}", flush=True)
    print(f"  non_wheel_body_names={non_wheel_body_names}", flush=True)
    print(
        f"  wheel_contact_forces.data.net_forces_w_history.shape="
        f"{tuple(wheel_contact_sensor.data.net_forces_w_history.shape)}",
        flush=True,
    )
    print(
        f"  all_body_contact_forces.data.net_forces_w_history.shape="
        f"{tuple(all_body_contact_sensor.data.net_forces_w_history.shape)}",
        flush=True,
    )


def _set_env_root_height(robot, env_id: int, target_height: float) -> None:
    _validate_env_and_tensor("robot.data.root_pose_w", robot.data.root_pose_w, env_id, robot.data.root_pose_w.shape[0])
    _validate_env_and_tensor("robot.data.root_vel_w", robot.data.root_vel_w, env_id, robot.data.root_vel_w.shape[0])
    env_ids = _env_ids_tensor(env_id=env_id, num_envs=robot.data.root_pose_w.shape[0], device=robot.device)
    root_pose = robot.data.root_pose_w[env_id : env_id + 1].clone()
    root_velocity = robot.data.root_vel_w[env_id : env_id + 1].clone()
    root_pose[:, 2] = float(target_height)
    root_velocity.zero_()
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def _current_force_vectors(contact_sensor, env_id: int, body_ids: torch.Tensor, *, sensor_name: str) -> torch.Tensor:
    force_history = contact_sensor.data.net_forces_w_history
    _validate_env_and_tensor(f"{sensor_name}.data.net_forces_w_history", force_history, env_id, force_history.shape[0])
    if force_history.ndim != 4:
        raise RuntimeError(f"{sensor_name}.data.net_forces_w_history must be 4D, got shape={tuple(force_history.shape)}")
    sensor_body_count = int(force_history.shape[2])
    _validate_ids(f"{sensor_name}.body_ids", body_ids, sensor_body_count)
    return force_history[env_id, -1, body_ids, :]


def _force_summary(force_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    force_norm = torch.norm(force_vectors, dim=-1)
    force_z = force_vectors[:, 2]
    total_norm = force_norm.sum()
    total_z = force_z.sum()
    return force_norm, force_z, total_norm, total_z


def _termination_reason(unwrapped, env_id: int, terminated: bool, truncated: bool) -> str:
    reasons: list[str] = []
    if truncated:
        reasons.append("truncated")
    if terminated:
        for term_name in getattr(unwrapped.termination_manager, "active_terms", ()):
            term_value = unwrapped.termination_manager.get_term(term_name)
            if _as_bool(term_value, env_id):
                reasons.append(term_name)
    return ",".join(reasons) if reasons else ""


def _collect_row(
    env,
    env_id: int,
    requested_height: float,
    step: int,
    robot_total_mass: float,
    expected_weight_force: float,
    wheel_body_ids: torch.Tensor,
    all_body_contact_ids: torch.Tensor,
    non_wheel_contact_ids: torch.Tensor,
    hydraulic_joint_ids: torch.Tensor,
    robot_wheel_body_ids: torch.Tensor,
    terminated: bool = False,
    truncated: bool = False,
    termination_reason: str = "",
) -> dict[str, Any]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    wheel_contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    all_body_contact_sensor = unwrapped.scene.sensors["all_body_contact_forces"]
    leg_action_term = unwrapped.action_manager.get_term("leg_hydraulic")

    wheel_force_vectors = _current_force_vectors(
        wheel_contact_sensor, env_id, wheel_body_ids, sensor_name="wheel_contact_forces"
    )
    wheel_force_norm, wheel_force_z, wheel_total_norm, wheel_total_z = _force_summary(wheel_force_vectors)

    all_body_force_vectors = _current_force_vectors(
        all_body_contact_sensor, env_id, all_body_contact_ids, sensor_name="all_body_contact_forces"
    )
    all_body_force_norm, all_body_force_z, all_body_total_norm, all_body_total_z = _force_summary(all_body_force_vectors)

    if non_wheel_contact_ids.numel() > 0:
        non_wheel_force_vectors = _current_force_vectors(
            all_body_contact_sensor, env_id, non_wheel_contact_ids, sensor_name="all_body_contact_forces"
        )
        non_wheel_force_norm, non_wheel_force_z, non_wheel_total_norm, non_wheel_total_z = _force_summary(
            non_wheel_force_vectors
        )
    else:
        non_wheel_total_norm = torch.tensor(0.0, device=robot.device, dtype=torch.float32)
        non_wheel_total_z = torch.tensor(0.0, device=robot.device, dtype=torch.float32)
        non_wheel_force_norm = torch.zeros((0,), device=robot.device, dtype=torch.float32)
        non_wheel_force_z = torch.zeros((0,), device=robot.device, dtype=torch.float32)

    contact_bool = (wheel_force_norm > 1.0).to(torch.float32)
    _validate_env_and_tensor("robot.data.body_pos_w", robot.data.body_pos_w, env_id, unwrapped.num_envs)
    _validate_ids("robot_wheel_body_ids", robot_wheel_body_ids, robot.data.body_pos_w.shape[1])
    wheel_body_pos_w = robot.data.body_pos_w[env_id, robot_wheel_body_ids, :]
    _validate_env_and_tensor("leg_action_term.stroke_actual", leg_action_term.stroke_actual, env_id, unwrapped.num_envs)
    stroke_actual = leg_action_term.stroke_actual[env_id]
    _validate_env_and_tensor("robot.data.root_quat_w", robot.data.root_quat_w, env_id, unwrapped.num_envs)
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w[env_id].unsqueeze(0))
    _validate_env_and_tensor("robot.data.root_pos_w", robot.data.root_pos_w, env_id, unwrapped.num_envs)
    _validate_env_and_tensor("robot.data.root_lin_vel_w", robot.data.root_lin_vel_w, env_id, unwrapped.num_envs)
    _validate_env_and_tensor("robot.data.joint_pos", robot.data.joint_pos, env_id, unwrapped.num_envs)
    _validate_ids("hydraulic_joint_ids", hydraulic_joint_ids, robot.data.joint_pos.shape[1])

    row = {
        "requested_height": float(requested_height),
        "step": int(step),
        "terminated": int(terminated),
        "truncated": int(truncated),
        "termination_reason": termination_reason,
        "root_height": float(robot.data.root_pos_w[env_id, 2].item()),
        "root_lin_vel_z": float(robot.data.root_lin_vel_w[env_id, 2].item()),
        "base_roll": float(roll[0].item()),
        "base_pitch": float(pitch[0].item()),
        "hydraulic_stroke_lf": float(stroke_actual[0].item()),
        "hydraulic_stroke_lr": float(stroke_actual[1].item()),
        "hydraulic_stroke_rf": float(stroke_actual[2].item()),
        "hydraulic_stroke_rr": float(stroke_actual[3].item()),
        "wheel_link_world_z_lf": float(wheel_body_pos_w[0, 2].item()),
        "wheel_link_world_z_lr": float(wheel_body_pos_w[1, 2].item()),
        "wheel_link_world_z_rf": float(wheel_body_pos_w[2, 2].item()),
        "wheel_link_world_z_rr": float(wheel_body_pos_w[3, 2].item()),
        "robot_total_mass": float(robot_total_mass),
        "expected_weight_force": float(expected_weight_force),
        "wheel_contact_force_norm_lf": float(wheel_force_norm[0].item()),
        "wheel_contact_force_norm_lr": float(wheel_force_norm[1].item()),
        "wheel_contact_force_norm_rf": float(wheel_force_norm[2].item()),
        "wheel_contact_force_norm_rr": float(wheel_force_norm[3].item()),
        "wheel_contact_force_z_lf": float(wheel_force_z[0].item()),
        "wheel_contact_force_z_lr": float(wheel_force_z[1].item()),
        "wheel_contact_force_z_rf": float(wheel_force_z[2].item()),
        "wheel_contact_force_z_rr": float(wheel_force_z[3].item()),
        "wheel_contact_force_total_norm": float(wheel_total_norm.item()),
        "wheel_contact_force_total_z": float(wheel_total_z.item()),
        "wheel_contact_force_total_z_over_weight": float(wheel_total_z.item() / max(expected_weight_force, 1.0e-6)),
        "contact_bool_lf": float(contact_bool[0].item()),
        "contact_bool_lr": float(contact_bool[1].item()),
        "contact_bool_rf": float(contact_bool[2].item()),
        "contact_bool_rr": float(contact_bool[3].item()),
        "min_contact_force": float(wheel_force_norm.min().item()),
        "all_body_contact_force_total_norm": float(all_body_total_norm.item()),
        "all_body_contact_force_total_z": float(all_body_total_z.item()),
        "all_body_contact_force_total_z_over_weight": float(all_body_total_z.item() / max(expected_weight_force, 1.0e-6)),
        "non_wheel_contact_force_total_norm": float(non_wheel_total_norm.item()),
        "non_wheel_contact_force_total_z": float(non_wheel_total_z.item()),
        "non_wheel_contact_force_total_z_over_weight": float(
            non_wheel_total_z.item() / max(expected_weight_force, 1.0e-6)
        ),
        "hydraulic_joint_pos_lf": float(robot.data.joint_pos[env_id, hydraulic_joint_ids[0]].item()),
        "hydraulic_joint_pos_lr": float(robot.data.joint_pos[env_id, hydraulic_joint_ids[1]].item()),
        "hydraulic_joint_pos_rf": float(robot.data.joint_pos[env_id, hydraulic_joint_ids[2]].item()),
        "hydraulic_joint_pos_rr": float(robot.data.joint_pos[env_id, hydraulic_joint_ids[3]].item()),
    }
    return row


def _print_row(height: float, row: dict[str, Any]) -> None:
    print(
        f"[HEIGHT {height:.2f}][STEP {row['step']}] "
        f"root_height={row['root_height']:.4f} "
        f"root_lin_vel_z={row['root_lin_vel_z']:+.4f} "
        f"roll={row['base_roll']:+.4f} "
        f"pitch={row['base_pitch']:+.4f} "
        f"terminated={row['terminated']} truncated={row['truncated']} "
        f"reason={row['termination_reason'] or 'none'}",
        flush=True,
    )
    print(
        "  hydraulic_stroke(lf,lr,rf,rr)=("
        f"{row['hydraulic_stroke_lf']:.4f}, {row['hydraulic_stroke_lr']:.4f}, "
        f"{row['hydraulic_stroke_rf']:.4f}, {row['hydraulic_stroke_rr']:.4f})",
        flush=True,
    )
    print(
        "  wheel_world_z(lf,lr,rf,rr)=("
        f"{row['wheel_link_world_z_lf']:.4f}, {row['wheel_link_world_z_lr']:.4f}, "
        f"{row['wheel_link_world_z_rf']:.4f}, {row['wheel_link_world_z_rr']:.4f})",
        flush=True,
    )
    print(
        "  wheel_force_norm(lf,lr,rf,rr)=("
        f"{row['wheel_contact_force_norm_lf']:.2f}, {row['wheel_contact_force_norm_lr']:.2f}, "
        f"{row['wheel_contact_force_norm_rf']:.2f}, {row['wheel_contact_force_norm_rr']:.2f})",
        flush=True,
    )
    print(
        "  wheel_force_z(lf,lr,rf,rr)=("
        f"{row['wheel_contact_force_z_lf']:+.2f}, {row['wheel_contact_force_z_lr']:+.2f}, "
        f"{row['wheel_contact_force_z_rf']:+.2f}, {row['wheel_contact_force_z_rr']:+.2f})",
        flush=True,
    )
    print(
        f"  wheel_total_norm={row['wheel_contact_force_total_norm']:.2f} "
        f"wheel_total_z={row['wheel_contact_force_total_z']:+.2f} "
        f"wheel_total_z_over_weight={row['wheel_contact_force_total_z_over_weight']:+.4f}",
        flush=True,
    )
    print(
        f"  all_body_total_norm={row['all_body_contact_force_total_norm']:.2f} "
        f"all_body_total_z={row['all_body_contact_force_total_z']:+.2f} "
        f"all_body_total_z_over_weight={row['all_body_contact_force_total_z_over_weight']:+.4f}",
        flush=True,
    )
    print(
        f"  non_wheel_total_norm={row['non_wheel_contact_force_total_norm']:.2f} "
        f"non_wheel_total_z={row['non_wheel_contact_force_total_z']:+.2f} "
        f"non_wheel_total_z_over_weight={row['non_wheel_contact_force_total_z_over_weight']:+.4f}",
        flush=True,
    )
    print(
        "  contact_bool(lf,lr,rf,rr)=("
        f"{int(row['contact_bool_lf'])}, {int(row['contact_bool_lr'])}, "
        f"{int(row['contact_bool_rf'])}, {int(row['contact_bool_rr'])}) "
        f"min_contact_force={row['min_contact_force']:.2f}",
        flush=True,
    )


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / float(len(rows))


def _build_summary(requested_height: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_row = rows[-1]
    tail_rows = rows[-20:] if len(rows) >= 20 else rows
    summary = {
        "requested_height": float(requested_height),
        "num_rows": int(len(rows)),
        "final_step": int(final_row["step"]),
        "terminated": int(final_row["terminated"]),
        "truncated": int(final_row["truncated"]),
        "termination_reason": str(final_row["termination_reason"]),
    }
    for key in PRINT_KEYS:
        summary[f"final/{key}"] = float(final_row[key]) if isinstance(final_row[key], (int, float)) else final_row[key]
        summary[f"mean_last20/{key}"] = _mean(tail_rows, key) if isinstance(final_row[key], (int, float)) else ""
    return summary


def _run_height_case(
    env,
    env_id: int,
    requested_height: float,
    print_steps: set[int],
    num_steps: int,
    run_root: Path,
    robot_total_mass: float,
    expected_weight_force: float,
) -> dict[str, Any]:
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    wheel_contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    all_body_contact_sensor = unwrapped.scene.sensors["all_body_contact_forces"]
    zero_actions = _build_zero_actions(env)

    if int(unwrapped.num_envs) != 1:
        raise RuntimeError(f"contact_force_sanity_test expects num_envs=1 inside case, got {unwrapped.num_envs}")
    env_ids = _env_ids_tensor(env_id=env_id, num_envs=unwrapped.num_envs, device=robot.device)
    if int(env_ids.item()) != 0:
        raise RuntimeError(f"contact_force_sanity_test expects env_ids=[0], got {env_ids.detach().cpu().tolist()}")

    robot_wheel_body_ids = _robot_body_ids(robot, WHEEL_BODY_NAMES_LF_LR_RF_RR, device=robot.device)
    wheel_sensor_body_ids = _sensor_local_body_ids(
        wheel_contact_sensor,
        WHEEL_BODY_NAMES_LF_LR_RF_RR,
        device=robot.device,
        sensor_name="wheel_contact_forces",
    )
    hydraulic_joint_ids = _joint_ids(robot, HYDRAULIC_JOINT_NAMES_LF_LR_RF_RR, device=robot.device)
    all_body_sensor_names = list(all_body_contact_sensor.body_names)
    all_body_contact_ids = _sensor_local_body_ids(
        all_body_contact_sensor,
        all_body_sensor_names,
        device=robot.device,
        sensor_name="all_body_contact_forces",
    )
    non_wheel_body_names = [name for name in all_body_sensor_names if name not in WHEEL_BODY_NAME_SET]
    if non_wheel_body_names:
        non_wheel_contact_ids = _sensor_local_body_ids(
            all_body_contact_sensor,
            non_wheel_body_names,
            device=robot.device,
            sensor_name="all_body_contact_forces",
        )
    else:
        non_wheel_contact_ids = torch.empty((0,), device=robot.device, dtype=torch.long)

    _set_env_root_height(robot, env_id=env_id, target_height=float(requested_height))

    rows: list[dict[str, Any]] = []
    row0 = _collect_row(
        env,
        env_id=env_id,
        requested_height=requested_height,
        step=0,
        robot_total_mass=robot_total_mass,
        expected_weight_force=expected_weight_force,
        wheel_body_ids=wheel_sensor_body_ids,
        all_body_contact_ids=all_body_contact_ids,
        non_wheel_contact_ids=non_wheel_contact_ids,
        hydraulic_joint_ids=hydraulic_joint_ids,
        robot_wheel_body_ids=robot_wheel_body_ids,
    )
    rows.append(row0)
    if 0 in print_steps:
        _print_row(requested_height, row0)

    with torch.no_grad():
        for step in range(1, int(num_steps) + 1):
            _, _, terminated, truncated, _ = _step_env(env, zero_actions)
            terminated_flag = _as_bool(terminated, env_id)
            truncated_flag = _as_bool(truncated, env_id)
            termination_reason = _termination_reason(
                unwrapped, env_id=env_id, terminated=terminated_flag, truncated=truncated_flag
            )
            row = _collect_row(
                env,
                env_id=env_id,
                requested_height=requested_height,
                step=step,
                robot_total_mass=robot_total_mass,
                expected_weight_force=expected_weight_force,
                wheel_body_ids=wheel_sensor_body_ids,
                all_body_contact_ids=all_body_contact_ids,
                non_wheel_contact_ids=non_wheel_contact_ids,
                hydraulic_joint_ids=hydraulic_joint_ids,
                robot_wheel_body_ids=robot_wheel_body_ids,
                terminated=terminated_flag,
                truncated=truncated_flag,
                termination_reason=termination_reason,
            )
            rows.append(row)
            if step in print_steps:
                _print_row(requested_height, row)
            if terminated_flag or truncated_flag:
                break

    case_csv_path = run_root / f"height_{requested_height:.2f}.csv"
    _write_csv(case_csv_path, rows)
    summary = _build_summary(requested_height=requested_height, rows=rows)

    print(f"[HEIGHT {requested_height:.2f}] final frame", flush=True)
    _print_row(requested_height, rows[-1])
    print(f"[HEIGHT {requested_height:.2f}] last-20 mean", flush=True)
    for key in PRINT_KEYS:
        if isinstance(rows[-1][key], (int, float)):
            print(f"  {key}={_format_float(summary[f'mean_last20/{key}'])}", flush=True)

    return summary


def main() -> None:
    run_root = Path(args_cli.output_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 10.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        unwrapped = env.unwrapped
        if unwrapped.num_envs != 1:
            raise RuntimeError(f"contact_force_sanity_test expects num_envs=1, got {unwrapped.num_envs}.")
        if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
            raise ValueError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}.")

        robot = unwrapped.scene["robot"]
        wheel_contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
        all_body_contact_sensor = unwrapped.scene.sensors["all_body_contact_forces"]
        non_wheel_body_names = [name for name in all_body_contact_sensor.body_names if name not in WHEEL_BODY_NAME_SET]
        robot_total_mass = _parse_robot_total_mass(RANGER_URDF_PATH)
        expected_weight_force = robot_total_mass * 9.81

        _print_sensor_audit(wheel_contact_sensor, all_body_contact_sensor, non_wheel_body_names)
        print(
            f"[AUDIT] robot_total_mass={robot_total_mass:.4f} expected_weight_force={expected_weight_force:.4f}",
            flush=True,
        )

        summary_rows: list[dict[str, Any]] = []
        print_steps = {int(step) for step in args_cli.print_steps}
        for requested_height in args_cli.heights:
            print(f"[HEIGHT {float(requested_height):.2f}] starting static contact audit", flush=True)
            summary_rows.append(
                _run_height_case(
                    env=env,
                    env_id=args_cli.env_id,
                    requested_height=float(requested_height),
                    print_steps=print_steps,
                    num_steps=int(args_cli.num_steps),
                    run_root=run_root,
                    robot_total_mass=robot_total_mass,
                    expected_weight_force=expected_weight_force,
                )
            )

        _write_csv(run_root / "summary.csv", summary_rows)
        print(f"[DONE] wrote summary: {run_root / 'summary.csv'}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        print(f"[ERROR] contact_force_sanity_test failed: {type(exc).__name__}: {exc!r}", flush=True)
        traceback.print_exc()
        raise
