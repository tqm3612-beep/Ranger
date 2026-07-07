#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug-only sanity tests for Ranger hydraulic stroke dynamics."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import math
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

faulthandler.enable()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run debug-only Ranger stroke dynamics sanity tests.")
parser.add_argument("--task", type=str, default="Template-Ranger-Stand-v0", help="Task used for the sanity test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to report.")
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/stroke_dynamics_sanity",
    help="Directory where CSV files will be written.",
)
parser.add_argument("--static_steps", type=int, default=400, help="Steps for each fixed-stroke static case.")
parser.add_argument("--initial_settle_steps", type=int, default=200, help="Initial settle steps before dynamic phases.")
parser.add_argument("--ramp_segment_steps", type=int, default=150, help="Steps per slow-ramp segment.")
parser.add_argument("--negative_action_steps", type=int, default=250, help="Steps per constant-negative-action replay.")
parser.add_argument(
    "--base_height",
    type=float,
    default=0.85,
    help="Raised initial root height used before letting the robot settle.",
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
OPTIONAL_LEG_BODY_NAMES = ["lf_Link", "lb_Link", "rf_Link", "rb_Link"]
ACTION_ORDER_LR_LF_RF_RR = ("lr", "lf", "rf", "rr")
FIXED_STROKES = (0.50, 0.45, 0.40, 0.35)
RAMP_STROKES = (0.50, 0.48, 0.46, 0.44, 0.42, 0.40)
NEGATIVE_ACTION_MAGNITUDES = (-0.2, -0.4, -0.6)
FIXED_PRINT_STEPS = {0, 1, 2, 10, 50, 100, 200}
NEGATIVE_PRINT_STEPS = {0, 1, 2, 5, 10, 20, 50, 100, 200}
TRACKING_ERROR_THRESHOLD = 0.05
BOUNCE_VEL_THRESHOLD = 0.5
JOINT_LIMIT_MARGIN = 0.03


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


def _get_action_slice(unwrapped_env, term_name: str) -> slice:
    start = 0
    for active_name, dim in zip(unwrapped_env.action_manager.active_terms, unwrapped_env.action_manager.action_term_dim):
        if active_name == term_name:
            return slice(start, start + dim)
        start += dim
    raise KeyError(f"Action term '{term_name}' not found in action manager.")


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


def _env_ids_tensor(env_id: int, num_envs: int, device: torch.device) -> torch.Tensor:
    if not 0 <= int(env_id) < int(num_envs):
        raise RuntimeError(f"env_id out of bounds: env_id={env_id}, num_envs={num_envs}")
    env_ids = torch.tensor([int(env_id)], device=device, dtype=torch.long)
    if env_ids.ndim != 1 or env_ids.numel() != 1:
        raise RuntimeError(f"invalid env_ids tensor: {env_ids}")
    return env_ids


def _sensor_local_body_ids(sensor, body_names: list[str], *, device: torch.device, sensor_name: str) -> torch.Tensor:
    body_ids, resolved_body_names = sensor.find_bodies(body_names, preserve_order=True)
    if list(resolved_body_names) != list(body_names):
        raise RuntimeError(
            f"{sensor_name} resolved unexpected body order: requested={body_names}, resolved={list(resolved_body_names)}"
        )
    body_ids = torch.as_tensor(body_ids, device=device, dtype=torch.long)
    _validate_ids(f"{sensor_name}.body_ids", body_ids, len(sensor.body_names))
    return body_ids


def _robot_body_ids(robot, body_names: list[str], *, device: torch.device) -> torch.Tensor:
    body_ids, resolved_body_names = robot.find_bodies(body_names, preserve_order=True)
    if list(resolved_body_names) != list(body_names):
        raise RuntimeError(
            f"robot resolved unexpected body order: requested={body_names}, resolved={list(resolved_body_names)}"
        )
    body_ids = torch.as_tensor(body_ids, device=device, dtype=torch.long)
    _validate_ids("robot.body_ids", body_ids, len(robot.body_names))
    return body_ids


def _optional_robot_body_ids(robot, body_names: list[str], *, device: torch.device) -> torch.Tensor | None:
    try:
        return _robot_body_ids(robot, body_names, device=device)
    except Exception:
        return None


def _joint_ids(robot, joint_names: list[str], *, device: torch.device) -> torch.Tensor:
    joint_ids, resolved_joint_names = robot.find_joints(joint_names, preserve_order=True)
    if list(resolved_joint_names) != list(joint_names):
        raise RuntimeError(
            f"robot resolved unexpected joint order: requested={joint_names}, resolved={list(resolved_joint_names)}"
        )
    joint_ids = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
    _validate_ids("robot.joint_ids", joint_ids, len(robot.joint_names))
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


def _write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _normalized_action_from_stroke(stroke: float, leg_action_term) -> float:
    return float((stroke - leg_action_term._stroke_mid) / max(leg_action_term._stroke_half_range, 1.0e-6))


def _leg_action_tensor_lr_lf_rf_rr(device: torch.device, values_lf_lr_rf_rr: tuple[float, float, float, float]) -> torch.Tensor:
    lf, lr, rf, rr = values_lf_lr_rf_rr
    return torch.tensor([lr, lf, rf, rr], device=device, dtype=torch.float32)


def _build_actions_from_leg_command(
    env,
    leg_action_slice: slice,
    wheel_action_slice: slice,
    leg_command_lr_lf_rf_rr: torch.Tensor,
) -> torch.Tensor:
    actions = _build_zero_actions(env)
    actions[:, leg_action_slice] = leg_command_lr_lf_rf_rr.unsqueeze(0).repeat(actions.shape[0], 1)
    actions[:, wheel_action_slice] = 0.0
    return actions


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


def _initialize_uniform_stroke_target(
    leg_action_term,
    env_id: int,
    target_stroke_lf_lr_rf_rr: tuple[float, float, float, float],
) -> None:
    target_stroke_lr_lf_rf_rr = torch.tensor(
        [target_stroke_lf_lr_rf_rr[1], target_stroke_lf_lr_rf_rr[0], target_stroke_lf_lr_rf_rr[2], target_stroke_lf_lr_rf_rr[3]],
        device=leg_action_term.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    target_stroke_lr_lf_rf_rr = torch.clamp(
        target_stroke_lr_lf_rf_rr, min=leg_action_term._stroke_min, max=leg_action_term._stroke_max
    )
    position_target = leg_action_term._interp_stroke_to_joint_pos(target_stroke_lr_lf_rf_rr) * leg_action_term._joint_target_sign
    raw_actions = (target_stroke_lr_lf_rf_rr - leg_action_term._stroke_mid) / max(leg_action_term._stroke_half_range, 1.0e-6)
    raw_actions = torch.clamp(raw_actions, min=-1.0, max=1.0)
    env_ids = _env_ids_tensor(env_id=env_id, num_envs=leg_action_term.num_envs, device=leg_action_term.device)
    leg_action_term._stroke_actual[env_ids, :] = target_stroke_lr_lf_rf_rr
    leg_action_term._position_target[env_ids, :] = position_target
    leg_action_term._raw_actions[env_ids, :] = raw_actions
    leg_action_term._processed_actions[env_ids, :] = 0.0
    leg_action_term._effort_actual[env_ids, :] = 0.0
    leg_action_term._asset.set_joint_position_target(leg_action_term._position_target, joint_ids=leg_action_term._joint_ids)
    leg_action_term._asset.set_joint_effort_target(leg_action_term._processed_actions, joint_ids=leg_action_term._joint_ids)


def _current_force_vectors(contact_sensor, env_id: int, body_ids: torch.Tensor, *, sensor_name: str) -> torch.Tensor:
    force_history = contact_sensor.data.net_forces_w_history
    _validate_env_and_tensor(f"{sensor_name}.data.net_forces_w_history", force_history, env_id, force_history.shape[0])
    if force_history.ndim != 4:
        raise RuntimeError(f"{sensor_name}.data.net_forces_w_history must be 4D, got shape={tuple(force_history.shape)}")
    _validate_ids(f"{sensor_name}.body_ids", body_ids, force_history.shape[2])
    return force_history[env_id, -1, body_ids, :]


def _force_summary(force_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    force_norm = torch.norm(force_vectors, dim=-1)
    force_z = force_vectors[:, 2]
    total_norm = force_norm.sum()
    total_z = force_z.sum()
    return force_norm, force_z, total_norm, total_z


def _safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / float(len(rows))


def _collect_row(
    env,
    env_id: int,
    group_name: str,
    case_name: str,
    segment_name: str,
    step: int,
    commanded_action_lf_lr_rf_rr: tuple[float, float, float, float],
    target_stroke_lf_lr_rf_rr: tuple[float, float, float, float],
    wheel_sensor_body_ids: torch.Tensor,
    all_body_sensor_ids: torch.Tensor,
    non_wheel_sensor_ids: torch.Tensor,
    robot_wheel_body_ids: torch.Tensor,
    optional_leg_body_ids: torch.Tensor | None,
    hydraulic_joint_ids: torch.Tensor,
    robot_total_mass: float,
    expected_weight_force: float,
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
        wheel_contact_sensor, env_id, wheel_sensor_body_ids, sensor_name="wheel_contact_forces"
    )
    wheel_force_norm, wheel_force_z, wheel_total_norm, wheel_total_z = _force_summary(wheel_force_vectors)
    all_body_force_vectors = _current_force_vectors(
        all_body_contact_sensor, env_id, all_body_sensor_ids, sensor_name="all_body_contact_forces"
    )
    _, _, all_body_total_norm, all_body_total_z = _force_summary(all_body_force_vectors)
    if non_wheel_sensor_ids.numel() > 0:
        non_wheel_force_vectors = _current_force_vectors(
            all_body_contact_sensor, env_id, non_wheel_sensor_ids, sensor_name="all_body_contact_forces"
        )
        _, _, non_wheel_total_norm, non_wheel_total_z = _force_summary(non_wheel_force_vectors)
    else:
        non_wheel_total_norm = torch.tensor(0.0, device=robot.device)
        non_wheel_total_z = torch.tensor(0.0, device=robot.device)

    _validate_env_and_tensor("robot.data.body_pos_w", robot.data.body_pos_w, env_id, unwrapped.num_envs)
    _validate_ids("robot_wheel_body_ids", robot_wheel_body_ids, robot.data.body_pos_w.shape[1])
    wheel_body_pos_w = robot.data.body_pos_w[env_id, robot_wheel_body_ids, :]
    wheel_world_z = wheel_body_pos_w[:, 2]

    base_link_world_z = float("nan")
    leg_link_world_z_lf_lr_rf_rr = [float("nan")] * 4
    if optional_leg_body_ids is not None:
        leg_body_pos_w = robot.data.body_pos_w[env_id, optional_leg_body_ids, :]
        leg_link_world_z_lf_lr_rf_rr = [float(v.item()) for v in leg_body_pos_w[:, 2]]
    try:
        base_link_id = robot.body_names.index("base_link")
        base_link_world_z = float(robot.data.body_pos_w[env_id, base_link_id, 2].item())
    except Exception:
        pass

    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w[env_id].unsqueeze(0))
    stroke_actual_lr_lf_rf_rr = leg_action_term.stroke_actual[env_id]
    position_target_lr_lf_rf_rr = leg_action_term.position_target[env_id]
    effort_actual_lr_lf_rf_rr = leg_action_term.effort_actual[env_id]

    articulation_joint_pos = robot.data.joint_pos[env_id, hydraulic_joint_ids]
    articulation_joint_vel = robot.data.joint_vel[env_id, hydraulic_joint_ids]
    articulation_joint_pos_target = robot.data.joint_pos_target[env_id, hydraulic_joint_ids]
    applied_torque = robot.data.applied_torque[env_id, hydraulic_joint_ids]
    joint_limits = robot.data.joint_pos_limits[env_id, hydraulic_joint_ids, :]

    stroke_actual_lf_lr_rf_rr = stroke_actual_lr_lf_rf_rr[[1, 0, 2, 3]]
    position_target_lf_lr_rf_rr = position_target_lr_lf_rf_rr[[1, 0, 2, 3]]
    effort_actual_lf_lr_rf_rr = effort_actual_lr_lf_rf_rr[[1, 0, 2, 3]]
    articulation_joint_pos_lf_lr_rf_rr = articulation_joint_pos
    articulation_joint_vel_lf_lr_rf_rr = articulation_joint_vel
    articulation_joint_pos_target_lf_lr_rf_rr = articulation_joint_pos_target
    applied_torque_lf_lr_rf_rr = applied_torque
    joint_lower_lf_lr_rf_rr = joint_limits[:, 0]
    joint_upper_lf_lr_rf_rr = joint_limits[:, 1]
    joint_pos_minus_target = articulation_joint_pos_lf_lr_rf_rr - articulation_joint_pos_target_lf_lr_rf_rr

    commanded_action_tensor = torch.tensor(commanded_action_lf_lr_rf_rr, device=robot.device, dtype=torch.float32)
    target_stroke_tensor = torch.tensor(target_stroke_lf_lr_rf_rr, device=robot.device, dtype=torch.float32)
    contact_bool = (wheel_force_norm > 1.0).to(torch.float32)
    min_limit_margin = torch.minimum(
        articulation_joint_pos_lf_lr_rf_rr - joint_lower_lf_lr_rf_rr,
        joint_upper_lf_lr_rf_rr - articulation_joint_pos_lf_lr_rf_rr,
    )
    near_joint_limit = min_limit_margin < JOINT_LIMIT_MARGIN
    near_joint_limit_flag = bool(torch.any(near_joint_limit).item())
    action_rate = float("nan")
    if hasattr(leg_action_term, "_prev_debug_command_lf_lr_rf_rr"):
        prev_cmd = leg_action_term._prev_debug_command_lf_lr_rf_rr[env_id]
        action_rate = float(torch.max(torch.abs(commanded_action_tensor - prev_cmd)).item() / max(unwrapped.step_dt, 1.0e-6))
        leg_action_term._prev_debug_command_lf_lr_rf_rr[env_id] = commanded_action_tensor
    else:
        leg_action_term._prev_debug_command_lf_lr_rf_rr = torch.zeros(
            (unwrapped.num_envs, 4), device=robot.device, dtype=torch.float32
        )
        leg_action_term._prev_debug_command_lf_lr_rf_rr[env_id] = commanded_action_tensor

    row = {
        "group_name": group_name,
        "case_name": case_name,
        "segment_name": segment_name,
        "step": int(step),
        "target_stroke": float(target_stroke_tensor.mean().item()),
        "commanded_action_lf": float(commanded_action_tensor[0].item()),
        "commanded_action_lr": float(commanded_action_tensor[1].item()),
        "commanded_action_rf": float(commanded_action_tensor[2].item()),
        "commanded_action_rr": float(commanded_action_tensor[3].item()),
        "root_height": float(robot.data.root_pos_w[env_id, 2].item()),
        "root_lin_vel_z": float(robot.data.root_lin_vel_w[env_id, 2].item()),
        "base_roll": float(roll[0].item()),
        "base_pitch": float(pitch[0].item()),
        "terminated": int(terminated),
        "truncated": int(truncated),
        "termination_reason": termination_reason,
        "actual_stroke_lf": float(stroke_actual_lf_lr_rf_rr[0].item()),
        "actual_stroke_lr": float(stroke_actual_lf_lr_rf_rr[1].item()),
        "actual_stroke_rf": float(stroke_actual_lf_lr_rf_rr[2].item()),
        "actual_stroke_rr": float(stroke_actual_lf_lr_rf_rr[3].item()),
        "target_stroke_lf": float(target_stroke_tensor[0].item()),
        "target_stroke_lr": float(target_stroke_tensor[1].item()),
        "target_stroke_rf": float(target_stroke_tensor[2].item()),
        "target_stroke_rr": float(target_stroke_tensor[3].item()),
        "joint_pos_g_lf": float(articulation_joint_pos_lf_lr_rf_rr[0].item()),
        "joint_pos_g_lb": float(articulation_joint_pos_lf_lr_rf_rr[1].item()),
        "joint_pos_g_rf": float(articulation_joint_pos_lf_lr_rf_rr[2].item()),
        "joint_pos_g_rb": float(articulation_joint_pos_lf_lr_rf_rr[3].item()),
        "joint_target_g_lf": float(articulation_joint_pos_target_lf_lr_rf_rr[0].item()),
        "joint_target_g_lb": float(articulation_joint_pos_target_lf_lr_rf_rr[1].item()),
        "joint_target_g_rf": float(articulation_joint_pos_target_lf_lr_rf_rr[2].item()),
        "joint_target_g_rb": float(articulation_joint_pos_target_lf_lr_rf_rr[3].item()),
        "joint_pos_minus_target_g_lf": float(joint_pos_minus_target[0].item()),
        "joint_pos_minus_target_g_lb": float(joint_pos_minus_target[1].item()),
        "joint_pos_minus_target_g_rf": float(joint_pos_minus_target[2].item()),
        "joint_pos_minus_target_g_rb": float(joint_pos_minus_target[3].item()),
        "joint_vel_g_lf": float(articulation_joint_vel_lf_lr_rf_rr[0].item()),
        "joint_vel_g_lb": float(articulation_joint_vel_lf_lr_rf_rr[1].item()),
        "joint_vel_g_rf": float(articulation_joint_vel_lf_lr_rf_rr[2].item()),
        "joint_vel_g_rb": float(articulation_joint_vel_lf_lr_rf_rr[3].item()),
        "joint_effort_g_lf": float(applied_torque_lf_lr_rf_rr[0].item()),
        "joint_effort_g_lb": float(applied_torque_lf_lr_rf_rr[1].item()),
        "joint_effort_g_rf": float(applied_torque_lf_lr_rf_rr[2].item()),
        "joint_effort_g_rb": float(applied_torque_lf_lr_rf_rr[3].item()),
        "joint_limit_lower_g_lf": float(joint_lower_lf_lr_rf_rr[0].item()),
        "joint_limit_lower_g_lb": float(joint_lower_lf_lr_rf_rr[1].item()),
        "joint_limit_lower_g_rf": float(joint_lower_lf_lr_rf_rr[2].item()),
        "joint_limit_lower_g_rb": float(joint_lower_lf_lr_rf_rr[3].item()),
        "joint_limit_upper_g_lf": float(joint_upper_lf_lr_rf_rr[0].item()),
        "joint_limit_upper_g_lb": float(joint_upper_lf_lr_rf_rr[1].item()),
        "joint_limit_upper_g_rf": float(joint_upper_lf_lr_rf_rr[2].item()),
        "joint_limit_upper_g_rb": float(joint_upper_lf_lr_rf_rr[3].item()),
        "near_joint_limit_g_lf": int(near_joint_limit[0].item()),
        "near_joint_limit_g_lb": int(near_joint_limit[1].item()),
        "near_joint_limit_g_rf": int(near_joint_limit[2].item()),
        "near_joint_limit_g_rb": int(near_joint_limit[3].item()),
        "near_joint_limit": int(near_joint_limit_flag),
        "hydraulic_term_target_pos_lf": float(position_target_lf_lr_rf_rr[0].item()),
        "hydraulic_term_target_pos_lr": float(position_target_lf_lr_rf_rr[1].item()),
        "hydraulic_term_target_pos_rf": float(position_target_lf_lr_rf_rr[2].item()),
        "hydraulic_term_target_pos_rr": float(position_target_lf_lr_rf_rr[3].item()),
        "hydraulic_term_effort_lf": float(effort_actual_lf_lr_rf_rr[0].item()),
        "hydraulic_term_effort_lr": float(effort_actual_lf_lr_rf_rr[1].item()),
        "hydraulic_term_effort_rf": float(effort_actual_lf_lr_rf_rr[2].item()),
        "hydraulic_term_effort_rr": float(effort_actual_lf_lr_rf_rr[3].item()),
        "wheel_contact_force_z_lf": float(wheel_force_z[0].item()),
        "wheel_contact_force_z_lr": float(wheel_force_z[1].item()),
        "wheel_contact_force_z_rf": float(wheel_force_z[2].item()),
        "wheel_contact_force_z_rr": float(wheel_force_z[3].item()),
        "wheel_contact_force_norm_lf": float(wheel_force_norm[0].item()),
        "wheel_contact_force_norm_lr": float(wheel_force_norm[1].item()),
        "wheel_contact_force_norm_rf": float(wheel_force_norm[2].item()),
        "wheel_contact_force_norm_rr": float(wheel_force_norm[3].item()),
        "wheel_contact_force_total_z": float(wheel_total_z.item()),
        "wheel_contact_force_total_z_over_weight": float(wheel_total_z.item() / max(expected_weight_force, 1.0e-6)),
        "wheel_contact_force_total_norm": float(wheel_total_norm.item()),
        "all_body_contact_force_total_z": float(all_body_total_z.item()),
        "all_body_contact_force_total_z_over_weight": float(all_body_total_z.item() / max(expected_weight_force, 1.0e-6)),
        "non_wheel_contact_force_total_z": float(non_wheel_total_z.item()),
        "non_wheel_contact_force_total_z_over_weight": float(non_wheel_total_z.item() / max(expected_weight_force, 1.0e-6)),
        "contact_bool_lf": int(contact_bool[0].item()),
        "contact_bool_lr": int(contact_bool[1].item()),
        "contact_bool_rf": int(contact_bool[2].item()),
        "contact_bool_rr": int(contact_bool[3].item()),
        "min_contact_force": float(wheel_force_norm.min().item()),
        "wheel_link_world_z_lf": float(wheel_world_z[0].item()),
        "wheel_link_world_z_lr": float(wheel_world_z[1].item()),
        "wheel_link_world_z_rf": float(wheel_world_z[2].item()),
        "wheel_link_world_z_rr": float(wheel_world_z[3].item()),
        "base_link_world_z": base_link_world_z,
        "leg_link_world_z_lf": leg_link_world_z_lf_lr_rf_rr[0],
        "leg_link_world_z_lr": leg_link_world_z_lf_lr_rf_rr[1],
        "leg_link_world_z_rf": leg_link_world_z_lf_lr_rf_rr[2],
        "leg_link_world_z_rr": leg_link_world_z_lf_lr_rf_rr[3],
        "robot_total_mass": float(robot_total_mass),
        "expected_weight_force": float(expected_weight_force),
        "max_abs_joint_pos_minus_target": float(torch.max(torch.abs(joint_pos_minus_target)).item()),
        "max_abs_joint_vel": float(torch.max(torch.abs(articulation_joint_vel_lf_lr_rf_rr)).item()),
        "max_abs_joint_effort": float(torch.max(torch.abs(applied_torque_lf_lr_rf_rr)).item()),
        "max_abs_hydraulic_action": float(torch.max(torch.abs(commanded_action_tensor)).item()),
        "action_rate": float(action_rate),
    }
    return row


def _is_print_step(step: int, total_steps: int, print_steps: set[int]) -> bool:
    return step in print_steps or step == total_steps - 1


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"[{row['group_name']}][{row['case_name']}][{row['segment_name']}][step={row['step']}] "
        f"h={row['root_height']:.4f} vz={row['root_lin_vel_z']:+.4f} "
        f"stroke(lf,lr,rf,rr)=({row['actual_stroke_lf']:.4f}, {row['actual_stroke_lr']:.4f}, "
        f"{row['actual_stroke_rf']:.4f}, {row['actual_stroke_rr']:.4f}) "
        f"wheel_z_over_w={row['wheel_contact_force_total_z_over_weight']:+.4f} "
        f"contact=({row['contact_bool_lf']},{row['contact_bool_lr']},{row['contact_bool_rf']},{row['contact_bool_rr']}) "
        f"max|q-q*|={row['max_abs_joint_pos_minus_target']:.4f} "
        f"term={row['termination_reason'] or 'none'}",
        flush=True,
    )


def _make_case_summary(group_name: str, case_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_row = rows[-1]
    last20 = rows[-20:] if len(rows) >= 20 else rows
    min_contact_count_last20 = min(
        int(row["contact_bool_lf"]) + int(row["contact_bool_lr"]) + int(row["contact_bool_rf"]) + int(row["contact_bool_rr"])
        for row in last20
    )
    stable_contact = (
        0.8 <= _safe_mean(last20, "wheel_contact_force_total_z_over_weight") <= 1.2
        and min_contact_count_last20 == 4
        and abs(_safe_mean(last20, "root_lin_vel_z")) < 0.1
    )
    unloaded = (
        _safe_mean(last20, "wheel_contact_force_total_z_over_weight") < 0.2
        or min_contact_count_last20 < 2
    )
    max_wheel_z_over_weight = max(float(row["wheel_contact_force_total_z_over_weight"]) for row in rows)
    mean_last20_wheel_z_over_weight = _safe_mean(last20, "wheel_contact_force_total_z_over_weight")
    bouncing = (
        max(abs(float(row["root_lin_vel_z"])) for row in rows) > BOUNCE_VEL_THRESHOLD
        or (max_wheel_z_over_weight > 1.5 and mean_last20_wheel_z_over_weight < 0.2)
    )
    near_joint_limit = any(int(row["near_joint_limit"]) for row in rows)
    actuator_tracking_bad = max(float(row["max_abs_joint_pos_minus_target"]) for row in rows) > TRACKING_ERROR_THRESHOLD
    summary = {
        "group_name": group_name,
        "case_name": case_name,
        "final_step": int(final_row["step"]),
        "final_root_height": float(final_row["root_height"]),
        "final_root_lin_vel_z": float(final_row["root_lin_vel_z"]),
        "final_target_stroke": float(final_row["target_stroke"]),
        "final_wheel_contact_force_total_z_over_weight": float(final_row["wheel_contact_force_total_z_over_weight"]),
        "final_all_body_contact_force_total_z_over_weight": float(final_row["all_body_contact_force_total_z_over_weight"]),
        "final_non_wheel_contact_force_total_z_over_weight": float(final_row["non_wheel_contact_force_total_z_over_weight"]),
        "mean_last20_wheel_contact_force_total_z_over_weight": mean_last20_wheel_z_over_weight,
        "mean_last20_root_lin_vel_z": _safe_mean(last20, "root_lin_vel_z"),
        "max_abs_root_lin_vel_z": max(abs(float(row["root_lin_vel_z"])) for row in rows),
        "max_abs_joint_pos_minus_target": max(float(row["max_abs_joint_pos_minus_target"]) for row in rows),
        "max_abs_joint_vel": max(float(row["max_abs_joint_vel"]) for row in rows),
        "max_abs_joint_effort": max(float(row["max_abs_joint_effort"]) for row in rows),
        "max_abs_hydraulic_action": max(float(row["max_abs_hydraulic_action"]) for row in rows),
        "stable_contact": int(stable_contact),
        "unloaded": int(unloaded),
        "bouncing": int(bouncing),
        "near_joint_limit": int(near_joint_limit),
        "actuator_tracking_bad": int(actuator_tracking_bad),
        "terminated": int(any(int(row["terminated"]) for row in rows)),
        "truncated": int(any(int(row["truncated"]) for row in rows)),
        "termination_reason": ",".join(sorted({str(row["termination_reason"]) for row in rows if row["termination_reason"]})),
        "failure_flags": ",".join(
            flag
            for flag, active in (
                ("stable_contact", not stable_contact),
                ("unloaded", unloaded),
                ("bouncing", bouncing),
                ("near_joint_limit", near_joint_limit),
                ("actuator_tracking_bad", actuator_tracking_bad),
            )
            if active
        ),
    }
    return summary


def _print_summary_table(group_name: str, summaries: list[dict[str, Any]]) -> None:
    print(f"\n[SUMMARY TABLE] {group_name}", flush=True)
    for summary in summaries:
        print(
            f"  {summary['case_name']}: "
            f"stable_contact={summary['stable_contact']} "
            f"unloaded={summary['unloaded']} "
            f"bouncing={summary['bouncing']} "
            f"near_joint_limit={summary['near_joint_limit']} "
            f"actuator_tracking_bad={summary['actuator_tracking_bad']} "
            f"final_h={summary['final_root_height']:.4f} "
            f"final_wz/weight={summary['final_wheel_contact_force_total_z_over_weight']:+.4f} "
            f"reason={summary['termination_reason'] or 'none'} "
            f"failures={summary['failure_flags'] or 'none'}",
            flush=True,
        )


def _run_steps_with_actions(
    env,
    total_steps: int,
    print_steps: set[int],
    group_name: str,
    case_name: str,
    segment_name_fn,
    action_tensor_fn,
    commanded_action_fn,
    target_stroke_fn,
    static_context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for step in range(total_steps):
            actions = action_tensor_fn(step)
            _, _, terminated, truncated, _ = _step_env(env, actions)
            terminated_flag = _as_bool(terminated, static_context["env_id"])
            truncated_flag = _as_bool(truncated, static_context["env_id"])
            reason = _termination_reason(env.unwrapped, static_context["env_id"], terminated_flag, truncated_flag)
            row = _collect_row(
                env,
                env_id=static_context["env_id"],
                group_name=group_name,
                case_name=case_name,
                segment_name=segment_name_fn(step),
                step=step,
                commanded_action_lf_lr_rf_rr=commanded_action_fn(step),
                target_stroke_lf_lr_rf_rr=target_stroke_fn(step),
                wheel_sensor_body_ids=static_context["wheel_sensor_body_ids"],
                all_body_sensor_ids=static_context["all_body_sensor_ids"],
                non_wheel_sensor_ids=static_context["non_wheel_sensor_ids"],
                robot_wheel_body_ids=static_context["robot_wheel_body_ids"],
                optional_leg_body_ids=static_context["optional_leg_body_ids"],
                hydraulic_joint_ids=static_context["hydraulic_joint_ids"],
                robot_total_mass=static_context["robot_total_mass"],
                expected_weight_force=static_context["expected_weight_force"],
                terminated=terminated_flag,
                truncated=truncated_flag,
                termination_reason=reason,
            )
            rows.append(row)
            if _is_print_step(step, total_steps, print_steps):
                _print_row(row)
            if terminated_flag or truncated_flag:
                break
    return rows


def _build_static_context(env):
    env.reset()
    unwrapped = env.unwrapped
    if int(unwrapped.num_envs) != 1:
        raise RuntimeError(f"stroke_dynamics_sanity_test expects num_envs=1, got {unwrapped.num_envs}")
    if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
        raise RuntimeError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}")
    robot = unwrapped.scene["robot"]
    wheel_contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    all_body_contact_sensor = unwrapped.scene.sensors["all_body_contact_forces"]
    leg_action_term = unwrapped.action_manager.get_term("leg_hydraulic")
    static_context = {
        "env_id": args_cli.env_id,
        "robot": robot,
        "wheel_sensor_body_ids": _sensor_local_body_ids(
            wheel_contact_sensor, WHEEL_BODY_NAMES_LF_LR_RF_RR, device=robot.device, sensor_name="wheel_contact_forces"
        ),
        "all_body_sensor_ids": _sensor_local_body_ids(
            all_body_contact_sensor,
            list(all_body_contact_sensor.body_names),
            device=robot.device,
            sensor_name="all_body_contact_forces",
        ),
        "non_wheel_sensor_ids": _sensor_local_body_ids(
            all_body_contact_sensor,
            [name for name in all_body_contact_sensor.body_names if name not in WHEEL_BODY_NAME_SET],
            device=robot.device,
            sensor_name="all_body_contact_forces",
        )
        if any(name not in WHEEL_BODY_NAME_SET for name in all_body_contact_sensor.body_names)
        else torch.empty((0,), device=robot.device, dtype=torch.long),
        "robot_wheel_body_ids": _robot_body_ids(robot, WHEEL_BODY_NAMES_LF_LR_RF_RR, device=robot.device),
        "optional_leg_body_ids": _optional_robot_body_ids(robot, OPTIONAL_LEG_BODY_NAMES, device=robot.device),
        "hydraulic_joint_ids": _joint_ids(robot, HYDRAULIC_JOINT_NAMES_LF_LR_RF_RR, device=robot.device),
        "leg_action_term": leg_action_term,
        "leg_action_slice": _get_action_slice(unwrapped, "leg_hydraulic"),
        "wheel_action_slice": _get_action_slice(unwrapped, "wheel_motor_csv"),
        "robot_total_mass": _parse_robot_total_mass(RANGER_URDF_PATH),
    }
    static_context["expected_weight_force"] = static_context["robot_total_mass"] * 9.81
    static_context["zero_actions"] = _build_zero_actions(env)
    return static_context


def _run_fixed_stroke_group(env, run_root: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
    group_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for target_stroke in FIXED_STROKES:
        env.reset()
        _set_env_root_height(context["robot"], context["env_id"], args_cli.base_height)
        target_tuple = (target_stroke, target_stroke, target_stroke, target_stroke)
        _initialize_uniform_stroke_target(context["leg_action_term"], context["env_id"], target_tuple)
        action_value = _normalized_action_from_stroke(target_stroke, context["leg_action_term"])
        commanded_action = (action_value, action_value, action_value, action_value)
        action_lr_lf_rf_rr = _leg_action_tensor_lr_lf_rf_rr(env.unwrapped.device, commanded_action)
        actions = _build_actions_from_leg_command(
            env,
            context["leg_action_slice"],
            context["wheel_action_slice"],
            action_lr_lf_rf_rr,
        )
        case_name = f"fixed_stroke_{target_stroke:.2f}"
        print(f"[CASE] {case_name}", flush=True)
        rows = _run_steps_with_actions(
            env=env,
            total_steps=int(args_cli.static_steps),
            print_steps=FIXED_PRINT_STEPS,
            group_name="fixed_stroke",
            case_name=case_name,
            segment_name_fn=lambda _step: "hold",
            action_tensor_fn=lambda _step, actions=actions: actions,
            commanded_action_fn=lambda _step, commanded_action=commanded_action: commanded_action,
            target_stroke_fn=lambda _step, target_tuple=target_tuple: target_tuple,
            static_context=context,
        )
        group_rows.extend(rows)
        summary = _make_case_summary("fixed_stroke", case_name, rows)
        summaries.append(summary)
    _write_csv(run_root / "fixed_stroke_timeseries.csv", group_rows)
    _print_summary_table("fixed_stroke", summaries)
    return summaries


def _run_slow_ramp_group(env, run_root: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
    env.reset()
    _set_env_root_height(context["robot"], context["env_id"], args_cli.base_height)
    initial_tuple = (0.50, 0.50, 0.50, 0.50)
    _initialize_uniform_stroke_target(context["leg_action_term"], context["env_id"], initial_tuple)
    segment_targets = [initial_tuple] + [(s, s, s, s) for s in RAMP_STROKES[1:]]
    segment_steps = [int(args_cli.initial_settle_steps)] + [int(args_cli.ramp_segment_steps)] * (len(segment_targets) - 1)
    cumulative_steps: list[int] = []
    total = 0
    for steps in segment_steps:
        total += steps
        cumulative_steps.append(total)

    def segment_index(step: int) -> int:
        for idx, end_step in enumerate(cumulative_steps):
            if step < end_step:
                return idx
        return len(cumulative_steps) - 1

    def current_target(step: int) -> tuple[float, float, float, float]:
        return segment_targets[segment_index(step)]

    def current_segment_name(step: int) -> str:
        idx = segment_index(step)
        if idx == 0:
            return "settle_0.50"
        return f"ramp_to_{segment_targets[idx][0]:.2f}"

    def current_actions(step: int):
        target = current_target(step)
        action_value = _normalized_action_from_stroke(target[0], context["leg_action_term"])
        command = (action_value, action_value, action_value, action_value)
        action_lr_lf_rf_rr = _leg_action_tensor_lr_lf_rf_rr(env.unwrapped.device, command)
        return _build_actions_from_leg_command(env, context["leg_action_slice"], context["wheel_action_slice"], action_lr_lf_rf_rr)

    def current_command(step: int) -> tuple[float, float, float, float]:
        target = current_target(step)
        action_value = _normalized_action_from_stroke(target[0], context["leg_action_term"])
        return (action_value, action_value, action_value, action_value)

    print("[CASE] slow_ramp_0.50_to_0.40", flush=True)
    rows = _run_steps_with_actions(
        env=env,
        total_steps=cumulative_steps[-1],
        print_steps=FIXED_PRINT_STEPS,
        group_name="slow_ramp",
        case_name="slow_ramp_0.50_to_0.40",
        segment_name_fn=current_segment_name,
        action_tensor_fn=current_actions,
        commanded_action_fn=current_command,
        target_stroke_fn=current_target,
        static_context=context,
    )
    _write_csv(run_root / "slow_ramp_timeseries.csv", rows)

    summaries: list[dict[str, Any]] = []
    start = 0
    for idx, end in enumerate(cumulative_steps):
        segment_rows = rows[start:end]
        if not segment_rows:
            break
        segment_case_name = current_segment_name(start)
        summary = _make_case_summary("slow_ramp", segment_case_name, segment_rows)
        summaries.append(summary)
        start = end
    _print_summary_table("slow_ramp", summaries)
    return summaries


def _run_negative_action_group(env, run_root: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
    group_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for action_value in NEGATIVE_ACTION_MAGNITUDES:
        env.reset()
        _set_env_root_height(context["robot"], context["env_id"], args_cli.base_height)
        initial_tuple = (0.50, 0.50, 0.50, 0.50)
        _initialize_uniform_stroke_target(context["leg_action_term"], context["env_id"], initial_tuple)
        settle_action_lr_lf_rf_rr = _leg_action_tensor_lr_lf_rf_rr(env.unwrapped.device, (0.0, 0.0, 0.0, 0.0))
        replay_action = (action_value, action_value, action_value, action_value)
        replay_action_lr_lf_rf_rr = _leg_action_tensor_lr_lf_rf_rr(env.unwrapped.device, replay_action)
        settle_actions = _build_actions_from_leg_command(
            env, context["leg_action_slice"], context["wheel_action_slice"], settle_action_lr_lf_rf_rr
        )
        replay_actions = _build_actions_from_leg_command(
            env, context["leg_action_slice"], context["wheel_action_slice"], replay_action_lr_lf_rf_rr
        )
        total_steps = int(args_cli.initial_settle_steps + args_cli.negative_action_steps)
        case_name = f"negative_action_{action_value:+.1f}"
        print(f"[CASE] {case_name}", flush=True)

        def actions_for_step(step: int):
            return settle_actions if step < int(args_cli.initial_settle_steps) else replay_actions

        def command_for_step(step: int):
            return (0.0, 0.0, 0.0, 0.0) if step < int(args_cli.initial_settle_steps) else replay_action

        def target_stroke_for_step(step: int):
            if step < int(args_cli.initial_settle_steps):
                return initial_tuple
            stroke_target = 0.5 + 0.5 * action_value
            return (stroke_target, stroke_target, stroke_target, stroke_target)

        def segment_name(step: int):
            return "settle_0.50" if step < int(args_cli.initial_settle_steps) else "negative_action_replay"

        rows = _run_steps_with_actions(
            env=env,
            total_steps=total_steps,
            print_steps=NEGATIVE_PRINT_STEPS,
            group_name="negative_action",
            case_name=case_name,
            segment_name_fn=segment_name,
            action_tensor_fn=actions_for_step,
            commanded_action_fn=command_for_step,
            target_stroke_fn=target_stroke_for_step,
            static_context=context,
        )
        group_rows.extend(rows)
        summary = _make_case_summary("negative_action", case_name, rows)
        summaries.append(summary)
    _write_csv(run_root / "negative_action_timeseries.csv", group_rows)
    _print_summary_table("negative_action", summaries)
    return summaries


def main() -> None:
    run_root = Path(args_cli.output_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 12.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        context = _build_static_context(env)
        print(
            f"[AUDIT] robot_total_mass={context['robot_total_mass']:.4f} "
            f"expected_weight_force={context['expected_weight_force']:.4f}",
            flush=True,
        )
        print(
            f"[AUDIT] wheel_contact_sensor.body_names={list(env.unwrapped.scene.sensors['wheel_contact_forces'].body_names)}",
            flush=True,
        )
        print(
            f"[AUDIT] all_body_contact_sensor.body_names={list(env.unwrapped.scene.sensors['all_body_contact_forces'].body_names)}",
            flush=True,
        )

        summary_rows: list[dict[str, Any]] = []
        summary_rows.extend(_run_fixed_stroke_group(env, run_root, context))
        summary_rows.extend(_run_slow_ramp_group(env, run_root, context))
        summary_rows.extend(_run_negative_action_group(env, run_root, context))
        _write_csv(run_root / "summary.csv", summary_rows)
        print(f"[DONE] wrote summary: {run_root / 'summary.csv'}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        print(f"[ERROR] stroke_dynamics_sanity_test failed: {type(exc).__name__}: {exc!r}", flush=True)
        traceback.print_exc()
        raise
