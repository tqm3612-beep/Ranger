#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run fixed Ranger hydraulic kinematics sanity tests without policy involvement."""

from __future__ import annotations

import argparse
import csv
import traceback
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run fixed Ranger hydraulic kinematics sanity tests.")
parser.add_argument("--task", type=str, default="Template-Ranger-Stand-v0", help="Task used for the sanity test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to report.")
parser.add_argument("--settle_s", type=float, default=1.0, help="Seconds to hold zero action before each case.")
parser.add_argument("--command_s", type=float, default=2.0, help="Seconds to hold the hydraulic command in each case.")
parser.add_argument("--print_every", type=int, default=15, help="Print one line every N steps.")
parser.add_argument(
    "--command_magnitude",
    type=float,
    default=1.0,
    help="Absolute normalized hydraulic command magnitude used in all cases.",
)
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/hydraulic_kinematics_sanity",
    help="Directory where CSV files will be written.",
)
parser.add_argument(
    "--include_lifted_no_contact",
    action="store_true",
    default=False,
    help="Append lifted no-contact cases to inspect relative geometry without ground support.",
)
parser.add_argument(
    "--lifted_base_z_offset",
    type=float,
    default=0.30,
    help="Extra base height applied before lifted no-contact cases.",
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
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import Ranger.tasks  # noqa: F401
from Ranger.tasks.manager_based.ranger import forward_debug_env as ranger_forward_debug_env

if not hasattr(ranger_forward_debug_env, "SceneEntityCfg"):
    ranger_forward_debug_env.SceneEntityCfg = SceneEntityCfg


WHEEL_BODY_NAMES_LF_LR_RF_RR = ["w_lf", "w_lb", "w_rf", "w_rb"]
HYDRAULIC_JOINT_NAMES_LF_LR_RF_RR = ["g_lf", "g_lb", "g_rf", "g_rb"]


def _case_definitions(command_magnitude: float, include_lifted_no_contact: bool) -> tuple[dict[str, Any], ...]:
    mag = float(command_magnitude)
    cases = [
        {"name": "all_plus", "actions": {"lf": mag, "lr": mag, "rf": mag, "rr": mag}, "lifted_base": False},
        {"name": "all_minus", "actions": {"lf": -mag, "lr": -mag, "rf": -mag, "rr": -mag}, "lifted_base": False},
        {"name": "lf_plus", "actions": {"lf": mag, "lr": 0.0, "rf": 0.0, "rr": 0.0}, "lifted_base": False},
        {"name": "lf_minus", "actions": {"lf": -mag, "lr": 0.0, "rf": 0.0, "rr": 0.0}, "lifted_base": False},
        {"name": "lr_plus", "actions": {"lf": 0.0, "lr": mag, "rf": 0.0, "rr": 0.0}, "lifted_base": False},
        {"name": "lr_minus", "actions": {"lf": 0.0, "lr": -mag, "rf": 0.0, "rr": 0.0}, "lifted_base": False},
        {"name": "rf_plus", "actions": {"lf": 0.0, "lr": 0.0, "rf": mag, "rr": 0.0}, "lifted_base": False},
        {"name": "rf_minus", "actions": {"lf": 0.0, "lr": 0.0, "rf": -mag, "rr": 0.0}, "lifted_base": False},
        {"name": "rr_plus", "actions": {"lf": 0.0, "lr": 0.0, "rf": 0.0, "rr": mag}, "lifted_base": False},
        {"name": "rr_minus", "actions": {"lf": 0.0, "lr": 0.0, "rf": 0.0, "rr": -mag}, "lifted_base": False},
        {"name": "front_plus", "actions": {"lf": mag, "lr": 0.0, "rf": mag, "rr": 0.0}, "lifted_base": False},
        {"name": "front_minus", "actions": {"lf": -mag, "lr": 0.0, "rf": -mag, "rr": 0.0}, "lifted_base": False},
        {"name": "rear_plus", "actions": {"lf": 0.0, "lr": mag, "rf": 0.0, "rr": mag}, "lifted_base": False},
        {"name": "rear_minus", "actions": {"lf": 0.0, "lr": -mag, "rf": 0.0, "rr": -mag}, "lifted_base": False},
    ]
    if include_lifted_no_contact:
        cases.extend(
            [
                {
                    "name": "lifted_no_contact_all_plus",
                    "actions": {"lf": mag, "lr": mag, "rf": mag, "rr": mag},
                    "lifted_base": True,
                },
                {
                    "name": "lifted_no_contact_all_minus",
                    "actions": {"lf": -mag, "lr": -mag, "rf": -mag, "rr": -mag},
                    "lifted_base": True,
                },
            ]
        )
    return tuple(cases)


def _get_action_slice(unwrapped_env, term_name: str) -> slice:
    start = 0
    for active_name, dim in zip(unwrapped_env.action_manager.active_terms, unwrapped_env.action_manager.action_term_dim):
        if active_name == term_name:
            return slice(start, start + dim)
        start += dim
    raise KeyError(f"Action term '{term_name}' not found in action manager.")


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


def _case_action_tensor_lr_lf_rf_rr(device: torch.device, case_actions_lf_lr_rf_rr: dict[str, float]) -> torch.Tensor:
    return torch.tensor(
        [
            float(case_actions_lf_lr_rf_rr["lr"]),
            float(case_actions_lf_lr_rf_rr["lf"]),
            float(case_actions_lf_lr_rf_rr["rf"]),
            float(case_actions_lf_lr_rf_rr["rr"]),
        ],
        device=device,
        dtype=torch.float32,
    )


def _build_case_actions(
    env,
    leg_action_slice: slice,
    wheel_action_slice: slice,
    case_actions_lf_lr_rf_rr: dict[str, float],
) -> torch.Tensor:
    unwrapped = env.unwrapped
    actions = _build_zero_actions(env)
    leg_actions = _case_action_tensor_lr_lf_rf_rr(unwrapped.device, case_actions_lf_lr_rf_rr)
    actions[:, leg_action_slice] = leg_actions.unsqueeze(0).repeat(actions.shape[0], 1)
    actions[:, wheel_action_slice] = 0.0
    return actions


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


def _peak_contact_force(contact_sensor, env_id: int, body_ids: torch.Tensor) -> torch.Tensor:
    force_history = contact_sensor.data.net_forces_w_history[env_id, :, body_ids, :]
    force_norm = torch.norm(force_history, dim=-1)
    return torch.max(force_norm, dim=0).values


def _write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(json_path: Path, payload: dict[str, Any]) -> None:
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)


def _safe_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _safe_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _all_joint_metadata_rows(robot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    joint_limits = robot.data.joint_pos_limits[0]
    soft_limits = robot.data.soft_joint_pos_limits[0]
    joint_stiffness = robot.data.joint_stiffness[0]
    joint_damping = robot.data.joint_damping[0]
    for joint_id, joint_name in enumerate(robot.joint_names):
        rows.append(
            {
                "joint_id": int(joint_id),
                "joint_name": str(joint_name),
                "limit_lower": float(joint_limits[joint_id, 0].item()),
                "limit_upper": float(joint_limits[joint_id, 1].item()),
                "soft_limit_lower": float(soft_limits[joint_id, 0].item()),
                "soft_limit_upper": float(soft_limits[joint_id, 1].item()),
                "joint_stiffness": float(joint_stiffness[joint_id].item()),
                "joint_damping": float(joint_damping[joint_id].item()),
            }
        )
    return rows


def _hydraulic_actuator_metadata(unwrapped, hydraulic_joint_ids: torch.Tensor, hydraulic_joint_names: list[str]) -> dict[str, Any]:
    robot = unwrapped.scene["robot"]
    leg_action_term = unwrapped.action_manager.get_term("leg_hydraulic")
    hydraulic_joint_id_list = _safe_int_list(hydraulic_joint_ids)
    metadata: dict[str, Any] = {
        "resolved_hydraulic_joint_names": [str(name) for name in hydraulic_joint_names],
        "resolved_hydraulic_joint_ids": hydraulic_joint_id_list,
        "all_articulation_joint_names_with_index": [
            {"joint_id": int(joint_id), "joint_name": str(joint_name)}
            for joint_id, joint_name in enumerate(robot.joint_names)
        ],
        "leg_hydraulic_cfg_joint_names": list(getattr(leg_action_term.cfg, "joint_names", [])),
        "leg_hydraulic_cfg_preserve_order": bool(getattr(leg_action_term.cfg, "preserve_order", False)),
        "leg_hydraulic_action_term_joint_names": [str(name) for name in getattr(leg_action_term, "_joint_names", [])],
        "leg_hydraulic_action_term_joint_ids": _safe_int_list(getattr(leg_action_term, "_joint_ids", [])),
        "stroke_table": _safe_list(getattr(leg_action_term, "_stroke_table", [])),
        "joint_pos_table": _safe_list(getattr(leg_action_term, "_joint_pos_table", [])),
        "joint_target_sign_lf_lr_rf_rr": _safe_list(getattr(leg_action_term, "_joint_target_sign_lf_lr_rf_rr", [])),
        "joint_target_sign_raw_order": _safe_list(getattr(leg_action_term, "_joint_target_sign", [])),
        "stroke_min": float(getattr(leg_action_term, "_stroke_min", 0.0)),
        "stroke_max": float(getattr(leg_action_term, "_stroke_max", 0.0)),
        "stroke_rate_limit": float(getattr(leg_action_term, "_stroke_rate_limit", 0.0)),
        "time_constant": float(getattr(leg_action_term, "_time_constant", 0.0)),
        "max_effort": float(getattr(leg_action_term, "_max_effort", 0.0)),
        "impedance_kp": float(getattr(leg_action_term, "_impedance_kp", 0.0)),
        "impedance_kd": float(getattr(leg_action_term, "_impedance_kd", 0.0)),
        "target_semantics": (
            "stroke increase = hydraulic cylinder extension = body side lowers; "
            "active-leg wheel_relative_z should increase with stroke."
        ),
        "hydraulic_joint_limits": [],
        "actuators": [],
    }

    joint_limits = robot.data.joint_pos_limits[0, hydraulic_joint_ids]
    soft_limits = robot.data.soft_joint_pos_limits[0, hydraulic_joint_ids]
    joint_stiffness = robot.data.joint_stiffness[0, hydraulic_joint_ids]
    joint_damping = robot.data.joint_damping[0, hydraulic_joint_ids]
    for local_idx, joint_name in enumerate(hydraulic_joint_names):
        metadata["hydraulic_joint_limits"].append(
            {
                "joint_name": str(joint_name),
                "joint_id": int(hydraulic_joint_id_list[local_idx]),
                "limit_lower": float(joint_limits[local_idx, 0].item()),
                "limit_upper": float(joint_limits[local_idx, 1].item()),
                "soft_limit_lower": float(soft_limits[local_idx, 0].item()),
                "soft_limit_upper": float(soft_limits[local_idx, 1].item()),
                "joint_stiffness": float(joint_stiffness[local_idx].item()),
                "joint_damping": float(joint_damping[local_idx].item()),
            }
        )

    for actuator_name, actuator in robot.actuators.items():
        metadata["actuators"].append(
            {
                "actuator_name": str(actuator_name),
                "joint_names": [str(name) for name in actuator.joint_names],
                "joint_indices": _safe_int_list(actuator.joint_indices),
                "stiffness": _safe_list(actuator.stiffness[0] if isinstance(actuator.stiffness, torch.Tensor) else actuator.stiffness),
                "damping": _safe_list(actuator.damping[0] if isinstance(actuator.damping, torch.Tensor) else actuator.damping),
                "effort_limit": _safe_list(
                    actuator.effort_limit[0] if isinstance(actuator.effort_limit, torch.Tensor) else actuator.effort_limit
                ),
                "velocity_limit": _safe_list(
                    actuator.velocity_limit[0] if isinstance(actuator.velocity_limit, torch.Tensor) else actuator.velocity_limit
                ),
                "is_implicit_model": bool(getattr(actuator, "is_implicit_model", False)),
            }
        )
    return metadata


def _print_resolution_info(metadata: dict[str, Any]) -> None:
    print(
        f"[RESOLVE] leg_hydraulic cfg joint_names={metadata['leg_hydraulic_cfg_joint_names']} "
        f"term joint_names={metadata['leg_hydraulic_action_term_joint_names']} "
        f"term joint_ids={metadata['leg_hydraulic_action_term_joint_ids']}",
        flush=True,
    )
    print(
        f"[RESOLVE] hydraulic joint names={metadata['resolved_hydraulic_joint_names']} "
        f"hydraulic joint ids={metadata['resolved_hydraulic_joint_ids']}",
        flush=True,
    )
    print(
        f"[RESOLVE] joint_target_sign_lf_lr_rf_rr={metadata['joint_target_sign_lf_lr_rf_rr']} "
        f"joint_target_sign_raw_order={metadata['joint_target_sign_raw_order']}",
        flush=True,
    )
    print(f"[RESOLVE] target semantics: {metadata['target_semantics']}", flush=True)
    print("[RESOLVE] articulation joints:", flush=True)
    for item in metadata["all_articulation_joint_names_with_index"]:
        print(f"  idx={item['joint_id']:02d} name={item['joint_name']}", flush=True)


def _collect_step_row(
    env,
    env_id: int,
    case_name: str,
    step: int,
    phase: str,
    wheel_body_ids: torch.Tensor,
    contact_body_ids: torch.Tensor,
    hydraulic_joint_ids: torch.Tensor,
    hydraulic_joint_names: list[str],
    commanded_leg_actions_lr_lf_rf_rr: torch.Tensor,
    lifted_base: bool,
    terminated: bool = False,
    truncated: bool = False,
    termination_reason: str = "",
) -> dict[str, Any]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    leg_action_term = unwrapped.action_manager.get_term("leg_hydraulic")
    contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]

    root_height = float(robot.data.root_pos_w[env_id, 2].item())
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w[env_id].unsqueeze(0))
    stroke_actual_lr_lf_rf_rr = leg_action_term.stroke_actual[env_id]
    joint_target_lr_lf_rf_rr = leg_action_term.position_target[env_id]
    joint_pos_lf_lr_rf_rr = robot.data.joint_pos[env_id, hydraulic_joint_ids]
    joint_vel_lf_lr_rf_rr = robot.data.joint_vel[env_id, hydraulic_joint_ids]
    articulation_joint_target_lf_lr_rf_rr = robot.data.joint_pos_target[env_id, hydraulic_joint_ids]
    articulation_joint_effort_lf_lr_rf_rr = robot.data.joint_effort_target[env_id, hydraulic_joint_ids]
    computed_torque_lf_lr_rf_rr = robot.data.computed_torque[env_id, hydraulic_joint_ids]
    applied_torque_lf_lr_rf_rr = robot.data.applied_torque[env_id, hydraulic_joint_ids]

    wheel_body_pos_w = robot.data.body_pos_w[env_id, wheel_body_ids]
    wheel_world_z_lf_lr_rf_rr = wheel_body_pos_w[:, 2]
    wheel_rel_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[env_id].unsqueeze(0),
        robot.data.root_quat_w[env_id].unsqueeze(0),
        wheel_body_pos_w.unsqueeze(0),
    )
    wheel_relative_z_lf_lr_rf_rr = wheel_rel_pos_b[0, :, 2]
    peak_contact_force_lf_lr_rf_rr = _peak_contact_force(contact_sensor, env_id, contact_body_ids)

    commanded_leg_actions_lf_lr_rf_rr = commanded_leg_actions_lr_lf_rf_rr[[1, 0, 2, 3]]
    stroke_actual_lf_lr_rf_rr = stroke_actual_lr_lf_rf_rr[[1, 0, 2, 3]]
    joint_target_lf_lr_rf_rr = joint_target_lr_lf_rf_rr[[1, 0, 2, 3]]

    contact_force_total = float(torch.sum(peak_contact_force_lf_lr_rf_rr).item())
    front_marker_z = float(0.5 * (wheel_world_z_lf_lr_rf_rr[0] + wheel_world_z_lf_lr_rf_rr[2]).item())
    rear_marker_z = float(0.5 * (wheel_world_z_lf_lr_rf_rr[1] + wheel_world_z_lf_lr_rf_rr[3]).item())

    return {
        "case": case_name,
        "step": int(step),
        "phase": phase,
        "lifted_base": int(lifted_base),
        "state_source": "live",
        "end_was_pre_reset_state": 0,
        "terminated": int(terminated),
        "truncated": int(truncated),
        "termination_reason": termination_reason,
        "hydraulic_joint_names_lf_lr_rf_rr": ",".join(str(name) for name in hydraulic_joint_names),
        "root_height": root_height,
        "base_roll": float(roll[0].item()),
        "base_pitch": float(pitch[0].item()),
        "hydraulic_action_lf": float(commanded_leg_actions_lf_lr_rf_rr[0].item()),
        "hydraulic_action_lr": float(commanded_leg_actions_lf_lr_rf_rr[1].item()),
        "hydraulic_action_rf": float(commanded_leg_actions_lf_lr_rf_rr[2].item()),
        "hydraulic_action_rr": float(commanded_leg_actions_lf_lr_rf_rr[3].item()),
        "hydraulic_stroke_lf": float(stroke_actual_lf_lr_rf_rr[0].item()),
        "hydraulic_stroke_lr": float(stroke_actual_lf_lr_rf_rr[1].item()),
        "hydraulic_stroke_rf": float(stroke_actual_lf_lr_rf_rr[2].item()),
        "hydraulic_stroke_rr": float(stroke_actual_lf_lr_rf_rr[3].item()),
        "hydraulic_joint_target_lf": float(joint_target_lf_lr_rf_rr[0].item()),
        "hydraulic_joint_target_lr": float(joint_target_lf_lr_rf_rr[1].item()),
        "hydraulic_joint_target_rf": float(joint_target_lf_lr_rf_rr[2].item()),
        "hydraulic_joint_target_rr": float(joint_target_lf_lr_rf_rr[3].item()),
        "hydraulic_joint_pos_lf": float(joint_pos_lf_lr_rf_rr[0].item()),
        "hydraulic_joint_pos_lr": float(joint_pos_lf_lr_rf_rr[1].item()),
        "hydraulic_joint_pos_rf": float(joint_pos_lf_lr_rf_rr[2].item()),
        "hydraulic_joint_pos_rr": float(joint_pos_lf_lr_rf_rr[3].item()),
        "hydraulic_joint_vel_lf": float(joint_vel_lf_lr_rf_rr[0].item()),
        "hydraulic_joint_vel_lr": float(joint_vel_lf_lr_rf_rr[1].item()),
        "hydraulic_joint_vel_rf": float(joint_vel_lf_lr_rf_rr[2].item()),
        "hydraulic_joint_vel_rr": float(joint_vel_lf_lr_rf_rr[3].item()),
        "articulation_joint_pos_target_lf": float(articulation_joint_target_lf_lr_rf_rr[0].item()),
        "articulation_joint_pos_target_lr": float(articulation_joint_target_lf_lr_rf_rr[1].item()),
        "articulation_joint_pos_target_rf": float(articulation_joint_target_lf_lr_rf_rr[2].item()),
        "articulation_joint_pos_target_rr": float(articulation_joint_target_lf_lr_rf_rr[3].item()),
        "articulation_joint_effort_target_lf": float(articulation_joint_effort_lf_lr_rf_rr[0].item()),
        "articulation_joint_effort_target_lr": float(articulation_joint_effort_lf_lr_rf_rr[1].item()),
        "articulation_joint_effort_target_rf": float(articulation_joint_effort_lf_lr_rf_rr[2].item()),
        "articulation_joint_effort_target_rr": float(articulation_joint_effort_lf_lr_rf_rr[3].item()),
        "computed_torque_lf": float(computed_torque_lf_lr_rf_rr[0].item()),
        "computed_torque_lr": float(computed_torque_lf_lr_rf_rr[1].item()),
        "computed_torque_rf": float(computed_torque_lf_lr_rf_rr[2].item()),
        "computed_torque_rr": float(computed_torque_lf_lr_rf_rr[3].item()),
        "applied_torque_lf": float(applied_torque_lf_lr_rf_rr[0].item()),
        "applied_torque_lr": float(applied_torque_lf_lr_rf_rr[1].item()),
        "applied_torque_rf": float(applied_torque_lf_lr_rf_rr[2].item()),
        "applied_torque_rr": float(applied_torque_lf_lr_rf_rr[3].item()),
        "wheel_world_z_lf": float(wheel_world_z_lf_lr_rf_rr[0].item()),
        "wheel_world_z_lr": float(wheel_world_z_lf_lr_rf_rr[1].item()),
        "wheel_world_z_rf": float(wheel_world_z_lf_lr_rf_rr[2].item()),
        "wheel_world_z_rr": float(wheel_world_z_lf_lr_rf_rr[3].item()),
        "wheel_relative_z_lf": float(wheel_relative_z_lf_lr_rf_rr[0].item()),
        "wheel_relative_z_lr": float(wheel_relative_z_lf_lr_rf_rr[1].item()),
        "wheel_relative_z_rf": float(wheel_relative_z_lf_lr_rf_rr[2].item()),
        "wheel_relative_z_rr": float(wheel_relative_z_lf_lr_rf_rr[3].item()),
        "contact_force_lf": float(peak_contact_force_lf_lr_rf_rr[0].item()),
        "contact_force_lr": float(peak_contact_force_lf_lr_rf_rr[1].item()),
        "contact_force_rf": float(peak_contact_force_lf_lr_rf_rr[2].item()),
        "contact_force_rr": float(peak_contact_force_lf_lr_rf_rr[3].item()),
        "contact_force_total": contact_force_total,
        "base_lin_vel_x": float(robot.data.root_lin_vel_b[env_id, 0].item()),
        "base_lin_vel_y": float(robot.data.root_lin_vel_b[env_id, 1].item()),
        "base_lin_vel_z": float(robot.data.root_lin_vel_b[env_id, 2].item()),
        "base_ang_vel_x": float(robot.data.root_ang_vel_b[env_id, 0].item()),
        "base_ang_vel_y": float(robot.data.root_ang_vel_b[env_id, 1].item()),
        "base_ang_vel_z": float(robot.data.root_ang_vel_b[env_id, 2].item()),
        "front_marker_z": front_marker_z,
        "rear_marker_z": rear_marker_z,
        "front_minus_rear_marker_z": front_marker_z - rear_marker_z,
    }


def _delta(start_row: dict[str, Any], end_row: dict[str, Any], key: str) -> float:
    return float(end_row[key]) - float(start_row[key])


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / float(len(rows))


def _signed_direction(value: float, eps: float = 1.0e-6) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _make_summary_row(case_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    command_rows = [row for row in rows if row["phase"] == "command"]
    start_row = command_rows[0] if command_rows else rows[0]
    end_row = command_rows[-1] if command_rows else rows[-1]
    event_rows = [row for row in rows if row["terminated"] or row["truncated"]]
    event_row = event_rows[-1] if event_rows else None

    summary = {
        "case": case_name,
        "lifted_base": int(start_row["lifted_base"]),
        "terminated": int(bool(event_row and event_row["terminated"])),
        "truncated": int(bool(event_row and event_row["truncated"])),
        "termination_reason": str(event_row["termination_reason"]) if event_row is not None else "",
        "terminated_step": int(event_row["step"]) if event_row is not None else -1,
        "end_was_pre_reset_state": int(end_row.get("end_was_pre_reset_state", 0)),
        "delta_root_height": _delta(start_row, end_row, "root_height"),
        "delta_base_roll": _delta(start_row, end_row, "base_roll"),
        "delta_base_pitch": _delta(start_row, end_row, "base_pitch"),
        "delta_contact_force_total": _delta(start_row, end_row, "contact_force_total"),
        "delta_front_marker_z": _delta(start_row, end_row, "front_marker_z"),
        "delta_rear_marker_z": _delta(start_row, end_row, "rear_marker_z"),
        "delta_front_minus_rear_marker_z": _delta(start_row, end_row, "front_minus_rear_marker_z"),
        "mean_contact_force_total": _mean(command_rows, "contact_force_total"),
    }
    for key in ("lf", "lr", "rf", "rr"):
        summary[f"delta_stroke_{key}"] = _delta(start_row, end_row, f"hydraulic_stroke_{key}")
        summary[f"delta_joint_pos_{key}"] = _delta(start_row, end_row, f"hydraulic_joint_pos_{key}")
        summary[f"delta_joint_vel_{key}"] = _delta(start_row, end_row, f"hydraulic_joint_vel_{key}")
        summary[f"delta_joint_target_term_{key}"] = _delta(start_row, end_row, f"hydraulic_joint_target_{key}")
        summary[f"delta_joint_target_articulation_{key}"] = _delta(
            start_row, end_row, f"articulation_joint_pos_target_{key}"
        )
        summary[f"delta_computed_torque_{key}"] = _delta(start_row, end_row, f"computed_torque_{key}")
        summary[f"delta_applied_torque_{key}"] = _delta(start_row, end_row, f"applied_torque_{key}")
        summary[f"delta_wheel_world_z_{key}"] = _delta(start_row, end_row, f"wheel_world_z_{key}")
        summary[f"delta_wheel_relative_z_{key}"] = _delta(start_row, end_row, f"wheel_relative_z_{key}")
        summary[f"delta_contact_force_{key}"] = _delta(start_row, end_row, f"contact_force_{key}")
        delta_stroke = float(summary[f"delta_stroke_{key}"])
        delta_rel_z = float(summary[f"delta_wheel_relative_z_{key}"])
        is_active = abs(delta_stroke) > 0.05
        summary[f"sign_check_active_{key}"] = int(is_active)
        summary[f"delta_wheel_relative_z_per_delta_stroke_{key}"] = (
            delta_rel_z / delta_stroke if abs(delta_stroke) > 1.0e-6 else float("nan")
        )
        if not is_active:
            summary[f"sign_consistent_with_stroke_increase_{key}"] = float("nan")
        else:
            summary[f"sign_consistent_with_stroke_increase_{key}"] = float(
                (delta_stroke > 0.0 and delta_rel_z > 0.0) or (delta_stroke < 0.0 and delta_rel_z < 0.0)
            )
    return summary


def _print_case_summary(case_name: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    command_rows = [row for row in rows if row["phase"] == "command"]
    start_row = command_rows[0] if command_rows else rows[0]
    end_row = command_rows[-1] if command_rows else rows[-1]
    print(
        f"[SUMMARY][{case_name}] lifted_base={summary['lifted_base']} terminated={summary['terminated']} "
        f"truncated={summary['truncated']} terminated_step={summary['terminated_step']} "
        f"end_was_pre_reset_state={summary['end_was_pre_reset_state']} "
        f"reason={summary['termination_reason'] or 'none'}",
        flush=True,
    )
    print(
        f"  root_height start={start_row['root_height']:.4f} end={end_row['root_height']:.4f} "
        f"delta={summary['delta_root_height']:+.4f}",
        flush=True,
    )
    print(
        f"  base_pitch start={start_row['base_pitch']:+.4f} end={end_row['base_pitch']:+.4f} "
        f"delta={summary['delta_base_pitch']:+.4f}",
        flush=True,
    )
    print(
        "  delta_stroke(lf,lr,rf,rr)=("
        f"{summary['delta_stroke_lf']:+.4f}, {summary['delta_stroke_lr']:+.4f}, "
        f"{summary['delta_stroke_rf']:+.4f}, {summary['delta_stroke_rr']:+.4f})",
        flush=True,
    )
    print(
        "  delta_joint_target_term(lf,lr,rf,rr)=("
        f"{summary['delta_joint_target_term_lf']:+.4f}, {summary['delta_joint_target_term_lr']:+.4f}, "
        f"{summary['delta_joint_target_term_rf']:+.4f}, {summary['delta_joint_target_term_rr']:+.4f})",
        flush=True,
    )
    print(
        "  delta_joint_target_articulation(lf,lr,rf,rr)=("
        f"{summary['delta_joint_target_articulation_lf']:+.4f}, {summary['delta_joint_target_articulation_lr']:+.4f}, "
        f"{summary['delta_joint_target_articulation_rf']:+.4f}, {summary['delta_joint_target_articulation_rr']:+.4f})",
        flush=True,
    )
    print(
        "  delta_joint_pos(lf,lr,rf,rr)=("
        f"{summary['delta_joint_pos_lf']:+.4f}, {summary['delta_joint_pos_lr']:+.4f}, "
        f"{summary['delta_joint_pos_rf']:+.4f}, {summary['delta_joint_pos_rr']:+.4f})",
        flush=True,
    )
    print(
        "  delta_applied_torque(lf,lr,rf,rr)=("
        f"{summary['delta_applied_torque_lf']:+.2f}, {summary['delta_applied_torque_lr']:+.2f}, "
        f"{summary['delta_applied_torque_rf']:+.2f}, {summary['delta_applied_torque_rr']:+.2f})",
        flush=True,
    )
    print(
        "  delta_wheel_relative_z(lf,lr,rf,rr)=("
        f"{summary['delta_wheel_relative_z_lf']:+.4f}, {summary['delta_wheel_relative_z_lr']:+.4f}, "
        f"{summary['delta_wheel_relative_z_rf']:+.4f}, {summary['delta_wheel_relative_z_rr']:+.4f})",
        flush=True,
    )
    print(
        "  delta_wheel_relative_z_per_delta_stroke(lf,lr,rf,rr)=("
        f"{summary['delta_wheel_relative_z_per_delta_stroke_lf']:+.4f}, "
        f"{summary['delta_wheel_relative_z_per_delta_stroke_lr']:+.4f}, "
        f"{summary['delta_wheel_relative_z_per_delta_stroke_rf']:+.4f}, "
        f"{summary['delta_wheel_relative_z_per_delta_stroke_rr']:+.4f})",
        flush=True,
    )
    print(
        "  sign_consistent_with_stroke_increase(lf,lr,rf,rr)=("
        f"{summary['sign_consistent_with_stroke_increase_lf']}, "
        f"{summary['sign_consistent_with_stroke_increase_lr']}, "
        f"{summary['sign_consistent_with_stroke_increase_rf']}, "
        f"{summary['sign_consistent_with_stroke_increase_rr']})",
        flush=True,
    )
    print(
        "  expected semantics: stroke increase -> wheel_relative_z increase -> body side lowers; "
        "stroke decrease -> wheel_relative_z decrease -> body side lifts",
        flush=True,
    )
    print(f"  mean_contact_force_total={summary['mean_contact_force_total']:.2f}", flush=True)
    inconsistent = [
        key
        for key in ("lf", "lr", "rf", "rr")
        if int(summary[f"sign_check_active_{key}"])
        and float(summary[f"sign_consistent_with_stroke_increase_{key}"]) < 0.5
    ]
    if inconsistent:
        print(
            f"WARNING {case_name} has inconsistent stroke/relative-z sign on legs: {','.join(inconsistent)}",
            flush=True,
        )
    if int(summary["lifted_base"]) and float(summary["mean_contact_force_total"]) > 1.0:
        print("WARNING lifted_no_contact case has contact during command phase.", flush=True)


def _run_single_case(
    env,
    case_dir: Path,
    case_cfg: dict[str, Any],
    settle_steps: int,
    command_steps: int,
) -> dict[str, Any]:
    case_name = str(case_cfg["name"])
    case_actions_lf_lr_rf_rr = dict(case_cfg["actions"])
    lifted_base = bool(case_cfg.get("lifted_base", False))

    env.reset()
    unwrapped = env.unwrapped
    if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
        raise ValueError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}.")

    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    wheel_body_ids, _ = robot.find_bodies(WHEEL_BODY_NAMES_LF_LR_RF_RR, preserve_order=True)
    contact_body_ids, _ = contact_sensor.find_bodies(WHEEL_BODY_NAMES_LF_LR_RF_RR, preserve_order=True)
    hydraulic_joint_ids, hydraulic_joint_names = robot.find_joints(HYDRAULIC_JOINT_NAMES_LF_LR_RF_RR, preserve_order=True)
    leg_action_slice = _get_action_slice(unwrapped, "leg_hydraulic")
    wheel_action_slice = _get_action_slice(unwrapped, "wheel_motor_csv")

    effective_settle_steps = settle_steps
    if lifted_base:
        root_pose = robot.data.root_pose_w.clone()
        root_pose[args_cli.env_id, 2] += max(float(args_cli.lifted_base_z_offset), 0.5)
        root_velocity = robot.data.root_vel_w.clone()
        root_velocity[args_cli.env_id] = 0.0
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(root_velocity)
        effective_settle_steps = min(settle_steps, 1)

    zero_actions = _build_zero_actions(env)
    test_actions = _build_case_actions(
        env,
        leg_action_slice=leg_action_slice,
        wheel_action_slice=wheel_action_slice,
        case_actions_lf_lr_rf_rr=case_actions_lf_lr_rf_rr,
    )
    zero_leg_actions_lr_lf_rf_rr = torch.zeros(4, device=unwrapped.device, dtype=torch.float32)
    command_leg_actions_lr_lf_rf_rr = _case_action_tensor_lr_lf_rf_rr(unwrapped.device, case_actions_lf_lr_rf_rr)

    print(
        f"[CASE] {case_name}: hydraulic(lf,lr,rf,rr)="
        f"({case_actions_lf_lr_rf_rr['lf']:+.2f}, {case_actions_lf_lr_rf_rr['lr']:+.2f}, "
        f"{case_actions_lf_lr_rf_rr['rf']:+.2f}, {case_actions_lf_lr_rf_rr['rr']:+.2f}) "
        f"lifted_base={int(lifted_base)} "
        f"[semantics: +action -> +stroke -> body side lowers]",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    total_steps = int(effective_settle_steps + command_steps)
    with torch.no_grad():
        for step in range(total_steps):
            phase = "settle" if step < effective_settle_steps else "command"
            actions = zero_actions if phase == "settle" else test_actions
            commanded_leg_actions = zero_leg_actions_lr_lf_rf_rr if phase == "settle" else command_leg_actions_lr_lf_rf_rr
            pre_step_row = _collect_step_row(
                env,
                env_id=args_cli.env_id,
                case_name=case_name,
                step=step,
                phase=phase,
                wheel_body_ids=wheel_body_ids,
                contact_body_ids=contact_body_ids,
                hydraulic_joint_ids=hydraulic_joint_ids,
                hydraulic_joint_names=hydraulic_joint_names,
                commanded_leg_actions_lr_lf_rf_rr=commanded_leg_actions,
                lifted_base=lifted_base,
                terminated=False,
                truncated=False,
                termination_reason="",
            )
            pre_step_row["state_source"] = "pre_step"
            _, _, terminated, truncated, _ = _step_env(env, actions)
            terminated_flag = _as_bool(terminated, args_cli.env_id)
            truncated_flag = _as_bool(truncated, args_cli.env_id)
            reason = _termination_reason(unwrapped, args_cli.env_id, terminated_flag, truncated_flag)
            if terminated_flag or truncated_flag:
                row = dict(pre_step_row)
                row["terminated"] = int(terminated_flag)
                row["truncated"] = int(truncated_flag)
                row["termination_reason"] = reason
                row["state_source"] = "pre_reset_terminal"
                row["end_was_pre_reset_state"] = 1
            else:
                row = _collect_step_row(
                    env,
                    env_id=args_cli.env_id,
                    case_name=case_name,
                    step=step,
                    phase=phase,
                    wheel_body_ids=wheel_body_ids,
                    contact_body_ids=contact_body_ids,
                    hydraulic_joint_ids=hydraulic_joint_ids,
                    hydraulic_joint_names=hydraulic_joint_names,
                    commanded_leg_actions_lr_lf_rf_rr=commanded_leg_actions,
                    lifted_base=lifted_base,
                    terminated=False,
                    truncated=False,
                    termination_reason="",
                )
                row["state_source"] = "post_step"
            rows.append(row)

            if step % max(args_cli.print_every, 1) == 0 or terminated_flag or truncated_flag:
                print(
                    f"[STEP {step:04d}] phase={phase} h={row['root_height']:.4f} "
                    f"roll={row['base_roll']:+.4f} pitch={row['base_pitch']:+.4f} "
                    f"stroke(lf,lr,rf,rr)=({row['hydraulic_stroke_lf']:.4f}, {row['hydraulic_stroke_lr']:.4f}, "
                    f"{row['hydraulic_stroke_rf']:.4f}, {row['hydraulic_stroke_rr']:.4f}) "
                    f"joint_target_term(lf,lr,rf,rr)=({row['hydraulic_joint_target_lf']:+.4f}, {row['hydraulic_joint_target_lr']:+.4f}, "
                    f"{row['hydraulic_joint_target_rf']:+.4f}, {row['hydraulic_joint_target_rr']:+.4f}) "
                    f"joint_target_art(lf,lr,rf,rr)=({row['articulation_joint_pos_target_lf']:+.4f}, {row['articulation_joint_pos_target_lr']:+.4f}, "
                    f"{row['articulation_joint_pos_target_rf']:+.4f}, {row['articulation_joint_pos_target_rr']:+.4f}) "
                    f"joint_pos(lf,lr,rf,rr)=({row['hydraulic_joint_pos_lf']:+.4f}, {row['hydraulic_joint_pos_lr']:+.4f}, "
                    f"{row['hydraulic_joint_pos_rf']:+.4f}, {row['hydraulic_joint_pos_rr']:+.4f}) "
                    f"applied_tau(lf,lr,rf,rr)=({row['applied_torque_lf']:+.2f}, {row['applied_torque_lr']:+.2f}, "
                    f"{row['applied_torque_rf']:+.2f}, {row['applied_torque_rr']:+.2f}) "
                    f"wheel_rel_z(lf,lr,rf,rr)=({row['wheel_relative_z_lf']:+.4f}, {row['wheel_relative_z_lr']:+.4f}, "
                    f"{row['wheel_relative_z_rf']:+.4f}, {row['wheel_relative_z_rr']:+.4f}) "
                    f"contact_total={row['contact_force_total']:.2f} term={reason or 'none'}",
                    flush=True,
                )
            if terminated_flag or truncated_flag:
                break

    timeseries_path = case_dir / f"{case_name}_timeseries.csv"
    _write_csv(timeseries_path, rows)
    summary = _make_summary_row(case_name, rows)
    _print_case_summary(case_name, rows, summary)
    print(f"[CASE] wrote timeseries: {timeseries_path}", flush=True)
    return summary


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = Path(args_cli.output_root) / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = max(float(args_cli.settle_s + args_cli.command_s + 1.0), float(env_cfg.episode_length_s))

    print(f"[INFO] Creating env for task={args_cli.task} num_envs={args_cli.num_envs}", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        print("[INFO] Env created; preparing hydraulic kinematics cases", flush=True)
        step_dt = float(env.unwrapped.step_dt)
        settle_steps = max(int(round(float(args_cli.settle_s) / max(step_dt, 1.0e-6))), 1)
        command_steps = max(int(round(float(args_cli.command_s) / max(step_dt, 1.0e-6))), 1)

        robot = env.unwrapped.scene["robot"]
        hydraulic_joint_ids, hydraulic_joint_names = robot.find_joints(
            HYDRAULIC_JOINT_NAMES_LF_LR_RF_RR, preserve_order=True
        )
        metadata = _hydraulic_actuator_metadata(env.unwrapped, hydraulic_joint_ids, hydraulic_joint_names)
        _print_resolution_info(metadata)
        _write_json(run_root / "metadata.json", metadata)
        _write_csv(run_root / "all_articulation_joints.csv", _all_joint_metadata_rows(robot))

        case_cfgs = list(_case_definitions(args_cli.command_magnitude, args_cli.include_lifted_no_contact))
        if not case_cfgs:
            raise RuntimeError("hydraulic kinematics case list is empty")
        print(f"[INFO] Running {len(case_cfgs)} hydraulic kinematics cases", flush=True)

        summary_rows: list[dict[str, Any]] = []
        for case_cfg in case_cfgs:
            print(f"[CASE] {case_cfg['name']}", flush=True)
            case_dir = run_root / str(case_cfg["name"])
            case_dir.mkdir(parents=True, exist_ok=True)
            summary_rows.append(
                _run_single_case(
                    env,
                    case_dir=case_dir,
                    case_cfg=case_cfg,
                    settle_steps=settle_steps,
                    command_steps=command_steps,
                )
            )

        summary_path = run_root / "summary.csv"
        _write_csv(summary_path, summary_rows)
        summary_by_case = {str(row["case"]): row for row in summary_rows}
        if "all_plus" in summary_by_case:
            all_plus_row = summary_by_case["all_plus"]
            all_plus_signs = {
                key: (
                    _signed_direction(float(all_plus_row[f"delta_wheel_relative_z_{key}"]))
                    if int(all_plus_row[f"sign_check_active_{key}"])
                    else 0
                )
                for key in ("lf", "lr", "rf", "rr")
            }
            active_plus_signs = [sign for sign in all_plus_signs.values() if sign != 0]
            if active_plus_signs and len(set(active_plus_signs)) > 1:
                print(f"WARNING all_plus wheel_relative_z signs are inconsistent: {all_plus_signs}", flush=True)
        if "all_plus" in summary_by_case and "all_minus" in summary_by_case:
            all_plus_row = summary_by_case["all_plus"]
            all_minus_row = summary_by_case["all_minus"]
            opposite_ok = True
            opposite_detail: dict[str, tuple[int, int]] = {}
            for key in ("lf", "lr", "rf", "rr"):
                plus_sign = (
                    _signed_direction(float(all_plus_row[f"delta_wheel_relative_z_{key}"]))
                    if int(all_plus_row[f"sign_check_active_{key}"])
                    else 0
                )
                minus_sign = (
                    _signed_direction(float(all_minus_row[f"delta_wheel_relative_z_{key}"]))
                    if int(all_minus_row[f"sign_check_active_{key}"])
                    else 0
                )
                opposite_detail[key] = (plus_sign, minus_sign)
                if plus_sign != 0 and minus_sign != 0 and plus_sign != -minus_sign:
                    opposite_ok = False
            if not opposite_ok:
                print(f"WARNING all_minus wheel_relative_z signs are not opposite of all_plus: {opposite_detail}", flush=True)
        print(f"[DONE] wrote summary: {summary_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"[ERROR] hydraulic_kinematics_sanity_test failed: {type(exc).__name__}: {exc!r}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        simulation_app.close()
