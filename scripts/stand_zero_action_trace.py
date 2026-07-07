#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a standalone Stand zero-action trace without PPO or train.py."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import traceback
from pathlib import Path
from typing import Any

faulthandler.enable()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a zero-action trace for Template-Ranger-Stand-v0.")
parser.add_argument("--task", type=str, default="Template-Ranger-Stand-v0", help="Task used for the trace.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to create.")
parser.add_argument("--steps", type=int, default=500, help="Number of zero-action rollout steps after reset.")
parser.add_argument(
    "--output",
    type=str,
    default="logs/stand_trace/stand_zero_action_trace_env0.csv",
    help="CSV path for the env0 trace output.",
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


WHEEL_BODY_NAMES_LF_LR_RF_RR = ["w_lf", "w_lb", "w_rf", "w_rb"]
WHEEL_BODY_NAME_SET = {"w_lb", "w_lf", "w_rf", "w_rb"}
HYDRAULIC_JOINT_NAMES_LF_LR_RF_RR = ["g_lf", "g_lb", "g_rf", "g_rb"]


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
    _validate_ids(f"{sensor_name}.body_ids", sensor_body_ids, len(sensor.body_names))
    if list(resolved_body_names) != list(body_names):
        raise RuntimeError(
            f"{sensor_name} resolved unexpected body order: requested={body_names}, resolved={list(resolved_body_names)}"
        )
    return sensor_body_ids


def _robot_body_ids(robot, body_names: list[str], *, device: torch.device) -> torch.Tensor:
    body_ids, resolved_body_names = robot.find_bodies(body_names, preserve_order=True)
    body_ids = torch.as_tensor(body_ids, device=device, dtype=torch.long)
    _validate_ids("robot.body_ids", body_ids, len(robot.body_names))
    if list(resolved_body_names) != list(body_names):
        raise RuntimeError(
            f"robot resolved unexpected body order: requested={body_names}, resolved={list(resolved_body_names)}"
        )
    return body_ids


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


def _termination_reason(unwrapped, env_id: int, terminated: bool, truncated: bool) -> str:
    reasons: list[str] = []
    if truncated:
        reasons.append("truncated")
    if terminated:
        for term_name in getattr(unwrapped.termination_manager, "active_terms", ()):
            term_value = unwrapped.termination_manager.get_term(term_name)
            if _as_bool(term_value, env_id):
                reasons.append(term_name)
    return "|".join(reasons) if reasons else ""


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


def _ground_top_z(unwrapped) -> float:
    ground_cfg = getattr(unwrapped.cfg.scene, "ground", None)
    if ground_cfg is None:
        return 0.0
    ground_pos_z = float(getattr(getattr(ground_cfg, "init_state", None), "pos", (0.0, 0.0, 0.0))[2])
    ground_size_z = float(getattr(getattr(ground_cfg, "spawn", None), "size", (0.0, 0.0, 0.0))[2])
    return ground_pos_z + 0.5 * ground_size_z


def _collect_row(
    env,
    env_id: int,
    trace_step: int,
    wheel_sensor_body_ids: torch.Tensor,
    all_body_contact_ids: torch.Tensor,
    non_wheel_contact_ids: torch.Tensor,
    robot_wheel_body_ids: torch.Tensor,
    base_link_body_ids: torch.Tensor,
    hydraulic_joint_ids: torch.Tensor,
    expected_weight_force: float,
    terminated: bool = False,
    truncated: bool = False,
    termination_reason: str = "",
) -> dict[str, Any]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    leg_action_term = unwrapped.action_manager.get_term("leg_hydraulic")
    wheel_contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    all_body_contact_sensor = unwrapped.scene.sensors["all_body_contact_forces"]

    wheel_force_vectors = _current_force_vectors(
        wheel_contact_sensor, env_id, wheel_sensor_body_ids, sensor_name="wheel_contact_forces"
    )
    wheel_force_norm, wheel_force_z, wheel_total_norm, wheel_total_z = _force_summary(wheel_force_vectors)

    if non_wheel_contact_ids.numel() > 0:
        non_wheel_force_vectors = _current_force_vectors(
            all_body_contact_sensor, env_id, non_wheel_contact_ids, sensor_name="all_body_contact_forces"
        )
        non_wheel_force_norm, non_wheel_force_z, non_wheel_total_norm, non_wheel_total_z = _force_summary(
            non_wheel_force_vectors
        )
    else:
        non_wheel_force_norm = torch.zeros((0,), device=robot.device, dtype=torch.float32)
        non_wheel_force_z = torch.zeros((0,), device=robot.device, dtype=torch.float32)
        non_wheel_total_norm = torch.tensor(0.0, device=robot.device, dtype=torch.float32)
        non_wheel_total_z = torch.tensor(0.0, device=robot.device, dtype=torch.float32)

    _validate_env_and_tensor("robot.data.body_pos_w", robot.data.body_pos_w, env_id, unwrapped.num_envs)
    _validate_ids("robot_wheel_body_ids", robot_wheel_body_ids, robot.data.body_pos_w.shape[1])
    _validate_ids("base_link_body_ids", base_link_body_ids, robot.data.body_pos_w.shape[1])
    wheel_body_pos_w = robot.data.body_pos_w[env_id, robot_wheel_body_ids, :]
    base_link_world_z = robot.data.body_pos_w[env_id, base_link_body_ids[0], 2]

    _validate_env_and_tensor("robot.data.root_pos_w", robot.data.root_pos_w, env_id, unwrapped.num_envs)
    _validate_env_and_tensor("robot.data.root_lin_vel_b", robot.data.root_lin_vel_b, env_id, unwrapped.num_envs)
    _validate_env_and_tensor("robot.data.root_quat_w", robot.data.root_quat_w, env_id, unwrapped.num_envs)
    _validate_env_and_tensor("leg_action_term.stroke_actual", leg_action_term.stroke_actual, env_id, unwrapped.num_envs)
    _validate_env_and_tensor("leg_action_term.stroke_desired", leg_action_term.stroke_desired, env_id, unwrapped.num_envs)
    _validate_ids("hydraulic_joint_ids", hydraulic_joint_ids, robot.data.joint_pos.shape[1])

    root_pos_w = robot.data.root_pos_w[env_id]
    wheel_relative_pos = wheel_body_pos_w - root_pos_w.unsqueeze(0)
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w[env_id].unsqueeze(0))
    stroke_actual = leg_action_term.stroke_actual[env_id]
    stroke_desired = leg_action_term.stroke_desired[env_id]
    env_origin = unwrapped.scene.env_origins[env_id]
    terrain_height = _ground_top_z(unwrapped)
    contact_bool = (wheel_force_norm > 1.0).to(torch.float32)
    episode_step = int(unwrapped.episode_length_buf[env_id].item())

    return {
        "trace_step": int(trace_step),
        "episode_step": episode_step,
        "root_height": float(root_pos_w[2].item()),
        "root_world_z": float(root_pos_w[2].item()),
        "root_lin_vel_z": float(robot.data.root_lin_vel_b[env_id, 2].item()),
        "env_origin_x": float(env_origin[0].item()),
        "env_origin_y": float(env_origin[1].item()),
        "env_origin_z": float(env_origin[2].item()),
        "terrain_height_under_robot": float(terrain_height),
        "root_world_z_minus_env_origin_z": float((root_pos_w[2] - env_origin[2]).item()),
        "root_world_z_minus_terrain_height": float((root_pos_w[2] - terrain_height).item()),
        "base_link_world_z": float(base_link_world_z.item()),
        "wheel_world_z_lf": float(wheel_body_pos_w[0, 2].item()),
        "wheel_world_z_lr": float(wheel_body_pos_w[1, 2].item()),
        "wheel_world_z_rf": float(wheel_body_pos_w[2, 2].item()),
        "wheel_world_z_rr": float(wheel_body_pos_w[3, 2].item()),
        "wheel_relative_z_lf": float(wheel_relative_pos[0, 2].item()),
        "wheel_relative_z_lr": float(wheel_relative_pos[1, 2].item()),
        "wheel_relative_z_rf": float(wheel_relative_pos[2, 2].item()),
        "wheel_relative_z_rr": float(wheel_relative_pos[3, 2].item()),
        "stroke_lf": float(stroke_actual[1].item()),
        "stroke_lr": float(stroke_actual[0].item()),
        "stroke_rf": float(stroke_actual[2].item()),
        "stroke_rr": float(stroke_actual[3].item()),
        "target_stroke_lf": float(stroke_desired[1].item()),
        "target_stroke_lr": float(stroke_desired[0].item()),
        "target_stroke_rf": float(stroke_desired[2].item()),
        "target_stroke_rr": float(stroke_desired[3].item()),
        "wheel_contact_force_norm_lf": float(wheel_force_norm[0].item()),
        "wheel_contact_force_norm_lr": float(wheel_force_norm[1].item()),
        "wheel_contact_force_norm_rf": float(wheel_force_norm[2].item()),
        "wheel_contact_force_norm_rr": float(wheel_force_norm[3].item()),
        "wheel_contact_force_z_lf": float(wheel_force_z[0].item()),
        "wheel_contact_force_z_lr": float(wheel_force_z[1].item()),
        "wheel_contact_force_z_rf": float(wheel_force_z[2].item()),
        "wheel_contact_force_z_rr": float(wheel_force_z[3].item()),
        "wheel_contact_force_norm_over_weight_lf": float(wheel_force_norm[0].item() / max(expected_weight_force, 1.0e-6)),
        "wheel_contact_force_norm_over_weight_lr": float(wheel_force_norm[1].item() / max(expected_weight_force, 1.0e-6)),
        "wheel_contact_force_norm_over_weight_rf": float(wheel_force_norm[2].item() / max(expected_weight_force, 1.0e-6)),
        "wheel_contact_force_norm_over_weight_rr": float(wheel_force_norm[3].item() / max(expected_weight_force, 1.0e-6)),
        "wheel_contact_force_z_over_weight_lf": float(wheel_force_z[0].item() / max(expected_weight_force, 1.0e-6)),
        "wheel_contact_force_z_over_weight_lr": float(wheel_force_z[1].item() / max(expected_weight_force, 1.0e-6)),
        "wheel_contact_force_z_over_weight_rf": float(wheel_force_z[2].item() / max(expected_weight_force, 1.0e-6)),
        "wheel_contact_force_z_over_weight_rr": float(wheel_force_z[3].item() / max(expected_weight_force, 1.0e-6)),
        "wheel_contact_force_total_norm_over_weight": float(wheel_total_norm.item() / max(expected_weight_force, 1.0e-6)),
        "wheel_contact_force_total_z_over_weight": float(wheel_total_z.item() / max(expected_weight_force, 1.0e-6)),
        "wheel_contact_force_over_weight": float(wheel_total_norm.item() / max(expected_weight_force, 1.0e-6)),
        "non_wheel_contact_force_total_norm_over_weight": float(
            non_wheel_total_norm.item() / max(expected_weight_force, 1.0e-6)
        ),
        "non_wheel_contact_force_total_z_over_weight": float(
            non_wheel_total_z.item() / max(expected_weight_force, 1.0e-6)
        ),
        "non_wheel_contact_force_over_weight": float(non_wheel_total_norm.item() / max(expected_weight_force, 1.0e-6)),
        "contact_bool_lf": int(contact_bool[0].item()),
        "contact_bool_lr": int(contact_bool[1].item()),
        "contact_bool_rf": int(contact_bool[2].item()),
        "contact_bool_rr": int(contact_bool[3].item()),
        "termination_any": int(bool(terminated or truncated)),
        "termination_reason": termination_reason,
    }


def _write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / float(len(rows))


def main() -> None:
    output_path = Path(args_cli.output)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        env.reset()
        unwrapped = env.unwrapped
        env_id = 0

        zero_actions = _build_zero_actions(env)
        robot = unwrapped.scene["robot"]
        wheel_contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
        all_body_contact_sensor = unwrapped.scene.sensors["all_body_contact_forces"]

        wheel_sensor_body_ids = _sensor_local_body_ids(
            wheel_contact_sensor, WHEEL_BODY_NAMES_LF_LR_RF_RR, device=robot.device, sensor_name="wheel_contact_forces"
        )
        robot_wheel_body_ids = _robot_body_ids(robot, WHEEL_BODY_NAMES_LF_LR_RF_RR, device=robot.device)
        base_link_body_ids = _robot_body_ids(robot, ["base_link"], device=robot.device)
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

        expected_weight_force = float(getattr(unwrapped, "_stand_debug_expected_weight_force", 0.0))
        if expected_weight_force <= 0.0:
            raise RuntimeError("Expected positive _stand_debug_expected_weight_force from environment.")

        rows: list[dict[str, Any]] = []
        rows.append(
            _collect_row(
                env=env,
                env_id=env_id,
                trace_step=0,
                wheel_sensor_body_ids=wheel_sensor_body_ids,
                all_body_contact_ids=all_body_contact_ids,
                non_wheel_contact_ids=non_wheel_contact_ids,
                robot_wheel_body_ids=robot_wheel_body_ids,
                base_link_body_ids=base_link_body_ids,
                hydraulic_joint_ids=hydraulic_joint_ids,
                expected_weight_force=expected_weight_force,
            )
        )

        with torch.no_grad():
            for trace_step in range(1, int(args_cli.steps) + 1):
                _, _, terminated, truncated, _ = _step_env(env, zero_actions)
                terminated_flag = _as_bool(terminated, env_id)
                truncated_flag = _as_bool(truncated, env_id)
                termination_reason = _termination_reason(
                    unwrapped, env_id=env_id, terminated=terminated_flag, truncated=truncated_flag
                )
                rows.append(
                    _collect_row(
                        env=env,
                        env_id=env_id,
                        trace_step=trace_step,
                        wheel_sensor_body_ids=wheel_sensor_body_ids,
                        all_body_contact_ids=all_body_contact_ids,
                        non_wheel_contact_ids=non_wheel_contact_ids,
                        robot_wheel_body_ids=robot_wheel_body_ids,
                        base_link_body_ids=base_link_body_ids,
                        hydraulic_joint_ids=hydraulic_joint_ids,
                        expected_weight_force=expected_weight_force,
                        terminated=terminated_flag,
                        truncated=truncated_flag,
                        termination_reason=termination_reason,
                    )
                )
                if terminated_flag or truncated_flag:
                    break

        _write_csv(output_path, rows)

        last50_rows = rows[-50:] if len(rows) >= 50 else rows
        first_contact_step = next(
            (int(row["trace_step"]) for row in rows if float(row["wheel_contact_force_over_weight"]) > 0.0),
            -1,
        )
        first_non_wheel_contact_step = next(
            (int(row["trace_step"]) for row in rows if float(row["non_wheel_contact_force_over_weight"]) > 0.0),
            -1,
        )
        print(f"[SUMMARY] first_contact_step={first_contact_step}", flush=True)
        print(f"[SUMMARY] first_non_wheel_contact_step={first_non_wheel_contact_step}", flush=True)
        print(
            f"[SUMMARY] max_wheel_contact_over_weight={max(float(row['wheel_contact_force_over_weight']) for row in rows):.6f}",
            flush=True,
        )
        print(
            f"[SUMMARY] max_non_wheel_contact_over_weight={max(float(row['non_wheel_contact_force_over_weight']) for row in rows):.6f}",
            flush=True,
        )
        print(f"[SUMMARY] root_height_first={float(rows[0]['root_height']):.6f}", flush=True)
        print(f"[SUMMARY] root_height_min={min(float(row['root_height']) for row in rows):.6f}", flush=True)
        print(f"[SUMMARY] root_height_max={max(float(row['root_height']) for row in rows):.6f}", flush=True)
        print(f"[SUMMARY] root_height_last50_mean={_mean(last50_rows, 'root_height'):.6f}", flush=True)
        print(f"[SUMMARY] root_lin_vel_z_last50_mean={_mean(last50_rows, 'root_lin_vel_z'):.6f}", flush=True)
        print(
            f"[SUMMARY] wheel_contact_over_weight_last50_mean={_mean(last50_rows, 'wheel_contact_force_over_weight'):.6f}",
            flush=True,
        )
        print(
            f"[SUMMARY] non_wheel_contact_over_weight_last50_mean={_mean(last50_rows, 'non_wheel_contact_force_over_weight'):.6f}",
            flush=True,
        )
        print(f"[DONE] wrote trace: {output_path}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        print(f"[ERROR] stand_zero_action_trace failed: {type(exc).__name__}: {exc!r}", flush=True)
        traceback.print_exc()
        raise
