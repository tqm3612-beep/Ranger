#!/usr/bin/env python3
"""Run a simultaneous left/right open-loop mirror test for Ranger wheel dynamics."""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Template-Ranger-ShortGoalFlat-v1")
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--settle_s", type=float, default=1.0)
parser.add_argument("--duration_s", type=float, default=5.0)
parser.add_argument(
    "--analysis_start_s",
    type=float,
    default=1.0,
    help="Ignore the initial action-ramp interval when computing summaries.",
)
parser.add_argument("--common", type=float, default=0.20, help="Normalized common-mode wheel action.")
parser.add_argument("--turn", type=float, default=0.20, help="Normalized absolute turn-mode wheel action.")
parser.add_argument("--wheel_velocity_limit", type=float, default=20.0)
parser.add_argument(
    "--conditions",
    type=str,
    default="straight,arc,spin",
    help="Comma-separated subset of: straight,arc,spin.",
)
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/debug_wheel_turn_mirror",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

import Ranger.tasks  # noqa: F401
from Ranger.tasks.manager_based.ranger.wheel_semantics import ranger_wheel_joint_to_semantic


WHEEL_NAMES = ("w_lb", "w_lf", "w_rf", "w_rb")
MIRROR_WHEEL_INDEX = (3, 2, 1, 0)
VALID_CONDITIONS = {"straight", "arc", "spin"}


def _safe_step(env: Any, actions: torch.Tensor):
    step_out = env.step(actions)
    if len(step_out) == 5:
        return step_out
    if len(step_out) == 4:
        obs, reward, dones, info = step_out
        terminated = dones
        truncated = torch.zeros_like(dones, dtype=torch.bool)
        return obs, reward, terminated, truncated, info
    raise RuntimeError(f"Unexpected env.step return length: {len(step_out)}")


def _get_action_slice(unwrapped: Any, term_name: str) -> slice:
    start = 0
    for active_name, dim in zip(unwrapped.action_manager.active_terms, unwrapped.action_manager.action_term_dim):
        dim = int(dim)
        if active_name == term_name:
            return slice(start, start + dim)
        start += dim
    raise KeyError(f"Action term {term_name!r} not found: {list(unwrapped.action_manager.active_terms)}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def _relative_error(a: float, b: float, eps: float = 1.0e-6) -> float:
    return abs(a - b) / max(0.5 * (abs(a) + abs(b)), eps)


def _copy_env0_state_to_pair(unwrapped: Any) -> None:
    """Make env 0 and env 1 identical in local coordinates before each condition."""

    robot = unwrapped.scene["robot"]
    env_ids = torch.tensor([0, 1], device=unwrapped.device, dtype=torch.long)
    origins = unwrapped.scene.env_origins[env_ids]

    local_root_pos = robot.data.root_pos_w[0] - unwrapped.scene.env_origins[0]
    root_pose = robot.data.root_state_w[0:1, :7].repeat(2, 1).clone()
    root_pose[:, :3] = origins + local_root_pos.unsqueeze(0)
    root_velocity = torch.zeros((2, 6), device=unwrapped.device, dtype=root_pose.dtype)

    joint_pos = robot.data.joint_pos[0:1].repeat(2, 1).clone()
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    unwrapped.scene.write_data_to_sim()


def _wheel_pattern(condition: str, common: float, turn: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if condition == "straight":
        left_test = (common, common, common, common)
        right_test = left_test
    elif condition == "arc":
        left_test = (common - turn, common - turn, common + turn, common + turn)
        right_test = (common + turn, common + turn, common - turn, common - turn)
    elif condition == "spin":
        left_test = (-turn, -turn, turn, turn)
        right_test = (turn, turn, -turn, -turn)
    else:
        raise ValueError(condition)

    for name, pattern in (("left", left_test), ("right", right_test)):
        if max(abs(value) for value in pattern) > 1.0:
            raise ValueError(f"{condition} {name} wheel action exceeds [-1, 1]: {pattern}")
    return left_test, right_test


def _contact_force_raw_order(unwrapped: Any) -> torch.Tensor:
    sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    body_ids, _ = sensor.find_bodies(list(WHEEL_NAMES), preserve_order=True)
    body_ids_t = torch.as_tensor(body_ids, device=unwrapped.device, dtype=torch.long)
    forces = sensor.data.net_forces_w_history[:, :, body_ids_t, :]
    return torch.max(torch.norm(forces, dim=-1), dim=1).values


def _collect_rows(
    env: Any,
    *,
    condition: str,
    step_idx: int,
    elapsed_s: float,
    start_pos_w: torch.Tensor,
    path_length: torch.Tensor,
) -> list[dict[str, Any]]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    wheel_term = unwrapped.action_manager.get_term("wheel_motor_csv")
    leg_term = unwrapped.action_manager.get_term("leg_hydraulic")

    wheel_joint_ids, _ = robot.find_joints(list(WHEEL_NAMES), preserve_order=True)
    wheel_body_ids, _ = robot.find_bodies(list(WHEEL_NAMES), preserve_order=True)
    wheel_joint_ids_t = torch.as_tensor(wheel_joint_ids, device=unwrapped.device, dtype=torch.long)
    wheel_body_ids_t = torch.as_tensor(wheel_body_ids, device=unwrapped.device, dtype=torch.long)

    wheel_target_semantic = ranger_wheel_joint_to_semantic(wheel_term.velocity_target)
    wheel_joint_vel_semantic = ranger_wheel_joint_to_semantic(robot.data.joint_vel[:, wheel_joint_ids_t])
    wheel_surface_speed = wheel_joint_vel_semantic * float(getattr(unwrapped.cfg, "short_goal_wheel_radius", 0.2024))

    root_forward_b = torch.zeros((2, 3), device=unwrapped.device, dtype=torch.float32)
    root_forward_b[:, 0] = 1.0
    root_forward_w = quat_apply(robot.data.root_quat_w[:2], root_forward_b)
    wheel_hub_vel_w = robot.data.body_lin_vel_w[:2, wheel_body_ids_t, :]
    wheel_hub_forward_speed = torch.sum(wheel_hub_vel_w * root_forward_w.unsqueeze(1), dim=-1)
    wheel_slip_speed = torch.abs(wheel_surface_speed - wheel_hub_forward_speed)

    contact_force = _contact_force_raw_order(unwrapped)[:2]
    contact_total = torch.clamp(contact_force.sum(dim=1, keepdim=True), min=1.0e-6)
    contact_ratio = contact_force / contact_total
    stroke = leg_term.stroke_actual[:2]

    roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w[:2])
    displacement = robot.data.root_pos_w[:2] - start_pos_w

    rows: list[dict[str, Any]] = []
    for env_id, turn_label in ((0, "left"), (1, "right")):
        surface_abs = torch.abs(wheel_surface_speed[env_id])
        hub_abs = torch.abs(wheel_hub_forward_speed[env_id])
        valid = (contact_force[env_id] > 20.0) & (surface_abs > 0.20)
        efficiency = torch.where(
            valid,
            hub_abs / torch.clamp(surface_abs, min=0.20),
            torch.zeros_like(surface_abs),
        )
        valid_count = max(int(valid.sum().item()), 1)
        efficiency_mean = float((efficiency * valid.to(efficiency.dtype)).sum().item() / valid_count)

        row: dict[str, Any] = {
            "condition": condition,
            "env_id": env_id,
            "turn_label": turn_label,
            "step": step_idx,
            "time_s": elapsed_s,
            "path_length_m": float(path_length[env_id].item()),
            "displacement_x_m": float(displacement[env_id, 0].item()),
            "displacement_y_m": float(displacement[env_id, 1].item()),
            "base_lin_vel_x_mps": float(robot.data.root_lin_vel_b[env_id, 0].item()),
            "base_lin_vel_y_mps": float(robot.data.root_lin_vel_b[env_id, 1].item()),
            "base_yaw_rate_radps": float(robot.data.root_ang_vel_b[env_id, 2].item()),
            "yaw_rad": float(yaw[env_id].item()),
            "roll_rad": float(roll[env_id].item()),
            "pitch_rad": float(pitch[env_id].item()),
            "root_height_m": float(robot.data.root_pos_w[env_id, 2].item()),
            "wheel_surface_speed_abs_mean_mps": float(surface_abs.mean().item()),
            "wheel_hub_speed_abs_mean_mps": float(hub_abs.mean().item()),
            "wheel_slip_speed_mean_mps": float(wheel_slip_speed[env_id].mean().item()),
            "traction_speed_efficiency_mean": efficiency_mean,
            "wheel_contact_min_ratio": float(contact_ratio[env_id].min().item()),
            "stroke_max_m": float(stroke[env_id].max().item()),
            "stroke_range_m": float((stroke[env_id].max() - stroke[env_id].min()).item()),
        }
        for wheel_idx, wheel_name in enumerate(WHEEL_NAMES):
            row[f"target_semantic_{wheel_name}_radps"] = float(wheel_target_semantic[env_id, wheel_idx].item())
            row[f"joint_semantic_{wheel_name}_radps"] = float(wheel_joint_vel_semantic[env_id, wheel_idx].item())
            row[f"surface_{wheel_name}_mps"] = float(wheel_surface_speed[env_id, wheel_idx].item())
            row[f"hub_{wheel_name}_mps"] = float(wheel_hub_forward_speed[env_id, wheel_idx].item())
            row[f"slip_{wheel_name}_mps"] = float(wheel_slip_speed[env_id, wheel_idx].item())
            row[f"contact_force_{wheel_name}_n"] = float(contact_force[env_id, wheel_idx].item())
            row[f"contact_ratio_{wheel_name}"] = float(contact_ratio[env_id, wheel_idx].item())
            row[f"stroke_{wheel_name[2:]}_m"] = float(stroke[env_id, wheel_idx].item())
        rows.append(row)
    return rows


def _summarize_condition(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    initial_yaw: tuple[float, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for env_id, turn_label in ((0, "left"), (1, "right")):
        env_rows = [row for row in rows if int(row["env_id"]) == env_id]
        final_yaw = float(env_rows[-1]["yaw_rad"])
        delta_yaw = float(wrap_to_pi(torch.tensor(final_yaw - initial_yaw[env_id])).item())
        path_length = float(env_rows[-1]["path_length_m"])
        summary: dict[str, Any] = {
            "condition": condition,
            "env_id": env_id,
            "turn_label": turn_label,
            "mean_yaw_rate_radps": _mean(env_rows, "base_yaw_rate_radps"),
            "mean_abs_yaw_rate_radps": _mean(
                [{"value": abs(float(row["base_yaw_rate_radps"]))} for row in env_rows], "value"
            ),
            "delta_yaw_rad": delta_yaw,
            "path_length_m": path_length,
            "curvature_rad_per_m": delta_yaw / max(path_length, 1.0e-6),
            "mean_forward_speed_mps": _mean(env_rows, "base_lin_vel_x_mps"),
            "mean_lateral_speed_mps": _mean(env_rows, "base_lin_vel_y_mps"),
            "wheel_surface_speed_abs_mean_mps": _mean(env_rows, "wheel_surface_speed_abs_mean_mps"),
            "wheel_hub_speed_abs_mean_mps": _mean(env_rows, "wheel_hub_speed_abs_mean_mps"),
            "wheel_slip_speed_mean_mps": _mean(env_rows, "wheel_slip_speed_mean_mps"),
            "traction_speed_efficiency_mean": _mean(env_rows, "traction_speed_efficiency_mean"),
            "wheel_contact_min_ratio": _mean(env_rows, "wheel_contact_min_ratio"),
            "stroke_max_m": _mean(env_rows, "stroke_max_m"),
            "stroke_range_m": _mean(env_rows, "stroke_range_m"),
            "termination_any": max(int(row.get("termination_any", 0)) for row in env_rows),
        }
        for wheel_name in WHEEL_NAMES:
            for prefix in ("surface", "hub", "slip", "contact_force", "contact_ratio", "stroke"):
                suffix = "m" if prefix == "stroke" else (
                    "n" if prefix == "contact_force" else ("mps" if prefix in {"surface", "hub", "slip"} else "")
                )
                key = f"{prefix}_{wheel_name[2:]}_{suffix}" if prefix == "stroke" else f"{prefix}_{wheel_name}_{suffix}"
                key = key.rstrip("_")
                if key in env_rows[0]:
                    summary[key] = _mean(env_rows, key)
        summaries.append(summary)

    left = summaries[0]
    right = summaries[1]
    yaw_left_abs = abs(float(left["mean_yaw_rate_radps"]))
    yaw_right_abs = abs(float(right["mean_yaw_rate_radps"]))
    comparison: dict[str, Any] = {
        "condition": condition,
        "yaw_sign_left": math.copysign(1.0, float(left["mean_yaw_rate_radps"])) if yaw_left_abs > 1.0e-6 else 0.0,
        "yaw_sign_right": math.copysign(1.0, float(right["mean_yaw_rate_radps"])) if yaw_right_abs > 1.0e-6 else 0.0,
        "yaw_rate_mirror_abs_error_radps": abs(
            float(left["mean_yaw_rate_radps"]) + float(right["mean_yaw_rate_radps"])
        ),
        "yaw_rate_magnitude_relative_error": _relative_error(yaw_left_abs, yaw_right_abs),
        "delta_yaw_magnitude_relative_error": _relative_error(
            abs(float(left["delta_yaw_rad"])), abs(float(right["delta_yaw_rad"]))
        ),
        "path_length_relative_error": _relative_error(
            float(left["path_length_m"]), float(right["path_length_m"])
        ),
        "slip_relative_error": _relative_error(
            float(left["wheel_slip_speed_mean_mps"]), float(right["wheel_slip_speed_mean_mps"])
        ),
        "traction_efficiency_relative_error": _relative_error(
            float(left["traction_speed_efficiency_mean"]), float(right["traction_speed_efficiency_mean"])
        ),
        "contact_min_ratio_relative_error": _relative_error(
            float(left["wheel_contact_min_ratio"]), float(right["wheel_contact_min_ratio"])
        ),
        "stroke_max_relative_error": _relative_error(
            float(left["stroke_max_m"]), float(right["stroke_max_m"])
        ),
    }

    mirror_metric_errors: list[float] = []
    for left_idx, right_idx in enumerate(MIRROR_WHEEL_INDEX):
        left_name = WHEEL_NAMES[left_idx]
        right_name = WHEEL_NAMES[right_idx]
        for prefix, suffix in (
            ("slip", "mps"),
            ("contact_force", "n"),
            ("contact_ratio", ""),
            ("surface", "mps"),
            ("hub", "mps"),
        ):
            left_key = f"{prefix}_{left_name}_{suffix}".rstrip("_")
            right_key = f"{prefix}_{right_name}_{suffix}".rstrip("_")
            if left_key in left and right_key in right:
                mirror_metric_errors.append(_relative_error(float(left[left_key]), float(right[right_key])))
    comparison["mean_per_wheel_mirror_relative_error"] = (
        sum(mirror_metric_errors) / max(len(mirror_metric_errors), 1)
    )
    return summaries, comparison


def _print_comparison(comparison: dict[str, Any]) -> None:
    condition = str(comparison["condition"])
    yaw_error = float(comparison["yaw_rate_magnitude_relative_error"])
    wheel_error = float(comparison["mean_per_wheel_mirror_relative_error"])
    print(
        f"[MIRROR] {condition}: "
        f"yaw_magnitude_rel_error={yaw_error:.3f}, "
        f"delta_yaw_rel_error={float(comparison['delta_yaw_magnitude_relative_error']):.3f}, "
        f"slip_rel_error={float(comparison['slip_relative_error']):.3f}, "
        f"traction_eff_rel_error={float(comparison['traction_efficiency_relative_error']):.3f}, "
        f"per_wheel_mirror_rel_error={wheel_error:.3f}",
        flush=True,
    )
    if condition == "straight":
        print(
            "  straight baseline: mean signed yaw should remain near zero in both environments.",
            flush=True,
        )
    elif yaw_error <= 0.15 and wheel_error <= 0.20:
        print("  verdict: approximately mirror-symmetric physical response.", flush=True)
    elif yaw_error <= 0.30:
        print("  verdict: moderate left/right physical asymmetry; inspect per-wheel rows.", flush=True)
    else:
        print("  verdict: strong left/right physical asymmetry under identical mirrored commands.", flush=True)


def main() -> None:
    conditions = [item.strip().lower() for item in args_cli.conditions.split(",") if item.strip()]
    invalid = [item for item in conditions if item not in VALID_CONDITIONS]
    if not conditions or invalid:
        raise ValueError(f"Invalid conditions={invalid}; valid={sorted(VALID_CONDITIONS)}")
    if args_cli.analysis_start_s < 0.0 or args_cli.analysis_start_s >= args_cli.duration_s:
        raise ValueError("analysis_start_s must satisfy 0 <= analysis_start_s < duration_s")

    run_root = Path(args_cli.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=2)
    env_cfg.scene.num_envs = 2
    env_cfg.seed = int(args_cli.seed)
    env_cfg.episode_length_s = max(
        float(env_cfg.episode_length_s),
        float(args_cli.settle_s + args_cli.duration_s + 2.0),
    )
    if hasattr(env_cfg.terminations, "goal_reached"):
        env_cfg.terminations.goal_reached = None
    if hasattr(env_cfg.terminations, "stopped_goal_reached"):
        env_cfg.terminations.stopped_goal_reached = None
    env_cfg.actions.wheel_motor_csv.velocity_limit = float(args_cli.wheel_velocity_limit)

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        if unwrapped.num_envs != 2:
            raise RuntimeError(f"Mirror test requires exactly 2 envs, got {unwrapped.num_envs}.")

        action_dim = int(unwrapped.action_manager.total_action_dim)
        zero_actions = torch.zeros((2, action_dim), device=unwrapped.device, dtype=torch.float32)
        leg_slice = _get_action_slice(unwrapped, "leg_hydraulic")
        wheel_slice = _get_action_slice(unwrapped, "wheel_motor_csv")
        step_dt = float(unwrapped.step_dt)
        settle_steps = max(int(round(float(args_cli.settle_s) / max(step_dt, 1.0e-6))), 1)
        run_steps = max(int(round(float(args_cli.duration_s) / max(step_dt, 1.0e-6))), 1)
        analysis_start_step = int(round(float(args_cli.analysis_start_s) / max(step_dt, 1.0e-6)))

        all_timeseries_rows: list[dict[str, Any]] = []
        all_summary_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []

        for condition in conditions:
            env.reset()
            _copy_env0_state_to_pair(unwrapped)
            for _ in range(settle_steps):
                _safe_step(env, zero_actions)

            actions = torch.zeros_like(zero_actions)
            actions[:, leg_slice] = 0.0
            left_pattern, right_pattern = _wheel_pattern(
                condition,
                common=float(args_cli.common),
                turn=float(args_cli.turn),
            )
            actions[0, wheel_slice] = torch.tensor(left_pattern, device=unwrapped.device)
            actions[1, wheel_slice] = torch.tensor(right_pattern, device=unwrapped.device)

            robot = unwrapped.scene["robot"]
            start_pos_w = robot.data.root_pos_w[:2].clone()
            previous_pos_w = start_pos_w.clone()
            path_length = torch.zeros(2, device=unwrapped.device, dtype=torch.float32)
            _, _, initial_yaw_tensor = euler_xyz_from_quat(robot.data.root_quat_w[:2])
            initial_yaw = (float(initial_yaw_tensor[0].item()), float(initial_yaw_tensor[1].item()))

            condition_rows: list[dict[str, Any]] = []
            for step_idx in range(1, run_steps + 1):
                _, _, terminated, truncated, _ = _safe_step(env, actions)
                current_pos_w = robot.data.root_pos_w[:2].clone()
                path_length += torch.norm(current_pos_w[:, :2] - previous_pos_w[:, :2], dim=1)
                previous_pos_w = current_pos_w
                rows = _collect_rows(
                    env,
                    condition=condition,
                    step_idx=step_idx,
                    elapsed_s=step_idx * step_dt,
                    start_pos_w=start_pos_w,
                    path_length=path_length,
                )
                for env_id, row in enumerate(rows):
                    row["termination_any"] = int((terminated[env_id] | truncated[env_id]).item())
                condition_rows.extend(rows)
                if bool(torch.any(terminated | truncated).item()):
                    print(f"[WARN] termination detected during {condition} at step {step_idx}", flush=True)
                    break

            analysis_rows = [
                row for row in condition_rows if int(row["step"]) >= analysis_start_step
            ]
            summaries, comparison = _summarize_condition(
                analysis_rows,
                condition=condition,
                initial_yaw=initial_yaw,
            )
            all_timeseries_rows.extend(condition_rows)
            all_summary_rows.extend(summaries)
            comparison_rows.append(comparison)
            _print_comparison(comparison)

        timeseries_path = run_root / "timeseries.csv"
        summary_path = run_root / "summary_by_side.csv"
        comparison_path = run_root / "mirror_comparison.csv"
        _write_csv(timeseries_path, all_timeseries_rows)
        _write_csv(summary_path, all_summary_rows)
        _write_csv(comparison_path, comparison_rows)
        print(f"[DONE] {timeseries_path}", flush=True)
        print(f"[DONE] {summary_path}", flush=True)
        print(f"[DONE] {comparison_path}", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
