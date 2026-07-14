#!/usr/bin/env python3
"""Deterministic single-environment zero-action smoke test for the rebuilt Ranger asset."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate the rebuilt Ranger USD with a deterministic zero-action rollout.")
parser.add_argument("--task", type=str, default="Template-Ranger-v0", help="Base task used to instantiate the scene.")
parser.add_argument("--steps", type=int, default=1000, help="Number of zero-action environment steps.")
parser.add_argument("--spawn_height", type=float, default=0.700, help="Initial base_link/root height in metres.")
parser.add_argument("--contact_threshold", type=float, default=1.0, help="Wheel contact threshold in newtons.")
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/ranger_new_asset_zero_action",
    help="Output directory for CSV and JSON reports.",
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
from Ranger.assets.ranger.ranger_cfg import RANGER_URDF_PATH, RANGER_USD_PATH


WHEEL_NAMES = ("w_lb", "w_lf", "w_rf", "w_rb")
LEG_JOINT_NAMES = ("g_lb", "g_lf", "g_rf", "g_rb")
MOTION_JOINT_NAMES = (*LEG_JOINT_NAMES, *WHEEL_NAMES)


def _set_zero_range(event_term: Any) -> None:
    if event_term is None:
        return
    event_term.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    event_term.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }


def _configure_env() -> Any:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, float(args_cli.spawn_height))
    env_cfg.scene.robot.init_state.joint_pos = {name: 0.0 for name in MOTION_JOINT_NAMES}
    env_cfg.scene.robot.init_state.joint_vel = {".*": 0.0}

    # The test must not inherit Stand-task reset logic or any reset randomization.
    if hasattr(env_cfg, "stand_training_task"):
        env_cfg.stand_training_task = False
    if hasattr(env_cfg, "enable_reset_settle"):
        env_cfg.enable_reset_settle = False
    if hasattr(env_cfg, "enable_initial_stroke_randomization"):
        env_cfg.enable_initial_stroke_randomization = False
    if hasattr(env_cfg, "terrain_like_reset_enabled"):
        env_cfg.terrain_like_reset_enabled = False
    if hasattr(env_cfg, "sync_reset_root_height_to_initial_stroke"):
        env_cfg.sync_reset_root_height_to_initial_stroke = False

    _set_zero_range(getattr(env_cfg.events, "reset_base", None))
    reset_joints = getattr(env_cfg.events, "reset_robot_joints", None)
    if reset_joints is not None:
        reset_joints.params["position_range"] = (0.0, 0.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)

    for group_name in ("policy_state", "policy_map", "critic_privileged"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None and hasattr(group, "enable_corruption"):
            group.enable_corruption = False

    for sensor_name in ("mid360_lidar", "avia_lidar", "d435i_camera", "wheel_contact_forces", "all_body_contact_forces"):
        sensor_cfg = getattr(env_cfg.scene, sensor_name, None)
        if sensor_cfg is not None and hasattr(sensor_cfg, "debug_vis"):
            sensor_cfg.debug_vis = False

    step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(args_cli.steps) * step_dt + 2.0)
    return env_cfg


def _action_tensor(env: Any) -> torch.Tensor:
    total_dim = int(env.unwrapped.action_manager.total_action_dim)
    return torch.zeros((1, total_dim), device=env.unwrapped.device, dtype=torch.float32)


def _sensor_force_vectors(sensor: Any, env_id: int, body_ids: torch.Tensor) -> torch.Tensor:
    history = getattr(sensor.data, "net_forces_w_history", None)
    if history is not None:
        return history[env_id, -1, body_ids, :]
    current = getattr(sensor.data, "net_forces_w", None)
    if current is None:
        raise RuntimeError(f"Contact sensor {sensor.cfg.prim_path!r} exposes no force tensor.")
    return current[env_id, body_ids, :]


def _find_ids(owner: Any, names: tuple[str, ...] | list[str], *, kind: str, device: torch.device) -> torch.Tensor:
    if kind == "joint":
        ids, resolved = owner.find_joints(list(names), preserve_order=True)
    elif kind == "body":
        ids, resolved = owner.find_bodies(list(names), preserve_order=True)
    else:
        raise ValueError(kind)
    if list(resolved) != list(names):
        raise RuntimeError(f"Failed to preserve {kind} order: requested={list(names)}, resolved={list(resolved)}")
    return torch.as_tensor(ids, device=device, dtype=torch.long)


def _safe_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def _max_abs(rows: list[dict[str, Any]], key: str) -> float:
    return max((abs(float(row[key])) for row in rows), default=0.0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    if not RANGER_USD_PATH.is_file():
        raise FileNotFoundError(f"Ranger USD asset not found: {RANGER_USD_PATH}")
    if not RANGER_URDF_PATH.is_file():
        raise FileNotFoundError(f"Ranger URDF asset not found: {RANGER_URDF_PATH}")

    output_root = Path(args_cli.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "zero_action_trace.csv"
    json_path = output_root / "zero_action_summary.json"

    env_cfg = _configure_env()
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        device = robot.device
        wheel_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
        all_body_sensor = unwrapped.scene.sensors["all_body_contact_forces"]

        joint_ids = _find_ids(robot, MOTION_JOINT_NAMES, kind="joint", device=device)
        wheel_body_ids_robot = _find_ids(robot, WHEEL_NAMES, kind="body", device=device)
        wheel_body_ids_sensor = _find_ids(wheel_sensor, WHEEL_NAMES, kind="body", device=device)

        all_body_names = list(all_body_sensor.body_names)
        _find_ids(all_body_sensor, all_body_names, kind="body", device=device)
        non_wheel_names = [name for name in all_body_names if name not in WHEEL_NAMES]
        non_wheel_ids = (
            _find_ids(all_body_sensor, non_wheel_names, kind="body", device=device)
            if non_wheel_names
            else torch.empty((0,), device=device, dtype=torch.long)
        )

        masses = robot.data.default_mass
        if masses.ndim == 2:
            total_mass = float(masses[0].sum().item())
        else:
            total_mass = float(masses.sum().item())
        gravity_z = float(env_cfg.sim.gravity[2])
        expected_weight = total_mass * abs(gravity_z)

        action_terms = list(unwrapped.action_manager.active_terms)
        action_dims = [int(value) for value in unwrapped.action_manager.action_term_dim]
        sensor_names = list(unwrapped.scene.sensors.keys())

        print(f"[ASSET] usd_path={RANGER_USD_PATH}", flush=True)
        print(f"[ASSET] urdf_path={RANGER_URDF_PATH}", flush=True)
        print(f"[ASSET] robot_prim={robot.cfg.prim_path}", flush=True)
        print(f"[ASSET] joint_names={list(robot.joint_names)}", flush=True)
        print(f"[ASSET] body_names={list(robot.body_names)}", flush=True)
        print(f"[ASSET] action_terms={action_terms} action_dims={action_dims}", flush=True)
        print(f"[ASSET] sensors={sensor_names}", flush=True)
        print(f"[ASSET] total_mass={total_mass:.6f} expected_weight={expected_weight:.6f}", flush=True)

        first_contact_step: dict[str, int | None] = {name: None for name in WHEEL_NAMES}
        rows: list[dict[str, Any]] = []
        zero_actions = _action_tensor(env)
        nonfinite_step: int | None = None
        termination_step: int | None = None

        with torch.inference_mode():
            for step in range(int(args_cli.steps) + 1):
                if step > 0:
                    step_out = env.step(zero_actions)
                    if len(step_out) == 5:
                        _, _, terminated, truncated, _ = step_out
                    elif len(step_out) == 4:
                        _, _, done, _ = step_out
                        terminated = done
                        truncated = torch.zeros_like(done, dtype=torch.bool)
                    else:
                        raise RuntimeError(f"Unexpected env.step return length: {len(step_out)}")
                    if bool(terminated[0].item()) or bool(truncated[0].item()):
                        termination_step = step
                else:
                    terminated = torch.zeros((1,), device=device, dtype=torch.bool)
                    truncated = torch.zeros((1,), device=device, dtype=torch.bool)

                wheel_force_vectors = _sensor_force_vectors(wheel_sensor, 0, wheel_body_ids_sensor)
                wheel_force_norm = torch.linalg.vector_norm(wheel_force_vectors, dim=-1)
                wheel_force_z = wheel_force_vectors[:, 2]
                non_wheel_force_vectors = (
                    _sensor_force_vectors(all_body_sensor, 0, non_wheel_ids)
                    if non_wheel_ids.numel() > 0
                    else torch.zeros((0, 3), device=device)
                )
                non_wheel_force_norm = torch.linalg.vector_norm(non_wheel_force_vectors, dim=-1)

                roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
                wheel_pos_w = robot.data.body_pos_w[0, wheel_body_ids_robot, :]
                joint_pos = robot.data.joint_pos[0, joint_ids]
                joint_vel = robot.data.joint_vel[0, joint_ids]
                root_lin_vel_b = robot.data.root_lin_vel_b[0]
                root_ang_vel_b = robot.data.root_ang_vel_b[0]

                finite_tensors = (
                    robot.data.root_pos_w[0],
                    robot.data.root_quat_w[0],
                    root_lin_vel_b,
                    root_ang_vel_b,
                    wheel_force_vectors,
                    non_wheel_force_vectors,
                    joint_pos,
                    joint_vel,
                )
                finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in finite_tensors)
                if not finite and nonfinite_step is None:
                    nonfinite_step = step

                for index, name in enumerate(WHEEL_NAMES):
                    if first_contact_step[name] is None and float(wheel_force_norm[index].item()) > float(args_cli.contact_threshold):
                        first_contact_step[name] = step

                row: dict[str, Any] = {
                    "step": step,
                    "root_x": _safe_float(robot.data.root_pos_w[0, 0]),
                    "root_y": _safe_float(robot.data.root_pos_w[0, 1]),
                    "root_z": _safe_float(robot.data.root_pos_w[0, 2]),
                    "roll": _safe_float(roll[0]),
                    "pitch": _safe_float(pitch[0]),
                    "yaw": _safe_float(yaw[0]),
                    "lin_vel_x_b": _safe_float(root_lin_vel_b[0]),
                    "lin_vel_y_b": _safe_float(root_lin_vel_b[1]),
                    "lin_vel_z_b": _safe_float(root_lin_vel_b[2]),
                    "ang_vel_x_b": _safe_float(root_ang_vel_b[0]),
                    "ang_vel_y_b": _safe_float(root_ang_vel_b[1]),
                    "ang_vel_z_b": _safe_float(root_ang_vel_b[2]),
                    "wheel_z_lb": _safe_float(wheel_pos_w[0, 2]),
                    "wheel_z_lf": _safe_float(wheel_pos_w[1, 2]),
                    "wheel_z_rf": _safe_float(wheel_pos_w[2, 2]),
                    "wheel_z_rb": _safe_float(wheel_pos_w[3, 2]),
                    "force_norm_lb": _safe_float(wheel_force_norm[0]),
                    "force_norm_lf": _safe_float(wheel_force_norm[1]),
                    "force_norm_rf": _safe_float(wheel_force_norm[2]),
                    "force_norm_rb": _safe_float(wheel_force_norm[3]),
                    "force_z_lb": _safe_float(wheel_force_z[0]),
                    "force_z_lf": _safe_float(wheel_force_z[1]),
                    "force_z_rf": _safe_float(wheel_force_z[2]),
                    "force_z_rb": _safe_float(wheel_force_z[3]),
                    "wheel_force_total_norm": _safe_float(wheel_force_norm.sum()),
                    "wheel_force_total_z": _safe_float(wheel_force_z.sum()),
                    "non_wheel_force_total_norm": _safe_float(non_wheel_force_norm.sum()),
                    "joint_pos_g_lb": _safe_float(joint_pos[0]),
                    "joint_pos_g_lf": _safe_float(joint_pos[1]),
                    "joint_pos_g_rf": _safe_float(joint_pos[2]),
                    "joint_pos_g_rb": _safe_float(joint_pos[3]),
                    "joint_vel_w_lb": _safe_float(joint_vel[4]),
                    "joint_vel_w_lf": _safe_float(joint_vel[5]),
                    "joint_vel_w_rf": _safe_float(joint_vel[6]),
                    "joint_vel_w_rb": _safe_float(joint_vel[7]),
                    "finite": int(finite),
                    "terminated": int(bool(terminated[0].item())),
                    "truncated": int(bool(truncated[0].item())),
                }
                rows.append(row)
                if termination_step is not None or nonfinite_step is not None:
                    break

        _write_csv(csv_path, rows)
        tail = rows[-min(100, len(rows)) :]
        contact_steps = [value for value in first_contact_step.values() if value is not None]
        contact_spread = max(contact_steps) - min(contact_steps) if len(contact_steps) == len(WHEEL_NAMES) else None
        tail_force = {name: _mean(tail, f"force_norm_{name[2:]}") for name in WHEEL_NAMES}
        tail_force_ratio = {name: value / max(expected_weight, 1.0e-9) for name, value in tail_force.items()}

        diagonal_lb_rf = tail_force["w_lb"] + tail_force["w_rf"]
        diagonal_lf_rb = tail_force["w_lf"] + tail_force["w_rb"]
        left_force = tail_force["w_lb"] + tail_force["w_lf"]
        right_force = tail_force["w_rf"] + tail_force["w_rb"]
        front_force = tail_force["w_lf"] + tail_force["w_rf"]
        rear_force = tail_force["w_lb"] + tail_force["w_rb"]

        checks = {
            "usd_path_exists": RANGER_USD_PATH.is_file(),
            "all_8_motion_joints_found": len(joint_ids) == len(MOTION_JOINT_NAMES)
            and set(MOTION_JOINT_NAMES).issubset(set(robot.joint_names)),
            "action_shape_4_plus_4": action_terms == ["leg_hydraulic", "wheel_motor_csv"] and action_dims == [4, 4],
            "all_wheels_contacted": len(contact_steps) == len(WHEEL_NAMES),
            "contact_spread_le_10_steps": contact_spread is not None and contact_spread <= 10,
            "no_nonfinite": nonfinite_step is None,
            "no_termination": termination_step is None,
            "root_drop_lt_5cm": float(rows[0]["root_z"]) - min(float(row["root_z"]) for row in rows) < 0.05,
            "non_wheel_tail_force_lt_2pct_weight": _mean(tail, "non_wheel_force_total_norm") < 0.02 * expected_weight,
            "all_wheels_loaded_in_tail": all(value > float(args_cli.contact_threshold) for value in tail_force.values()),
        }

        summary: dict[str, Any] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "task": args_cli.task,
            "usd_path": str(RANGER_USD_PATH),
            "urdf_path": str(RANGER_URDF_PATH),
            "spawn_height": float(args_cli.spawn_height),
            "steps_requested": int(args_cli.steps),
            "steps_recorded": len(rows) - 1,
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "action_terms": action_terms,
            "action_dims": action_dims,
            "sensor_names": sensor_names,
            "sensor_prim_paths": {
                name: getattr(unwrapped.scene.sensors[name].cfg, "prim_path", "") for name in sensor_names
            },
            "total_mass": total_mass,
            "expected_weight": expected_weight,
            "first_contact_step": first_contact_step,
            "first_contact_spread_steps": contact_spread,
            "root_z_first": float(rows[0]["root_z"]),
            "root_z_min": min(float(row["root_z"]) for row in rows),
            "root_z_max": max(float(row["root_z"]) for row in rows),
            "root_z_tail_mean": _mean(tail, "root_z"),
            "roll_tail_abs_max": _max_abs(tail, "roll"),
            "pitch_tail_abs_max": _max_abs(tail, "pitch"),
            "lin_vel_x_tail_mean": _mean(tail, "lin_vel_x_b"),
            "lin_vel_y_tail_mean": _mean(tail, "lin_vel_y_b"),
            "lin_vel_z_tail_mean": _mean(tail, "lin_vel_z_b"),
            "wheel_force_tail_mean_n": tail_force,
            "wheel_force_tail_ratio_of_weight": tail_force_ratio,
            "wheel_force_total_tail_ratio_of_weight": _mean(tail, "wheel_force_total_norm") / max(expected_weight, 1.0e-9),
            "non_wheel_force_tail_ratio_of_weight": _mean(tail, "non_wheel_force_total_norm") / max(expected_weight, 1.0e-9),
            "diagonal_force_difference_ratio": abs(diagonal_lb_rf - diagonal_lf_rb) / max(diagonal_lb_rf + diagonal_lf_rb, 1.0e-9),
            "left_right_force_difference_ratio": abs(left_force - right_force) / max(left_force + right_force, 1.0e-9),
            "front_rear_force_difference_ratio": abs(front_force - rear_force) / max(front_force + rear_force, 1.0e-9),
            "nonfinite_step": nonfinite_step,
            "termination_step": termination_step,
            "csv_path": str(csv_path),
        }
        _write_json(json_path, summary)
        print("[SUMMARY] " + json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"[DONE] csv={csv_path}", flush=True)
        print(f"[DONE] json={json_path}", flush=True)

        if summary["status"] != "PASS":
            raise RuntimeError("Ranger rebuilt-asset zero-action checks did not all pass; inspect the JSON report.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        print(f"[ERROR] ranger_new_asset_zero_action failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
