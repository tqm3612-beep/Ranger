#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open-loop Ranger wheel turn debug with velocity-limit sweep.")
parser.add_argument("--task", type=str, default="Template-Ranger-ShortGoalFlat-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_id", type=int, default=0)
parser.add_argument("--duration", "--duration_s", dest="duration_s", type=float, default=5.0)
parser.add_argument("--settle_s", type=float, default=0.5)
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/debug_wheel_turn_open_loop",
    help="Directory for summary/timeseries CSV output.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

import Ranger.tasks  # noqa: F401
from Ranger.tasks.manager_based.ranger import mdp


VELOCITY_LIMIT_OVERRIDES = (20.0, 40.0, 60.0, 80.0, 100.0, 130.0)
PATTERNS = (
    ("straight_forward", (0.4, 0.4, 0.4, 0.4)),
    ("arc_left", (0.1, 0.1, 0.5, 0.5)),
    ("arc_right", (0.5, 0.5, 0.1, 0.1)),
    ("spin_a", (-0.4, -0.4, 0.4, 0.4)),
    ("spin_b", (0.4, 0.4, -0.4, -0.4)),
)


def _safe_step(env, actions: torch.Tensor):
    step_out = env.step(actions)
    if len(step_out) == 5:
        return step_out
    if len(step_out) == 4:
        obs, reward, dones, info = step_out
        terminated = dones
        truncated = torch.zeros_like(dones, dtype=torch.bool)
        return obs, reward, terminated, truncated, info
    raise RuntimeError(f"Unexpected env.step return length: {len(step_out)}")


def _resolve_action_dim(env) -> int:
    unwrapped = env.unwrapped
    manager_dim = int(unwrapped.action_manager.total_action_dim)
    action_space_shape = tuple(int(dim) for dim in getattr(env.action_space, "shape", ()))
    if len(action_space_shape) == 0:
        return manager_dim
    if action_space_shape[-1] != manager_dim:
        raise RuntimeError(
            f"Action-space shape {action_space_shape} is inconsistent with action_manager.total_action_dim={manager_dim}."
        )
    return manager_dim


def _get_action_slice(unwrapped_env, term_name: str) -> slice:
    start = 0
    for active_name, dim in zip(unwrapped_env.action_manager.active_terms, unwrapped_env.action_manager.action_term_dim):
        if active_name == term_name:
            return slice(start, start + dim)
        start += dim
    raise KeyError(f"Action term '{term_name}' not found.")


def _format_vec(values: list[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _semantic_sign(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return mdp.wheel_semantic_sign_lr_lf_rf_rr(device=device, dtype=dtype).squeeze(0)


def _wheel_semantic_raw_order(values_w_lb_w_lf_w_rf_w_rb: torch.Tensor) -> torch.Tensor:
    return values_w_lb_w_lf_w_rf_w_rb * _semantic_sign(
        device=values_w_lb_w_lf_w_rf_w_rb.device,
        dtype=values_w_lb_w_lf_w_rf_w_rb.dtype,
    )


def _wheel_joint_ids(robot) -> list[int]:
    joint_ids, _ = robot.find_joints(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
    return [int(idx) for idx in joint_ids]


def _contact_forces_and_bools(unwrapped, env_id: int):
    sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    body_ids, _ = sensor.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
    body_ids_t = torch.as_tensor(body_ids, device=unwrapped.device, dtype=torch.long)
    net_forces = sensor.data.net_forces_w_history[env_id, :, body_ids_t, :]
    force_norm = torch.max(torch.norm(net_forces, dim=-1), dim=0).values
    contact_bool = force_norm > 1.0
    return force_norm, contact_bool


def _strokes_lf_lr_rf_rr(unwrapped, env_id: int) -> torch.Tensor:
    leg_action_term = unwrapped.action_manager.get_term("leg_hydraulic")
    return leg_action_term.stroke_actual[env_id, [1, 0, 2, 3]]


def _row(
    env,
    *,
    env_id: int,
    velocity_limit_override: float,
    pattern_name: str,
    step_idx: int,
) -> dict[str, float | int | str]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    wheel_action_term = unwrapped.action_manager.get_term("wheel_motor_csv")
    wheel_joint_ids = _wheel_joint_ids(robot)

    roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    wheel_target = wheel_action_term.velocity_target[env_id]
    wheel_joint_vel = robot.data.joint_vel[env_id, wheel_joint_ids]
    semantic_wheel_target = _wheel_semantic_raw_order(wheel_target)
    semantic_wheel_joint_vel = _wheel_semantic_raw_order(wheel_joint_vel)
    wheel_tracking_error = wheel_target - wheel_joint_vel
    wheel_force, wheel_contact_bool = _contact_forces_and_bools(unwrapped, env_id)
    stroke_lf_lr_rf_rr = _strokes_lf_lr_rf_rr(unwrapped, env_id)

    return {
        "velocity_limit_override": velocity_limit_override,
        "pattern": pattern_name,
        "step": step_idx,
        "base_lin_vel_x": float(robot.data.root_lin_vel_b[env_id, 0].item()),
        "base_lin_vel_y": float(robot.data.root_lin_vel_b[env_id, 1].item()),
        "base_yaw_rate": float(robot.data.root_ang_vel_b[env_id, 2].item()),
        "yaw": float(yaw[env_id].item()),
        "root_height": float(robot.data.root_pos_w[env_id, 2].item()),
        "roll": float(roll[env_id].item()),
        "pitch": float(pitch[env_id].item()),
        "wheel_target_w_lb": float(wheel_target[0].item()),
        "wheel_target_w_lf": float(wheel_target[1].item()),
        "wheel_target_w_rf": float(wheel_target[2].item()),
        "wheel_target_w_rb": float(wheel_target[3].item()),
        "wheel_joint_vel_w_lb": float(wheel_joint_vel[0].item()),
        "wheel_joint_vel_w_lf": float(wheel_joint_vel[1].item()),
        "wheel_joint_vel_w_rf": float(wheel_joint_vel[2].item()),
        "wheel_joint_vel_w_rb": float(wheel_joint_vel[3].item()),
        "wheel_tracking_error_w_lb": float(wheel_tracking_error[0].item()),
        "wheel_tracking_error_w_lf": float(wheel_tracking_error[1].item()),
        "wheel_tracking_error_w_rf": float(wheel_tracking_error[2].item()),
        "wheel_tracking_error_w_rb": float(wheel_tracking_error[3].item()),
        "wheel_contact_force_w_lb": float(wheel_force[0].item()),
        "wheel_contact_force_w_lf": float(wheel_force[1].item()),
        "wheel_contact_force_w_rf": float(wheel_force[2].item()),
        "wheel_contact_force_w_rb": float(wheel_force[3].item()),
        "wheel_contact_bool_w_lb": int(wheel_contact_bool[0].item()),
        "wheel_contact_bool_w_lf": int(wheel_contact_bool[1].item()),
        "wheel_contact_bool_w_rf": int(wheel_contact_bool[2].item()),
        "wheel_contact_bool_w_rb": int(wheel_contact_bool[3].item()),
        "stroke_lf": float(stroke_lf_lr_rf_rr[0].item()),
        "stroke_lr": float(stroke_lf_lr_rf_rr[1].item()),
        "stroke_rf": float(stroke_lf_lr_rf_rr[2].item()),
        "stroke_rr": float(stroke_lf_lr_rf_rr[3].item()),
        "semantic_wheel_target_lr": float(semantic_wheel_target[0].item()),
        "semantic_wheel_target_lf": float(semantic_wheel_target[1].item()),
        "semantic_wheel_target_rf": float(semantic_wheel_target[2].item()),
        "semantic_wheel_target_rr": float(semantic_wheel_target[3].item()),
        "semantic_wheel_joint_vel_lr": float(semantic_wheel_joint_vel[0].item()),
        "semantic_wheel_joint_vel_lf": float(semantic_wheel_joint_vel[1].item()),
        "semantic_wheel_joint_vel_rf": float(semantic_wheel_joint_vel[2].item()),
        "semantic_wheel_joint_vel_rr": float(semantic_wheel_joint_vel[3].item()),
    }


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    preferred_prefix = [
        "velocity_limit_override",
        "pattern",
        "step",
        "termination_any",
        "base_lin_vel_x",
        "base_lin_vel_y",
        "base_yaw_rate",
        "yaw",
        "delta_yaw",
        "root_height",
        "roll",
        "pitch",
    ]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(str(key) for key in row.keys())
    fieldnames = [key for key in preferred_prefix if key in all_keys]
    fieldnames.extend(sorted(key for key in all_keys if key not in fieldnames))
    normalized_rows = [{field: row.get(field, "") for field in fieldnames} for row in rows]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized_rows)


def _mean(rows: list[dict[str, float | int | str]], key: str) -> float:
    return sum(float(r[key]) for r in rows) / max(len(rows), 1)


def _apply_velocity_limit_override(unwrapped, velocity_limit_override: float) -> None:
    wheel_action_term = unwrapped.action_manager.get_term("wheel_motor_csv")
    wheel_action_term._velocity_limit = float(velocity_limit_override)
    if hasattr(unwrapped.cfg.actions.wheel_motor_csv, "velocity_limit"):
        unwrapped.cfg.actions.wheel_motor_csv.velocity_limit = float(velocity_limit_override)


def _run_pattern(
    env,
    *,
    run_root: Path,
    env_id: int,
    velocity_limit_override: float,
    pattern_name: str,
    wheel_raw_pattern: tuple[float, float, float, float],
    zero_actions: torch.Tensor,
    leg_action_slice: slice,
    wheel_action_slice: slice,
    settle_steps: int,
    run_steps: int,
) -> dict[str, float | int | str]:
    unwrapped = env.unwrapped
    _apply_velocity_limit_override(unwrapped, velocity_limit_override)
    env.reset()
    for _ in range(settle_steps):
        _safe_step(env, zero_actions)

    actions = torch.zeros_like(zero_actions)
    actions[:, leg_action_slice] = 0.0
    actions[:, wheel_action_slice] = torch.tensor(
        wheel_raw_pattern, device=zero_actions.device, dtype=torch.float32
    ).unsqueeze(0)

    rows: list[dict[str, float | int | str]] = []
    initial_row = _row(
        env,
        env_id=env_id,
        velocity_limit_override=velocity_limit_override,
        pattern_name=pattern_name,
        step_idx=0,
    )
    rows.append(initial_row)
    initial_yaw = float(initial_row["yaw"])

    print(
        f"[PATTERN] velocity_limit_override={velocity_limit_override:.1f} pattern={pattern_name} raw={wheel_raw_pattern}",
        flush=True,
    )
    for step_idx in range(1, run_steps + 1):
        _, _, terminated, truncated, _ = _safe_step(env, actions)
        current_row = _row(
            env,
            env_id=env_id,
            velocity_limit_override=velocity_limit_override,
            pattern_name=pattern_name,
            step_idx=step_idx,
        )
        current_row["termination_any"] = int((terminated[env_id] | truncated[env_id]).item())
        rows.append(current_row)

    timeseries_path = run_root / (
        f"velocity_limit_{int(round(velocity_limit_override))}_{pattern_name}.csv"
    )
    _write_csv(timeseries_path, rows)

    active_rows = rows[1:] if len(rows) > 1 else rows
    final_yaw = float(active_rows[-1]["yaw"])
    delta_yaw = float(wrap_to_pi(torch.tensor(final_yaw - initial_yaw)).item())

    summary = {
        "velocity_limit_override": velocity_limit_override,
        "pattern": pattern_name,
        "mean_base_lin_vel_x": _mean(active_rows, "base_lin_vel_x"),
        "mean_base_lin_vel_y": _mean(active_rows, "base_lin_vel_y"),
        "mean_base_yaw_rate": _mean(active_rows, "base_yaw_rate"),
        "max_abs_base_yaw_rate": max(abs(float(r["base_yaw_rate"])) for r in active_rows),
        "delta_yaw": delta_yaw,
        "root_height_mean": _mean(active_rows, "root_height"),
        "roll_mean": _mean(active_rows, "roll"),
        "pitch_mean": _mean(active_rows, "pitch"),
        "wheel_target_w_lb": _mean(active_rows, "wheel_target_w_lb"),
        "wheel_target_w_lf": _mean(active_rows, "wheel_target_w_lf"),
        "wheel_target_w_rf": _mean(active_rows, "wheel_target_w_rf"),
        "wheel_target_w_rb": _mean(active_rows, "wheel_target_w_rb"),
        "wheel_joint_vel_w_lb": _mean(active_rows, "wheel_joint_vel_w_lb"),
        "wheel_joint_vel_w_lf": _mean(active_rows, "wheel_joint_vel_w_lf"),
        "wheel_joint_vel_w_rf": _mean(active_rows, "wheel_joint_vel_w_rf"),
        "wheel_joint_vel_w_rb": _mean(active_rows, "wheel_joint_vel_w_rb"),
        "wheel_tracking_error_w_lb": _mean(active_rows, "wheel_tracking_error_w_lb"),
        "wheel_tracking_error_w_lf": _mean(active_rows, "wheel_tracking_error_w_lf"),
        "wheel_tracking_error_w_rf": _mean(active_rows, "wheel_tracking_error_w_rf"),
        "wheel_tracking_error_w_rb": _mean(active_rows, "wheel_tracking_error_w_rb"),
        "wheel_contact_force_w_lb": _mean(active_rows, "wheel_contact_force_w_lb"),
        "wheel_contact_force_w_lf": _mean(active_rows, "wheel_contact_force_w_lf"),
        "wheel_contact_force_w_rf": _mean(active_rows, "wheel_contact_force_w_rf"),
        "wheel_contact_force_w_rb": _mean(active_rows, "wheel_contact_force_w_rb"),
        "wheel_contact_bool_w_lb": _mean(active_rows, "wheel_contact_bool_w_lb"),
        "wheel_contact_bool_w_lf": _mean(active_rows, "wheel_contact_bool_w_lf"),
        "wheel_contact_bool_w_rf": _mean(active_rows, "wheel_contact_bool_w_rf"),
        "wheel_contact_bool_w_rb": _mean(active_rows, "wheel_contact_bool_w_rb"),
        "stroke_lf": _mean(active_rows, "stroke_lf"),
        "stroke_lr": _mean(active_rows, "stroke_lr"),
        "stroke_rf": _mean(active_rows, "stroke_rf"),
        "stroke_rr": _mean(active_rows, "stroke_rr"),
        "semantic_wheel_target_lr": _mean(active_rows, "semantic_wheel_target_lr"),
        "semantic_wheel_target_lf": _mean(active_rows, "semantic_wheel_target_lf"),
        "semantic_wheel_target_rf": _mean(active_rows, "semantic_wheel_target_rf"),
        "semantic_wheel_target_rr": _mean(active_rows, "semantic_wheel_target_rr"),
        "semantic_wheel_joint_vel_lr": _mean(active_rows, "semantic_wheel_joint_vel_lr"),
        "semantic_wheel_joint_vel_lf": _mean(active_rows, "semantic_wheel_joint_vel_lf"),
        "semantic_wheel_joint_vel_rf": _mean(active_rows, "semantic_wheel_joint_vel_rf"),
        "semantic_wheel_joint_vel_rr": _mean(active_rows, "semantic_wheel_joint_vel_rr"),
        "termination_any": max(int(r.get("termination_any", 0)) for r in active_rows),
        "timeseries_csv": str(timeseries_path),
    }

    print(
        "[RESULT] "
        f"velocity_limit_override={velocity_limit_override:.1f} pattern={pattern_name}: "
        f"mean_vx={summary['mean_base_lin_vel_x']:.4f}, "
        f"mean_vy={summary['mean_base_lin_vel_y']:.4f}, "
        f"mean_yaw_rate={summary['mean_base_yaw_rate']:.4f}, "
        f"max_abs_yaw_rate={summary['max_abs_base_yaw_rate']:.4f}, "
        f"delta_yaw={summary['delta_yaw']:.4f}, "
        f"target={_format_vec([summary['wheel_target_w_lb'], summary['wheel_target_w_lf'], summary['wheel_target_w_rf'], summary['wheel_target_w_rb']])}, "
        f"joint_vel={_format_vec([summary['wheel_joint_vel_w_lb'], summary['wheel_joint_vel_w_lf'], summary['wheel_joint_vel_w_rf'], summary['wheel_joint_vel_w_rb']])}, "
        f"tracking_err={_format_vec([summary['wheel_tracking_error_w_lb'], summary['wheel_tracking_error_w_lf'], summary['wheel_tracking_error_w_rf'], summary['wheel_tracking_error_w_rb']])}, "
        f"contact={_format_vec([summary['wheel_contact_force_w_lb'], summary['wheel_contact_force_w_lf'], summary['wheel_contact_force_w_rf'], summary['wheel_contact_force_w_rb']])}",
        flush=True,
    )
    return summary


def _largest_tracking_wheels(summary_rows: list[dict[str, float | int | str]]) -> str:
    accum = {"w_lb": 0.0, "w_lf": 0.0, "w_rf": 0.0, "w_rb": 0.0}
    for row in summary_rows:
        for joint_name in accum:
            accum[joint_name] += abs(float(row[f"wheel_tracking_error_{joint_name}"]))
    ordered = sorted(accum.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{name}({value / max(len(summary_rows), 1):.3f})" for name, value in ordered)


def _print_conclusions(summary_rows: list[dict[str, float | int | str]]) -> None:
    print("[CONCLUSION]", flush=True)
    if not summary_rows:
        print("  no summary rows collected", flush=True)
        return

    turning_rows = [
        row for row in summary_rows
        if row["pattern"] in {"arc_left", "arc_right", "spin_a", "spin_b"}
    ]
    usable_turn_rows = [
        row for row in turning_rows if abs(float(row["mean_base_yaw_rate"])) > 0.2
    ]
    if usable_turn_rows:
        min_usable = min(usable_turn_rows, key=lambda row: float(row["velocity_limit_override"]))
        print(
            f"  minimal usable turn velocity_limit_override (|mean_yaw_rate|>0.2): "
            f"{float(min_usable['velocity_limit_override']):.1f} via {min_usable['pattern']} "
            f"(mean_yaw_rate={float(min_usable['mean_base_yaw_rate']):.4f})",
            flush=True,
        )
    else:
        print("  minimal usable turn velocity_limit_override (|mean_yaw_rate|>0.2): none", flush=True)

    straight_rows = [row for row in summary_rows if row["pattern"] == "straight_forward"]
    if straight_rows:
        best_straight = min(straight_rows, key=lambda row: abs(float(row["mean_base_yaw_rate"])))
        print(
            f"  straight_forward smallest |yaw| override: {float(best_straight['velocity_limit_override']):.1f} "
            f"(mean_yaw_rate={float(best_straight['mean_base_yaw_rate']):.4f})",
            flush=True,
        )

    for spin_name in ("spin_a", "spin_b"):
        spin_rows = [row for row in summary_rows if row["pattern"] == spin_name]
        if spin_rows:
            strongest = max(spin_rows, key=lambda row: abs(float(row["mean_base_yaw_rate"])))
            yaw = float(strongest["mean_base_yaw_rate"])
            yaw_sign = "positive" if yaw > 0.0 else "negative" if yaw < 0.0 else "zero"
            print(
                f"  {spin_name} strongest yaw sign: {yaw_sign} at override={float(strongest['velocity_limit_override']):.1f} "
                f"(mean_yaw_rate={yaw:.4f})",
                flush=True,
            )

    imbalance_rows = []
    for row in summary_rows:
        forces = [
            float(row["wheel_contact_force_w_lb"]),
            float(row["wheel_contact_force_w_lf"]),
            float(row["wheel_contact_force_w_rf"]),
            float(row["wheel_contact_force_w_rb"]),
        ]
        denom = max(sum(forces), 1.0e-6)
        imbalance = (max(forces) - min(forces)) / denom
        if imbalance > 0.35:
            imbalance_rows.append((imbalance, row))
    if imbalance_rows:
        imbalance_rows.sort(key=lambda item: item[0], reverse=True)
        worst_imbalance, worst_row = imbalance_rows[0]
        print(
            f"  severe contact-force imbalance detected: yes, worst={worst_imbalance:.3f} "
            f"at override={float(worst_row['velocity_limit_override']):.1f} pattern={worst_row['pattern']}",
            flush=True,
        )
    else:
        print("  severe contact-force imbalance detected: no (threshold=0.35)", flush=True)

    print(f"  largest average target-tracking-error wheels: {_largest_tracking_wheels(summary_rows)}", flush=True)


def main() -> None:
    run_root = Path(args_cli.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(args_cli.settle_s + args_cli.duration_s + 1.0))

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        unwrapped = env.unwrapped
        if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
            raise ValueError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}.")

        device = unwrapped.device
        action_dim = _resolve_action_dim(env)
        zero_actions = torch.zeros((unwrapped.num_envs, action_dim), device=device, dtype=torch.float32)
        leg_action_slice = _get_action_slice(unwrapped, "leg_hydraulic")
        wheel_action_slice = _get_action_slice(unwrapped, "wheel_motor_csv")
        step_dt = float(unwrapped.step_dt)
        settle_steps = max(int(round(float(args_cli.settle_s) / max(step_dt, 1.0e-6))), 1)
        run_steps = max(int(round(float(args_cli.duration_s) / max(step_dt, 1.0e-6))), 1)

        print(
            f"[INFO] single-env-create velocity_limit sweep task={args_cli.task} "
            f"num_envs={unwrapped.num_envs} env_id={args_cli.env_id} step_dt={step_dt:.6f} "
            f"settle_steps={settle_steps} run_steps={run_steps}",
            flush=True,
        )

        summary_rows: list[dict[str, float | int | str]] = []
        for velocity_limit_override in VELOCITY_LIMIT_OVERRIDES:
            print(f"[INFO] testing velocity_limit_override={velocity_limit_override:.1f}", flush=True)
            for pattern_name, wheel_raw_pattern in PATTERNS:
                summary_rows.append(
                    _run_pattern(
                        env,
                        run_root=run_root,
                        env_id=args_cli.env_id,
                        velocity_limit_override=velocity_limit_override,
                        pattern_name=pattern_name,
                        wheel_raw_pattern=wheel_raw_pattern,
                        zero_actions=zero_actions,
                        leg_action_slice=leg_action_slice,
                        wheel_action_slice=wheel_action_slice,
                        settle_steps=settle_steps,
                        run_steps=run_steps,
                    )
                )

        summary_path = run_root / "summary.csv"
        _write_csv(summary_path, summary_rows)
        print(f"[DONE] wrote summary: {summary_path}", flush=True)
        _print_conclusions(summary_rows)
    except BaseException:
        print("[ERROR] debug_wheel_turn_open_loop failed", flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
