#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run fixed Ranger wheel-target sanity tests without policy involvement."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run fixed Ranger wheel-target sanity tests and record videos.")
parser.add_argument("--task", type=str, default="Template-Ranger-Stand-v0", help="Task used for the sanity test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to report.")
parser.add_argument("--duration_s", type=float, default=4.0, help="Seconds to run each fixed wheel-target case.")
parser.add_argument("--settle_s", type=float, default=1.0, help="Seconds to hold zero wheel target before each case.")
parser.add_argument(
    "--neutral_leg_action",
    type=float,
    default=-0.34,
    help="Constant normalized hydraulic action applied to all four legs during the test.",
)
parser.add_argument("--print_every", type=int, default=15, help="Print one line every N steps.")
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/wheel_action_sanity",
    help="Directory where videos and CSV files will be written.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Ranger.tasks  # noqa: F401
from Ranger.tasks.manager_based.ranger import mdp


CASE_DEFINITIONS = (
    ("all_same_low", {"lf": 2.0, "lr": 2.0, "rf": 2.0, "rr": 2.0}),
    ("all_same_medium", {"lf": 5.0, "lr": 5.0, "rf": 5.0, "rr": 5.0}),
    ("diff_low", {"lf": 2.0, "lr": 2.0, "rf": -2.0, "rr": -2.0}),
    ("diff_medium", {"lf": 5.0, "lr": 5.0, "rf": -5.0, "rr": -5.0}),
)


def _get_action_slice(unwrapped_env, term_name: str) -> slice:
    start = 0
    for active_name, dim in zip(unwrapped_env.action_manager.active_terms, unwrapped_env.action_manager.action_term_dim):
        if active_name == term_name:
            return slice(start, start + dim)
        start += dim
    raise KeyError(f"Action term '{term_name}' not found in action manager.")


def _case_target_tensor(device: torch.device, case_targets: dict[str, float]) -> torch.Tensor:
    """Return wheel targets in the wheel action term order: [lr, lf, rf, rr]."""

    return torch.tensor(
        [
            float(case_targets["lr"]),
            float(case_targets["lf"]),
            float(case_targets["rf"]),
            float(case_targets["rr"]),
        ],
        device=device,
        dtype=torch.float32,
    )


def _build_actions(
    env,
    leg_action_slice: slice,
    wheel_action_slice: slice,
    wheel_targets_rad_s: torch.Tensor,
    neutral_leg_action: float,
) -> torch.Tensor:
    unwrapped = env.unwrapped
    wheel_velocity_limit = max(float(unwrapped.cfg.actions.wheel_motor_csv.velocity_limit), 1.0e-6)
    wheel_forward_sign = mdp.wheel_semantic_sign_lr_lf_rf_rr(
        device=unwrapped.device, dtype=torch.float32
    ).squeeze(0)
    actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
    actions[:, leg_action_slice] = float(neutral_leg_action)
    wheel_raw_actions = (wheel_targets_rad_s / wheel_velocity_limit) * wheel_forward_sign
    actions[:, wheel_action_slice] = wheel_raw_actions.unsqueeze(0).repeat(actions.shape[0], 1)
    return actions


def _peak_contact_force_and_bool(contact_sensor, env_id: int, body_ids: torch.Tensor, threshold: float = 1.0):
    force_history = contact_sensor.data.net_forces_w_history[env_id, :, body_ids, :]
    force_norm = torch.norm(force_history, dim=-1)
    peak_force_norm = torch.max(force_norm, dim=0).values
    contact_bool = peak_force_norm > float(threshold)
    return peak_force_norm, contact_bool


def _collect_step_row(
    env,
    env_id: int,
    case_name: str,
    step: int,
    wheel_joint_ids: torch.Tensor,
    wheel_body_ids: torch.Tensor,
) -> dict[str, float | int | str]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    wheel_action_term = unwrapped.action_manager.get_term("wheel_motor_csv")
    contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]

    wheel_target = wheel_action_term.velocity_target[env_id]
    wheel_actual = robot.data.joint_vel[env_id, wheel_joint_ids]
    peak_contact_force, contact_bool = _peak_contact_force_and_bool(contact_sensor, env_id, wheel_body_ids)

    # Internal order: [lr, lf, rf, rr]
    target_lr = float(wheel_target[0].item())
    target_lf = float(wheel_target[1].item())
    target_rf = float(wheel_target[2].item())
    target_rr = float(wheel_target[3].item())
    actual_lr = float(wheel_actual[0].item())
    actual_lf = float(wheel_actual[1].item())
    actual_rf = float(wheel_actual[2].item())
    actual_rr = float(wheel_actual[3].item())
    contact_force_lr = float(peak_contact_force[0].item())
    contact_force_lf = float(peak_contact_force[1].item())
    contact_force_rf = float(peak_contact_force[2].item())
    contact_force_rr = float(peak_contact_force[3].item())
    contact_bool_lr = float(contact_bool[0].to(torch.float32).item())
    contact_bool_lf = float(contact_bool[1].to(torch.float32).item())
    contact_bool_rf = float(contact_bool[2].to(torch.float32).item())
    contact_bool_rr = float(contact_bool[3].to(torch.float32).item())

    velocity_tracking_abs_error_mean = float(torch.mean(torch.abs(wheel_target - wheel_actual)).item())
    semantic_wheel_target = wheel_target * mdp.wheel_semantic_sign_lr_lf_rf_rr(
        device=wheel_target.device, dtype=wheel_target.dtype
    ).squeeze(0)
    semantic_wheel_actual = wheel_actual * mdp.wheel_semantic_sign_lr_lf_rf_rr(
        device=wheel_actual.device, dtype=wheel_actual.dtype
    ).squeeze(0)
    semantic_target_left_mean = float(0.5 * (semantic_wheel_target[0].item() + semantic_wheel_target[1].item()))
    semantic_target_right_mean = float(0.5 * (semantic_wheel_target[2].item() + semantic_wheel_target[3].item()))
    semantic_target_diff = semantic_target_right_mean - semantic_target_left_mean
    semantic_target_common = 0.5 * (semantic_target_left_mean + semantic_target_right_mean)
    semantic_actual_left_mean = float(0.5 * (semantic_wheel_actual[0].item() + semantic_wheel_actual[1].item()))
    semantic_actual_right_mean = float(0.5 * (semantic_wheel_actual[2].item() + semantic_wheel_actual[3].item()))

    return {
        "case": case_name,
        "step": step,
        "base_lin_vel_x": float(robot.data.root_lin_vel_b[env_id, 0].item()),
        "base_lin_vel_y": float(robot.data.root_lin_vel_b[env_id, 1].item()),
        "base_ang_vel_z": float(robot.data.root_ang_vel_b[env_id, 2].item()),
        "target_lf": target_lf,
        "target_lr": target_lr,
        "target_rf": target_rf,
        "target_rr": target_rr,
        "actual_lf": actual_lf,
        "actual_lr": actual_lr,
        "actual_rf": actual_rf,
        "actual_rr": actual_rr,
        "contact_force_lf": contact_force_lf,
        "contact_force_lr": contact_force_lr,
        "contact_force_rf": contact_force_rf,
        "contact_force_rr": contact_force_rr,
        "contact_bool_lf": contact_bool_lf,
        "contact_bool_lr": contact_bool_lr,
        "contact_bool_rf": contact_bool_rf,
        "contact_bool_rr": contact_bool_rr,
        "actual_left_front_rear_diff": abs(actual_lf - actual_lr),
        "actual_right_front_rear_diff": abs(actual_rf - actual_rr),
        "velocity_tracking_abs_error_mean": velocity_tracking_abs_error_mean,
        "semantic_target_left_mean": semantic_target_left_mean,
        "semantic_target_right_mean": semantic_target_right_mean,
        "semantic_target_diff": semantic_target_diff,
        "semantic_target_common": semantic_target_common,
        "semantic_actual_left_mean": semantic_actual_left_mean,
        "semantic_actual_right_mean": semantic_actual_right_mean,
    }


def _write_case_csv(csv_path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _move_recorded_videos(run_root: Path, case_video_steps: dict[str, int]) -> None:
    video_root = run_root / "videos"
    for case_name, start_step in case_video_steps.items():
        src = video_root / f"rl-video-step-{start_step}.mp4"
        dst_dir = run_root / case_name / "video"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{case_name}.mp4"
        if src.exists():
            src.rename(dst)


def _run_all_cases(
    run_root: Path,
    case_total_steps: int,
    settle_steps: int,
) -> list[dict[str, float]]:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = max(float(args_cli.duration_s + args_cli.settle_s + 1.0), float(env_cfg.episode_length_s))
    for case_name, _ in CASE_DEFINITIONS:
        (run_root / case_name).mkdir(parents=True, exist_ok=True)

    case_video_steps = {case_name: index * case_total_steps for index, (case_name, _) in enumerate(CASE_DEFINITIONS)}
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(run_root / "videos"),
        step_trigger=lambda step: step in set(case_video_steps.values()),
        video_length=case_total_steps,
        disable_logger=True,
    )

    try:
        env.reset()
        unwrapped = env.unwrapped
        if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
            raise ValueError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}.")

        robot = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
        wheel_joint_ids, _ = robot.find_joints(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
        wheel_body_ids, _ = contact_sensor.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
        leg_action_slice = _get_action_slice(unwrapped, "leg_hydraulic")
        wheel_action_slice = _get_action_slice(unwrapped, "wheel_motor_csv")
        zero_wheel_target = torch.zeros(4, device=unwrapped.device, dtype=torch.float32)
        zero_actions = _build_actions(
            env,
            leg_action_slice=leg_action_slice,
            wheel_action_slice=wheel_action_slice,
            wheel_targets_rad_s=zero_wheel_target,
            neutral_leg_action=args_cli.neutral_leg_action,
        )

        summaries = []
        with torch.inference_mode():
            for case_name, case_targets in CASE_DEFINITIONS:
                env.reset()
                print(
                    f"[CASE] {case_name}: targets(lf,lr,rf,rr)="
                    f"({case_targets['lf']:+.2f}, {case_targets['lr']:+.2f}, {case_targets['rf']:+.2f}, {case_targets['rr']:+.2f}) rad/s",
                    flush=True,
                )
                print(f"[CASE] video dir: {run_root / case_name / 'video'}", flush=True)

                rows: list[dict[str, float | int | str]] = []
                test_actions = _build_actions(
                    env,
                    leg_action_slice=leg_action_slice,
                    wheel_action_slice=wheel_action_slice,
                    wheel_targets_rad_s=_case_target_tensor(unwrapped.device, case_targets),
                    neutral_leg_action=args_cli.neutral_leg_action,
                )

                for step in range(case_total_steps):
                    actions = zero_actions if step < settle_steps else test_actions
                    env.step(actions)
                    row = _collect_step_row(
                        env,
                        env_id=args_cli.env_id,
                        case_name=case_name,
                        step=step,
                        wheel_joint_ids=wheel_joint_ids,
                        wheel_body_ids=wheel_body_ids,
                    )
                    row["phase"] = "settle" if step < settle_steps else "command"
                    rows.append(row)

                    if step % max(args_cli.print_every, 1) == 0:
                        print(
                            f"[STEP {step:04d}] phase={row['phase']} "
                            f"vx={row['base_lin_vel_x']:+.3f} vy={row['base_lin_vel_y']:+.3f} wz={row['base_ang_vel_z']:+.3f} "
                            f"tgt(lf,lr,rf,rr)=({row['target_lf']:+.2f}, {row['target_lr']:+.2f}, {row['target_rf']:+.2f}, {row['target_rr']:+.2f}) "
                            f"act(lf,lr,rf,rr)=({row['actual_lf']:+.2f}, {row['actual_lr']:+.2f}, {row['actual_rf']:+.2f}, {row['actual_rr']:+.2f}) "
                            f"sem(diff,common)=({row['semantic_target_diff']:+.2f}, {row['semantic_target_common']:+.2f}) "
                            f"cf(lf,lr,rf,rr)=({row['contact_force_lf']:.1f}, {row['contact_force_lr']:.1f}, {row['contact_force_rf']:.1f}, {row['contact_force_rr']:.1f})",
                            flush=True,
                        )

                csv_path = run_root / case_name / "metrics.csv"
                _write_case_csv(csv_path, rows)
                print(f"[CASE] wrote metrics: {csv_path}", flush=True)

                command_rows = [row for row in rows if row["phase"] == "command"]
                summaries.append(
                    {
                        "case": case_name,
                        "mean_base_lin_vel_x": sum(float(row["base_lin_vel_x"]) for row in command_rows)
                        / max(len(command_rows), 1),
                        "mean_base_lin_vel_y": sum(float(row["base_lin_vel_y"]) for row in command_rows)
                        / max(len(command_rows), 1),
                        "mean_base_ang_vel_z": sum(float(row["base_ang_vel_z"]) for row in command_rows)
                        / max(len(command_rows), 1),
                        "mean_actual_left_front_rear_diff": sum(
                            float(row["actual_left_front_rear_diff"]) for row in command_rows
                        )
                        / max(len(command_rows), 1),
                        "mean_actual_right_front_rear_diff": sum(
                            float(row["actual_right_front_rear_diff"]) for row in command_rows
                        )
                        / max(len(command_rows), 1),
                        "mean_velocity_tracking_abs_error": sum(
                            float(row["velocity_tracking_abs_error_mean"]) for row in command_rows
                        )
                        / max(len(command_rows), 1),
                        "mean_semantic_target_diff": sum(float(row["semantic_target_diff"]) for row in command_rows)
                        / max(len(command_rows), 1),
                        "mean_semantic_target_common": sum(float(row["semantic_target_common"]) for row in command_rows)
                        / max(len(command_rows), 1),
                    }
                )
        return summaries
    finally:
        env.close()
        _move_recorded_videos(run_root, case_video_steps)


def main() -> None:
    run_root = Path(args_cli.output_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root.mkdir(parents=True, exist_ok=True)

    dummy_env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    step_dt = float(dummy_env_cfg.sim.dt) * float(dummy_env_cfg.decimation)
    total_steps = max(int(round(float(args_cli.duration_s) / step_dt)), 1)
    settle_steps = max(int(round(float(args_cli.settle_s) / step_dt)), 0)

    print(f"[INFO] output root: {run_root}", flush=True)
    print(f"[INFO] task: {args_cli.task}", flush=True)
    print(f"[INFO] step_dt: {step_dt:.6f} s", flush=True)
    print(f"[INFO] settle_steps: {settle_steps}", flush=True)
    print(f"[INFO] command_steps: {total_steps}", flush=True)
    print(f"[INFO] neutral_leg_action: {args_cli.neutral_leg_action:+.3f}", flush=True)

    summaries = _run_all_cases(run_root=run_root, case_total_steps=total_steps + settle_steps, settle_steps=settle_steps)

    print("\n[SUMMARY] fixed wheel-target sanity test", flush=True)
    for summary in summaries:
        print(
            f"{summary['case']}: "
            f"mean_vx={summary['mean_base_lin_vel_x']:+.3f} "
            f"mean_vy={summary['mean_base_lin_vel_y']:+.3f} "
            f"mean_wz={summary['mean_base_ang_vel_z']:+.3f} "
            f"mean_semantic_diff={summary['mean_semantic_target_diff']:+.3f} "
            f"mean_semantic_common={summary['mean_semantic_target_common']:+.3f} "
            f"mean_left_fr_diff={summary['mean_actual_left_front_rear_diff']:.3f} "
            f"mean_right_fr_diff={summary['mean_actual_right_front_rear_diff']:.3f} "
            f"mean_tracking_abs_error={summary['mean_velocity_tracking_abs_error']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
