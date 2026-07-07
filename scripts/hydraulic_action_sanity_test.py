#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run fixed Ranger hydraulic-action sanity tests without policy involvement."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

faulthandler.enable()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run fixed Ranger hydraulic-action sanity tests.")
parser.add_argument("--task", type=str, default="Template-Ranger-Stand-v0", help="Task used for the sanity test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to report.")
parser.add_argument("--duration_s", type=float, default=4.0, help="Seconds to run each fixed action case.")
parser.add_argument("--settle_s", type=float, default=1.0, help="Seconds to hold zero action before each case.")
parser.add_argument("--print_every", type=int, default=15, help="Print one line every N steps.")
parser.add_argument(
    "--mode",
    type=str,
    default="full",
    choices=("zero_step_only", "full"),
    help="Run a minimal zero-action smoke test first, or all hydraulic action cases.",
)
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/hydraulic_action_sanity",
    help="Directory where CSV files will be written.",
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
from isaaclab_tasks.utils import parse_env_cfg

import Ranger.tasks  # noqa: F401
from Ranger.tasks.manager_based.ranger import forward_debug_env as ranger_forward_debug_env

if not hasattr(ranger_forward_debug_env, "SceneEntityCfg"):
    ranger_forward_debug_env.SceneEntityCfg = SceneEntityCfg


CASE_DEFINITIONS = (
    ("all_plus_one", {"lf": 1.0, "lr": 1.0, "rf": 1.0, "rr": 1.0}),
    ("all_minus_one", {"lf": -1.0, "lr": -1.0, "rf": -1.0, "rr": -1.0}),
    ("only_lf_plus_one", {"lf": 1.0, "lr": 0.0, "rf": 0.0, "rr": 0.0}),
    ("only_lr_plus_one", {"lf": 0.0, "lr": 1.0, "rf": 0.0, "rr": 0.0}),
    ("only_rf_plus_one", {"lf": 0.0, "lr": 0.0, "rf": 1.0, "rr": 0.0}),
    ("only_rr_plus_one", {"lf": 0.0, "lr": 0.0, "rf": 0.0, "rr": 1.0}),
)

WHEEL_BODY_NAMES = ["w_lb", "w_lf", "w_rf", "w_rb"]
ACTION_ORDER = ("lr", "lf", "rf", "rr")


def _get_action_slice(unwrapped_env, term_name: str) -> slice:
    start = 0
    for active_name, dim in zip(unwrapped_env.action_manager.active_terms, unwrapped_env.action_manager.action_term_dim):
        if active_name == term_name:
            return slice(start, start + dim)
        start += dim
    raise KeyError(f"Action term '{term_name}' not found in action manager.")


def _case_action_tensor(device: torch.device, case_actions: dict[str, float]) -> torch.Tensor:
    return torch.tensor(
        [float(case_actions[key]) for key in ACTION_ORDER],
        device=device,
        dtype=torch.float32,
    )


def _peak_contact_force_and_bool(contact_sensor, env_id: int, body_ids: torch.Tensor, threshold: float = 1.0):
    force_history = contact_sensor.data.net_forces_w_history[env_id, :, body_ids, :]
    force_norm = torch.norm(force_history, dim=-1)
    peak_force_norm = torch.max(force_norm, dim=0).values
    contact_bool = peak_force_norm > float(threshold)
    return peak_force_norm, contact_bool


def _write_case_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: Any, env_id: int) -> bool:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return bool(value.item())
        return bool(value[env_id].item())
    if isinstance(value, (list, tuple)):
        return bool(value[env_id])
    return bool(value)


def _build_zero_actions(env) -> torch.Tensor:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "device"):
        device = unwrapped.device
    elif hasattr(env, "device"):
        device = env.device
    else:
        raise RuntimeError("Unable to determine environment device for action tensor creation.")
    action_space_shape = tuple(int(v) for v in env.action_space.shape)
    if len(action_space_shape) == 2:
        action_shape = action_space_shape
    elif len(action_space_shape) == 1:
        action_shape = (int(unwrapped.num_envs), action_space_shape[0])
    else:
        raise RuntimeError(f"Unsupported action_space shape: {action_space_shape}")
    return torch.zeros(action_shape, device=device, dtype=torch.float32)


def _build_case_actions(env, leg_action_slice: slice, wheel_action_slice: slice, case_actions: dict[str, float]) -> torch.Tensor:
    unwrapped = env.unwrapped
    actions = _build_zero_actions(env)
    leg_actions = _case_action_tensor(unwrapped.device, case_actions)
    actions[:, leg_action_slice] = leg_actions.unsqueeze(0).repeat(actions.shape[0], 1)
    actions[:, wheel_action_slice] = 0.0
    return actions


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


def _collect_step_row(
    env,
    env_id: int,
    case_name: str,
    step: int,
    phase: str,
    wheel_body_ids: torch.Tensor,
    commanded_leg_actions_lr_lf_rf_rr: torch.Tensor,
    terminated: bool = False,
    truncated: bool = False,
    termination_reason: str = "",
) -> dict[str, Any]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    leg_action_term = unwrapped.action_manager.get_term("leg_hydraulic")
    contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]

    stroke_actual = leg_action_term.stroke_actual[env_id]
    peak_contact_force, contact_bool = _peak_contact_force_and_bool(contact_sensor, env_id, wheel_body_ids)
    root_lin_vel_b = robot.data.root_lin_vel_b[env_id]
    root_ang_vel_b = robot.data.root_ang_vel_b[env_id]

    row: dict[str, Any] = {
        "case": case_name,
        "step": int(step),
        "phase": phase,
        "terminated": int(terminated),
        "truncated": int(truncated),
        "termination_reason": termination_reason,
        "root_height": float(robot.data.root_pos_w[env_id, 2].item()),
        "base_lin_vel_x": float(root_lin_vel_b[0].item()),
        "base_lin_vel_y": float(root_lin_vel_b[1].item()),
        "base_lin_vel_z": float(root_lin_vel_b[2].item()),
        "base_ang_vel_x": float(root_ang_vel_b[0].item()),
        "base_ang_vel_y": float(root_ang_vel_b[1].item()),
        "base_ang_vel_z": float(root_ang_vel_b[2].item()),
        "hydraulic_action_lr": float(commanded_leg_actions_lr_lf_rf_rr[0].item()),
        "hydraulic_action_lf": float(commanded_leg_actions_lr_lf_rf_rr[1].item()),
        "hydraulic_action_rf": float(commanded_leg_actions_lr_lf_rf_rr[2].item()),
        "hydraulic_action_rr": float(commanded_leg_actions_lr_lf_rf_rr[3].item()),
        "hydraulic_stroke_lr": float(stroke_actual[0].item()),
        "hydraulic_stroke_lf": float(stroke_actual[1].item()),
        "hydraulic_stroke_rf": float(stroke_actual[2].item()),
        "hydraulic_stroke_rr": float(stroke_actual[3].item()),
        "wheel_target_lr": 0.0,
        "wheel_target_lf": 0.0,
        "wheel_target_rf": 0.0,
        "wheel_target_rr": 0.0,
        "contact_force_lr": float(peak_contact_force[0].item()),
        "contact_force_lf": float(peak_contact_force[1].item()),
        "contact_force_rf": float(peak_contact_force[2].item()),
        "contact_force_rr": float(peak_contact_force[3].item()),
        "contact_bool_lr": float(contact_bool[0].to(torch.float32).item()),
        "contact_bool_lf": float(contact_bool[1].to(torch.float32).item()),
        "contact_bool_rf": float(contact_bool[2].to(torch.float32).item()),
        "contact_bool_rr": float(contact_bool[3].to(torch.float32).item()),
    }
    return row


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / float(len(rows))


def _case_summary(case_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    initial_row = rows[0]
    final_row = rows[-1]
    command_rows = [row for row in rows if row["phase"] == "command"]
    event_rows = [row for row in rows if row["terminated"] or row["truncated"]]
    event_row = event_rows[-1] if event_rows else None
    return {
        "case": case_name,
        "terminated_step": int(event_row["step"]) if event_row is not None else -1,
        "termination_reason": str(event_row["termination_reason"]) if event_row is not None else "",
        "completed_all_command_steps": int(len(command_rows) > 0 and event_row is None),
        "initial_hydraulic_stroke_lf": float(initial_row["hydraulic_stroke_lf"]),
        "initial_hydraulic_stroke_lr": float(initial_row["hydraulic_stroke_lr"]),
        "initial_hydraulic_stroke_rf": float(initial_row["hydraulic_stroke_rf"]),
        "initial_hydraulic_stroke_rr": float(initial_row["hydraulic_stroke_rr"]),
        "final_hydraulic_stroke_lf": float(final_row["hydraulic_stroke_lf"]),
        "final_hydraulic_stroke_lr": float(final_row["hydraulic_stroke_lr"]),
        "final_hydraulic_stroke_rf": float(final_row["hydraulic_stroke_rf"]),
        "final_hydraulic_stroke_rr": float(final_row["hydraulic_stroke_rr"]),
        "delta_hydraulic_stroke_lf": float(final_row["hydraulic_stroke_lf"]) - float(initial_row["hydraulic_stroke_lf"]),
        "delta_hydraulic_stroke_lr": float(final_row["hydraulic_stroke_lr"]) - float(initial_row["hydraulic_stroke_lr"]),
        "delta_hydraulic_stroke_rf": float(final_row["hydraulic_stroke_rf"]) - float(initial_row["hydraulic_stroke_rf"]),
        "delta_hydraulic_stroke_rr": float(final_row["hydraulic_stroke_rr"]) - float(initial_row["hydraulic_stroke_rr"]),
        "initial_root_height": float(initial_row["root_height"]),
        "final_root_height": float(final_row["root_height"]),
        "delta_root_height": float(final_row["root_height"]) - float(initial_row["root_height"]),
        "mean_contact_force_lf": _mean(command_rows, "contact_force_lf"),
        "mean_contact_force_lr": _mean(command_rows, "contact_force_lr"),
        "mean_contact_force_rf": _mean(command_rows, "contact_force_rf"),
        "mean_contact_force_rr": _mean(command_rows, "contact_force_rr"),
        "mean_base_lin_vel_x": _mean(command_rows, "base_lin_vel_x"),
        "mean_base_lin_vel_y": _mean(command_rows, "base_lin_vel_y"),
        "mean_base_lin_vel_z": _mean(command_rows, "base_lin_vel_z"),
        "mean_base_ang_vel_x": _mean(command_rows, "base_ang_vel_x"),
        "mean_base_ang_vel_y": _mean(command_rows, "base_ang_vel_y"),
        "mean_base_ang_vel_z": _mean(command_rows, "base_ang_vel_z"),
    }


def _print_case_summary(summary: dict[str, Any]) -> None:
    print(
        f"[SUMMARY][{summary['case']}] "
        f"terminated_step={summary['terminated_step']} "
        f"completed_all_command_steps={summary['completed_all_command_steps']} "
        f"reason={summary['termination_reason'] or 'none'}",
        flush=True,
    )
    print(
        f"  initial_stroke(lf,lr,rf,rr)=("
        f"{summary['initial_hydraulic_stroke_lf']:.4f}, {summary['initial_hydraulic_stroke_lr']:.4f}, "
        f"{summary['initial_hydraulic_stroke_rf']:.4f}, {summary['initial_hydraulic_stroke_rr']:.4f})",
        flush=True,
    )
    print(
        f"  final_stroke(lf,lr,rf,rr)=("
        f"{summary['final_hydraulic_stroke_lf']:.4f}, {summary['final_hydraulic_stroke_lr']:.4f}, "
        f"{summary['final_hydraulic_stroke_rf']:.4f}, {summary['final_hydraulic_stroke_rr']:.4f})",
        flush=True,
    )
    print(
        f"  delta_stroke(lf,lr,rf,rr)=("
        f"{summary['delta_hydraulic_stroke_lf']:+.4f}, {summary['delta_hydraulic_stroke_lr']:+.4f}, "
        f"{summary['delta_hydraulic_stroke_rf']:+.4f}, {summary['delta_hydraulic_stroke_rr']:+.4f})",
        flush=True,
    )
    print(
        f"  root_height initial={summary['initial_root_height']:.4f} "
        f"final={summary['final_root_height']:.4f} delta={summary['delta_root_height']:+.4f}",
        flush=True,
    )
    print(
        f"  mean_contact_force(lf,lr,rf,rr)=("
        f"{summary['mean_contact_force_lf']:.2f}, {summary['mean_contact_force_lr']:.2f}, "
        f"{summary['mean_contact_force_rf']:.2f}, {summary['mean_contact_force_rr']:.2f})",
        flush=True,
    )
    print(
        f"  mean_base_lin_vel(x,y,z)=("
        f"{summary['mean_base_lin_vel_x']:+.4f}, {summary['mean_base_lin_vel_y']:+.4f}, {summary['mean_base_lin_vel_z']:+.4f})",
        flush=True,
    )
    print(
        f"  mean_base_ang_vel(x,y,z)=("
        f"{summary['mean_base_ang_vel_x']:+.4f}, {summary['mean_base_ang_vel_y']:+.4f}, {summary['mean_base_ang_vel_z']:+.4f})",
        flush=True,
    )


def _make_env():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = max(float(args_cli.duration_s + args_cli.settle_s + 1.0), float(env_cfg.episode_length_s))
    print("[DEBUG] before gym.make()", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[DEBUG] after gym.make()", flush=True)
    return env, env_cfg


def _run_zero_step_only(zero_step_count: int = 10) -> None:
    env, _ = _make_env()
    try:
        print("[DEBUG] before env.reset()", flush=True)
        try:
            env.reset()
        except SystemExit as exc:
            print(f"[DEBUG] SystemExit during env.reset(): {exc!r}", flush=True)
            raise
        except BaseException as exc:
            print(f"[DEBUG] reset exception type={type(exc)!r} repr={exc!r}", flush=True)
            print(traceback.format_exc(), flush=True)
            raise
        print("[DEBUG] after env.reset()", flush=True)
        unwrapped = env.unwrapped
        print("[DEBUG] before contact_sensor lookup", flush=True)
        contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
        print("[DEBUG] after contact_sensor lookup", flush=True)
        print("[DEBUG] before find_bodies", flush=True)
        wheel_body_ids, _ = contact_sensor.find_bodies(WHEEL_BODY_NAMES, preserve_order=True)
        print(f"[DEBUG] after find_bodies ids={wheel_body_ids}", flush=True)
        print("[DEBUG] before build_zero_actions", flush=True)
        zero_actions = _build_zero_actions(env)
        print("[DEBUG] after build_zero_actions", flush=True)
        print(f"[ZERO_STEP] action_space_shape={tuple(int(v) for v in env.action_space.shape)}", flush=True)
        print(f"[ZERO_STEP] zero_actions_shape={tuple(int(v) for v in zero_actions.shape)}", flush=True)
        with torch.no_grad():
            for step in range(zero_step_count):
                print(f"[ZERO_STEP] i={step} before env.step()", flush=True)
                try:
                    _step_env(env, zero_actions)
                except SystemExit as exc:
                    print(f"[ZERO_STEP] SystemExit: {exc!r}", flush=True)
                    raise
                except BaseException as exc:
                    print(f"[ZERO_STEP] step exception type={type(exc)!r} repr={exc!r}", flush=True)
                    print(traceback.format_exc(), flush=True)
                    raise
                row = _collect_step_row(
                    env,
                    env_id=args_cli.env_id,
                    case_name="zero_step_only",
                    step=step,
                    phase="zero",
                    wheel_body_ids=wheel_body_ids,
                    commanded_leg_actions_lr_lf_rf_rr=torch.zeros(4, device=unwrapped.device, dtype=torch.float32),
                )
                print(
                    f"[ZERO_STEP] i={step} after env.step() "
                    f"root_height={row['root_height']:.4f} "
                    f"stroke(lf,lr,rf,rr)=("
                    f"{row['hydraulic_stroke_lf']:.4f}, {row['hydraulic_stroke_lr']:.4f}, "
                    f"{row['hydraulic_stroke_rf']:.4f}, {row['hydraulic_stroke_rr']:.4f})",
                    flush=True,
                )
    finally:
        env.close()


def _run_single_case(
    env,
    case_dir: Path,
    case_name: str,
    case_actions: dict[str, float],
    command_steps: int,
    settle_steps: int,
) -> dict[str, Any]:
    print(f"[DEBUG][{case_name}] before env.reset()", flush=True)
    try:
        env.reset()
    except SystemExit as exc:
        print(f"[DEBUG][{case_name}] SystemExit during env.reset(): {exc!r}", flush=True)
        raise
    except BaseException as exc:
        print(f"[DEBUG][{case_name}] reset exception type={type(exc)!r} repr={exc!r}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    print(f"[DEBUG][{case_name}] after env.reset()", flush=True)
    unwrapped = env.unwrapped
    if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
        raise ValueError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}.")

    print(f"[DEBUG][{case_name}] before contact_sensor lookup", flush=True)
    contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    print(f"[DEBUG][{case_name}] after contact_sensor lookup", flush=True)
    print(f"[DEBUG][{case_name}] before find_bodies", flush=True)
    wheel_body_ids, _ = contact_sensor.find_bodies(WHEEL_BODY_NAMES, preserve_order=True)
    print(f"[DEBUG][{case_name}] after find_bodies ids={wheel_body_ids}", flush=True)
    leg_action_slice = _get_action_slice(unwrapped, "leg_hydraulic")
    wheel_action_slice = _get_action_slice(unwrapped, "wheel_motor_csv")
    print(f"[DEBUG][{case_name}] before build_zero_actions", flush=True)
    zero_actions = _build_zero_actions(env)
    print(f"[DEBUG][{case_name}] after build_zero_actions", flush=True)
    test_actions = _build_case_actions(env, leg_action_slice, wheel_action_slice, case_actions)
    zero_leg_actions = torch.zeros(4, device=unwrapped.device, dtype=torch.float32)
    command_leg_actions = _case_action_tensor(unwrapped.device, case_actions)

    print(
        f"[CASE] {case_name}: hydraulic(lf,lr,rf,rr)="
        f"({case_actions['lf']:+.2f}, {case_actions['lr']:+.2f}, {case_actions['rf']:+.2f}, {case_actions['rr']:+.2f})",
        flush=True,
    )
    print(
        f"[CASE] action_space_shape={tuple(int(v) for v in env.action_space.shape)} "
        f"zero_actions_shape={tuple(int(v) for v in zero_actions.shape)} "
        f"test_actions_shape={tuple(int(v) for v in test_actions.shape)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    total_steps = settle_steps + command_steps
    with torch.no_grad():
        for step in range(total_steps):
            phase = "settle" if step < settle_steps else "command"
            actions = zero_actions if phase == "settle" else test_actions
            commanded_leg_actions = zero_leg_actions if phase == "settle" else command_leg_actions
            if step == 0:
                print("[CASE] before first env.step()", flush=True)
            try:
                _, _, terminated, truncated, _ = _step_env(env, actions)
            except SystemExit as exc:
                print(f"[CASE] SystemExit during step: {exc!r}", flush=True)
                raise
            except BaseException as exc:
                print(f"[CASE] step exception type={type(exc)!r} repr={exc!r}", flush=True)
                print(traceback.format_exc(), flush=True)
                raise
            if step == 0:
                print("[CASE] after first env.step()", flush=True)

            terminated_flag = _as_bool(terminated, args_cli.env_id)
            truncated_flag = _as_bool(truncated, args_cli.env_id)
            termination_reason = _termination_reason(unwrapped, args_cli.env_id, terminated_flag, truncated_flag)
            row = _collect_step_row(
                env,
                env_id=args_cli.env_id,
                case_name=case_name,
                step=step,
                phase=phase,
                wheel_body_ids=wheel_body_ids,
                commanded_leg_actions_lr_lf_rf_rr=commanded_leg_actions,
                terminated=terminated_flag,
                truncated=truncated_flag,
                termination_reason=termination_reason,
            )
            rows.append(row)

            if step == 0:
                print(
                    f"[CASE] initial "
                    f"h={row['root_height']:.4f} "
                    f"stroke(lf,lr,rf,rr)=({row['hydraulic_stroke_lf']:.4f}, {row['hydraulic_stroke_lr']:.4f}, "
                    f"{row['hydraulic_stroke_rf']:.4f}, {row['hydraulic_stroke_rr']:.4f})",
                    flush=True,
                )
            if step % max(args_cli.print_every, 1) == 0 or terminated_flag or truncated_flag:
                print(
                    f"[STEP {step:04d}] phase={phase} "
                    f"h={row['root_height']:.3f} "
                    f"lin(x,y,z)=({row['base_lin_vel_x']:+.3f}, {row['base_lin_vel_y']:+.3f}, {row['base_lin_vel_z']:+.3f}) "
                    f"ang(x,y,z)=({row['base_ang_vel_x']:+.3f}, {row['base_ang_vel_y']:+.3f}, {row['base_ang_vel_z']:+.3f}) "
                    f"act(lf,lr,rf,rr)=({row['hydraulic_action_lf']:+.2f}, {row['hydraulic_action_lr']:+.2f}, "
                    f"{row['hydraulic_action_rf']:+.2f}, {row['hydraulic_action_rr']:+.2f}) "
                    f"stroke(lf,lr,rf,rr)=({row['hydraulic_stroke_lf']:.3f}, {row['hydraulic_stroke_lr']:.3f}, "
                    f"{row['hydraulic_stroke_rf']:.3f}, {row['hydraulic_stroke_rr']:.3f}) "
                    f"contact(lf,lr,rf,rr)=({row['contact_force_lf']:.1f}, {row['contact_force_lr']:.1f}, "
                    f"{row['contact_force_rf']:.1f}, {row['contact_force_rr']:.1f}) "
                    f"term={termination_reason or 'none'}",
                    flush=True,
                )
            if terminated_flag or truncated_flag:
                break

    csv_path = case_dir / "timeseries.csv"
    _write_case_csv(csv_path, rows)
    print(f"[CASE] wrote timeseries: {csv_path}", flush=True)
    summary = _case_summary(case_name, rows)
    _print_case_summary(summary)
    return summary


def _run_all_cases(run_root: Path, command_steps: int, settle_steps: int) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    env, _ = _make_env()
    try:
        for case_name, case_actions in CASE_DEFINITIONS:
            case_dir = run_root / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            summaries.append(
                _run_single_case(
                    env=env,
                    case_dir=case_dir,
                    case_name=case_name,
                    case_actions=case_actions,
                    command_steps=command_steps,
                    settle_steps=settle_steps,
                )
            )
    finally:
        env.close()
    return summaries


def main() -> None:
    run_root = Path(args_cli.output_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root.mkdir(parents=True, exist_ok=True)

    dummy_env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    step_dt = float(dummy_env_cfg.sim.dt) * float(dummy_env_cfg.decimation)
    command_steps = max(int(round(float(args_cli.duration_s) / step_dt)), 1)
    settle_steps = max(int(round(float(args_cli.settle_s) / step_dt)), 0)

    print(f"[INFO] output root: {run_root}", flush=True)
    print(f"[INFO] task: {args_cli.task}", flush=True)
    print(f"[INFO] step_dt: {step_dt:.6f} s", flush=True)
    print(f"[INFO] settle_steps: {settle_steps}", flush=True)
    print(f"[INFO] command_steps: {command_steps}", flush=True)
    print(f"[INFO] mode: {args_cli.mode}", flush=True)

    if args_cli.mode == "zero_step_only":
        _run_zero_step_only()
        return

    summaries = _run_all_cases(run_root=run_root, command_steps=command_steps, settle_steps=settle_steps)
    summary_csv_path = run_root / "summary.csv"
    _write_case_csv(summary_csv_path, summaries)
    print(f"\n[SUMMARY] wrote case summary: {summary_csv_path}", flush=True)
    for summary in summaries:
        print(
            f"{summary['case']}: "
            f"completed_all_command_steps={summary['completed_all_command_steps']} "
            f"delta_root_height={summary['delta_root_height']:+.4f} "
            f"delta_stroke(lf,lr,rf,rr)=("
            f"{summary['delta_hydraulic_stroke_lf']:+.4f}, "
            f"{summary['delta_hydraulic_stroke_lr']:+.4f}, "
            f"{summary['delta_hydraulic_stroke_rf']:+.4f}, "
            f"{summary['delta_hydraulic_stroke_rr']:+.4f}) "
            f"terminated_step={summary['terminated_step']} "
            f"reason={summary['termination_reason'] or 'none'}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
