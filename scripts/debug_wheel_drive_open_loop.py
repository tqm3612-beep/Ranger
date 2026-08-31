#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run open-loop Ranger wheel-drive patterns without PPO.")
parser.add_argument("--task", type=str, default="Template-Ranger-ShortGoalFlat-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--env_id", type=int, default=0)
parser.add_argument("--duration", "--duration_s", dest="duration_s", type=float, default=2.0)
parser.add_argument("--settle_s", type=float, default=1.0)
parser.add_argument("--free_spin_lift", type=float, default=0.8, help="Extra root height added for free-spin tests.")
parser.add_argument("--print_every", type=int, default=20)
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/debug_wheel_drive_open_loop",
    help="Directory for per-pattern CSV output.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from pxr import UsdPhysics

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Ranger.tasks  # noqa: F401
from Ranger.tasks.manager_based.ranger import mdp


PATTERNS = (
    ("semantic_forward", (1.0, 1.0, 1.0, 1.0)),
    ("semantic_backward", (-1.0, -1.0, -1.0, -1.0)),
    ("turn_left", (-1.0, -1.0, 1.0, 1.0)),
    ("turn_right", (1.0, 1.0, -1.0, -1.0)),
)

SWEEP_PATTERNS = (
    ("semantic_forward", (1.0, 1.0, 1.0, 1.0)),
    ("semantic_backward", (-1.0, -1.0, -1.0, -1.0)),
)

SCALES = (20.0, 50.0, 100.0, 200.0, 400.0)


def _get_action_slice(unwrapped_env, term_name: str) -> slice:
    start = 0
    for active_name, dim in zip(unwrapped_env.action_manager.active_terms, unwrapped_env.action_manager.action_term_dim):
        if active_name == term_name:
            return slice(start, start + dim)
        start += dim
    raise KeyError(f"Action term '{term_name}' not found.")


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


def _format_vec(values: list[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


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


def _format_scalar(value: float) -> str:
    return f"{value:.4f}"


def _usd_joint_debug_info(stage, joint_name: str) -> dict[str, object]:
    joint_prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/joints/{joint_name}")
    if not joint_prim.IsValid():
        return {"joint_name": joint_name, "warning": f"joint prim not found at {joint_prim.GetPath()}"}

    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        joint_api = UsdPhysics.RevoluteJoint(joint_prim)
        drive_kind = "angular"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        joint_api = UsdPhysics.PrismaticJoint(joint_prim)
        drive_kind = "linear"
    else:
        joint_api = None
        drive_kind = None

    info: dict[str, object] = {
        "joint_name": joint_name,
        "joint_type": joint_prim.GetTypeName(),
        "axis": joint_api.GetAxisAttr().Get() if joint_api else None,
        "lower_limit": joint_api.GetLowerLimitAttr().Get() if joint_api else None,
        "upper_limit": joint_api.GetUpperLimitAttr().Get() if joint_api else None,
    }
    attr_names = {
        "drive_stiffness": f"drive:{drive_kind}:physics:stiffness" if drive_kind else None,
        "drive_damping": f"drive:{drive_kind}:physics:damping" if drive_kind else None,
        "drive_max_force": f"drive:{drive_kind}:physics:maxForce" if drive_kind else None,
        "joint_friction": "physxJoint:jointFriction",
        "armature": "physxJoint:armature",
    }
    for key, attr_name in attr_names.items():
        if attr_name is None:
            info[key] = None
            continue
        attr = joint_prim.GetAttribute(attr_name)
        info[key] = attr.Get() if attr.IsValid() else None
    return info


def _wheel_actuator_info(env) -> dict[str, object]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    wheel_action_cfg = unwrapped.cfg.actions.wheel_motor_csv
    asset_cfg = getattr(robot, "cfg", None)
    wheel_asset_actuator_cfg = None
    if asset_cfg is not None and hasattr(asset_cfg, "actuators"):
        wheel_asset_actuator_cfg = asset_cfg.actuators.get("wheel_joints")

    stage = getattr(getattr(unwrapped, "sim", None), "stage", None)
    wheel_joint_ids, wheel_joint_names = robot.find_joints(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)

    joint_rows: list[dict[str, object]] = []
    for joint_name, joint_id in zip(wheel_joint_names, wheel_joint_ids):
        row = {
            "joint_name": joint_name,
            "joint_id": int(joint_id),
        }
        if stage is not None:
            row.update(_usd_joint_debug_info(stage, joint_name))
        joint_rows.append(row)

    return {
        "wheel_joint_names": list(wheel_joint_names),
        "wheel_joint_ids": [int(idx) for idx in wheel_joint_ids],
        "wheel_actuator_cfg_type": type(wheel_asset_actuator_cfg).__name__ if wheel_asset_actuator_cfg is not None else "unknown",
        "wheel_action_term_type": type(unwrapped.action_manager.get_term("wheel_motor_csv")).__name__,
        "wheel_action_control_mode": getattr(unwrapped.action_manager.get_term("wheel_motor_csv"), "control_mode", "unknown"),
        "wheel_action_apply_mode": (
            "set_joint_velocity_target"
            if getattr(unwrapped.action_manager.get_term("wheel_motor_csv"), "uses_velocity_target", False)
            else "set_joint_effort_target"
        ),
        "wheel_action_cfg_velocity_limit": float(wheel_action_cfg.velocity_limit),
        "wheel_action_cfg_acceleration_limit": float(wheel_action_cfg.acceleration_limit),
        "wheel_action_cfg_velocity_kp": float(wheel_action_cfg.velocity_kp),
        "wheel_action_cfg_velocity_damping": float(wheel_action_cfg.velocity_damping),
        "wheel_action_cfg_viscous_friction": float(wheel_action_cfg.viscous_friction),
        "wheel_action_cfg_effort_limit": float(wheel_action_cfg.effort_limit),
        "wheel_asset_effort_limit": getattr(wheel_asset_actuator_cfg, "effort_limit", None) if wheel_asset_actuator_cfg is not None else None,
        "wheel_asset_effort_limit_sim": getattr(wheel_asset_actuator_cfg, "effort_limit_sim", None) if wheel_asset_actuator_cfg is not None else None,
        "wheel_asset_velocity_limit": getattr(wheel_asset_actuator_cfg, "velocity_limit", None) if wheel_asset_actuator_cfg is not None else None,
        "wheel_asset_velocity_limit_sim": getattr(wheel_asset_actuator_cfg, "velocity_limit_sim", None) if wheel_asset_actuator_cfg is not None else None,
        "wheel_asset_stiffness": getattr(wheel_asset_actuator_cfg, "stiffness", None) if wheel_asset_actuator_cfg is not None else None,
        "wheel_asset_damping": getattr(wheel_asset_actuator_cfg, "damping", None) if wheel_asset_actuator_cfg is not None else None,
        "joint_rows": joint_rows,
    }


def _print_wheel_actuator_info(info: dict[str, object]) -> None:
    print("[INFO] wheel drive chain audit", flush=True)
    print(f"  wheel_joint_names={info['wheel_joint_names']}", flush=True)
    print(f"  wheel_joint_ids={info['wheel_joint_ids']}", flush=True)
    print(f"  wheel_actuator_type={info['wheel_actuator_cfg_type']}", flush=True)
    print(f"  wheel_action_term_type={info['wheel_action_term_type']}", flush=True)
    print(f"  wheel_action_control_mode={info['wheel_action_control_mode']}", flush=True)
    print(f"  wheel_action_apply_mode={info['wheel_action_apply_mode']}", flush=True)
    print(
        "  wheel_action_cfg: "
        f"velocity_limit={info['wheel_action_cfg_velocity_limit']} "
        f"acceleration_limit={info['wheel_action_cfg_acceleration_limit']} "
        f"velocity_kp={info['wheel_action_cfg_velocity_kp']} "
        f"velocity_damping={info['wheel_action_cfg_velocity_damping']} "
        f"viscous_friction={info['wheel_action_cfg_viscous_friction']} "
        f"effort_limit={info['wheel_action_cfg_effort_limit']}",
        flush=True,
    )
    print(
        "  wheel_asset_actuator_cfg: "
        f"effort_limit={info['wheel_asset_effort_limit']} "
        f"effort_limit_sim={info['wheel_asset_effort_limit_sim']} "
        f"velocity_limit={info['wheel_asset_velocity_limit']} "
        f"velocity_limit_sim={info['wheel_asset_velocity_limit_sim']} "
        f"stiffness={info['wheel_asset_stiffness']} "
        f"damping={info['wheel_asset_damping']}",
        flush=True,
    )
    for row in info["joint_rows"]:
        if "warning" in row:
            print(f"  WARNING {row['joint_name']}: {row['warning']}", flush=True)
            continue
        print(
            f"  {row['joint_name']}: joint_id={row['joint_id']} type={row['joint_type']} axis={row['axis']} "
            f"limits=({_format_scalar(float(row['lower_limit'])) if row['lower_limit'] is not None else 'None'}, "
            f"{_format_scalar(float(row['upper_limit'])) if row['upper_limit'] is not None else 'None'}) "
            f"drive_stiffness={row['drive_stiffness']} drive_damping={row['drive_damping']} "
            f"drive_max_force={row['drive_max_force']} joint_friction={row['joint_friction']} armature={row['armature']}",
            flush=True,
        )


def _set_env_root_lift(robot, env_id: int, lift_amount: float) -> None:
    env_ids = torch.tensor([env_id], device=robot.device, dtype=torch.long)
    root_pose = robot.data.root_pose_w[env_id : env_id + 1].clone()
    root_velocity = robot.data.root_vel_w[env_id : env_id + 1].clone()
    root_pose[:, 2] = root_pose[:, 2] + float(lift_amount)
    root_velocity.zero_()
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def _set_env_root_height(robot, env_id: int, target_height: float) -> None:
    env_ids = torch.tensor([env_id], device=robot.device, dtype=torch.long)
    root_pose = robot.data.root_pose_w[env_id : env_id + 1].clone()
    root_velocity = robot.data.root_vel_w[env_id : env_id + 1].clone()
    root_pose[:, 2] = float(target_height)
    root_velocity.zero_()
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def _peak_contact_force_and_bool(contact_sensor, env_id: int, body_ids: torch.Tensor, threshold: float = 1.0):
    force_history = contact_sensor.data.net_forces_w_history[env_id, :, body_ids, :]
    force_norm = torch.norm(force_history, dim=-1)
    peak_force_norm = torch.max(force_norm, dim=0).values
    contact_bool = peak_force_norm > float(threshold)
    return peak_force_norm, contact_bool


def _collect_row(
    env,
    env_id: int,
    pattern_name: str,
    step_idx: int,
    wheel_joint_ids: torch.Tensor,
    wheel_body_ids: torch.Tensor,
    *,
    mode: str,
    scale: float,
):
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    wheel_action_term = unwrapped.action_manager.get_term("wheel_motor_csv")
    contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    termination_manager = unwrapped.termination_manager

    wheel_target = wheel_action_term.velocity_target[env_id]
    wheel_joint_vel = robot.data.joint_vel[env_id, wheel_joint_ids]
    semantic_wheel_joint_vel = wheel_joint_vel * mdp.wheel_semantic_sign_lr_lf_rf_rr(
        device=wheel_joint_vel.device, dtype=wheel_joint_vel.dtype
    ).squeeze(0)
    peak_contact_force, contact_bool = _peak_contact_force_and_bool(contact_sensor, env_id, wheel_body_ids)

    bad_orientation = (
        float(termination_manager.get_term("bad_orientation")[env_id].float().item())
        if "bad_orientation" in termination_manager.active_terms
        else 0.0
    )
    root_height_low = (
        float(termination_manager.get_term("root_height_low")[env_id].float().item())
        if "root_height_low" in termination_manager.active_terms
        else 0.0
    )

    return {
        "mode": mode,
        "scale": float(scale),
        "pattern": pattern_name,
        "step": step_idx,
        "base_lin_vel_x": float(robot.data.root_lin_vel_b[env_id, 0].item()),
        "base_lin_vel_y": float(robot.data.root_lin_vel_b[env_id, 1].item()),
        "base_yaw_rate": float(robot.data.root_ang_vel_b[env_id, 2].item()),
        "root_height": float(robot.data.root_pos_w[env_id, 2].item()),
        "wheel_target_w_lb": float(wheel_target[0].item()),
        "wheel_target_w_lf": float(wheel_target[1].item()),
        "wheel_target_w_rf": float(wheel_target[2].item()),
        "wheel_target_w_rb": float(wheel_target[3].item()),
        "wheel_joint_vel_w_lb": float(wheel_joint_vel[0].item()),
        "wheel_joint_vel_w_lf": float(wheel_joint_vel[1].item()),
        "wheel_joint_vel_w_rf": float(wheel_joint_vel[2].item()),
        "wheel_joint_vel_w_rb": float(wheel_joint_vel[3].item()),
        "semantic_wheel_joint_vel_lr": float(semantic_wheel_joint_vel[0].item()),
        "semantic_wheel_joint_vel_lf": float(semantic_wheel_joint_vel[1].item()),
        "semantic_wheel_joint_vel_rf": float(semantic_wheel_joint_vel[2].item()),
        "semantic_wheel_joint_vel_rr": float(semantic_wheel_joint_vel[3].item()),
        "wheel_contact_force_w_lb": float(peak_contact_force[0].item()),
        "wheel_contact_force_w_lf": float(peak_contact_force[1].item()),
        "wheel_contact_force_w_rf": float(peak_contact_force[2].item()),
        "wheel_contact_force_w_rb": float(peak_contact_force[3].item()),
        "contact_bool_w_lb": int(contact_bool[0].item()),
        "contact_bool_w_lf": int(contact_bool[1].item()),
        "contact_bool_w_rf": int(contact_bool[2].item()),
        "contact_bool_w_rb": int(contact_bool[3].item()),
        "raw_wheel_action_abs_mean": float(torch.mean(torch.abs(wheel_action_term.raw_actions[env_id])).item()),
        "bad_orientation": bad_orientation,
        "root_height_low": root_height_low,
        "terminated_any": int(bool(bad_orientation > 0.5 or root_height_low > 0.5)),
    }


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_step_debug(row: dict[str, float | int | str]) -> None:
    print(
        "[STEP] "
        f"mode={row['mode']} pattern={row['pattern']} scale={float(row['scale']):.1f} step={row['step']} "
        f"vx={float(row['base_lin_vel_x']):.4f} vy={float(row['base_lin_vel_y']):.4f} "
        f"yaw={float(row['base_yaw_rate']):.4f} "
        f"root_h={float(row['root_height']):.4f} "
        f"wheel_target={_format_vec([float(row['wheel_target_w_lb']), float(row['wheel_target_w_lf']), float(row['wheel_target_w_rf']), float(row['wheel_target_w_rb'])])} "
        f"wheel_vel={_format_vec([float(row['wheel_joint_vel_w_lb']), float(row['wheel_joint_vel_w_lf']), float(row['wheel_joint_vel_w_rf']), float(row['wheel_joint_vel_w_rb'])])} "
        f"semantic_wheel_vel={_format_vec([float(row['semantic_wheel_joint_vel_lr']), float(row['semantic_wheel_joint_vel_lf']), float(row['semantic_wheel_joint_vel_rf']), float(row['semantic_wheel_joint_vel_rr'])])} "
        f"contact={_format_vec([float(row['wheel_contact_force_w_lb']), float(row['wheel_contact_force_w_lf']), float(row['wheel_contact_force_w_rf']), float(row['wheel_contact_force_w_rb'])], precision=1)}",
        flush=True,
    )


def _run_case(
    env,
    *,
    run_root: Path,
    env_id: int,
    mode: str,
    pattern_name: str,
    raw_pattern: tuple[float, float, float, float],
    scale: float,
    settle_steps: int,
    case_steps: int,
    print_every: int,
    zero_actions: torch.Tensor,
    leg_action_slice: slice,
    wheel_action_slice: slice,
    wheel_joint_ids: torch.Tensor,
    wheel_body_ids: torch.Tensor,
    lift_root: bool,
    free_spin_lift: float,
) -> dict[str, float | int | str]:
    env.reset()
    robot = env.unwrapped.scene["robot"]
    if lift_root:
        _set_env_root_lift(robot, env_id=env_id, lift_amount=free_spin_lift)
        for _ in range(3):
            _safe_step(env, zero_actions)
            _set_env_root_lift(robot, env_id=env_id, lift_amount=0.0)
    for _ in range(settle_steps):
        _safe_step(env, zero_actions)

    if lift_root:
        target_free_spin_height = float(robot.data.root_pos_w[env_id, 2].item()) + float(free_spin_lift)
        _set_env_root_height(robot, env_id=env_id, target_height=target_free_spin_height)
    else:
        target_free_spin_height = None

    pattern_actions = torch.zeros_like(zero_actions)
    pattern_actions[:, leg_action_slice] = 0.0
    scaled_pattern = torch.tensor(raw_pattern, device=zero_actions.device, dtype=torch.float32) * float(scale)
    pattern_actions[:, wheel_action_slice] = scaled_pattern.unsqueeze(0)

    rows: list[dict[str, float | int | str]] = []
    terminated_step = -1
    truncated_step = -1
    for step_idx in range(case_steps):
        if lift_root and target_free_spin_height is not None:
            _set_env_root_height(robot, env_id=env_id, target_height=target_free_spin_height)
        _, _, terminated, truncated, _ = _safe_step(env, pattern_actions)
        if lift_root and target_free_spin_height is not None:
            _set_env_root_height(robot, env_id=env_id, target_height=target_free_spin_height)
        row = _collect_row(
            env,
            env_id,
            pattern_name,
            step_idx,
            wheel_joint_ids,
            wheel_body_ids,
            mode=mode,
            scale=scale,
        )
        rows.append(row)
        if step_idx % max(print_every, 1) == 0 or step_idx == case_steps - 1:
            _print_step_debug(row)
        if terminated_step < 0 and bool(terminated[env_id].item()):
            terminated_step = step_idx
        if truncated_step < 0 and bool(truncated[env_id].item()):
            truncated_step = step_idx

    if not rows:
        raise RuntimeError(f"{mode}:{pattern_name}:scale={scale} produced zero rows.")

    csv_name = f"{mode}_{pattern_name}_scale_{int(scale)}.csv"
    _write_csv(run_root / csv_name, rows)
    tail_rows = rows[max(len(rows) // 2, 0) :]
    mean_vx = sum(float(r["base_lin_vel_x"]) for r in tail_rows) / max(len(tail_rows), 1)
    mean_vy = sum(float(r["base_lin_vel_y"]) for r in tail_rows) / max(len(tail_rows), 1)
    mean_yaw = sum(float(r["base_yaw_rate"]) for r in tail_rows) / max(len(tail_rows), 1)
    mean_wheel_vel_abs = sum(
        0.25
        * (
            abs(float(r["wheel_joint_vel_w_lb"]))
            + abs(float(r["wheel_joint_vel_w_lf"]))
            + abs(float(r["wheel_joint_vel_w_rf"]))
            + abs(float(r["wheel_joint_vel_w_rb"]))
        )
        for r in tail_rows
    ) / max(len(tail_rows), 1)
    mean_contact = sum(
        0.25
        * (
            float(r["wheel_contact_force_w_lb"])
            + float(r["wheel_contact_force_w_lf"])
            + float(r["wheel_contact_force_w_rf"])
            + float(r["wheel_contact_force_w_rb"])
        )
        for r in tail_rows
    ) / max(len(tail_rows), 1)
    final_row = rows[-1]
    summary = {
        "mode": mode,
        "pattern": pattern_name,
        "scale": float(scale),
        "mean_base_lin_vel_x": mean_vx,
        "mean_base_lin_vel_y": mean_vy,
        "mean_base_yaw_rate": mean_yaw,
        "mean_wheel_joint_vel_abs": mean_wheel_vel_abs,
        "mean_contact_force": mean_contact,
        "final_root_height": float(final_row["root_height"]),
        "final_wheel_target_w_lb": float(final_row["wheel_target_w_lb"]),
        "final_wheel_target_w_lf": float(final_row["wheel_target_w_lf"]),
        "final_wheel_target_w_rf": float(final_row["wheel_target_w_rf"]),
        "final_wheel_target_w_rb": float(final_row["wheel_target_w_rb"]),
        "final_wheel_joint_vel_w_lb": float(final_row["wheel_joint_vel_w_lb"]),
        "final_wheel_joint_vel_w_lf": float(final_row["wheel_joint_vel_w_lf"]),
        "final_wheel_joint_vel_w_rf": float(final_row["wheel_joint_vel_w_rf"]),
        "final_wheel_joint_vel_w_rb": float(final_row["wheel_joint_vel_w_rb"]),
        "terminated_step": terminated_step,
        "truncated_step": truncated_step,
        "bad_orientation_triggered": int(any(int(r["bad_orientation"]) for r in rows)),
        "root_height_low_triggered": int(any(int(r["root_height_low"]) for r in rows)),
        "csv_path": str(run_root / csv_name),
    }
    print(
        "[RESULT] "
        f"mode={mode} pattern={pattern_name} scale={scale:.1f}: "
        f"vx={mean_vx:.4f}, vy={mean_vy:.4f}, yaw={mean_yaw:.4f}, "
        f"wheel_vel_abs={mean_wheel_vel_abs:.4f}, contact={mean_contact:.2f}, "
        f"root_h={float(final_row['root_height']):.4f}, "
        f"terminated_step={terminated_step}, truncated_step={truncated_step}",
        flush=True,
    )
    return summary


def main():
    run_root = Path(args_cli.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(args_cli.settle_s + args_cli.duration_s + 1.0))
    env = None

    summaries: list[dict[str, float | int | str]] = []
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
        robot = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
        wheel_joint_ids, _ = robot.find_joints(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
        wheel_body_ids, _ = contact_sensor.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
        wheel_info = _wheel_actuator_info(env)
        _print_wheel_actuator_info(wheel_info)
        step_dt = float(unwrapped.step_dt)
        settle_steps = max(int(round(float(args_cli.settle_s) / max(step_dt, 1.0e-6))), 1)
        case_steps = max(int(round(float(args_cli.duration_s) / max(step_dt, 1.0e-6))), 1)
        print(
            f"[INFO] step_dt={step_dt:.6f} settle_steps={settle_steps} case_steps={case_steps} "
            f"duration={float(args_cli.duration_s):.3f}s print_every={int(args_cli.print_every)} "
            f"action_space_shape={tuple(getattr(env.action_space, 'shape', ())) if hasattr(env.action_space, 'shape') else 'n/a'} "
            f"resolved_action_dim={action_dim}",
            flush=True,
        )

        for pattern_name, raw_pattern in PATTERNS:
            print(f"[PATTERN] mode=ground pattern={pattern_name} raw={raw_pattern} scale=1.0", flush=True)
            try:
                summaries.append(
                    _run_case(
                        env,
                        run_root=run_root,
                        env_id=args_cli.env_id,
                        mode="ground",
                        pattern_name=pattern_name,
                        raw_pattern=raw_pattern,
                        scale=1.0,
                        settle_steps=settle_steps,
                        case_steps=case_steps,
                        print_every=int(args_cli.print_every),
                        zero_actions=zero_actions,
                        leg_action_slice=leg_action_slice,
                        wheel_action_slice=wheel_action_slice,
                        wheel_joint_ids=wheel_joint_ids,
                        wheel_body_ids=wheel_body_ids,
                        lift_root=False,
                        free_spin_lift=float(args_cli.free_spin_lift),
                    )
                )
            except BaseException:
                print(f"[ERROR] ground pattern={pattern_name} failed", flush=True)
                traceback.print_exc()
                raise

        print("[INFO] running wheel target sweep on ground", flush=True)
        for scale in SCALES:
            for pattern_name, raw_pattern in SWEEP_PATTERNS:
                print(f"[PATTERN] mode=ground_sweep pattern={pattern_name} raw={raw_pattern} scale={scale}", flush=True)
                try:
                    summaries.append(
                        _run_case(
                            env,
                            run_root=run_root,
                            env_id=args_cli.env_id,
                            mode="ground_sweep",
                            pattern_name=pattern_name,
                            raw_pattern=raw_pattern,
                            scale=scale,
                            settle_steps=settle_steps,
                            case_steps=case_steps,
                            print_every=int(args_cli.print_every),
                            zero_actions=zero_actions,
                            leg_action_slice=leg_action_slice,
                            wheel_action_slice=wheel_action_slice,
                            wheel_joint_ids=wheel_joint_ids,
                            wheel_body_ids=wheel_body_ids,
                            lift_root=False,
                            free_spin_lift=float(args_cli.free_spin_lift),
                        )
                    )
                except BaseException:
                    print(f"[ERROR] ground_sweep pattern={pattern_name} scale={scale} failed", flush=True)
                    traceback.print_exc()
                    raise

        print("[INFO] running free-spin wheel target sweep", flush=True)
        for scale in SCALES:
            for pattern_name, raw_pattern in SWEEP_PATTERNS:
                print(f"[PATTERN] mode=free_spin pattern={pattern_name} raw={raw_pattern} scale={scale}", flush=True)
                try:
                    summaries.append(
                        _run_case(
                            env,
                            run_root=run_root,
                            env_id=args_cli.env_id,
                            mode="free_spin",
                            pattern_name=pattern_name,
                            raw_pattern=raw_pattern,
                            scale=scale,
                            settle_steps=settle_steps,
                            case_steps=case_steps,
                            print_every=int(args_cli.print_every),
                            zero_actions=zero_actions,
                            leg_action_slice=leg_action_slice,
                            wheel_action_slice=wheel_action_slice,
                            wheel_joint_ids=wheel_joint_ids,
                            wheel_body_ids=wheel_body_ids,
                            lift_root=True,
                            free_spin_lift=float(args_cli.free_spin_lift),
                        )
                    )
                except BaseException:
                    print(f"[ERROR] free_spin pattern={pattern_name} scale={scale} failed", flush=True)
                    traceback.print_exc()
                    raise

        summary_path = run_root / "summary.csv"
        _write_csv(summary_path, summaries)
        best_forward = max(summaries, key=lambda item: float(item["mean_base_lin_vel_x"]))
        print(
            "[RESULT] strongest forward pattern = "
            f"{best_forward['mode']}:{best_forward['pattern']}:scale={float(best_forward['scale']):.1f} "
            f"(mean_base_lin_vel_x={float(best_forward['mean_base_lin_vel_x']):.4f})",
            flush=True,
        )
        print(f"[DONE] wrote summary: {summary_path}", flush=True)
    except BaseException:
        print("[ERROR] debug_wheel_drive_open_loop.py failed", flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
