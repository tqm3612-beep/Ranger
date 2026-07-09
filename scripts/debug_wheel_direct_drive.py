#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Ranger wheel direct-drive modes under free-spin conditions.")
parser.add_argument("--task", type=str, default="Template-Ranger-ShortGoalFlat-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_id", type=int, default=0)
parser.add_argument("--duration", "--duration_s", dest="duration_s", type=float, default=1.0)
parser.add_argument("--settle_s", type=float, default=1.0)
parser.add_argument("--free_spin_lift", type=float, default=0.8)
parser.add_argument("--print_every", type=int, default=20)
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/debug_wheel_direct_drive",
    help="Directory for per-case CSV output.",
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


CURRENT_EFFORT_SCALES = (20.0, 50.0, 100.0, 200.0, 400.0)
DIRECT_EFFORT_VALUES = (20.0, 50.0, 100.0, 130.0, 300.0)
DIRECT_VELOCITY_VALUES = (5.0, 10.0, 20.0)
DAMPING_VALUES = (1000000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.0)
SEMANTIC_PATTERNS = (
    ("semantic_forward", (-1.0, -1.0, 1.0, 1.0)),
    ("semantic_backward", (1.0, 1.0, -1.0, -1.0)),
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


def _format_vec(values: list[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _format_scalar(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{float(value):.4f}"


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


def _env_cfg_with_optional_damping(damping_override: float | None):
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(args_cli.settle_s + args_cli.duration_s + 1.0))
    if damping_override is not None:
        env_cfg.scene.robot.actuators["wheel_joints"].damping = float(damping_override)
    return env_cfg


def _create_env(damping_override: float | None):
    env_cfg = _env_cfg_with_optional_damping(damping_override)
    return gym.make(args_cli.task, cfg=env_cfg)


def _set_env_root_height(robot, env_id: int, target_height: float) -> None:
    env_ids = torch.tensor([env_id], device=robot.device, dtype=torch.long)
    root_pose = robot.data.root_pose_w[env_id : env_id + 1].clone()
    root_velocity = robot.data.root_vel_w[env_id : env_id + 1].clone()
    root_pose[:, 2] = float(target_height)
    root_velocity.zero_()
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def _pin_free_spin_height(robot, env_id: int, target_height: float) -> None:
    _set_env_root_height(robot, env_id=env_id, target_height=target_height)


def _peak_contact_force_and_bool(contact_sensor, env_id: int, body_ids: torch.Tensor, threshold: float = 1.0):
    force_history = contact_sensor.data.net_forces_w_history[env_id, :, body_ids, :]
    force_norm = torch.norm(force_history, dim=-1)
    peak_force_norm = torch.max(force_norm, dim=0).values
    contact_bool = peak_force_norm > float(threshold)
    return peak_force_norm, contact_bool


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
        row = {"joint_name": joint_name, "joint_id": int(joint_id)}
        if stage is not None:
            row.update(_usd_joint_debug_info(stage, joint_name))
        joint_rows.append(row)

    return {
        "wheel_joint_names": list(wheel_joint_names),
        "wheel_joint_ids": [int(idx) for idx in wheel_joint_ids],
        "wheel_actuator_cfg_type": type(wheel_asset_actuator_cfg).__name__ if wheel_asset_actuator_cfg is not None else "unknown",
        "wheel_action_term_type": type(unwrapped.action_manager.get_term("wheel_motor_csv")).__name__,
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


def _print_wheel_actuator_info(info: dict[str, object], *, mode_name: str, damping_override: float | None) -> None:
    print(f"[INFO] wheel drive audit mode_env={mode_name} damping_override={damping_override}", flush=True)
    print(f"  wheel_joint_names={info['wheel_joint_names']}", flush=True)
    print(f"  wheel_joint_ids={info['wheel_joint_ids']}", flush=True)
    print(f"  wheel_actuator_type={info['wheel_actuator_cfg_type']}", flush=True)
    print(f"  wheel_action_term_type={info['wheel_action_term_type']}", flush=True)
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
            f"limits=({_format_scalar(row['lower_limit'])}, {_format_scalar(row['upper_limit'])}) "
            f"drive_stiffness={row['drive_stiffness']} drive_damping={row['drive_damping']} "
            f"drive_max_force={row['drive_max_force']} joint_friction={row['joint_friction']} armature={row['armature']}",
            flush=True,
        )


def _collect_row(
    env,
    env_id: int,
    *,
    mode: str,
    pattern_name: str,
    command_kind: str,
    command_value: float,
    wheel_joint_ids: torch.Tensor,
    wheel_body_ids: torch.Tensor,
    command_tensor: torch.Tensor,
) -> dict[str, float | int | str]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    peak_contact_force, contact_bool = _peak_contact_force_and_bool(contact_sensor, env_id, wheel_body_ids)
    wheel_joint_vel = robot.data.joint_vel[env_id, wheel_joint_ids]
    semantic_wheel_joint_vel = wheel_joint_vel * mdp.wheel_semantic_sign_lr_lf_rf_rr(
        device=wheel_joint_vel.device, dtype=wheel_joint_vel.dtype
    ).squeeze(0)

    return {
        "mode": mode,
        "pattern": pattern_name,
        "command_kind": command_kind,
        "command_value": float(command_value),
        "root_height": float(robot.data.root_pos_w[env_id, 2].item()),
        "wheel_target_w_lb": float(command_tensor[0].item()),
        "wheel_target_w_lf": float(command_tensor[1].item()),
        "wheel_target_w_rf": float(command_tensor[2].item()),
        "wheel_target_w_rb": float(command_tensor[3].item()),
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
    }


def _print_step(row: dict[str, float | int | str], step_idx: int, *, target_mode: str) -> None:
    print(
        "[STEP] "
        f"mode={row['mode']} target_mode={target_mode} pattern={row['pattern']} "
        f"value={float(row['command_value']):.3f} step={step_idx} "
        f"root_h={float(row['root_height']):.4f} "
        f"wheel_target={_format_vec([float(row['wheel_target_w_lb']), float(row['wheel_target_w_lf']), float(row['wheel_target_w_rf']), float(row['wheel_target_w_rb'])])} "
        f"wheel_vel={_format_vec([float(row['wheel_joint_vel_w_lb']), float(row['wheel_joint_vel_w_lf']), float(row['wheel_joint_vel_w_rf']), float(row['wheel_joint_vel_w_rb'])])} "
        f"semantic_wheel_vel={_format_vec([float(row['semantic_wheel_joint_vel_lr']), float(row['semantic_wheel_joint_vel_lf']), float(row['semantic_wheel_joint_vel_rf']), float(row['semantic_wheel_joint_vel_rr'])])} "
        f"contact={_format_vec([float(row['wheel_contact_force_w_lb']), float(row['wheel_contact_force_w_lf']), float(row['wheel_contact_force_w_rf']), float(row['wheel_contact_force_w_rb'])], precision=1)}",
        flush=True,
    )


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_zero_actions(env) -> torch.Tensor:
    unwrapped = env.unwrapped
    action_dim = _resolve_action_dim(env)
    return torch.zeros((unwrapped.num_envs, action_dim), device=unwrapped.device, dtype=torch.float32)


def _semantic_command_tensor(pattern: tuple[float, float, float, float], magnitude: float, device, dtype) -> torch.Tensor:
    return torch.tensor(pattern, device=device, dtype=dtype) * float(magnitude)


def _apply_current_effort_mode(env, actions: torch.Tensor):
    _safe_step(env, actions)


def _apply_direct_effort_mode(robot, joint_ids: torch.Tensor, env_id: int, command_tensor: torch.Tensor, unwrapped):
    env_ids = torch.tensor([env_id], device=robot.device, dtype=torch.long)
    robot.set_joint_effort_target(command_tensor.unsqueeze(0), joint_ids=joint_ids, env_ids=env_ids)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.step(render=False)
    unwrapped.scene.update(dt=unwrapped.physics_dt)


def _apply_direct_velocity_mode(robot, joint_ids: torch.Tensor, env_id: int, command_tensor: torch.Tensor, unwrapped):
    env_ids = torch.tensor([env_id], device=robot.device, dtype=torch.long)
    if not hasattr(robot, "set_joint_velocity_target"):
        raise RuntimeError("Articulation does not expose set_joint_velocity_target.")
    robot.set_joint_velocity_target(command_tensor.unsqueeze(0), joint_ids=joint_ids, env_ids=env_ids)
    robot.set_joint_effort_target(torch.zeros_like(command_tensor).unsqueeze(0), joint_ids=joint_ids, env_ids=env_ids)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.step(render=False)
    unwrapped.scene.update(dt=unwrapped.physics_dt)


def _run_single_case(
    env,
    *,
    run_root: Path,
    mode: str,
    target_mode: str,
    pattern_name: str,
    pattern: tuple[float, float, float, float],
    command_kind: str,
    command_value: float,
    damping_override: float | None,
) -> dict[str, float | int | str]:
    env.reset()
    unwrapped = env.unwrapped
    env_id = args_cli.env_id
    device = unwrapped.device
    robot = unwrapped.scene["robot"]
    zero_actions = _build_zero_actions(env)
    leg_action_slice = _get_action_slice(unwrapped, "leg_hydraulic")
    wheel_action_slice = _get_action_slice(unwrapped, "wheel_motor_csv")
    wheel_joint_ids, _ = robot.find_joints(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
    wheel_body_ids, _ = unwrapped.scene.sensors["wheel_contact_forces"].find_bodies(
        ["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True
    )
    wheel_joint_ids = torch.as_tensor(wheel_joint_ids, device=device, dtype=torch.long)
    wheel_body_ids = torch.as_tensor(wheel_body_ids, device=device, dtype=torch.long)
    step_dt = float(unwrapped.step_dt)
    settle_steps = max(int(round(float(args_cli.settle_s) / max(step_dt, 1.0e-6))), 1)
    case_steps = max(int(round(float(args_cli.duration_s) / max(step_dt, 1.0e-6))), 1)

    for _ in range(settle_steps):
        _safe_step(env, zero_actions)

    target_free_spin_height = float(robot.data.root_pos_w[env_id, 2].item()) + float(args_cli.free_spin_lift)
    command_tensor = _semantic_command_tensor(pattern, command_value, device=device, dtype=torch.float32)

    rows: list[dict[str, float | int | str]] = []
    for step_idx in range(case_steps):
        _pin_free_spin_height(robot, env_id=env_id, target_height=target_free_spin_height)
        if mode == "current_effort_mode":
            actions = torch.zeros_like(zero_actions)
            actions[:, leg_action_slice] = 0.0
            actions[:, wheel_action_slice] = command_tensor.unsqueeze(0)
            _apply_current_effort_mode(env, actions)
        elif mode == "direct_effort_mode" or mode == "wheel_damping_override_test":
            _apply_direct_effort_mode(robot, wheel_joint_ids, env_id, command_tensor, unwrapped)
        elif mode == "direct_velocity_mode":
            _apply_direct_velocity_mode(robot, wheel_joint_ids, env_id, command_tensor, unwrapped)
        else:
            raise RuntimeError(f"Unsupported mode: {mode}")
        _pin_free_spin_height(robot, env_id=env_id, target_height=target_free_spin_height)
        row = _collect_row(
            env,
            env_id,
            mode=mode,
            pattern_name=pattern_name,
            command_kind=command_kind,
            command_value=command_value,
            wheel_joint_ids=wheel_joint_ids,
            wheel_body_ids=wheel_body_ids,
            command_tensor=command_tensor,
        )
        row["target_mode"] = target_mode
        row["damping_override"] = "" if damping_override is None else float(damping_override)
        row["step"] = step_idx
        rows.append(row)
        if step_idx % max(int(args_cli.print_every), 1) == 0 or step_idx == case_steps - 1:
            _print_step(row, step_idx, target_mode=target_mode)

    command_value_tag = str(command_value).replace(".", "p")
    damping_tag = "" if damping_override is None else f"_damp_{str(damping_override).replace('.', 'p')}"
    csv_name = f"{mode}_{target_mode}_{pattern_name}_{command_kind}_{command_value_tag}{damping_tag}.csv"
    _write_csv(run_root / csv_name, rows)
    tail_rows = rows[max(len(rows) // 2, 0) :]
    mean_wheel_joint_vel_abs = sum(
        0.25
        * (
            abs(float(r["wheel_joint_vel_w_lb"]))
            + abs(float(r["wheel_joint_vel_w_lf"]))
            + abs(float(r["wheel_joint_vel_w_rf"]))
            + abs(float(r["wheel_joint_vel_w_rb"]))
        )
        for r in tail_rows
    ) / max(len(tail_rows), 1)
    mean_contact_force = sum(
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
        "target_mode": target_mode,
        "pattern": pattern_name,
        "command_kind": command_kind,
        "command_value": float(command_value),
        "damping_override": "" if damping_override is None else float(damping_override),
        "mean_wheel_joint_vel_abs": mean_wheel_joint_vel_abs,
        "mean_contact_force": mean_contact_force,
        "final_root_height": float(final_row["root_height"]),
        "final_wheel_joint_vel_w_lb": float(final_row["wheel_joint_vel_w_lb"]),
        "final_wheel_joint_vel_w_lf": float(final_row["wheel_joint_vel_w_lf"]),
        "final_wheel_joint_vel_w_rf": float(final_row["wheel_joint_vel_w_rf"]),
        "final_wheel_joint_vel_w_rb": float(final_row["wheel_joint_vel_w_rb"]),
        "csv_path": str(run_root / csv_name),
    }
    print(
        "[RESULT] "
        f"mode={mode} target_mode={target_mode} pattern={pattern_name} "
        f"{command_kind}={command_value:.3f} damping_override={damping_override} "
        f"wheel_joint_vel_abs={mean_wheel_joint_vel_abs:.6f} contact={mean_contact_force:.3f} "
        f"root_h={float(final_row['root_height']):.4f}",
        flush=True,
    )
    return summary


def _run_env_suite(run_root: Path, mode_name: str, damping_override: float | None) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []
    env = None
    try:
        env = _create_env(damping_override=damping_override)
        env.reset()
        if args_cli.env_id < 0 or args_cli.env_id >= env.unwrapped.num_envs:
            raise ValueError(f"env_id must be in [0, {env.unwrapped.num_envs - 1}], got {args_cli.env_id}.")
        _print_wheel_actuator_info(_wheel_actuator_info(env), mode_name=mode_name, damping_override=damping_override)

        if mode_name == "default_suite":
            for pattern_name, pattern in SEMANTIC_PATTERNS:
                for scale in CURRENT_EFFORT_SCALES:
                    summaries.append(
                        _run_single_case(
                            env,
                            run_root=run_root,
                            mode="current_effort_mode",
                            target_mode="set_joint_effort_target_via_WheelMotorCSVAction",
                            pattern_name=pattern_name,
                            pattern=pattern,
                            command_kind="raw_action_scale",
                            command_value=scale,
                            damping_override=damping_override,
                        )
                    )
            for pattern_name, pattern in SEMANTIC_PATTERNS:
                for effort in DIRECT_EFFORT_VALUES:
                    summaries.append(
                        _run_single_case(
                            env,
                            run_root=run_root,
                            mode="direct_effort_mode",
                            target_mode="set_joint_effort_target_direct",
                            pattern_name=pattern_name,
                            pattern=pattern,
                            command_kind="direct_effort",
                            command_value=effort,
                            damping_override=damping_override,
                        )
                    )
            for pattern_name, pattern in SEMANTIC_PATTERNS:
                for velocity in DIRECT_VELOCITY_VALUES:
                    summaries.append(
                        _run_single_case(
                            env,
                            run_root=run_root,
                            mode="direct_velocity_mode",
                            target_mode="set_joint_velocity_target_direct",
                            pattern_name=pattern_name,
                            pattern=pattern,
                            command_kind="direct_velocity",
                            command_value=velocity,
                            damping_override=damping_override,
                        )
                    )
        elif mode_name == "damping_override_suite":
            for effort in DIRECT_EFFORT_VALUES:
                summaries.append(
                    _run_single_case(
                        env,
                        run_root=run_root,
                        mode="wheel_damping_override_test",
                        target_mode="set_joint_effort_target_direct",
                        pattern_name="semantic_forward",
                        pattern=SEMANTIC_PATTERNS[0][1],
                        command_kind="direct_effort",
                        command_value=effort,
                        damping_override=damping_override,
                    )
                )
        else:
            raise RuntimeError(f"Unsupported mode_name: {mode_name}")
    finally:
        if env is not None:
            env.close()
    return summaries


def main():
    run_root = Path(args_cli.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, float | int | str]] = []
    try:
        summaries.extend(_run_env_suite(run_root, mode_name="default_suite", damping_override=None))
        for damping in DAMPING_VALUES:
            summaries.extend(_run_env_suite(run_root, mode_name="damping_override_suite", damping_override=damping))
        summary_path = run_root / "summary.csv"
        _write_csv(summary_path, summaries)
        print(f"[DONE] wrote summary: {summary_path}", flush=True)
    except BaseException:
        print("[ERROR] debug_wheel_direct_drive.py failed", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
