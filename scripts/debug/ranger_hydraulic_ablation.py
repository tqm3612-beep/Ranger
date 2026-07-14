#!/usr/bin/env python3
"""Run Ranger hydraulic/wheel action-source ablations and write per-step diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run Ranger hydraulic A/B/C/D ablation diagnostics.")
parser.add_argument("--task", type=str, required=True, help="Gym task id.")
parser.add_argument("--checkpoint", type=str, default=None, help="RSL-RL checkpoint for policy action sources.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--steps", type=int, default=600, help="Number of policy steps to run.")
parser.add_argument("--warmup_steps", type=int, default=120, help="Steps excluded from summary statistics.")
parser.add_argument("--seed", type=int, default=2, help="Environment and agent seed.")
parser.add_argument(
    "--hydraulic_source",
    type=str,
    choices=("policy", "zero", "fixed"),
    default="policy",
    help="Source for action dimensions 0:4.",
)
parser.add_argument(
    "--wheel_source",
    type=str,
    choices=("policy", "zero", "fixed"),
    default="policy",
    help="Source for action dimensions 4:8.",
)
parser.add_argument(
    "--fixed_hydraulic_action",
    type=str,
    default="0,0,0,0",
    help="Comma-separated LB,LF,RF,RB hydraulic action used when hydraulic_source=fixed.",
)
parser.add_argument(
    "--fixed_wheel_action",
    type=str,
    default="0,0,0,0",
    help="Comma-separated LB,LF,RF,RB wheel action used when wheel_source=fixed.",
)
parser.add_argument("--output_csv", type=str, required=True, help="Path for per-step CSV output.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import isaaclab_tasks  # noqa: F401
import Ranger.tasks  # noqa: F401
from Ranger.tasks.manager_based.ranger import mdp
from Ranger.tasks.manager_based.ranger.agents import RangerTerrainActorCritic

rsl_on_policy_runner.RangerTerrainActorCritic = RangerTerrainActorCritic


SUMMARY_KEYS = (
    "hydraulic_action_abs_mean",
    "hydraulic_action_diag_mode",
    "stroke_range",
    "stroke_diag_mode",
    "contact_force_lb",
    "contact_force_lf",
    "contact_force_rf",
    "contact_force_rb",
    "contact_diag_ratio",
    "zero_contact_ratio",
    "contact_force_imbalance",
    "base_xy_speed",
)


def _parse_four_floats(text: str, *, name: str) -> list[float]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if len(values) != 4:
        raise ValueError(f"{name} must contain exactly four comma-separated values, got: {text!r}")
    return [float(value) for value in values]


def _get_action_slice(unwrapped_env, term_name: str) -> slice:
    start = 0
    for active_name, dim in zip(unwrapped_env.action_manager.active_terms, unwrapped_env.action_manager.action_term_dim):
        if active_name == term_name:
            return slice(start, start + int(dim))
        start += int(dim)
    raise KeyError(f"Action term {term_name!r} not found. Active terms: {list(unwrapped_env.action_manager.active_terms)}")


def _to_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _safe_stats(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(finite),
        "std": pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def _make_policy(env, agent_cfg):
    if args_cli.checkpoint is None and (args_cli.hydraulic_source == "policy" or args_cli.wheel_source == "policy"):
        raise ValueError("--checkpoint is required when either action source is 'policy'.")
    if args_cli.checkpoint is None:
        return None
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(Path(args_cli.checkpoint).expanduser()))
    return runner.get_inference_policy(device=env.unwrapped.device)


def _collect_rows(unwrapped, step: int, summary_active: bool) -> list[dict[str, float | int]]:
    robot = unwrapped.scene["robot"]
    leg_term = unwrapped.action_manager.get_term("leg_hydraulic")
    wheel_term = unwrapped.action_manager.get_term("wheel_motor_csv")
    leg_ids = torch.as_tensor(getattr(leg_term, "_joint_ids"), device=unwrapped.device, dtype=torch.long)
    wheel_ids = torch.as_tensor(getattr(wheel_term, "_joint_ids"), device=unwrapped.device, dtype=torch.long)
    contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    contact_body_ids, _ = contact_sensor.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
    contact_body_ids = torch.as_tensor(contact_body_ids, device=unwrapped.device, dtype=torch.long)

    if bool(getattr(unwrapped, "_is_short_goal_task", False)):
        _, goal_distance, heading_error = mdp.short_goal_target_body(unwrapped)
    else:
        goal_distance = torch.zeros((unwrapped.num_envs,), dtype=torch.float32, device=unwrapped.device)
        heading_error = torch.zeros_like(goal_distance)

    root_lin_vel_b = robot.data.root_lin_vel_b
    base_xy_speed = torch.linalg.vector_norm(root_lin_vel_b[:, :2], dim=1)
    base_yaw_rate = robot.data.root_ang_vel_b[:, 2]

    hydraulic_action = leg_term.raw_actions
    stroke_desired = leg_term.stroke_desired
    stroke_actual = leg_term.stroke_actual
    joint_position_target = robot.data.joint_pos_target[:, leg_ids]
    joint_position = robot.data.joint_pos[:, leg_ids]
    wheel_target = wheel_term.velocity_target
    wheel_joint_velocity = robot.data.joint_vel[:, wheel_ids]
    contact_force = torch.max(
        torch.norm(contact_sensor.data.net_forces_w_history[:, :, contact_body_ids, :], dim=-1),
        dim=1,
    )[0]
    contact_total = contact_force.sum(dim=1)
    contact_diag_force = contact_force[:, 0] + contact_force[:, 2] - contact_force[:, 1] - contact_force[:, 3]
    contact_diag_ratio = contact_diag_force / torch.clamp(contact_total, min=1.0e-6)
    contact_force_imbalance = (contact_force.max(dim=1).values - contact_force.min(dim=1).values) / torch.clamp(
        contact_total, min=1.0e-6
    )
    zero_contact_ratio = (contact_force < 1.0).to(torch.float32).mean(dim=1)
    hydraulic_action_diag_mode = 0.5 * (
        hydraulic_action[:, 0] + hydraulic_action[:, 2] - hydraulic_action[:, 1] - hydraulic_action[:, 3]
    )
    stroke_diag_mode = 0.5 * (
        stroke_actual[:, 0] + stroke_actual[:, 2] - stroke_actual[:, 1] - stroke_actual[:, 3]
    )
    stroke_range = stroke_actual.max(dim=1).values - stroke_actual.min(dim=1).values
    hydraulic_action_abs_mean = torch.mean(torch.abs(hydraulic_action), dim=1)

    rows: list[dict[str, float | int]] = []
    for env_id in range(unwrapped.num_envs):
        row: dict[str, float | int] = {
            "step": int(step),
            "env_id": int(env_id),
            "summary_active": int(summary_active),
            "goal_distance": float(goal_distance[env_id].item()),
            "heading_error": float(heading_error[env_id].item()),
            "base_xy_speed": float(base_xy_speed[env_id].item()),
            "base_yaw_rate": float(base_yaw_rate[env_id].item()),
            "zero_contact_ratio": float(zero_contact_ratio[env_id].item()),
            "contact_force_imbalance": float(contact_force_imbalance[env_id].item()),
            "hydraulic_action_diag_mode": float(hydraulic_action_diag_mode[env_id].item()),
            "stroke_diag_mode": float(stroke_diag_mode[env_id].item()),
            "stroke_range": float(stroke_range[env_id].item()),
            "contact_diag_force": float(contact_diag_force[env_id].item()),
            "contact_diag_ratio": float(contact_diag_ratio[env_id].item()),
            "hydraulic_action_abs_mean": float(hydraulic_action_abs_mean[env_id].item()),
        }
        for idx, name in enumerate(("lb", "lf", "rf", "rb")):
            row[f"hydraulic_action_{name}"] = float(hydraulic_action[env_id, idx].item())
            row[f"stroke_desired_{name}"] = float(stroke_desired[env_id, idx].item())
            row[f"stroke_actual_{name}"] = float(stroke_actual[env_id, idx].item())
            row[f"joint_position_target_{name}"] = float(joint_position_target[env_id, idx].item())
            row[f"joint_position_{name}"] = float(joint_position[env_id, idx].item())
            row[f"wheel_target_{name}"] = float(wheel_target[env_id, idx].item())
            row[f"wheel_joint_velocity_{name}"] = float(wheel_joint_velocity[env_id, idx].item())
            row[f"contact_force_{name}"] = float(contact_force[env_id, idx].item())
        rows.append(row)
    return rows


def _print_summary(rows: list[dict[str, float | int]]) -> None:
    active_rows = [row for row in rows if int(row["summary_active"]) == 1]
    print("[Ablation] summary rows:", len(active_rows), flush=True)
    print("[Ablation] key, mean, std, min, max", flush=True)
    for key in SUMMARY_KEYS:
        stats = _safe_stats([float(row[key]) for row in active_rows])
        print(
            f"[Ablation] {key}: mean={stats['mean']:+.6f} std={stats['std']:.6f} "
            f"min={stats['min']:+.6f} max={stats['max']:+.6f}",
            flush=True,
        )


def main() -> None:
    uses_policy = args_cli.hydraulic_source == "policy" or args_cli.wheel_source == "policy"
    fixed_hydraulic = torch.tensor(_parse_four_floats(args_cli.fixed_hydraulic_action, name="fixed_hydraulic_action"))
    fixed_wheel = torch.tensor(_parse_four_floats(args_cli.fixed_wheel_action, name="fixed_wheel_action"))
    output_csv = Path(args_cli.output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg = None
    if uses_policy:
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args_cli.seed
        agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    print("[Ablation] task:", args_cli.task, flush=True)
    print("[Ablation] checkpoint:", args_cli.checkpoint, flush=True)
    print("[Ablation] num_envs:", args_cli.num_envs, "steps:", args_cli.steps, "warmup_steps:", args_cli.warmup_steps, flush=True)
    print("[Ablation] hydraulic_source:", args_cli.hydraulic_source, "wheel_source:", args_cli.wheel_source, flush=True)
    print("[Ablation] fixed_hydraulic_action:", _to_list(fixed_hydraulic), flush=True)
    print("[Ablation] fixed_wheel_action:", _to_list(fixed_wheel), flush=True)

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    print("[Ablation] environment created", flush=True)
    if uses_policy:
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        raw_env = env.unwrapped
        policy = _make_policy(env, agent_cfg)
        print("[Ablation] policy runner loaded", flush=True)
    else:
        env = base_env
        raw_env = env.unwrapped
        policy = None
        env.reset()
        print("[Ablation] running without policy runner", flush=True)

    leg_slice = _get_action_slice(raw_env, "leg_hydraulic")
    wheel_slice = _get_action_slice(raw_env, "wheel_motor_csv")
    action_dim = sum(int(dim) for dim in raw_env.action_manager.action_term_dim)
    print(
        f"[Ablation] action slices: leg={leg_slice} wheel={wheel_slice} action_dim={action_dim}",
        flush=True,
    )
    fixed_hydraulic = fixed_hydraulic.to(device=raw_env.device, dtype=torch.float32)
    fixed_wheel = fixed_wheel.to(device=raw_env.device, dtype=torch.float32)

    rows: list[dict[str, float | int]] = []
    try:
        if uses_policy:
            obs_result = env.get_observations()
            obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result
        else:
            obs = None
        with torch.inference_mode():
            for step in range(args_cli.steps):
                if step == 0:
                    print("[Ablation] before first action build", flush=True)
                if policy is None:
                    actions = torch.zeros((raw_env.num_envs, action_dim), device=raw_env.device)
                else:
                    actions = policy(obs)
                actions = actions.clone()
                if args_cli.hydraulic_source == "zero":
                    actions[:, leg_slice] = 0.0
                elif args_cli.hydraulic_source == "fixed":
                    actions[:, leg_slice] = fixed_hydraulic.unsqueeze(0).repeat(raw_env.num_envs, 1)
                if args_cli.wheel_source == "zero":
                    actions[:, wheel_slice] = 0.0
                elif args_cli.wheel_source == "fixed":
                    actions[:, wheel_slice] = fixed_wheel.unsqueeze(0).repeat(raw_env.num_envs, 1)

                if step == 0:
                    print("[Ablation] before first env.step", flush=True)
                step_result = env.step(actions)
                if step == 0:
                    print("[Ablation] after first env.step", flush=True)
                if uses_policy:
                    obs = step_result[0]
                rows.extend(_collect_rows(raw_env, step=step, summary_active=step >= args_cli.warmup_steps))
                if step == 0:
                    print("[Ablation] after first row collect", flush=True)

        with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = list(rows[0].keys()) if rows else []
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print("[Ablation] wrote:", output_csv, flush=True)
        _print_summary(rows)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
