# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate posture-stability metrics for two Ranger locomotion tasks on matched terrain seeds."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare posture-stability metrics between two Ranger tasks.")
parser.add_argument(
    "--task_simple",
    type=str,
    default="Template-Ranger-SimpleTerrain-v0",
    help="Stage-3 task to evaluate.",
)
parser.add_argument(
    "--task_map",
    type=str,
    default="Template-Ranger-MapPosture-v0",
    help="Stage-4 task to evaluate.",
)
parser.add_argument("--checkpoint_simple", type=str, required=True, help="Checkpoint path for the stage-3 policy.")
parser.add_argument("--checkpoint_map", type=str, required=True, help="Checkpoint path for the stage-4 policy.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of vectorized environments to evaluate.")
parser.add_argument("--num_episodes", type=int, default=64, help="Minimum number of finished episodes to aggregate.")
parser.add_argument("--max_steps", type=int, default=10000, help="Safety limit on environment steps per evaluation.")
parser.add_argument("--seed", type=int, default=42, help="Shared evaluation seed for both tasks.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--output", type=str, default=None, help="Optional JSON file to store the comparison result.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import Ranger.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401


EVAL_METRICS = (
    "roll_rms_deg",
    "pitch_rms_deg",
    "roll_max_abs_deg",
    "pitch_max_abs_deg",
    "base_vertical_velocity_rms",
    "base_roll_pitch_ang_vel_rms",
    "average_forward_speed",
    "velocity_tracking_error_rms",
    "joint_limit_margin_penalty_mean",
    "joint_limit_margin_count_mean",
    "episode_length",
    "success_rate",
)


def _set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_agent_cfg(task_name: str):
    agent_cfg = load_cfg_from_registry(task_name.split(":")[-1], "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    return agent_cfg


def _load_runner(agent_cfg, checkpoint_path: str, env) -> OnPolicyRunner:
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    return runner


def _evaluate_task(task_name: str, checkpoint_path: str) -> dict[str, float]:
    _set_eval_seed(args_cli.seed)
    resolved_checkpoint = retrieve_file_path(checkpoint_path)
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    agent_cfg = _load_agent_cfg(task_name)
    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = _load_runner(agent_cfg, resolved_checkpoint, env)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    weighted_metric_sums = {name: 0.0 for name in EVAL_METRICS}
    completed_episodes = 0
    sim_steps = 0

    try:
        obs, _ = env.get_observations()
        while simulation_app.is_running() and completed_episodes < args_cli.num_episodes and sim_steps < args_cli.max_steps:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, extras = env.step(actions)

            sim_steps += 1
            log_data = extras.get("log", {})
            episode_batch = int(round(float(log_data.get("Metrics/eval/num_episodes", 0.0))))
            if episode_batch <= 0:
                continue

            completed_episodes += episode_batch
            for metric_name in EVAL_METRICS:
                log_key = f"Metrics/eval/{metric_name}"
                if log_key in log_data:
                    weighted_metric_sums[metric_name] += float(log_data[log_key]) * episode_batch

        if completed_episodes == 0:
            raise RuntimeError(f"No finished episodes were collected for task '{task_name}' within {args_cli.max_steps} steps.")

        results = {
            metric_name: weighted_metric_sums[metric_name] / completed_episodes
            for metric_name in EVAL_METRICS
        }
        results["completed_episodes"] = float(completed_episodes)
        results["sim_steps"] = float(sim_steps)
        results["target_speed"] = float(env.unwrapped.cfg.rewards.velocity_tracking.params["target_speed"])
        return results
    finally:
        env.close()


def _format_metric_value(metric_name: str, value: float) -> str:
    if metric_name in {"completed_episodes", "sim_steps"}:
        return f"{int(round(value))}"
    if metric_name == "success_rate":
        return f"{value * 100.0:.2f}%"
    return f"{value:.4f}"


def _print_comparison(simple_metrics: dict[str, float], map_metrics: dict[str, float]) -> None:
    print("\n=== Ranger Posture Stability Comparison ===")
    print(f"seed={args_cli.seed} num_envs={args_cli.num_envs} target_episodes>={args_cli.num_episodes}")
    print(f"simple_task={args_cli.task_simple}")
    print(f"map_task={args_cli.task_map}")
    print("")
    header = f"{'metric':<34} {'simple':>14} {'map_posture':>14} {'delta(map-simple)':>18}"
    print(header)
    print("-" * len(header))
    for metric_name in EVAL_METRICS:
        simple_value = simple_metrics[metric_name]
        map_value = map_metrics[metric_name]
        delta_value = map_value - simple_value
        delta_str = f"{delta_value:+.4f}"
        if metric_name == "success_rate":
            delta_str = f"{delta_value * 100.0:+.2f}%"
        print(
            f"{metric_name:<34} "
            f"{_format_metric_value(metric_name, simple_value):>14} "
            f"{_format_metric_value(metric_name, map_value):>14} "
            f"{delta_str:>18}"
        )
    print("")
    print(
        f"completed_episodes: simple={int(round(simple_metrics['completed_episodes']))}, "
        f"map_posture={int(round(map_metrics['completed_episodes']))}"
    )
    print(
        f"sim_steps: simple={int(round(simple_metrics['sim_steps']))}, "
        f"map_posture={int(round(map_metrics['sim_steps']))}"
    )


def main() -> None:
    simple_metrics = _evaluate_task(args_cli.task_simple, args_cli.checkpoint_simple)
    map_metrics = _evaluate_task(args_cli.task_map, args_cli.checkpoint_map)
    _print_comparison(simple_metrics, map_metrics)

    if args_cli.output is not None:
        output_path = Path(args_cli.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "num_episodes": args_cli.num_episodes,
            "task_simple": args_cli.task_simple,
            "task_map": args_cli.task_map,
            "checkpoint_simple": str(Path(args_cli.checkpoint_simple).expanduser()),
            "checkpoint_map": str(Path(args_cli.checkpoint_map).expanduser()),
            "simple_metrics": simple_metrics,
            "map_metrics": map_metrics,
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"\nSaved comparison JSON to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
