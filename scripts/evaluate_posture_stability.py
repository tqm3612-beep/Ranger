# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate posture-stability metrics for two Ranger locomotion tasks on matched terrain seeds."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate or compare Ranger posture-stability metrics.")
parser.add_argument("--task", type=str, default=None, help="Single-task evaluation task name.")
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
parser.add_argument("--checkpoint", type=str, default=None, help="Single-task checkpoint path.")
parser.add_argument("--checkpoint_simple", type=str, default=None, help="Checkpoint path for the stage-3 policy.")
parser.add_argument("--checkpoint_map", type=str, default=None, help="Checkpoint path for the stage-4 policy.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of vectorized environments to evaluate.")
parser.add_argument("--num_episodes", type=int, default=64, help="Minimum number of finished episodes to aggregate.")
parser.add_argument("--max_steps", type=int, default=10000, help="Safety limit on environment steps per evaluation.")
parser.add_argument("--seed", type=int, default=42, help="Shared evaluation seed for both tasks.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--output", type=str, default=None, help="Optional JSON file to store the comparison result.")
parser.add_argument("--simple_json", type=str, default=None, help="Single-task JSON result for SimpleTerrain.")
parser.add_argument("--map_json", type=str, default=None, help="Single-task JSON result for MapPosture.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

offline_compare = args_cli.simple_json is not None or args_cli.map_json is not None
single_task_eval = args_cli.task is not None or args_cli.checkpoint is not None

if offline_compare and (args_cli.simple_json is None or args_cli.map_json is None):
    parser.error("--simple_json and --map_json must be provided together.")
if offline_compare and single_task_eval:
    parser.error("Offline comparison mode cannot be combined with --task/--checkpoint.")
if single_task_eval and (args_cli.task is None or args_cli.checkpoint is None):
    parser.error("--task and --checkpoint must be provided together.")
if not offline_compare and not single_task_eval and (args_cli.checkpoint_simple is None or args_cli.checkpoint_map is None):
    parser.error("Provide either --task/--checkpoint, --simple_json/--map_json, or --checkpoint_simple/--checkpoint_map.")

if offline_compare:
    simulation_app = None
else:
    # Posture metric evaluation does not need an interactive viewport. Force
    # headless startup to avoid RTX/SceneDB/GUI plugin crashes on display-less
    # or fragile rendering setups.
    if hasattr(args_cli, "headless"):
        args_cli.headless = True
    if hasattr(args_cli, "enable_cameras"):
        args_cli.enable_cameras = False
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
    print(f"[Eval] Loading checkpoint: {checkpoint_path}", flush=True)
    runner.load(checkpoint_path)
    print("[Eval] Checkpoint loaded.", flush=True)
    return runner


def _reset_env(env):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out
    if isinstance(obs, dict):
        obs = obs["policy"]
    if obs is None:
        obs, _ = env.get_observations()
    return obs


def _step_env(env, actions):
    step_out = env.step(actions)
    if len(step_out) == 4:
        obs, reward, dones, extras = step_out
    elif len(step_out) == 5:
        obs, reward, terminated, truncated, extras = step_out
        dones = terminated | truncated
    else:
        raise RuntimeError(f"Unexpected env.step return length: {len(step_out)}")
    if isinstance(obs, dict):
        obs = obs["policy"]
    return obs, reward, dones, extras


def _evaluate_task(task_name: str, checkpoint_path: str) -> dict[str, float]:
    _set_eval_seed(args_cli.seed)
    resolved_checkpoint = retrieve_file_path(checkpoint_path)
    print(f"\n[Eval] Preparing task: {task_name}", flush=True)
    env = None
    runner = None
    policy = None
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
    print(f"[Eval] Starting rollout: {task_name}", flush=True)

    weighted_metric_sums = {name: 0.0 for name in EVAL_METRICS}
    completed_episodes = 0
    sim_steps = 0
    progress_interval = max(1, min(args_cli.num_envs, args_cli.num_episodes))
    next_progress = progress_interval

    try:
        obs = _reset_env(env)
        while completed_episodes < args_cli.num_episodes and sim_steps < args_cli.max_steps:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, extras = _step_env(env, actions)

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

            if completed_episodes >= next_progress or completed_episodes >= args_cli.num_episodes:
                print(
                    f"[Eval] {task_name}: collected {min(completed_episodes, args_cli.num_episodes)}/"
                    f"{args_cli.num_episodes} episodes",
                    flush=True,
                )
                while next_progress <= completed_episodes:
                    next_progress += progress_interval

        if completed_episodes == 0:
            raise RuntimeError(f"No finished episodes were collected for task '{task_name}' within {args_cli.max_steps} steps.")

        results = {
            metric_name: weighted_metric_sums[metric_name] / completed_episodes
            for metric_name in EVAL_METRICS
        }
        results["completed_episodes"] = float(completed_episodes)
        results["sim_steps"] = float(sim_steps)
        results["target_speed"] = float(env.unwrapped.cfg.rewards.velocity_tracking.params["target_speed"])
        print(f"[Eval] Finished {task_name}: {completed_episodes} episodes in {sim_steps} steps.", flush=True)
        return results
    finally:
        if env is not None:
            env.close()
        del policy
        del runner
        del env
        gc.collect()
        if "torch" in globals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_comparison_output(simple_metrics: dict[str, float] | None, map_metrics: dict[str, float] | None) -> None:
    if args_cli.output is None:
        return

    checkpoint_simple = str(Path(args_cli.checkpoint_simple).expanduser()) if args_cli.checkpoint_simple else None
    checkpoint_map = str(Path(args_cli.checkpoint_map).expanduser()) if args_cli.checkpoint_map else None
    output_path = Path(args_cli.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "num_episodes": args_cli.num_episodes,
        "task_simple": args_cli.task_simple,
        "task_map": args_cli.task_map,
        "checkpoint_simple": checkpoint_simple,
        "checkpoint_map": checkpoint_map,
        "simple_json": str(Path(args_cli.simple_json).expanduser()) if args_cli.simple_json else None,
        "map_json": str(Path(args_cli.map_json).expanduser()) if args_cli.map_json else None,
        "simple_metrics": simple_metrics,
        "map_metrics": map_metrics,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"\nSaved comparison JSON to: {output_path}", flush=True)


def _write_single_output(task_name: str, checkpoint_path: str, metrics: dict[str, float]) -> None:
    if args_cli.output is None:
        return

    output_path = Path(args_cli.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "num_episodes": args_cli.num_episodes,
        "task": task_name,
        "checkpoint": str(Path(checkpoint_path).expanduser()),
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"\nSaved single-task JSON to: {output_path}", flush=True)


def _read_json(path: str) -> dict:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _extract_metrics(payload: dict, label: str) -> dict[str, float]:
    if "metrics" in payload and payload["metrics"] is not None:
        return payload["metrics"]

    key = f"{label}_metrics"
    if key in payload and payload[key] is not None:
        return payload[key]

    raise KeyError(f"Could not find metrics in {label} JSON. Expected 'metrics' or '{key}'.")


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
    if offline_compare:
        simple_payload = _read_json(args_cli.simple_json)
        map_payload = _read_json(args_cli.map_json)
        simple_metrics = _extract_metrics(simple_payload, "simple")
        map_metrics = _extract_metrics(map_payload, "map")
        _print_comparison(simple_metrics, map_metrics)
        _write_comparison_output(simple_metrics, map_metrics)
        return

    if single_task_eval:
        metrics = _evaluate_task(args_cli.task, args_cli.checkpoint)
        _write_single_output(args_cli.task, args_cli.checkpoint, metrics)
        return

    simple_metrics = _evaluate_task(args_cli.task_simple, args_cli.checkpoint_simple)
    _write_comparison_output(simple_metrics, None)
    map_metrics = _evaluate_task(args_cli.task_map, args_cli.checkpoint_map)
    _print_comparison(simple_metrics, map_metrics)
    _write_comparison_output(simple_metrics, map_metrics)


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
