#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug Ranger policy observations and validate their ranges."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Ranger policy observation health and shape.")
parser.add_argument("--task", type=str, default="Template-Ranger-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--steps", type=int, default=2, help="Number of zero-action steps before inspection.")
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
from isaaclab_tasks.utils import parse_env_cfg

import Ranger.tasks  # noqa: F401
from Ranger.assets.ranger.ranger_cfg import RANGER_URDF_PATH, RANGER_USD_PATH
from Ranger.tasks.manager_based.ranger import mdp
from Ranger.tasks.manager_based.ranger.ranger_env_cfg import (
    GOAL_STATE_PARAMS,
    LOCAL_NAVIGATION_MAP_PARAMS,
    _expected_policy_obs_dim,
)


def _require_all_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN/Inf values.")


def _require_range(name: str, tensor: torch.Tensor, lower: float, upper: float, atol: float = 1.0e-5) -> None:
    _require_all_finite(name, tensor)
    min_val = float(tensor.min().item())
    max_val = float(tensor.max().item())
    if min_val < lower - atol or max_val > upper + atol:
        raise RuntimeError(f"{name} range out of bounds: min={min_val:.6f}, max={max_val:.6f}, expected [{lower}, {upper}].")


def _require_binary(name: str, tensor: torch.Tensor, atol: float = 1.0e-6) -> None:
    _require_all_finite(name, tensor)
    is_binary = torch.logical_or(torch.isclose(tensor, torch.zeros_like(tensor), atol=atol), torch.isclose(tensor, torch.ones_like(tensor), atol=atol))
    if not is_binary.all():
        unique_vals = torch.unique(tensor).detach().cpu().tolist()
        raise RuntimeError(f"{name} is not binary 0/1. Unique sample values: {unique_vals[:10]}")


def _print_stats(name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach()
    print(
        f"[CHECK] {name:<20} shape={tuple(tensor.shape)} "
        f"min={float(tensor.min().item()): .6f} "
        f"max={float(tensor.max().item()): .6f}"
    )


def main() -> None:
    """Create the environment and validate the policy observation interface."""
    if not RANGER_USD_PATH.exists():
        raise FileNotFoundError(
            f"Ranger USD not found: {RANGER_USD_PATH}\n"
            f"Place the URDF at {RANGER_URDF_PATH} and convert it to this USD path first."
        )

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        obs_dict, _ = env.reset()
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                obs_dict, _, _, _, _ = env.step(actions)

        policy_obs = obs_dict["policy"]
        obs_manager = env.unwrapped.observation_manager
        expected_shape = _expected_policy_obs_dim()

        if tuple(obs_manager.group_obs_dim["policy"]) != (expected_shape,):
            raise RuntimeError(
                f"ObservationManager policy shape mismatch: {obs_manager.group_obs_dim['policy']} vs expected {(expected_shape,)}"
            )
        if policy_obs.shape[1] != expected_shape:
            raise RuntimeError(f"Policy observation tensor has shape {tuple(policy_obs.shape)}, expected second dim {expected_shape}.")
        _require_all_finite("policy_obs", policy_obs)

        goal_obs = mdp.goal_state(env.unwrapped, **GOAL_STATE_PARAMS)
        nav_layers = mdp.local_navigation_map_layers(env.unwrapped, **LOCAL_NAVIGATION_MAP_PARAMS)

        _print_stats("policy_obs", policy_obs)
        _print_stats("goal_state", goal_obs)
        _print_stats("height", nav_layers["height"])
        _print_stats("slope", nav_layers["slope"])
        _print_stats("roughness", nav_layers["roughness"])
        _print_stats("step", nav_layers["step"])
        _print_stats("traversability", nav_layers["traversability"])
        _print_stats("valid_mask", nav_layers["valid_mask"])

        _require_all_finite("goal_state", goal_obs)
        _require_range("goal_state[:,:5]", goal_obs[:, :5], -1.0, 1.0)
        _require_binary("goal_state[:,5]", goal_obs[:, 5])

        _require_all_finite("local_navigation_map.height", nav_layers["height"])
        _require_range("local_navigation_map.slope", nav_layers["slope"], 0.0, 1.0)
        _require_range("local_navigation_map.roughness", nav_layers["roughness"], 0.0, 1.0)
        _require_range("local_navigation_map.step", nav_layers["step"], 0.0, 1.0)
        _require_range("local_navigation_map.traversability", nav_layers["traversability"], 0.0, 1.0)
        _require_binary("local_navigation_map.valid_mask", nav_layers["valid_mask"])

        height_min = float(nav_layers["height"].min().item())
        height_max = float(nav_layers["height"].max().item())
        if height_min < -1.0 or height_max > 1.0:
            print(
                "[WARN] local_navigation_map.height extends outside [-1, 1]. "
                f"Observed min/max: {height_min:.6f}, {height_max:.6f}"
            )

        print(f"[PASS] policy observation shape = {expected_shape}")
        print("[PASS] goal_state and local_navigation_map checks completed without NaN/Inf or range violations.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
