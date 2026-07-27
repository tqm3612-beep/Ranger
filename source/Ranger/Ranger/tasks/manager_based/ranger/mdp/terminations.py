# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-local termination helpers for Ranger."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from .observations import short_goal_target_body

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum_with_grace(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    grace_time_s: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate on low root height after an initial post-reset grace window."""

    asset: RigidObject = env.scene[asset_cfg.name]
    low_height = asset.data.root_pos_w[:, 2] < float(minimum_height)
    if grace_time_s <= 0.0:
        return low_height

    grace_steps = max(int(math.ceil(float(grace_time_s) / max(float(env.step_dt), 1.0e-6))), 0)
    if grace_steps <= 0:
        return low_height

    grace_active = env.episode_length_buf < grace_steps
    return torch.logical_and(low_height, ~grace_active)


def short_goal_reached(
    env: ManagerBasedRLEnv,
    success_distance: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the short flat goal is reached."""

    _, distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    return distance < float(success_distance)


def short_goal_stopped(
    env: ManagerBasedRLEnv,
    success_distance: float = 0.25,
    max_xy_speed: float = 0.15,
    max_yaw_rate: float = 0.20,
    required_hold_steps: int = 24,
    max_roll: float | None = None,
    max_pitch: float | None = None,
    max_stroke_range: float | None = None,
    max_stroke_tracking_error: float | None = None,
    action_name: str = "leg_hydraulic",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate after the robot remains settled inside the success radius."""

    asset: RigidObject = env.scene[asset_cfg.name]
    _, distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    base_xy_speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    yaw_rate_abs = torch.abs(asset.data.root_ang_vel_b[:, 2])
    posture_ok = torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)
    if max_roll is not None or max_pitch is not None:
        roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
        if max_roll is not None:
            posture_ok &= torch.abs(roll) <= float(max_roll)
        if max_pitch is not None:
            posture_ok &= torch.abs(pitch) <= float(max_pitch)
    action_term = None
    if max_stroke_range is not None or max_stroke_tracking_error is not None:
        action_term = env.action_manager.get_term(action_name)
    if max_stroke_range is not None:
        stroke_actual = action_term.stroke_actual
        stroke_range = stroke_actual.max(dim=1).values - stroke_actual.min(dim=1).values
        posture_ok &= stroke_range <= float(max_stroke_range)
    if max_stroke_tracking_error is not None:
        tracking_error = torch.max(torch.abs(action_term.stroke_desired - action_term.stroke_actual), dim=1).values
        posture_ok &= tracking_error <= float(max_stroke_tracking_error)
    stable_steps = getattr(env, "_short_goal_stop_phase_stable_steps", None)
    if stable_steps is None:
        return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    return (
        (distance < float(success_distance))
        & (base_xy_speed < float(max_xy_speed))
        & (yaw_rate_abs < float(max_yaw_rate))
        & posture_ok
        & (stable_steps >= int(required_hold_steps))
    )


def never_terminate(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Termination helper that never fires."""

    return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
