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


def never_terminate(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Termination helper that never fires."""

    return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
