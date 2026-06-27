# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    speed_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward positive forward body-frame velocity for the first locomotion stage."""

    asset: Articulation = env.scene[asset_cfg.name]
    forward_speed = asset.data.root_lin_vel_b[:, 0]
    return torch.clamp(forward_speed / speed_scale, min=0.0)


def forward_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    target_speed: float = 0.45,
    std: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward body-frame forward speed that stays close to a target cruising speed."""

    asset: Articulation = env.scene[asset_cfg.name]
    forward_speed = asset.data.root_lin_vel_b[:, 0]
    speed_error = forward_speed - target_speed
    std_sq = max(std * std, 1e-6)
    return torch.exp(-torch.square(speed_error) / std_sq)


def overspeed_l2(
    env: ManagerBasedRLEnv,
    target_speed: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body-frame forward speed only when it exceeds the target cruising speed."""

    asset: Articulation = env.scene[asset_cfg.name]
    forward_speed = asset.data.root_lin_vel_b[:, 0]
    overspeed = torch.clamp(forward_speed - target_speed, min=0.0)
    return torch.square(overspeed)


def base_height_below_target_l2(
    env: ManagerBasedRLEnv,
    target_height: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the base only when it drops below the target height."""

    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.clamp(target_height - asset.data.root_pos_w[:, 2], min=0.0)
    return torch.square(height_error)


def lin_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize lateral base velocity using an L2 squared kernel."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def joint_limit_margin_penalty(
    env: ManagerBasedRLEnv,
    margin_ratio: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joints that enter a configurable margin near their soft position limits."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    dist_to_lower = joint_pos - joint_limits[..., 0]
    dist_to_upper = joint_limits[..., 1] - joint_pos
    min_margin = torch.minimum(dist_to_lower, dist_to_upper)
    half_range = 0.5 * (joint_limits[..., 1] - joint_limits[..., 0])
    normalized_margin = min_margin / torch.clamp(half_range, min=1e-6)
    return torch.sum(torch.clamp(margin_ratio - normalized_margin, min=0.0), dim=1)


def wheel_semantic_velocity_symmetry_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize mismatch between left/right mean semantic wheel velocity targets."""

    wheel_action_term = env.action_manager.get_term("wheel_motor_csv")
    wheel_forward_sign = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float32, device=env.device).unsqueeze(0)
    semantic_target = wheel_action_term.velocity_target * wheel_forward_sign
    left_mean = semantic_target[:, :2].mean(dim=1)
    right_mean = semantic_target[:, 2:].mean(dim=1)
    return torch.square(left_mean - right_mean)


def wheel_contact_count_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the fraction of wheel bodies that maintain ground contact."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold
    return contacts.to(torch.float32).mean(dim=1)


def wheel_contact_force_balance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward balanced normal contact distribution across wheels that are touching the ground."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_norm = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
    contacts = force_norm > threshold

    contact_count = contacts.sum(dim=1)
    mean_force = force_norm.sum(dim=1) / torch.clamp(contact_count, min=1)
    squared_error = torch.square(force_norm - mean_force.unsqueeze(1)) * contacts
    variance = squared_error.sum(dim=1) / torch.clamp(contact_count, min=1)
    balance_reward = 1.0 / (1.0 + variance)
    # Do not reward non-contact cases; the separate contact-count reward handles that.
    return torch.where(contact_count > 0, balance_reward, torch.zeros_like(balance_reward))
