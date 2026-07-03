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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi
from .observations import _pack_local_map_params, get_local_map_manager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_wheel_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_body_names: tuple[str, str, str, str] = ("w_lb", "w_lf", "w_rf", "w_rb"),
) -> torch.Tensor:
    """Return wheel body positions expressed in the base frame using a fixed wheel ordering."""

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_body_ids, _ = asset.find_bodies(list(wheel_body_names), preserve_order=True)
    wheel_pos_w = asset.data.body_pos_w[:, wheel_body_ids, :]
    wheel_pos_w_rel = wheel_pos_w - asset.data.root_pos_w.unsqueeze(1)
    return quat_apply_inverse(
        asset.data.root_quat_w.unsqueeze(1).expand(-1, len(wheel_body_ids), -1).reshape(-1, 4),
        wheel_pos_w_rel.reshape(-1, 3),
    ).reshape(wheel_pos_w.shape[0], len(wheel_body_ids), 3)


def get_flat_terrain_weight(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    asset_name: str = "robot",
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
    step_threshold: float = 0.08,
    height_reference_x_range: tuple[float, float] = (0.0, 0.4),
    height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
    slope_normalization: float = 0.6,
    roughness_normalization: float = 0.05,
    step_normalization: float = 0.15,
    slope_weight: float = 0.4,
    roughness_weight: float = 0.3,
    step_weight: float = 0.3,
    height_range_weight: float = 0.2,
    flatness_gain: float = 4.0,
) -> torch.Tensor:
    """Estimate how flat the local terrain is, with flat terrain near 1 and complex terrain near 0."""

    manager = get_local_map_manager(env)
    params = _pack_local_map_params(
        sensor_names=sensor_names,
        asset_name=asset_name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        step_threshold=step_threshold,
        height_reference_x_range=height_reference_x_range,
        height_reference_y_range=height_reference_y_range,
        slope_normalization=slope_normalization,
        roughness_normalization=roughness_normalization,
        step_normalization=step_normalization,
        apply_noise=False,
        height_noise_std=0.0,
        risk_noise_std=0.0,
        valid_dropout_prob=0.0,
        slope_weight=slope_weight,
        roughness_weight=roughness_weight,
        step_weight=step_weight,
        unknown_penalty=1.0,
    )
    return manager.get_flat_terrain_weight(
        height_range_weight=height_range_weight,
        flatness_gain=flatness_gain,
        **params,
    )


def get_local_ground_height(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    asset_name: str = "robot",
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
    height_reference_x_range: tuple[float, float] = (0.0, 0.4),
    height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
) -> torch.Tensor:
    """Estimate the local ground height near the robot in the base-frame z convention."""

    manager = get_local_map_manager(env)
    params = _pack_local_map_params(
        sensor_names=sensor_names,
        asset_name=asset_name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        step_threshold=0.08,
        height_reference_x_range=height_reference_x_range,
        height_reference_y_range=height_reference_y_range,
        slope_normalization=0.6,
        roughness_normalization=0.05,
        step_normalization=0.15,
        slope_weight=0.4,
        roughness_weight=0.3,
        step_weight=0.3,
        unknown_penalty=1.0,
    )
    return manager.get_local_ground_height(**params)


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


def underspeed_l2(
    env: ManagerBasedRLEnv,
    target_speed: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body-frame forward speed only when it falls below the target cruising speed."""

    asset: Articulation = env.scene[asset_cfg.name]
    forward_speed = asset.data.root_lin_vel_b[:, 0]
    underspeed = torch.clamp(target_speed - forward_speed, min=0.0)
    return torch.square(underspeed)


def base_height_below_target_l2(
    env: ManagerBasedRLEnv,
    target_height: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the base only when it drops below the target height."""

    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.clamp(target_height - asset.data.root_pos_w[:, 2], min=0.0)
    return torch.square(height_error)


def pitch_angle_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base pitch angle only."""

    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(pitch)


def roll_angle_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base roll angle only."""

    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(roll)


def roll_angle_limit_l2(
    env: ManagerBasedRLEnv,
    allowed_roll_deg: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll angle only when it exceeds the allowed absolute threshold."""

    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    allowed_roll = torch.deg2rad(torch.tensor(allowed_roll_deg, device=roll.device, dtype=roll.dtype))
    roll_excess = torch.clamp(torch.abs(roll) - allowed_roll, min=0.0)
    return torch.square(roll_excess)


def pitch_angle_limit_l2(
    env: ManagerBasedRLEnv,
    allowed_pitch_deg: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize pitch angle only when it exceeds the allowed absolute threshold."""

    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    allowed_pitch = torch.deg2rad(torch.tensor(allowed_pitch_deg, device=pitch.device, dtype=pitch.dtype))
    pitch_excess = torch.clamp(torch.abs(pitch) - allowed_pitch, min=0.0)
    return torch.square(pitch_excess)


def flat_stroke_nominal_l2(
    env: ManagerBasedRLEnv,
    stroke_nominal: float = 0.4,
    sensor_names: tuple[str, ...] = ("mid360_lidar", "avia_lidar", "d435i_camera"),
    asset_name: str = "robot",
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
    step_threshold: float = 0.08,
    height_reference_x_range: tuple[float, float] = (0.0, 0.4),
    height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
    slope_normalization: float = 0.6,
    roughness_normalization: float = 0.05,
    step_normalization: float = 0.15,
    slope_weight: float = 0.4,
    roughness_weight: float = 0.3,
    step_weight: float = 0.3,
    height_range_weight: float = 0.2,
    flatness_gain: float = 4.0,
) -> torch.Tensor:
    """Pull suspension stroke back toward nominal only when the local terrain is flat."""

    flat_weight = get_flat_terrain_weight(
        env=env,
        sensor_names=sensor_names,
        asset_name=asset_name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        step_threshold=step_threshold,
        height_reference_x_range=height_reference_x_range,
        height_reference_y_range=height_reference_y_range,
        slope_normalization=slope_normalization,
        roughness_normalization=roughness_normalization,
        step_normalization=step_normalization,
        slope_weight=slope_weight,
        roughness_weight=roughness_weight,
        step_weight=step_weight,
        height_range_weight=height_range_weight,
        flatness_gain=flatness_gain,
    )
    stroke_command = env.action_manager.get_term("leg_hydraulic").stroke_command
    stroke_error = torch.mean(torch.square(stroke_command - stroke_nominal), dim=1)
    return flat_weight * stroke_error


def flat_stroke_high_l2(
    env: ManagerBasedRLEnv,
    stroke_mean_limit: float = 0.55,
    sensor_names: tuple[str, ...] = ("mid360_lidar", "avia_lidar", "d435i_camera"),
    asset_name: str = "robot",
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
    step_threshold: float = 0.08,
    height_reference_x_range: tuple[float, float] = (0.0, 0.4),
    height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
    slope_normalization: float = 0.6,
    roughness_normalization: float = 0.05,
    step_normalization: float = 0.15,
    slope_weight: float = 0.4,
    roughness_weight: float = 0.3,
    step_weight: float = 0.3,
    height_range_weight: float = 0.2,
    flatness_gain: float = 4.0,
) -> torch.Tensor:
    """Penalize globally high suspension stroke on flat terrain only."""

    flat_weight = get_flat_terrain_weight(
        env=env,
        sensor_names=sensor_names,
        asset_name=asset_name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        step_threshold=step_threshold,
        height_reference_x_range=height_reference_x_range,
        height_reference_y_range=height_reference_y_range,
        slope_normalization=slope_normalization,
        roughness_normalization=roughness_normalization,
        step_normalization=step_normalization,
        slope_weight=slope_weight,
        roughness_weight=roughness_weight,
        step_weight=step_weight,
        height_range_weight=height_range_weight,
        flatness_gain=flatness_gain,
    )
    stroke_mean = env.action_manager.get_term("leg_hydraulic").stroke_command.mean(dim=1)
    stroke_high = torch.clamp(stroke_mean - stroke_mean_limit, min=0.0)
    return flat_weight * torch.square(stroke_high)


def stroke_soft_limit_penalty(
    env: ManagerBasedRLEnv,
    limit: float = 0.5,
    quadratic_gain: float = 4.0,
) -> torch.Tensor:
    """Penalize only suspension stroke commands above a soft limit."""

    stroke_command = env.action_manager.get_term("leg_hydraulic").stroke_command
    stroke_high = torch.clamp(stroke_command - limit, min=0.0)
    return torch.mean(stroke_high + quadratic_gain * torch.square(stroke_high), dim=1)


def actual_stroke_nominal_l2(
    env: ManagerBasedRLEnv,
    stroke_nominal: float = 0.4,
) -> torch.Tensor:
    """Pull measured suspension stroke toward a nominal operating point."""

    stroke_actual = env.action_manager.get_term("leg_hydraulic").stroke_measured
    return torch.mean(torch.square(stroke_actual - stroke_nominal), dim=1)


def actual_stroke_soft_limit_penalty(
    env: ManagerBasedRLEnv,
    limit: float = 0.5,
    quadratic_gain: float = 4.0,
) -> torch.Tensor:
    """Penalize only measured suspension stroke above a soft limit."""

    stroke_actual = env.action_manager.get_term("leg_hydraulic").stroke_measured
    stroke_high = torch.clamp(stroke_actual - limit, min=0.0)
    return torch.mean(stroke_high + quadratic_gain * torch.square(stroke_high), dim=1)


def actual_stroke_rate_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize measured suspension stroke rate."""

    stroke_actual = env.action_manager.get_term(action_name).stroke_measured
    attr_name = "_ranger_prev_measured_stroke"
    prev_stroke = getattr(env, attr_name, None)
    if prev_stroke is None or prev_stroke.shape != stroke_actual.shape:
        prev_stroke = stroke_actual.clone()
        setattr(env, attr_name, prev_stroke)

    episode_start = env.episode_length_buf <= 1
    if torch.any(episode_start):
        prev_stroke[episode_start] = stroke_actual[episode_start]

    stroke_rate = (stroke_actual - prev_stroke) / max(env.step_dt, 1.0e-6)
    penalty = torch.mean(torch.square(stroke_rate), dim=1)
    prev_stroke[:] = stroke_actual
    return penalty


def flat_base_clearance_l2(
    env: ManagerBasedRLEnv,
    clearance_nominal: float = 0.85,
    sensor_names: tuple[str, ...] = ("mid360_lidar", "avia_lidar", "d435i_camera"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
    step_threshold: float = 0.08,
    height_reference_x_range: tuple[float, float] = (0.0, 0.4),
    height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
    slope_normalization: float = 0.6,
    roughness_normalization: float = 0.05,
    step_normalization: float = 0.15,
    slope_weight: float = 0.4,
    roughness_weight: float = 0.3,
    step_weight: float = 0.3,
    height_range_weight: float = 0.2,
    flatness_gain: float = 4.0,
) -> torch.Tensor:
    """Pull base clearance back toward a nominal value only when the local terrain is flat."""

    flat_weight = get_flat_terrain_weight(
        env=env,
        sensor_names=sensor_names,
        asset_name=asset_cfg.name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        step_threshold=step_threshold,
        height_reference_x_range=height_reference_x_range,
        height_reference_y_range=height_reference_y_range,
        slope_normalization=slope_normalization,
        roughness_normalization=roughness_normalization,
        step_normalization=step_normalization,
        slope_weight=slope_weight,
        roughness_weight=roughness_weight,
        step_weight=step_weight,
        height_range_weight=height_range_weight,
        flatness_gain=flatness_gain,
    )
    asset: Articulation = env.scene[asset_cfg.name]
    local_ground_height_b = get_local_ground_height(
        env=env,
        sensor_names=sensor_names,
        asset_name=asset_cfg.name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        height_reference_x_range=height_reference_x_range,
        height_reference_y_range=height_reference_y_range,
    )
    base_clearance = -local_ground_height_b
    return flat_weight * torch.square(base_clearance - clearance_nominal)


def flat_root_height_l2(
    env: ManagerBasedRLEnv,
    root_height_nominal: float = 0.85,
    sensor_names: tuple[str, ...] = ("mid360_lidar", "avia_lidar", "d435i_camera"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
    step_threshold: float = 0.08,
    height_reference_x_range: tuple[float, float] = (0.0, 0.4),
    height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
    slope_normalization: float = 0.6,
    roughness_normalization: float = 0.05,
    step_normalization: float = 0.15,
    slope_weight: float = 0.4,
    roughness_weight: float = 0.3,
    step_weight: float = 0.3,
    height_range_weight: float = 0.2,
    flatness_gain: float = 4.0,
) -> torch.Tensor:
    """Pull root height back toward a nominal value only when the local terrain is flat."""

    flat_weight = get_flat_terrain_weight(
        env=env,
        sensor_names=sensor_names,
        asset_name=asset_cfg.name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        step_threshold=step_threshold,
        height_reference_x_range=height_reference_x_range,
        height_reference_y_range=height_reference_y_range,
        slope_normalization=slope_normalization,
        roughness_normalization=roughness_normalization,
        step_normalization=step_normalization,
        slope_weight=slope_weight,
        roughness_weight=roughness_weight,
        step_weight=step_weight,
        height_range_weight=height_range_weight,
        flatness_gain=flatness_gain,
    )
    asset: Articulation = env.scene[asset_cfg.name]
    return flat_weight * torch.square(asset.data.root_pos_w[:, 2] - root_height_nominal)


def flat_attitude_l2(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...] = ("mid360_lidar", "avia_lidar", "d435i_camera"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
    step_threshold: float = 0.08,
    height_reference_x_range: tuple[float, float] = (0.0, 0.4),
    height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
    slope_normalization: float = 0.6,
    roughness_normalization: float = 0.05,
    step_normalization: float = 0.15,
    slope_weight: float = 0.4,
    roughness_weight: float = 0.3,
    step_weight: float = 0.3,
    height_range_weight: float = 0.2,
    flatness_gain: float = 4.0,
) -> torch.Tensor:
    """Pull roll and pitch back toward zero only when the local terrain is flat."""

    flat_weight = get_flat_terrain_weight(
        env=env,
        sensor_names=sensor_names,
        asset_name=asset_cfg.name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        step_threshold=step_threshold,
        height_reference_x_range=height_reference_x_range,
        height_reference_y_range=height_reference_y_range,
        slope_normalization=slope_normalization,
        roughness_normalization=roughness_normalization,
        step_normalization=step_normalization,
        slope_weight=slope_weight,
        roughness_weight=roughness_weight,
        step_weight=step_weight,
        height_range_weight=height_range_weight,
        flatness_gain=flatness_gain,
    )
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return flat_weight * (torch.square(roll) + torch.square(pitch))


def front_rear_wheel_height_balance_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize front/rear wheel height difference in the base frame."""

    wheel_pos_b = _get_wheel_pos_b(env, asset_cfg)
    front_mean = wheel_pos_b[:, 1:3, 2].mean(dim=1)
    rear_mean = wheel_pos_b[:, [0, 3], 2].mean(dim=1)
    return torch.square(front_mean - rear_mean)


def left_right_wheel_height_balance_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize left/right wheel height difference in the base frame."""

    wheel_pos_b = _get_wheel_pos_b(env, asset_cfg)
    left_mean = wheel_pos_b[:, :2, 2].mean(dim=1)
    right_mean = wheel_pos_b[:, 2:, 2].mean(dim=1)
    return torch.square(left_mean - right_mean)


def front_rear_stroke_balance_l2(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...] = ("mid360_lidar", "avia_lidar", "d435i_camera"),
    asset_name: str = "robot",
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
    step_threshold: float = 0.08,
    height_reference_x_range: tuple[float, float] = (0.0, 0.4),
    height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
    slope_normalization: float = 0.6,
    roughness_normalization: float = 0.05,
    step_normalization: float = 0.15,
    slope_weight: float = 0.4,
    roughness_weight: float = 0.3,
    step_weight: float = 0.3,
    height_range_weight: float = 0.2,
    flatness_gain: float = 4.0,
) -> torch.Tensor:
    """Penalize front/rear hydraulic stroke difference only on flat terrain."""

    flat_weight = get_flat_terrain_weight(
        env=env,
        sensor_names=sensor_names,
        asset_name=asset_name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        step_threshold=step_threshold,
        height_reference_x_range=height_reference_x_range,
        height_reference_y_range=height_reference_y_range,
        slope_normalization=slope_normalization,
        roughness_normalization=roughness_normalization,
        step_normalization=step_normalization,
        slope_weight=slope_weight,
        roughness_weight=roughness_weight,
        step_weight=step_weight,
        height_range_weight=height_range_weight,
        flatness_gain=flatness_gain,
    )
    leg_action_term = env.action_manager.get_term("leg_hydraulic")
    stroke_command = leg_action_term.stroke_command
    front_mean = stroke_command[:, 1:3].mean(dim=1)
    rear_mean = stroke_command[:, [0, 3]].mean(dim=1)
    return flat_weight * torch.square(front_mean - rear_mean)


def base_pitch_stroke_compensation_l2(
    env: ManagerBasedRLEnv,
    k_pitch: float = 1.0,
    target_diff_limit: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Match front/rear stroke split to the measured base pitch."""

    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    stroke_command = env.action_manager.get_term("leg_hydraulic").stroke_command
    front_mean = stroke_command[:, [1, 2]].mean(dim=1)
    rear_mean = stroke_command[:, [0, 3]].mean(dim=1)
    actual_diff = front_mean - rear_mean
    target_diff = torch.clamp(k_pitch * pitch, min=-target_diff_limit, max=target_diff_limit)
    return torch.square(actual_diff - target_diff)


def left_right_stroke_balance_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize left/right hydraulic stroke difference."""

    leg_action_term = env.action_manager.get_term("leg_hydraulic")
    stroke_command = leg_action_term.stroke_command
    left_mean = stroke_command[:, :2].mean(dim=1)
    right_mean = stroke_command[:, 2:].mean(dim=1)
    return torch.square(left_mean - right_mean)


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
