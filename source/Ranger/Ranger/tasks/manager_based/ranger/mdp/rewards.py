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
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .observations import GOAL_HEADING_PREV_HEADING_ERROR_ATTR, goal_heading_target_body, speed_command

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _wheel_semantic_sign_from_names(
    wheel_names: list[str],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return semantic forward sign derived from Ranger wheel joint names."""

    forward_sign_by_name = {
        "w_lb": -1.0,
        "w_lf": -1.0,
        "w_rf": 1.0,
        "w_rb": 1.0,
    }
    return torch.tensor([forward_sign_by_name[name] for name in wheel_names], device=device, dtype=dtype).unsqueeze(0)


def wheel_semantic_sign_lf_lr_rf_rr(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return wheel semantic sign for ``[lf, lr, rf, rr]`` ordered tensors."""

    return _wheel_semantic_sign_from_names(["w_lf", "w_lb", "w_rf", "w_rb"], device=device, dtype=dtype)


def wheel_semantic_sign_lr_lf_rf_rr(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return wheel semantic sign for raw action/joint order ``[lr, lf, rf, rr]``."""

    return _wheel_semantic_sign_from_names(["w_lb", "w_lf", "w_rf", "w_rb"], device=device, dtype=dtype)


def wheel_raw_to_semantic_lf_lr_rf_rr(wheel_tensor: torch.Tensor) -> torch.Tensor:
    """Convert raw wheel values in ``[lf, lr, rf, rr]`` order to semantic values."""

    sign = wheel_semantic_sign_lf_lr_rf_rr(device=wheel_tensor.device, dtype=wheel_tensor.dtype)
    return wheel_tensor * sign


def wheel_raw_to_semantic_lr_lf_rf_rr(wheel_tensor: torch.Tensor) -> torch.Tensor:
    """Convert raw wheel values in ``[lr, lf, rf, rr]`` order to semantic values."""

    sign = wheel_semantic_sign_lr_lf_rf_rr(device=wheel_tensor.device, dtype=wheel_tensor.dtype)
    return wheel_tensor * sign


def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    speed_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward positive forward body-frame velocity for the first locomotion stage."""

    asset: Articulation = env.scene[asset_cfg.name]
    forward_speed = asset.data.root_lin_vel_b[:, 0]
    return torch.clamp(forward_speed / speed_scale, min=0.0)


def lin_vel_x_command_tracking_exp(
    env: ManagerBasedRLEnv,
    std_sq: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward body-frame forward speed tracking the dynamic command."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = speed_command(env)
    error = asset.data.root_lin_vel_b[:, 0] - command[:, 0]
    return torch.exp(-torch.square(error) / max(float(std_sq), 1.0e-6))


def lin_vel_x_command_error_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return absolute forward-speed command tracking error."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = speed_command(env)
    return torch.abs(asset.data.root_lin_vel_b[:, 0] - command[:, 0])


def low_cmd_speed_error_l1(
    env: ManagerBasedRLEnv,
    command_threshold: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize speed tracking error only for low forward-speed commands."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = speed_command(env)
    low_mask = (command[:, 0] < float(command_threshold)).to(torch.float32)
    return low_mask * torch.abs(asset.data.root_lin_vel_b[:, 0] - command[:, 0])


def mid_cmd_speed_error_l1(
    env: ManagerBasedRLEnv,
    command_min: float = 0.12,
    command_max: float = 0.28,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize speed tracking error only for mid forward-speed commands."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = speed_command(env)
    mid_mask = ((command[:, 0] >= float(command_min)) & (command[:, 0] < float(command_max))).to(torch.float32)
    return mid_mask * torch.abs(asset.data.root_lin_vel_b[:, 0] - command[:, 0])


def normalized_cmd_speed_error_l1(
    env: ManagerBasedRLEnv,
    min_command: float = 0.08,
    max_error: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize command-relative speed error so low-speed commands are not ignored."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = speed_command(env)
    error = torch.abs(asset.data.root_lin_vel_b[:, 0] - command[:, 0])
    normalized_error = error / torch.clamp(command[:, 0], min=float(min_command))
    return torch.clamp(normalized_error, max=float(max_error))


def overspeed_command_penalty(
    env: ManagerBasedRLEnv,
    margin: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize forward speed that exceeds the dynamic command by a margin."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = speed_command(env)
    return torch.clamp(asset.data.root_lin_vel_b[:, 0] - command[:, 0] - float(margin), min=0.0)


def backward_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize backward body-frame motion."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.root_lin_vel_b[:, 0], min=0.0)


def yaw_rate_command_tracking_exp(
    env: ManagerBasedRLEnv,
    std_sq: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward body-frame yaw-rate tracking the dynamic command."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = speed_command(env)
    error = asset.data.root_ang_vel_b[:, 2] - command[:, 1]
    return torch.exp(-torch.square(error) / max(float(std_sq), 1.0e-6))


def yaw_rate_command_error_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return absolute yaw-rate command tracking error."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = speed_command(env)
    return torch.abs(asset.data.root_ang_vel_b[:, 2] - command[:, 1])


def turn_direction_reward(
    env: ManagerBasedRLEnv,
    yaw_cmd_deadband: float = 0.1,
    yaw_rate_deadband: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward matching the requested yaw direction for non-zero yaw commands."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = speed_command(env)
    yaw_cmd = command[:, 1]
    yaw_rate = asset.data.root_ang_vel_b[:, 2]
    active = (torch.abs(yaw_cmd) > float(yaw_cmd_deadband)) & (torch.abs(yaw_rate) > float(yaw_rate_deadband))
    return (active & (torch.sign(yaw_rate) == torch.sign(yaw_cmd))).to(torch.float32)


def expected_speed_command_wheel_velocity(
    env: ManagerBasedRLEnv,
    forward_gain: float = 9.0,
    yaw_gain: float = 2.5,
    max_abs_speed: float = 6.0,
    forward_sign: float = 1.0,
) -> torch.Tensor:
    """Return expected physical wheel velocity target for the current command."""

    command = speed_command(env)
    forward_speed = float(forward_sign) * float(forward_gain) * command[:, 0]
    turn_speed = float(yaw_gain) * command[:, 1]
    left_semantic = forward_speed - turn_speed
    right_semantic = forward_speed + turn_speed
    semantic_expected = torch.stack(
        (left_semantic, left_semantic, right_semantic, right_semantic),
        dim=1,
    )
    expected = semantic_expected * wheel_semantic_sign_lr_lf_rf_rr(device=env.device)
    return torch.clamp(expected, min=-float(max_abs_speed), max=float(max_abs_speed))


def expected_speed_command_wheel_velocity_semantic(
    env: ManagerBasedRLEnv,
    forward_gain: float = 9.0,
    yaw_gain: float = 2.5,
    max_abs_speed: float = 6.0,
    forward_sign: float = 1.0,
) -> torch.Tensor:
    """Return expected semantic wheel speed for debugging and sign checks."""

    command = speed_command(env)
    forward_speed = float(forward_sign) * float(forward_gain) * command[:, 0]
    turn_speed = float(yaw_gain) * command[:, 1]
    left_semantic = forward_speed - turn_speed
    right_semantic = forward_speed + turn_speed
    expected = torch.stack(
        (left_semantic, left_semantic, right_semantic, right_semantic),
        dim=1,
    )
    return torch.clamp(expected, min=-float(max_abs_speed), max=float(max_abs_speed))


def wheel_command_tracking_exp(
    env: ManagerBasedRLEnv,
    forward_gain: float = 9.0,
    yaw_gain: float = 2.5,
    max_abs_speed: float = 6.0,
    forward_sign: float = 1.0,
    std_sq: float = 4.0,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Reward scaled wheel velocity targets that follow the command prior."""

    wheel_action_term = env.action_manager.get_term(action_name)
    expected = expected_speed_command_wheel_velocity(
        env=env,
        forward_gain=forward_gain,
        yaw_gain=yaw_gain,
        max_abs_speed=max_abs_speed,
        forward_sign=forward_sign,
    )
    error_sq = torch.mean(torch.square(wheel_action_term.velocity_target - expected), dim=1)
    return torch.exp(-error_sq / max(float(std_sq), 1.0e-6))


def wheel_command_error_l1(
    env: ManagerBasedRLEnv,
    forward_gain: float = 9.0,
    yaw_gain: float = 2.5,
    max_abs_speed: float = 6.0,
    forward_sign: float = 1.0,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Penalize scaled wheel velocity target deviation from the command prior."""

    wheel_action_term = env.action_manager.get_term(action_name)
    expected = expected_speed_command_wheel_velocity_semantic(
        env=env,
        forward_gain=forward_gain,
        yaw_gain=yaw_gain,
        max_abs_speed=max_abs_speed,
        forward_sign=forward_sign,
    )
    semantic_target = wheel_raw_to_semantic_lr_lf_rf_rr(wheel_action_term.velocity_target)
    return torch.mean(torch.abs(semantic_target - expected), dim=1)


def wheel_target_magnitude_l1(env: ManagerBasedRLEnv, action_name: str = "wheel_motor_csv") -> torch.Tensor:
    """Penalize large scaled wheel velocity targets."""

    wheel_action_term = env.action_manager.get_term(action_name)
    return torch.mean(torch.abs(wheel_action_term.velocity_target), dim=1)


def wheel_turn_difference_command_l1(
    env: ManagerBasedRLEnv,
    yaw_gain: float = 2.5,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Penalize left-right wheel target difference error for yaw commands."""

    wheel_action_term = env.action_manager.get_term(action_name)
    target = wheel_action_term.velocity_target
    left_mean = target[:, :2].mean(dim=1)
    right_mean = target[:, 2:].mean(dim=1)
    expected_diff = -2.0 * float(yaw_gain) * speed_command(env)[:, 1]
    return torch.abs((left_mean - right_mean) - expected_diff)


def goal_heading_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot facing the sampled target direction."""

    _, _, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
    return torch.cos(heading_error)


def goal_heading_velocity_towards_target(
    env: ManagerBasedRLEnv,
    max_velocity: float = 1.0,
    min_reward: float = -1.0,
    max_reward: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward body-frame velocity projected onto the target direction."""

    asset: Articulation = env.scene[asset_cfg.name]
    target_vec_b, distance, _ = goal_heading_target_body(env, asset_cfg=asset_cfg)
    target_dir_b = target_vec_b[:, :2] / torch.clamp(distance.unsqueeze(1), min=1.0e-6)
    velocity_towards_target = torch.sum(asset.data.root_lin_vel_b[:, :2] * target_dir_b, dim=1)
    normalized_velocity = velocity_towards_target / max(float(max_velocity), 1.0e-6)
    return torch.clamp(normalized_velocity, min=float(min_reward), max=float(max_reward))


def goal_heading_turn_direction(
    env: ManagerBasedRLEnv,
    heading_deadband: float = 0.08,
    yaw_rate_deadband: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw-rate sign matching the target bearing sign."""

    asset: Articulation = env.scene[asset_cfg.name]
    _, _, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
    yaw_rate = asset.data.root_ang_vel_b[:, 2]
    active = (torch.abs(heading_error) > float(heading_deadband)) & (torch.abs(yaw_rate) > float(yaw_rate_deadband))
    return (active & (torch.sign(yaw_rate) == torch.sign(heading_error))).to(torch.float32)


def goal_heading_error_progress(
    env: ManagerBasedRLEnv,
    min_reward: float = 0.0,
    max_reward: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reduction in absolute target heading error from the previous step."""

    _, _, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
    prev_heading_error = getattr(env, GOAL_HEADING_PREV_HEADING_ERROR_ATTR, None)
    if prev_heading_error is None or prev_heading_error.shape != (env.num_envs,):
        prev_heading_error = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
        setattr(env, GOAL_HEADING_PREV_HEADING_ERROR_ATTR, prev_heading_error)

    prev_abs_error = torch.abs(prev_heading_error)
    current_abs_error = torch.abs(heading_error)
    dt = max(float(getattr(env, "step_dt", 1.0 / 60.0)), 1.0e-6)
    valid_prev = torch.isfinite(prev_abs_error)
    progress = torch.where(
        valid_prev,
        (prev_abs_error - current_abs_error) / dt,
        torch.zeros_like(current_abs_error),
    )
    prev_heading_error[:] = heading_error
    return torch.clamp(progress, min=float(min_reward), max=float(max_reward))


def goal_heading_wheel_prior_exp(
    env: ManagerBasedRLEnv,
    forward_gain: float = 4.0,
    turn_gain: float = 3.0,
    max_abs_speed: float = 6.0,
    std_sq: float = 4.0,
    action_name: str = "wheel_motor_csv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward semantic wheel targets that encode target-heading differential drive."""

    wheel_action_term = env.action_manager.get_term(action_name)
    _, distance, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
    forward_speed = float(forward_gain) * torch.clamp(torch.cos(heading_error), min=0.0)
    forward_speed = forward_speed * torch.clamp(distance / 2.0, max=1.0)
    turn_speed = float(turn_gain) * torch.sin(heading_error)
    left_semantic = forward_speed - turn_speed
    right_semantic = forward_speed + turn_speed
    expected = torch.stack((left_semantic, left_semantic, right_semantic, right_semantic), dim=1)
    expected = torch.clamp(expected, min=-float(max_abs_speed), max=float(max_abs_speed))
    semantic_target = wheel_raw_to_semantic_lr_lf_rf_rr(wheel_action_term.velocity_target)
    error_sq = torch.mean(torch.square(semantic_target - expected), dim=1)
    return torch.exp(-error_sq / max(float(std_sq), 1.0e-6))


def goal_heading_front_wheel_diff_l1(
    env: ManagerBasedRLEnv,
    heading_deadband: float = 0.08,
    action_name: str = "wheel_motor_csv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize semantic wheel differential only when the target is nearly straight ahead."""

    wheel_action_term = env.action_manager.get_term(action_name)
    _, _, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
    front_mask = (torch.abs(heading_error) <= float(heading_deadband)).to(torch.float32)
    semantic_target = wheel_raw_to_semantic_lr_lf_rf_rr(wheel_action_term.velocity_target)
    left_mean = semantic_target[:, :2].mean(dim=1)
    right_mean = semantic_target[:, 2:].mean(dim=1)
    return front_mask * torch.abs(left_mean - right_mean)


def goal_heading_wheel_diff_l1(
    env: ManagerBasedRLEnv,
    turn_gain: float = 3.0,
    max_abs_diff: float = 6.0,
    action_name: str = "wheel_motor_csv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize semantic left-right wheel difference mismatch for target heading."""

    wheel_action_term = env.action_manager.get_term(action_name)
    _, _, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
    semantic_target = wheel_raw_to_semantic_lr_lf_rf_rr(wheel_action_term.velocity_target)
    left_mean = semantic_target[:, :2].mean(dim=1)
    right_mean = semantic_target[:, 2:].mean(dim=1)
    expected_diff = torch.clamp(-2.0 * float(turn_gain) * torch.sin(heading_error), -float(max_abs_diff), float(max_abs_diff))
    return torch.abs((left_mean - right_mean) - expected_diff)


def turn_to_target_heading_alignment_exp(
    env: ManagerBasedRLEnv,
    sigma: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward exponential alignment with the sampled target heading."""

    _, _, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
    sigma_sq = max(float(sigma) * float(sigma), 1.0e-6)
    return torch.exp(-torch.square(heading_error) / sigma_sq)


def turn_to_target_heading_progress(
    env: ManagerBasedRLEnv,
    min_progress: float = -0.25,
    max_progress: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward step-wise reduction in absolute heading error."""

    _, _, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
    prev_heading_error = getattr(env, GOAL_HEADING_PREV_HEADING_ERROR_ATTR, None)
    if prev_heading_error is None or prev_heading_error.shape != (env.num_envs,):
        prev_heading_error = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
        setattr(env, GOAL_HEADING_PREV_HEADING_ERROR_ATTR, prev_heading_error)

    prev_abs_error = torch.abs(prev_heading_error)
    current_abs_error = torch.abs(heading_error)
    progress = torch.where(
        torch.isfinite(prev_abs_error),
        prev_abs_error - current_abs_error,
        torch.zeros_like(current_abs_error),
    )
    prev_heading_error[:] = heading_error
    return torch.clamp(progress, min=float(min_progress), max=float(max_progress))


def turn_to_target_yaw_rate_command(
    env: ManagerBasedRLEnv,
    k_yaw: float = 1.5,
    yaw_rate_max: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the bounded yaw-rate command implied by the current heading error."""

    _, _, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
    return torch.clamp(float(k_yaw) * heading_error, min=-float(yaw_rate_max), max=float(yaw_rate_max))


def turn_to_target_yaw_tracking_exp(
    env: ManagerBasedRLEnv,
    k_yaw: float = 1.5,
    yaw_rate_max: float = 0.8,
    sigma_yaw: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward body yaw-rate tracking for the heading-derived yaw command."""

    asset: Articulation = env.scene[asset_cfg.name]
    yaw_rate_cmd = turn_to_target_yaw_rate_command(
        env=env,
        k_yaw=k_yaw,
        yaw_rate_max=yaw_rate_max,
        asset_cfg=asset_cfg,
    )
    sigma_sq = max(float(sigma_yaw) * float(sigma_yaw), 1.0e-6)
    yaw_error = asset.data.root_ang_vel_b[:, 2] - yaw_rate_cmd
    return torch.exp(-torch.square(yaw_error) / sigma_sq)


def _wheel_joint_vel_lf_lr_rf_rr(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return 4 wheel joint velocities ordered as ``[lf, lr, rf, rr]``."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(["w_lf", "w_lb", "w_rf", "w_rb"], preserve_order=True)
    wheel_vel = asset.data.joint_vel[:, joint_ids]
    assert wheel_vel.ndim == 2
    assert wheel_vel.shape[1] == 4
    return wheel_vel


def _wheel_target_vel_lf_lr_rf_rr(
    env: ManagerBasedRLEnv,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Return 4 wheel velocity targets ordered as ``[lf, lr, rf, rr]``."""

    action_term = env.action_manager.get_term(action_name)
    wheel_target = action_term.velocity_target[:, [1, 0, 2, 3]]
    assert wheel_target.ndim == 2
    assert wheel_target.shape[1] == 4
    return wheel_target


def _wheel_contact_force_lf_lr_rf_rr(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return 4 wheel contact-force magnitudes ordered as ``[lf, lr, rf, rr]``."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids, _ = contact_sensor.find_bodies(["w_lf", "w_lb", "w_rf", "w_rb"], preserve_order=True)
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, body_ids, :]
    force_norm = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
    assert force_norm.ndim == 2
    assert force_norm.shape[1] == 4
    return force_norm


def _wheel_contact_force_ratio_lf_lr_rf_rr(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return wheel force, total force, normalized ratios, and a valid-total mask in ``[lf, lr, rf, rr]`` order."""

    force_norm = _wheel_contact_force_lf_lr_rf_rr(env, sensor_cfg)
    total_force = force_norm.sum(dim=1)
    valid_total_mask = total_force > float(eps)
    safe_total_force = torch.where(valid_total_mask, total_force, torch.ones_like(total_force))
    force_ratio = torch.where(valid_total_mask.unsqueeze(1), force_norm / safe_total_force.unsqueeze(1), torch.zeros_like(force_norm))
    return force_norm, total_force, force_ratio, valid_total_mask


def wheel_same_side_front_rear_diff_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
) -> torch.Tensor:
    """Penalize front-rear velocity mismatch on each side without constraining left-right differential drive."""

    wheel_vel = _wheel_joint_vel_lf_lr_rf_rr(env, asset_cfg)
    left_front_rear_diff = torch.abs(wheel_vel[:, 0] - wheel_vel[:, 1])
    right_front_rear_diff = torch.abs(wheel_vel[:, 2] - wheel_vel[:, 3])
    return left_front_rear_diff + right_front_rear_diff


def unloaded_wheel_spin_penalty(
    env: ManagerBasedRLEnv,
    force_threshold: float = 20.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
) -> torch.Tensor:
    """Penalize wheel spin when contact force is very small."""

    force_norm = _wheel_contact_force_lf_lr_rf_rr(env, sensor_cfg)
    wheel_vel = torch.abs(_wheel_joint_vel_lf_lr_rf_rr(env, asset_cfg))
    unloaded_mask = (force_norm < float(force_threshold)).to(wheel_vel.dtype)
    assert unloaded_mask.ndim == 2
    assert unloaded_mask.shape[1] == 4
    assert wheel_vel.shape == unloaded_mask.shape
    return torch.mean(unloaded_mask * wheel_vel, dim=1)


def wheel_target_overspeed_penalty(
    env: ManagerBasedRLEnv,
    safe_target_vel: float = 10.0,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Penalize only excessive wheel velocity targets while preserving normal differential steering."""

    wheel_target = _wheel_target_vel_lf_lr_rf_rr(env, action_name=action_name)
    overspeed = torch.relu(torch.abs(wheel_target) - float(safe_target_vel))
    return torch.mean(torch.square(overspeed), dim=1)


def wheel_target_common_mode_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Penalize common-mode wheel target magnitude to discourage turn-only policies from using shared fore-aft motion."""

    raw_target = _wheel_target_vel_lf_lr_rf_rr(env, action_name=action_name)
    semantic_target = wheel_raw_to_semantic_lf_lr_rf_rr(raw_target)
    target_left_mean = semantic_target[:, :2].mean(dim=1)
    target_right_mean = semantic_target[:, 2:].mean(dim=1)
    target_common = 0.5 * (target_left_mean + target_right_mean)
    return torch.abs(target_common)


def roll_angle_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base roll angle."""

    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(roll)


def pitch_angle_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base pitch angle."""

    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(pitch)


def base_height_below_target_l2(
    env: ManagerBasedRLEnv,
    target_height: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root height only when below a target."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(torch.clamp(float(target_height) - asset.data.root_pos_w[:, 2], min=0.0))


def base_height_above_target_l1(
    env: ManagerBasedRLEnv,
    target_height: float = 0.80,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root height only when above a target."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_pos_w[:, 2] - float(target_height), min=0.0)


def root_height_tracking_exp(
    env: ManagerBasedRLEnv,
    target_height: float = 0.74,
    std_sq: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward root height tracking around a nominal flat-ground support height."""

    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.root_pos_w[:, 2] - float(target_height)
    return torch.exp(-torch.square(error) / max(float(std_sq), 1.0e-6))


def root_height_band_piecewise(
    env: ManagerBasedRLEnv,
    peak_min: float = 0.74,
    peak_max: float = 0.76,
    healthy_min: float = 0.70,
    healthy_max: float = 0.80,
    low_floor: float = 0.65,
    high_penalty_scale: float = 6.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Piecewise stand-height shaping with a narrow peak band and gentle healthy shoulders."""

    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]

    reward = torch.zeros_like(height)

    low_shoulder = (height >= float(healthy_min)) & (height < float(peak_min))
    reward = torch.where(
        low_shoulder,
        0.4 + 0.6 * (height - float(healthy_min)) / max(float(peak_min - healthy_min), 1.0e-6),
        reward,
    )

    peak_band = (height >= float(peak_min)) & (height <= float(peak_max))
    reward = torch.where(peak_band, torch.ones_like(reward), reward)

    high_shoulder = (height > float(peak_max)) & (height <= float(healthy_max))
    reward = torch.where(
        high_shoulder,
        0.4 + 0.6 * (float(healthy_max) - height) / max(float(healthy_max - peak_max), 1.0e-6),
        reward,
    )

    low_band = (height >= float(low_floor)) & (height < float(healthy_min))
    reward = torch.where(
        low_band,
        -(float(healthy_min) - height) / max(float(healthy_min - low_floor), 1.0e-6),
        reward,
    )

    below_floor = height < float(low_floor)
    reward = torch.where(
        below_floor,
        -1.0 - 3.0 * (float(low_floor) - height) / max(float(low_floor), 1.0e-6),
        reward,
    )

    above_healthy = height > float(healthy_max)
    reward = torch.where(
        above_healthy,
        -float(high_penalty_scale) * torch.square(height - float(healthy_max)),
        reward,
    )
    return reward


def actual_stroke_nominal_l2(
    env: ManagerBasedRLEnv,
    stroke_nominal: float = 0.4,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize measured hydraulic stroke away from the nominal working point."""

    action_term = env.action_manager.get_term(action_name)
    return torch.mean(torch.square(action_term.stroke_actual - float(stroke_nominal)), dim=1)


def actual_stroke_soft_limit_penalty(
    env: ManagerBasedRLEnv,
    limit: float = 0.55,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize measured hydraulic stroke above a soft upper limit."""

    action_term = env.action_manager.get_term(action_name)
    over = torch.clamp(action_term.stroke_actual - float(limit), min=0.0)
    return torch.mean(over + 4.0 * torch.square(over), dim=1)


def stroke_range_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize large per-env spread across the four measured hydraulic strokes."""

    action_term = env.action_manager.get_term(action_name)
    stroke_actual = action_term.stroke_actual
    return stroke_actual.max(dim=1).values - stroke_actual.min(dim=1).values


def stroke_diagonal_balance_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize diagonal split in measured hydraulic stroke using ``[lf, lr, rf, rr]`` semantics."""

    stroke_actual, _ = _hydraulic_stroke_action_lf_lr_rf_rr(env, action_name=action_name)
    diag_a = stroke_actual[:, 0] + stroke_actual[:, 3]
    diag_b = stroke_actual[:, 1] + stroke_actual[:, 2]
    return torch.abs(diag_a - diag_b)


def hydraulic_action_magnitude_l1(
    env: ManagerBasedRLEnv,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize large absolute hydraulic actions."""

    action_term = env.action_manager.get_term(action_name)
    return torch.mean(torch.abs(action_term.raw_actions), dim=1)


def hydraulic_action_range_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize wide spread across per-leg hydraulic actions."""

    action_term = env.action_manager.get_term(action_name)
    raw_actions = action_term.raw_actions
    return raw_actions.max(dim=1).values - raw_actions.min(dim=1).values


def _hydraulic_stroke_action_lf_lr_rf_rr(
    env: ManagerBasedRLEnv,
    action_name: str = "leg_hydraulic",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return measured hydraulic stroke and actuator raw action in ``[lf, lr, rf, rr]`` order."""

    action_term = env.action_manager.get_term(action_name)
    stroke_actual = action_term.stroke_actual[:, [1, 0, 2, 3]]
    hydraulic_action = action_term.raw_actions[:, [1, 0, 2, 3]]
    return stroke_actual, hydraulic_action


def low_stroke_negative_hydraulic_action_penalty(
    env: ManagerBasedRLEnv,
    stroke_threshold: float = 0.35,
    stroke_margin: float = 0.10,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize negative hydraulic action on legs whose measured stroke is already low."""

    stroke_actual, hydraulic_action = _hydraulic_stroke_action_lf_lr_rf_rr(env, action_name=action_name)
    stroke_low_factor = torch.relu(float(stroke_threshold) - stroke_actual) / max(float(stroke_margin), 1.0e-6)
    bad_action = torch.relu(-hydraulic_action)
    penalty_per_leg = stroke_low_factor * bad_action
    return penalty_per_leg.mean(dim=1)


def height_gated_low_stroke_negative_hydraulic_action_penalty(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.72,
    height_margin: float = 0.07,
    stroke_threshold: float = 0.35,
    stroke_margin: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize negative hydraulic action on low-stroke legs more strongly when root height is already low."""

    stroke_actual, hydraulic_action = _hydraulic_stroke_action_lf_lr_rf_rr(env, action_name=action_name)
    asset: Articulation = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]
    stroke_low_factor = torch.relu(float(stroke_threshold) - stroke_actual) / max(float(stroke_margin), 1.0e-6)
    height_low_factor = torch.relu(float(height_threshold) - root_height) / max(float(height_margin), 1.0e-6)
    bad_action = torch.relu(-hydraulic_action)
    penalty_per_leg = height_low_factor.unsqueeze(1) * stroke_low_factor * bad_action
    return penalty_per_leg.mean(dim=1)


def high_height_low_stroke_negative_hydraulic_action_penalty(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.80,
    height_margin: float = 0.10,
    stroke_threshold: float = 0.45,
    stroke_margin: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize negative hydraulic action on already low-stroke legs when body height is too high."""

    stroke_actual, hydraulic_action = _hydraulic_stroke_action_lf_lr_rf_rr(env, action_name=action_name)
    asset: Articulation = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]
    height_high_factor = torch.relu(root_height - float(height_threshold)) / max(float(height_margin), 1.0e-6)
    stroke_low_factor = torch.relu(float(stroke_threshold) - stroke_actual) / max(float(stroke_margin), 1.0e-6)
    bad_action = torch.relu(-hydraulic_action)
    penalty_per_leg = height_high_factor.unsqueeze(1) * stroke_low_factor * bad_action
    return penalty_per_leg.mean(dim=1)


def height_low_high_stroke_positive_hydraulic_action_penalty(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.72,
    height_margin: float = 0.07,
    stroke_high_threshold: float = 0.55,
    stroke_margin: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize positive hydraulic action on already high-stroke legs when root height is low."""

    stroke_actual, hydraulic_action = _hydraulic_stroke_action_lf_lr_rf_rr(env, action_name=action_name)
    asset: Articulation = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]
    height_low_factor = torch.relu(float(height_threshold) - root_height) / max(float(height_margin), 1.0e-6)
    stroke_high_factor = torch.relu(stroke_actual - float(stroke_high_threshold)) / max(float(stroke_margin), 1.0e-6)
    bad_action = torch.relu(hydraulic_action)
    penalty_per_leg = height_low_factor.unsqueeze(1) * stroke_high_factor * bad_action
    return penalty_per_leg.mean(dim=1)


def stroke_away_from_nominal_hydraulic_action_penalty(
    env: ManagerBasedRLEnv,
    nominal_stroke: float = 0.50,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize hydraulic action that pushes stroke farther away from a nominal operating point."""

    stroke_actual, hydraulic_action = _hydraulic_stroke_action_lf_lr_rf_rr(env, action_name=action_name)
    stroke_error = stroke_actual - float(nominal_stroke)
    bad_direction = torch.relu(stroke_error * hydraulic_action)
    return bad_direction.mean(dim=1)


def front_rear_stroke_balance_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize front-rear split in mean hydraulic stroke."""

    stroke_actual, _ = _hydraulic_stroke_action_lf_lr_rf_rr(env, action_name=action_name)
    front_mean = 0.5 * (stroke_actual[:, 0] + stroke_actual[:, 2])
    rear_mean = 0.5 * (stroke_actual[:, 1] + stroke_actual[:, 3])
    return torch.abs(rear_mean - front_mean)


def front_rear_stroke_split_action_penalty(
    env: ManagerBasedRLEnv,
    diff_threshold: float = 0.10,
    diff_margin: float = 0.20,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize hydraulic actions that further amplify an existing front-rear stroke split."""

    stroke_actual, hydraulic_action = _hydraulic_stroke_action_lf_lr_rf_rr(env, action_name=action_name)
    front_mean = 0.5 * (stroke_actual[:, 0] + stroke_actual[:, 2])
    rear_mean = 0.5 * (stroke_actual[:, 1] + stroke_actual[:, 3])
    split = rear_mean - front_mean

    front_negative_action = 0.5 * (torch.relu(-hydraulic_action[:, 0]) + torch.relu(-hydraulic_action[:, 2]))
    rear_positive_action = 0.5 * (torch.relu(hydraulic_action[:, 1]) + torch.relu(hydraulic_action[:, 3]))
    front_positive_action = 0.5 * (torch.relu(hydraulic_action[:, 0]) + torch.relu(hydraulic_action[:, 2]))
    rear_negative_action = 0.5 * (torch.relu(-hydraulic_action[:, 1]) + torch.relu(-hydraulic_action[:, 3]))

    positive_factor = torch.clamp(
        torch.relu(split - float(diff_threshold)) / max(float(diff_margin), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    negative_factor = torch.clamp(
        torch.relu(-split - float(diff_threshold)) / max(float(diff_margin), 1.0e-6),
        min=0.0,
        max=1.0,
    )

    return positive_factor * (front_negative_action + rear_positive_action) + negative_factor * (
        front_positive_action + rear_negative_action
    )


def wheel_semantic_velocity_symmetry_l2(env: ManagerBasedRLEnv, action_name: str = "wheel_motor_csv") -> torch.Tensor:
    """Penalize left/right semantic wheel target mismatch."""

    wheel_action_term = env.action_manager.get_term(action_name)
    semantic_target = wheel_raw_to_semantic_lr_lf_rf_rr(wheel_action_term.velocity_target)
    left_mean = semantic_target[:, :2].mean(dim=1)
    right_mean = semantic_target[:, 2:].mean(dim=1)
    return torch.square(left_mean - right_mean)


def lin_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize lateral base velocity using an L2 squared kernel."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def base_yaw_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body-frame yaw angular velocity for stand stabilization."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def base_xy_speed_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar base speed magnitude for stand stabilization."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def stand_yaw_drift_abs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw drift away from the reset heading in stand training."""

    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    reset_yaw = getattr(env, "_stand_reset_yaw", None)
    if reset_yaw is None:
        return torch.zeros_like(yaw)
    return torch.abs(torch.atan2(torch.sin(yaw - reset_yaw), torch.cos(yaw - reset_yaw)))


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


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


def wheel_all_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward only when all selected wheel bodies maintain ground contact."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold
    return torch.all(contacts, dim=1).to(torch.float32)


def wheel_contact_force_low_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]),
    min_force: float = 80.0,
) -> torch.Tensor:
    """Penalize selected wheel contact forces below a small flat-ground support threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_norm = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
    return torch.mean(torch.clamp(float(min_force) - force_norm, min=0.0), dim=1)


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


def contact_force_diag_balance_penalty(
    env: ManagerBasedRLEnv,
    eps: float = 1.0e-6,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
) -> torch.Tensor:
    """Penalize diagonal support imbalance: ``(lf + rr) - (lr + rf)`` normalized by total support."""

    force_norm = _wheel_contact_force_lf_lr_rf_rr(env, sensor_cfg)
    diag_diff = (force_norm[:, 0] + force_norm[:, 3]) - (force_norm[:, 1] + force_norm[:, 2])
    total_force = force_norm.sum(dim=1) + float(eps)
    return torch.abs(diag_diff) / total_force


def contact_force_left_right_balance_penalty(
    env: ManagerBasedRLEnv,
    eps: float = 1.0e-6,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
) -> torch.Tensor:
    """Penalize left-right support imbalance normalized by total support."""

    force_norm = _wheel_contact_force_lf_lr_rf_rr(env, sensor_cfg)
    left_right_diff = (force_norm[:, 0] + force_norm[:, 1]) - (force_norm[:, 2] + force_norm[:, 3])
    total_force = force_norm.sum(dim=1) + float(eps)
    return torch.abs(left_right_diff) / total_force


def contact_force_min_support_penalty(
    env: ManagerBasedRLEnv,
    min_force_threshold: float = 50.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
) -> torch.Tensor:
    """Penalize wheels whose contact force falls below a conservative minimum support threshold."""

    force_norm = _wheel_contact_force_lf_lr_rf_rr(env, sensor_cfg)
    min_force = max(float(min_force_threshold), 1.0e-6)
    penalty = torch.relu(min_force - force_norm) / min_force
    return penalty.mean(dim=1)


def contact_force_min_ratio_support_penalty(
    env: ManagerBasedRLEnv,
    min_ratio_threshold: float = 0.10,
    eps: float = 1.0e-6,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
) -> torch.Tensor:
    """Penalize any wheel whose share of total support force falls below a minimum ratio."""

    _, _, force_ratio, _ = _wheel_contact_force_ratio_lf_lr_rf_rr(env, sensor_cfg, eps=eps)
    min_ratio = force_ratio.min(dim=1).values
    min_ratio_threshold = max(float(min_ratio_threshold), 1.0e-6)
    return torch.relu(min_ratio_threshold - min_ratio) / min_ratio_threshold


def contact_force_range_balance_penalty(
    env: ManagerBasedRLEnv,
    eps: float = 1.0e-6,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
) -> torch.Tensor:
    """Penalize large spread between the most and least loaded wheel."""

    force_norm = _wheel_contact_force_lf_lr_rf_rr(env, sensor_cfg)
    total_force = force_norm.sum(dim=1) + float(eps)
    return (force_norm.max(dim=1).values - force_norm.min(dim=1).values) / total_force
