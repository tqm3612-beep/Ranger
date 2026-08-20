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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

from .observations import (
    GOAL_HEADING_PREV_HEADING_ERROR_ATTR,
    SHORT_GOAL_PREV_HEADING_ERROR_ATTR,
    SHORT_GOAL_PREV_DISTANCE_ATTR,
    SHORT_GOAL_REACHED_ATTR,
    goal_heading_target_body,
    short_goal_target_body,
    speed_command,
    yaw_rate_command,
)

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


def short_goal_large_heading_common_mode_penalty(
    env: ManagerBasedRLEnv,
    heading_start: float = 0.70,
    heading_full: float = 1.05,
    action_name: str = "wheel_motor_csv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize translational wheel common-mode while a large heading correction is still required.

    Unlike state-only yaw/progress terms, this term assigns the cost directly to the wheel target
    issued at the current step. The gate is zero below heading_start and reaches one at
    heading_full. The common-mode is normalized by the wheel target velocity limit.
    """

    wheel_action_term = env.action_manager.get_term(action_name)
    _, _, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    heading_abs = torch.abs(heading_error)
    heading_gate = torch.clamp(
        (heading_abs - float(heading_start)) / max(float(heading_full) - float(heading_start), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    semantic_target = wheel_raw_to_semantic_lr_lf_rf_rr(wheel_action_term.velocity_target)
    common_mode = semantic_target.mean(dim=1)
    velocity_limit = max(float(getattr(wheel_action_term, "_velocity_limit", 1.0)), 1.0e-6)
    normalized_common = torch.abs(common_mode) / velocity_limit
    penalty = heading_gate * normalized_common
    return _short_goal_navigation_gate(env, dtype=penalty.dtype) * penalty


def short_goal_large_heading_turn_mode_prior_l1(
    env: ManagerBasedRLEnv,
    heading_start: float = 0.70,
    heading_full: float = 1.05,
    turn_gain: float = 0.75,
    action_name: str = "wheel_motor_csv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Require strong, correctly signed wheel turn-mode while heading error is large."""

    raw_target = _wheel_target_vel_lf_lr_rf_rr(env, action_name=action_name)
    semantic_target = wheel_raw_to_semantic_lf_lr_rf_rr(raw_target)
    wheel_action_term = env.action_manager.get_term(action_name)
    velocity_limit = max(float(getattr(wheel_action_term, "_velocity_limit", 1.0)), 1.0e-6)
    left_mean = semantic_target[:, :2].mean(dim=1) / velocity_limit
    right_mean = semantic_target[:, 2:].mean(dim=1) / velocity_limit
    actual_turn_mode = 0.5 * (right_mean - left_mean)
    _, _, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    heading_abs = torch.abs(heading_error)
    heading_gate = torch.clamp(
        (heading_abs - float(heading_start)) / max(float(heading_full) - float(heading_start), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    desired_turn_mode = float(turn_gain) * torch.sin(heading_error)
    penalty = heading_gate * torch.abs(actual_turn_mode - desired_turn_mode)
    return _short_goal_navigation_gate(env, dtype=penalty.dtype) * penalty


def short_goal_progress_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reduction in short-goal distance."""

    _, current_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    prev_goal_distance = getattr(env, SHORT_GOAL_PREV_DISTANCE_ATTR, None)
    if prev_goal_distance is None or prev_goal_distance.shape != (env.num_envs,):
        prev_goal_distance = current_distance.clone()
        setattr(env, SHORT_GOAL_PREV_DISTANCE_ATTR, prev_goal_distance)
    progress = prev_goal_distance - current_distance
    prev_goal_distance[:] = current_distance
    return progress


def _short_goal_navigation_gate(env: ManagerBasedRLEnv, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return one before the latched stop phase and zero after entering it."""

    stop_phase_active = getattr(env, "_short_goal_stop_phase_active", None)
    if stop_phase_active is None or stop_phase_active.shape != (env.num_envs,):
        return torch.ones((env.num_envs,), dtype=dtype, device=env.device)
    return (~stop_phase_active).to(dtype)


def _short_goal_stop_phase_gate(env: ManagerBasedRLEnv, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return one only after the short-goal stop phase has been latched."""

    stop_phase_active = getattr(env, "_short_goal_stop_phase_active", None)
    if stop_phase_active is None or stop_phase_active.shape != (env.num_envs,):
        return torch.zeros((env.num_envs,), dtype=dtype, device=env.device)
    return stop_phase_active.to(dtype)


def short_goal_progress_reward_heading_gated(
    env: ManagerBasedRLEnv,
    heading_error_threshold: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward distance reduction only when heading error is not too large."""

    progress = short_goal_progress_reward(env, asset_cfg=asset_cfg)
    _, _, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    active_mask = (torch.abs(heading_error) <= float(heading_error_threshold)).to(progress.dtype)
    return active_mask * progress


def short_goal_progress_reward_alignment_gated(
    env: ManagerBasedRLEnv,
    heading_deadband: float = 0.20,
    heading_full: float = 0.80,
    alignment_floor: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward distance reduction with a non-zero heading-alignment floor."""

    progress = short_goal_progress_reward(env, asset_cfg=asset_cfg)
    _, _, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    alignment_gate = torch.clamp(
        (float(heading_full) - torch.abs(heading_error))
        / max(float(heading_full) - float(heading_deadband), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    floor = min(max(float(alignment_floor), 0.0), 1.0)
    alignment_scale = floor + (1.0 - floor) * alignment_gate
    return _short_goal_navigation_gate(env, dtype=progress.dtype) * alignment_scale * progress


def short_goal_success_reward(
    env: ManagerBasedRLEnv,
    success_distance: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the first time each environment reaches the short goal."""

    _, current_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    goal_reached = getattr(env, SHORT_GOAL_REACHED_ATTR, None)
    if goal_reached is None or goal_reached.shape != (env.num_envs,):
        goal_reached = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
        setattr(env, SHORT_GOAL_REACHED_ATTR, goal_reached)
    newly_reached = (current_distance < float(success_distance)) & (~goal_reached)
    goal_reached |= newly_reached
    return newly_reached.to(torch.float32)


def short_goal_precision_reach_reward(
    env: ManagerBasedRLEnv,
    precision_distance: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Give a one-time episode-level bonus for reaching the precision radius."""

    newly_reached = short_goal_success_reward(
        env,
        success_distance=precision_distance,
        asset_cfg=asset_cfg,
    )
    # RewardManager multiplies terms by step_dt. Divide here so the configured
    # weight is the actual one-time episode bonus rather than a time-scaled value.
    return newly_reached / max(float(env.step_dt), 1.0e-6)


def short_goal_stopped_success_reward(
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
    """Reward settling inside the success radius, with or without capture override."""

    asset: Articulation = env.scene[asset_cfg.name]
    _, current_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
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
    stopped_now = (
        (current_distance < float(success_distance))
        & (base_xy_speed < float(max_xy_speed))
        & (yaw_rate_abs < float(max_yaw_rate))
        & posture_ok
    )
    stable_steps = getattr(env, "_short_goal_stop_phase_stable_steps", None)
    if stable_steps is None or stable_steps.shape != (env.num_envs,):
        stopped_success = stopped_now
    else:
        stopped_success = stopped_now & (stable_steps >= int(required_hold_steps))
    # RewardManager integrates every term with step_dt. This event occurs for one
    # step only, so divide by step_dt to make the configured weight a true
    # episode-level success bonus rather than a time-scaled continuous reward.
    return stopped_success.to(torch.float32) / max(float(env.step_dt), 1.0e-6)


def short_goal_stop_phase_xy_speed_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar base motion after the goal region has been entered."""

    asset: Articulation = env.scene[asset_cfg.name]
    base_xy_speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    return _short_goal_stop_phase_gate(env, dtype=base_xy_speed.dtype) * base_xy_speed


def _short_goal_stop_phase_linear_allowance(
    goal_distance: torch.Tensor,
    enter_distance: float,
    zero_distance: float,
    value_at_enter: float,
) -> torch.Tensor:
    """Return a linear allowance that reaches zero at the precision radius."""

    if float(enter_distance) <= float(zero_distance):
        raise ValueError("enter_distance must be greater than zero_distance.")
    distance_fraction = torch.clamp(
        (goal_distance - float(zero_distance)) / (float(enter_distance) - float(zero_distance)),
        min=0.0,
        max=1.0,
    )
    return max(float(value_at_enter), 0.0) * distance_fraction


def short_goal_stop_phase_xy_speed_envelope_penalty(
    env: ManagerBasedRLEnv,
    enter_distance: float = 0.50,
    zero_distance: float = 0.30,
    allowed_speed_at_enter: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar speed above a distance-dependent stop-phase envelope."""

    asset: Articulation = env.scene[asset_cfg.name]
    _, goal_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    base_xy_speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    allowance = _short_goal_stop_phase_linear_allowance(
        goal_distance,
        enter_distance=enter_distance,
        zero_distance=zero_distance,
        value_at_enter=allowed_speed_at_enter,
    )
    excess = torch.relu(base_xy_speed - allowance)
    scale = max(float(allowed_speed_at_enter), 1.0e-6)
    penalty = torch.square(excess / scale)
    return _short_goal_stop_phase_gate(env, dtype=penalty.dtype) * penalty


def short_goal_stop_phase_yaw_rate_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw motion after the goal region has been entered."""

    asset: Articulation = env.scene[asset_cfg.name]
    yaw_rate_abs = torch.abs(asset.data.root_ang_vel_b[:, 2])
    return _short_goal_stop_phase_gate(env, dtype=yaw_rate_abs.dtype) * yaw_rate_abs


def short_goal_stop_phase_yaw_rate_envelope_penalty(
    env: ManagerBasedRLEnv,
    enter_distance: float = 0.50,
    zero_distance: float = 0.30,
    allowed_yaw_rate_at_enter: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw rate above a distance-dependent stop-phase envelope."""

    asset: Articulation = env.scene[asset_cfg.name]
    _, goal_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    yaw_rate_abs = torch.abs(asset.data.root_ang_vel_b[:, 2])
    allowance = _short_goal_stop_phase_linear_allowance(
        goal_distance,
        enter_distance=enter_distance,
        zero_distance=zero_distance,
        value_at_enter=allowed_yaw_rate_at_enter,
    )
    excess = torch.relu(yaw_rate_abs - allowance)
    scale = max(float(allowed_yaw_rate_at_enter), 1.0e-6)
    penalty = torch.square(excess / scale)
    return _short_goal_stop_phase_gate(env, dtype=penalty.dtype) * penalty


def short_goal_stop_phase_roll_error_penalty(
    env: ManagerBasedRLEnv,
    reference_angle: float = 0.035,
    max_normalized_error: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize normalized squared roll error only during the latched stop phase."""

    if float(reference_angle) <= 0.0:
        raise ValueError("reference_angle must be positive.")
    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    normalized_error = torch.square(roll / float(reference_angle))
    normalized_error = torch.clamp(normalized_error, max=float(max_normalized_error))
    return _short_goal_stop_phase_gate(env, dtype=normalized_error.dtype) * normalized_error


def short_goal_stop_phase_pitch_error_penalty(
    env: ManagerBasedRLEnv,
    reference_angle: float = 0.035,
    max_normalized_error: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize normalized squared pitch error only during the latched stop phase."""

    if float(reference_angle) <= 0.0:
        raise ValueError("reference_angle must be positive.")
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    normalized_error = torch.square(pitch / float(reference_angle))
    normalized_error = torch.clamp(normalized_error, max=float(max_normalized_error))
    return _short_goal_stop_phase_gate(env, dtype=normalized_error.dtype) * normalized_error


def short_goal_stop_phase_wheel_target_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Penalize the policy's raw wheel request during the latched stop phase."""

    raw_policy_wheel_action = getattr(env, "_short_goal_policy_wheel_action_raw", None)
    if raw_policy_wheel_action is not None and raw_policy_wheel_action.shape == (env.num_envs, 4):
        target_abs_mean = torch.mean(torch.abs(raw_policy_wheel_action), dim=1)
    else:
        action_term = env.action_manager.get_term(action_name)
        velocity_limit = max(float(getattr(action_term, "_velocity_limit", 1.0)), 1.0e-6)
        target_abs_mean = torch.mean(torch.abs(action_term.velocity_target), dim=1) / velocity_limit
    return _short_goal_stop_phase_gate(env, dtype=target_abs_mean.dtype) * target_abs_mean


def short_goal_stop_phase_wheel_target_envelope_penalty(
    env: ManagerBasedRLEnv,
    enter_distance: float = 0.50,
    zero_distance: float = 0.30,
    allowed_action_at_enter: float = 0.20,
    action_name: str = "wheel_motor_csv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize wheel requests above an allowance that shrinks to zero near the target."""

    _, goal_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    raw_policy_wheel_action = getattr(env, "_short_goal_policy_wheel_action_raw", None)
    if raw_policy_wheel_action is not None and raw_policy_wheel_action.shape == (env.num_envs, 4):
        target_abs_mean = torch.mean(torch.abs(raw_policy_wheel_action), dim=1)
    else:
        action_term = env.action_manager.get_term(action_name)
        velocity_limit = max(float(getattr(action_term, "_velocity_limit", 1.0)), 1.0e-6)
        target_abs_mean = torch.mean(torch.abs(action_term.velocity_target), dim=1) / velocity_limit
    allowance = _short_goal_stop_phase_linear_allowance(
        goal_distance,
        enter_distance=enter_distance,
        zero_distance=zero_distance,
        value_at_enter=allowed_action_at_enter,
    )
    excess = torch.relu(target_abs_mean - allowance)
    scale = max(float(allowed_action_at_enter), 1.0e-6)
    penalty = torch.square(excess / scale)
    return _short_goal_stop_phase_gate(env, dtype=penalty.dtype) * penalty


def short_goal_stop_phase_precision_closeness_reward(
    env: ManagerBasedRLEnv,
    enter_distance: float = 0.50,
    precision_distance: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward continuous closeness from the stop boundary to the precision radius."""

    if float(enter_distance) <= float(precision_distance):
        raise ValueError("enter_distance must be greater than precision_distance.")
    _, goal_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    closeness = torch.clamp(
        (float(enter_distance) - goal_distance) / (float(enter_distance) - float(precision_distance)),
        min=0.0,
        max=1.0,
    )
    return _short_goal_stop_phase_gate(env, dtype=closeness.dtype) * closeness


def short_goal_stop_phase_distance_drift_penalty(
    env: ManagerBasedRLEnv,
    enter_distance: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize moving away from the target after entering the stop phase."""

    _, current_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    drift = torch.relu(current_distance - float(enter_distance)) / max(float(enter_distance), 1.0e-6)
    return _short_goal_stop_phase_gate(env, dtype=drift.dtype) * drift


def failure_termination_penalty(
    env: ManagerBasedRLEnv,
    excluded_terms: tuple[str, ...] = ("time_out", "stopped_goal_reached"),
) -> torch.Tensor:
    """Return one only for undesired termination terms, excluding timeout and success."""

    failed = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    active_terms = tuple(getattr(env.termination_manager, "active_terms", ()))
    term_dones = getattr(env.termination_manager, "_term_dones", None)
    if term_dones is None:
        return failed.to(torch.float32)
    excluded = set(excluded_terms)
    for term_idx, term_name in enumerate(active_terms):
        if term_name in excluded:
            continue
        failed |= term_dones[:, term_idx]
    return failed.to(torch.float32)


def short_goal_near_stop_penalty(
    env: ManagerBasedRLEnv,
    stop_distance: float = 0.5,
    yaw_weight: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize residual planar speed and yaw rate when already near the goal."""

    asset: Articulation = env.scene[asset_cfg.name]
    _, current_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    near_mask = (current_distance < float(stop_distance)).to(torch.float32)
    base_xy_speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
    return near_mask * (base_xy_speed + float(yaw_weight) * yaw_rate)


def short_goal_heading_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Lightly reward facing toward the short goal."""

    _, _, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    return torch.cos(heading_error)


def short_goal_velocity_towards_target(
    env: ManagerBasedRLEnv,
    max_velocity: float = 1.0,
    min_reward: float = -1.0,
    max_reward: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward planar body velocity projected onto the current short-goal direction."""

    asset: Articulation = env.scene[asset_cfg.name]
    target_vec_b, distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    target_dir_b = target_vec_b[:, :2] / torch.clamp(distance.unsqueeze(1), min=1.0e-6)
    velocity_towards_target = torch.sum(asset.data.root_lin_vel_b[:, :2] * target_dir_b, dim=1)
    normalized_velocity = velocity_towards_target / max(float(max_velocity), 1.0e-6)
    return torch.clamp(normalized_velocity, min=float(min_reward), max=float(max_reward))


def short_goal_velocity_towards_target_heading_gated(
    env: ManagerBasedRLEnv,
    max_velocity: float = 1.0,
    min_reward: float = -1.0,
    max_reward: float = 1.5,
    heading_error_threshold: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward target-direction velocity only when the body is roughly facing the goal."""

    velocity_reward = short_goal_velocity_towards_target(
        env,
        max_velocity=max_velocity,
        min_reward=min_reward,
        max_reward=max_reward,
        asset_cfg=asset_cfg,
    )
    _, _, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    active_mask = (torch.abs(heading_error) <= float(heading_error_threshold)).to(velocity_reward.dtype)
    return active_mask * velocity_reward


def _short_goal_braking_desired_speed(
    goal_distance: torch.Tensor,
    heading_error: torch.Tensor,
    stop_distance: float,
    braking_acceleration: float,
    max_speed: float,
    heading_deadband: float,
    heading_full: float,
    alignment_floor: float = 0.30,
) -> torch.Tensor:
    """Return a distance-based speed envelope without allowing heading error to force zero speed."""

    remaining_distance = torch.relu(goal_distance - float(stop_distance))
    desired_speed = torch.sqrt(2.0 * max(float(braking_acceleration), 1.0e-6) * remaining_distance)
    desired_speed = torch.clamp(desired_speed, max=float(max_speed))
    heading_abs = torch.abs(heading_error)
    alignment_gate = torch.clamp(
        (float(heading_full) - heading_abs) / max(float(heading_full) - float(heading_deadband), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    floor = min(max(float(alignment_floor), 0.0), 1.0)
    return desired_speed * (floor + (1.0 - floor) * alignment_gate)


def short_goal_speed_profile_penalty(
    env: ManagerBasedRLEnv,
    stop_distance: float = 0.30,
    braking_acceleration: float = 0.60,
    reaction_time: float = 0.20,
    braking_margin: float = 0.08,
    near_distance: float = 1.0,
    yaw_rate_ref: float = 0.80,
    yaw_component_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only when the robot cannot stop inside the captured goal region."""

    asset: Articulation = env.scene[asset_cfg.name]
    target_vec_b, goal_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    target_dir_b = target_vec_b[:, :2] / torch.clamp(goal_distance.unsqueeze(1), min=1.0e-6)
    velocity_toward_goal = torch.sum(asset.data.root_lin_vel_b[:, :2] * target_dir_b, dim=1)
    positive_velocity = torch.relu(velocity_toward_goal)
    yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])

    braking_accel = max(float(braking_acceleration), 1.0e-6)
    # The vehicle may cross the capture boundary with finite speed and brake
    # inside the goal region. Requiring zero speed exactly at stop_distance
    # creates the 0.30-0.60 m low-speed dead zone seen in the previous run.
    available_braking_distance = torch.relu(goal_distance)
    required_braking_distance = (
        torch.square(positive_velocity) / (2.0 * braking_accel)
        + max(float(reaction_time), 0.0) * positive_velocity
        + max(float(braking_margin), 0.0)
    )
    braking_excess = torch.relu(required_braking_distance - available_braking_distance)
    linear_penalty = torch.square(braking_excess / max(float(near_distance), 1.0e-6))
    near_gate = torch.clamp(
        (float(near_distance) - goal_distance) / max(float(near_distance) - float(stop_distance), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    yaw_penalty = near_gate * torch.square(yaw_rate / max(float(yaw_rate_ref), 1.0e-6))
    penalty = linear_penalty + float(yaw_component_weight) * yaw_penalty
    return _short_goal_navigation_gate(env, dtype=penalty.dtype) * penalty


def short_goal_cruise_underspeed_penalty(
    env: ManagerBasedRLEnv,
    capture_distance: float = 0.30,
    approach_full_distance: float = 0.60,
    cruise_full_distance: float = 1.20,
    approach_speed: float = 0.35,
    cruise_speed: float = 0.80,
    heading_deadband: float = 0.20,
    heading_full: float = 0.80,
    alignment_floor: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize stopping outside the capture radius with a piecewise speed floor."""

    asset: Articulation = env.scene[asset_cfg.name]
    target_vec_b, goal_distance, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    target_dir_b = target_vec_b[:, :2] / torch.clamp(goal_distance.unsqueeze(1), min=1.0e-6)
    velocity_toward_goal = torch.sum(asset.data.root_lin_vel_b[:, :2] * target_dir_b, dim=1)

    heading_abs = torch.abs(heading_error)
    alignment_gate = torch.clamp(
        (float(heading_full) - heading_abs) / max(float(heading_full) - float(heading_deadband), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    floor = min(max(float(alignment_floor), 0.0), 1.0)
    blend = torch.clamp(
        (goal_distance - float(approach_full_distance))
        / max(float(cruise_full_distance) - float(approach_full_distance), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    base_reference_speed = float(approach_speed) + blend * (float(cruise_speed) - float(approach_speed))
    reference_speed = base_reference_speed * (floor + (1.0 - floor) * alignment_gate)
    underspeed = torch.relu(reference_speed - velocity_toward_goal)
    outside_capture = (goal_distance > float(capture_distance)).to(underspeed.dtype)
    penalty = outside_capture * torch.square(underspeed / max(float(cruise_speed), 1.0e-6))
    return _short_goal_navigation_gate(env, dtype=penalty.dtype) * penalty


def short_goal_near_goal_away_speed_penalty(
    env: ManagerBasedRLEnv,
    stop_distance: float = 0.30,
    active_distance: float = 1.0,
    speed_reference: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize moving away from the target near the goal, independent of drive direction."""

    asset: Articulation = env.scene[asset_cfg.name]
    target_vec_b, goal_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    target_dir_b = target_vec_b[:, :2] / torch.clamp(goal_distance.unsqueeze(1), min=1.0e-6)
    velocity_toward_goal = torch.sum(asset.data.root_lin_vel_b[:, :2] * target_dir_b, dim=1)
    near_gate = torch.clamp(
        (float(active_distance) - goal_distance)
        / max(float(active_distance) - float(stop_distance), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    away_speed = torch.relu(-velocity_toward_goal)
    penalty = near_gate * torch.square(away_speed / max(float(speed_reference), 1.0e-6))
    return _short_goal_navigation_gate(env, dtype=penalty.dtype) * penalty


def short_goal_approach_wheel_target_excess_penalty(
    env: ManagerBasedRLEnv,
    stop_distance: float = 0.25,
    braking_acceleration: float = 0.6,
    max_speed: float = 1.0,
    approach_distance: float = 1.0,
    wheel_radius: float = 0.2024,
    heading_deadband: float = 0.20,
    heading_full: float = 0.60,
    action_name: str = "wheel_motor_csv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize wheel targets that exceed the braking-profile speed in the approach region."""

    _, goal_distance, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    desired_speed = _short_goal_braking_desired_speed(
        goal_distance,
        heading_error,
        stop_distance=stop_distance,
        braking_acceleration=braking_acceleration,
        max_speed=max_speed,
        heading_deadband=heading_deadband,
        heading_full=heading_full,
    )
    action_term = env.action_manager.get_term(action_name)
    semantic_target = wheel_raw_to_semantic_lf_lr_rf_rr(action_term.velocity_target)
    velocity_limit = max(float(getattr(action_term, "_velocity_limit", 1.0)), 1.0e-6)
    allowed_wheel_speed = desired_speed / max(float(wheel_radius), 1.0e-6)
    target_excess = torch.relu(torch.abs(semantic_target) - allowed_wheel_speed.unsqueeze(1))
    approach_gate = (goal_distance < float(approach_distance)).to(target_excess.dtype)
    penalty = approach_gate * torch.mean(target_excess, dim=1) / velocity_limit
    return _short_goal_navigation_gate(env, dtype=penalty.dtype) * penalty


def short_goal_heading_error_reduction(
    env: ManagerBasedRLEnv,
    min_progress: float = -0.5,
    max_progress: float = 0.5,
    use_turn_distance_gate: bool = False,
    turn_gate_start_distance: float = 0.35,
    turn_gate_full_distance: float = 0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward step-wise reduction in absolute heading error for short-goal turning."""

    _, goal_distance, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    prev_heading_error = getattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, None)
    if prev_heading_error is None or prev_heading_error.shape != (env.num_envs,):
        prev_heading_error = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
        setattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, prev_heading_error)

    prev_abs_error = torch.abs(prev_heading_error)
    current_abs_error = torch.abs(heading_error)
    progress = torch.where(
        torch.isfinite(prev_abs_error),
        prev_abs_error - current_abs_error,
        torch.zeros_like(current_abs_error),
    )
    prev_heading_error[:] = heading_error
    progress = torch.clamp(progress, min=float(min_progress), max=float(max_progress))
    if use_turn_distance_gate:
        progress = progress * _short_goal_turn_distance_gate(
            goal_distance,
            start_distance=turn_gate_start_distance,
            full_distance=turn_gate_full_distance,
        )
    return _short_goal_navigation_gate(env, dtype=progress.dtype) * progress


def short_goal_heading_error_cost(
    env: ManagerBasedRLEnv,
    fade_start_distance: float = 0.50,
    full_distance: float = 0.80,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a smooth heading cost that fades out near the position target."""

    _, goal_distance, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    distance_gate = _short_goal_turn_distance_gate(
        goal_distance,
        start_distance=fade_start_distance,
        full_distance=full_distance,
    )
    cost = distance_gate * (1.0 - torch.cos(heading_error))
    return _short_goal_navigation_gate(env, dtype=cost.dtype) * cost


def short_goal_turn_toward_goal(
    env: ManagerBasedRLEnv,
    min_reward: float = -1.0,
    max_reward: float = 1.0,
    use_turn_distance_gate: bool = False,
    turn_gate_start_distance: float = 0.35,
    turn_gate_full_distance: float = 0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw-rate direction that turns toward the side goal."""

    asset: Articulation = env.scene[asset_cfg.name]
    target_vec_b, goal_distance, _ = short_goal_target_body(env, asset_cfg=asset_cfg)
    turn_direction = torch.sign(target_vec_b[:, 1]) * asset.data.root_ang_vel_b[:, 2]
    reward = torch.clamp(turn_direction, min=float(min_reward), max=float(max_reward))
    if use_turn_distance_gate:
        reward = reward * _short_goal_turn_distance_gate(
            goal_distance,
            start_distance=turn_gate_start_distance,
            full_distance=turn_gate_full_distance,
        )
    return reward


def short_goal_continuous_yaw_rate_tracking_penalty(
    env: ManagerBasedRLEnv,
    yaw_rate_max: float = 0.45,
    heading_deadband: float = 0.04,
    heading_scale: float = 0.30,
    yaw_rate_reference: float = 0.35,
    max_normalized_error: float = 2.0,
    use_turn_distance_gate: bool = False,
    turn_gate_start_distance: float = 0.50,
    turn_gate_full_distance: float = 0.80,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track a smooth signed yaw-rate reference that vanishes as heading error closes.

    Large heading errors request a near-saturated yaw rate, intermediate errors
    request a proportionally smaller rate, and errors inside the deadband request
    zero yaw rate. The returned value is a non-negative penalty.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    _, goal_distance, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    heading_abs = torch.abs(heading_error)
    effective_error = torch.sign(heading_error) * torch.relu(
        heading_abs - max(float(heading_deadband), 0.0)
    )
    desired_yaw_rate = float(yaw_rate_max) * torch.tanh(
        effective_error / max(float(heading_scale), 1.0e-6)
    )
    actual_yaw_rate = asset.data.root_ang_vel_b[:, 2]
    normalized_error = torch.abs(actual_yaw_rate - desired_yaw_rate) / max(
        float(yaw_rate_reference), 1.0e-6
    )
    normalized_error = torch.clamp(
        normalized_error,
        max=max(float(max_normalized_error), 0.0),
    )
    penalty = torch.square(normalized_error)
    if use_turn_distance_gate:
        penalty = penalty * _short_goal_turn_distance_gate(
            goal_distance,
            start_distance=turn_gate_start_distance,
            full_distance=turn_gate_full_distance,
        )
    return _short_goal_navigation_gate(env, dtype=penalty.dtype) * penalty


def _short_goal_turn_distance_gate(
    goal_distance: torch.Tensor,
    start_distance: float = 0.35,
    full_distance: float = 0.60,
) -> torch.Tensor:
    """Return a near-goal gate that fades turn-only shaping out near the target."""

    return torch.clamp(
        (goal_distance - float(start_distance)) / max(float(full_distance) - float(start_distance), 1.0e-6),
        min=0.0,
        max=1.0,
    )


def short_goal_wheel_diff_prior_l1(
    env: ManagerBasedRLEnv,
    turn_gain: float = 20.0,
    action_name: str = "wheel_motor_csv",
    use_turn_distance_gate: bool = False,
    turn_gate_start_distance: float = 0.35,
    turn_gate_full_distance: float = 0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize semantic left-right wheel target mismatch for the short-goal heading error."""

    raw_target = _wheel_target_vel_lf_lr_rf_rr(env, action_name=action_name)
    semantic_target = wheel_raw_to_semantic_lf_lr_rf_rr(raw_target)
    left_mean = semantic_target[:, :2].mean(dim=1)
    right_mean = semantic_target[:, 2:].mean(dim=1)
    _, goal_distance, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    expected_diff = -2.0 * float(turn_gain) * torch.sin(heading_error)
    actual_diff = left_mean - right_mean
    error = torch.abs(actual_diff - expected_diff)
    wheel_action_term = env.action_manager.get_term(action_name)
    velocity_limit = float(getattr(wheel_action_term, "_velocity_limit", 1.0))
    penalty = error / max(2.0 * velocity_limit, 1.0)
    if use_turn_distance_gate:
        penalty = penalty * _short_goal_turn_distance_gate(
            goal_distance,
            start_distance=turn_gate_start_distance,
            full_distance=turn_gate_full_distance,
        )
    return penalty


def short_goal_wheel_turn_mode_prior_l1(
    env: ManagerBasedRLEnv,
    turn_gain: float = 0.5,
    action_name: str = "wheel_motor_csv",
    use_turn_distance_gate: bool = False,
    turn_gate_start_distance: float = 0.35,
    turn_gate_full_distance: float = 0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize normalized wheel turn-mode error implied by the target heading."""

    raw_target = _wheel_target_vel_lf_lr_rf_rr(env, action_name=action_name)
    semantic_target = wheel_raw_to_semantic_lf_lr_rf_rr(raw_target)
    wheel_action_term = env.action_manager.get_term(action_name)
    velocity_limit = max(float(getattr(wheel_action_term, "_velocity_limit", 1.0)), 1.0e-6)
    left_mean = semantic_target[:, :2].mean(dim=1) / velocity_limit
    right_mean = semantic_target[:, 2:].mean(dim=1) / velocity_limit
    actual_turn_mode = 0.5 * (right_mean - left_mean)
    _, goal_distance, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    desired_turn_mode = float(turn_gain) * torch.sin(heading_error)
    penalty = torch.abs(actual_turn_mode - desired_turn_mode)
    if use_turn_distance_gate:
        penalty = penalty * _short_goal_turn_distance_gate(
            goal_distance,
            start_distance=turn_gate_start_distance,
            full_distance=turn_gate_full_distance,
        )
    return _short_goal_navigation_gate(env, dtype=penalty.dtype) * penalty


def wheel_same_side_target_consistency_l1(
    env: ManagerBasedRLEnv,
    deadband: float = 0.0,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Penalize excessive front/rear target disagreement while allowing a normalized deadband."""

    raw_target = _wheel_target_vel_lf_lr_rf_rr(env, action_name=action_name)
    semantic_target = wheel_raw_to_semantic_lf_lr_rf_rr(raw_target)
    wheel_action_term = env.action_manager.get_term(action_name)
    velocity_limit = max(float(getattr(wheel_action_term, "_velocity_limit", 1.0)), 1.0e-6)
    left_diff_norm = torch.abs(semantic_target[:, 0] - semantic_target[:, 1]) / velocity_limit
    right_diff_norm = torch.abs(semantic_target[:, 2] - semantic_target[:, 3]) / velocity_limit
    left_excess = torch.relu(left_diff_norm - float(deadband))
    right_excess = torch.relu(right_diff_norm - float(deadband))
    return 0.5 * (left_excess + right_excess)


def wheel_same_side_opposite_sign_penalty(
    env: ManagerBasedRLEnv,
    margin: float = 0.03,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Penalize the normalized same-side component that can only exist under sign opposition."""

    raw_target = _wheel_target_vel_lf_lr_rf_rr(env, action_name=action_name)
    semantic_target = wheel_raw_to_semantic_lf_lr_rf_rr(raw_target)
    wheel_action_term = env.action_manager.get_term(action_name)
    velocity_limit = max(float(getattr(wheel_action_term, "_velocity_limit", 1.0)), 1.0e-6)
    target_norm = semantic_target / velocity_limit

    left_front, left_rear = target_norm[:, 0], target_norm[:, 1]
    right_front, right_rear = target_norm[:, 2], target_norm[:, 3]
    left_mean = 0.5 * (left_front + left_rear)
    right_mean = 0.5 * (right_front + right_rear)
    left_antisymmetric = 0.5 * (left_front - left_rear)
    right_antisymmetric = 0.5 * (right_front - right_rear)
    margin_value = max(float(margin), 0.0)
    left_opposition = torch.relu(torch.abs(left_antisymmetric) - torch.abs(left_mean) - margin_value)
    right_opposition = torch.relu(torch.abs(right_antisymmetric) - torch.abs(right_mean) - margin_value)
    return 0.5 * (left_opposition + right_opposition)


def loaded_wheel_longitudinal_slip_penalty(
    env: ManagerBasedRLEnv,
    wheel_radius: float = 0.2024,
    min_contact_force: float = 20.0,
    min_motion_speed: float = 0.20,
    absolute_margin: float = 0.10,
    relative_margin: float = 0.15,
    excess_speed_reference: float = 0.50,
    max_normalized_excess: float = 2.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]
    ),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize loaded-wheel overspeed without penalizing normal rolling or braking mismatch.

    The term is zero while the wheel-surface speed stays within an absolute plus
    speed-proportional margin above the corresponding wheel-hub forward speed.
    Only positive wheel overspeed is penalized, so a stationary policy must still
    be rejected by the separate progress and cruise-underspeed objectives.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_joint_vel = _wheel_joint_vel_lf_lr_rf_rr(
        env,
        SceneEntityCfg("robot", joint_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
    )
    semantic_wheel_joint_vel = wheel_raw_to_semantic_lf_lr_rf_rr(wheel_joint_vel)

    wheel_body_ids, _ = asset.find_bodies(["w_lf", "w_lb", "w_rf", "w_rb"], preserve_order=True)
    wheel_hub_lin_vel_w = asset.data.body_lin_vel_w[:, wheel_body_ids, :]
    root_forward_axis_b = torch.zeros((env.num_envs, 3), dtype=wheel_joint_vel.dtype, device=env.device)
    root_forward_axis_b[:, 0] = 1.0
    root_forward_axis_w = quat_apply(asset.data.root_quat_w, root_forward_axis_b)
    wheel_hub_forward_speed = torch.sum(wheel_hub_lin_vel_w * root_forward_axis_w.unsqueeze(1), dim=-1)

    wheel_surface_speed_abs = torch.abs(semantic_wheel_joint_vel * float(wheel_radius))
    wheel_hub_forward_speed_abs = torch.abs(wheel_hub_forward_speed)
    motion_scale = torch.maximum(wheel_surface_speed_abs, wheel_hub_forward_speed_abs)
    contact_force = _wheel_contact_force_lf_lr_rf_rr(env, sensor_cfg)
    active = (contact_force > float(min_contact_force)) & (motion_scale > float(min_motion_speed))

    allowed_overspeed = max(float(absolute_margin), 0.0) + max(float(relative_margin), 0.0) * wheel_hub_forward_speed_abs
    overspeed_excess = torch.relu(wheel_surface_speed_abs - wheel_hub_forward_speed_abs - allowed_overspeed)
    normalized_excess = overspeed_excess / max(float(excess_speed_reference), 1.0e-6)
    normalized_excess = torch.clamp(
        normalized_excess,
        max=max(float(max_normalized_excess), 0.0),
    )
    active_f = active.to(normalized_excess.dtype)
    penalty = torch.sum(torch.square(normalized_excess) * active_f, dim=1) / torch.clamp(
        active_f.sum(dim=1), min=1.0
    )
    return _short_goal_navigation_gate(env, dtype=penalty.dtype) * penalty


def wheel_target_rate_l1(
    env: ManagerBasedRLEnv,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Penalize per-step wheel target changes in normalized target units."""

    action_term = env.action_manager.get_term(action_name)
    velocity_limit = max(float(getattr(action_term, "_velocity_limit", 1.0)), 1.0e-6)
    previous_target = getattr(action_term, "previous_velocity_target", None)
    if previous_target is None:
        return torch.zeros((env.num_envs,), dtype=action_term.velocity_target.dtype, device=env.device)
    rate = torch.mean(torch.abs(action_term.velocity_target - previous_target), dim=1) / velocity_limit
    # Do not penalize the abrupt target reduction needed when entering the stop phase.
    return _short_goal_navigation_gate(env, dtype=rate.dtype) * rate


def short_goal_forward_common_mode_penalty(
    env: ManagerBasedRLEnv,
    heading_error_threshold: float = 0.35,
    action_name: str = "wheel_motor_csv",
    use_turn_distance_gate: bool = False,
    turn_gate_start_distance: float = 0.35,
    turn_gate_full_distance: float = 0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize shared forward wheel target motion while the short-goal heading error is large."""

    raw_target = _wheel_target_vel_lf_lr_rf_rr(env, action_name=action_name)
    semantic_target = wheel_raw_to_semantic_lf_lr_rf_rr(raw_target)
    left_mean = semantic_target[:, :2].mean(dim=1)
    right_mean = semantic_target[:, 2:].mean(dim=1)
    _, goal_distance, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    active = (torch.abs(heading_error) > float(heading_error_threshold)).to(semantic_target.dtype)
    common_mode = torch.abs(0.5 * (left_mean + right_mean))
    wheel_action_term = env.action_manager.get_term(action_name)
    velocity_limit = float(getattr(wheel_action_term, "_velocity_limit", 1.0))
    penalty = active * common_mode / max(velocity_limit, 1.0)
    if use_turn_distance_gate:
        penalty = penalty * _short_goal_turn_distance_gate(
            goal_distance,
            start_distance=turn_gate_start_distance,
            full_distance=turn_gate_full_distance,
        )
    return _short_goal_navigation_gate(env, dtype=penalty.dtype) * penalty


def _short_goal_signed_yaw_targets(
    env: ManagerBasedRLEnv,
    target_yaw_rate: float = 0.35,
    heading_deadband: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return desired signed yaw-rate signals for short-goal turning tasks."""

    asset: Articulation = env.scene[asset_cfg.name]
    target_vec_b, _, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
    desired_sign = torch.sign(target_vec_b[:, 1])
    heading_abs = torch.abs(heading_error)
    active_mask = heading_abs > float(heading_deadband)
    desired_yaw_rate = torch.where(
        active_mask,
        desired_sign * float(target_yaw_rate),
        torch.zeros_like(desired_sign),
    )
    base_yaw_rate = asset.data.root_ang_vel_b[:, 2]
    return desired_yaw_rate, desired_sign, active_mask, base_yaw_rate


def short_goal_signed_yaw_rate_tracking(
    env: ManagerBasedRLEnv,
    target_yaw_rate: float = 0.35,
    heading_deadband: float = 0.10,
    sigma: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bounded positive reward for tracking the signed yaw-rate implied by the goal side."""

    desired_yaw_rate, desired_sign, active_mask, base_yaw_rate = _short_goal_signed_yaw_targets(
        env,
        target_yaw_rate=target_yaw_rate,
        heading_deadband=heading_deadband,
        asset_cfg=asset_cfg,
    )
    sigma_sq = max(float(sigma) * float(sigma), 1.0e-6)
    signed_yaw = desired_sign * base_yaw_rate
    directional_tracking = torch.exp(-torch.square(signed_yaw - float(target_yaw_rate)) / sigma_sq)
    tracking = torch.where(signed_yaw > 0.0, directional_tracking, torch.zeros_like(directional_tracking))
    inactive_tracking = torch.exp(-torch.square(base_yaw_rate) / sigma_sq)
    return torch.where(active_mask, tracking, inactive_tracking)


def short_goal_too_small_yaw_rate_when_error_large_penalty(
    env: ManagerBasedRLEnv,
    min_yaw_rate: float = 0.12,
    heading_threshold: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the low-motion local optimum when heading error is still large."""

    desired_yaw_rate, desired_sign, active_mask, base_yaw_rate = _short_goal_signed_yaw_targets(
        env,
        target_yaw_rate=min_yaw_rate,
        heading_deadband=heading_threshold,
        asset_cfg=asset_cfg,
    )
    del desired_yaw_rate
    signed_yaw = desired_sign * base_yaw_rate
    wrong_direction = signed_yaw < 0.0
    too_small = signed_yaw < float(min_yaw_rate)
    active = active_mask & (wrong_direction | too_small)
    penalty = torch.relu(float(min_yaw_rate) - signed_yaw) / max(float(min_yaw_rate), 1.0e-6)
    return torch.where(active, torch.clamp(penalty, max=2.0), torch.zeros_like(penalty))


def short_goal_wrong_direction_yaw_penalty(
    env: ManagerBasedRLEnv,
    heading_deadband: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw-rate whose sign turns away from the side goal."""

    desired_yaw_rate, desired_sign, active_mask, base_yaw_rate = _short_goal_signed_yaw_targets(
        env,
        target_yaw_rate=0.0,
        heading_deadband=heading_deadband,
        asset_cfg=asset_cfg,
    )
    del desired_yaw_rate
    wrong_direction = torch.relu(-desired_sign * base_yaw_rate)
    return torch.where(active_mask, wrong_direction, torch.zeros_like(wrong_direction))


def _yaw_turn_support_free_mode_gate(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Return 1 only for YawTurnSupportFlat free-hydraulic mode, else 0."""

    is_yaw_turn_support = bool(getattr(env, "_is_yaw_turn_support_task", False))
    mode = str(getattr(env, "_yaw_turn_support_hydraulic_mode", "")).strip().lower()
    gate_value = 1.0 if is_yaw_turn_support and mode == "free" else 0.0
    return torch.full((env.num_envs,), gate_value, device=env.device, dtype=torch.float32)


def yaw_turn_free_mode_hydraulic_action_magnitude_l1(
    env: ManagerBasedRLEnv,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Mode-gated hydraulic action magnitude penalty for free-hydraulic yaw-turn ablations."""

    return _yaw_turn_support_free_mode_gate(env) * hydraulic_action_magnitude_l1(env, action_name=action_name)


def yaw_turn_free_mode_hydraulic_action_rate_l1(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Mode-gated L1 hydraulic action-rate penalty using the first four action dimensions."""

    delta = torch.abs(env.action_manager.action[:, :4] - env.action_manager.prev_action[:, :4])
    return _yaw_turn_support_free_mode_gate(env) * torch.mean(delta, dim=1)


def yaw_turn_free_mode_hydraulic_action_range_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Mode-gated per-leg hydraulic action spread penalty."""

    return _yaw_turn_support_free_mode_gate(env) * hydraulic_action_range_penalty(env, action_name=action_name)


def yaw_turn_free_mode_actual_stroke_nominal_l2(
    env: ManagerBasedRLEnv,
    stroke_nominal: float = 0.5,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Mode-gated nominal stroke regularizer."""

    return _yaw_turn_support_free_mode_gate(env) * actual_stroke_nominal_l2(
        env, stroke_nominal=stroke_nominal, action_name=action_name
    )


def yaw_turn_free_mode_actual_stroke_soft_limit_penalty(
    env: ManagerBasedRLEnv,
    limit: float = 0.55,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Mode-gated soft-limit regularizer."""

    return _yaw_turn_support_free_mode_gate(env) * actual_stroke_soft_limit_penalty(
        env, limit=limit, action_name=action_name
    )


def yaw_turn_wheel_common_mode_target_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Penalize shared left/right wheel target motion while allowing differential steering."""

    return wheel_target_common_mode_penalty(env, action_name=action_name)


def yaw_rate_command_tracking_reward(
    env: ManagerBasedRLEnv,
    active_threshold: float = 0.03,
    sigma: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Direction-gated yaw-rate command tracking reward."""

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = yaw_rate_command(env)
    yaw = asset.data.root_ang_vel_b[:, 2]
    active = torch.abs(cmd) > float(active_threshold)
    desired_sign = torch.sign(cmd)
    signed_yaw = desired_sign * yaw
    sigma_sq = max(float(sigma) * float(sigma), 1.0e-6)
    active_reward = torch.where(
        signed_yaw > 0.0,
        torch.exp(-torch.square(yaw - cmd) / sigma_sq),
        torch.zeros_like(yaw),
    )
    inactive_reward = torch.exp(-torch.square(yaw) / sigma_sq)
    return torch.where(active, active_reward, inactive_reward)


def wrong_direction_yaw_command_penalty(
    env: ManagerBasedRLEnv,
    active_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw-rate that rotates opposite to the command sign."""

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = yaw_rate_command(env)
    yaw = asset.data.root_ang_vel_b[:, 2]
    active = torch.abs(cmd) > float(active_threshold)
    penalty = torch.relu(-(torch.sign(cmd) * yaw))
    return torch.where(active, penalty, torch.zeros_like(penalty))


def too_small_yaw_rate_command_penalty(
    env: ManagerBasedRLEnv,
    active_threshold: float = 0.10,
    min_yaw_rate_min: float = 0.08,
    min_yaw_rate_max: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize too-small signed yaw when a non-trivial yaw-rate command is active."""

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = yaw_rate_command(env)
    yaw = asset.data.root_ang_vel_b[:, 2]
    active = torch.abs(cmd) > float(active_threshold)
    desired_sign = torch.sign(cmd)
    signed_yaw = desired_sign * yaw
    min_yaw = torch.clamp(
        0.5 * torch.abs(cmd),
        min=float(min_yaw_rate_min),
        max=float(min_yaw_rate_max),
    )
    penalty = torch.relu(min_yaw - signed_yaw) / torch.clamp(min_yaw, min=1.0e-6)
    penalty = torch.clamp(penalty, min=0.0, max=2.0)
    return torch.where(active, penalty, torch.zeros_like(penalty))


def wheel_forward_mode_target_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "wheel_motor_csv",
) -> torch.Tensor:
    """Penalize semantic wheel forward-mode target magnitude."""

    raw_target = _wheel_target_vel_lf_lr_rf_rr(env, action_name=action_name)
    semantic_target = wheel_raw_to_semantic_lf_lr_rf_rr(raw_target)
    semantic_left_forward_target = semantic_target[:, :2].mean(dim=1)
    semantic_right_forward_target = semantic_target[:, 2:].mean(dim=1)
    semantic_forward_mode = 0.5 * (semantic_left_forward_target + semantic_right_forward_target)
    action_term = env.action_manager.get_term(action_name)
    velocity_limit = float(getattr(action_term, "_velocity_limit", 1.0))
    return torch.abs(semantic_forward_mode) / max(velocity_limit, 1.0)


def wheel_turn_mode_target_soft_limit_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "wheel_motor_csv",
    soft_limit: float = 8.0,
) -> torch.Tensor:
    """Penalize excessive semantic differential turn mode target magnitude."""

    raw_target = _wheel_target_vel_lf_lr_rf_rr(env, action_name=action_name)
    semantic_target = wheel_raw_to_semantic_lf_lr_rf_rr(raw_target)
    semantic_left_forward_target = semantic_target[:, :2].mean(dim=1)
    semantic_right_forward_target = semantic_target[:, 2:].mean(dim=1)
    semantic_turn_mode = 0.5 * (semantic_right_forward_target - semantic_left_forward_target)
    action_term = env.action_manager.get_term(action_name)
    velocity_limit = float(getattr(action_term, "_velocity_limit", 1.0))
    penalty = torch.relu(torch.abs(semantic_turn_mode) - float(soft_limit))
    return penalty / max(velocity_limit, 1.0)


def wheel_target_abs_soft_limit_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "wheel_motor_csv",
    soft_limit: float = 8.0,
) -> torch.Tensor:
    """Penalize excessive mean absolute wheel velocity target magnitude."""

    wheel_action_term = env.action_manager.get_term(action_name)
    target_abs = torch.mean(torch.abs(wheel_action_term.velocity_target), dim=1)
    velocity_limit = float(getattr(wheel_action_term, "_velocity_limit", 1.0))
    penalty = torch.relu(target_abs - float(soft_limit))
    return penalty / max(velocity_limit, 1.0)


def wheel_joint_vel_abs_soft_limit_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "wheel_motor_csv",
    soft_limit: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize excessive mean absolute actual wheel joint velocity magnitude."""

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_action_term = env.action_manager.get_term(action_name)
    wheel_joint_vel = asset.data.joint_vel[:, wheel_action_term._joint_ids]
    vel_abs = torch.mean(torch.abs(wheel_joint_vel), dim=1)
    velocity_limit = float(getattr(wheel_action_term, "_velocity_limit", 1.0))
    penalty = torch.relu(vel_abs - float(soft_limit))
    return penalty / max(velocity_limit, 1.0)


def wasted_turn_when_yaw_small_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "wheel_motor_csv",
    active_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize large semantic turn usage when commanded yaw is still not produced."""

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = yaw_rate_command(env)
    yaw = asset.data.root_ang_vel_b[:, 2]
    desired_sign = torch.sign(cmd)
    signed_yaw = desired_sign * yaw
    required_yaw = 0.5 * torch.abs(cmd)
    yaw_deficit = torch.relu(required_yaw - signed_yaw) / torch.clamp(required_yaw, min=1.0e-6)
    yaw_deficit = torch.clamp(yaw_deficit, min=0.0, max=2.0)
    raw_target = _wheel_target_vel_lf_lr_rf_rr(env, action_name=action_name)
    semantic_target = wheel_raw_to_semantic_lf_lr_rf_rr(raw_target)
    semantic_left_forward_target = semantic_target[:, :2].mean(dim=1)
    semantic_right_forward_target = semantic_target[:, 2:].mean(dim=1)
    semantic_turn_mode = 0.5 * (semantic_right_forward_target - semantic_left_forward_target)
    action_term = env.action_manager.get_term(action_name)
    velocity_limit = float(getattr(action_term, "_velocity_limit", 1.0))
    turn_usage = torch.abs(semantic_turn_mode) / max(velocity_limit, 1.0)
    active = (torch.abs(cmd) > float(active_threshold)).to(torch.float32)
    return turn_usage * yaw_deficit * active


def short_goal_excessive_yaw_rate_penalty(
    env: ManagerBasedRLEnv,
    free_yaw_rate: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize excessive absolute yaw rate beyond a small free band."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.relu(torch.abs(asset.data.root_ang_vel_b[:, 2]) - float(free_yaw_rate))


def short_goal_forward_velocity_during_turn_penalty(
    env: ManagerBasedRLEnv,
    free_speed: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize forward/backward body speed during a turn-only task."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.relu(torch.abs(asset.data.root_lin_vel_b[:, 0]) - float(free_speed))


def short_goal_base_xy_speed_penalty(
    env: ManagerBasedRLEnv,
    free_speed: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar body speed beyond a small allowance."""

    asset: Articulation = env.scene[asset_cfg.name]
    base_xy_speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    return torch.relu(base_xy_speed - float(free_speed))


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


def stroke_high_soft_limit_penalty(
    env: ManagerBasedRLEnv,
    soft_start: float = 0.55,
    hard_reference: float = 0.60,
    action_name: str = "leg_hydraulic",
) -> torch.Tensor:
    """Penalize high stroke; larger Ranger stroke lowers the body and sensor clearance."""

    if float(hard_reference) <= float(soft_start):
        raise ValueError("hard_reference must be greater than soft_start.")
    action_term = env.action_manager.get_term(action_name)
    scaled_excess = torch.relu(action_term.stroke_actual - float(soft_start)) / float(
        hard_reference - soft_start
    )
    return torch.mean(torch.square(scaled_excess), dim=1)


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


def hydraulic_action_rate_l1(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize L1 hydraulic action-rate using the first four action dimensions."""

    delta = torch.abs(env.action_manager.action[:, :4] - env.action_manager.prev_action[:, :4])
    return torch.mean(delta, dim=1)


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
