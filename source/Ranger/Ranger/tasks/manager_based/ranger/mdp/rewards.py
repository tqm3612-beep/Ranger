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
    wheel_forward_sign = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float32, device=env.device).unsqueeze(0)
    expected = semantic_expected * wheel_forward_sign
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
    wheel_forward_sign = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float32, device=env.device).unsqueeze(0)
    semantic_target = wheel_action_term.velocity_target * wheel_forward_sign
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
    wheel_forward_sign = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float32, device=env.device).unsqueeze(0)
    semantic_target = wheel_action_term.velocity_target * wheel_forward_sign
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
    wheel_forward_sign = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float32, device=env.device).unsqueeze(0)
    semantic_target = wheel_action_term.velocity_target * wheel_forward_sign
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
    wheel_forward_sign = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float32, device=env.device).unsqueeze(0)
    semantic_target = wheel_action_term.velocity_target * wheel_forward_sign
    left_mean = semantic_target[:, :2].mean(dim=1)
    right_mean = semantic_target[:, 2:].mean(dim=1)
    expected_diff = torch.clamp(-2.0 * float(turn_gain) * torch.sin(heading_error), -float(max_abs_diff), float(max_abs_diff))
    return torch.abs((left_mean - right_mean) - expected_diff)


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


def wheel_semantic_velocity_symmetry_l2(env: ManagerBasedRLEnv, action_name: str = "wheel_motor_csv") -> torch.Tensor:
    """Penalize left/right semantic wheel target mismatch."""

    wheel_action_term = env.action_manager.get_term(action_name)
    wheel_forward_sign = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float32, device=env.device).unsqueeze(0)
    semantic_target = wheel_action_term.velocity_target * wheel_forward_sign
    left_mean = semantic_target[:, :2].mean(dim=1)
    right_mean = semantic_target[:, 2:].mean(dim=1)
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
