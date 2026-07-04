# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for Ranger perception."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


SPEED_COMMAND_ATTR = "_ranger_speed_command"
SPEED_COMMAND_TARGET_ATTR = "_ranger_speed_command_target"
SPEED_COMMAND_TIMER_ATTR = "_ranger_speed_command_timer"
SPEED_COMMAND_DURATION_ATTR = "_ranger_speed_command_duration"
GOAL_HEADING_TARGET_ATTR = "_ranger_goal_heading_target_pos_w"
GOAL_HEADING_PREV_HEADING_ERROR_ATTR = "_ranger_goal_heading_prev_heading_error"


def _as_env_ids(env: ManagerBasedEnv, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _ensure_speed_command_buffers(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    command = getattr(env, SPEED_COMMAND_ATTR, None)
    command_target = getattr(env, SPEED_COMMAND_TARGET_ATTR, None)
    command_timer = getattr(env, SPEED_COMMAND_TIMER_ATTR, None)
    command_duration = getattr(env, SPEED_COMMAND_DURATION_ATTR, None)
    needs_init = (
        command is None
        or command_target is None
        or command_timer is None
        or command_duration is None
        or command.shape != (env.num_envs, 2)
        or command_target.shape != (env.num_envs, 2)
        or command_timer.shape != (env.num_envs,)
        or command_duration.shape != (env.num_envs,)
    )
    if needs_init:
        command = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
        command_target = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
        command_timer = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        command_duration = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
        setattr(env, SPEED_COMMAND_ATTR, command)
        setattr(env, SPEED_COMMAND_TARGET_ATTR, command_target)
        setattr(env, SPEED_COMMAND_TIMER_ATTR, command_timer)
        setattr(env, SPEED_COMMAND_DURATION_ATTR, command_duration)
    return command, command_target, command_timer, command_duration


def _sample_speed_command_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    stage: str,
    v_x_range: tuple[float, float],
    yaw_rate_range: tuple[float, float],
    command_duration_range: tuple[float, float],
    small_yaw_prob: float = 0.0,
    small_yaw_range: tuple[float, float] = (0.03, 0.08),
    v_x_bins: tuple[tuple[float, float], ...] | None = None,
) -> None:
    command, command_target, command_timer, command_duration = _ensure_speed_command_buffers(env)
    if env_ids.numel() == 0:
        return

    num = env_ids.numel()
    if v_x_bins:
        selector = torch.randint(len(v_x_bins), (num,), device=env.device)
        v_x_target = torch.empty(num, device=env.device, dtype=torch.float32)
        for bin_id, speed_range in enumerate(v_x_bins):
            mask = selector == bin_id
            if torch.any(mask):
                v_x_target[mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
                    float(speed_range[0]), float(speed_range[1])
                )[mask]
        command_target[env_ids, 0] = v_x_target
    else:
        command_target[env_ids, 0] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
            float(v_x_range[0]), float(v_x_range[1])
        )
    if stage.upper() == "A":
        selector = torch.rand(num, device=env.device)
        yaw_target = torch.zeros(num, device=env.device, dtype=torch.float32)
        half_prob = 0.5 * max(float(small_yaw_prob), 0.0)
        neg_mask = selector < half_prob
        pos_mask = (selector >= half_prob) & (selector < 2.0 * half_prob)
        if torch.any(neg_mask):
            yaw_target[neg_mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
                -float(small_yaw_range[1]), -float(small_yaw_range[0])
            )[neg_mask]
        if torch.any(pos_mask):
            yaw_target[pos_mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
                float(small_yaw_range[0]), float(small_yaw_range[1])
            )[pos_mask]
        command_target[env_ids, 1] = yaw_target
    else:
        selector = torch.rand(num, device=env.device)
        yaw_target = torch.zeros(num, device=env.device, dtype=torch.float32)
        neg_mask = (selector >= 0.4) & (selector < 0.7)
        pos_mask = selector >= 0.7
        if torch.any(neg_mask):
            yaw_target[neg_mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
                float(yaw_rate_range[0]), -0.12
            )[neg_mask]
        if torch.any(pos_mask):
            yaw_target[pos_mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
                0.12, float(yaw_rate_range[1])
            )[pos_mask]
        command_target[env_ids, 1] = yaw_target

    command_timer[env_ids] = 0.0
    command_duration[env_ids] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
        float(command_duration_range[0]), float(command_duration_range[1])
    )
    just_initialized = torch.all(command[env_ids] == 0.0, dim=1)
    if torch.any(just_initialized):
        command[env_ids[just_initialized]] = command_target[env_ids[just_initialized]]


def reset_speed_command(
    env: ManagerBasedEnv,
    env_ids,
    stage: str = "A",
    v_x_range: tuple[float, float] = (0.05, 0.4),
    yaw_rate_range: tuple[float, float] = (-0.4, 0.4),
    command_duration_range: tuple[float, float] = (2.0, 5.0),
    small_yaw_prob: float = 0.0,
    small_yaw_range: tuple[float, float] = (0.03, 0.08),
    v_x_bins: tuple[tuple[float, float], ...] | None = None,
) -> None:
    """Initialize per-env speed-command buffers at reset."""

    env_ids = _as_env_ids(env, env_ids)
    command, _, _, _ = _ensure_speed_command_buffers(env)
    command[env_ids] = 0.0
    _sample_speed_command_targets(
        env=env,
        env_ids=env_ids,
        stage=stage,
        v_x_range=v_x_range,
        yaw_rate_range=yaw_rate_range,
        command_duration_range=command_duration_range,
        small_yaw_prob=small_yaw_prob,
        small_yaw_range=small_yaw_range,
        v_x_bins=v_x_bins,
    )


def update_speed_command(
    env: ManagerBasedEnv,
    stage: str = "A",
    v_x_range: tuple[float, float] = (0.05, 0.4),
    yaw_rate_range: tuple[float, float] = (-0.4, 0.4),
    command_duration_range: tuple[float, float] = (2.0, 5.0),
    smoothing_alpha: float = 0.1,
    small_yaw_prob: float = 0.0,
    small_yaw_range: tuple[float, float] = (0.03, 0.08),
    v_x_bins: tuple[tuple[float, float], ...] | None = None,
) -> torch.Tensor:
    """Update and return the smoothed dynamic speed command."""

    command, command_target, command_timer, command_duration = _ensure_speed_command_buffers(env)
    dt = float(getattr(env, "step_dt", 1.0 / 60.0))
    command_timer += dt
    expired = command_timer >= command_duration
    if torch.any(expired):
        _sample_speed_command_targets(
            env=env,
            env_ids=expired.nonzero(as_tuple=False).squeeze(-1),
            stage=stage,
            v_x_range=v_x_range,
            yaw_rate_range=yaw_rate_range,
            command_duration_range=command_duration_range,
            small_yaw_prob=small_yaw_prob,
            small_yaw_range=small_yaw_range,
            v_x_bins=v_x_bins,
        )
    alpha = float(smoothing_alpha)
    command[:] = command + alpha * (command_target - command)
    return command


def speed_command(env: ManagerBasedEnv) -> torch.Tensor:
    """Return current smoothed ``[v_x_cmd, yaw_rate_cmd]`` without updating it."""

    command, _, _, _ = _ensure_speed_command_buffers(env)
    return command


def speed_command_target(env: ManagerBasedEnv) -> torch.Tensor:
    """Return current command target ``[v_x_cmd_target, yaw_rate_cmd_target]``."""

    _, command_target, _, _ = _ensure_speed_command_buffers(env)
    return command_target


def speed_command_time_left(env: ManagerBasedEnv) -> torch.Tensor:
    """Return seconds until each command target is resampled."""

    _, _, command_timer, command_duration = _ensure_speed_command_buffers(env)
    return torch.clamp(command_duration - command_timer, min=0.0)


def speed_command_state(
    env: ManagerBasedEnv,
    stage: str = "A",
    v_x_range: tuple[float, float] = (0.05, 0.4),
    yaw_rate_range: tuple[float, float] = (-0.4, 0.4),
    command_duration_range: tuple[float, float] = (2.0, 5.0),
    smoothing_alpha: float = 0.1,
    max_command_duration: float = 5.0,
    small_yaw_prob: float = 0.0,
    small_yaw_range: tuple[float, float] = (0.03, 0.08),
    v_x_bins: tuple[tuple[float, float], ...] | None = None,
) -> torch.Tensor:
    """Return six-dimensional dynamic velocity-command observation."""

    command = update_speed_command(
        env=env,
        stage=stage,
        v_x_range=v_x_range,
        yaw_rate_range=yaw_rate_range,
        command_duration_range=command_duration_range,
        smoothing_alpha=smoothing_alpha,
        small_yaw_prob=small_yaw_prob,
        small_yaw_range=small_yaw_range,
        v_x_bins=v_x_bins,
    )
    _, command_target, _, _ = _ensure_speed_command_buffers(env)
    time_left = speed_command_time_left(env)
    obs = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
    obs[:, 0] = torch.clamp(command[:, 0] / 0.45, min=-1.0, max=1.0)
    obs[:, 1] = torch.clamp(command[:, 1], min=-1.0, max=1.0)
    obs[:, 2] = torch.sin(command[:, 1])
    obs[:, 3] = torch.cos(command[:, 1])
    obs[:, 4] = torch.clamp(command_target[:, 0] / 0.45, min=-1.0, max=1.0)
    obs[:, 5] = torch.clamp(time_left / max(float(max_command_duration), 1.0e-6), min=0.0, max=1.0)
    return obs


def _ensure_goal_heading_target(env: ManagerBasedEnv) -> torch.Tensor:
    target_pos_w = getattr(env, GOAL_HEADING_TARGET_ATTR, None)
    if target_pos_w is None or target_pos_w.shape != (env.num_envs, 3):
        target_pos_w = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
        setattr(env, GOAL_HEADING_TARGET_ATTR, target_pos_w)
    return target_pos_w


def reset_goal_heading_target(
    env: ManagerBasedEnv,
    env_ids,
    distance_range: tuple[float, float] = (2.0, 5.0),
    heading_range: tuple[float, float] = (-0.7853981633974483, 0.7853981633974483),
    heading_bins: tuple[tuple[float, float], ...] | None = None,
    heading_bin_probs: tuple[float, ...] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample a per-episode target point in front of the robot's reset heading."""

    env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_w = _ensure_goal_heading_target(env)
    num = env_ids.numel()
    distance = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
        float(distance_range[0]), float(distance_range[1])
    )
    if heading_bins:
        if heading_bin_probs is None:
            probabilities = torch.ones(len(heading_bins), device=env.device, dtype=torch.float32)
        else:
            if len(heading_bin_probs) != len(heading_bins):
                raise ValueError("heading_bin_probs must have the same length as heading_bins.")
            probabilities = torch.tensor(heading_bin_probs, device=env.device, dtype=torch.float32)
        probabilities = probabilities / torch.clamp(probabilities.sum(), min=1.0e-6)
        selector = torch.multinomial(probabilities, num, replacement=True)
        heading = torch.empty(num, device=env.device, dtype=torch.float32)
        for bin_id, angle_range in enumerate(heading_bins):
            mask = selector == bin_id
            if torch.any(mask):
                heading[mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
                    float(angle_range[0]), float(angle_range[1])
                )[mask]
    else:
        heading = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
            float(heading_range[0]), float(heading_range[1])
        )
    target_vec_b = torch.zeros((num, 3), device=env.device, dtype=torch.float32)
    target_vec_b[:, 0] = distance * torch.cos(heading)
    target_vec_b[:, 1] = distance * torch.sin(heading)
    target_vec_w = math_utils.quat_apply_yaw(asset.data.root_quat_w[env_ids], target_vec_b)
    target_pos_w[env_ids] = asset.data.root_pos_w[env_ids] + target_vec_w
    prev_heading_error = getattr(env, GOAL_HEADING_PREV_HEADING_ERROR_ATTR, None)
    if prev_heading_error is not None and prev_heading_error.shape == (env.num_envs,):
        prev_heading_error[env_ids] = float("nan")


def goal_heading_target_pos_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Return sampled goal-heading target positions in world frame."""

    return _ensure_goal_heading_target(env)


def goal_heading_target_body(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target vector, distance, and heading error in the robot body frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_w = _ensure_goal_heading_target(env)
    target_vec_w = target_pos_w - asset.data.root_pos_w
    target_vec_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, target_vec_w)
    target_xy_b = target_vec_b[:, :2]
    distance = torch.norm(target_xy_b, dim=1)
    heading_error = torch.atan2(target_xy_b[:, 1], target_xy_b[:, 0])
    return target_vec_b, distance, heading_error


def goal_heading_state(
    env: ManagerBasedEnv,
    goal_range: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return six-dimensional target-heading descriptor in the body frame."""

    if goal_range <= 0.0:
        raise ValueError(f"goal_range must be positive. Received: {goal_range}")

    target_vec_b, distance, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
    obs = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
    obs[:, 0] = torch.clamp(target_vec_b[:, 0] / float(goal_range), min=-1.0, max=1.0)
    obs[:, 1] = torch.clamp(target_vec_b[:, 1] / float(goal_range), min=-1.0, max=1.0)
    obs[:, 2] = torch.clamp(distance / float(goal_range), min=0.0, max=1.0)
    obs[:, 3] = torch.sin(heading_error)
    obs[:, 4] = torch.cos(heading_error)
    obs[:, 5] = 1.0
    return obs


def base_lin_vel_normalized(
    env: ManagerBasedEnv,
    scale: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return normalized base linear velocity in the body frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_b / scale, min=-1.0, max=1.0)


def base_ang_vel_normalized(
    env: ManagerBasedEnv,
    scale: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return normalized base angular velocity in the body frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_ang_vel_b / scale, min=-1.0, max=1.0)


def projected_gravity_normalized(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return clipped projected gravity vector."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.projected_gravity_b, min=-1.0, max=1.0)


def joint_pos_rel_normalized(
    env: ManagerBasedEnv,
    scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return normalized relative joint positions for the selected joints."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.clamp(joint_pos_rel / scale, min=-1.0, max=1.0)


def joint_vel_rel_normalized(
    env: ManagerBasedEnv,
    scale: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return normalized relative joint velocities for the selected joints."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_rel = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    return torch.clamp(joint_vel_rel / scale, min=-1.0, max=1.0)


def hydraulic_stroke_state(env: ManagerBasedEnv, action_name: str = "leg_hydraulic") -> torch.Tensor:
    """Return the current equivalent hydraulic-cylinder stroke state."""

    action_term = env.action_manager.get_term(action_name)
    if not hasattr(action_term, "stroke_actual"):
        raise AttributeError(f"Action term '{action_name}' does not expose 'stroke_actual'.")
    return action_term.stroke_actual


def hydraulic_stroke_state_normalized(
    env: ManagerBasedEnv,
    action_name: str = "leg_hydraulic",
    stroke_min: float = 0.0,
    stroke_max: float = 1.0,
) -> torch.Tensor:
    """Return normalized equivalent hydraulic-cylinder stroke state."""

    stroke_actual = hydraulic_stroke_state(env=env, action_name=action_name)
    normalized = (stroke_actual - stroke_min) / max(stroke_max - stroke_min, 1.0e-6)
    return torch.clamp(normalized, min=0.0, max=1.0)


def hydraulic_effort_state_normalized(
    env: ManagerBasedEnv,
    action_name: str = "leg_hydraulic",
    effort_limit: float = 300.0,
) -> torch.Tensor:
    """Return normalized equivalent hydraulic effort state."""

    action_term = env.action_manager.get_term(action_name)
    if not hasattr(action_term, "effort_actual"):
        raise AttributeError(f"Action term '{action_name}' does not expose 'effort_actual'.")
    return torch.clamp(action_term.effort_actual / max(effort_limit, 1.0e-6), min=-1.0, max=1.0)


def wheel_velocity_target_normalized(
    env: ManagerBasedEnv,
    action_name: str = "wheel_motor_csv",
    velocity_limit: float = 20.0,
) -> torch.Tensor:
    """Return normalized wheel CSV target velocity state."""

    action_term = env.action_manager.get_term(action_name)
    if not hasattr(action_term, "velocity_target"):
        raise AttributeError(f"Action term '{action_name}' does not expose 'velocity_target'.")
    return torch.clamp(action_term.velocity_target / max(velocity_limit, 1.0e-6), min=-1.0, max=1.0)


def wheel_torque_state_normalized(
    env: ManagerBasedEnv,
    action_name: str = "wheel_motor_csv",
    effort_limit: float = 100.0,
) -> torch.Tensor:
    """Return normalized wheel torque state from the local CSV loop."""

    action_term = env.action_manager.get_term(action_name)
    if not hasattr(action_term, "torque_actual"):
        raise AttributeError(f"Action term '{action_name}' does not expose 'torque_actual'.")
    return torch.clamp(action_term.torque_actual / max(effort_limit, 1.0e-6), min=-1.0, max=1.0)


def goal_state(
    env: ManagerBasedEnv,
    goal_x_body: float = 0.0,
    goal_y_body: float = 0.0,
    goal_range: float = 5.0,
    goal_enabled: bool = False,
) -> torch.Tensor:
    """Return a fixed six-dimensional goal descriptor in the body frame.

    This keeps the policy interface stable for later staged training. When
    ``goal_enabled`` is False, the term returns a neutral placeholder:
    ``[0, 0, 0, 0, 1, 0]``.
    """

    if goal_range <= 0.0:
        raise ValueError(f"goal_range must be positive. Received: {goal_range}")

    goal_obs = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
    if not goal_enabled:
        goal_obs[:, 4] = 1.0
        return goal_obs

    goal_x_body_tensor = torch.full((env.num_envs,), float(goal_x_body), device=env.device)
    goal_y_body_tensor = torch.full((env.num_envs,), float(goal_y_body), device=env.device)
    distance = torch.sqrt(goal_x_body_tensor.square() + goal_y_body_tensor.square())
    bearing = torch.atan2(goal_y_body_tensor, goal_x_body_tensor)

    goal_obs[:, 0] = torch.clamp(goal_x_body_tensor / goal_range, min=-1.0, max=1.0)
    goal_obs[:, 1] = torch.clamp(goal_y_body_tensor / goal_range, min=-1.0, max=1.0)
    goal_obs[:, 2] = torch.clamp(distance / goal_range, min=0.0, max=1.0)
    goal_obs[:, 3] = torch.sin(bearing)
    goal_obs[:, 4] = torch.cos(bearing)
    goal_obs[:, 5] = 1.0
    return goal_obs


def last_action_normalized(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """Return clipped previous action history."""

    if action_name is None:
        return torch.clamp(env.action_manager.action, min=-1.0, max=1.0)
    return torch.clamp(env.action_manager.get_term(action_name).raw_actions, min=-1.0, max=1.0)


def local_sensor_visibility_maps(
    env: ManagerBasedEnv,
    sensor_names: tuple[str, ...],
    asset_name: str = "robot",
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Return one local valid-mask grid per sensor for visibility debugging."""

    visibility_maps = {}
    for sensor_name in sensor_names:
        _, valid_mask = _build_local_height_map(
            env=env,
            sensor_names=(sensor_name,),
            asset_name=asset_name,
            x_range=x_range,
            y_range=y_range,
            resolution=resolution,
        )
        visibility_maps[sensor_name] = valid_mask.to(torch.float32)
    return visibility_maps


def local_navigation_map_layers(
    env: ManagerBasedEnv,
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
    apply_noise: bool = False,
    height_noise_std: float = 0.0,
    risk_noise_std: float = 0.0,
    valid_dropout_prob: float = 0.0,
    slope_weight: float = 0.4,
    roughness_weight: float = 0.3,
    step_weight: float = 0.3,
    unknown_penalty: float = 1.0,
    use_neutral_map: bool = False,
) -> dict[str, torch.Tensor]:
    """Return the unflattened local navigation layers for planning-aware policies."""

    if use_neutral_map:
        return _build_neutral_navigation_map_layers(
            env=env,
            x_range=x_range,
            y_range=y_range,
            resolution=resolution,
        )

    height_map_raw, valid_mask = _build_local_height_map(
        env=env,
        sensor_names=sensor_names,
        asset_name=asset_name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
    )
    height_map = _normalize_height_map(
        height_map_raw=height_map_raw,
        valid_mask=valid_mask,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        reference_x_range=height_reference_x_range,
        reference_y_range=height_reference_y_range,
    )
    slope_map = _normalize_risk_map(
        _compute_slope_map(height_map, valid_mask, resolution),
        valid_mask,
        normalization=slope_normalization,
    )
    roughness_map = _normalize_risk_map(
        _compute_roughness_map(height_map, valid_mask),
        valid_mask,
        normalization=roughness_normalization,
    )
    step_map = _normalize_risk_map(
        _compute_step_map(height_map, valid_mask, step_threshold),
        valid_mask,
        normalization=step_normalization,
    )
    height_map, slope_map, roughness_map, step_map, valid_mask = _apply_local_map_noise(
        height_map=height_map,
        slope_map=slope_map,
        roughness_map=roughness_map,
        step_map=step_map,
        valid_mask=valid_mask,
        apply_noise=apply_noise,
        height_noise_std=height_noise_std,
        risk_noise_std=risk_noise_std,
        valid_dropout_prob=valid_dropout_prob,
    )
    traversability_map = _compute_traversability_map(
        slope_map=slope_map,
        roughness_map=roughness_map,
        step_map=step_map,
        valid_mask=valid_mask,
        slope_weight=slope_weight,
        roughness_weight=roughness_weight,
        step_weight=step_weight,
        unknown_penalty=unknown_penalty,
    )

    return {
        "height": height_map,
        "slope": slope_map,
        "roughness": roughness_map,
        "step": step_map,
        "traversability": traversability_map,
        "valid_mask": valid_mask.to(height_map.dtype),
    }


def local_navigation_map(
    env: ManagerBasedEnv,
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
    apply_noise: bool = False,
    height_noise_std: float = 0.0,
    risk_noise_std: float = 0.0,
    valid_dropout_prob: float = 0.0,
    slope_weight: float = 0.4,
    roughness_weight: float = 0.3,
    step_weight: float = 0.3,
    unknown_penalty: float = 1.0,
    use_neutral_map: bool = False,
) -> torch.Tensor:
    """Build a local six-layer navigation map from ray-based terrain perception."""

    layers_dict = local_navigation_map_layers(
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
        apply_noise=apply_noise,
        height_noise_std=height_noise_std,
        risk_noise_std=risk_noise_std,
        valid_dropout_prob=valid_dropout_prob,
        slope_weight=slope_weight,
        roughness_weight=roughness_weight,
        step_weight=step_weight,
        unknown_penalty=unknown_penalty,
        use_neutral_map=use_neutral_map,
    )
    layers = torch.stack(
        (
            layers_dict["height"],
            layers_dict["slope"],
            layers_dict["roughness"],
            layers_dict["step"],
            layers_dict["traversability"],
            layers_dict["valid_mask"],
        ),
        dim=1,
    )
    return layers.reshape(env.num_envs, -1)


def local_geometric_map_layers(
    env: ManagerBasedEnv,
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
    apply_noise: bool = False,
    height_noise_std: float = 0.0,
    risk_noise_std: float = 0.0,
    valid_dropout_prob: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Backward-compatible wrapper that returns the original five-layer map."""

    layers_dict = local_navigation_map_layers(
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
        apply_noise=apply_noise,
        height_noise_std=height_noise_std,
        risk_noise_std=risk_noise_std,
        valid_dropout_prob=valid_dropout_prob,
    )
    return {
        "height": layers_dict["height"],
        "slope": layers_dict["slope"],
        "roughness": layers_dict["roughness"],
        "step": layers_dict["step"],
        "valid_mask": layers_dict["valid_mask"],
    }


def local_geometric_map(
    env: ManagerBasedEnv,
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
    apply_noise: bool = False,
    height_noise_std: float = 0.0,
    risk_noise_std: float = 0.0,
    valid_dropout_prob: float = 0.0,
) -> torch.Tensor:
    """Backward-compatible wrapper that returns the original five-layer map."""

    layers_dict = local_geometric_map_layers(
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
        apply_noise=apply_noise,
        height_noise_std=height_noise_std,
        risk_noise_std=risk_noise_std,
        valid_dropout_prob=valid_dropout_prob,
    )
    layers = torch.stack(
        (
            layers_dict["height"],
            layers_dict["slope"],
            layers_dict["roughness"],
            layers_dict["step"],
            layers_dict["valid_mask"],
        ),
        dim=1,
    )
    return layers.reshape(env.num_envs, -1)


def _build_local_height_map(
    env: ManagerBasedEnv,
    sensor_names: tuple[str, ...],
    asset_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    invalid_height: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[asset_name]
    ray_hits_w = []
    for sensor_name in sensor_names:
        sensor: RayCaster = env.scene.sensors[sensor_name]
        if hasattr(sensor.data, "ray_hits_w"):
            hits_w = sensor.data.ray_hits_w
        else:
            # RayCasterCamera stores world-space hits on the sensor object and may
            # update a subset of environments internally, so refresh all envs here.
            sensor._update_buffers_impl(slice(None))
            hits_w = sensor.ray_hits_w
        ray_hits_w.append(hits_w)
    points_w = torch.cat(ray_hits_w, dim=1)

    points_rel_w = points_w - asset.data.root_pos_w.unsqueeze(1)
    num_rays = points_rel_w.shape[1]
    # Use a gravity-aligned local frame: keep yaw, remove roll and pitch.
    root_yaw_quat_w = math_utils.yaw_quat(asset.data.root_quat_w)
    points_b = math_utils.quat_apply_inverse(
        root_yaw_quat_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4),
        points_rel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_rays, 3)

    x_min, x_max = x_range
    y_min, y_max = y_range
    num_x = int(round((x_max - x_min) / resolution)) + 1
    num_y = int(round((y_max - y_min) / resolution)) + 1
    num_cells = num_x * num_y

    x = points_b[..., 0]
    y = points_b[..., 1]
    z = points_b[..., 2]
    valid_points = (
        torch.isfinite(points_b).all(dim=-1)
        & (x >= x_min)
        & (x <= x_max)
        & (y >= y_min)
        & (y <= y_max)
    )

    ix = torch.round((x - x_min) / resolution).long().clamp(0, num_x - 1)
    iy = torch.round((y - y_min) / resolution).long().clamp(0, num_y - 1)
    local_index = ix * num_y + iy

    env_offsets = torch.arange(env.num_envs, device=points_b.device).unsqueeze(1) * num_cells
    flat_index = (local_index + env_offsets).reshape(-1)
    flat_valid = valid_points.reshape(-1)
    flat_z = z.reshape(-1)

    flat_height = torch.full((env.num_envs * num_cells,), -torch.inf, device=points_b.device)
    if flat_valid.any():
        flat_height.scatter_reduce_(
            0,
            flat_index[flat_valid],
            flat_z[flat_valid],
            reduce="amax",
            include_self=True,
        )

    valid_mask = torch.isfinite(flat_height).reshape(env.num_envs, num_x, num_y)
    height_map = flat_height.reshape(env.num_envs, num_x, num_y)
    height_map = torch.where(valid_mask, height_map, torch.full_like(height_map, invalid_height))
    return height_map, valid_mask


def _compute_slope_map(height_map: torch.Tensor, valid_mask: torch.Tensor, resolution: float) -> torch.Tensor:
    slope_x = torch.zeros_like(height_map)
    valid_x = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]
    diff_x = torch.abs(height_map[:, 1:, :] - height_map[:, :-1, :]) / resolution
    slope_x[:, 1:, :] = torch.where(valid_x, diff_x, torch.zeros_like(diff_x))

    slope_y = torch.zeros_like(height_map)
    valid_y = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]
    diff_y = torch.abs(height_map[:, :, 1:] - height_map[:, :, :-1]) / resolution
    slope_y[:, :, 1:] = torch.where(valid_y, diff_y, torch.zeros_like(diff_y))
    return torch.maximum(slope_x, slope_y)


def _normalize_height_map(
    height_map_raw: torch.Tensor,
    valid_mask: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    reference_x_range: tuple[float, float],
    reference_y_range: tuple[float, float],
) -> torch.Tensor:
    """Shift height so locally flat support terrain stays near zero."""

    reference_height = _compute_height_reference(
        height_map_raw=height_map_raw,
        valid_mask=valid_mask,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        reference_x_range=reference_x_range,
        reference_y_range=reference_y_range,
    )
    height_map = height_map_raw - reference_height[:, None, None]
    return torch.where(valid_mask, height_map, torch.zeros_like(height_map))


def _compute_height_reference(
    height_map_raw: torch.Tensor,
    valid_mask: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    reference_x_range: tuple[float, float],
    reference_y_range: tuple[float, float],
) -> torch.Tensor:
    """Estimate the local ground reference height from a small anchor region near the robot."""

    device = height_map_raw.device
    num_x = height_map_raw.shape[1]
    num_y = height_map_raw.shape[2]
    x_coords = torch.linspace(x_range[0], x_range[1], num_x, device=device)
    y_coords = torch.linspace(y_range[0], y_range[1], num_y, device=device)
    x_mask = (x_coords >= reference_x_range[0]) & (x_coords <= reference_x_range[1])
    y_mask = (y_coords >= reference_y_range[0]) & (y_coords <= reference_y_range[1])
    reference_region_mask = x_mask[:, None] & y_mask[None, :]

    fallback_region_mask = (x_coords >= x_range[0]) & (x_coords <= min(x_range[1], reference_x_range[1] + resolution))
    fallback_region_mask = fallback_region_mask[:, None] & torch.ones((1, num_y), dtype=torch.bool, device=device)

    reference_heights = []
    for env_id in range(height_map_raw.shape[0]):
        region_valid = valid_mask[env_id] & reference_region_mask
        if region_valid.any():
            reference_heights.append(height_map_raw[env_id][region_valid].median())
            continue

        fallback_valid = valid_mask[env_id] & fallback_region_mask
        if fallback_valid.any():
            reference_heights.append(height_map_raw[env_id][fallback_valid].median())
            continue

        all_valid = valid_mask[env_id]
        if all_valid.any():
            reference_heights.append(height_map_raw[env_id][all_valid].median())
        else:
            reference_heights.append(torch.tensor(0.0, device=device, dtype=height_map_raw.dtype))

    return torch.stack(reference_heights, dim=0)


def _normalize_risk_map(risk_map: torch.Tensor, valid_mask: torch.Tensor, normalization: float) -> torch.Tensor:
    """Clamp a raw geometric risk layer into a 0..1 intensity map."""

    if normalization <= 0.0:
        raise ValueError(f"Risk normalization must be positive. Received: {normalization}")
    normalized = torch.clamp(risk_map / normalization, min=0.0, max=1.0)
    return torch.where(valid_mask, normalized, torch.zeros_like(normalized))


def _build_neutral_navigation_map_layers(
    env: ManagerBasedEnv,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
) -> dict[str, torch.Tensor]:
    """Create a stage-1 neutral navigation map with fully traversable support."""

    num_x, num_y = _navigation_map_grid_shape(x_range=x_range, y_range=y_range, resolution=resolution)
    zeros = torch.zeros((env.num_envs, num_x, num_y), device=env.device, dtype=torch.float32)
    ones = torch.ones_like(zeros)
    return {
        "height": zeros,
        "slope": zeros,
        "roughness": zeros,
        "step": zeros,
        "traversability": ones,
        "valid_mask": ones,
    }


def _compute_traversability_map(
    slope_map: torch.Tensor,
    roughness_map: torch.Tensor,
    step_map: torch.Tensor,
    valid_mask: torch.Tensor,
    slope_weight: float,
    roughness_weight: float,
    step_weight: float,
    unknown_penalty: float,
) -> torch.Tensor:
    """Fuse geometric risk layers into a simple traversability intensity."""

    valid_mask_float = valid_mask.to(slope_map.dtype)
    cost = (
        slope_weight * slope_map
        + roughness_weight * roughness_map
        + step_weight * step_map
        + unknown_penalty * (1.0 - valid_mask_float)
    )
    cost = torch.clamp(cost, min=0.0, max=1.0)
    traversability = 1.0 - cost
    return torch.where(valid_mask, traversability, torch.zeros_like(traversability))


def _navigation_map_grid_shape(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
) -> tuple[int, int]:
    """Compute the discrete local map grid shape from metric bounds."""

    num_x = int(round((x_range[1] - x_range[0]) / resolution)) + 1
    num_y = int(round((y_range[1] - y_range[0]) / resolution)) + 1
    return num_x, num_y


def _apply_local_map_noise(
    height_map: torch.Tensor,
    slope_map: torch.Tensor,
    roughness_map: torch.Tensor,
    step_map: torch.Tensor,
    valid_mask: torch.Tensor,
    apply_noise: bool,
    height_noise_std: float,
    risk_noise_std: float,
    valid_dropout_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply lightweight noise and random visibility dropout to the local map."""

    if not apply_noise:
        return height_map, slope_map, roughness_map, step_map, valid_mask

    valid_mask_noisy = valid_mask
    if valid_dropout_prob > 0.0:
        dropout = torch.rand_like(valid_mask.to(torch.float32)) < valid_dropout_prob
        valid_mask_noisy = valid_mask & (~dropout)

    if height_noise_std > 0.0:
        height_map = height_map + torch.randn_like(height_map) * height_noise_std

    if risk_noise_std > 0.0:
        slope_map = slope_map + torch.randn_like(slope_map) * risk_noise_std
        roughness_map = roughness_map + torch.randn_like(roughness_map) * risk_noise_std
        step_map = step_map + torch.randn_like(step_map) * risk_noise_std

    slope_map = torch.clamp(slope_map, min=0.0, max=1.0)
    roughness_map = torch.clamp(roughness_map, min=0.0, max=1.0)
    step_map = torch.clamp(step_map, min=0.0, max=1.0)

    height_map = torch.where(valid_mask_noisy, height_map, torch.zeros_like(height_map))
    slope_map = torch.where(valid_mask_noisy, slope_map, torch.zeros_like(slope_map))
    roughness_map = torch.where(valid_mask_noisy, roughness_map, torch.zeros_like(roughness_map))
    step_map = torch.where(valid_mask_noisy, step_map, torch.zeros_like(step_map))

    return height_map, slope_map, roughness_map, step_map, valid_mask_noisy


def _compute_roughness_map(height_map: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    height = height_map.unsqueeze(1)
    mask = valid_mask.to(height_map.dtype).unsqueeze(1)
    count = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) * 9.0
    height_sum = F.avg_pool2d(height * mask, kernel_size=3, stride=1, padding=1) * 9.0
    mean = height_sum / torch.clamp(count, min=1.0)
    var_sum = F.avg_pool2d(((height - mean) ** 2) * mask, kernel_size=3, stride=1, padding=1) * 9.0
    roughness = torch.sqrt(var_sum / torch.clamp(count, min=1.0))
    return torch.where(valid_mask, roughness.squeeze(1), torch.zeros_like(height_map))


def _compute_step_map(height_map: torch.Tensor, valid_mask: torch.Tensor, threshold: float) -> torch.Tensor:
    step_x = torch.zeros_like(height_map)
    valid_x = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]
    diff_x = torch.abs(height_map[:, 1:, :] - height_map[:, :-1, :])
    step_x[:, 1:, :] = torch.where(valid_x & (diff_x > threshold), diff_x, torch.zeros_like(diff_x))

    step_y = torch.zeros_like(height_map)
    valid_y = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]
    diff_y = torch.abs(height_map[:, :, 1:] - height_map[:, :, :-1])
    step_y[:, :, 1:] = torch.where(valid_y & (diff_y > threshold), diff_y, torch.zeros_like(diff_y))
    return torch.maximum(step_x, step_y)
