# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for Ranger perception."""

from __future__ import annotations

import math
import os
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
SHORT_GOAL_TARGET_ATTR = "_ranger_short_goal_pos_w"
SHORT_GOAL_PREV_DISTANCE_ATTR = "_ranger_short_goal_prev_distance"
SHORT_GOAL_REACHED_ATTR = "_ranger_short_goal_reached"
SHORT_GOAL_PREV_HEADING_ERROR_ATTR = "_ranger_short_goal_prev_heading_error"
YAW_RATE_COMMAND_ATTR = "_ranger_yaw_rate_command"
SUSPENSION_STROKE_PREV_ATTR = "_ranger_suspension_stroke_prev"
SUSPENSION_STROKE_RATE_ATTR = "_ranger_suspension_stroke_rate"
SUSPENSION_STROKE_RATE_STEP_ATTR = "_ranger_suspension_stroke_rate_step"
COMMAND_OBS_CACHE_ATTR = "_ranger_command_obs_cache"
COMMAND_OBS_CACHE_STEP_ATTR = "_ranger_command_obs_cache_step"
WHEEL_CONTACT_SENSOR_BODY_IDS_ATTR = "_ranger_wheel_contact_sensor_body_ids"


def _obs_debug_enabled() -> bool:
    return os.getenv("RANGER_OBS_DEBUG", "0") == "1"


def _obs_debug(msg: str) -> None:
    if _obs_debug_enabled():
        print(f"[OBS_DEBUG] {msg}", flush=True)


def _policy_state_disabled() -> bool:
    return os.getenv("RANGER_DISABLE_POLICY_STATE", "0") == "1"


def _policy_map_disabled() -> bool:
    return os.getenv("RANGER_DISABLE_POLICY_MAP", "0") == "1"


def _critic_privileged_disabled() -> bool:
    return os.getenv("RANGER_DISABLE_CRITIC_PRIVILEGED", "0") == "1"


def _as_env_ids(env: ManagerBasedEnv, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return float(default)
    return float(raw)


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
    max_command_duration: float = 5.0,
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


def _ensure_short_goal_target(env: ManagerBasedEnv) -> torch.Tensor:
    target_pos_w = getattr(env, SHORT_GOAL_TARGET_ATTR, None)
    if target_pos_w is None or target_pos_w.shape != (env.num_envs, 3):
        target_pos_w = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
        setattr(env, SHORT_GOAL_TARGET_ATTR, target_pos_w)
    return target_pos_w


def _ensure_short_goal_buffers(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_pos_w = _ensure_short_goal_target(env)
    prev_goal_distance = getattr(env, SHORT_GOAL_PREV_DISTANCE_ATTR, None)
    goal_reached = getattr(env, SHORT_GOAL_REACHED_ATTR, None)
    needs_init = (
        prev_goal_distance is None
        or goal_reached is None
        or prev_goal_distance.shape != (env.num_envs,)
        or goal_reached.shape != (env.num_envs,)
    )
    if needs_init:
        prev_goal_distance = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        goal_reached = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
        setattr(env, SHORT_GOAL_PREV_DISTANCE_ATTR, prev_goal_distance)
        setattr(env, SHORT_GOAL_REACHED_ATTR, goal_reached)
    return target_pos_w, prev_goal_distance, goal_reached


def _ensure_yaw_rate_command_buffer(env: ManagerBasedEnv) -> torch.Tensor:
    command = getattr(env, YAW_RATE_COMMAND_ATTR, None)
    if command is None or command.shape != (env.num_envs,):
        command = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        setattr(env, YAW_RATE_COMMAND_ATTR, command)
    return command


def reset_yaw_rate_command(
    env: ManagerBasedEnv,
    env_ids,
    min_abs: float = 0.08,
    max_abs: float = 0.15,
    zero_rate: float = 0.0,
    sign_mode: str = "balanced",
) -> None:
    """Sample a per-episode constant yaw-rate command."""

    env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    min_abs = _env_float("RANGER_YAW_CMD_MIN_ABS", min_abs)
    max_abs = _env_float("RANGER_YAW_CMD_MAX_ABS", max_abs)
    zero_rate = _env_float("RANGER_YAW_CMD_ZERO_RATE", zero_rate)
    sign_mode = os.getenv("RANGER_YAW_CMD_SIGN_MODE", sign_mode).strip().lower()
    if sign_mode not in {"balanced", "positive", "negative"}:
        sign_mode = "balanced"
    if max_abs < min_abs:
        min_abs, max_abs = max_abs, min_abs
    zero_rate = min(max(float(zero_rate), 0.0), 1.0)

    command = _ensure_yaw_rate_command_buffer(env)
    num = env_ids.numel()
    abs_cmd = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(float(min_abs), float(max_abs))
    if sign_mode == "positive":
        sign = torch.ones(num, device=env.device, dtype=torch.float32)
    elif sign_mode == "negative":
        sign = -torch.ones(num, device=env.device, dtype=torch.float32)
    else:
        sign = torch.where(
            torch.rand(num, device=env.device) < 0.5,
            -torch.ones(num, device=env.device, dtype=torch.float32),
            torch.ones(num, device=env.device, dtype=torch.float32),
        )
    cmd = sign * abs_cmd
    if zero_rate > 0.0:
        zero_mask = torch.rand(num, device=env.device) < float(zero_rate)
        cmd = torch.where(zero_mask, torch.zeros_like(cmd), cmd)
    command[env_ids] = cmd


def yaw_rate_command(env: ManagerBasedEnv) -> torch.Tensor:
    """Return per-env constant yaw-rate command."""

    return _ensure_yaw_rate_command_buffer(env)


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


def reset_short_goal_target(
    env: ManagerBasedEnv,
    env_ids,
    distance_range: tuple[float, float] = (0.5, 2.0),
    heading_range: tuple[float, float] = (-0.7853981633974483, 0.7853981633974483),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample a short-range flat-goal target and initialize per-env progress buffers."""

    env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_w, prev_goal_distance, goal_reached = _ensure_short_goal_buffers(env)
    num = env_ids.numel()
    distance_min = _env_float("RANGER_GOAL_DISTANCE_MIN", float(distance_range[0]))
    distance_max = _env_float("RANGER_GOAL_DISTANCE_MAX", float(distance_range[1]))
    angle_min_deg = _env_float("RANGER_GOAL_ANGLE_MIN_DEG", float(torch.rad2deg(torch.tensor(float(heading_range[0]))).item()))
    angle_max_deg = _env_float("RANGER_GOAL_ANGLE_MAX_DEG", float(torch.rad2deg(torch.tensor(float(heading_range[1]))).item()))
    if distance_max < distance_min:
        distance_min, distance_max = distance_max, distance_min
    if angle_max_deg < angle_min_deg:
        angle_min_deg, angle_max_deg = angle_max_deg, angle_min_deg
    heading_min = math.radians(angle_min_deg)
    heading_max = math.radians(angle_max_deg)
    distance = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
        float(distance_min), float(distance_max)
    )
    heading = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
        float(heading_min), float(heading_max)
    )
    target_vec_b = torch.zeros((num, 3), device=env.device, dtype=torch.float32)
    target_vec_b[:, 0] = distance * torch.cos(heading)
    target_vec_b[:, 1] = distance * torch.sin(heading)
    target_vec_w = math_utils.quat_apply_yaw(asset.data.root_quat_w[env_ids], target_vec_b)
    target_pos_w[env_ids] = asset.data.root_pos_w[env_ids] + target_vec_w
    prev_goal_distance[env_ids] = distance
    goal_reached[env_ids] = False
    prev_heading_error = getattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, None)
    if prev_heading_error is not None and prev_heading_error.shape == (env.num_envs,):
        prev_heading_error[env_ids] = float("nan")


def reset_short_goal_turn_target(
    env: ManagerBasedEnv,
    env_ids,
    distance_range: tuple[float, float] = (1.0, 1.5),
    left_heading_range_deg: tuple[float, float] = (25.0, 45.0),
    right_heading_range_deg: tuple[float, float] = (-45.0, -25.0),
    paired_sides: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample a side-only short-goal target for differential turning practice."""

    env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_w, prev_goal_distance, goal_reached = _ensure_short_goal_buffers(env)
    num = env_ids.numel()

    distance = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
        float(distance_range[0]), float(distance_range[1])
    )
    if paired_sides:
        left_mask = torch.remainder(env_ids, 2) == 0
    else:
        left_mask = torch.rand((num,), device=env.device) < 0.5
    heading = torch.empty(num, device=env.device, dtype=torch.float32)

    if torch.any(left_mask):
        heading[left_mask] = torch.empty(int(left_mask.sum().item()), device=env.device, dtype=torch.float32).uniform_(
            math.radians(float(left_heading_range_deg[0])),
            math.radians(float(left_heading_range_deg[1])),
        )
    right_mask = ~left_mask
    if torch.any(right_mask):
        heading[right_mask] = torch.empty(
            int(right_mask.sum().item()), device=env.device, dtype=torch.float32
        ).uniform_(
            math.radians(float(right_heading_range_deg[0])),
            math.radians(float(right_heading_range_deg[1])),
        )

    target_vec_b = torch.zeros((num, 3), device=env.device, dtype=torch.float32)
    target_vec_b[:, 0] = distance * torch.cos(heading)
    target_vec_b[:, 1] = distance * torch.sin(heading)
    target_vec_w = math_utils.quat_apply_yaw(asset.data.root_quat_w[env_ids], target_vec_b)
    target_pos_w[env_ids] = asset.data.root_pos_w[env_ids] + target_vec_w
    prev_goal_distance[env_ids] = distance
    goal_reached[env_ids] = False

    prev_heading_error = getattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, None)
    if prev_heading_error is None or prev_heading_error.shape != (env.num_envs,):
        prev_heading_error = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
        setattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, prev_heading_error)
    prev_heading_error[env_ids] = heading


def _sample_weighted_uniform_bands(
    num: int,
    bands: tuple[tuple[float, float], ...],
    weights: tuple[float, ...],
    device: str,
) -> torch.Tensor:
    """Sample uniformly inside one of several weighted scalar bands."""

    if not bands:
        raise ValueError("At least one sampling band is required.")
    if len(weights) != len(bands):
        raise ValueError(f"Expected {len(bands)} band weights, got {len(weights)}.")
    bounds = torch.tensor(bands, device=device, dtype=torch.float32)
    if torch.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(f"Each sampling band must satisfy high > low, got {bands}.")
    probabilities = torch.tensor(weights, device=device, dtype=torch.float32)
    if torch.any(probabilities < 0.0) or float(probabilities.sum().item()) <= 0.0:
        raise ValueError(f"Band weights must be non-negative with positive sum, got {weights}.")
    probabilities = probabilities / probabilities.sum()
    band_ids = torch.multinomial(probabilities, num_samples=num, replacement=True)
    selected = bounds[band_ids]
    return selected[:, 0] + torch.rand((num,), device=device) * (selected[:, 1] - selected[:, 0])


def reset_short_goal_stratified_target(
    env: ManagerBasedEnv,
    env_ids,
    distance_bands: tuple[tuple[float, float], ...] = ((5.0, 8.0),),
    distance_weights: tuple[float, ...] = (1.0,),
    heading_bands_deg: tuple[tuple[float, float], ...] = ((-45.0, -25.0), (25.0, 45.0)),
    heading_weights: tuple[float, ...] = (0.5, 0.5),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample short-goal distance and heading from weighted curriculum bands."""

    env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_w, prev_goal_distance, goal_reached = _ensure_short_goal_buffers(env)
    num = env_ids.numel()
    distance = _sample_weighted_uniform_bands(num, distance_bands, distance_weights, env.device)
    heading_deg = _sample_weighted_uniform_bands(num, heading_bands_deg, heading_weights, env.device)
    heading = torch.deg2rad(heading_deg)

    target_vec_b = torch.zeros((num, 3), device=env.device, dtype=torch.float32)
    target_vec_b[:, 0] = distance * torch.cos(heading)
    target_vec_b[:, 1] = distance * torch.sin(heading)
    target_vec_w = math_utils.quat_apply_yaw(asset.data.root_quat_w[env_ids], target_vec_b)
    target_pos_w[env_ids] = asset.data.root_pos_w[env_ids] + target_vec_w
    prev_goal_distance[env_ids] = distance
    goal_reached[env_ids] = False

    prev_heading_error = getattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, None)
    if prev_heading_error is None or prev_heading_error.shape != (env.num_envs,):
        prev_heading_error = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
        setattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, prev_heading_error)
    prev_heading_error[env_ids] = heading


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


def short_goal_target_pos_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Return sampled short-goal target positions in world frame."""

    return _ensure_short_goal_target(env)


def short_goal_target_body(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return short-goal vector, distance, and heading error in the robot body frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_w = _ensure_short_goal_target(env)
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


def turn_to_target_goal_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the raw six-dimensional turn-to-target descriptor in the body frame."""

    target_vec_b, distance, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
    obs = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
    obs[:, 0] = target_vec_b[:, 0]
    obs[:, 1] = target_vec_b[:, 1]
    obs[:, 2] = distance
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

    _obs_debug("enter critic_privileged/base_lin_vel")
    if _critic_privileged_disabled():
        out = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit critic_privileged/base_lin_vel disabled shape={tuple(out.shape)}")
        return out
    asset: Articulation = env.scene[asset_cfg.name]
    out = torch.clamp(asset.data.root_lin_vel_b / scale, min=-1.0, max=1.0)
    _obs_debug(f"exit critic_privileged/base_lin_vel shape={tuple(out.shape)}")
    return out


def base_ang_vel_normalized(
    env: ManagerBasedEnv,
    scale: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return normalized base angular velocity in the body frame."""

    _obs_debug("enter policy_state/base_ang_vel")
    if _policy_state_disabled():
        out = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit policy_state/base_ang_vel disabled shape={tuple(out.shape)}")
        return out
    asset: Articulation = env.scene[asset_cfg.name]
    out = torch.clamp(asset.data.root_ang_vel_b / scale, min=-1.0, max=1.0)
    _obs_debug(f"exit policy_state/base_ang_vel shape={tuple(out.shape)}")
    return out


def projected_gravity_normalized(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return clipped projected gravity vector."""

    _obs_debug("enter policy_state/projected_gravity")
    if _policy_state_disabled():
        out = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit policy_state/projected_gravity disabled shape={tuple(out.shape)}")
        return out
    asset: Articulation = env.scene[asset_cfg.name]
    out = torch.clamp(asset.data.projected_gravity_b, min=-1.0, max=1.0)
    _obs_debug(f"exit policy_state/projected_gravity shape={tuple(out.shape)}")
    return out


def joint_pos_rel_normalized(
    env: ManagerBasedEnv,
    scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return normalized relative joint positions for the selected joints."""

    _obs_debug(f"enter policy_state/joint_pos_rel joint_count={len(asset_cfg.joint_ids)}")
    if _policy_state_disabled():
        out = torch.zeros((env.num_envs, len(asset_cfg.joint_ids)), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit policy_state/joint_pos_rel disabled shape={tuple(out.shape)}")
        return out
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    out = torch.clamp(joint_pos_rel / scale, min=-1.0, max=1.0)
    _obs_debug(f"exit policy_state/joint_pos_rel shape={tuple(out.shape)}")
    return out


def joint_vel_rel_normalized(
    env: ManagerBasedEnv,
    scale: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return normalized relative joint velocities for the selected joints."""

    _obs_debug(f"enter policy_state/joint_vel_rel joint_count={len(asset_cfg.joint_ids)}")
    if _policy_state_disabled():
        out = torch.zeros((env.num_envs, len(asset_cfg.joint_ids)), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit policy_state/joint_vel_rel disabled shape={tuple(out.shape)}")
        return out
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_rel = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    out = torch.clamp(joint_vel_rel / scale, min=-1.0, max=1.0)
    _obs_debug(f"exit policy_state/joint_vel_rel shape={tuple(out.shape)}")
    return out


def hydraulic_stroke_state(env: ManagerBasedEnv, action_name: str = "leg_hydraulic") -> torch.Tensor:
    """Return the current equivalent hydraulic-cylinder stroke state."""

    action_term = env.action_manager.get_term(action_name)
    if not hasattr(action_term, "stroke_actual"):
        raise AttributeError(f"Action term '{action_name}' does not expose 'stroke_actual'.")
    return action_term.stroke_actual


def suspension_stroke_state(env: ManagerBasedEnv, action_name: str = "leg_hydraulic") -> torch.Tensor:
    """Return the measured suspension/EHA stroke in normalized physical stroke units."""

    _obs_debug("enter policy_state/suspension_stroke")
    if _policy_state_disabled():
        out = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit policy_state/suspension_stroke disabled shape={tuple(out.shape)}")
        return out
    out = hydraulic_stroke_state(env=env, action_name=action_name)
    _obs_debug(f"exit policy_state/suspension_stroke shape={tuple(out.shape)}")
    return out


def suspension_stroke_rate_state(
    env: ManagerBasedEnv,
    action_name: str = "leg_hydraulic",
    clip: float | None = 5.0,
) -> torch.Tensor:
    """Return per-step suspension stroke rate using a per-env cached finite difference."""

    _obs_debug("enter policy_state/suspension_stroke_rate")
    if _policy_state_disabled():
        out = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit policy_state/suspension_stroke_rate disabled shape={tuple(out.shape)}")
        return out
    stroke = suspension_stroke_state(env=env, action_name=action_name)
    prev_stroke = getattr(env, SUSPENSION_STROKE_PREV_ATTR, None)
    stroke_rate = getattr(env, SUSPENSION_STROKE_RATE_ATTR, None)
    last_step = getattr(env, SUSPENSION_STROKE_RATE_STEP_ATTR, None)

    if (
        prev_stroke is None
        or stroke_rate is None
        or prev_stroke.shape != stroke.shape
        or stroke_rate.shape != stroke.shape
        or last_step is None
    ):
        prev_stroke = stroke.clone()
        stroke_rate = torch.zeros_like(stroke)
        setattr(env, SUSPENSION_STROKE_PREV_ATTR, prev_stroke)
        setattr(env, SUSPENSION_STROKE_RATE_ATTR, stroke_rate)
        setattr(env, SUSPENSION_STROKE_RATE_STEP_ATTR, int(env.common_step_counter))
        _obs_debug(f"exit policy_state/suspension_stroke_rate init shape={tuple(stroke_rate.shape)}")
        return stroke_rate

    current_step = int(env.common_step_counter)
    if int(last_step) != current_step:
        dt = max(float(getattr(env, "step_dt", 1.0 / 60.0)), 1.0e-6)
        stroke_rate[:] = (stroke - prev_stroke) / dt
        reset_mask = env.episode_length_buf == 0
        if torch.any(reset_mask):
            stroke_rate[reset_mask] = 0.0
        if clip is not None:
            stroke_rate[:] = torch.clamp(stroke_rate, min=-float(clip), max=float(clip))
        prev_stroke[:] = stroke
        setattr(env, SUSPENSION_STROKE_RATE_STEP_ATTR, current_step)
    else:
        reset_mask = env.episode_length_buf == 0
        if torch.any(reset_mask):
            stroke_rate[reset_mask] = 0.0
            prev_stroke[reset_mask] = stroke[reset_mask]

    _obs_debug(f"exit policy_state/suspension_stroke_rate shape={tuple(stroke_rate.shape)}")
    return stroke_rate


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


def command_observation(
    env: ManagerBasedEnv,
    command_mode: str = "zero",
    goal_source: str = "none",
    stage: str = "A",
    v_x_range: tuple[float, float] = (0.05, 0.4),
    yaw_rate_range: tuple[float, float] = (-0.4, 0.4),
    command_duration_range: tuple[float, float] = (2.0, 5.0),
    smoothing_alpha: float = 0.1,
    max_command_duration: float = 5.0,
    small_yaw_prob: float = 0.0,
    small_yaw_range: tuple[float, float] = (0.03, 0.08),
    v_x_bins: tuple[tuple[float, float], ...] | None = None,
    goal_x_body: float = 0.0,
    goal_y_body: float = 0.0,
    short_goal_encoding_mode: str = "legacy",
    short_goal_observation_max_distance: float | None = None,
    short_goal_near_distance_range: float = 3.0,
    short_goal_global_distance_unit: float = 1.0,
    short_goal_velocity_reference: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the fixed eight-dimensional command/goal observation for actor state."""

    _obs_debug("enter policy_state/command_state")
    if _policy_state_disabled():
        out = torch.zeros((env.num_envs, 8), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit policy_state/command_state disabled shape={tuple(out.shape)}")
        return out

    del max_command_duration

    current_step = int(env.common_step_counter)
    cached_obs = getattr(env, COMMAND_OBS_CACHE_ATTR, None)
    cached_step = getattr(env, COMMAND_OBS_CACHE_STEP_ATTR, None)
    if (
        cached_obs is not None
        and cached_obs.shape == (env.num_envs, 8)
        and cached_step == current_step
        and not torch.any(env.episode_length_buf == 0)
    ):
        _obs_debug(f"exit policy_state/command_state cached shape={tuple(cached_obs.shape)}")
        return cached_obs

    obs = torch.zeros((env.num_envs, 8), device=env.device, dtype=torch.float32)

    command_mode = command_mode.lower()
    if command_mode == "speed_command":
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
        obs[:, 0] = command[:, 0]
        obs[:, 1] = 0.0
        obs[:, 2] = command[:, 1]
    elif command_mode == "yaw_rate_command":
        command = yaw_rate_command(env)
        obs[:, 0] = 0.0
        obs[:, 1] = 0.0
        obs[:, 2] = command
    elif command_mode != "zero":
        raise ValueError(f"Unsupported command_mode: {command_mode}")

    goal_source = goal_source.lower()
    if goal_source == "dynamic":
        target_vec_b, distance, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
        obs[:, 3] = target_vec_b[:, 0]
        obs[:, 4] = target_vec_b[:, 1]
        obs[:, 5] = distance
        obs[:, 6] = torch.sin(heading_error)
        obs[:, 7] = torch.cos(heading_error)
    elif goal_source == "short_goal":
        target_vec_b, distance, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
        stop_phase_active = getattr(env, "_short_goal_stop_phase_active", None)
        if stop_phase_active is not None and stop_phase_active.shape == (env.num_envs,):
            # ShortGoal does not use the three command slots; reuse slot 2 for the
            # latched stop-phase flag without changing the observation dimension.
            obs[:, 2] = stop_phase_active.to(obs.dtype)

        encoding_mode = short_goal_encoding_mode.lower()
        if encoding_mode == "long_term":
            distance_safe = torch.clamp(distance, min=1.0e-6)
            goal_direction = target_vec_b[:, :2] / distance_safe.unsqueeze(1)
            near_range = max(float(short_goal_near_distance_range), 1.0e-6)
            global_distance_unit = max(float(short_goal_global_distance_unit), 1.0e-6)
            velocity_reference = max(float(short_goal_velocity_reference), 1.0e-6)
            near_distance = torch.clamp(distance / near_range, min=0.0, max=1.0)
            global_log_distance = torch.log2(1.0 + distance / global_distance_unit)

            asset: Articulation = env.scene[asset_cfg.name]
            velocity_toward_goal = torch.sum(asset.data.root_lin_vel_b[:, :2] * goal_direction, dim=1)
            velocity_toward_goal = torch.clamp(
                velocity_toward_goal / velocity_reference,
                min=-1.0,
                max=1.0,
            )

            # Fixed long-term encoding:
            # [goal_dir_x, goal_dir_y, near_distance, log2(1 + distance / unit), velocity_toward_goal].
            obs[:, 3] = goal_direction[:, 0]
            obs[:, 4] = goal_direction[:, 1]
            obs[:, 5] = near_distance
            obs[:, 6] = global_log_distance
            obs[:, 7] = velocity_toward_goal
        elif encoding_mode == "legacy":
            target_vec_obs = target_vec_b[:, :2]
            distance_obs = distance
            if short_goal_observation_max_distance is not None:
                max_distance = max(float(short_goal_observation_max_distance), 1.0e-6)
                radial_scale = torch.clamp(max_distance / torch.clamp(distance, min=1.0e-6), max=1.0)
                target_vec_obs = target_vec_obs * radial_scale.unsqueeze(1)
                distance_obs = torch.clamp(distance, max=max_distance)

            obs[:, 3] = target_vec_obs[:, 0]
            obs[:, 4] = target_vec_obs[:, 1]
            obs[:, 5] = distance_obs
            obs[:, 6] = torch.sin(heading_error)
            obs[:, 7] = torch.cos(heading_error)
        else:
            raise ValueError(f"Unsupported short_goal_encoding_mode: {short_goal_encoding_mode}")
    elif goal_source == "fixed":
        goal_x_body_tensor = torch.full((env.num_envs,), float(goal_x_body), device=env.device)
        goal_y_body_tensor = torch.full((env.num_envs,), float(goal_y_body), device=env.device)
        distance = torch.sqrt(goal_x_body_tensor.square() + goal_y_body_tensor.square())
        heading_error = torch.atan2(goal_y_body_tensor, goal_x_body_tensor)
        obs[:, 3] = goal_x_body_tensor
        obs[:, 4] = goal_y_body_tensor
        obs[:, 5] = distance
        obs[:, 6] = torch.sin(heading_error)
        obs[:, 7] = torch.cos(heading_error)
    elif goal_source == "none":
        obs[:, 7] = 1.0
    else:
        raise ValueError(f"Unsupported goal_source: {goal_source}")

    setattr(env, COMMAND_OBS_CACHE_ATTR, obs)
    setattr(env, COMMAND_OBS_CACHE_STEP_ATTR, current_step)
    _obs_debug(f"exit policy_state/command_state shape={tuple(obs.shape)}")
    return obs


def last_action_normalized(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """Return clipped previous action history."""

    _obs_debug("enter policy_state/previous_action")
    if _policy_state_disabled():
        out = torch.zeros((env.num_envs, 8), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit policy_state/previous_action disabled shape={tuple(out.shape)}")
        return out
    if action_name is None:
        out = torch.clamp(env.action_manager.action, min=-1.0, max=1.0)
    else:
        out = torch.clamp(env.action_manager.get_term(action_name).raw_actions, min=-1.0, max=1.0)
    _obs_debug(f"exit policy_state/previous_action shape={tuple(out.shape)}")
    return out


def _wheel_contact_sensor_body_ids(env: ManagerBasedEnv, sensor_name: str = "wheel_contact_forces") -> torch.Tensor:
    cached_body_ids = getattr(env, WHEEL_CONTACT_SENSOR_BODY_IDS_ATTR, None)
    if cached_body_ids is not None and cached_body_ids.numel() == 4:
        return cached_body_ids

    contact_sensor = env.scene.sensors[sensor_name]
    body_ids, _ = contact_sensor.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
    cached_body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
    setattr(env, WHEEL_CONTACT_SENSOR_BODY_IDS_ATTR, cached_body_ids)
    return cached_body_ids


def wheel_contact_force_over_weight(
    env: ManagerBasedEnv,
    sensor_name: str = "wheel_contact_forces",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return wheel-contact force norm per wheel normalized by total robot weight."""

    _obs_debug("enter critic_privileged/wheel_contact_force")
    if _critic_privileged_disabled():
        out = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit critic_privileged/wheel_contact_force disabled shape={tuple(out.shape)}")
        return out
    contact_sensor = env.scene.sensors[sensor_name]
    body_ids = _wheel_contact_sensor_body_ids(env, sensor_name=sensor_name)
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, body_ids, :]
    contact_force = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]

    asset: Articulation = env.scene[asset_cfg.name]
    expected_weight = torch.sum(asset.root_physx_view.get_masses(), dim=1).to(env.device) * 9.81
    out = contact_force / torch.clamp(expected_weight.unsqueeze(1), min=1.0e-6)
    _obs_debug(f"exit critic_privileged/wheel_contact_force shape={tuple(out.shape)}")
    return out


def wheel_contact_bool(
    env: ManagerBasedEnv,
    sensor_name: str = "wheel_contact_forces",
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return binary wheel contact state in semantic ``[lr, lf, rf, rr]`` sensor order."""

    _obs_debug("enter critic_privileged/wheel_contact_bool")
    if _critic_privileged_disabled():
        out = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit critic_privileged/wheel_contact_bool disabled shape={tuple(out.shape)}")
        return out
    del asset_cfg
    contact_sensor = env.scene.sensors[sensor_name]
    body_ids = _wheel_contact_sensor_body_ids(env, sensor_name=sensor_name)
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, body_ids, :]
    force_norm = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
    out = (force_norm > float(threshold)).to(torch.float32)
    _obs_debug(f"exit critic_privileged/wheel_contact_bool shape={tuple(out.shape)}")
    return out


def root_height_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root height used by the environment in world z coordinates."""

    _obs_debug("enter critic_privileged/root_height")
    if _critic_privileged_disabled():
        out = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit critic_privileged/root_height disabled shape={tuple(out.shape)}")
        return out
    asset: Articulation = env.scene[asset_cfg.name]
    out = asset.data.root_pos_w[:, 2:3]
    _obs_debug(f"exit critic_privileged/root_height shape={tuple(out.shape)}")
    return out


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

    _obs_debug("enter policy_map/local_navigation_map_layers")
    if _policy_map_disabled():
        num_x, num_y = _navigation_map_grid_shape(x_range=x_range, y_range=y_range, resolution=resolution)
        zeros = torch.zeros((env.num_envs, num_x, num_y), device=env.device, dtype=torch.float32)
        out = {
            "height": zeros,
            "slope": zeros,
            "roughness": zeros,
            "step": zeros,
            "geometric_traversability": zeros,
            "valid_mask": zeros,
            "semantic_traversability": zeros,
            "confidence": zeros,
            "traversability": zeros,
        }
        _obs_debug(
            "exit policy_map/local_navigation_map_layers disabled "
            f"grid_shape={tuple(out['height'].shape)}"
        )
        return out

    if use_neutral_map:
        out = _build_neutral_navigation_map_layers(
            env=env,
            x_range=x_range,
            y_range=y_range,
            resolution=resolution,
        )
        _obs_debug(
            "exit policy_map/local_navigation_map_layers neutral "
            f"grid_shape={tuple(out['height'].shape)}"
        )
        return out

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
    _obs_debug("enter policy_map/geometric_traversability")
    geometric_traversability = _compute_traversability_map(
        slope_map=slope_map,
        roughness_map=roughness_map,
        step_map=step_map,
        valid_mask=valid_mask,
        slope_weight=slope_weight,
        roughness_weight=roughness_weight,
        step_weight=step_weight,
        unknown_penalty=unknown_penalty,
    )
    _obs_debug(f"exit policy_map/geometric_traversability shape={tuple(geometric_traversability.shape)}")
    semantic_traversability = torch.zeros_like(geometric_traversability)
    confidence = valid_mask.to(height_map.dtype)
    _obs_debug(f"policy_map/valid_mask shape={tuple(valid_mask.shape)}")
    _obs_debug(f"policy_map/confidence shape={tuple(confidence.shape)}")

    out = {
        "height": height_map,
        "slope": slope_map,
        "roughness": roughness_map,
        "step": step_map,
        "geometric_traversability": geometric_traversability,
        "valid_mask": valid_mask.to(height_map.dtype),
        "semantic_traversability": semantic_traversability,
        "confidence": confidence,
        # Backward-compatible alias for existing debug utilities that still reference the old name.
        "traversability": geometric_traversability,
    }
    _obs_debug(f"exit policy_map/local_navigation_map_layers grid_shape={tuple(height_map.shape)}")
    return out


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
    """Build the fixed eight-channel local navigation map used by the V1 terrain-CNN policy."""

    _obs_debug("enter policy_map")
    if _policy_map_disabled():
        out = torch.zeros((env.num_envs, 2184), device=env.device, dtype=torch.float32)
        _obs_debug(f"exit policy_map disabled shape={tuple(out.shape)}")
        return out
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
            layers_dict["geometric_traversability"],
            layers_dict["valid_mask"],
            layers_dict["semantic_traversability"],
            layers_dict["confidence"],
        ),
        dim=1,
    )
    out = layers.reshape(env.num_envs, -1)
    _obs_debug(f"exit policy_map shape={tuple(out.shape)}")
    return out


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
        "geometric_traversability": ones,
        "valid_mask": ones,
        "semantic_traversability": zeros,
        "confidence": ones,
        "traversability": ones,
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

    _obs_debug("enter policy_map/_compute_traversability_map")
    valid_mask_float = valid_mask.to(slope_map.dtype)
    cost = (
        slope_weight * slope_map
        + roughness_weight * roughness_map
        + step_weight * step_map
        + unknown_penalty * (1.0 - valid_mask_float)
    )
    cost = torch.clamp(cost, min=0.0, max=1.0)
    traversability = 1.0 - cost
    out = torch.where(valid_mask, traversability, torch.zeros_like(traversability))
    _obs_debug(f"exit policy_map/_compute_traversability_map shape={tuple(out.shape)}")
    return out


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
