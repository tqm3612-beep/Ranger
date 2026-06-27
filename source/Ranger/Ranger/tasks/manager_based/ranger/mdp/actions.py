# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action terms for the Ranger task."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedEnv


class WheelMotorCSVAction(ActionTerm):
    """Equivalent wheel-motor action for cyclic synchronous velocity control.

    Assumption for this project version:
    the policy outputs a normalized wheel-velocity command. The action term
    filters the command with a first-order response and acceleration limit, then
    closes a local velocity loop that converts the velocity target into joint
    torque sent to the wheel joints.
    """

    cfg: "WheelMotorCSVActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "WheelMotorCSVActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints == 0:
            raise ValueError(f"No joints matched {self.cfg.joint_names} for WheelMotorCSVAction.")

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._velocity_target = torch.zeros_like(self._raw_actions)
        self._wheel_forward_sign = torch.tensor(
            [-1.0, -1.0, 1.0, 1.0],
            device=self._raw_actions.device,
            dtype=self._raw_actions.dtype,
        ).unsqueeze(0)
        self._torque_actual = torch.zeros_like(self._raw_actions)

        self._velocity_limit = float(self.cfg.velocity_limit)
        self._acceleration_limit = float(self.cfg.acceleration_limit)
        self._command_time_constant = float(self.cfg.command_time_constant)
        self._velocity_kp = float(self.cfg.velocity_kp)
        self._velocity_damping = float(self.cfg.velocity_damping)
        self._viscous_friction = float(self.cfg.viscous_friction)
        self._effort_limit = float(self.cfg.effort_limit)

        if self._velocity_limit <= 0.0 or self._acceleration_limit <= 0.0 or self._effort_limit <= 0.0:
            raise ValueError("Wheel motor limits must be positive.")
        if self._command_time_constant < 0.0:
            raise ValueError("Wheel motor command_time_constant must be non-negative.")
        if self._velocity_kp <= 0.0:
            raise ValueError("Wheel motor velocity_kp must be positive.")

        self._clip = _build_clip_tensor(
            clip=self.cfg.clip,
            joint_names=self._joint_names,
            num_envs=self.num_envs,
            action_dim=self.action_dim,
            device=self.device,
        )

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Wheel joint torques sent to the simulator."""
        return self._processed_actions

    @property
    def velocity_target(self) -> torch.Tensor:
        """Equivalent CSV velocity target state."""
        return self._velocity_target

    @property
    def torque_actual(self) -> torch.Tensor:
        """Equivalent wheel torque command after the local velocity loop."""
        return self._torque_actual

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        if self._clip is not None:
            self._raw_actions[:] = torch.clamp(self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

        raw_wheel_actions = torch.clamp(self._raw_actions, min=-1.0, max=1.0)
        left_cmd = raw_wheel_actions[:, 0:2].mean(dim=1, keepdim=True)
        right_cmd = raw_wheel_actions[:, 2:4].mean(dim=1, keepdim=True)
        semantic_wheel_cmd = torch.cat([left_cmd, left_cmd, right_cmd, right_cmd], dim=1)
        velocity_des = semantic_wheel_cmd * self._velocity_limit * self._wheel_forward_sign
        if self._command_time_constant > 0.0:
            alpha = self._env.step_dt / (self._command_time_constant + self._env.step_dt)
            velocity_cmd = self._velocity_target + alpha * (velocity_des - self._velocity_target)
        else:
            velocity_cmd = velocity_des

        wheel_speed = self._asset.data.joint_vel[:, self._joint_ids]
        max_delta = self._acceleration_limit * self._env.step_dt
        velocity_delta = torch.clamp(velocity_cmd - self._velocity_target, min=-max_delta, max=max_delta)
        self._velocity_target[:] = torch.clamp(
            self._velocity_target + velocity_delta,
            min=-self._velocity_limit,
            max=self._velocity_limit,
        )

        velocity_error = self._velocity_target - wheel_speed
        torque = self._velocity_kp * velocity_error - self._velocity_damping * wheel_speed
        torque = torque - self._viscous_friction * wheel_speed
        self._torque_actual[:] = torch.clamp(torque, min=-self._effort_limit, max=self._effort_limit)
        self._processed_actions[:] = self._torque_actual

    def apply_actions(self) -> None:
        self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._velocity_target[env_ids] = 0.0
        self._torque_actual[env_ids] = 0.0


class HydraulicActuatorAction(ActionTerm):
    """Equivalent hydraulic actuator action using position targets.

    Policy outputs normalized suspension commands. The action term maps them to
    physical desired cylinder strokes ``stroke_des``, optionally filters and
    rate-limits the command, and maps the stroke command to equivalent g_*
    joint position targets. ``joint_position_sign`` is applied per joint so
    larger ``stroke_des`` always means raising the corresponding wheel in the
    unified policy semantics. The existing implicit actuator executes the low-
    level position/impedance behavior in simulation.
    """

    cfg: "HydraulicActuatorActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "HydraulicActuatorActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints == 0:
            raise ValueError(f"No joints matched {self.cfg.joint_names} for HydraulicActuatorAction.")

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._stroke_des = torch.zeros_like(self._raw_actions)
        self._stroke_command = torch.zeros_like(self._raw_actions)
        self._position_target = torch.zeros_like(self._raw_actions)
        self._effort_actual = torch.zeros_like(self._raw_actions)

        self._stroke_table = torch.tensor(self.cfg.stroke_table, dtype=torch.float32, device=self.device)
        self._joint_pos_table = torch.tensor(self.cfg.joint_pos_table, dtype=torch.float32, device=self.device)
        self._joint_position_sign = torch.tensor(
            self.cfg.joint_position_sign, dtype=torch.float32, device=self.device
        ).view(1, -1)
        if self._stroke_table.ndim != 1 or self._joint_pos_table.ndim != 1:
            raise ValueError("stroke_table and joint_pos_table must be one-dimensional.")
        if self._stroke_table.numel() != self._joint_pos_table.numel():
            raise ValueError("stroke_table and joint_pos_table must have the same length.")
        if self._stroke_table.numel() < 2:
            raise ValueError("stroke_table and joint_pos_table must contain at least two points.")
        if not torch.all(self._stroke_table[1:] > self._stroke_table[:-1]):
            raise ValueError("stroke_table must be strictly increasing.")
        if self._joint_position_sign.shape[1] != self._num_joints:
            raise ValueError(
                "joint_position_sign must have the same length as the number of matched hydraulic joints."
            )

        self._stroke_min = float(self.cfg.stroke_min)
        self._stroke_max = float(self.cfg.stroke_max)
        self._stroke_rate_limit = float(self.cfg.stroke_rate_limit)
        self._time_constant = float(self.cfg.time_constant)
        self._max_effort = float(self.cfg.max_effort)
        self._impedance_kp = float(self.cfg.impedance_kp)
        self._impedance_kd = float(self.cfg.impedance_kd)

        if self._stroke_max <= self._stroke_min:
            raise ValueError("stroke_max must be greater than stroke_min.")
        if self._stroke_rate_limit <= 0.0:
            raise ValueError("stroke_rate_limit must be positive.")
        if self._time_constant < 0.0:
            raise ValueError("time_constant must be non-negative.")
        if self._max_effort <= 0.0:
            raise ValueError("max_effort must be positive.")

        self._stroke_mid = 0.5 * (self._stroke_min + self._stroke_max)
        self._stroke_half_range = 0.5 * (self._stroke_max - self._stroke_min)
        self._stroke_des[:] = self._stroke_mid
        self._stroke_command[:] = self._stroke_mid
        self._position_target[:] = self._interp_stroke_to_joint_pos(self._stroke_command) * self._joint_position_sign

        self._clip = _build_clip_tensor(
            clip=self.cfg.clip,
            joint_names=self._joint_names,
            num_envs=self.num_envs,
            action_dim=self.action_dim,
            device=self.device,
        )

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Current joint position targets applied on the equivalent hydraulic joints."""
        return self._processed_actions

    @property
    def stroke_des(self) -> torch.Tensor:
        """Current physical desired cylinder stroke commanded by the policy."""
        return self._stroke_des

    @property
    def stroke_command(self) -> torch.Tensor:
        """Current filtered and rate-limited stroke command used inside simulation."""
        return self._stroke_command

    @property
    def stroke_cmd(self) -> torch.Tensor:
        """Alias for the filtered stroke command."""
        return self._stroke_command

    @property
    def stroke_actual(self) -> torch.Tensor:
        """Backward-compatible alias for the filtered stroke command, not a measured physical stroke."""
        return self._stroke_command

    @property
    def position_target(self) -> torch.Tensor:
        """Current mapped joint position target from the stroke command."""
        return self._position_target

    @property
    def joint_position_sign(self) -> torch.Tensor:
        """Per-joint sign used to keep ``stroke_des`` semantics consistent across legs."""
        return self._joint_position_sign

    @property
    def effort_actual(self) -> torch.Tensor:
        """Compatibility tensor; not used in position-target mode."""
        return self._effort_actual

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        if self._clip is not None:
            self._raw_actions[:] = torch.clamp(self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

        stroke_des = self._stroke_mid + torch.clamp(self._raw_actions, min=-1.0, max=1.0) * self._stroke_half_range
        self._stroke_des[:] = torch.clamp(stroke_des, min=self._stroke_min, max=self._stroke_max)

        if self._time_constant > 0.0:
            alpha = self._env.step_dt / (self._time_constant + self._env.step_dt)
            stroke_target = self._stroke_command + alpha * (self._stroke_des - self._stroke_command)
        else:
            stroke_target = self._stroke_des

        max_delta = self._stroke_rate_limit * self._env.step_dt
        stroke_delta = torch.clamp(stroke_target - self._stroke_command, min=-max_delta, max=max_delta)
        self._stroke_command[:] = torch.clamp(
            self._stroke_command + stroke_delta, min=self._stroke_min, max=self._stroke_max
        )
        base_position_target = self._interp_stroke_to_joint_pos(self._stroke_command)
        position_target = base_position_target * self._joint_position_sign
        joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._position_target[:] = torch.clamp(position_target, min=joint_limits[..., 0], max=joint_limits[..., 1])
        self._effort_actual.zero_()
        self._processed_actions[:] = self._position_target

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._effort_actual[env_ids] = 0.0
        self._stroke_des[env_ids] = self._stroke_mid
        self._stroke_command[env_ids] = self._stroke_mid
        self._position_target[env_ids] = (
            self._interp_stroke_to_joint_pos(self._stroke_command[env_ids]) * self._joint_position_sign
        )

    def _interp_stroke_to_joint_pos(self, stroke: torch.Tensor) -> torch.Tensor:
        stroke_clamped = torch.clamp(stroke, min=self._stroke_table[0], max=self._stroke_table[-1])
        flat_stroke = stroke_clamped.reshape(-1)
        upper_ids = torch.bucketize(flat_stroke, self._stroke_table)
        upper_ids = torch.clamp(upper_ids, min=1, max=self._stroke_table.numel() - 1)
        lower_ids = upper_ids - 1

        stroke_lower = self._stroke_table[lower_ids]
        stroke_upper = self._stroke_table[upper_ids]
        joint_lower = self._joint_pos_table[lower_ids]
        joint_upper = self._joint_pos_table[upper_ids]
        ratio = (flat_stroke - stroke_lower) / (stroke_upper - stroke_lower)
        joint_pos = joint_lower + ratio * (joint_upper - joint_lower)
        return joint_pos.reshape_as(stroke)


@configclass
class WheelMotorCSVActionCfg(ActionTermCfg):
    """Configuration for :class:`WheelMotorCSVAction`."""

    class_type: type[ActionTerm] = WheelMotorCSVAction

    joint_names: list[str] = MISSING
    preserve_order: bool = False

    velocity_limit: float = 20.0
    acceleration_limit: float = 80.0
    command_time_constant: float = 0.02
    velocity_kp: float = 10.0
    velocity_damping: float = 0.2
    viscous_friction: float = 0.05
    effort_limit: float = 100.0


@configclass
class HydraulicActuatorActionCfg(ActionTermCfg):
    """Configuration for :class:`HydraulicActuatorAction`."""

    class_type: type[ActionTerm] = HydraulicActuatorAction

    joint_names: list[str] = MISSING
    preserve_order: bool = False

    stroke_min: float = 0.0
    """Physical desired stroke lower bound."""
    stroke_max: float = 1.0
    """Physical desired stroke upper bound."""
    stroke_rate_limit: float = 1.0
    """Command safety limit applied to ``stroke_des`` before position-target mapping."""
    time_constant: float = 0.08
    """First-order command filter applied to ``stroke_des`` before position-target mapping."""

    stroke_table: tuple[float, ...] = (0.0, 0.5, 1.0)
    """Calibration lookup from desired cylinder stroke to equivalent joint angle."""
    joint_pos_table: tuple[float, ...] = (-1.0, 0.0, 1.0)
    """Equivalent joint-angle calibration corresponding to :attr:`stroke_table`."""
    joint_position_sign: tuple[float, ...] = (1.0, -1.0, 1.0, -1.0)
    """Per-joint sign that makes larger ``stroke_des`` mean raising the corresponding wheel."""

    max_effort: float = 300.0
    impedance_kp: float = 250.0
    """Fixed low-level impedance proportional gain."""
    impedance_kd: float = 30.0
    """Fixed low-level impedance derivative gain."""


def _build_clip_tensor(
    clip: dict[str, tuple[float, float]] | None,
    joint_names: list[str],
    num_envs: int,
    action_dim: int,
    device: str,
) -> torch.Tensor | None:
    """Build a per-joint clipping tensor for normalized actions."""

    if clip is None:
        return None
    clip_tensor = torch.tensor([[-float("inf"), float("inf")]], device=device).repeat(num_envs, action_dim, 1)
    index_list, _, value_list = string_utils.resolve_matching_names_values(clip, joint_names)
    clip_tensor[:, index_list] = torch.tensor(value_list, device=device)
    return clip_tensor
