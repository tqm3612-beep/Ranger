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

from ..wheel_semantics import ranger_wheel_semantic_to_joint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedEnv


class WheelMotorCSVAction(ActionTerm):
    """Equivalent wheel-motor action for cyclic synchronous velocity control.

    Assumption for this project version:
    the policy outputs a normalized wheel-velocity command. The action term
    filters the command with a first-order response and acceleration limit, then
    sends the resulting wheel-velocity target to the articulation through
    ``set_joint_velocity_target``.

    Deprecated compatibility note:
    older config fields such as ``control_mode``, ``velocity_kp``,
    ``velocity_damping``, ``viscous_friction``, and ``effort_limit`` are still
    accepted so existing task configs keep loading, but they no longer affect
    the control calculation. The local velocity loop / effort-control path is
    intentionally disabled.
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
        self._previous_velocity_target = torch.zeros_like(self._raw_actions)
        self._torque_actual = torch.zeros_like(self._raw_actions)

        self._velocity_limit = float(self.cfg.velocity_limit)
        self._acceleration_limit = float(self.cfg.acceleration_limit)
        self._command_time_constant = float(self.cfg.command_time_constant)
        self._velocity_kp = float(self.cfg.velocity_kp)  # deprecated / ignored
        self._velocity_damping = float(self.cfg.velocity_damping)  # deprecated / ignored
        self._viscous_friction = float(self.cfg.viscous_friction)  # deprecated / ignored
        self._effort_limit = float(self.cfg.effort_limit)  # deprecated / ignored
        requested_control_mode = str(self.cfg.control_mode).strip().lower()
        self._control_mode = "velocity"

        if self._velocity_limit <= 0.0 or self._acceleration_limit <= 0.0:
            raise ValueError("Wheel motor limits must be positive.")
        if self._command_time_constant < 0.0:
            raise ValueError("Wheel motor command_time_constant must be non-negative.")

        self._clip = _build_clip_tensor(
            clip=self.cfg.clip,
            joint_names=self._joint_names,
            num_envs=self.num_envs,
            action_dim=self.action_dim,
            device=self.device,
        )
        if requested_control_mode != "velocity":
            print(
                f"[WheelMotorCSVAction] requested control_mode={requested_control_mode!r} is deprecated; "
                "forcing velocity-target control.",
                flush=True,
            )
        print(
            f"[WheelMotorCSVAction] joints={self._joint_names} "
            f"control_mode={self._control_mode} "
            "target_mode=set_joint_velocity_target "
            f"velocity_limit={self._velocity_limit} "
            "local_velocity_loop=disabled",
            flush=True,
        )

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Wheel velocity target sent to the simulator."""
        return self._processed_actions

    @property
    def velocity_target(self) -> torch.Tensor:
        """Equivalent CSV velocity target state."""
        return self._velocity_target

    @property
    def previous_velocity_target(self) -> torch.Tensor:
        """Velocity target from the previous policy step."""
        return self._previous_velocity_target

    @property
    def torque_actual(self) -> torch.Tensor:
        """Deprecated debug tensor retained for compatibility; always zero."""
        return self._torque_actual

    @property
    def control_mode(self) -> str:
        return self._control_mode

    @property
    def uses_velocity_target(self) -> bool:
        return True

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        if self._clip is not None:
            self._raw_actions[:] = torch.clamp(self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

        self._previous_velocity_target[:] = self._velocity_target

        # Policy wheel actions are semantic [lb, lf, rf, rb] commands: positive means
        # that wheel should drive the vehicle forward. Convert to mirrored physical
        # Isaac-Sim joint coordinates only at this simulator boundary.
        semantic_wheel_actions = torch.clamp(self._raw_actions, min=-1.0, max=1.0)
        velocity_des = ranger_wheel_semantic_to_joint(semantic_wheel_actions) * self._velocity_limit
        if self._command_time_constant > 0.0:
            alpha = self._env.step_dt / (self._command_time_constant + self._env.step_dt)
            velocity_cmd = self._velocity_target + alpha * (velocity_des - self._velocity_target)
        else:
            velocity_cmd = velocity_des

        max_delta = self._acceleration_limit * self._env.step_dt
        velocity_delta = torch.clamp(velocity_cmd - self._velocity_target, min=-max_delta, max=max_delta)
        self._velocity_target[:] = torch.clamp(
            self._velocity_target + velocity_delta,
            min=-self._velocity_limit,
            max=self._velocity_limit,
        )
        self._torque_actual.zero_()
        self._processed_actions[:] = self._velocity_target

    def apply_actions(self) -> None:
        self._asset.set_joint_velocity_target(self._processed_actions, joint_ids=self._joint_ids)

    def _canonicalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).flatten()
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()

    def _validate_per_env_buffer(self, name: str, buffer: torch.Tensor, env_ids: torch.Tensor) -> None:
        if buffer.ndim != 2 or buffer.shape[0] != self.num_envs or buffer.shape[1] != self.action_dim:
            env_min = int(env_ids.min().item()) if env_ids.numel() > 0 else -1
            env_max = int(env_ids.max().item()) if env_ids.numel() > 0 else -1
            raise RuntimeError(
                f"{self.__class__.__name__}.{name} has invalid shape {tuple(buffer.shape)}; "
                f"expected ({self.num_envs}, {self.action_dim}), env_ids_min={env_min}, env_ids_max={env_max}"
            )
        if env_ids.numel() > 0 and (int(env_ids.min().item()) < 0 or int(env_ids.max().item()) >= buffer.shape[0]):
            raise RuntimeError(
                f"{self.__class__.__name__}.{name} env_ids out of bounds for shape {tuple(buffer.shape)}; "
                f"num_envs={self.num_envs}, env_ids_min={int(env_ids.min().item())}, "
                f"env_ids_max={int(env_ids.max().item())}"
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = self._canonicalize_env_ids(env_ids)
        self._validate_per_env_buffer("_raw_actions", self._raw_actions, env_ids)
        self._validate_per_env_buffer("_processed_actions", self._processed_actions, env_ids)
        self._validate_per_env_buffer("_velocity_target", self._velocity_target, env_ids)
        self._validate_per_env_buffer("_previous_velocity_target", self._previous_velocity_target, env_ids)
        self._validate_per_env_buffer("_torque_actual", self._torque_actual, env_ids)
        self._raw_actions[env_ids, :] = 0.0
        self._processed_actions[env_ids, :] = 0.0
        self._velocity_target[env_ids, :] = 0.0
        self._previous_velocity_target[env_ids, :] = 0.0
        self._torque_actual[env_ids, :] = 0.0


class HydraulicActuatorAction(ActionTerm):
    """Equivalent hydraulic actuator action using a fixed impedance controller.

    In this project version the policy outputs a normalized stroke target. The
    action term converts it to a joint-angle target through the stroke lookup,
    then applies impedance torque on the equivalent hydraulic joints.
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
        self._clipped_actions = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._stroke_desired = torch.zeros_like(self._raw_actions)
        self._stroke_actual = torch.zeros_like(self._raw_actions)
        self._position_target = torch.zeros_like(self._raw_actions)
        self._effort_actual = torch.zeros_like(self._raw_actions)

        self._stroke_table = torch.tensor(self.cfg.stroke_table, dtype=torch.float32, device=self.device)
        self._joint_pos_table = torch.tensor(self.cfg.joint_pos_table, dtype=torch.float32, device=self.device)
        self._joint_target_sign_lf_lr_rf_rr = torch.tensor(
            self.cfg.joint_target_sign, dtype=torch.float32, device=self.device
        )
        if self._stroke_table.ndim != 1 or self._joint_pos_table.ndim != 1:
            raise ValueError("stroke_table and joint_pos_table must be one-dimensional.")
        if self._stroke_table.numel() != self._joint_pos_table.numel():
            raise ValueError("stroke_table and joint_pos_table must have the same length.")
        if self._stroke_table.numel() < 2:
            raise ValueError("stroke_table and joint_pos_table must contain at least two points.")
        if not torch.all(self._stroke_table[1:] > self._stroke_table[:-1]):
            raise ValueError("stroke_table must be strictly increasing.")
        if self._joint_target_sign_lf_lr_rf_rr.numel() != self._num_joints:
            raise ValueError(
                "joint_target_sign must have the same length as the resolved hydraulic joints "
                f"({self._num_joints}), got {self._joint_target_sign_lf_lr_rf_rr.numel()}."
            )
        # Config uses semantic [lf, lr, rf, rr] order while the action term resolves
        # joints in raw articulation order [lr, lf, rf, rr].
        self._joint_target_sign = self._joint_target_sign_lf_lr_rf_rr[[1, 0, 2, 3]].unsqueeze(0)

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
        self._clipped_actions.zero_()
        self._stroke_desired[:] = self._stroke_mid
        self._stroke_actual[:] = self._stroke_mid
        self._position_target[:] = self._interp_stroke_to_joint_pos(self._stroke_actual) * self._joint_target_sign

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
        """Current simulator command buffer for the hydraulic action term."""
        return self._processed_actions

    @property
    def clipped_actions(self) -> torch.Tensor:
        """Normalized action after safety clipping and before stroke update."""
        return self._clipped_actions

    @property
    def stroke_desired(self) -> torch.Tensor:
        """Desired stroke before first-order and rate-limit dynamics."""
        return self._stroke_desired

    @property
    def stroke_actual(self) -> torch.Tensor:
        """Current virtual cylinder stroke after actuator dynamics."""
        return self._stroke_actual

    @property
    def position_target(self) -> torch.Tensor:
        """Current mapped joint position target from the stroke state."""
        return self._position_target

    @property
    def effort_actual(self) -> torch.Tensor:
        """Current equivalent effort command applied by the actuator term."""
        return self._effort_actual

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        clipped_actions = self._raw_actions
        if self._clip is not None:
            clipped_actions = torch.clamp(clipped_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        self._clipped_actions[:] = torch.clamp(clipped_actions, min=-1.0, max=1.0)

        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]

        stroke_des = self._stroke_mid + self._clipped_actions * self._stroke_half_range
        self._stroke_desired[:] = torch.clamp(stroke_des, min=self._stroke_min, max=self._stroke_max)

        if self._time_constant > 0.0:
            alpha = self._env.step_dt / (self._time_constant + self._env.step_dt)
            stroke_target = self._stroke_actual + alpha * (self._stroke_desired - self._stroke_actual)
        else:
            stroke_target = self._stroke_desired

        max_delta = self._stroke_rate_limit * self._env.step_dt
        stroke_delta = torch.clamp(stroke_target - self._stroke_actual, min=-max_delta, max=max_delta)
        self._stroke_actual[:] = torch.clamp(
            self._stroke_actual + stroke_delta, min=self._stroke_min, max=self._stroke_max
        )
        self._position_target[:] = self._interp_stroke_to_joint_pos(self._stroke_actual) * self._joint_target_sign

        effort = self._impedance_kp * (self._position_target - joint_pos) - self._impedance_kd * joint_vel
        self._effort_actual[:] = torch.clamp(effort, min=-self._max_effort, max=self._max_effort)

        # The Ranger leg joints are driven by IsaacLab implicit actuators, which
        # follow the articulation joint position targets. Keep the internally
        # computed effort as debug state only and drive physics through the
        # articulated position target path.
        self._processed_actions.zero_()

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._position_target, joint_ids=self._joint_ids)
        self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)

    def _canonicalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).flatten()
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()

    def _validate_per_env_buffer(self, name: str, buffer: torch.Tensor, env_ids: torch.Tensor) -> None:
        if buffer.ndim != 2 or buffer.shape[0] != self.num_envs or buffer.shape[1] != self.action_dim:
            env_min = int(env_ids.min().item()) if env_ids.numel() > 0 else -1
            env_max = int(env_ids.max().item()) if env_ids.numel() > 0 else -1
            raise RuntimeError(
                f"{self.__class__.__name__}.{name} has invalid shape {tuple(buffer.shape)}; "
                f"expected ({self.num_envs}, {self.action_dim}), env_ids_min={env_min}, env_ids_max={env_max}"
            )
        if env_ids.numel() > 0 and (int(env_ids.min().item()) < 0 or int(env_ids.max().item()) >= buffer.shape[0]):
            raise RuntimeError(
                f"{self.__class__.__name__}.{name} env_ids out of bounds for shape {tuple(buffer.shape)}; "
                f"num_envs={self.num_envs}, env_ids_min={int(env_ids.min().item())}, "
                f"env_ids_max={int(env_ids.max().item())}"
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = self._canonicalize_env_ids(env_ids)
        self._validate_per_env_buffer("_raw_actions", self._raw_actions, env_ids)
        self._validate_per_env_buffer("_clipped_actions", self._clipped_actions, env_ids)
        self._validate_per_env_buffer("_processed_actions", self._processed_actions, env_ids)
        self._validate_per_env_buffer("_stroke_desired", self._stroke_desired, env_ids)
        self._validate_per_env_buffer("_stroke_actual", self._stroke_actual, env_ids)
        self._validate_per_env_buffer("_position_target", self._position_target, env_ids)
        self._validate_per_env_buffer("_effort_actual", self._effort_actual, env_ids)
        self._raw_actions[env_ids, :] = 0.0
        self._clipped_actions[env_ids, :] = 0.0
        self._processed_actions[env_ids, :] = 0.0
        self._effort_actual[env_ids, :] = 0.0
        self._stroke_desired[env_ids, :] = self._stroke_mid
        self._stroke_actual[env_ids, :] = self._stroke_mid
        self._position_target[env_ids, :] = (
            self._interp_stroke_to_joint_pos(self._stroke_actual[env_ids, :]) * self._joint_target_sign
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
    # Deprecated / ignored: kept only so older config files still load cleanly.
    velocity_kp: float = 10.0
    velocity_damping: float = 0.2
    viscous_friction: float = 0.05
    effort_limit: float = 100.0
    control_mode: str = "velocity"


@configclass
class HydraulicActuatorActionCfg(ActionTermCfg):
    """Configuration for :class:`HydraulicActuatorAction`."""

    class_type: type[ActionTerm] = HydraulicActuatorAction

    joint_names: list[str] = MISSING
    preserve_order: bool = False

    stroke_min: float = 0.0
    stroke_max: float = 1.0
    stroke_rate_limit: float = 1.0
    time_constant: float = 0.08

    stroke_table: tuple[float, ...] = (0.0, 0.5, 1.0)
    joint_pos_table: tuple[float, ...] = (-1.0, 0.0, 1.0)
    joint_target_sign: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)

    max_effort: float = 300.0
    impedance_kp: float = 250.0
    impedance_kd: float = 30.0


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
