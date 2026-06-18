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


class HydraulicCylinderAction(ActionTerm):
    """Equivalent hydraulic-cylinder action for Ranger wheel-leg joints.

    The policy commands a virtual cylinder stroke. This term applies stroke
    limits, first-order lag, stroke-rate limits, and a stroke-to-joint-angle
    lookup before sending position targets to the wheel-leg joints.
    """

    cfg: "HydraulicCylinderActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "HydraulicCylinderActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints == 0:
            raise ValueError(f"No joints matched {self.cfg.joint_names} for HydraulicCylinderAction.")

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._stroke_table = torch.tensor(self.cfg.stroke_table, dtype=torch.float32, device=self.device)
        self._joint_pos_table = torch.tensor(self.cfg.joint_pos_table, dtype=torch.float32, device=self.device)
        if self._stroke_table.ndim != 1 or self._joint_pos_table.ndim != 1:
            raise ValueError("stroke_table and joint_pos_table must be one-dimensional.")
        if self._stroke_table.numel() != self._joint_pos_table.numel():
            raise ValueError("stroke_table and joint_pos_table must have the same length.")
        if self._stroke_table.numel() < 2:
            raise ValueError("stroke_table and joint_pos_table must contain at least two points.")
        if not torch.all(self._stroke_table[1:] > self._stroke_table[:-1]):
            raise ValueError("stroke_table must be strictly increasing.")

        self._stroke_min = float(self.cfg.stroke_min)
        self._stroke_max = float(self.cfg.stroke_max)
        if self._stroke_max <= self._stroke_min:
            raise ValueError("stroke_max must be greater than stroke_min.")
        if self.cfg.stroke_rate_limit <= 0.0:
            raise ValueError("stroke_rate_limit must be positive.")
        if self.cfg.time_constant < 0.0:
            raise ValueError("time_constant must be non-negative.")

        self._stroke_mid = 0.5 * (self._stroke_min + self._stroke_max)
        self._stroke_half_range = 0.5 * (self._stroke_max - self._stroke_min)
        self._stroke_actual = torch.full_like(self._raw_actions, self._stroke_mid)
        self._processed_actions[:] = self._interp_stroke_to_joint_pos(self._stroke_actual)

        if isinstance(self.cfg.clip, dict):
            self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self.action_dim, 1
            )
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
            self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
        elif self.cfg.clip is not None:
            raise ValueError(f"Unsupported clip type: {type(self.cfg.clip)}. Supported type is dict.")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Joint position targets computed from the virtual cylinder stroke."""
        return self._processed_actions

    @property
    def stroke_actual(self) -> torch.Tensor:
        """Current virtual cylinder stroke after actuator dynamics."""
        return self._stroke_actual

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        if self.cfg.clip is not None:
            self._raw_actions[:] = torch.clamp(self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

        # 把RL输出映射成目标行程s_des
        stroke_des = self._stroke_mid + self._raw_actions * self._stroke_half_range
        stroke_des = torch.clamp(stroke_des, min=self._stroke_min, max=self._stroke_max)

        # 模拟一阶系统响应和行程速率限制，更新当前行程s_actual
        dt = self._env.step_dt
        if self.cfg.time_constant > 0.0:
            alpha = dt / (self.cfg.time_constant + dt)
            stroke_target = self._stroke_actual + alpha * (stroke_des - self._stroke_actual)
        else:
            stroke_target = stroke_des

        max_delta = self.cfg.stroke_rate_limit * dt
        stroke_delta = torch.clamp(stroke_target - self._stroke_actual, min=-max_delta, max=max_delta)
        self._stroke_actual[:] = torch.clamp(
            self._stroke_actual + stroke_delta, min=self._stroke_min, max=self._stroke_max
        )
        
        # 根据当前行程s_actual通过查找表计算轮-腿关节位置，并保存到_processed_actions中
        self._processed_actions[:] = self._interp_stroke_to_joint_pos(self._stroke_actual)

    # 把角度目标发给isaac lab的g_*关节
    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._stroke_actual[env_ids] = self._stroke_mid
        self._processed_actions[env_ids] = self._interp_stroke_to_joint_pos(self._stroke_actual[env_ids])

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
class HydraulicCylinderActionCfg(ActionTermCfg):
    """Configuration for :class:`HydraulicCylinderAction`."""

    class_type: type[ActionTerm] = HydraulicCylinderAction

    joint_names: list[str] = MISSING
    """Joint names or regex expressions controlled by the equivalent hydraulic cylinders."""

    preserve_order: bool = False
    """Whether to preserve the provided joint order in the action vector."""

    stroke_min: float = 0.0
    """Minimum virtual cylinder stroke."""

    stroke_max: float = 1.0
    """Maximum virtual cylinder stroke."""

    stroke_rate_limit: float = 1.0
    """Maximum stroke speed in stroke units per second."""

    time_constant: float = 0.08
    """First-order response time constant in seconds."""

    stroke_table: tuple[float, ...] = (0.0, 0.5, 1.0)
    """Lookup table input: virtual cylinder stroke."""

    joint_pos_table: tuple[float, ...] = (-1.0, 0.0, 1.0)
    """Lookup table output: wheel-leg joint target position in radians."""
