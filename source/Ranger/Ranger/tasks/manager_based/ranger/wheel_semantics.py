"""Canonical Ranger wheel-coordinate conventions.

The RL/control stack uses semantic wheel commands ordered as ``[lb, lf, rf, rb]``:
positive means that wheel should drive the vehicle forward. Isaac Sim joint axes
are mirrored left-vs-right, so only the simulator boundary converts between
semantic and physical joint coordinates.
"""

from __future__ import annotations

import torch


RANGER_WHEEL_ORDER_LB_LF_RF_RB = ("w_lb", "w_lf", "w_rf", "w_rb")
RANGER_WHEEL_JOINT_FORWARD_SIGN = (-1.0, -1.0, 1.0, 1.0)


def _validate_four_wheels(wheel_tensor: torch.Tensor) -> None:
    if wheel_tensor.shape[-1] != 4:
        raise ValueError(f"Expected four Ranger wheel values, got shape={tuple(wheel_tensor.shape)}")


def ranger_wheel_joint_forward_sign_like(wheel_tensor: torch.Tensor) -> torch.Tensor:
    """Return the physical-joint sign corresponding to semantic forward motion."""

    _validate_four_wheels(wheel_tensor)
    return wheel_tensor.new_tensor(RANGER_WHEEL_JOINT_FORWARD_SIGN)


def ranger_wheel_semantic_to_joint(wheel_semantic: torch.Tensor) -> torch.Tensor:
    """Convert semantic wheel values to physical Isaac-Sim joint coordinates."""

    return wheel_semantic * ranger_wheel_joint_forward_sign_like(wheel_semantic)


def ranger_wheel_joint_to_semantic(wheel_joint: torch.Tensor) -> torch.Tensor:
    """Convert physical Isaac-Sim wheel joint values to forward-positive semantics."""

    return wheel_joint * ranger_wheel_joint_forward_sign_like(wheel_joint)


def ranger_wheel_semantic_modes(wheel_semantic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return semantic common and turn modes for ``[lb, lf, rf, rb]`` wheel values."""

    _validate_four_wheels(wheel_semantic)
    left = wheel_semantic[..., :2].mean(dim=-1)
    right = wheel_semantic[..., 2:].mean(dim=-1)
    common = 0.5 * (left + right)
    turn = 0.5 * (right - left)
    return common, turn


def ranger_wheel_semantic_from_modes(common: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
    """Construct ``[lb, lf, rf, rb]`` semantic wheel values from common/turn modes."""

    left = common - turn
    right = common + turn
