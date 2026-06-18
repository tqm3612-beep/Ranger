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
from isaaclab.sensors import RayCaster


def local_height_scan(
    env: ManagerBasedEnv,
    sensor_names: tuple[str, ...],
    asset_name: str = "robot",
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
    invalid_height: float = 0.0,
) -> torch.Tensor:
    """Build a local height scan from FOV-limited LiDAR ray hits."""

    height_map, _ = _build_local_height_map(
        env=env,
        sensor_names=sensor_names,
        asset_name=asset_name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        invalid_height=invalid_height,
    )
    return height_map.reshape(env.num_envs, -1)


def local_height_scan_valid_mask(
    env: ManagerBasedEnv,
    sensor_names: tuple[str, ...],
    asset_name: str = "robot",
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
) -> torch.Tensor:
    """Return valid cells for the local height scan."""

    _, valid_mask = _build_local_height_map(
        env=env,
        sensor_names=sensor_names,
        asset_name=asset_name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
    )
    return valid_mask.to(torch.float32).reshape(env.num_envs, -1)


def local_geometric_map(
    env: ManagerBasedEnv,
    sensor_names: tuple[str, ...],
    asset_name: str = "robot",
    x_range: tuple[float, float] = (0.0, 2.0),
    y_range: tuple[float, float] = (-0.6, 0.6),
    resolution: float = 0.1,
    step_threshold: float = 0.08,
) -> torch.Tensor:
    """Build a local five-layer geometric map from LiDAR ray hits.

    The returned layers are height, slope, roughness, step, and valid mask.
    Points outside the local map bounds are discarded, so the policy only sees
    geometry observed through the configured sensor rays.
    """

    height_map, valid_mask = _build_local_height_map(
        env=env,
        sensor_names=sensor_names,
        asset_name=asset_name,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
    )

    slope_map = _compute_slope_map(height_map, valid_mask, resolution)
    roughness_map = _compute_roughness_map(height_map, valid_mask)
    step_map = _compute_step_map(height_map, valid_mask, step_threshold)

    layers = torch.stack(
        (
            height_map,
            slope_map,
            roughness_map,
            step_map,
            valid_mask.to(height_map.dtype),
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
        ray_hits_w.append(sensor.data.ray_hits_w)
    points_w = torch.cat(ray_hits_w, dim=1)

    points_rel_w = points_w - asset.data.root_pos_w.unsqueeze(1)
    num_rays = points_rel_w.shape[1]
    points_b = math_utils.quat_apply_inverse(
        asset.data.root_quat_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4),
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
