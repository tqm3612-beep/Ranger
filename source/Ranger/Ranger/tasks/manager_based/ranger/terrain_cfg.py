# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain generator configs for Ranger perception adaptation tasks."""

from __future__ import annotations

import math

import numpy as np
import trimesh

from isaaclab.terrains import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfTerrainBaseCfg,
    MeshPlaneTerrainCfg,
    TerrainGeneratorCfg,
)
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.utils import configclass


RANGER_WAVE_AMPLITUDE_M = 0.18
RANGER_WAVE_WAVELENGTH_M = 6.0
RANGER_WAVE_START_X_M = 3.0
RANGER_WAVE_GOAL_X_M = 30.0
RANGER_WAVE_GOAL_MARKER_HEIGHT_M = 0.2


def ranger_wave_height(x: float | np.ndarray) -> float | np.ndarray:
    """Continuous wave surface height in world x coordinates."""

    return RANGER_WAVE_AMPLITUDE_M * np.sin(2.0 * math.pi * x / RANGER_WAVE_WAVELENGTH_M)


def wave_height_field(difficulty: float, cfg: "HfWaveSurfaceTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate h(x)=A*sin(2*pi*x/lambda) as a continuous height field."""

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
    x = np.linspace(0.0, cfg.size[0], width_pixels, dtype=np.float32)
    z = ranger_wave_height(x).reshape(width_pixels, 1)
    z = np.repeat(z, length_pixels, axis=1)
    heights = np.rint(z / cfg.vertical_scale).astype(np.int16)
    vertices, triangles = convert_height_field_to_mesh(
        heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    origin = np.array(
        [
            RANGER_WAVE_START_X_M,
            0.5 * cfg.size[1],
            float(ranger_wave_height(RANGER_WAVE_START_X_M)),
        ],
        dtype=np.float32,
    )
    return [mesh], origin


@configclass
class HfWaveSurfaceTerrainCfg(HfTerrainBaseCfg):
    """Deterministic continuous wave terrain for first-stage terrain adaptation."""

    function = wave_height_field


RANGER_WAVE_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=2,
    curriculum=False,
    size=(36.0, 20.0),
    border_width=0.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={
        "wave": HfWaveSurfaceTerrainCfg(proportion=1.0),
    },
)


RANGER_STAGE2_TERRAIN_P0_CFG = TerrainGeneratorCfg(
    seed=2,
    curriculum=True,
    size=(32.0, 32.0),
    border_width=5.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.50),
        "gentle_up_slope": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(math.tan(math.radians(2.0)), math.tan(math.radians(6.0))),
            platform_width=3.0,
        ),
        "gentle_down_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(math.tan(math.radians(2.0)), math.tan(math.radians(6.0))),
            platform_width=3.0,
        ),
    },
)
"""P0 terrain mix: flat plus deterministic columns of gentle ascent and descent."""


RANGER_STAGE2_TERRAIN_P1_CFG = TerrainGeneratorCfg(
    seed=3,
    curriculum=True,
    size=(32.0, 32.0),
    border_width=5.0,
    num_rows=4,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.20),
        "moderate_up_slope": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(math.tan(math.radians(5.0)), math.tan(math.radians(10.0))),
            platform_width=3.0,
        ),
        "moderate_down_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(math.tan(math.radians(5.0)), math.tan(math.radians(10.0))),
            platform_width=3.0,
        ),
        "ascending_stairs": HfInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.03, 0.08),
            step_width=0.8,
            platform_width=3.0,
        ),
        "descending_stairs": HfPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.03, 0.08),
            step_width=0.8,
            platform_width=3.0,
        ),
    },
)
"""P1 terrain mix: flat, moderate bidirectional slopes, and bidirectional stairs."""
