# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize Ranger stage-1 geometric visibility and local geometric maps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize Ranger local visibility and geometric map layers.")
parser.add_argument("--task", type=str, default="Template-Ranger-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--steps", type=int, default=2, help="Number of zero-action simulation steps before capture.")
parser.add_argument(
    "--terrain_case",
    type=str,
    default="flat",
    choices=("flat", "step", "ramp"),
    help="Local terrain case used for perception debugging.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs/perception_debug",
    help="Directory for saved visualization artifacts.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab_tasks  # noqa: F401
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_tasks.utils import parse_env_cfg

import Ranger.tasks  # noqa: F401
from Ranger.assets.ranger.ranger_cfg import RANGER_URDF_PATH, RANGER_USD_PATH
from Ranger.tasks.manager_based.ranger import mdp


def _masked_layer(layer: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Return a visualization-only layer where invalid cells are hidden."""

    masked = layer.astype(np.float32, copy=True)
    masked[valid_mask <= 0.0] = np.nan
    return masked


def _configure_sensor_mesh_targets(env_cfg) -> None:
    """Disable sensor debug visualization to keep the debug script stable."""

    for sensor_cfg_name in ("mid360_lidar", "avia_lidar", "d435i_camera"):
        getattr(env_cfg.scene, sensor_cfg_name).debug_vis = False


def _terrain_cfg_for_case(terrain_case: str) -> TerrainImporterCfg:
    """Build a single-mesh terrain config that RayCaster can consume reliably."""

    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="average",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35))

    if terrain_case == "flat":
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=physics_material,
            visual_material=visual_material,
            debug_vis=False,
        )

    if terrain_case == "ramp":
        terrain_generator = terrain_gen.TerrainGeneratorCfg(
            seed=7,
            curriculum=False,
            size=(4.0, 4.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.05,
            vertical_scale=0.005,
            slope_threshold=None,
            use_cache=False,
            sub_terrains={
                "ramp": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=1.0,
                    slope_range=(0.20, 0.20),
                    platform_width=0.4,
                    border_width=0.0,
                ),
            },
        )
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_generator,
            collision_group=-1,
            physics_material=physics_material,
            visual_material=visual_material,
            debug_vis=False,
        )

    if terrain_case == "step":
        terrain_generator = terrain_gen.TerrainGeneratorCfg(
            seed=11,
            curriculum=False,
            size=(4.0, 4.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.05,
            vertical_scale=0.005,
            slope_threshold=None,
            use_cache=False,
            sub_terrains={
                "step": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
                    proportion=1.0,
                    step_height_range=(0.10, 0.10),
                    step_width=0.25,
                    platform_width=0.4,
                    border_width=0.0,
                ),
            },
        )
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_generator,
            collision_group=-1,
            physics_material=physics_material,
            visual_material=visual_material,
            debug_vis=False,
        )

    raise ValueError(f"Unsupported terrain_case: {terrain_case}")


def _configure_terrain(env_cfg, terrain_case: str) -> None:
    """Replace the default flat cuboid with a single terrain prim under /World/ground."""

    env_cfg.scene.ground = None
    env_cfg.scene.terrain = _terrain_cfg_for_case(terrain_case)


def _save_layer_figure(
    output_path: Path,
    sensor_visibility: dict[str, np.ndarray],
    fused_layers: dict[str, np.ndarray],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    terrain_case: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    extent = (x_range[0], x_range[1], y_range[0], y_range[1])
    x_label = "x in gravity-aligned base frame [m]"
    y_label = "y in gravity-aligned base frame [m]"

    visibility_items = list(sensor_visibility.items())
    for axis, (sensor_name, mask) in zip(axes[0, :3], visibility_items, strict=True):
        axis.imshow(mask.T, origin="lower", extent=extent, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_title(f"{sensor_name} visibility")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)

    fused_valid = fused_layers["valid_mask"]
    axes[0, 3].imshow(fused_valid.T, origin="lower", extent=extent, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0, 3].set_title("fused valid_mask")
    axes[0, 3].set_xlabel(x_label)
    axes[0, 3].set_ylabel(y_label)

    bottom_titles = ("height", "slope", "roughness", "step")
    fused_valid_mask = fused_layers["valid_mask"]
    for axis, layer_name in zip(axes[1], bottom_titles, strict=True):
        masked_layer = _masked_layer(fused_layers[layer_name], fused_valid_mask)
        image = axis.imshow(
            masked_layer.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
        )
        axis.set_title(layer_name)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        fig.colorbar(image, ax=axis, shrink=0.84)

    fig.suptitle(f"Ranger stage-1 local perception ({terrain_case})")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Create the environment, capture local visibility, and save visualizations."""
    if not RANGER_USD_PATH.exists():
        raise FileNotFoundError(
            f"Ranger USD not found: {RANGER_USD_PATH}\n"
            f"Place the URDF at {RANGER_URDF_PATH} and convert it to this USD path first."
        )

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.policy.local_geometric_map.params["apply_noise"] = False
    _configure_sensor_mesh_targets(env_cfg)
    _configure_terrain(env_cfg, args_cli.terrain_case)
    print(f"[DEBUG]: building env for terrain_case={args_cli.terrain_case}", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("[DEBUG]: env created", flush=True)
        obs_dict, _ = env.reset()
        print("[DEBUG]: env reset complete", flush=True)

        policy_cfg = env.unwrapped.cfg.observations.policy
        map_params = dict(policy_cfg.local_geometric_map.params)
        sensor_names = tuple(map_params["sensor_names"])
        x_range = tuple(map_params["x_range"])
        y_range = tuple(map_params["y_range"])
        resolution = float(map_params["resolution"])
        step_threshold = float(map_params["step_threshold"])

        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)
            print("[DEBUG]: rollout steps complete", flush=True)

            visibility_maps = mdp.local_sensor_visibility_maps(
                env=env.unwrapped,
                sensor_names=sensor_names,
                x_range=x_range,
                y_range=y_range,
                resolution=resolution,
            )
            fused_layers = mdp.local_geometric_map_layers(
                env=env.unwrapped,
                sensor_names=sensor_names,
                x_range=x_range,
                y_range=y_range,
                resolution=resolution,
                step_threshold=step_threshold,
            )

        env_index = 0
        visibility_np = {name: tensor[env_index].detach().cpu().numpy() for name, tensor in visibility_maps.items()}
        layers_np = {name: tensor[env_index].detach().cpu().numpy() for name, tensor in fused_layers.items()}

        grid_shape = layers_np["height"].shape
        policy_state_obs = obs_dict["policy_state"]
        policy_map_obs = obs_dict["policy_map"]
        observation_dim = int(policy_state_obs.shape[-1] + policy_map_obs.shape[-1])
        fused_valid_cells = int(layers_np["valid_mask"].sum())

        print(f"[INFO]: terrain case = {args_cli.terrain_case}")
        print(
            "[INFO]: actor observation dim = "
            f"{observation_dim} (state={int(policy_state_obs.shape[-1])}, map={int(policy_map_obs.shape[-1])})"
        )
        print(f"[INFO]: local map grid shape = {grid_shape}")
        print(f"[INFO]: fused valid cells = {fused_valid_cells}")
        for sensor_name, mask in visibility_np.items():
            print(f"[INFO]: {sensor_name} visible cells = {int(mask.sum())}")

        npz_path = output_dir / f"ranger_local_perception_{args_cli.terrain_case}_env0.npz"
        png_path = output_dir / f"ranger_local_perception_{args_cli.terrain_case}_env0.png"
        np.savez_compressed(npz_path, **visibility_np, **layers_np)
        _save_layer_figure(png_path, visibility_np, layers_np, x_range, y_range, args_cli.terrain_case)

        print(f"[INFO]: saved arrays to {npz_path}")
        print(f"[INFO]: saved figure to {png_path}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
