#!/usr/bin/env python3
"""Audit Ranger runtime asset symmetry without modifying project assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Audit Ranger asset geometry, collision, inertia, and mapping symmetry.")
parser.add_argument("--task", type=str, default="Template-Ranger-ShortGoalFlat-v1", help="Gym task id.")
parser.add_argument(
    "--mode",
    type=str,
    choices=("all", "static", "runtime", "coplanar", "collision", "yaw", "stiffness"),
    default="all",
    help="Audit subset to run.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to create for runtime audits.")
parser.add_argument("--seed", type=int, default=2, help="Environment seed.")
parser.add_argument("--steps", type=int, default=240, help="Steps per runtime condition.")
parser.add_argument("--warmup_steps", type=int, default=120, help="Steps excluded from contact summaries.")
parser.add_argument("--strokes", type=str, default="0.25,0.50,0.75", help="Comma-separated stroke values.")
parser.add_argument("--stiffness_values", type=str, default="1e6,3e5,1e5,3e4", help="Comma-separated leg stiffness values.")
parser.add_argument("--base_height", type=float, default=0.75, help="Pinned base height for runtime tests.")
parser.add_argument("--output_json", type=str, default="/tmp/ranger_asset_symmetry_audit.json", help="JSON report path.")
parser.add_argument("--output_csv", type=str, default="/tmp/ranger_asset_symmetry_audit.csv", help="Runtime row CSV path.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Ranger.tasks  # noqa: F401
from Ranger.assets.ranger.ranger_cfg import RANGER_CFG, RANGER_URDF_PATH, RANGER_USD_PATH


SEMANTIC_CORNERS = ("lb", "lf", "rf", "rb")
LEG_JOINTS = ("g_lb", "g_lf", "g_rf", "g_rb")
WHEEL_JOINTS = ("w_lb", "w_lf", "w_rf", "w_rb")
WHEEL_BODIES = ("w_lb", "w_lf", "w_rf", "w_rb")
LEG_BODIES = ("lb_Link", "lf_Link", "rf_Link", "rb_Link")


def _parse_float_list(text: str, *, name: str) -> list[float]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return [float(value) for value in values]


def _tensor_to_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_tensor_to_list(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _tensor_to_list(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_tensor_to_list(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": mean(finite), "std": pstdev(finite) if len(finite) > 1 else 0.0, "min": min(finite), "max": max(finite)}


def _extract_attr(block: str, tag: str, attr: str) -> str | None:
    match = re.search(rf"<{tag}\b[^>]*\b{attr}=\"([^\"]+)\"", block)
    return match.group(1) if match else None


def _parse_xyz(text: str | None) -> list[float] | None:
    if text is None:
        return None
    return [float(part) for part in text.split()]


def _parse_urdf_loose(urdf_path: Path) -> dict[str, Any]:
    """Parse the Ranger URDF with a tolerant regex path.

    The current URDF contains duplicated XML fragments, so strict XML parsers
    may fail. This parser only extracts the fields needed for the audit.
    """

    text = urdf_path.read_text(encoding="utf-8", errors="replace")
    links: dict[str, dict[str, Any]] = {}
    joints: dict[str, dict[str, Any]] = {}

    for match in re.finditer(r"<link\b[^>]*name=\"([^\"]+)\"[^>]*>(.*?)</link>", text, flags=re.S):
        name = match.group(1)
        block = match.group(2)
        inertial = re.search(r"<inertial>(.*?)</inertial>", block, flags=re.S)
        collisions = []
        for collision in re.finditer(r"<collision>(.*?)</collision>", block, flags=re.S):
            collision_block = collision.group(1)
            collisions.append(
                {
                    "origin_xyz": _parse_xyz(_extract_attr(collision_block, "origin", "xyz")),
                    "origin_rpy": _parse_xyz(_extract_attr(collision_block, "origin", "rpy")),
                    "mesh": _extract_attr(collision_block, "mesh", "filename"),
                }
            )
        links[name] = {
            "mass": float(_extract_attr(inertial.group(1), "mass", "value")) if inertial and _extract_attr(inertial.group(1), "mass", "value") else None,
            "inertial_origin_xyz": _parse_xyz(_extract_attr(inertial.group(1), "origin", "xyz")) if inertial else None,
            "inertial_origin_rpy": _parse_xyz(_extract_attr(inertial.group(1), "origin", "rpy")) if inertial else None,
            "inertia": {
                key: (float(_extract_attr(inertial.group(1), "inertia", key)) if inertial and _extract_attr(inertial.group(1), "inertia", key) else None)
                for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
            },
            "collisions": collisions,
        }

    for match in re.finditer(r"<joint\b[^>]*name=\"([^\"]+)\"[^>]*type=\"([^\"]+)\"[^>]*>(.*?)</joint>", text, flags=re.S):
        name = match.group(1)
        block = match.group(3)
        joints[name] = {
            "type": match.group(2),
            "parent": _extract_attr(block, "parent", "link"),
            "child": _extract_attr(block, "child", "link"),
            "origin_xyz": _parse_xyz(_extract_attr(block, "origin", "xyz")),
            "origin_rpy": _parse_xyz(_extract_attr(block, "origin", "rpy")),
            "axis": _parse_xyz(_extract_attr(block, "axis", "xyz")),
            "limit_lower": float(_extract_attr(block, "limit", "lower")) if _extract_attr(block, "limit", "lower") else None,
            "limit_upper": float(_extract_attr(block, "limit", "upper")) if _extract_attr(block, "limit", "upper") else None,
            "limit_effort": float(_extract_attr(block, "limit", "effort")) if _extract_attr(block, "limit", "effort") else None,
            "limit_velocity": float(_extract_attr(block, "limit", "velocity")) if _extract_attr(block, "limit", "velocity") else None,
            "damping": float(_extract_attr(block, "dynamics", "damping")) if _extract_attr(block, "dynamics", "damping") else None,
            "friction": float(_extract_attr(block, "dynamics", "friction")) if _extract_attr(block, "dynamics", "friction") else None,
        }
    return {"links": links, "joints": joints}


def _static_asset_report() -> dict[str, Any]:
    urdf = _parse_urdf_loose(RANGER_URDF_PATH)
    usd_config = RANGER_USD_PATH.parent / "config.yaml"
    spawn_cfg = RANGER_CFG.spawn
    return {
        "runtime_spawn_type": type(spawn_cfg).__name__,
        "runtime_usd_path": str(RANGER_USD_PATH),
        "source_urdf_path": str(RANGER_URDF_PATH),
        "usd_exists": RANGER_USD_PATH.exists(),
        "urdf_exists": RANGER_URDF_PATH.exists(),
        "usd_config_path": str(usd_config),
        "usd_config": usd_config.read_text(encoding="utf-8") if usd_config.exists() else "",
        "python_overrides": {
            "spawn": {
                "usd_path": getattr(spawn_cfg, "usd_path", None),
                "activate_contact_sensors": getattr(spawn_cfg, "activate_contact_sensors", None),
                "rigid_props": str(getattr(spawn_cfg, "rigid_props", None)),
                "articulation_props": str(getattr(spawn_cfg, "articulation_props", None)),
            },
            "soft_joint_pos_limit_factor": RANGER_CFG.soft_joint_pos_limit_factor,
            "init_joint_pos": dict(RANGER_CFG.init_state.joint_pos),
            "init_pos": tuple(RANGER_CFG.init_state.pos),
            "actuators": {
                name: {
                    "type": type(cfg).__name__,
                    "joint_names_expr": list(cfg.joint_names_expr),
                    "effort_limit": getattr(cfg, "effort_limit", None),
                    "effort_limit_sim": getattr(cfg, "effort_limit_sim", None),
                    "velocity_limit": getattr(cfg, "velocity_limit", None),
                    "velocity_limit_sim": getattr(cfg, "velocity_limit_sim", None),
                    "stiffness": getattr(cfg, "stiffness", None),
                    "damping": getattr(cfg, "damping", None),
                }
                for name, cfg in RANGER_CFG.actuators.items()
            },
        },
        "urdf_semantic_joints": {name: urdf["joints"].get(name) for name in (*LEG_JOINTS, *WHEEL_JOINTS)},
        "urdf_semantic_links": {name: urdf["links"].get(name) for name in (*LEG_BODIES, *WHEEL_BODIES, "base_link")},
        "urdf_note": "Runtime uses the generated USD path above; editing the URDF alone will not affect runtime until USD is regenerated/repointed.",
    }


def _get_action_slice(unwrapped_env, term_name: str) -> slice:
    start = 0
    for active_name, dim in zip(unwrapped_env.action_manager.active_terms, unwrapped_env.action_manager.action_term_dim):
        if active_name == term_name:
            return slice(start, start + int(dim))
        start += int(dim)
    raise KeyError(f"Action term {term_name!r} not found. Active terms: {list(unwrapped_env.action_manager.active_terms)}")


def _make_env(*, task: str, num_envs: int, seed: int, device: str | None, disable_fabric: bool, gravity: tuple[float, float, float] | None = None, leg_stiffness: float | None = None):
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs, use_fabric=not disable_fabric)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    if device is not None:
        env_cfg.sim.device = device
    if gravity is not None:
        env_cfg.sim.gravity = gravity
    if leg_stiffness is not None:
        env_cfg.scene.robot.actuators["leg_joints"].stiffness = float(leg_stiffness)
    env = gym.make(task, cfg=env_cfg)
    env.reset()
    return env


def _find_ids(robot, names: tuple[str, ...], *, kind: str) -> tuple[torch.Tensor, list[str]]:
    if kind == "joint":
        ids, resolved = robot.find_joints(list(names), preserve_order=True)
    elif kind == "body":
        ids, resolved = robot.find_bodies(list(names), preserve_order=True)
    else:
        raise ValueError(kind)
    return torch.as_tensor(ids, device=robot.device, dtype=torch.long), list(resolved)


def _joint_limits(robot: Any, ids: torch.Tensor) -> list[list[float]] | None:
    for attr in ("soft_joint_pos_limits", "joint_pos_limits", "default_joint_pos_limits"):
        value = getattr(robot.data, attr, None)
        if value is not None:
            try:
                return value[0, ids].detach().cpu().tolist()
            except Exception:
                try:
                    return value[:, ids].detach().cpu().tolist()
                except Exception:
                    pass
    return None


def _stroke_mapping_table(leg_term: Any) -> list[dict[str, float | str]]:
    rows = []
    actions = (-1.0, -0.5, 0.0, 0.5, 1.0)
    stroke_mid = float(getattr(leg_term, "_stroke_mid"))
    stroke_half = float(getattr(leg_term, "_stroke_half_range"))
    stroke_min = float(getattr(leg_term, "_stroke_min"))
    stroke_max = float(getattr(leg_term, "_stroke_max"))
    stroke_table = getattr(leg_term, "_stroke_table").detach().cpu()
    joint_pos_table = getattr(leg_term, "_joint_pos_table").detach().cpu()
    signs = getattr(leg_term, "_joint_target_sign").detach().cpu().reshape(-1).tolist()
    for corner, joint_name, sign in zip(SEMANTIC_CORNERS, LEG_JOINTS, signs):
        for action in actions:
            stroke_desired = max(stroke_min, min(stroke_max, stroke_mid + action * stroke_half))
            # Tables are currently linear, but keep generic interpolation for audit clarity.
            upper = int(torch.bucketize(torch.tensor([stroke_desired]), stroke_table).item())
            upper = max(1, min(upper, int(stroke_table.numel()) - 1))
            lower = upper - 1
            ratio = (stroke_desired - float(stroke_table[lower])) / (float(stroke_table[upper]) - float(stroke_table[lower]))
            joint_pos = float(joint_pos_table[lower]) + ratio * (float(joint_pos_table[upper]) - float(joint_pos_table[lower]))
            rows.append(
                {
                    "corner": corner,
                    "joint": joint_name,
                    "action": action,
                    "stroke_desired": stroke_desired,
                    "stroke_actual_if_settled": stroke_desired,
                    "joint_target_sign": float(sign),
                    "position_target_if_settled": joint_pos * float(sign),
                }
            )
    return rows


def _runtime_report(env: Any) -> dict[str, Any]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    leg_term = unwrapped.action_manager.get_term("leg_hydraulic")
    wheel_term = unwrapped.action_manager.get_term("wheel_motor_csv")
    leg_ids, leg_names = _find_ids(robot, LEG_JOINTS, kind="joint")
    wheel_joint_ids, wheel_joint_names = _find_ids(robot, WHEEL_JOINTS, kind="joint")
    wheel_body_ids, wheel_body_names = _find_ids(robot, WHEEL_BODIES, kind="body")
    leg_body_ids, leg_body_names = _find_ids(robot, LEG_BODIES, kind="body")
    contact_body_ids, contact_body_names = unwrapped.scene.sensors["wheel_contact_forces"].find_bodies(
        list(WHEEL_BODIES), preserve_order=True
    )
    root_view = getattr(robot, "root_physx_view", None)
    masses = robot.data.default_mass[0].detach().cpu().tolist() if getattr(robot.data, "default_mass", None) is not None else None
    inertias = robot.data.default_inertia[0].detach().cpu().tolist() if getattr(robot.data, "default_inertia", None) is not None else None
    body_com_pos_b = robot.data.body_com_pos_b[0].detach().cpu().tolist()
    total_mass = sum(masses) if masses is not None else None
    total_com_b = None
    if masses is not None and total_mass and total_mass > 0.0:
        weighted = torch.as_tensor(masses, dtype=torch.float64).unsqueeze(1) * robot.data.body_com_pos_b[0].detach().cpu().to(torch.float64)
        total_com_b = (weighted.sum(dim=0) / total_mass).tolist()

    return {
        "action_terms": list(unwrapped.action_manager.active_terms),
        "action_term_dims": [int(dim) for dim in unwrapped.action_manager.action_term_dim],
        "leg_action_slice": [int(_get_action_slice(unwrapped, "leg_hydraulic").start), int(_get_action_slice(unwrapped, "leg_hydraulic").stop)],
        "wheel_action_slice": [int(_get_action_slice(unwrapped, "wheel_motor_csv").start), int(_get_action_slice(unwrapped, "wheel_motor_csv").stop)],
        "joint_names_all": list(robot.joint_names),
        "body_names_all": list(robot.body_names),
        "leg_joint_names_semantic": leg_names,
        "leg_joint_ids_semantic": [int(v) for v in leg_ids.detach().cpu().tolist()],
        "wheel_joint_names_semantic": wheel_joint_names,
        "wheel_joint_ids_semantic": [int(v) for v in wheel_joint_ids.detach().cpu().tolist()],
        "leg_body_names_semantic": leg_body_names,
        "leg_body_ids_semantic": [int(v) for v in leg_body_ids.detach().cpu().tolist()],
        "wheel_body_names_semantic": wheel_body_names,
        "wheel_body_ids_semantic": [int(v) for v in wheel_body_ids.detach().cpu().tolist()],
        "wheel_contact_sensor_body_names_semantic": list(contact_body_names),
        "wheel_contact_sensor_body_ids_semantic": [int(v) for v in contact_body_ids],
        "leg_action_term_joint_names": list(getattr(leg_term, "_joint_names")),
        "leg_action_term_joint_ids": [int(v) for v in getattr(leg_term, "_joint_ids")],
        "wheel_action_term_joint_names": list(getattr(wheel_term, "_joint_names")),
        "wheel_action_term_joint_ids": [int(v) for v in getattr(wheel_term, "_joint_ids")],
        "joint_limits_for_leg_ids": _joint_limits(robot, leg_ids),
        "init_joint_pos_semantic": robot.data.default_joint_pos[0, leg_ids].detach().cpu().tolist(),
        "actuator_leg_stiffness_cfg": getattr(unwrapped.cfg.scene.robot.actuators["leg_joints"], "stiffness", None),
        "actuator_leg_damping_cfg": getattr(unwrapped.cfg.scene.robot.actuators["leg_joints"], "damping", None),
        "stroke_mapping_table": _stroke_mapping_table(leg_term),
        "link_paths_env0": list(getattr(root_view, "link_paths", [[]])[0]) if root_view is not None else [],
        "body_mass": {name: masses[i] for i, name in enumerate(robot.body_names)} if masses is not None else {},
        "body_com_pos_b": {name: body_com_pos_b[i] for i, name in enumerate(robot.body_names)},
        "body_inertia": {name: inertias[i] for i, name in enumerate(robot.body_names)} if inertias is not None else {},
        "total_mass": total_mass,
        "total_com_b": total_com_b,
    }


def _pin_root(robot: Any, env_id: int, *, yaw: float, height: float) -> None:
    env_ids = torch.tensor([env_id], device=robot.device, dtype=torch.long)
    pose = robot.data.default_root_state[env_id : env_id + 1, :7].clone()
    pose[:, 2] = float(height)
    pose[:, 3] = math.cos(0.5 * yaw)
    pose[:, 4] = 0.0
    pose[:, 5] = 0.0
    pose[:, 6] = math.sin(0.5 * yaw)
    vel = torch.zeros((1, 6), device=robot.device, dtype=pose.dtype)
    robot.write_root_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)


def _zero_actions(env: Any) -> torch.Tensor:
    action_dim = sum(int(dim) for dim in env.unwrapped.action_manager.action_term_dim)
    return torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device, dtype=torch.float32)


def _step_pinned(env: Any, actions: torch.Tensor, *, yaw: float, height: float) -> None:
    robot = env.unwrapped.scene["robot"]
    _pin_root(robot, 0, yaw=yaw, height=height)
    env.step(actions)
    _pin_root(robot, 0, yaw=yaw, height=height)
    env.unwrapped.scene.write_data_to_sim()


def _set_exact_geometry_pose(env: Any, *, stroke: float, height: float) -> dict[str, Any]:
    """Set env 0 to an exact, contact-free articulation pose for geometry inspection."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    leg_term = unwrapped.action_manager.get_term("leg_hydraulic")
    env_ids = torch.tensor([0], device=robot.device, dtype=torch.long)
    leg_term_ids = torch.as_tensor(getattr(leg_term, "_joint_ids"), device=robot.device, dtype=torch.long)
    wheel_ids, _ = _find_ids(robot, WHEEL_JOINTS, kind="joint")

    stroke_tensor = torch.full(
        (1, int(leg_term_ids.numel())), float(stroke), device=robot.device, dtype=robot.data.joint_pos.dtype
    )
    position_target = leg_term._interp_stroke_to_joint_pos(stroke_tensor) * leg_term._joint_target_sign
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    joint_pos[:, leg_term_ids] = position_target
    joint_pos[:, wheel_ids] = 0.0

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(position_target, joint_ids=leg_term_ids, env_ids=env_ids)
    robot.set_joint_velocity_target(
        torch.zeros((1, int(wheel_ids.numel())), device=robot.device, dtype=joint_pos.dtype),
        joint_ids=wheel_ids,
        env_ids=env_ids,
    )
    _pin_root(robot, 0, yaw=0.0, height=height)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.step(render=False)
    unwrapped.scene.update(dt=unwrapped.physics_dt)

    # Re-assert once after synchronization so actuator dynamics do not contaminate the snapshot.
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    _pin_root(robot, 0, yaw=0.0, height=height)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.step(render=False)
    unwrapped.scene.update(dt=unwrapped.physics_dt)

    semantic_leg_ids, semantic_leg_names = _find_ids(robot, LEG_JOINTS, kind="joint")
    cached_leg_pos = robot.data.joint_pos[0, semantic_leg_ids]
    root_view = getattr(robot, "root_physx_view", None)
    if root_view is None or not hasattr(root_view, "get_dof_positions"):
        raise RuntimeError("Runtime articulation does not expose root_physx_view.get_dof_positions().")
    live_leg_pos = root_view.get_dof_positions()[0, semantic_leg_ids]
    return {
        "stroke": float(stroke),
        "base_height_requested": float(height),
        "base_height_actual": float(robot.data.root_pos_w[0, 2].item()),
        "base_quat_wxyz_actual": robot.data.root_link_quat_w[0].detach().cpu().tolist(),
        "leg_joint_names_semantic": semantic_leg_names,
        "leg_joint_pos_semantic": cached_leg_pos.detach().cpu().tolist(),
        "leg_joint_pos_physx_semantic": live_leg_pos.detach().cpu().tolist(),
        "leg_joint_cache_error_semantic": (live_leg_pos - cached_leg_pos).detach().cpu().tolist(),
        "leg_joint_target_term_order": position_target[0].detach().cpu().tolist(),
    }


def _runtime_link_paths(robot: Any, body_names: tuple[str, ...]) -> dict[str, str]:
    root_view = getattr(robot, "root_physx_view", None)
    if root_view is None or not getattr(root_view, "link_paths", None):
        raise RuntimeError("Runtime articulation does not expose root_physx_view.link_paths.")
    link_paths = [str(path) for path in root_view.link_paths[0]]
    by_leaf_name = {path.rstrip("/").split("/")[-1]: path for path in link_paths}
    if all(name in by_leaf_name for name in body_names):
        return {name: by_leaf_name[name] for name in body_names}
    if len(link_paths) != len(robot.body_names):
        raise RuntimeError(
            f"Unable to map link paths by prim leaf name and lengths differ: "
            f"{len(link_paths)} != {len(robot.body_names)}; "
            f"body_names={list(robot.body_names)} link_paths={link_paths}"
        )
    return {name: link_paths[list(robot.body_names).index(name)] for name in body_names}


def _runtime_link_indices(robot: Any, link_paths_by_name: dict[str, str]) -> dict[str, int]:
    """Map runtime link paths to their actual PhysX tensor indices."""
    root_view = getattr(robot, "root_physx_view", None)
    if root_view is None or not getattr(root_view, "link_paths", None):
        raise RuntimeError("Runtime articulation does not expose root_physx_view.link_paths.")
    physx_link_paths = [str(path) for path in root_view.link_paths[0]]
    path_to_index = {path: index for index, path in enumerate(physx_link_paths)}
    missing = [path for path in link_paths_by_name.values() if path not in path_to_index]
    if missing:
        raise RuntimeError(f"Runtime link paths missing from PhysX tensor order: {missing}; all={physx_link_paths}")
    return {name: int(path_to_index[path]) for name, path in link_paths_by_name.items()}


def _get_live_link_pose_w(robot: Any, physx_link_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one link pose directly from the PhysX tensor view.

    PhysX returns quaternion values in xyzw order; the audit uses wxyz.
    """
    root_view = getattr(robot, "root_physx_view", None)
    if root_view is None or not hasattr(root_view, "get_link_transforms"):
        raise RuntimeError("Runtime articulation does not expose root_physx_view.get_link_transforms().")
    transforms = root_view.get_link_transforms()
    pose = transforms[0, int(physx_link_index)]
    position = pose[0:3]
    quat_xyzw = pose[3:7]
    quat_wxyz = torch.cat((quat_xyzw[3:4], quat_xyzw[0:3]), dim=0)
    return position, quat_wxyz


def _get_live_root_position_w(robot: Any) -> torch.Tensor:
    root_view = getattr(robot, "root_physx_view", None)
    if root_view is None:
        raise RuntimeError("Runtime articulation does not expose root_physx_view.")
    if hasattr(root_view, "get_root_transforms"):
        return root_view.get_root_transforms()[0, 0:3]
    if hasattr(root_view, "get_link_transforms"):
        return root_view.get_link_transforms()[0, 0, 0:3]
    raise RuntimeError("Runtime articulation does not expose root or link transforms.")


def _transform_point_to_ancestor_local(prim: Usd.Prim, ancestor: Usd.Prim, point: Gf.Vec3d) -> Gf.Vec3d:
    """Transform a point from ``prim`` local space into ``ancestor`` local space.

    Only authored local xform ops along the parent chain are used. No world-space
    transform or articulation state participates in this conversion.
    """
    current = prim
    transformed = Gf.Vec3d(point)
    ancestor_path = ancestor.GetPath()
    visited: list[str] = []
    while current and current.IsValid() and current.GetPath() != ancestor_path:
        visited.append(str(current.GetPath()))
        xformable = UsdGeom.Xformable(current)
        if xformable:
            local_to_parent, _ = xformable.GetLocalTransformation(Usd.TimeCode.Default())
            transformed = local_to_parent.Transform(transformed)
        current = current.GetParent()
    if not current or not current.IsValid() or current.GetPath() != ancestor_path:
        raise RuntimeError(
            f"Collision prim {prim.GetPath()} is not below expected wheel link {ancestor_path}; visited={visited}"
        )
    return transformed


def _collision_prim_bounds(
    stage: Usd.Stage,
    link_path: str,
    *,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
) -> list[dict[str, Any]]:
    """Return world-space bounds for enabled collision prims below one wheel link.

    Mesh colliders are evaluated from authored mesh points, their USD transform
    relative to the wheel link, and the live PhysX wheel-link pose. This avoids
    mixing stale USD world transforms with current articulation transforms.
    """
    link_prim = stage.GetPrimAtPath(link_path)
    if not link_prim or not link_prim.IsValid():
        raise RuntimeError(f"Invalid runtime link prim path: {link_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    records: list[dict[str, Any]] = []
    scanned_prims: list[dict[str, Any]] = []
    traversal = Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies())
    for prim in traversal:
        scanned_prims.append(
            {
                "prim_path": str(prim.GetPath()),
                "prim_type": prim.GetTypeName(),
                "is_instance": bool(prim.IsInstance()),
                "is_instance_proxy": bool(prim.IsInstanceProxy()),
                "applied_schemas": [str(schema) for schema in prim.GetAppliedSchemas()],
            }
        )
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_api = UsdPhysics.CollisionAPI(prim)
        enabled_value = collision_api.GetCollisionEnabledAttr().Get()
        if enabled_value is not None and not bool(enabled_value):
            continue

        approximation = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()

        points_world: list[torch.Tensor] = []
        points_link_min: tuple[float, float, float] | None = None
        points_link_max: tuple[float, float, float] | None = None
        bound_source = ""
        mesh = UsdGeom.Mesh(prim)
        if mesh:
            points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
            if points:
                points_link = []
                for point in points:
                    point_link = _transform_point_to_ancestor_local(prim, link_prim, Gf.Vec3d(point))
                    points_link.append((float(point_link[0]), float(point_link[1]), float(point_link[2])))

                points_link_tensor = torch.tensor(
                    points_link,
                    device=body_pos_w.device,
                    dtype=body_pos_w.dtype,
                )
                points_link_min = tuple(float(value) for value in points_link_tensor.min(dim=0).values.tolist())
                points_link_max = tuple(float(value) for value in points_link_tensor.max(dim=0).values.tolist())
                quat_xyz = body_quat_w[1:4].expand_as(points_link_tensor)
                t = 2.0 * torch.cross(quat_xyz, points_link_tensor, dim=-1)
                rotated = points_link_tensor + body_quat_w[0] * t + torch.cross(quat_xyz, t, dim=-1)
                points_world_tensor = rotated + body_pos_w.unsqueeze(0)
                points_world = [point for point in points_world_tensor]
                bound_source = "mesh_points_plus_runtime_body_pose"

        if points_world:
            xs = [float(point[0].item()) for point in points_world]
            ys = [float(point[1].item()) for point in points_world]
            zs = [float(point[2].item()) for point in points_world]
            minimum = (min(xs), min(ys), min(zs))
            maximum = (max(xs), max(ys), max(zs))
        else:
            # Primitive colliders or collision APIs applied to an enclosing Xform
            # still use a local one-prim bbox fallback. This path is not expected
            # for the current wheel meshes, but keeps the audit generic.
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
                useExtentsHint=False,
                ignoreVisibility=True,
            )
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            bbox_minimum = aligned.GetMin()
            bbox_maximum = aligned.GetMax()
            minimum = (float(bbox_minimum[0]), float(bbox_minimum[1]), float(bbox_minimum[2]))
            maximum = (float(bbox_maximum[0]), float(bbox_maximum[1]), float(bbox_maximum[2]))
            bound_source = "bbox_fallback"

        if not all(math.isfinite(float(value)) for value in (*minimum, *maximum)):
            continue
        records.append(
            {
                "prim_path": str(prim.GetPath()),
                "prim_type": prim.GetTypeName(),
                "is_boundable": bool(UsdGeom.Boundable(prim)),
                "approximation": None if approximation is None else str(approximation),
                "bound_source": bound_source,
                "point_count": len(points_world),
                "link_local_aabb_min": None if points_link_min is None else list(points_link_min),
                "link_local_aabb_max": None if points_link_max is None else list(points_link_max),
                "world_aabb_min": [float(minimum[0]), float(minimum[1]), float(minimum[2])],
                "world_aabb_max": [float(maximum[0]), float(maximum[1]), float(maximum[2])],
                "lowest_world_z": float(minimum[2]),
            }
        )
    if not records:
        print(
            "[AssetAudit][collision] no CollisionAPI prim found; scanned subtree="
            + json.dumps(scanned_prims[:200], indent=2),
            flush=True,
        )
    return records


def _collision_lowest_audit(env: Any, strokes: list[float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Unable to access the live USD stage.")

    print("[AssetAudit][collision] resolving runtime wheel link paths...", flush=True)
    link_paths = _runtime_link_paths(robot, WHEEL_BODIES)
    physx_link_indices = _runtime_link_indices(robot, link_paths)
    wheel_body_ids, _ = _find_ids(robot, WHEEL_BODIES, kind="body")
    body_ids_by_name = {name: int(wheel_body_ids[index].item()) for index, name in enumerate(WHEEL_BODIES)}
    print(f"[AssetAudit][collision] wheel_link_paths={link_paths}", flush=True)
    print(
        f"[AssetAudit][collision] isaaclab_body_ids={body_ids_by_name} physx_link_indices={physx_link_indices}",
        flush=True,
    )
    effective_height = max(float(args_cli.base_height), 1.5)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for stroke in strokes:
        print(f"[AssetAudit][collision] setting exact pose for stroke={stroke:.3f}...", flush=True)
        pose = _set_exact_geometry_pose(env, stroke=stroke, height=effective_height)
        print(f"[AssetAudit][collision] exact pose synchronized for stroke={stroke:.3f}", flush=True)
        live_root_pos_w = _get_live_root_position_w(robot)
        base_z = float(live_root_pos_w[2].item())
        per_wheel: dict[str, Any] = {}
        lowest: dict[str, float] = {}

        for wheel_index, (corner, body_name) in enumerate(zip(SEMANTIC_CORNERS, WHEEL_BODIES)):
            print(
                f"[AssetAudit][collision] scanning {corner}/{body_name} at {link_paths[body_name]}...",
                flush=True,
            )
            body_id = int(wheel_body_ids[wheel_index].item())
            physx_link_index = int(physx_link_indices[body_name])
            live_body_pos_w, live_body_quat_w = _get_live_link_pose_w(robot, physx_link_index)
            cached_body_pos_w = robot.data.body_pos_w[0, body_id]
            cache_position_error = float(torch.norm(live_body_pos_w - cached_body_pos_w).item())
            collision_records = _collision_prim_bounds(
                stage,
                link_paths[body_name],
                body_pos_w=live_body_pos_w,
                body_quat_w=live_body_quat_w,
            )
            print(
                f"[AssetAudit][collision] {corner} enabled_collision_prims={len(collision_records)}",
                flush=True,
            )
            if not collision_records:
                raise RuntimeError(
                    f"No enabled CollisionAPI prims found below wheel body {body_name!r} at {link_paths[body_name]!r}."
                )
            lowest_world_z = min(float(record["lowest_world_z"]) for record in collision_records)
            local_lowest_candidates = [
                float(record["link_local_aabb_min"][2])
                for record in collision_records
                if record.get("link_local_aabb_min") is not None
            ]
            link_local_lowest_z = min(local_lowest_candidates) if local_lowest_candidates else float("nan")
            lowest[corner] = lowest_world_z - base_z
            per_wheel[corner] = {
                "body_name": body_name,
                "body_id": body_id,
                "physx_link_index": physx_link_index,
                "link_path": link_paths[body_name],
                "collision_prim_count": len(collision_records),
                "live_body_pos_w": live_body_pos_w.detach().cpu().tolist(),
                "live_body_quat_wxyz": live_body_quat_w.detach().cpu().tolist(),
                "cached_body_pos_w": cached_body_pos_w.detach().cpu().tolist(),
                "cache_position_error": cache_position_error,
                "link_local_lowest_z": link_local_lowest_z,
                "lowest_world_z": lowest_world_z,
                "lowest_base_rel_z": lowest[corner],
                "collision_prims": collision_records,
            }

        mean_lowest = mean(lowest.values())
        z_range = max(lowest.values()) - min(lowest.values())
        diag_diff = lowest["lb"] + lowest["rf"] - lowest["lf"] - lowest["rb"]
        label = f"collision_stroke_{stroke:.2f}"
        row: dict[str, Any] = {
            "label": label,
            "stroke": float(stroke),
            "base_height": base_z,
            "collision_lowest_mean_base_rel_z": mean_lowest,
            "collision_lowest_z_range": z_range,
            "collision_lowest_z_diag_diff": diag_diff,
        }
        for corner_index, corner in enumerate(SEMANTIC_CORNERS):
            row[f"joint_pos_{corner}"] = float(pose["leg_joint_pos_semantic"][corner_index])
            row[f"physx_link_index_{corner}"] = per_wheel[corner]["physx_link_index"]
            row[f"live_body_world_z_{corner}"] = float(per_wheel[corner]["live_body_pos_w"][2])
            row[f"cached_body_world_z_{corner}"] = float(per_wheel[corner]["cached_body_pos_w"][2])
            row[f"link_local_lowest_z_{corner}"] = float(per_wheel[corner]["link_local_lowest_z"])
            row[f"collision_lowest_base_rel_z_{corner}"] = lowest[corner]
            row[f"collision_prim_count_{corner}"] = per_wheel[corner]["collision_prim_count"]
            row[f"body_cache_position_error_{corner}"] = per_wheel[corner]["cache_position_error"]
        rows.append(row)
        summaries[label] = {
            "pose": pose,
            "effective_base_height": effective_height,
            "wheel_link_paths": link_paths,
            "collision_lowest_base_rel_z": lowest,
            "collision_lowest_mean_base_rel_z": mean_lowest,
            "collision_lowest_mean_base_rel_z_mm": 1000.0 * mean_lowest,
            "collision_lowest_z_range": z_range,
            "collision_lowest_z_diag_diff": diag_diff,
            "collision_lowest_z_range_mm": 1000.0 * z_range,
            "collision_lowest_z_diag_diff_mm": 1000.0 * diag_diff,
            "per_wheel": per_wheel,
            "note": (
                "Bounds use only enabled runtime prims carrying UsdPhysics.CollisionAPI. "
                "Run with --disable_fabric so articulation transforms are mirrored to the live USD stage."
            ),
        }
    return rows, summaries


def _collect_contact_row(env: Any, *, label: str, step: int, yaw_deg: float, stroke: float | None = None, stiffness: float | None = None) -> dict[str, Any]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    wheel_body_ids, _ = _find_ids(robot, WHEEL_BODIES, kind="body")
    leg_ids, _ = _find_ids(robot, LEG_JOINTS, kind="joint")
    contact_sensor = unwrapped.scene.sensors["wheel_contact_forces"]
    contact_body_ids, _ = contact_sensor.find_bodies(list(WHEEL_BODIES), preserve_order=True)
    contact_body_ids = torch.as_tensor(contact_body_ids, device=unwrapped.device, dtype=torch.long)
    force = torch.max(torch.norm(contact_sensor.data.net_forces_w_history[0, :, contact_body_ids, :], dim=-1), dim=0).values
    total = torch.clamp(force.sum(), min=1.0e-6)
    diag_force = force[0] + force[2] - force[1] - force[3]
    wheel_pos = robot.data.body_pos_w[0, wheel_body_ids, :]
    root_pos = robot.data.root_pos_w[0]
    row: dict[str, Any] = {
        "label": label,
        "step": int(step),
        "yaw_deg": float(yaw_deg),
        "stroke": "" if stroke is None else float(stroke),
        "stiffness": "" if stiffness is None else float(stiffness),
        "contact_diag_force": float(diag_force.item()),
        "contact_diag_ratio": float((diag_force / total).item()),
        "zero_contact_ratio": float((force < 1.0).to(torch.float32).mean().item()),
        "contact_force_imbalance": float(((force.max() - force.min()) / total).item()),
        "base_roll": float(robot.data.root_link_quat_w[0, 1].item()),
        "base_pitch": float(robot.data.root_link_quat_w[0, 2].item()),
    }
    for idx, corner in enumerate(SEMANTIC_CORNERS):
        row[f"contact_force_{corner}"] = float(force[idx].item())
        row[f"wheel_hub_world_z_{corner}"] = float(wheel_pos[idx, 2].item())
        row[f"wheel_hub_base_rel_z_{corner}"] = float((wheel_pos[idx, 2] - root_pos[2]).item())
        row[f"joint_pos_{corner}"] = float(robot.data.joint_pos[0, leg_ids[idx]].item())
        row[f"joint_target_{corner}"] = float(robot.data.joint_pos_target[0, leg_ids[idx]].item())
    row["joint_tracking_error_mean"] = mean(abs(row[f"joint_target_{corner}"] - row[f"joint_pos_{corner}"]) for corner in SEMANTIC_CORNERS)
    row["hub_z_range"] = max(row[f"wheel_hub_base_rel_z_{corner}"] for corner in SEMANTIC_CORNERS) - min(
        row[f"wheel_hub_base_rel_z_{corner}"] for corner in SEMANTIC_CORNERS
    )
    row["hub_z_diag_diff"] = (
        row["wheel_hub_base_rel_z_lb"]
        + row["wheel_hub_base_rel_z_rf"]
        - row["wheel_hub_base_rel_z_lf"]
        - row["wheel_hub_base_rel_z_rb"]
    )
    return row


def _run_stroke_condition(env: Any, *, stroke: float, yaw: float, steps: int, height: float, label: str, stiffness: float | None = None) -> list[dict[str, Any]]:
    unwrapped = env.unwrapped
    leg_slice = _get_action_slice(unwrapped, "leg_hydraulic")
    wheel_slice = _get_action_slice(unwrapped, "wheel_motor_csv")
    actions = _zero_actions(env)
    action_value = max(-1.0, min(1.0, 2.0 * float(stroke) - 1.0))
    actions[:, leg_slice] = action_value
    actions[:, wheel_slice] = 0.0
    rows = []
    for step in range(steps):
        _step_pinned(env, actions, yaw=yaw, height=height)
        rows.append(_collect_contact_row(env, label=label, step=step, yaw_deg=math.degrees(yaw), stroke=stroke, stiffness=stiffness))
    return rows


def _summarize_rows(rows: list[dict[str, Any]], *, warmup_steps: int) -> dict[str, Any]:
    active = [row for row in rows if int(row["step"]) >= int(warmup_steps)]
    keys = [
        "contact_force_lb",
        "contact_force_lf",
        "contact_force_rf",
        "contact_force_rb",
        "contact_diag_ratio",
        "zero_contact_ratio",
        "contact_force_imbalance",
        "hub_z_range",
        "hub_z_diag_diff",
        "joint_tracking_error_mean",
    ]
    return {key: _stats([float(row[key]) for row in active]) for key in keys}


def _coplanar_audit(env: Any, strokes: list[float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for stroke in strokes:
        label = f"coplanar_stroke_{stroke:.2f}"
        condition_rows = _run_stroke_condition(
            env, stroke=stroke, yaw=0.0, steps=args_cli.steps, height=args_cli.base_height, label=label
        )
        rows.extend(condition_rows)
        final = condition_rows[-1]
        summaries[label] = {
            "final_wheel_hub_base_rel_z": {corner: final[f"wheel_hub_base_rel_z_{corner}"] for corner in SEMANTIC_CORNERS},
            "final_hub_z_range": final["hub_z_range"],
            "final_hub_z_diag_diff": final["hub_z_diag_diff"],
            "contact_summary": _summarize_rows(condition_rows, warmup_steps=args_cli.warmup_steps),
            "note": "Hub z uses runtime body pose. Collision lowest-z/AABB requires USD bbox extraction and is not used for dynamics here.",
        }
    return rows, summaries


def _yaw_audit(env: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for yaw_deg, yaw in ((0.0, 0.0), (180.0, math.pi)):
        label = f"yaw_{int(yaw_deg)}"
        condition_rows = _run_stroke_condition(
            env, stroke=0.5, yaw=yaw, steps=args_cli.steps, height=args_cli.base_height, label=label
        )
        rows.extend(condition_rows)
        summaries[label] = _summarize_rows(condition_rows, warmup_steps=args_cli.warmup_steps)
    return rows, summaries


def _stiffness_audit(stiffness_values: list[float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for stiffness in stiffness_values:
        env = _make_env(
            task=args_cli.task,
            num_envs=1,
            seed=args_cli.seed,
            device=args_cli.device,
            disable_fabric=args_cli.disable_fabric,
            leg_stiffness=stiffness,
        )
        try:
            label = f"stiffness_{stiffness:.3g}"
            condition_rows = _run_stroke_condition(
                env, stroke=0.5, yaw=0.0, steps=args_cli.steps, height=args_cli.base_height, label=label, stiffness=stiffness
            )
            rows.extend(condition_rows)
            summaries[label] = _summarize_rows(condition_rows, warmup_steps=args_cli.warmup_steps)
        finally:
            env.close()
    return rows, summaries


def main() -> None:
    output_json = Path(args_cli.output_json).expanduser()
    output_csv = Path(args_cli.output_csv).expanduser()
    strokes = _parse_float_list(args_cli.strokes, name="strokes")
    stiffness_values = _parse_float_list(args_cli.stiffness_values, name="stiffness_values")
    report: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "task": args_cli.task,
        "mode": args_cli.mode,
        "static_asset": _static_asset_report(),
        "not_implemented_runtime_ablations": {
            "cylinder_collision_replacement": "Not applied. This script does not mutate or overlay the runtime USD collision geometry yet.",
            "inertia_equalization": "Not applied. Runtime mass/inertia readout is reported; temporary mass/inertia rewriting is left for a separate controlled script.",
        },
    }
    all_rows: list[dict[str, Any]] = []

    needs_runtime = args_cli.mode in ("all", "runtime", "coplanar", "collision", "yaw")
    env = None
    if needs_runtime:
        gravity = (0.0, 0.0, 0.0) if args_cli.mode in ("coplanar", "collision") else None
        env = _make_env(
            task=args_cli.task,
            num_envs=args_cli.num_envs,
            seed=args_cli.seed,
            device=args_cli.device,
            disable_fabric=args_cli.disable_fabric,
            gravity=gravity,
        )
        try:
            if args_cli.mode == "collision":
                print("[AssetAudit][collision] environment ready; entering minimal geometry path...", flush=True)
                report["runtime"] = {
                    "joint_names_all": list(env.unwrapped.scene["robot"].joint_names),
                    "body_names_all": list(env.unwrapped.scene["robot"].body_names),
                    "note": "Full runtime mass/inertia report skipped in collision-only mode to isolate geometry diagnostics.",
                }
            else:
                report["runtime"] = _runtime_report(env)
            if args_cli.mode in ("all", "coplanar"):
                coplanar_rows, coplanar_summary = _coplanar_audit(env, strokes)
                all_rows.extend(coplanar_rows)
                report["coplanar"] = coplanar_summary
            if args_cli.mode in ("all", "collision"):
                collision_rows, collision_summary = _collision_lowest_audit(env, strokes)
                all_rows.extend(collision_rows)
                report["collision"] = collision_summary
            if args_cli.mode in ("all", "yaw"):
                yaw_rows, yaw_summary = _yaw_audit(env)
                all_rows.extend(yaw_rows)
                report["yaw"] = yaw_summary
        finally:
            env.close()

    if args_cli.mode in ("all", "stiffness"):
        stiffness_rows, stiffness_summary = _stiffness_audit(stiffness_values)
        all_rows.extend(stiffness_rows)
        report["stiffness"] = stiffness_summary

    _write_json(output_json, report)
    _write_csv(output_csv, all_rows)
    print("[AssetAudit] wrote JSON:", output_json, flush=True)
    if all_rows:
        print("[AssetAudit] wrote CSV:", output_csv, flush=True)
    print("[AssetAudit] runtime asset:", RANGER_USD_PATH, flush=True)
    print("[AssetAudit] source URDF:", RANGER_URDF_PATH, flush=True)
    if "runtime" in report and "leg_joint_ids_semantic" in report["runtime"]:
        print("[AssetAudit] leg_joint_ids_semantic:", report["runtime"]["leg_joint_ids_semantic"], flush=True)
        print("[AssetAudit] wheel_joint_ids_semantic:", report["runtime"]["wheel_joint_ids_semantic"], flush=True)
        print("[AssetAudit] total_com_b:", report["runtime"]["total_com_b"], flush=True)
    if "collision" in report:
        for label, summary in report["collision"].items():
            print(
                f"[AssetAudit] {label} mean_lowest_base_rel_z_mm="
                f"{summary['collision_lowest_mean_base_rel_z_mm']:+.3f} "
                f"collision_lowest_range_mm={summary['collision_lowest_z_range_mm']:+.3f} "
                f"diag_diff_mm={summary['collision_lowest_z_diag_diff_mm']:+.3f}",
                flush=True,
            )
    if "yaw" in report:
        for label, summary in report["yaw"].items():
            print(
                f"[AssetAudit] {label} contact_diag_ratio_mean="
                f"{summary['contact_diag_ratio']['mean']:+.6f}",
                flush=True,
            )
    if "stiffness" in report:
        for label, summary in report["stiffness"].items():
            print(
                f"[AssetAudit] {label} contact_diag_ratio_mean="
                f"{summary['contact_diag_ratio']['mean']:+.6f} "
                f"tracking_error_mean={summary['joint_tracking_error_mean']['mean']:+.6f}",
                flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
