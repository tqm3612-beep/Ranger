#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug whether Ranger wheel DOFs can be driven by direct PhysX joint efforts."""

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug direct wheel joint efforts on suspended Ranger.")
parser.add_argument("--task", type=str, default="Template-Ranger-Forward-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to print.")
parser.add_argument("--init_height", type=float, default=1.5, help="Initial base height for the suspended test.")
parser.add_argument(
    "--inspect_only",
    action="store_true",
    default=False,
    help="Run only static inspection/metadata printing and skip one-wheel execution intentionally.",
)
parser.add_argument(
    "--pin_base",
    action="store_true",
    default=False,
    help="Force the floating base pose and velocities back to the initial suspended state before and after each step.",
)
parser.add_argument(
    "--disable_wheel_collisions",
    action="store_true",
    default=False,
    help="Temporarily disable collisionEnabled on the four wheel links to isolate wheel-joint motion from mesh contacts.",
)
parser.add_argument("--settle_steps", type=int, default=5, help="Number of zero-effort settle steps before tests.")
parser.add_argument("--steps_per_effort", type=int, default=40, help="Number of sim steps to run for each effort.")
parser.add_argument("--print_every", type=int, default=5, help="Print debug information every N steps.")
parser.add_argument(
    "--efforts",
    type=float,
    nargs=4,
    default=(20.0, -20.0, 100.0, -100.0),
    help="Wheel effort targets in N*m to test sequentially.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from pxr import Sdf
from pxr import UsdPhysics

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Ranger.tasks  # noqa: F401


def _get_joint_prim_type_name(joint_prim) -> str:
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        return "RevoluteJoint"
    if joint_prim.IsA(UsdPhysics.PrismaticJoint):
        return "PrismaticJoint"
    if joint_prim.IsA(UsdPhysics.FixedJoint):
        return "FixedJoint"
    return joint_prim.GetTypeName()


def _get_usd_joint_debug_info(stage, joint_name: str) -> dict[str, object]:
    joint_prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/joints/{joint_name}")
    if not joint_prim.IsValid():
        return {"warning": f"USD joint prim for {joint_name} not found at {joint_prim.GetPath()}."}

    info = {"type": _get_joint_prim_type_name(joint_prim)}
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        joint_api = UsdPhysics.RevoluteJoint(joint_prim)
        info["axis"] = joint_api.GetAxisAttr().Get()
        info["lower"] = joint_api.GetLowerLimitAttr().Get()
        info["upper"] = joint_api.GetUpperLimitAttr().Get()
        drive_model = "angular"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        joint_api = UsdPhysics.PrismaticJoint(joint_prim)
        info["axis"] = joint_api.GetAxisAttr().Get()
        info["lower"] = joint_api.GetLowerLimitAttr().Get()
        info["upper"] = joint_api.GetUpperLimitAttr().Get()
        drive_model = "linear"
    else:
        info["axis"] = None
        info["lower"] = None
        info["upper"] = None
        drive_model = None

    attr_names = {
        "drive_stiffness": f"drive:{drive_model}:physics:stiffness" if drive_model else None,
        "drive_damping": f"drive:{drive_model}:physics:damping" if drive_model else None,
        "max_force": f"drive:{drive_model}:physics:maxForce" if drive_model else None,
        "joint_friction": "physxJoint:jointFriction",
        "armature": "physxJoint:armature",
    }
    for key, attr_name in attr_names.items():
        if attr_name is None:
            info[key] = None
            continue
        attr = joint_prim.GetAttribute(attr_name)
        info[key] = attr.Get() if attr.IsValid() else None

    info["body0_targets"] = [str(path) for path in joint_prim.GetRelationship("physics:body0").GetTargets()]
    info["body1_targets"] = [str(path) for path in joint_prim.GetRelationship("physics:body1").GetTargets()]
    for key in ["local_pos0", "local_rot0", "local_pos1", "local_rot1"]:
        attr_name = {
            "local_pos0": "physics:localPos0",
            "local_rot0": "physics:localRot0",
            "local_pos1": "physics:localPos1",
            "local_rot1": "physics:localRot1",
        }[key]
        attr = joint_prim.GetAttribute(attr_name)
        info[key] = attr.Get() if attr.IsValid() else None

    return info


def _print_joint_inventory(robot, env_id: int, wheel_joint_names: list[str], wheel_joint_ids: list[int]) -> None:
    print("[INFO]: articulation joint inventory", flush=True)
    for joint_id, joint_name in enumerate(robot.data.joint_names):
        joint_pos = float(robot.data.joint_pos[env_id, joint_id].item())
        joint_vel = float(robot.data.joint_vel[env_id, joint_id].item())
        marker = " <wheel>" if joint_name in wheel_joint_names else ""
        print(
            f"  joint_name={joint_name:<8} joint_id={joint_id:<2} "
            f"joint_pos={joint_pos: .6f} joint_vel={joint_vel: .6f}{marker}",
            flush=True,
        )

    for wheel_name, wheel_id in zip(wheel_joint_names, wheel_joint_ids):
        if wheel_id >= len(robot.data.joint_names):
            print(f"WARNING: {wheel_name} joint_id={wheel_id} is outside articulation joint_names range.", flush=True)


def _print_wheel_usd_info(stage, wheel_joint_names: list[str]) -> None:
    print("[INFO]: USD wheel joint metadata", flush=True)
    for wheel_name in wheel_joint_names:
        info = _get_usd_joint_debug_info(stage, wheel_name)
        if "warning" in info:
            print(f"WARNING: {info['warning']}", flush=True)
            continue
        lower = info["lower"]
        upper = info["upper"]
        if lower == upper:
            print(f"WARNING: {wheel_name} has lower==upper=={lower}.", flush=True)
        print(
            f"  {wheel_name}: type={info['type']} axis={info['axis']} "
            f"lower={lower} upper={upper} "
            f"drive_stiffness={info['drive_stiffness']} drive_damping={info['drive_damping']} "
            f"maxForce={info['max_force']} joint_friction={info['joint_friction']} armature={info['armature']} "
            f"body0={info['body0_targets']} body1={info['body1_targets']} "
            f"localPos0={info['local_pos0']} localRot0={info['local_rot0']} "
            f"localPos1={info['local_pos1']} localRot1={info['local_rot1']}",
            flush=True,
        )


def _apply_direct_wheel_efforts(
    robot, env_id: int, num_joints: int, wheel_joint_ids: list[int], effort_targets_per_wheel: list[float]
) -> torch.Tensor:
    env_ids = torch.tensor([env_id], device=robot.device, dtype=torch.long)
    joint_ids = torch.tensor(wheel_joint_ids, device=robot.device, dtype=torch.long)
    effort_targets = torch.tensor(
        [effort_targets_per_wheel],
        dtype=robot.data.joint_pos.dtype,
        device=robot.device,
    )
    robot.set_joint_effort_target(effort_targets, joint_ids=joint_ids, env_ids=env_ids)
    return effort_targets[0].detach().clone()


def _find_robot_prim_by_name(stage, prim_name: str):
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    if not robot_prim.IsValid():
        return None

    stack = [robot_prim]
    while stack:
        prim = stack.pop()
        if prim.GetName() == prim_name:
            return prim
        children = list(prim.GetChildren())
        stack.extend(reversed(children))
    return None


def _get_link_collision_enabled(stage, body_name: str) -> object:
    body_prim = _find_robot_prim_by_name(stage, body_name)
    if body_prim is None:
        return None

    values: list[bool] = []
    stack = [body_prim]
    while stack:
        prim = stack.pop()
        attr = prim.GetAttribute("physics:collisionEnabled")
        if attr.IsValid():
            value = attr.Get()
            if value is not None:
                values.append(bool(value))
        stack.extend(list(prim.GetChildren()))

    if not values:
        return None
    if all(values):
        return True
    if not any(values):
        return False
    return values


def _set_link_collision_enabled(stage, body_name: str, enabled: bool) -> list[tuple[object, object]]:
    """Set collisionEnabled recursively for all collision-bearing prims under one wheel link."""

    body_prim = _find_robot_prim_by_name(stage, body_name)
    if body_prim is None:
        return []

    changed_attrs: list[tuple[object, object]] = []
    stack = [body_prim]
    while stack:
        prim = stack.pop()
        attr = prim.GetAttribute("physics:collisionEnabled")
        if attr.IsValid():
            previous_value = attr.Get()
            attr.Set(enabled)
            changed_attrs.append((attr, previous_value))
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            attr = prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool)
            attr.Set(enabled)
            changed_attrs.append((attr, None))
        stack.extend(list(prim.GetChildren()))
    return changed_attrs


def _restore_collision_attrs(changed_attrs: list[tuple[object, object]]) -> None:
    for attr, previous_value in changed_attrs:
        if previous_value is None:
            attr.Clear()
        else:
            attr.Set(previous_value)


def _tensor_row_to_diag3(values: torch.Tensor) -> tuple[float, float, float]:
    return (float(values[0].item()), float(values[4].item()), float(values[8].item()))


def _tensor_to_tuple(values: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(v.item()) for v in values)


def _print_wheel_structure_comparison(robot, stage, env_id: int, wheel_body_names: list[str], wheel_joint_names: list[str]) -> None:
    wheel_body_ids, resolved_body_names = robot.find_bodies(wheel_body_names, preserve_order=True)
    wheel_body_ids = list(wheel_body_ids)
    resolved_body_names = list(resolved_body_names)

    print("[INFO]: wheel link / joint structural comparison", flush=True)
    link_info_by_name: dict[str, dict[str, object]] = {}
    joint_info_by_name = {name: _get_usd_joint_debug_info(stage, name) for name in wheel_joint_names}

    for body_name, body_id in zip(resolved_body_names, wheel_body_ids):
        mass = float(robot.data.default_mass[env_id, body_id].item())
        inertia_diag = _tensor_row_to_diag3(robot.data.default_inertia[env_id, body_id])
        com_pos = _tensor_to_tuple(robot.data.body_com_pos_b[env_id, body_id])
        com_quat = _tensor_to_tuple(robot.data.body_com_quat_b[env_id, body_id])
        collision_enabled = _get_link_collision_enabled(stage, body_name)
        joint_info = joint_info_by_name[body_name]
        parent_link = joint_info.get("body0_targets", [])
        child_link = joint_info.get("body1_targets", [])
        joint_frame = {
            "localPos0": joint_info.get("local_pos0"),
            "localRot0": joint_info.get("local_rot0"),
            "localPos1": joint_info.get("local_pos1"),
            "localRot1": joint_info.get("local_rot1"),
        }

        link_info_by_name[body_name] = {
            "mass": mass,
            "inertia_diag": inertia_diag,
            "com_pos": com_pos,
            "com_quat": com_quat,
            "parent_link": parent_link,
            "child_link": child_link,
            "joint_frame": joint_frame,
            "collision_enabled": collision_enabled,
        }
        print(
            f"  {body_name}: mass={mass:.6f} inertia_diag={inertia_diag} "
            f"com_pos={com_pos} com_quat={com_quat} "
            f"parent_link={parent_link} child_link={child_link} "
            f"joint_frame={joint_frame} collision_enabled={collision_enabled}",
            flush=True,
        )

    print("[COMPARE]: right-side vs left-side wheel structural differences", flush=True)
    for left_name, right_name in [("w_lb", "w_rb"), ("w_lf", "w_rf")]:
        left = link_info_by_name.get(left_name)
        right = link_info_by_name.get(right_name)
        if left is None or right is None:
            print(f"  WARNING: cannot compare {left_name} vs {right_name}, body info missing.", flush=True)
            continue
        print(f"  pair {left_name} vs {right_name}", flush=True)
        for field in [
            "mass",
            "inertia_diag",
            "com_pos",
            "com_quat",
            "parent_link",
            "child_link",
            "joint_frame",
            "collision_enabled",
        ]:
            left_value = left[field]
            right_value = right[field]
            marker = "DIFF" if left_value != right_value else "same"
            print(f"    {field}: {marker} | left={left_value} | right={right_value}", flush=True)


def _print_effort_step(
    robot,
    env_id: int,
    step: int,
    wheel_joint_names: list[str],
    wheel_joint_ids: list[int],
    commanded_targets: list[float],
    effort_tensor: torch.Tensor,
    pin_base_enabled: bool,
    disable_wheel_collisions_enabled: bool,
) -> torch.Tensor:
    joint_pos = robot.data.joint_pos[env_id, wheel_joint_ids]
    joint_vel = robot.data.joint_vel[env_id, wheel_joint_ids]
    applied_torque = robot.data.applied_torque[env_id, wheel_joint_ids]
    computed_torque = None
    if hasattr(robot.data, "computed_torque") and robot.data.computed_torque is not None:
        computed_torque = robot.data.computed_torque[env_id, wheel_joint_ids]
    base_lin_vel_x = float(robot.data.root_lin_vel_b[env_id, 0].item())
    base_ang_vel_z = float(robot.data.root_ang_vel_b[env_id, 2].item())

    print(
        f"[STEP {step:04d}] base_lin_vel_x={base_lin_vel_x: .4f} "
        f"base_ang_vel_z={base_ang_vel_z: .4f} "
        f"pin_base={int(pin_base_enabled)} "
        f"disable_wheel_collisions={int(disable_wheel_collisions_enabled)}",
        flush=True,
    )
    print(f"  effort_tensor={effort_tensor.detach().cpu().tolist()}", flush=True)
    print(f"  applied_torque={applied_torque.detach().cpu().tolist()}", flush=True)
    if computed_torque is not None:
        print(f"  computed_torque={computed_torque.detach().cpu().tolist()}", flush=True)
    print(f"  joint_vel={joint_vel.detach().cpu().tolist()}", flush=True)
    for wheel_name, wheel_id, target_value, pos_value, vel_value in zip(
        wheel_joint_names, wheel_joint_ids, commanded_targets, joint_pos, joint_vel
    ):
        print(
            f"  {wheel_name}: joint_id={wheel_id} "
            f"effort_target={float(target_value): .4f} "
            f"joint_pos={float(pos_value.item()): .6f} "
            f"joint_vel={float(vel_value.item()): .6f}",
            flush=True,
        )
    return joint_vel.detach().clone()


def _warn_on_unexpected_wheel_motion(
    wheel_joint_names: list[str], wheel_joint_ids: list[int], commanded_targets: list[float], joint_vel: torch.Tensor
) -> None:
    movement_eps = 1e-3
    commanded_indices = [index for index, target in enumerate(commanded_targets) if abs(float(target)) > 1e-6]
    if len(commanded_indices) != 1:
        return

    commanded_index = commanded_indices[0]
    commanded_name = wheel_joint_names[commanded_index]
    commanded_id = wheel_joint_ids[commanded_index]
    commanded_speed = abs(float(joint_vel[commanded_index].item()))

    other_speeds = []
    for index, (wheel_name, wheel_id) in enumerate(zip(wheel_joint_names, wheel_joint_ids)):
        if index == commanded_index:
            continue
        other_speeds.append((wheel_name, wheel_id, abs(float(joint_vel[index].item()))))

    max_other_name, max_other_id, max_other_speed = max(other_speeds, key=lambda item: item[2])
    if commanded_speed < movement_eps and max_other_speed > movement_eps:
        print(
            f"WARNING: commanded only {commanded_name} (joint_id={commanded_id}) "
            f"but {max_other_name} (joint_id={max_other_id}) moved more "
            f"({max_other_speed:.6f} > {commanded_speed:.6f}).",
            flush=True,
        )
    elif max_other_speed > commanded_speed * 1.1 and max_other_speed > movement_eps:
        print(
            f"WARNING: unexpected coupling? commanded {commanded_name} (joint_id={commanded_id}) "
            f"but {max_other_name} (joint_id={max_other_id}) has larger |joint_vel| "
            f"({max_other_speed:.6f} > {commanded_speed:.6f}).",
            flush=True,
        )


def _build_pinned_root_state(robot, env_id: int, env_origins: torch.Tensor, init_height: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the suspended root pose/velocity used to isolate wheel motion from base coupling."""

    root_pose = robot.data.default_root_state[env_id : env_id + 1, :7].clone()
    root_pose[:, :3] = env_origins[env_id : env_id + 1] + torch.tensor(
        [[0.0, 0.0, init_height]], dtype=root_pose.dtype, device=root_pose.device
    )
    root_velocity = torch.zeros((1, 6), dtype=root_pose.dtype, device=root_pose.device)
    return root_pose, root_velocity


def _pin_robot_root_state(robot, env_id: int, root_pose: torch.Tensor, root_velocity: torch.Tensor) -> None:
    """Force the floating base back to the diagnostic suspended state."""

    env_ids = torch.tensor([env_id], device=robot.device, dtype=torch.long)
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def _run_direct_effort_step(
    robot,
    unwrapped_env,
    env_id: int,
    wheel_joint_ids: list[int],
    commanded_targets: list[float],
    pinned_root_pose: torch.Tensor | None,
    pinned_root_velocity: torch.Tensor | None,
) -> None:
    if pinned_root_pose is not None and pinned_root_velocity is not None:
        _pin_robot_root_state(robot, env_id=env_id, root_pose=pinned_root_pose, root_velocity=pinned_root_velocity)

    _apply_direct_wheel_efforts(
        robot=robot,
        env_id=env_id,
        num_joints=len(robot.data.joint_names),
        wheel_joint_ids=wheel_joint_ids,
        effort_targets_per_wheel=commanded_targets,
    )
    unwrapped_env.scene.write_data_to_sim()
    unwrapped_env.sim.step(render=False)
    unwrapped_env.scene.update(dt=unwrapped_env.physics_dt)

    if pinned_root_pose is not None and pinned_root_velocity is not None:
        _pin_robot_root_state(robot, env_id=env_id, root_pose=pinned_root_pose, root_velocity=pinned_root_velocity)
        unwrapped_env.scene.write_data_to_sim()


def _reset_wheel_state_for_case(
    env,
    unwrapped,
    robot,
    env_id: int,
    wheel_joint_ids: list[int],
    pinned_root_pose: torch.Tensor | None,
    pinned_root_velocity: torch.Tensor | None,
    settle_steps: int,
) -> None:
    """Clear wheel residual state before each one-wheel effort case."""

    print("[TRACE] env.reset before one-wheel case", flush=True)
    env.reset()
    print("[TRACE] env.reset after one-wheel case", flush=True)

    env_ids = torch.tensor([env_id], device=robot.device, dtype=torch.long)
    joint_ids = torch.tensor(wheel_joint_ids, device=robot.device, dtype=torch.long)
    zero_wheel_pos = torch.zeros((1, len(wheel_joint_ids)), dtype=robot.data.joint_pos.dtype, device=robot.device)
    zero_wheel_vel = torch.zeros_like(zero_wheel_pos)
    zero_wheel_effort = torch.zeros_like(zero_wheel_pos)

    # Clear any residual wheel state from the previous case before measuring isolated effort response.
    robot.write_joint_state_to_sim(zero_wheel_pos, zero_wheel_vel, joint_ids=joint_ids, env_ids=env_ids)
    robot.set_joint_effort_target(zero_wheel_effort, joint_ids=joint_ids, env_ids=env_ids)
    if pinned_root_pose is not None and pinned_root_velocity is not None:
        _pin_robot_root_state(robot, env_id=env_id, root_pose=pinned_root_pose, root_velocity=pinned_root_velocity)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()
    unwrapped.scene.update(dt=unwrapped.physics_dt)

    print(
        f"[TRACE] case reset state wheel_joint_ids={wheel_joint_ids} "
        f"wheel_joint_pos={zero_wheel_pos.tolist()} "
        f"wheel_joint_vel={zero_wheel_vel.tolist()} "
        f"wheel_effort={zero_wheel_effort.tolist()} "
        f"settle_steps={settle_steps}",
        flush=True,
    )

    for settle_step in range(settle_steps):
        _run_direct_effort_step(
            robot=robot,
            unwrapped_env=unwrapped,
            env_id=env_id,
            wheel_joint_ids=wheel_joint_ids,
            commanded_targets=[0.0, 0.0, 0.0, 0.0],
            pinned_root_pose=pinned_root_pose,
            pinned_root_velocity=pinned_root_velocity,
        )
        if settle_step == 0 or settle_step == settle_steps - 1:
            current_vel = robot.data.joint_vel[env_id, wheel_joint_ids].detach().cpu().tolist()
            print(
                f"[TRACE] case settle step={settle_step} "
                f"wheel_joint_ids={wheel_joint_ids} "
                f"joint_vel={current_vel}",
                flush=True,
            )


def _run_one_wheel_tests(
    env,
    unwrapped,
    robot,
    wheel_joint_names: list[str],
    wheel_joint_ids: list[int],
    pinned_root_pose: torch.Tensor | None,
    pinned_root_velocity: torch.Tensor | None,
) -> list[dict[str, object]]:
    single_wheel_efforts = [100.0, -100.0]
    case_summaries: list[dict[str, object]] = []
    print(
        f"[TRACE] run_one_wheel_tests start "
        f"len(wheel_joint_names)={len(wheel_joint_names)} "
        f"wheel_joint_ids={wheel_joint_ids} "
        f"effort_values={single_wheel_efforts} "
        f"steps_per_effort={args_cli.steps_per_effort} "
        f"print_every={args_cli.print_every}",
        flush=True,
    )

    for commanded_wheel_index, commanded_wheel_name in enumerate(wheel_joint_names):
        for effort in single_wheel_efforts:
            print(
                f"[TRACE] one-wheel case start wheel={commanded_wheel_name} "
                f"wheel_index={commanded_wheel_index} "
                f"joint_id={wheel_joint_ids[commanded_wheel_index]} "
                f"effort={effort: .4f}",
                flush=True,
            )
            print(
                f"[ONE_WHEEL] commanded_wheel={commanded_wheel_name} "
                f"commanded_joint_id={wheel_joint_ids[commanded_wheel_index]} "
                f"effort={effort: .4f} N*m",
                flush=True,
            )
            commanded_targets = [0.0] * len(wheel_joint_ids)
            commanded_targets[commanded_wheel_index] = float(effort)
            print(
                f"[TRACE] case command mapping commanded_wheel={commanded_wheel_name} "
                f"target_joint_id={wheel_joint_ids[commanded_wheel_index]} "
                f"wheel_joint_ids={wheel_joint_ids} "
                f"effort_tensor={[commanded_targets]}",
                flush=True,
            )
            _reset_wheel_state_for_case(
                env=env,
                unwrapped=unwrapped,
                robot=robot,
                env_id=args_cli.env_id,
                wheel_joint_ids=wheel_joint_ids,
                pinned_root_pose=pinned_root_pose,
                pinned_root_velocity=pinned_root_velocity,
                settle_steps=max(args_cli.settle_steps, 1),
            )
            max_abs_joint_vel = torch.zeros(len(wheel_joint_ids), dtype=torch.float32, device=robot.device)
            final_joint_vel = torch.zeros(len(wheel_joint_ids), dtype=torch.float32, device=robot.device)

            for step in range(args_cli.steps_per_effort):
                _run_direct_effort_step(
                    robot=robot,
                    unwrapped_env=unwrapped,
                    env_id=args_cli.env_id,
                    wheel_joint_ids=wheel_joint_ids,
                    commanded_targets=commanded_targets,
                    pinned_root_pose=pinned_root_pose,
                    pinned_root_velocity=pinned_root_velocity,
                )
                joint_vel_now = robot.data.joint_vel[args_cli.env_id, wheel_joint_ids]
                max_abs_joint_vel = torch.maximum(max_abs_joint_vel, torch.abs(joint_vel_now))
                final_joint_vel = joint_vel_now.detach().clone()

                if step % args_cli.print_every != 0:
                    continue

                joint_vel = _print_effort_step(
                    robot=robot,
                    env_id=args_cli.env_id,
                    step=step,
                    wheel_joint_names=wheel_joint_names,
                    wheel_joint_ids=wheel_joint_ids,
                    commanded_targets=commanded_targets,
                    effort_tensor=torch.tensor(
                        [commanded_targets],
                        dtype=robot.data.joint_pos.dtype,
                        device=robot.device,
                    )[0],
                    pin_base_enabled=args_cli.pin_base,
                    disable_wheel_collisions_enabled=args_cli.disable_wheel_collisions,
                )
                _warn_on_unexpected_wheel_motion(
                    wheel_joint_names=wheel_joint_names,
                    wheel_joint_ids=wheel_joint_ids,
                    commanded_targets=commanded_targets,
                    joint_vel=joint_vel,
                )
            case_summaries.append(
                {
                    "commanded_wheel": commanded_wheel_name,
                    "target_joint_id": wheel_joint_ids[commanded_wheel_index],
                    "effort": float(effort),
                    "max_abs_joint_vel": [float(value) for value in max_abs_joint_vel.detach().cpu().tolist()],
                    "final_joint_vel": [float(value) for value in final_joint_vel.detach().cpu().tolist()],
                }
            )

    print("[TRACE] run_one_wheel_tests end", flush=True)
    return case_summaries


def _print_case_summary_table(case_summaries: list[dict[str, object]], wheel_joint_names: list[str]) -> None:
    print("[SUMMARY] one-wheel direct-effort cases", flush=True)
    print(
        "case | target_joint_id | effort | max_abs_joint_vel[w_lb,w_lf,w_rf,w_rb] | final_joint_vel[w_lb,w_lf,w_rf,w_rb]",
        flush=True,
    )
    for case in case_summaries:
        print(
            f"{case['commanded_wheel']} | {case['target_joint_id']} | {case['effort']: .1f} | "
            f"{case['max_abs_joint_vel']} | {case['final_joint_vel']}",
            flush=True,
        )


def main() -> None:
    env = None
    exit_reason = "exception exit"
    wheel_collision_restore_attrs: list[tuple[object, object]] = []
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.init_height)

    print("[TRACE] before gym.make", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[TRACE] after gym.make", flush=True)

    try:
        print("[TRACE] before initial env.reset", flush=True)
        env.reset()
        print("[TRACE] after initial env.reset", flush=True)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        stage = unwrapped.sim.stage
        env_origins = unwrapped.scene.env_origins

        if args_cli.env_id < 0 or args_cli.env_id >= unwrapped.num_envs:
            raise ValueError(f"env_id must be in [0, {unwrapped.num_envs - 1}], got {args_cli.env_id}.")

        wheel_joint_ids, wheel_joint_names = robot.find_joints(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
        wheel_joint_ids = list(wheel_joint_ids)
        wheel_joint_names = list(wheel_joint_names)
        single_wheel_efforts = [100.0, -100.0]

        print(
            f"[TRACE] static config "
            f"len(wheel_joint_names)={len(wheel_joint_names)} "
            f"wheel_joint_ids={wheel_joint_ids} "
            f"effort_values={single_wheel_efforts} "
            f"steps_per_effort={args_cli.steps_per_effort} "
            f"print_every={args_cli.print_every}",
            flush=True,
        )
        assert len(wheel_joint_names) == 4, f"Expected 4 wheel joints, got {len(wheel_joint_names)}: {wheel_joint_names}"
        assert args_cli.steps_per_effort > 0, f"steps_per_effort must be > 0, got {args_cli.steps_per_effort}"

        print(f"[INFO]: task = {args_cli.task}", flush=True)
        print(f"[INFO]: testing env_id = {args_cli.env_id}", flush=True)
        print(f"[INFO]: suspended init height = {args_cli.init_height}", flush=True)
        print(f"[INFO]: inspect_only = {args_cli.inspect_only}", flush=True)
        print("[INFO]: sim gravity forced to (0, 0, 0) for this test.", flush=True)
        print(f"[INFO]: pin_base = {args_cli.pin_base}", flush=True)
        print(f"[INFO]: disable_wheel_collisions = {args_cli.disable_wheel_collisions}", flush=True)
        print(f"[INFO]: wheel joints = {wheel_joint_names}", flush=True)
        print(f"[INFO]: wheel joint ids = {wheel_joint_ids}", flush=True)

        missing_wheels = [name for name in ["w_lb", "w_lf", "w_rf", "w_rb"] if name not in robot.data.joint_names]
        if missing_wheels:
            print(f"WARNING: Wheel joints missing from articulation DOFs: {missing_wheels}", flush=True)

        _print_joint_inventory(robot, args_cli.env_id, wheel_joint_names, wheel_joint_ids)
        _print_wheel_usd_info(stage, wheel_joint_names)
        _print_wheel_structure_comparison(
            robot=robot,
            stage=stage,
            env_id=args_cli.env_id,
            wheel_body_names=wheel_joint_names,
            wheel_joint_names=wheel_joint_names,
        )

        pinned_root_pose = None
        pinned_root_velocity = None
        if args_cli.pin_base:
            pinned_root_pose, pinned_root_velocity = _build_pinned_root_state(
                robot=robot,
                env_id=args_cli.env_id,
                env_origins=env_origins,
                init_height=args_cli.init_height,
            )

        if args_cli.disable_wheel_collisions:
            for wheel_name in wheel_joint_names:
                wheel_collision_restore_attrs.extend(_set_link_collision_enabled(stage, wheel_name, enabled=False))
            print("[INFO]: disabled wheel collisions for current diagnostic run.", flush=True)
            for wheel_name in wheel_joint_names:
                print(
                    f"  collision_enabled[{wheel_name}]={_get_link_collision_enabled(stage, wheel_name)}",
                    flush=True,
                )

        with torch.inference_mode():
            print("[TRACE] before settle loop", flush=True)
            for _ in range(args_cli.settle_steps):
                _run_direct_effort_step(
                    robot=robot,
                    unwrapped_env=unwrapped,
                    env_id=args_cli.env_id,
                    wheel_joint_ids=wheel_joint_ids,
                    commanded_targets=[0.0, 0.0, 0.0, 0.0],
                    pinned_root_pose=pinned_root_pose,
                    pinned_root_velocity=pinned_root_velocity,
                )
            print("[TRACE] after settle loop", flush=True)
            if args_cli.inspect_only:
                print("[TRACE] inspect_only=True, skipping run_one_wheel_tests by explicit request.", flush=True)
            else:
                print("[TRACE] before run_one_wheel_tests", flush=True)
                case_summaries = _run_one_wheel_tests(
                    env=env,
                    unwrapped=unwrapped,
                    robot=robot,
                    wheel_joint_names=wheel_joint_names,
                    wheel_joint_ids=wheel_joint_ids,
                    pinned_root_pose=pinned_root_pose,
                    pinned_root_velocity=pinned_root_velocity,
                )
                _print_case_summary_table(case_summaries=case_summaries, wheel_joint_names=wheel_joint_names)
                print("[TRACE] after run_one_wheel_tests", flush=True)
        exit_reason = "normal exit"
        print(f"[TRACE] main completed with {exit_reason}", flush=True)
    except Exception:
        print("[TRACE] exception in main; printing traceback", flush=True)
        traceback.print_exc()
        print(f"[TRACE] main completed with {exit_reason}", flush=True)
        raise
    finally:
        if wheel_collision_restore_attrs:
            _restore_collision_attrs(wheel_collision_restore_attrs)
        print(f"[TRACE] finally: about to close env with status={exit_reason}", flush=True)
        if env is not None:
            env.close()
            print("[TRACE] finally: env.close completed", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
