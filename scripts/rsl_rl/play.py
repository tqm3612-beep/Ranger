# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_width", type=int, default=1920, help="Recorded viewport width in pixels.")
parser.add_argument("--video_height", type=int, default=1080, help="Recorded viewport height in pixels.")
parser.add_argument(
    "--show_sensor_debug",
    action="store_true",
    default=False,
    help="Keep lidar and ray-caster debug points visible in recorded videos.",
)
parser.add_argument(
    "--camera_mode",
    type=str,
    choices=("fixed", "follow", "overview", "overview_fixed"),
    default="follow",
    help=(
        "Recording camera mode. 'follow' tracks environment zero from behind; "
        "'overview' dynamically reframes all robots; 'overview_fixed' computes one overview pose "
        "from the initial robots and goals, then keeps it fixed."
    ),
)
parser.add_argument(
    "--overview_fixed_scale",
    type=float,
    default=None,
    help="Optional fixed overview camera scale. Larger values zoom farther out.",
)
parser.add_argument(
    "--overview_fixed_padding",
    type=float,
    default=2.0,
    help="Extra XY padding in meters when inferring the fixed overview framing.",
)
parser.add_argument(
    "--fixed_suspension_action",
    type=float,
    default=None,
    help="Evaluation-only override for all four suspension actions. Use 0.0 for nominal mid-stroke.",
)
parser.add_argument(
    "--stop_phase_suspension_action",
    type=float,
    default=None,
    help="Evaluation-only override applied to all four suspension actions after stop phase begins.",
)
parser.add_argument(
    "--suspension_action_scale",
    type=float,
    default=None,
    help="Evaluation-only multiplier for the four policy suspension actions. Use 1.0 for full authority.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop evaluation after this many policy steps even when video recording is disabled.",
)
parser.add_argument(
    "--evaluation_summary",
    action="store_true",
    default=False,
    help="Print termination counts and stopped-goal success rate at the end of play.",
)
parser.add_argument(
    "--evaluation_metrics_csv",
    type=str,
    default=None,
    help="Optional CSV path for episode-weighted full debug metrics collected during deterministic play.",
)
parser.add_argument(
    "--success_gate_trace_csv",
    type=str,
    default=None,
    help=(
        "Optional per-policy-step CSV trace for every stopped-goal success gate, including goal distance, "
        "base speed, yaw rate, roll/pitch, suspension tracking error, stop-phase state, and stable-step count."
    ),
)
parser.add_argument(
    "--video_subdir",
    type=str,
    default="play",
    help="Subdirectory below the checkpoint run's videos directory.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.fixed_suspension_action is not None and args_cli.stop_phase_suspension_action is not None:
    raise ValueError(
        "Use either --fixed_suspension_action or --stop_phase_suspension_action, not both."
    )
if args_cli.fixed_suspension_action is not None:
    if not -1.0 <= float(args_cli.fixed_suspension_action) <= 1.0:
        raise ValueError("--fixed_suspension_action must be within [-1, 1].")
    os.environ["RANGER_FIXED_SUSPENSION_ACTION"] = str(float(args_cli.fixed_suspension_action))
if args_cli.stop_phase_suspension_action is not None:
    if not -1.0 <= float(args_cli.stop_phase_suspension_action) <= 1.0:
        raise ValueError("--stop_phase_suspension_action must be within [-1, 1].")
    os.environ["RANGER_STOP_PHASE_SUSPENSION_ACTION"] = str(
        float(args_cli.stop_phase_suspension_action)
    )
if args_cli.suspension_action_scale is not None:
    if not 0.0 <= float(args_cli.suspension_action_scale) <= 1.0:
        raise ValueError("--suspension_action_scale must be within [0, 1].")
    os.environ["RANGER_SUSPENSION_ACTION_SCALE"] = str(float(args_cli.suspension_action_scale))
if args_cli.max_steps is not None and int(args_cli.max_steps) <= 0:
    raise ValueError("--max_steps must be positive when provided.")
# Always enable cameras to record video and pass the requested viewport resolution
# through to SimulationApp before AppLauncher is constructed.
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.width = max(int(args_cli.video_width), 320)
    args_cli.height = max(int(args_cli.video_height), 240)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import time
import torch

from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    def get_published_pretrained_checkpoint(*args, **kwargs):
        return None

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Ranger.tasks  # noqa: F401
from Ranger.tasks.manager_based.ranger import mdp as ranger_mdp
from Ranger.tasks.manager_based.ranger.agents import RangerTerrainActorCritic

rsl_on_policy_runner.RangerTerrainActorCritic = RangerTerrainActorCritic


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    if args_cli.video and not args_cli.show_sensor_debug:
        for sensor_name in ("mid360_lidar", "avia_lidar", "d435i_camera"):
            sensor_cfg = getattr(env_cfg.scene, sensor_name, None)
            if sensor_cfg is not None and hasattr(sensor_cfg, "debug_vis"):
                sensor_cfg.debug_vis = False

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", os.path.basename(args_cli.video_subdir)),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    obs_normalizer = getattr(ppo_runner, "obs_normalizer", None)
    if obs_normalizer is None:
        obs_normalizers = getattr(ppo_runner, "obs_normalizers", None)
        if isinstance(obs_normalizers, dict):
            obs_normalizer = obs_normalizers.get("policy", None)
        else:
            obs_normalizer = obs_normalizers
    try:
        export_policy_as_jit(policy_nn, obs_normalizer, path=export_model_dir, filename="policy.pt")
    except Exception as e:
        print(f"[WARN] Failed to export JIT policy, continue play without export: {e}")
    try:
        export_policy_as_onnx(policy_nn, normalizer=obs_normalizer, path=export_model_dir, filename="policy.onnx")
    except Exception as e:
        print(f"[WARN] Failed to export ONNX policy, continue play without export: {e}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs_result = env.get_observations()
    obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result
    timestep = 0
    termination_counts: dict[str, int] = {
        str(name): 0 for name in getattr(env.unwrapped.termination_manager, "active_terms", ())
    }
    completed_episodes = 0
    completed_episode_steps_sum = 0
    success_episode_steps_sum = 0
    success_stop_entry_steps_sum = 0
    success_stop_phase_steps_sum = 0
    success_with_stop_entry_count = 0
    evaluation_metric_weighted_sums: dict[str, float] = {}
    evaluation_metric_weights: dict[str, int] = {}
    episode_age_steps = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
    stop_entry_age_steps = torch.full(
        (env.unwrapped.num_envs,), -1, dtype=torch.long, device=env.unwrapped.device
    )
    episode_ids = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)

    success_gate_trace_file = None
    success_gate_trace_writer = None
    success_gate_trace_fields = [
        "policy_step",
        "env_id",
        "episode_id",
        "episode_step_before_action",
        "goal_side",
        "goal_distance_m",
        "heading_error_rad",
        "stop_phase_enter_distance_m",
        "success_distance_m",
        "distance_lt_stop_enter",
        "distance_ok",
        "velocity_toward_goal_mps",
        "velocity_tangential_to_goal_mps",
        "base_xy_speed_mps",
        "max_xy_speed_mps",
        "xy_speed_ok",
        "yaw_rate_signed_radps",
        "yaw_rate_abs_radps",
        "max_yaw_rate_radps",
        "yaw_rate_ok",
        "roll_abs_rad",
        "max_roll_rad",
        "roll_ok",
        "pitch_abs_rad",
        "max_pitch_rad",
        "pitch_ok",
        "stroke_tracking_error_max_m",
        "max_stroke_tracking_error_m",
        "stroke_tracking_ok",
        "posture_ok",
        "stop_phase_active",
        "stop_phase_first_observed_before_action",
        "stable_steps_before_action",
        "required_hold_steps",
        "stable_fraction_before_action",
        "settled_now_before_action",
        "success_ready_before_action",
        "failed_gates_before_action",
        "policy_wheel_raw_lb",
        "policy_wheel_raw_lf",
        "policy_wheel_raw_rf",
        "policy_wheel_raw_rb",
        "current_wheel_target_semantic_lb_radps",
        "current_wheel_target_semantic_lf_radps",
        "current_wheel_target_semantic_rf_radps",
        "current_wheel_target_semantic_rb_radps",
        "termination_any_after_step",
        "stopped_goal_reached_after_step",
        "time_out_after_step",
        "termination_terms_after_step",
    ]
    if args_cli.success_gate_trace_csv is not None:
        trace_path = os.path.abspath(os.path.expanduser(args_cli.success_gate_trace_csv))
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        success_gate_trace_file = open(trace_path, "w", newline="", encoding="utf-8")
        success_gate_trace_writer = csv.DictWriter(
            success_gate_trace_file,
            fieldnames=success_gate_trace_fields,
            extrasaction="ignore",
        )
        success_gate_trace_writer.writeheader()
        print(f"[SuccessGateTrace] CSV writing to: {trace_path}")

    camera_eye_state: torch.Tensor | None = None
    camera_target_state: torch.Tensor | None = None

    def update_recording_camera() -> None:
        """Track environment zero during video recording without affecting policy observations."""

        nonlocal camera_eye_state, camera_target_state
        if not args_cli.video or args_cli.camera_mode == "fixed":
            return
        if args_cli.camera_mode == "overview_fixed" and camera_eye_state is not None:
            return
        robot = env.unwrapped.scene["robot"]
        root_positions = robot.data.root_pos_w.detach().cpu()
        root_pos = root_positions[0]
        if args_cli.camera_mode in {"overview", "overview_fixed"}:
            framing_positions = root_positions
            if args_cli.camera_mode == "overview_fixed":
                goal_positions = getattr(env.unwrapped, "_ranger_short_goal_pos_w", None)
                if isinstance(goal_positions, torch.Tensor) and goal_positions.shape[0] == root_positions.shape[0]:
                    framing_positions = torch.cat((root_positions, goal_positions.detach().cpu()), dim=0)
            xy_min = framing_positions[:, :2].min(dim=0).values
            xy_max = framing_positions[:, :2].max(dim=0).values
            center_xy = 0.5 * (xy_min + xy_max)
            horizontal_extent = float(torch.max(xy_max - xy_min).item())
            if args_cli.camera_mode == "overview_fixed":
                horizontal_extent += 2.0 * max(float(args_cli.overview_fixed_padding), 0.0)
                if args_cli.overview_fixed_scale is None:
                    camera_scale = max(9.0, 1.35 * horizontal_extent + 6.0)
                else:
                    camera_scale = max(float(args_cli.overview_fixed_scale), 1.0)
            else:
                # Keep less empty margin than the original fleet view so each robot occupies
                # more pixels while still scaling out when the environments separate.
                camera_scale = max(9.0, 1.35 * horizontal_extent + 6.0)
            desired_target = torch.tensor([float(center_xy[0]), float(center_xy[1]), 0.7])
            # Oblique fleet view: keep all environments framed while exposing the wheel sidewalls
            # well enough to distinguish front/rear wheel rotation direction.
            desired_eye = desired_target + torch.tensor(
                [0.55 * camera_scale, -0.80 * camera_scale, 0.65 * camera_scale]
            )
        else:
            _, _, yaw_tensor = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
            yaw = float(yaw_tensor[0].item())
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            offset_body = (-4.5, -3.5)
            lookahead_body = (1.0, 0.0)
            offset_world = torch.tensor(
                [
                    cos_yaw * offset_body[0] - sin_yaw * offset_body[1],
                    sin_yaw * offset_body[0] + cos_yaw * offset_body[1],
                    2.8,
                ]
            )
            lookahead_world = torch.tensor(
                [
                    cos_yaw * lookahead_body[0] - sin_yaw * lookahead_body[1],
                    sin_yaw * lookahead_body[0] + cos_yaw * lookahead_body[1],
                    0.7,
                ]
            )
            desired_eye = root_pos + offset_world
            desired_target = root_pos + lookahead_world

        smoothing = 0.12
        if camera_eye_state is None:
            camera_eye_state = desired_eye
            camera_target_state = desired_target
        else:
            camera_eye_state = torch.lerp(camera_eye_state, desired_eye, smoothing)
            camera_target_state = torch.lerp(camera_target_state, desired_target, smoothing)
        env.unwrapped.sim.set_camera_view(camera_eye_state.tolist(), camera_target_state.tolist())

    def collect_success_gate_rows(
        actions: torch.Tensor,
        stop_phase_active: torch.Tensor,
        first_observed_stop: torch.Tensor,
    ) -> list[dict[str, object]]:
        """Collect every stopped-goal success gate before the current env.step()."""

        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        target_vec_b, goal_distance, heading_error = ranger_mdp.short_goal_target_body(unwrapped)
        base_xy_velocity = robot.data.root_lin_vel_b[:, :2]
        base_xy_speed = torch.linalg.vector_norm(base_xy_velocity, dim=1)
        target_dir_b = target_vec_b[:, :2] / torch.clamp(goal_distance.unsqueeze(1), min=1.0e-6)
        velocity_toward_goal = torch.sum(base_xy_velocity * target_dir_b, dim=1)
        velocity_tangential_to_goal = (
            target_dir_b[:, 0] * base_xy_velocity[:, 1]
            - target_dir_b[:, 1] * base_xy_velocity[:, 0]
        )
        yaw_rate_signed = robot.data.root_ang_vel_b[:, 2]
        yaw_rate_abs = torch.abs(yaw_rate_signed)
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        roll_abs = torch.abs(roll)
        pitch_abs = torch.abs(pitch)

        leg_term = unwrapped.action_manager.get_term("leg_hydraulic")
        stroke_error = torch.abs(leg_term.stroke_desired - leg_term.stroke_actual)
        stroke_tracking_error_max = torch.max(stroke_error, dim=1).values

        wheel_term = unwrapped.action_manager.get_term("wheel_motor_csv")
        wheel_forward_sign = getattr(unwrapped, "_wheel_forward_sign", None)
        if not isinstance(wheel_forward_sign, torch.Tensor):
            wheel_forward_sign = torch.ones(4, device=unwrapped.device, dtype=wheel_term.velocity_target.dtype)
        wheel_forward_sign = wheel_forward_sign.to(
            device=unwrapped.device,
            dtype=wheel_term.velocity_target.dtype,
        )
        current_wheel_target_semantic = wheel_term.velocity_target * wheel_forward_sign

        stop_enter_distance = float(getattr(unwrapped, "_short_goal_stop_phase_enter_distance", 0.30))
        success_distance = float(getattr(unwrapped, "_short_goal_stop_success_distance", 0.50))
        max_xy_speed = float(getattr(unwrapped, "_short_goal_stop_max_xy_speed", 0.15))
        max_yaw_rate = float(getattr(unwrapped, "_short_goal_stop_max_yaw_rate", 0.20))
        max_roll = float(getattr(unwrapped, "_short_goal_stop_max_roll", 0.035))
        max_pitch = float(getattr(unwrapped, "_short_goal_stop_max_pitch", 0.035))
        max_stroke_tracking_error = float(
            getattr(unwrapped, "_short_goal_stop_max_stroke_tracking_error", 0.02)
        )
        required_hold_steps = int(getattr(unwrapped, "_short_goal_stop_required_hold_steps", 60))
        stable_steps = getattr(unwrapped, "_short_goal_stop_phase_stable_steps", None)
        if not isinstance(stable_steps, torch.Tensor):
            stable_steps = torch.zeros(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)

        distance_lt_stop_enter = goal_distance < stop_enter_distance
        distance_ok = goal_distance < success_distance
        xy_speed_ok = base_xy_speed < max_xy_speed
        yaw_rate_ok = yaw_rate_abs < max_yaw_rate
        roll_ok = roll_abs <= max_roll
        pitch_ok = pitch_abs <= max_pitch
        stroke_tracking_ok = stroke_tracking_error_max <= max_stroke_tracking_error
        posture_ok = roll_ok & pitch_ok & stroke_tracking_ok
        settled_now = stop_phase_active & distance_ok & xy_speed_ok & yaw_rate_ok & posture_ok
        success_ready = settled_now & (stable_steps >= required_hold_steps)

        initial_side_sign = getattr(unwrapped, "_short_goal_initial_side_sign", None)
        if not isinstance(initial_side_sign, torch.Tensor):
            initial_side_sign = torch.sign(heading_error)

        rows: list[dict[str, object]] = []
        wheel_names = ("lb", "lf", "rf", "rb")
        for env_id in range(unwrapped.num_envs):
            side_sign = float(initial_side_sign[env_id].item())
            goal_side = "left" if side_sign > 0.0 else ("right" if side_sign < 0.0 else "straight")
            failed_gates: list[str] = []
            if not bool(stop_phase_active[env_id].item()):
                failed_gates.append("stop_phase")
            if not bool(distance_ok[env_id].item()):
                failed_gates.append("distance")
            if not bool(xy_speed_ok[env_id].item()):
                failed_gates.append("xy_speed")
            if not bool(yaw_rate_ok[env_id].item()):
                failed_gates.append("yaw_rate")
            if not bool(roll_ok[env_id].item()):
                failed_gates.append("roll")
            if not bool(pitch_ok[env_id].item()):
                failed_gates.append("pitch")
            if not bool(stroke_tracking_ok[env_id].item()):
                failed_gates.append("stroke_tracking")
            if bool(settled_now[env_id].item()) and int(stable_steps[env_id].item()) < required_hold_steps:
                failed_gates.append("hold_steps")
            row: dict[str, object] = {
                "policy_step": timestep,
                "env_id": env_id,
                "episode_id": int(episode_ids[env_id].item()),
                "episode_step_before_action": int(episode_age_steps[env_id].item()),
                "goal_side": goal_side,
                "goal_distance_m": float(goal_distance[env_id].item()),
                "heading_error_rad": float(heading_error[env_id].item()),
                "stop_phase_enter_distance_m": stop_enter_distance,
                "success_distance_m": success_distance,
                "distance_lt_stop_enter": int(distance_lt_stop_enter[env_id].item()),
                "distance_ok": int(distance_ok[env_id].item()),
                "velocity_toward_goal_mps": float(velocity_toward_goal[env_id].item()),
                "velocity_tangential_to_goal_mps": float(velocity_tangential_to_goal[env_id].item()),
                "base_xy_speed_mps": float(base_xy_speed[env_id].item()),
                "max_xy_speed_mps": max_xy_speed,
                "xy_speed_ok": int(xy_speed_ok[env_id].item()),
                "yaw_rate_signed_radps": float(yaw_rate_signed[env_id].item()),
                "yaw_rate_abs_radps": float(yaw_rate_abs[env_id].item()),
                "max_yaw_rate_radps": max_yaw_rate,
                "yaw_rate_ok": int(yaw_rate_ok[env_id].item()),
                "roll_abs_rad": float(roll_abs[env_id].item()),
                "max_roll_rad": max_roll,
                "roll_ok": int(roll_ok[env_id].item()),
                "pitch_abs_rad": float(pitch_abs[env_id].item()),
                "max_pitch_rad": max_pitch,
                "pitch_ok": int(pitch_ok[env_id].item()),
                "stroke_tracking_error_max_m": float(stroke_tracking_error_max[env_id].item()),
                "max_stroke_tracking_error_m": max_stroke_tracking_error,
                "stroke_tracking_ok": int(stroke_tracking_ok[env_id].item()),
                "posture_ok": int(posture_ok[env_id].item()),
                "stop_phase_active": int(stop_phase_active[env_id].item()),
                "stop_phase_first_observed_before_action": int(first_observed_stop[env_id].item()),
                "stable_steps_before_action": int(stable_steps[env_id].item()),
                "required_hold_steps": required_hold_steps,
                "stable_fraction_before_action": min(
                    float(stable_steps[env_id].item()) / max(float(required_hold_steps), 1.0),
                    1.0,
                ),
                "settled_now_before_action": int(settled_now[env_id].item()),
                "success_ready_before_action": int(success_ready[env_id].item()),
                "failed_gates_before_action": ";".join(failed_gates),
                "termination_any_after_step": 0,
                "stopped_goal_reached_after_step": 0,
                "time_out_after_step": 0,
                "termination_terms_after_step": "",
            }
            for wheel_idx, wheel_name in enumerate(wheel_names):
                row[f"policy_wheel_raw_{wheel_name}"] = float(actions[env_id, wheel_idx + 4].item())
                row[f"current_wheel_target_semantic_{wheel_name}_radps"] = float(
                    current_wheel_target_semantic[env_id, wheel_idx].item()
                )
            rows.append(row)
        return rows

    update_recording_camera()

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            collect_episode_state = args_cli.evaluation_summary or success_gate_trace_writer is not None
            stop_phase_active = torch.zeros(
                env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device
            )
            first_observed_stop = torch.zeros_like(stop_phase_active)
            if collect_episode_state:
                # Sample the latched stop-phase state before env.step(). Successful
                # environments are reset inside env.step(), so post-step sampling loses it.
                stop_phase_state = getattr(env.unwrapped, "_short_goal_stop_phase_active", None)
                if isinstance(stop_phase_state, torch.Tensor):
                    stop_phase_active = stop_phase_state.to(dtype=torch.bool)
                newly_entered_stop = stop_phase_active & (stop_entry_age_steps < 0)
                first_observed_stop = newly_entered_stop.clone()
                stop_entry_age_steps[newly_entered_stop] = episode_age_steps[newly_entered_stop]

            # agent stepping
            actions = policy(obs)
            success_gate_rows = None
            if success_gate_trace_writer is not None:
                success_gate_rows = collect_success_gate_rows(
                    actions,
                    stop_phase_active,
                    first_observed_stop,
                )

            # env stepping
            step_result = env.step(actions)
            if len(step_result) == 5:
                obs, _, _, _, step_info = step_result
            else:
                obs, _, _, step_info = step_result

            if collect_episode_state:
                episode_age_steps += 1
                term_manager = env.unwrapped.termination_manager
                term_any = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
                success_mask = torch.zeros_like(term_any)
                timeout_mask = torch.zeros_like(term_any)
                termination_names_by_env: list[list[str]] = [
                    [] for _ in range(env.unwrapped.num_envs)
                ]
                for term_idx, term_name in enumerate(getattr(term_manager, "active_terms", ())):
                    term_done = term_manager._term_dones[:, term_idx]
                    term_name_str = str(term_name)
                    if args_cli.evaluation_summary:
                        termination_counts[term_name_str] = termination_counts.get(term_name_str, 0) + int(
                            term_done.sum().item()
                        )
                    for env_id in torch.nonzero(term_done, as_tuple=False).flatten().tolist():
                        termination_names_by_env[int(env_id)].append(term_name_str)
                    if term_name_str == "stopped_goal_reached":
                        success_mask |= term_done
                    if term_name_str == "time_out":
                        timeout_mask |= term_done
                    term_any |= term_done

                if success_gate_rows is not None:
                    for env_id, row in enumerate(success_gate_rows):
                        row["termination_any_after_step"] = int(term_any[env_id].item())
                        row["stopped_goal_reached_after_step"] = int(success_mask[env_id].item())
                        row["time_out_after_step"] = int(timeout_mask[env_id].item())
                        row["termination_terms_after_step"] = ";".join(termination_names_by_env[env_id])
                    success_gate_trace_writer.writerows(success_gate_rows)
                    if timestep % 60 == 0:
                        success_gate_trace_file.flush()

                if args_cli.evaluation_summary:
                    completed_episode_steps_sum += int(episode_age_steps[term_any].sum().item())
                    success_episode_steps_sum += int(episode_age_steps[success_mask].sum().item())
                    valid_success_stop = success_mask & (stop_entry_age_steps >= 0)
                    success_stop_entry_steps_sum += int(stop_entry_age_steps[valid_success_stop].sum().item())
                    success_stop_phase_steps_sum += int(
                        (episode_age_steps[valid_success_stop] - stop_entry_age_steps[valid_success_stop]).sum().item()
                    )
                    success_with_stop_entry_count += int(valid_success_stop.sum().item())
                    completed_this_step = int(term_any.sum().item())
                    completed_episodes += completed_this_step
                    if completed_this_step > 0:
                        full_log = step_info.get("full_log", {}) if isinstance(step_info, dict) else {}
                        if not full_log:
                            env_extras = getattr(env.unwrapped, "extras", {})
                            full_log = env_extras.get("full_log", {}) if isinstance(env_extras, dict) else {}
                        if isinstance(full_log, dict):
                            for metric_name, metric_value in full_log.items():
                                if not str(metric_name).startswith("Metrics/short_goal/"):
                                    continue
                                try:
                                    scalar_value = float(metric_value)
                                except (TypeError, ValueError):
                                    continue
                                evaluation_metric_weighted_sums[metric_name] = (
                                    evaluation_metric_weighted_sums.get(metric_name, 0.0)
                                    + scalar_value * completed_this_step
                                )
                                evaluation_metric_weights[metric_name] = (
                                    evaluation_metric_weights.get(metric_name, 0) + completed_this_step
                                )
                episode_ids[term_any] += 1
                episode_age_steps[term_any] = 0
                stop_entry_age_steps[term_any] = -1
        update_recording_camera()
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_length:
                break
        if args_cli.max_steps is not None and timestep >= int(args_cli.max_steps):
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if success_gate_trace_file is not None:
        success_gate_trace_file.flush()
        success_gate_trace_file.close()

    if args_cli.evaluation_summary:
        success_count = int(termination_counts.get("stopped_goal_reached", 0))
        timeout_count = int(termination_counts.get("time_out", 0))
        other_failure_count = max(completed_episodes - success_count - timeout_count, 0)
        success_rate = success_count / max(completed_episodes, 1)
        mean_completed_steps = completed_episode_steps_sum / max(completed_episodes, 1)
        mean_success_steps = success_episode_steps_sum / max(success_count, 1)
        mean_stop_entry_steps = success_stop_entry_steps_sum / max(success_with_stop_entry_count, 1)
        mean_stop_phase_steps = success_stop_phase_steps_sum / max(success_with_stop_entry_count, 1)
        print("[EvaluationSummary]")
        print(f"  policy_steps: {timestep}")
        print(f"  completed_episodes: {completed_episodes}")
        print(f"  stopped_goal_reached: {success_count}")
        print(f"  time_out: {timeout_count}")
        print(f"  other_failures: {other_failure_count}")
        print(f"  stopped_goal_success_rate: {success_rate:.6f}")
        print(f"  mean_completed_episode_steps: {mean_completed_steps:.3f}")
        print(f"  mean_success_episode_steps: {mean_success_steps:.3f}")
        print(f"  success_with_stop_entry_count: {success_with_stop_entry_count}")
        print(f"  mean_steps_to_stop_phase: {mean_stop_entry_steps:.3f}")
        print(f"  mean_steps_stop_phase_to_success: {mean_stop_phase_steps:.3f}")
        print(f"  termination_counts: {termination_counts}")

        evaluation_metric_means = {
            name: evaluation_metric_weighted_sums[name] / max(evaluation_metric_weights[name], 1)
            for name in evaluation_metric_weighted_sums
        }
        if evaluation_metric_means:
            print("[EvaluationMetrics]")
            for metric_name in sorted(evaluation_metric_means):
                print(
                    f"  {metric_name}: {evaluation_metric_means[metric_name]:.8f} "
                    f"(episodes={evaluation_metric_weights[metric_name]})"
                )
        else:
            print("[EvaluationMetrics] No completed-episode full_log metrics were collected.")

        if args_cli.evaluation_metrics_csv is not None:
            metrics_csv_path = os.path.abspath(os.path.expanduser(args_cli.evaluation_metrics_csv))
            metrics_csv_dir = os.path.dirname(metrics_csv_path)
            if metrics_csv_dir:
                os.makedirs(metrics_csv_dir, exist_ok=True)
            with open(metrics_csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(("metric", "value", "episode_weight"))
                for metric_name in sorted(evaluation_metric_means):
                    writer.writerow(
                        (
                            metric_name,
                            f"{evaluation_metric_means[metric_name]:.10g}",
                            evaluation_metric_weights[metric_name],
                        )
                    )
            print(f"[EvaluationMetrics] CSV written to: {metrics_csv_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
