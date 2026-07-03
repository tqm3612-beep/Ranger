from __future__ import annotations

import os

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

from . import mdp


class RangerForwardDebugEnv(ManagerBasedRLEnv):
    """Forward-stage environment with episode-level debug metrics for locomotion diagnosis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        robot = self.scene["robot"]
        self._leg_joint_ids, _ = robot.find_joints(
            ["g_lb", "g_lf", "g_rf", "g_rb"],
            preserve_order=True,
        )
        self._wheel_joint_ids, _ = robot.find_joints(
            ["w_lb", "w_lf", "w_rf", "w_rb"],
            preserve_order=True,
        )
        self._wheel_body_ids, self._wheel_body_names = robot.find_bodies(
            ["w_lb", "w_lf", "w_rf", "w_rb"],
            preserve_order=True,
        )
        contact_sensor = self.scene.sensors["wheel_contact_forces"]
        self._wheel_contact_body_ids, _ = contact_sensor.find_bodies(
            self._wheel_body_names,
            preserve_order=True,
        )
        leg_action_term = self.action_manager.get_term("leg_hydraulic")
        self._prev_stroke_command = leg_action_term.stroke_command.clone()
        self._enable_forward_debug_metrics = os.environ.get("RANGER_DEBUG_METRICS", "0") == "1"

        # Wheel joint forward sign.
        # Joint order: [w_lb, w_lf, w_rf, w_rb]
        # Semantic positive wheel velocity means robot forward +x motion.
        self._wheel_forward_sign = torch.tensor(
            [-1.0, -1.0, 1.0, 1.0],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self._forward_debug_metric_names = (
            "base_lin_vel_x",
            "flat_weight",
            "local_height_range",
            "local_height_std",
            "local_height_max_abs",
            "terrain_level",
            "terrain_type",
            "local_ground_height_b",
            "local_ground_height_w",
            "base_clearance",
            "clearance_error",
            "root_height_w",
            "base_pitch_deg",
            "semantic_wheel_velocity_target_mean",
            "semantic_wheel_joint_vel_mean",
            "wheel_action_abs_mean",
            "roll_deg",
            "pitch_deg",
            "abs_roll_deg",
            "abs_pitch_deg",
            "stroke_min",
            "stroke_max",
            "stroke_mean",
            "stroke_abs_mean",
            "stroke_rate_mean",
            "stroke_over_0_5_ratio",
            "stroke_over_0_55_ratio",
            "actual_stroke_min",
            "actual_stroke_max",
            "actual_stroke_mean",
            "actual_stroke_over_0_5_ratio",
            "actual_stroke_over_0_55_ratio",
            "front_stroke_mean",
            "rear_stroke_mean",
            "target_front_rear_stroke_diff",
            "actual_front_rear_stroke_diff",
            "joint_pos_g_min",
            "joint_pos_g_max",
            "joint_pos_g_abs_max",
            "wheel_z_b_min",
            "wheel_z_b_max",
            "front_rear_stroke_diff",
            "left_right_stroke_diff",
            "front_rear_wheel_z_diff",
            "left_right_wheel_z_diff",
            "contact_force_front_mean",
            "contact_force_rear_mean",
            "contact_force_left_mean",
            "contact_force_right_mean",
        )
    
        self._forward_debug_metric_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for name in self._forward_debug_metric_names
        }
        self._forward_debug_metric_counts = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        self._eval_metric_names = (
            "roll_rms_deg",
            "pitch_rms_deg",
            "roll_max_abs_deg",
            "pitch_max_abs_deg",
            "base_vertical_velocity_rms",
            "base_roll_pitch_ang_vel_rms",
            "average_forward_speed",
            "velocity_tracking_error_rms",
            "joint_limit_margin_penalty_mean",
            "joint_limit_margin_count_mean",
            "episode_length",
            "success_rate",
        )
        self._eval_step_counts = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._eval_roll_sq_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._eval_pitch_sq_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._eval_roll_abs_max = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._eval_pitch_abs_max = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._eval_base_vertical_velocity_sq_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._eval_base_roll_pitch_ang_vel_sq_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._eval_forward_speed_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._eval_velocity_tracking_error_sq_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._eval_joint_limit_margin_penalty_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._eval_joint_limit_margin_count_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._eval_last_time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._eval_has_episode_result = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


    def _compute_forward_debug_metric_values(self) -> dict[str, torch.Tensor]:
        robot = self.scene["robot"]
        wheel_action_term = self.action_manager.get_term("wheel_motor_csv")
        leg_action_term = self.action_manager.get_term("leg_hydraulic")
        flat_base_clearance_params = self.cfg.rewards.flat_base_clearance.params
        flat_stroke_nominal_params = self.cfg.rewards.flat_stroke_nominal.params
        pitch_stroke_comp_params = self.cfg.rewards.base_pitch_stroke_compensation.params
        flat_stroke_asset_cfg = flat_stroke_nominal_params.get("asset_cfg", SceneEntityCfg("robot"))
        flat_base_asset_cfg = flat_base_clearance_params.get("asset_cfg", SceneEntityCfg("robot"))
        flat_stroke_asset_name = flat_stroke_asset_cfg.name
        flat_base_asset_name = flat_base_asset_cfg.name

        base_lin_vel_x = robot.data.root_lin_vel_b[:, 0]
        root_height_w = robot.data.root_pos_w[:, 2]
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        roll_deg = torch.rad2deg(roll)
        pitch_deg = torch.rad2deg(pitch)
        base_pitch_deg = pitch_deg
        flat_weight = mdp.get_flat_terrain_weight(
            env=self,
            sensor_names=flat_stroke_nominal_params["sensor_names"],
            asset_name=flat_stroke_asset_name,
            x_range=flat_stroke_nominal_params["x_range"],
            y_range=flat_stroke_nominal_params["y_range"],
            resolution=flat_stroke_nominal_params["resolution"],
            step_threshold=flat_stroke_nominal_params["step_threshold"],
            height_reference_x_range=flat_stroke_nominal_params["height_reference_x_range"],
            height_reference_y_range=flat_stroke_nominal_params["height_reference_y_range"],
            slope_normalization=flat_stroke_nominal_params["slope_normalization"],
            roughness_normalization=flat_stroke_nominal_params["roughness_normalization"],
            step_normalization=flat_stroke_nominal_params["step_normalization"],
            slope_weight=flat_stroke_nominal_params["slope_weight"],
            roughness_weight=flat_stroke_nominal_params["roughness_weight"],
            step_weight=flat_stroke_nominal_params["step_weight"],
            height_range_weight=flat_stroke_nominal_params["height_range_weight"],
            flatness_gain=flat_stroke_nominal_params["flatness_gain"],
        )
        local_map_layers = mdp.local_navigation_map_layers(
            env=self,
            sensor_names=flat_stroke_nominal_params["sensor_names"],
            asset_name=flat_stroke_asset_name,
            x_range=flat_stroke_nominal_params["x_range"],
            y_range=flat_stroke_nominal_params["y_range"],
            resolution=flat_stroke_nominal_params["resolution"],
            step_threshold=flat_stroke_nominal_params["step_threshold"],
            height_reference_x_range=flat_stroke_nominal_params["height_reference_x_range"],
            height_reference_y_range=flat_stroke_nominal_params["height_reference_y_range"],
            slope_normalization=flat_stroke_nominal_params["slope_normalization"],
            roughness_normalization=flat_stroke_nominal_params["roughness_normalization"],
            step_normalization=flat_stroke_nominal_params["step_normalization"],
            apply_noise=False,
            height_noise_std=0.0,
            risk_noise_std=0.0,
            valid_dropout_prob=0.0,
            slope_weight=flat_stroke_nominal_params["slope_weight"],
            roughness_weight=flat_stroke_nominal_params["roughness_weight"],
            step_weight=flat_stroke_nominal_params["step_weight"],
            unknown_penalty=1.0,
            use_neutral_map=False,
        )
        local_height = local_map_layers["height"]
        local_valid = local_map_layers["valid_mask"] > 0.5
        valid_count = torch.clamp(local_valid.sum(dim=(1, 2)).to(torch.float32), min=1.0)
        local_height_for_max = torch.where(local_valid, local_height, torch.full_like(local_height, -1.0e6))
        local_height_for_min = torch.where(local_valid, local_height, torch.full_like(local_height, 1.0e6))
        local_height_max = local_height_for_max.amax(dim=(1, 2))
        local_height_min = local_height_for_min.amin(dim=(1, 2))
        local_height_mean = (local_height * local_valid).sum(dim=(1, 2)) / valid_count
        local_height_var = (
            torch.square(torch.where(local_valid, local_height - local_height_mean.view(-1, 1, 1), torch.zeros_like(local_height)))
            .sum(dim=(1, 2))
            / valid_count
        )
        local_height_range = torch.clamp(local_height_max - local_height_min, min=0.0)
        local_height_std = torch.sqrt(torch.clamp(local_height_var, min=0.0))
        local_height_max_abs = torch.where(local_valid, torch.abs(local_height), torch.zeros_like(local_height)).amax(dim=(1, 2))
        terrain = self.scene.terrain
        terrain_level = (
            terrain.terrain_levels.to(torch.float32)
            if terrain is not None and hasattr(terrain, "terrain_levels")
            else torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        )
        terrain_type = (
            terrain.terrain_types.to(torch.float32)
            if terrain is not None and hasattr(terrain, "terrain_types")
            else torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        )
        local_ground_height_b = mdp.get_local_ground_height(
            env=self,
            sensor_names=flat_base_clearance_params["sensor_names"],
            asset_name=flat_base_asset_name,
            x_range=flat_base_clearance_params["x_range"],
            y_range=flat_base_clearance_params["y_range"],
            resolution=flat_base_clearance_params["resolution"],
            height_reference_x_range=flat_base_clearance_params["height_reference_x_range"],
            height_reference_y_range=flat_base_clearance_params["height_reference_y_range"],
        )
        local_ground_height_w = root_height_w + local_ground_height_b
        base_clearance = -local_ground_height_b
        clearance_error = base_clearance - flat_base_clearance_params["clearance_nominal"]

        wheel_velocity_target = wheel_action_term.velocity_target
        wheel_joint_vel = robot.data.joint_vel[:, self._wheel_joint_ids]
        leg_joint_pos = robot.data.joint_pos[:, self._leg_joint_ids]
        semantic_wheel_velocity_target = wheel_velocity_target * self._wheel_forward_sign
        semantic_wheel_joint_vel = wheel_joint_vel * self._wheel_forward_sign
        stroke_command = leg_action_term.stroke_command
        actual_stroke = leg_action_term.stroke_measured
        stroke_rate = torch.abs(stroke_command - self._prev_stroke_command) / max(self.step_dt, 1.0e-6)
        wheel_pos_w = robot.data.body_pos_w[:, self._wheel_body_ids, :]
        wheel_pos_w_rel = wheel_pos_w - robot.data.root_pos_w.unsqueeze(1)
        wheel_pos_b = quat_apply_inverse(
            robot.data.root_quat_w.unsqueeze(1).expand(-1, len(self._wheel_body_ids), -1).reshape(-1, 4),
            wheel_pos_w_rel.reshape(-1, 3),
        ).reshape(self.num_envs, len(self._wheel_body_ids), 3)
        wheel_contact_sensor = self.scene.sensors["wheel_contact_forces"]
        wheel_contact_force = torch.mean(
            torch.norm(
                wheel_contact_sensor.data.net_forces_w_history[:, :, self._wheel_contact_body_ids, :],
                dim=-1,
            ),
            dim=1,
        )

        # Semantic forward-direction mean.
        # Positive value means the wheels are commanded / rotating in the robot-forward direction.
        semantic_wheel_velocity_target_mean = (
            wheel_velocity_target * self._wheel_forward_sign
        ).mean(dim=1)

        semantic_wheel_joint_vel_mean = (
            wheel_joint_vel * self._wheel_forward_sign
        ).mean(dim=1)

        front_stroke_mean = stroke_command[:, [1, 2]].mean(dim=1)
        rear_stroke_mean = stroke_command[:, [0, 3]].mean(dim=1)
        target_front_rear_stroke_diff = torch.clamp(
            pitch_stroke_comp_params["k_pitch"] * pitch,
            min=-0.15,
            max=0.15,
        )
        actual_front_rear_stroke_diff = front_stroke_mean - rear_stroke_mean
        front_rear_stroke_diff = actual_front_rear_stroke_diff
        left_right_stroke_diff = stroke_command[:, :2].mean(dim=1) - stroke_command[:, 2:].mean(dim=1)
        stroke_min = torch.min(stroke_command, dim=1).values
        stroke_max = torch.max(stroke_command, dim=1).values
        stroke_mean = torch.mean(stroke_command, dim=1)
        stroke_abs_mean = torch.mean(torch.abs(stroke_command), dim=1)
        stroke_rate_mean = torch.mean(stroke_rate, dim=1)
        stroke_over_0_5_ratio = torch.mean((stroke_command > 0.5).to(torch.float32), dim=1)
        stroke_over_0_55_ratio = torch.mean((stroke_command > 0.55).to(torch.float32), dim=1)
        actual_stroke_min = torch.min(actual_stroke, dim=1).values
        actual_stroke_max = torch.max(actual_stroke, dim=1).values
        actual_stroke_mean = torch.mean(actual_stroke, dim=1)
        actual_stroke_over_0_5_ratio = torch.mean((actual_stroke > 0.5).to(torch.float32), dim=1)
        actual_stroke_over_0_55_ratio = torch.mean((actual_stroke > 0.55).to(torch.float32), dim=1)
        joint_pos_g_min = torch.min(leg_joint_pos, dim=1).values
        joint_pos_g_max = torch.max(leg_joint_pos, dim=1).values
        joint_pos_g_abs_max = torch.max(torch.abs(leg_joint_pos), dim=1).values
        wheel_z_b_min = torch.min(wheel_pos_b[:, :, 2], dim=1).values
        wheel_z_b_max = torch.max(wheel_pos_b[:, :, 2], dim=1).values
        front_rear_wheel_z_diff = wheel_pos_b[:, 1:3, 2].mean(dim=1) - wheel_pos_b[:, [0, 3], 2].mean(dim=1)
        left_right_wheel_z_diff = wheel_pos_b[:, :2, 2].mean(dim=1) - wheel_pos_b[:, 2:, 2].mean(dim=1)
        wheel_action_abs_mean = torch.mean(torch.abs(wheel_action_term.raw_actions), dim=1)
        contact_force_front_mean = wheel_contact_force[:, 1:3].mean(dim=1)
        contact_force_rear_mean = wheel_contact_force[:, [0, 3]].mean(dim=1)
        contact_force_left_mean = wheel_contact_force[:, :2].mean(dim=1)
        contact_force_right_mean = wheel_contact_force[:, 2:].mean(dim=1)
        self._prev_stroke_command.copy_(stroke_command)

        return {
            "base_lin_vel_x": base_lin_vel_x,
            "flat_weight": flat_weight,
            "local_height_range": local_height_range,
            "local_height_std": local_height_std,
            "local_height_max_abs": local_height_max_abs,
            "terrain_level": terrain_level,
            "terrain_type": terrain_type,
            "local_ground_height_b": local_ground_height_b,
            "local_ground_height_w": local_ground_height_w,
            "base_clearance": base_clearance,
            "clearance_error": clearance_error,
            "root_height_w": root_height_w,
            "base_pitch_deg": base_pitch_deg,
            "semantic_wheel_velocity_target_mean": semantic_wheel_velocity_target_mean,
            "semantic_wheel_joint_vel_mean": semantic_wheel_joint_vel_mean,
            "wheel_action_abs_mean": wheel_action_abs_mean,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "abs_roll_deg": torch.abs(roll_deg),
            "abs_pitch_deg": torch.abs(pitch_deg),
            "stroke_min": stroke_min,
            "stroke_max": stroke_max,
            "stroke_mean": stroke_mean,
            "stroke_abs_mean": stroke_abs_mean,
            "stroke_rate_mean": stroke_rate_mean,
            "stroke_over_0_5_ratio": stroke_over_0_5_ratio,
            "stroke_over_0_55_ratio": stroke_over_0_55_ratio,
            "actual_stroke_min": actual_stroke_min,
            "actual_stroke_max": actual_stroke_max,
            "actual_stroke_mean": actual_stroke_mean,
            "actual_stroke_over_0_5_ratio": actual_stroke_over_0_5_ratio,
            "actual_stroke_over_0_55_ratio": actual_stroke_over_0_55_ratio,
            "front_stroke_mean": front_stroke_mean,
            "rear_stroke_mean": rear_stroke_mean,
            "target_front_rear_stroke_diff": target_front_rear_stroke_diff,
            "actual_front_rear_stroke_diff": actual_front_rear_stroke_diff,
            "joint_pos_g_min": joint_pos_g_min,
            "joint_pos_g_max": joint_pos_g_max,
            "joint_pos_g_abs_max": joint_pos_g_abs_max,
            "wheel_z_b_min": wheel_z_b_min,
            "wheel_z_b_max": wheel_z_b_max,
            "front_rear_stroke_diff": front_rear_stroke_diff,
            "left_right_stroke_diff": left_right_stroke_diff,
            "front_rear_wheel_z_diff": front_rear_wheel_z_diff,
            "left_right_wheel_z_diff": left_right_wheel_z_diff,
            "contact_force_front_mean": contact_force_front_mean,
            "contact_force_rear_mean": contact_force_rear_mean,
            "contact_force_left_mean": contact_force_left_mean,
            "contact_force_right_mean": contact_force_right_mean,
        }

    def _accumulate_forward_debug_metrics(self) -> None:
        if not self._enable_forward_debug_metrics:
            self._accumulate_episode_eval_metrics(None)
            return

        metric_values = self._compute_forward_debug_metric_values()
        for name, value in metric_values.items():
            self._forward_debug_metric_sums[name] += value
        self._forward_debug_metric_counts += 1.0
        self._accumulate_episode_eval_metrics(metric_values)

    def _accumulate_episode_eval_metrics(self, metric_values: dict[str, torch.Tensor] | None) -> None:
        robot = self.scene["robot"]
        joint_limit_margin_params = self.cfg.rewards.joint_limit_margin.params
        joint_pos = robot.data.joint_pos[:, self._leg_joint_ids]
        joint_limits = robot.data.soft_joint_pos_limits[:, self._leg_joint_ids]
        dist_to_lower = joint_pos - joint_limits[..., 0]
        dist_to_upper = joint_limits[..., 1] - joint_pos
        min_margin = torch.minimum(dist_to_lower, dist_to_upper)
        half_range = 0.5 * (joint_limits[..., 1] - joint_limits[..., 0])
        normalized_margin = min_margin / torch.clamp(half_range, min=1.0e-6)
        margin_deficit = torch.clamp(joint_limit_margin_params["margin_ratio"] - normalized_margin, min=0.0)
        joint_limit_margin_penalty = torch.sum(margin_deficit, dim=1)
        joint_limit_margin_count = torch.sum(margin_deficit > 0.0, dim=1).to(torch.float32)

        target_speed = float(self.cfg.rewards.velocity_tracking.params["target_speed"])
        base_vertical_velocity = robot.data.root_lin_vel_b[:, 2]
        base_roll_pitch_ang_vel_sq = torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)
        if metric_values is None:
            base_lin_vel_x = robot.data.root_lin_vel_b[:, 0]
            roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
            roll_deg = torch.rad2deg(roll)
            pitch_deg = torch.rad2deg(pitch)
            abs_roll_deg = torch.abs(roll_deg)
            abs_pitch_deg = torch.abs(pitch_deg)
        else:
            base_lin_vel_x = metric_values["base_lin_vel_x"]
            roll_deg = metric_values["roll_deg"]
            pitch_deg = metric_values["pitch_deg"]
            abs_roll_deg = metric_values["abs_roll_deg"]
            abs_pitch_deg = metric_values["abs_pitch_deg"]
        velocity_tracking_error = base_lin_vel_x - target_speed

        self._eval_step_counts += 1.0
        self._eval_roll_sq_sum += torch.square(roll_deg)
        self._eval_pitch_sq_sum += torch.square(pitch_deg)
        self._eval_roll_abs_max = torch.maximum(self._eval_roll_abs_max, abs_roll_deg)
        self._eval_pitch_abs_max = torch.maximum(self._eval_pitch_abs_max, abs_pitch_deg)
        self._eval_base_vertical_velocity_sq_sum += torch.square(base_vertical_velocity)
        self._eval_base_roll_pitch_ang_vel_sq_sum += base_roll_pitch_ang_vel_sq
        self._eval_forward_speed_sum += base_lin_vel_x
        self._eval_velocity_tracking_error_sq_sum += torch.square(velocity_tracking_error)
        self._eval_joint_limit_margin_penalty_sum += joint_limit_margin_penalty
        self._eval_joint_limit_margin_count_sum += joint_limit_margin_count

    def _consume_forward_debug_logs(self, env_ids) -> dict[str, float]:
        if not self._enable_forward_debug_metrics:
            return {}

        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if env_ids.numel() == 0:
            return {}

        counts = torch.clamp(self._forward_debug_metric_counts[env_ids], min=1.0)
        logs = {}
        for name in self._forward_debug_metric_names:
            per_env_mean = self._forward_debug_metric_sums[name][env_ids] / counts
            logs[f"Metrics/forward_debug/{name}"] = float(per_env_mean.mean().item())
            self._forward_debug_metric_sums[name][env_ids] = 0.0
        self._forward_debug_metric_counts[env_ids] = 0.0
        return logs

    def _consume_episode_eval_logs(self, env_ids) -> dict[str, float]:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if env_ids.numel() == 0:
            return {}

        finished_mask = self._eval_has_episode_result[env_ids]
        if not torch.any(finished_mask):
            return {}

        env_ids = env_ids[finished_mask]
        counts = self._eval_step_counts[env_ids]
        valid_mask = counts > 0.0
        if not torch.any(valid_mask):
            return {}

        env_ids = env_ids[valid_mask]
        counts = torch.clamp(counts[valid_mask], min=1.0)

        success_rate = self._eval_last_time_outs[env_ids].to(torch.float32)
        logs = {
            "Metrics/eval/num_episodes": float(env_ids.numel()),
            "Metrics/eval/roll_rms_deg": float(torch.sqrt(self._eval_roll_sq_sum[env_ids] / counts).mean().item()),
            "Metrics/eval/pitch_rms_deg": float(torch.sqrt(self._eval_pitch_sq_sum[env_ids] / counts).mean().item()),
            "Metrics/eval/roll_max_abs_deg": float(self._eval_roll_abs_max[env_ids].mean().item()),
            "Metrics/eval/pitch_max_abs_deg": float(self._eval_pitch_abs_max[env_ids].mean().item()),
            "Metrics/eval/base_vertical_velocity_rms": float(
                torch.sqrt(self._eval_base_vertical_velocity_sq_sum[env_ids] / counts).mean().item()
            ),
            "Metrics/eval/base_roll_pitch_ang_vel_rms": float(
                torch.sqrt(self._eval_base_roll_pitch_ang_vel_sq_sum[env_ids] / counts).mean().item()
            ),
            "Metrics/eval/average_forward_speed": float((self._eval_forward_speed_sum[env_ids] / counts).mean().item()),
            "Metrics/eval/velocity_tracking_error_rms": float(
                torch.sqrt(self._eval_velocity_tracking_error_sq_sum[env_ids] / counts).mean().item()
            ),
            "Metrics/eval/joint_limit_margin_penalty_mean": float(
                (self._eval_joint_limit_margin_penalty_sum[env_ids] / counts).mean().item()
            ),
            "Metrics/eval/joint_limit_margin_count_mean": float(
                (self._eval_joint_limit_margin_count_sum[env_ids] / counts).mean().item()
            ),
            "Metrics/eval/episode_length": float(counts.mean().item()),
            "Metrics/eval/success_rate": float(success_rate.mean().item()),
        }

        self._eval_step_counts[env_ids] = 0.0
        self._eval_roll_sq_sum[env_ids] = 0.0
        self._eval_pitch_sq_sum[env_ids] = 0.0
        self._eval_roll_abs_max[env_ids] = 0.0
        self._eval_pitch_abs_max[env_ids] = 0.0
        self._eval_base_vertical_velocity_sq_sum[env_ids] = 0.0
        self._eval_base_roll_pitch_ang_vel_sq_sum[env_ids] = 0.0
        self._eval_forward_speed_sum[env_ids] = 0.0
        self._eval_velocity_tracking_error_sq_sum[env_ids] = 0.0
        self._eval_joint_limit_margin_penalty_sum[env_ids] = 0.0
        self._eval_joint_limit_margin_count_sum[env_ids] = 0.0
        self._eval_last_time_outs[env_ids] = False
        self._eval_has_episode_result[env_ids] = False
        return logs

    def step(self, action: torch.Tensor):
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # Accumulate locomotion diagnosis metrics before any terminated env is reset.
        self._accumulate_forward_debug_metrics()

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._eval_last_time_outs[reset_env_ids] = self.termination_manager.time_outs[reset_env_ids]
            self._eval_has_episode_result[reset_env_ids] = True
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids):
        debug_logs = self._consume_forward_debug_logs(env_ids)
        eval_logs = self._consume_episode_eval_logs(env_ids)
        super()._reset_idx(env_ids)
        mdp.reset_local_map_cache(self, env_ids)
        leg_action_term = self.action_manager.get_term("leg_hydraulic")
        self._prev_stroke_command[env_ids] = leg_action_term.stroke_command[env_ids]
        self.extras["log"].update(debug_logs)
        self.extras["log"].update(eval_logs)

    def render(self, recompute: bool = False) -> np.ndarray | None:
        return super().render(recompute=recompute)
