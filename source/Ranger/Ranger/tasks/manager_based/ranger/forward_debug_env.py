from __future__ import annotations

import os
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat

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
        contact_sensor = self.scene.sensors["wheel_contact_forces"]
        self._wheel_contact_body_ids, _ = contact_sensor.find_bodies(
            ["w_lb", "w_lf", "w_rf", "w_rb"],
            preserve_order=True,
        )

        # Wheel joint forward sign.
        # Joint order: [w_lb, w_lf, w_rf, w_rb]
        # Semantic positive wheel velocity means robot forward +x motion.
        self._wheel_forward_sign = torch.tensor(
            [-1.0, -1.0, 1.0, 1.0],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self._is_speed_command_task = hasattr(self.cfg.events, "reset_speed_command")
        self._is_goal_heading_task = hasattr(self.cfg.events, "reset_goal_heading_target")
        self._speed_command_obs_start = 37
        self._speed_command_obs_end = 43
        self._goal_heading_prev_distance = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._goal_heading_debug_prev_abs_error = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._turn_sanity_action = os.environ.get("RANGER_TURN_SANITY_ACTION", "").strip().lower()
        self._turn_sanity_leg_mode = os.environ.get("RANGER_TURN_SANITY_LEG_MODE", "stand").strip().lower()
        self._turn_sanity_settle_steps = max(int(os.environ.get("RANGER_TURN_SANITY_SETTLE_STEPS", "100")), 0)
        self._turn_sanity_stand_leg_action = float(os.environ.get("RANGER_TURN_SANITY_STAND_LEG_ACTION", "-0.34"))
        self._turn_sanity_semantic_cmd = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._turn_sanity_prev_yaw = torch.full((self.num_envs,), float("nan"), dtype=torch.float32, device=self.device)
        self._turn_sanity_term_names = tuple(getattr(self.termination_manager, "active_terms", ()))
        self._turn_sanity_root_height_initial = torch.full(
            (self.num_envs,), float("nan"), dtype=torch.float32, device=self.device
        )
        self._turn_sanity_root_height_step_0 = torch.full_like(self._turn_sanity_root_height_initial, float("nan"))
        self._turn_sanity_root_height_step_20 = torch.full_like(self._turn_sanity_root_height_initial, float("nan"))
        self._turn_sanity_root_height_step_50 = torch.full_like(self._turn_sanity_root_height_initial, float("nan"))
        self._turn_sanity_contact_step_0 = torch.full_like(self._turn_sanity_root_height_initial, float("nan"))
        self._turn_sanity_contact_step_20 = torch.full_like(self._turn_sanity_root_height_initial, float("nan"))
        self._turn_sanity_contact_step_50 = torch.full_like(self._turn_sanity_root_height_initial, float("nan"))
        speed_command_metric_names = (
            "v_x_cmd",
            "v_x_cmd_target",
            "yaw_rate_cmd",
            "yaw_rate_cmd_target",
            "command_time_left",
            "root_lin_vel_b_x",
            "root_ang_vel_b_z",
            "lin_vel_x_error",
            "yaw_rate_error",
            "overspeed_cmd_error",
            "average_forward_speed",
            "average_yaw_rate_abs",
            "expected_wheel_speed_abs_mean",
            "expected_wheel_speed_min",
            "expected_wheel_speed_max",
            "expected_left_wheel_speed",
            "expected_right_wheel_speed",
            "actual_left_wheel_target",
            "actual_right_wheel_target",
            "wheel_target_abs_mean",
            "wheel_command_abs_error",
            "wheel_velocity_target_mean",
            "wheel_velocity_target_min",
            "wheel_velocity_target_max",
            "wheel_joint_velocity_mean",
            "left_wheel_vel_mean",
            "right_wheel_vel_mean",
            "left_right_wheel_vel_diff",
            "root_height_w",
            "base_clearance",
            "stroke_mean",
            "actual_stroke_mean",
            "actual_stroke_over_0_5_ratio",
            "roll_deg",
            "pitch_deg",
            "wheel_contact_count",
            "fl_contact",
            "fr_contact",
            "rl_contact",
            "rr_contact",
            "fl_contact_force",
            "fr_contact_force",
            "rl_contact_force",
            "rr_contact_force",
            "fl_stroke",
            "fr_stroke",
            "rl_stroke",
            "rr_stroke",
            "fl_wheel_target",
            "fr_wheel_target",
            "rl_wheel_target",
            "rr_wheel_target",
            "semantic_fl_wheel_target",
            "semantic_fr_wheel_target",
            "semantic_rl_wheel_target",
            "semantic_rr_wheel_target",
            "semantic_left_wheel_target_mean",
            "semantic_right_wheel_target_mean",
            "semantic_left_right_wheel_diff",
            "semantic_wheel_target_abs_mean",
            "low_cmd_actual_vel_mean",
            "mid_cmd_actual_vel_mean",
            "high_cmd_actual_vel_mean",
            "low_cmd_v_x_cmd_mean",
            "mid_cmd_v_x_cmd_mean",
            "high_cmd_v_x_cmd_mean",
            "low_cmd_v_x_cmd_target_mean",
            "mid_cmd_v_x_cmd_target_mean",
            "high_cmd_v_x_cmd_target_mean",
            "policy_obs_cmd_v_x_feature_mean",
            "policy_obs_cmd_v_x_target_feature_mean",
            "policy_obs_cmd_valid_mean",
            "policy_obs_cmd_time_left_feature_mean",
            "policy_obs_cmd_slot0_mean",
            "policy_obs_cmd_slot1_mean",
            "policy_obs_cmd_slot2_mean",
            "policy_obs_cmd_slot3_mean",
            "policy_obs_cmd_slot4_mean",
            "policy_obs_cmd_slot5_mean",
            "policy_obs_cmd_low_feature_mean",
            "policy_obs_cmd_mid_feature_mean",
            "policy_obs_cmd_high_feature_mean",
            "low_cmd_vel_error_mean",
            "mid_cmd_vel_error_mean",
            "high_cmd_vel_error_mean",
            "low_cmd_env_ratio",
            "mid_cmd_env_ratio",
            "high_cmd_env_ratio",
            "root_height_low",
            "episode_length",
        )
        goal_heading_metric_names = (
            "target_dir_b_x",
            "target_dir_b_y",
            "distance_to_target",
            "heading_error",
            "heading_error_deg",
            "heading_error_abs",
            "heading_error_abs_progress_raw_mean",
            "heading_error_abs_progress_clamped_mean",
            "left_heading_error_abs_progress_clamped_mean",
            "right_heading_error_abs_progress_clamped_mean",
            "front_heading_error_abs_progress_clamped_mean",
            "left_env_ratio",
            "right_env_ratio",
            "front_env_ratio",
            "left_yaw_rate_mean",
            "right_yaw_rate_mean",
            "front_yaw_rate_mean",
            "left_turn_direction_score",
            "right_turn_direction_score",
            "left_target_progress_mean",
            "right_target_progress_mean",
            "front_target_progress_mean",
            "left_forward_speed_mean",
            "right_forward_speed_mean",
            "front_forward_speed_mean",
            "semantic_left_wheel_target_mean",
            "semantic_right_wheel_target_mean",
            "semantic_left_right_diff",
            "semantic_left_right_target_diff",
            "semantic_left_wheel_joint_vel_mean",
            "semantic_right_wheel_joint_vel_mean",
            "semantic_left_right_joint_vel_diff",
            "semantic_target_to_actual_abs_error_mean",
            "semantic_left_actual_over_target_ratio",
            "semantic_right_actual_over_target_ratio",
            "left_semantic_left_wheel_target_mean",
            "left_semantic_right_wheel_target_mean",
            "left_semantic_diff_mean",
            "right_semantic_left_wheel_target_mean",
            "right_semantic_right_wheel_target_mean",
            "right_semantic_diff_mean",
            "left_semantic_left_joint_vel_mean",
            "left_semantic_right_joint_vel_mean",
            "left_semantic_joint_diff_mean",
            "right_semantic_left_joint_vel_mean",
            "right_semantic_right_joint_vel_mean",
            "right_semantic_joint_diff_mean",
            "front_semantic_left_joint_vel_mean",
            "front_semantic_right_joint_vel_mean",
            "front_semantic_joint_diff_mean",
            "front_semantic_diff_mean",
            "root_height_w",
            "root_height_target",
            "root_height_error",
            "base_clearance",
            "fl_stroke",
            "fr_stroke",
            "rl_stroke",
            "rr_stroke",
            "fl_contact_force",
            "fr_contact_force",
            "rl_contact_force",
            "rr_contact_force",
            "contact_force_mean",
            "contact_force_min",
            "wheel_contact_count",
            "actual_stroke_mean",
            "actual_stroke_over_0_5_ratio",
            "root_height_low",
            "episode_length",
            "turn_sanity/semantic_left_cmd",
            "turn_sanity/semantic_right_cmd",
            "turn_sanity/semantic_cmd_diff",
            "turn_sanity/semantic_left_actual_joint_vel",
            "turn_sanity/semantic_right_actual_joint_vel",
            "turn_sanity/semantic_actual_joint_diff",
            "turn_sanity/semantic_tracking_abs_error_mean",
            "turn_sanity/semantic_left_actual_over_target_ratio",
            "turn_sanity/semantic_right_actual_over_target_ratio",
            "turn_sanity/root_lin_vel_b_x_mean",
            "turn_sanity/root_lin_vel_b_y_mean",
            "turn_sanity/root_ang_vel_b_z_mean",
            "turn_sanity/yaw_delta_mean",
            "turn_sanity/wheel_contact_count",
            "turn_sanity/root_height_w",
            "turn_sanity/root_height_initial_after_reset",
            "turn_sanity/root_height_step_0",
            "turn_sanity/root_height_step_20",
            "turn_sanity/root_height_step_50",
            "turn_sanity/root_height_after_settle",
            "turn_sanity/wheel_contact_count_step_0",
            "turn_sanity/wheel_contact_count_step_20",
            "turn_sanity/wheel_contact_count_step_50",
            "turn_sanity/wheel_contact_count_after_settle",
            "turn_sanity/posture_valid",
            "turn_sanity/posture_invalid_reason",
            "turn_sanity/stand_leg_action",
            "turn_sanity/hydraulic_action_fl",
            "turn_sanity/hydraulic_action_fr",
            "turn_sanity/hydraulic_action_rl",
            "turn_sanity/hydraulic_action_rr",
            "turn_sanity/termination_any",
            "turn_sanity/termination_time_out",
            "turn_sanity/termination_bad_orientation",
            "turn_sanity/termination_root_height_low",
            "turn_sanity/termination_other",
            "turn_sanity/termination_other_ratio",
            "turn_sanity/mean_episode_length",
        )
        goal_heading_metric_names = goal_heading_metric_names + tuple(
            f"turn_sanity/termination_term/{name}" for name in self._turn_sanity_term_names
        )
        forward_metric_names = (
            "base_lin_vel_x",
            "wheel_velocity_target_mean",
            "semantic_wheel_velocity_target_mean",
            "wheel_joint_vel_mean",
            "semantic_wheel_joint_vel_mean",
            "wheel_action_abs_mean",
            "forward_progress_raw",
            "target_w_lb",
            "target_w_lf",
            "target_w_rf",
            "target_w_rb",
            "joint_vel_w_lb",
            "joint_vel_w_lf",
            "joint_vel_w_rf",
            "joint_vel_w_rb",
            "semantic_target_w_lb",
            "semantic_target_w_lf",
            "semantic_target_w_rf",
            "semantic_target_w_rb",
            "semantic_joint_vel_w_lb",
            "semantic_joint_vel_w_lf",
            "semantic_joint_vel_w_rf",
            "semantic_joint_vel_w_rb",
            "raw_action_w_lb",
            "raw_action_w_lf",
            "raw_action_w_rf",
            "raw_action_w_rb",
        )
        if self._is_speed_command_task:
            self._forward_debug_metric_names = speed_command_metric_names
            self._debug_log_prefix = "forward_debug"
        elif self._is_goal_heading_task:
            self._forward_debug_metric_names = goal_heading_metric_names
            self._debug_log_prefix = "goal_heading_debug"
        else:
            self._forward_debug_metric_names = forward_metric_names
            self._debug_log_prefix = "forward_debug"
    
        self._forward_debug_metric_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for name in self._forward_debug_metric_names
        }
        self._forward_debug_metric_counts = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._forward_debug_metric_counts_by_name = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for name in self._forward_debug_metric_names
        }


    def _compute_forward_debug_metric_values(self) -> dict[str, torch.Tensor]:
        robot = self.scene["robot"]
        wheel_action_term = self.action_manager.get_term("wheel_motor_csv")
        leg_action_term = self.action_manager.get_term("leg_hydraulic")

        base_lin_vel_x = robot.data.root_lin_vel_b[:, 0]
        root_ang_vel_b_z = robot.data.root_ang_vel_b[:, 2]

        wheel_velocity_target = wheel_action_term.velocity_target
        wheel_joint_vel = robot.data.joint_vel[:, self._wheel_joint_ids]
        semantic_wheel_velocity_target = wheel_velocity_target * self._wheel_forward_sign
        semantic_wheel_joint_vel = wheel_joint_vel * self._wheel_forward_sign
        if self._is_speed_command_task:
            command = mdp.speed_command(self)
            command_target = mdp.speed_command_target(self)
            command_time_left = mdp.speed_command_time_left(self)
            wheel_command_params = self.cfg.rewards.wheel_command_tracking.params
            expected_wheel_speed = mdp.expected_speed_command_wheel_velocity(
                self,
                forward_gain=wheel_command_params.get("forward_gain", 9.0),
                yaw_gain=wheel_command_params.get("yaw_gain", 0.0),
                max_abs_speed=wheel_command_params.get("max_abs_speed", 6.0),
                forward_sign=wheel_command_params.get("forward_sign", 1.0),
            )
            roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
            wheel_contact_sensor = self.scene.sensors["wheel_contact_forces"]
            net_contact_forces = wheel_contact_sensor.data.net_forces_w_history[:, :, self._wheel_contact_body_ids, :]
            contact_force = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
            contacts = contact_force > 1.0
            root_height_low_limit = float(self.cfg.terminations.root_height_low.params["minimum_height"])
            stroke_actual = leg_action_term.stroke_actual
            lin_vel_x_error = base_lin_vel_x - command[:, 0]
            yaw_rate_error = root_ang_vel_b_z - command[:, 1]
            actual_left_wheel_target = wheel_velocity_target[:, :2].mean(dim=1)
            actual_right_wheel_target = wheel_velocity_target[:, 2:].mean(dim=1)
            semantic_left_wheel_target = semantic_wheel_velocity_target[:, :2].mean(dim=1)
            semantic_right_wheel_target = semantic_wheel_velocity_target[:, 2:].mean(dim=1)
            expected_left_wheel_speed = expected_wheel_speed[:, :2].mean(dim=1)
            expected_right_wheel_speed = expected_wheel_speed[:, 2:].mean(dim=1)
            overspeed_cmd_error = mdp.overspeed_command_penalty(
                self,
                **self.cfg.rewards.overspeed_cmd_penalty.params,
            )
            low_cmd_mask = (command[:, 0] >= 0.03) & (command[:, 0] < 0.12)
            mid_cmd_mask = (command[:, 0] >= 0.12) & (command[:, 0] < 0.28)
            high_cmd_mask = (command[:, 0] >= 0.28) & (command[:, 0] <= 0.45)

            def masked_mean_or_zero(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                masked_count = mask.to(torch.float32).sum()
                mean_value = torch.where(
                    masked_count > 0.0,
                    value[mask].sum() / torch.clamp(masked_count, min=1.0),
                    torch.zeros((), device=self.device, dtype=torch.float32),
                )
                return mean_value.expand(self.num_envs)

            low_cmd_ratio = low_cmd_mask.to(torch.float32).mean().expand(self.num_envs)
            mid_cmd_ratio = mid_cmd_mask.to(torch.float32).mean().expand(self.num_envs)
            high_cmd_ratio = high_cmd_mask.to(torch.float32).mean().expand(self.num_envs)
            abs_lin_vel_x_error = torch.abs(lin_vel_x_error)

            policy_obs = None
            obs_buf = getattr(self, "obs_buf", None)
            if hasattr(obs_buf, "get"):
                policy_obs = obs_buf.get("policy", None)
            elif isinstance(obs_buf, torch.Tensor):
                policy_obs = obs_buf
            if (
                policy_obs is not None
                and policy_obs.ndim == 2
                and policy_obs.shape[1] >= self._speed_command_obs_end
            ):
                policy_cmd_obs = policy_obs[:, self._speed_command_obs_start : self._speed_command_obs_end]
            else:
                policy_cmd_obs = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
            policy_obs_cmd_v_x_feature = policy_cmd_obs[:, 0]
            policy_obs_cmd_v_x_target_feature = policy_cmd_obs[:, 4]
            policy_obs_cmd_time_left_feature = policy_cmd_obs[:, 5]
            policy_obs_cmd_valid = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
            return {
                "v_x_cmd": command[:, 0],
                "v_x_cmd_target": command_target[:, 0],
                "yaw_rate_cmd": command[:, 1],
                "yaw_rate_cmd_target": command_target[:, 1],
                "command_time_left": command_time_left,
                "root_lin_vel_b_x": base_lin_vel_x,
                "root_ang_vel_b_z": root_ang_vel_b_z,
                "lin_vel_x_error": lin_vel_x_error,
                "yaw_rate_error": yaw_rate_error,
                "overspeed_cmd_error": overspeed_cmd_error,
                "average_forward_speed": base_lin_vel_x,
                "average_yaw_rate_abs": torch.abs(root_ang_vel_b_z),
                "expected_wheel_speed_abs_mean": torch.mean(torch.abs(expected_wheel_speed), dim=1),
                "expected_wheel_speed_min": torch.min(expected_wheel_speed, dim=1).values,
                "expected_wheel_speed_max": torch.max(expected_wheel_speed, dim=1).values,
                "expected_left_wheel_speed": expected_left_wheel_speed,
                "expected_right_wheel_speed": expected_right_wheel_speed,
                "actual_left_wheel_target": actual_left_wheel_target,
                "actual_right_wheel_target": actual_right_wheel_target,
                "wheel_target_abs_mean": torch.mean(torch.abs(wheel_velocity_target), dim=1),
                "wheel_command_abs_error": torch.mean(torch.abs(wheel_velocity_target - expected_wheel_speed), dim=1),
                "wheel_velocity_target_mean": wheel_velocity_target.mean(dim=1),
                "wheel_velocity_target_min": torch.min(wheel_velocity_target, dim=1).values,
                "wheel_velocity_target_max": torch.max(wheel_velocity_target, dim=1).values,
                "wheel_joint_velocity_mean": wheel_joint_vel.mean(dim=1),
                "left_wheel_vel_mean": actual_left_wheel_target,
                "right_wheel_vel_mean": actual_right_wheel_target,
                "left_right_wheel_vel_diff": actual_left_wheel_target - actual_right_wheel_target,
                "root_height_w": robot.data.root_pos_w[:, 2],
                "base_clearance": robot.data.root_pos_w[:, 2],
                "stroke_mean": stroke_actual.mean(dim=1),
                "actual_stroke_mean": stroke_actual.mean(dim=1),
                "actual_stroke_over_0_5_ratio": (stroke_actual > 0.5).to(torch.float32).mean(dim=1),
                "roll_deg": torch.rad2deg(roll),
                "pitch_deg": torch.rad2deg(pitch),
                "wheel_contact_count": contacts.to(torch.float32).mean(dim=1),
                "fl_contact": contacts[:, 1].to(torch.float32),
                "fr_contact": contacts[:, 2].to(torch.float32),
                "rl_contact": contacts[:, 0].to(torch.float32),
                "rr_contact": contacts[:, 3].to(torch.float32),
                "fl_contact_force": contact_force[:, 1],
                "fr_contact_force": contact_force[:, 2],
                "rl_contact_force": contact_force[:, 0],
                "rr_contact_force": contact_force[:, 3],
                "fl_stroke": stroke_actual[:, 1],
                "fr_stroke": stroke_actual[:, 2],
                "rl_stroke": stroke_actual[:, 0],
                "rr_stroke": stroke_actual[:, 3],
                "fl_wheel_target": wheel_velocity_target[:, 1],
                "fr_wheel_target": wheel_velocity_target[:, 2],
                "rl_wheel_target": wheel_velocity_target[:, 0],
                "rr_wheel_target": wheel_velocity_target[:, 3],
                "semantic_fl_wheel_target": semantic_wheel_velocity_target[:, 1],
                "semantic_fr_wheel_target": semantic_wheel_velocity_target[:, 2],
                "semantic_rl_wheel_target": semantic_wheel_velocity_target[:, 0],
                "semantic_rr_wheel_target": semantic_wheel_velocity_target[:, 3],
                "semantic_left_wheel_target_mean": semantic_left_wheel_target,
                "semantic_right_wheel_target_mean": semantic_right_wheel_target,
                "semantic_left_right_wheel_diff": semantic_left_wheel_target - semantic_right_wheel_target,
                "semantic_wheel_target_abs_mean": torch.mean(torch.abs(semantic_wheel_velocity_target), dim=1),
                "low_cmd_actual_vel_mean": masked_mean_or_zero(base_lin_vel_x, low_cmd_mask),
                "mid_cmd_actual_vel_mean": masked_mean_or_zero(base_lin_vel_x, mid_cmd_mask),
                "high_cmd_actual_vel_mean": masked_mean_or_zero(base_lin_vel_x, high_cmd_mask),
                "low_cmd_v_x_cmd_mean": masked_mean_or_zero(command[:, 0], low_cmd_mask),
                "mid_cmd_v_x_cmd_mean": masked_mean_or_zero(command[:, 0], mid_cmd_mask),
                "high_cmd_v_x_cmd_mean": masked_mean_or_zero(command[:, 0], high_cmd_mask),
                "low_cmd_v_x_cmd_target_mean": masked_mean_or_zero(command_target[:, 0], low_cmd_mask),
                "mid_cmd_v_x_cmd_target_mean": masked_mean_or_zero(command_target[:, 0], mid_cmd_mask),
                "high_cmd_v_x_cmd_target_mean": masked_mean_or_zero(command_target[:, 0], high_cmd_mask),
                "policy_obs_cmd_v_x_feature_mean": policy_obs_cmd_v_x_feature,
                "policy_obs_cmd_v_x_target_feature_mean": policy_obs_cmd_v_x_target_feature,
                "policy_obs_cmd_valid_mean": policy_obs_cmd_valid,
                "policy_obs_cmd_time_left_feature_mean": policy_obs_cmd_time_left_feature,
                "policy_obs_cmd_slot0_mean": policy_cmd_obs[:, 0],
                "policy_obs_cmd_slot1_mean": policy_cmd_obs[:, 1],
                "policy_obs_cmd_slot2_mean": policy_cmd_obs[:, 2],
                "policy_obs_cmd_slot3_mean": policy_cmd_obs[:, 3],
                "policy_obs_cmd_slot4_mean": policy_cmd_obs[:, 4],
                "policy_obs_cmd_slot5_mean": policy_cmd_obs[:, 5],
                "policy_obs_cmd_low_feature_mean": masked_mean_or_zero(policy_obs_cmd_v_x_feature, low_cmd_mask),
                "policy_obs_cmd_mid_feature_mean": masked_mean_or_zero(policy_obs_cmd_v_x_feature, mid_cmd_mask),
                "policy_obs_cmd_high_feature_mean": masked_mean_or_zero(policy_obs_cmd_v_x_feature, high_cmd_mask),
                "low_cmd_vel_error_mean": masked_mean_or_zero(abs_lin_vel_x_error, low_cmd_mask),
                "mid_cmd_vel_error_mean": masked_mean_or_zero(abs_lin_vel_x_error, mid_cmd_mask),
                "high_cmd_vel_error_mean": masked_mean_or_zero(abs_lin_vel_x_error, high_cmd_mask),
                "low_cmd_env_ratio": low_cmd_ratio,
                "mid_cmd_env_ratio": mid_cmd_ratio,
                "high_cmd_env_ratio": high_cmd_ratio,
                "root_height_low": (robot.data.root_pos_w[:, 2] < root_height_low_limit).to(torch.float32),
                "episode_length": self.episode_length_buf.to(torch.float32),
            }

        if self._is_goal_heading_task:
            target_vec_b, distance_to_target, heading_error = mdp.goal_heading_target_body(self)
            target_dir_b = target_vec_b[:, :2] / torch.clamp(distance_to_target.unsqueeze(1), min=1.0e-6)
            heading_abs = torch.abs(heading_error)
            left_mask = heading_error > 0.08
            right_mask = heading_error < -0.08
            front_mask = heading_abs <= 0.08

            def masked_mean_or_zero(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                masked_count = mask.to(torch.float32).sum()
                mean_value = torch.where(
                    masked_count > 0.0,
                    value[mask].sum() / torch.clamp(masked_count, min=1.0),
                    torch.zeros((), device=self.device, dtype=torch.float32),
                )
                return mean_value.expand(self.num_envs)

            wheel_contact_sensor = self.scene.sensors["wheel_contact_forces"]
            net_contact_forces = wheel_contact_sensor.data.net_forces_w_history[:, :, self._wheel_contact_body_ids, :]
            contact_force = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
            contacts = contact_force > 1.0
            stroke_actual = leg_action_term.stroke_actual
            root_height_low_limit = float(self.cfg.terminations.root_height_low.params["minimum_height"])
            semantic_left_wheel_target = semantic_wheel_velocity_target[:, :2].mean(dim=1)
            semantic_right_wheel_target = semantic_wheel_velocity_target[:, 2:].mean(dim=1)
            semantic_wheel_joint_vel = wheel_joint_vel * self._wheel_forward_sign
            semantic_left_wheel_joint_vel = semantic_wheel_joint_vel[:, :2].mean(dim=1)
            semantic_right_wheel_joint_vel = semantic_wheel_joint_vel[:, 2:].mean(dim=1)
            semantic_diff = semantic_left_wheel_target - semantic_right_wheel_target
            semantic_joint_diff = semantic_left_wheel_joint_vel - semantic_right_wheel_joint_vel
            semantic_target_to_actual_abs_error = torch.mean(
                torch.abs(semantic_wheel_velocity_target - semantic_wheel_joint_vel),
                dim=1,
            )
            left_ratio_den = torch.where(
                torch.abs(semantic_left_wheel_target) > 0.1,
                semantic_left_wheel_target,
                torch.ones_like(semantic_left_wheel_target),
            )
            right_ratio_den = torch.where(
                torch.abs(semantic_right_wheel_target) > 0.1,
                semantic_right_wheel_target,
                torch.ones_like(semantic_right_wheel_target),
            )
            semantic_left_actual_over_target_ratio = torch.where(
                torch.abs(semantic_left_wheel_target) > 0.1,
                semantic_left_wheel_joint_vel / left_ratio_den,
                torch.zeros_like(semantic_left_wheel_joint_vel),
            )
            semantic_right_actual_over_target_ratio = torch.where(
                torch.abs(semantic_right_wheel_target) > 0.1,
                semantic_right_wheel_joint_vel / right_ratio_den,
                torch.zeros_like(semantic_right_wheel_joint_vel),
            )
            velocity_towards_target = torch.sum(robot.data.root_lin_vel_b[:, :2] * target_dir_b, dim=1)

            prev_distance = self._goal_heading_prev_distance
            target_progress = torch.where(
                torch.isfinite(prev_distance),
                prev_distance - distance_to_target,
                velocity_towards_target * self.step_dt,
            )
            self._goal_heading_prev_distance[:] = distance_to_target
            heading_error_abs = torch.abs(heading_error)
            prev_heading_error_abs = self._goal_heading_debug_prev_abs_error
            raw_heading_progress = torch.where(
                torch.isfinite(prev_heading_error_abs),
                (prev_heading_error_abs - heading_error_abs) / max(float(self.step_dt), 1.0e-6),
                torch.zeros_like(heading_error_abs),
            )
            clamped_heading_progress = torch.clamp(raw_heading_progress, min=0.0, max=0.5)
            self._goal_heading_debug_prev_abs_error[:] = heading_error_abs

            left_turn_direction = (root_ang_vel_b_z > 0.02).to(torch.float32)
            right_turn_direction = (root_ang_vel_b_z < -0.02).to(torch.float32)
            left_ratio = left_mask.to(torch.float32).mean().expand(self.num_envs)
            right_ratio = right_mask.to(torch.float32).mean().expand(self.num_envs)
            front_ratio = front_mask.to(torch.float32).mean().expand(self.num_envs)
            root_height_target = float(self.cfg.rewards.root_height_tracking.params.get("target_height", 0.74))
            root_height_error = robot.data.root_pos_w[:, 2] - root_height_target
            _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
            yaw_delta = torch.where(
                torch.isfinite(self._turn_sanity_prev_yaw),
                torch.atan2(torch.sin(yaw - self._turn_sanity_prev_yaw), torch.cos(yaw - self._turn_sanity_prev_yaw)),
                torch.zeros_like(yaw),
            )
            self._turn_sanity_prev_yaw[:] = yaw
            after_settle = self.episode_length_buf >= self._turn_sanity_settle_steps
            root_height_ok = (robot.data.root_pos_w[:, 2] >= 0.74) & (robot.data.root_pos_w[:, 2] <= 0.85)
            contact_ok = contacts.to(torch.float32).mean(dim=1) > 0.9
            posture_valid = (after_settle & root_height_ok & contact_ok).to(torch.float32)
            posture_invalid_reason = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            posture_invalid_reason = torch.where(after_settle, posture_invalid_reason, torch.ones_like(posture_invalid_reason))
            posture_invalid_reason = torch.where(
                after_settle & (~root_height_ok) & contact_ok,
                torch.full_like(posture_invalid_reason, 2.0),
                posture_invalid_reason,
            )
            posture_invalid_reason = torch.where(
                after_settle & root_height_ok & (~contact_ok),
                torch.full_like(posture_invalid_reason, 3.0),
                posture_invalid_reason,
            )
            posture_invalid_reason = torch.where(
                after_settle & (~root_height_ok) & (~contact_ok),
                torch.full_like(posture_invalid_reason, 4.0),
                posture_invalid_reason,
            )

            root_height = robot.data.root_pos_w[:, 2]
            contact_count = contacts.to(torch.float32).mean(dim=1)
            episode_length = self.episode_length_buf.to(torch.float32)

            def capture_once(buffer: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> None:
                capture_mask = torch.isnan(buffer) & mask
                if torch.any(capture_mask):
                    buffer[capture_mask] = value[capture_mask]

            capture_once(self._turn_sanity_root_height_initial, root_height, torch.ones_like(after_settle))
            capture_once(self._turn_sanity_root_height_step_0, root_height, self.episode_length_buf <= 1)
            capture_once(self._turn_sanity_contact_step_0, contact_count, self.episode_length_buf <= 1)
            capture_once(self._turn_sanity_root_height_step_20, root_height, self.episode_length_buf >= 20)
            capture_once(self._turn_sanity_contact_step_20, contact_count, self.episode_length_buf >= 20)
            capture_once(self._turn_sanity_root_height_step_50, root_height, self.episode_length_buf >= 50)
            capture_once(self._turn_sanity_contact_step_50, contact_count, self.episode_length_buf >= 50)

            zeros = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            termination_any = self.reset_buf.to(torch.float32)
            termination_time_out = self.reset_time_outs.to(torch.float32)
            termination_term_values: dict[str, torch.Tensor] = {}
            for term_name in self._turn_sanity_term_names:
                termination_term_values[term_name] = self.termination_manager.get_term(term_name).to(torch.float32)
            termination_bad_orientation = termination_term_values.get("bad_orientation", zeros)
            termination_root_height_low = termination_term_values.get("root_height_low", zeros)
            termination_other_bool = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            for term_name, term_value in termination_term_values.items():
                if term_name not in ("time_out", "bad_orientation", "root_height_low"):
                    termination_other_bool |= term_value.to(torch.bool)
            termination_other = termination_other_bool.to(torch.float32)

            hydraulic_action = leg_action_term.raw_actions
            stand_leg_action = torch.full(
                (self.num_envs,),
                self._turn_sanity_stand_leg_action,
                dtype=torch.float32,
                device=self.device,
            )

            values = {
                "target_dir_b_x": target_dir_b[:, 0],
                "target_dir_b_y": target_dir_b[:, 1],
                "distance_to_target": distance_to_target,
                "heading_error": heading_error,
                "heading_error_deg": torch.rad2deg(heading_error),
                "heading_error_abs": heading_error_abs,
                "heading_error_abs_progress_raw_mean": raw_heading_progress,
                "heading_error_abs_progress_clamped_mean": clamped_heading_progress,
                "left_heading_error_abs_progress_clamped_mean": masked_mean_or_zero(clamped_heading_progress, left_mask),
                "right_heading_error_abs_progress_clamped_mean": masked_mean_or_zero(clamped_heading_progress, right_mask),
                "front_heading_error_abs_progress_clamped_mean": masked_mean_or_zero(clamped_heading_progress, front_mask),
                "left_env_ratio": left_ratio,
                "right_env_ratio": right_ratio,
                "front_env_ratio": front_ratio,
                "left_yaw_rate_mean": masked_mean_or_zero(root_ang_vel_b_z, left_mask),
                "right_yaw_rate_mean": masked_mean_or_zero(root_ang_vel_b_z, right_mask),
                "front_yaw_rate_mean": masked_mean_or_zero(root_ang_vel_b_z, front_mask),
                "left_turn_direction_score": masked_mean_or_zero(left_turn_direction, left_mask),
                "right_turn_direction_score": masked_mean_or_zero(right_turn_direction, right_mask),
                "left_target_progress_mean": masked_mean_or_zero(target_progress, left_mask),
                "right_target_progress_mean": masked_mean_or_zero(target_progress, right_mask),
                "front_target_progress_mean": masked_mean_or_zero(target_progress, front_mask),
                "left_forward_speed_mean": masked_mean_or_zero(base_lin_vel_x, left_mask),
                "right_forward_speed_mean": masked_mean_or_zero(base_lin_vel_x, right_mask),
                "front_forward_speed_mean": masked_mean_or_zero(base_lin_vel_x, front_mask),
                "semantic_left_wheel_target_mean": semantic_left_wheel_target,
                "semantic_right_wheel_target_mean": semantic_right_wheel_target,
                "semantic_left_right_diff": semantic_diff,
                "semantic_left_right_target_diff": semantic_diff,
                "semantic_left_wheel_joint_vel_mean": semantic_left_wheel_joint_vel,
                "semantic_right_wheel_joint_vel_mean": semantic_right_wheel_joint_vel,
                "semantic_left_right_joint_vel_diff": semantic_joint_diff,
                "semantic_target_to_actual_abs_error_mean": semantic_target_to_actual_abs_error,
                "semantic_left_actual_over_target_ratio": semantic_left_actual_over_target_ratio,
                "semantic_right_actual_over_target_ratio": semantic_right_actual_over_target_ratio,
                "left_semantic_left_wheel_target_mean": masked_mean_or_zero(semantic_left_wheel_target, left_mask),
                "left_semantic_right_wheel_target_mean": masked_mean_or_zero(semantic_right_wheel_target, left_mask),
                "left_semantic_diff_mean": masked_mean_or_zero(semantic_diff, left_mask),
                "right_semantic_left_wheel_target_mean": masked_mean_or_zero(semantic_left_wheel_target, right_mask),
                "right_semantic_right_wheel_target_mean": masked_mean_or_zero(semantic_right_wheel_target, right_mask),
                "right_semantic_diff_mean": masked_mean_or_zero(semantic_diff, right_mask),
                "left_semantic_left_joint_vel_mean": masked_mean_or_zero(semantic_left_wheel_joint_vel, left_mask),
                "left_semantic_right_joint_vel_mean": masked_mean_or_zero(semantic_right_wheel_joint_vel, left_mask),
                "left_semantic_joint_diff_mean": masked_mean_or_zero(semantic_joint_diff, left_mask),
                "right_semantic_left_joint_vel_mean": masked_mean_or_zero(semantic_left_wheel_joint_vel, right_mask),
                "right_semantic_right_joint_vel_mean": masked_mean_or_zero(semantic_right_wheel_joint_vel, right_mask),
                "right_semantic_joint_diff_mean": masked_mean_or_zero(semantic_joint_diff, right_mask),
                "front_semantic_left_joint_vel_mean": masked_mean_or_zero(semantic_left_wheel_joint_vel, front_mask),
                "front_semantic_right_joint_vel_mean": masked_mean_or_zero(semantic_right_wheel_joint_vel, front_mask),
                "front_semantic_joint_diff_mean": masked_mean_or_zero(semantic_joint_diff, front_mask),
                "front_semantic_diff_mean": masked_mean_or_zero(semantic_diff, front_mask),
                "root_height_w": robot.data.root_pos_w[:, 2],
                "root_height_target": torch.full((self.num_envs,), root_height_target, device=self.device, dtype=torch.float32),
                "root_height_error": root_height_error,
                "base_clearance": robot.data.root_pos_w[:, 2],
                "fl_stroke": stroke_actual[:, 1],
                "fr_stroke": stroke_actual[:, 2],
                "rl_stroke": stroke_actual[:, 0],
                "rr_stroke": stroke_actual[:, 3],
                "fl_contact_force": contact_force[:, 1],
                "fr_contact_force": contact_force[:, 2],
                "rl_contact_force": contact_force[:, 0],
                "rr_contact_force": contact_force[:, 3],
                "contact_force_mean": contact_force.mean(dim=1),
                "contact_force_min": contact_force.min(dim=1).values,
                "wheel_contact_count": contacts.to(torch.float32).mean(dim=1),
                "actual_stroke_mean": stroke_actual.mean(dim=1),
                "actual_stroke_over_0_5_ratio": (stroke_actual > 0.5).to(torch.float32).mean(dim=1),
                "root_height_low": (robot.data.root_pos_w[:, 2] < root_height_low_limit).to(torch.float32),
                "episode_length": episode_length,
                "turn_sanity/semantic_left_cmd": self._turn_sanity_semantic_cmd[:, 0],
                "turn_sanity/semantic_right_cmd": self._turn_sanity_semantic_cmd[:, 1],
                "turn_sanity/semantic_cmd_diff": self._turn_sanity_semantic_cmd[:, 0] - self._turn_sanity_semantic_cmd[:, 1],
                "turn_sanity/semantic_left_actual_joint_vel": semantic_left_wheel_joint_vel,
                "turn_sanity/semantic_right_actual_joint_vel": semantic_right_wheel_joint_vel,
                "turn_sanity/semantic_actual_joint_diff": semantic_joint_diff,
                "turn_sanity/semantic_tracking_abs_error_mean": semantic_target_to_actual_abs_error,
                "turn_sanity/semantic_left_actual_over_target_ratio": semantic_left_actual_over_target_ratio,
                "turn_sanity/semantic_right_actual_over_target_ratio": semantic_right_actual_over_target_ratio,
                "turn_sanity/root_lin_vel_b_x_mean": robot.data.root_lin_vel_b[:, 0],
                "turn_sanity/root_lin_vel_b_y_mean": robot.data.root_lin_vel_b[:, 1],
                "turn_sanity/root_ang_vel_b_z_mean": root_ang_vel_b_z,
                "turn_sanity/yaw_delta_mean": yaw_delta,
                "turn_sanity/wheel_contact_count": contacts.to(torch.float32).mean(dim=1),
                "turn_sanity/root_height_w": root_height,
                "turn_sanity/root_height_initial_after_reset": torch.nan_to_num(
                    self._turn_sanity_root_height_initial, nan=0.0
                ),
                "turn_sanity/root_height_step_0": torch.nan_to_num(self._turn_sanity_root_height_step_0, nan=0.0),
                "turn_sanity/root_height_step_20": torch.nan_to_num(self._turn_sanity_root_height_step_20, nan=0.0),
                "turn_sanity/root_height_step_50": torch.nan_to_num(self._turn_sanity_root_height_step_50, nan=0.0),
                "turn_sanity/root_height_after_settle": root_height,
                "turn_sanity/wheel_contact_count_step_0": torch.nan_to_num(self._turn_sanity_contact_step_0, nan=0.0),
                "turn_sanity/wheel_contact_count_step_20": torch.nan_to_num(
                    self._turn_sanity_contact_step_20, nan=0.0
                ),
                "turn_sanity/wheel_contact_count_step_50": torch.nan_to_num(
                    self._turn_sanity_contact_step_50, nan=0.0
                ),
                "turn_sanity/wheel_contact_count_after_settle": contact_count,
                "turn_sanity/posture_valid": posture_valid,
                "turn_sanity/posture_invalid_reason": posture_invalid_reason,
                "turn_sanity/stand_leg_action": stand_leg_action,
                "turn_sanity/hydraulic_action_fl": hydraulic_action[:, 1],
                "turn_sanity/hydraulic_action_fr": hydraulic_action[:, 2],
                "turn_sanity/hydraulic_action_rl": hydraulic_action[:, 0],
                "turn_sanity/hydraulic_action_rr": hydraulic_action[:, 3],
                "turn_sanity/termination_any": termination_any,
                "turn_sanity/termination_time_out": termination_time_out,
                "turn_sanity/termination_bad_orientation": termination_bad_orientation,
                "turn_sanity/termination_root_height_low": termination_root_height_low,
                "turn_sanity/termination_other": termination_other,
                "turn_sanity/termination_other_ratio": termination_other,
                "turn_sanity/mean_episode_length": episode_length,
            }
            for term_name, term_value in termination_term_values.items():
                values[f"turn_sanity/termination_term/{term_name}"] = term_value
            return values

        # Raw physical joint-space mean. This is kept for debugging, but after adding
        # wheel_forward_sign it is no longer a good "forward motion" indicator.
        wheel_velocity_target_mean = wheel_velocity_target.mean(dim=1)
        wheel_joint_vel_mean = wheel_joint_vel.mean(dim=1)

        # Semantic forward-direction mean.
        # Positive value means the wheels are commanded / rotating in the robot-forward direction.
        semantic_wheel_velocity_target_mean = (
            wheel_velocity_target * self._wheel_forward_sign
        ).mean(dim=1)

        semantic_wheel_joint_vel_mean = (
            wheel_joint_vel * self._wheel_forward_sign
        ).mean(dim=1)

        wheel_action_abs_mean = torch.mean(torch.abs(wheel_action_term.raw_actions), dim=1)
        forward_progress_raw = mdp.forward_velocity_reward(self, speed_scale=1.0)

        return {
            "base_lin_vel_x": base_lin_vel_x,
            "wheel_velocity_target_mean": wheel_velocity_target_mean,
            "semantic_wheel_velocity_target_mean": semantic_wheel_velocity_target_mean,
            "wheel_joint_vel_mean": wheel_joint_vel_mean,
            "semantic_wheel_joint_vel_mean": semantic_wheel_joint_vel_mean,
            "wheel_action_abs_mean": wheel_action_abs_mean,
            "forward_progress_raw": forward_progress_raw,
            
            # per-wheel physical joint-space target
            "target_w_lb": wheel_velocity_target[:, 0],
            "target_w_lf": wheel_velocity_target[:, 1],
            "target_w_rf": wheel_velocity_target[:, 2],
            "target_w_rb": wheel_velocity_target[:, 3],

            # per-wheel physical joint velocity
            "joint_vel_w_lb": wheel_joint_vel[:, 0],
            "joint_vel_w_lf": wheel_joint_vel[:, 1],
            "joint_vel_w_rf": wheel_joint_vel[:, 2],
            "joint_vel_w_rb": wheel_joint_vel[:, 3],

            # per-wheel semantic forward target
            "semantic_target_w_lb": semantic_wheel_velocity_target[:, 0],
            "semantic_target_w_lf": semantic_wheel_velocity_target[:, 1],
            "semantic_target_w_rf": semantic_wheel_velocity_target[:, 2],
            "semantic_target_w_rb": semantic_wheel_velocity_target[:, 3],

            # per-wheel semantic forward joint velocity
            "semantic_joint_vel_w_lb": semantic_wheel_joint_vel[:, 0],
            "semantic_joint_vel_w_lf": semantic_wheel_joint_vel[:, 1],
            "semantic_joint_vel_w_rf": semantic_wheel_joint_vel[:, 2],
            "semantic_joint_vel_w_rb": semantic_wheel_joint_vel[:, 3],

            "raw_action_w_lb": wheel_action_term.raw_actions[:, 0],
            "raw_action_w_lf": wheel_action_term.raw_actions[:, 1],
            "raw_action_w_rf": wheel_action_term.raw_actions[:, 2],
            "raw_action_w_rb": wheel_action_term.raw_actions[:, 3],
        }

    def _accumulate_forward_debug_metrics(self) -> None:
        metric_values = self._compute_forward_debug_metric_values()
        if self._is_goal_heading_task and self._turn_sanity_action and self._turn_sanity_settle_steps > 0:
            active_mask = self.episode_length_buf >= self._turn_sanity_settle_steps
            always_names = {
                "turn_sanity/root_height_initial_after_reset",
                "turn_sanity/root_height_step_0",
                "turn_sanity/root_height_step_20",
                "turn_sanity/root_height_step_50",
                "turn_sanity/wheel_contact_count_step_0",
                "turn_sanity/wheel_contact_count_step_20",
                "turn_sanity/wheel_contact_count_step_50",
                "turn_sanity/stand_leg_action",
                "turn_sanity/hydraulic_action_fl",
                "turn_sanity/hydraulic_action_fr",
                "turn_sanity/hydraulic_action_rl",
                "turn_sanity/hydraulic_action_rr",
            }
            episode_end_names = {
                "turn_sanity/termination_any",
                "turn_sanity/termination_time_out",
                "turn_sanity/termination_bad_orientation",
                "turn_sanity/termination_root_height_low",
                "turn_sanity/termination_other",
                "turn_sanity/termination_other_ratio",
                "turn_sanity/mean_episode_length",
            }
            reset_mask = self.reset_buf.to(torch.bool)
            for name, value in metric_values.items():
                if name in episode_end_names or name.startswith("turn_sanity/termination_term/"):
                    mask = reset_mask
                elif name in always_names:
                    mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
                else:
                    mask = active_mask
                if torch.any(mask):
                    self._forward_debug_metric_sums[name][mask] += value[mask]
                    self._forward_debug_metric_counts_by_name[name][mask] += 1.0
            return
        for name, value in metric_values.items():
            self._forward_debug_metric_sums[name] += value
            self._forward_debug_metric_counts_by_name[name] += 1.0
        self._forward_debug_metric_counts += 1.0

    def _consume_forward_debug_logs(self, env_ids) -> dict[str, float]:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if env_ids.numel() == 0:
            return {}

        logs = {}
        for name in self._forward_debug_metric_names:
            counts = torch.clamp(self._forward_debug_metric_counts_by_name[name][env_ids], min=1.0)
            per_env_mean = self._forward_debug_metric_sums[name][env_ids] / counts
            if name.startswith("turn_sanity/"):
                log_name = f"Metrics/{name}"
            else:
                log_name = f"Metrics/{self._debug_log_prefix}/{name}"
            logs[log_name] = float(per_env_mean.mean().item())
            self._forward_debug_metric_sums[name][env_ids] = 0.0
            self._forward_debug_metric_counts_by_name[name][env_ids] = 0.0
        self._forward_debug_metric_counts[env_ids] = 0.0
        return logs

    def _apply_goal_heading_turn_sanity_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self._is_goal_heading_task or not self._turn_sanity_action:
            self._turn_sanity_semantic_cmd.zero_()
            return action

        presets = {
            "forward": (2.0, 2.0),
            "left": (1.0, 2.5),
            "right": (2.5, 1.0),
            "left_strong": (0.5, 3.5),
            "right_strong": (3.5, 0.5),
        }
        if self._turn_sanity_action not in presets:
            if self.common_step_counter == 0:
                print(f"[WARN] Unsupported RANGER_TURN_SANITY_ACTION={self._turn_sanity_action!r}; ignoring override.")
            self._turn_sanity_semantic_cmd.zero_()
            return action

        settling = self.episode_length_buf < self._turn_sanity_settle_steps
        left_semantic, right_semantic = presets[self._turn_sanity_action]
        velocity_limit = max(float(self.cfg.actions.wheel_motor_csv.velocity_limit), 1.0e-6)
        action = action.clone()

        if self._turn_sanity_leg_mode == "zero":
            action[:, 0:4] = 0.0
        elif self._turn_sanity_leg_mode == "stand":
            action[:, 0:4] = self._turn_sanity_stand_leg_action
        elif self._turn_sanity_leg_mode != "policy":
            if self.common_step_counter == 0:
                print(
                    f"[WARN] Unsupported RANGER_TURN_SANITY_LEG_MODE={self._turn_sanity_leg_mode!r}; "
                    "using stand mode."
                )
            action[:, 0:4] = self._turn_sanity_stand_leg_action

        action[:, 4:8] = 0.0
        active_mask = ~settling
        if torch.any(active_mask):
            action[active_mask, 4:6] = float(left_semantic) / velocity_limit
            action[active_mask, 6:8] = float(right_semantic) / velocity_limit
        self._turn_sanity_semantic_cmd.zero_()
        self._turn_sanity_semantic_cmd[active_mask, 0] = float(left_semantic)
        self._turn_sanity_semantic_cmd[active_mask, 1] = float(right_semantic)
        return action

    def step(self, action: torch.Tensor):
        # process actions
        action = self._apply_goal_heading_turn_sanity_action(action.to(self.device))
        self.action_manager.process_action(action)

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
        super()._reset_idx(env_ids)
        if self._is_goal_heading_task:
            self._goal_heading_prev_distance[env_ids] = float("nan")
            self._goal_heading_debug_prev_abs_error[env_ids] = float("nan")
            self._turn_sanity_prev_yaw[env_ids] = float("nan")
            self._turn_sanity_root_height_initial[env_ids] = float("nan")
            self._turn_sanity_root_height_step_0[env_ids] = float("nan")
            self._turn_sanity_root_height_step_20[env_ids] = float("nan")
            self._turn_sanity_root_height_step_50[env_ids] = float("nan")
            self._turn_sanity_contact_step_0[env_ids] = float("nan")
            self._turn_sanity_contact_step_20[env_ids] = float("nan")
            self._turn_sanity_contact_step_50[env_ids] = float("nan")
        self.extras["log"].update(debug_logs)

    def render(self, recompute: bool = False) -> np.ndarray | None:
        return super().render(recompute=recompute)
