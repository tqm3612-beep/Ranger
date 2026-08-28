from __future__ import annotations

import csv
import datetime as dt
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz

from . import mdp
from .mdp.rewards import _wheel_contact_force_ratio_lf_lr_rf_rr
from Ranger.assets.ranger.ranger_cfg import RANGER_URDF_PATH


STAND_RESET_PATTERN_NAMES = (
    "left_right_slope",
    "front_back_slope",
    "diagonal_twist",
    "single_wheel_bump",
    "single_wheel_dip",
    "mild_random_mixed",
)


def _sum_urdf_link_masses(urdf_path: str | Path) -> float:
    """Return the total URDF link mass for lightweight debug comparisons."""

    root = ET.parse(str(urdf_path)).getroot()
    total_mass = 0.0
    for link_elem in root.findall("link"):
        inertial_elem = link_elem.find("inertial")
        if inertial_elem is None:
            continue
        mass_elem = inertial_elem.find("mass")
        if mass_elem is None:
            continue
        total_mass += float(mass_elem.attrib.get("value", "0.0"))
    return total_mass


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
        self._wheel_robot_body_ids_lf_lr_rf_rr, _ = robot.find_bodies(
            ["w_lf", "w_lb", "w_rf", "w_rb"],
            preserve_order=True,
        )
        self._wheel_robot_body_ids_lb_lf_rf_rb, _ = robot.find_bodies(
            ["w_lb", "w_lf", "w_rf", "w_rb"],
            preserve_order=True,
        )
        self._base_link_body_ids, _ = robot.find_bodies(
            ["base_link"],
            preserve_order=True,
        )
        contact_sensor = self.scene.sensors["wheel_contact_forces"]
        self._wheel_contact_body_ids, _ = contact_sensor.find_bodies(
            ["w_lb", "w_lf", "w_rf", "w_rb"],
            preserve_order=True,
        )
        self._robot_body_names_joined = "|".join(list(robot.body_names))
        self._wheel_contact_sensor_body_names_joined = "|".join(list(contact_sensor.body_names))
        all_body_contact_sensor = self.scene.sensors["all_body_contact_forces"]
        all_body_names = list(robot.body_names)
        self._all_body_contact_sensor_body_names_joined = "|".join(list(all_body_contact_sensor.body_names))
        all_body_contact_body_ids, _ = all_body_contact_sensor.find_bodies(
            all_body_names,
            preserve_order=True,
        )
        self._all_body_contact_body_ids = torch.as_tensor(
            all_body_contact_body_ids, device=self.device, dtype=torch.long
        )
        self._non_wheel_body_names = [name for name in all_body_names if name not in {"w_lb", "w_lf", "w_rf", "w_rb"}]
        if len(self._non_wheel_body_names) > 0:
            non_wheel_contact_body_ids, _ = all_body_contact_sensor.find_bodies(
                self._non_wheel_body_names,
                preserve_order=True,
            )
            self._non_wheel_contact_body_ids = torch.as_tensor(
                non_wheel_contact_body_ids, device=self.device, dtype=torch.long
            )
        else:
            self._non_wheel_contact_body_ids = torch.empty((0,), dtype=torch.long, device=self.device)
        self._stand_debug_robot_total_mass = float(_sum_urdf_link_masses(RANGER_URDF_PATH))
        self._stand_debug_expected_weight_force = self._stand_debug_robot_total_mass * 9.81
        ground_cfg = getattr(self.cfg.scene, "ground", None)
        ground_pos_z = 0.0
        ground_size_z = 0.0
        if ground_cfg is not None:
            ground_pos_z = float(getattr(getattr(ground_cfg, "init_state", None), "pos", (0.0, 0.0, 0.0))[2])
            ground_size_z = float(getattr(getattr(ground_cfg, "spawn", None), "size", (0.0, 0.0, 0.0))[2])
        self._stand_trace_ground_height = ground_pos_z + 0.5 * ground_size_z

        # Raw wheel order is [w_lb, w_lf, w_rf, w_rb].
        # Semantic wheel sign unifies left/right mirrored joint axes.
        self._wheel_forward_sign = mdp.wheel_semantic_sign_lr_lf_rf_rr(device=self.device)

        self._is_speed_command_task = getattr(self.cfg.events, "reset_speed_command", None) is not None
        self._is_goal_heading_task = getattr(self.cfg.events, "reset_goal_heading_target", None) is not None
        self._is_short_goal_task = getattr(self.cfg.events, "reset_short_goal_target", None) is not None
        self._is_yaw_rate_command_task = bool(getattr(self.cfg, "yaw_rate_command_task", False))
        self._is_short_goal_turn_task = bool(getattr(self.cfg, "short_goal_turn_task", False))
        self._is_yaw_turn_support_task = bool(getattr(self.cfg, "yaw_turn_support_task", False))
        self._is_turn_to_target_task = bool(getattr(self.cfg, "turn_to_target_task", False))
        self._is_stand_training_task = bool(getattr(self.cfg, "stand_training_task", False))
        self._support_contact_sign_mode = os.getenv("RANGER_SUPPORT_CONTACT_SIGN", "auto").strip().lower()
        if self._support_contact_sign_mode not in {"auto", "normal", "inverted"}:
            self._support_contact_sign_mode = "auto"
        self._short_goal_visualization_enabled = (
            self._is_short_goal_task and os.getenv("RANGER_VISUALIZE_GOAL", "0") == "1"
        )
        self._short_goal_goal_marker: VisualizationMarkers | None = None
        self._stand_debug_metrics_enabled = os.getenv("RANGER_DEBUG_METRICS", "0") == "1"
        self._command_obs_start = 26
        self._command_obs_end = 34
        self._goal_heading_prev_distance = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._goal_heading_debug_prev_abs_error = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._turn_sanity_action = os.environ.get("RANGER_TURN_SANITY_ACTION", "").strip().lower()
        self._turn_sanity_leg_mode = os.environ.get("RANGER_TURN_SANITY_LEG_MODE", "stand").strip().lower()
        self._turn_sanity_settle_steps = max(int(os.environ.get("RANGER_TURN_SANITY_SETTLE_STEPS", "100")), 0)
        self._turn_sanity_stand_leg_action = float(os.environ.get("RANGER_TURN_SANITY_STAND_LEG_ACTION", "-0.34"))
        fixed_suspension_action_text = os.environ.get("RANGER_FIXED_SUSPENSION_ACTION", "").strip()
        configured_fixed_suspension_action = getattr(self.cfg, "fixed_suspension_action", None)
        self._fixed_suspension_action_enabled = bool(fixed_suspension_action_text) or (
            configured_fixed_suspension_action is not None
        )
        self._fixed_suspension_action = 0.0
        if self._fixed_suspension_action_enabled:
            if fixed_suspension_action_text:
                self._fixed_suspension_action = float(fixed_suspension_action_text)
                fixed_suspension_source = "environment override"
            else:
                self._fixed_suspension_action = float(configured_fixed_suspension_action)
                fixed_suspension_source = "task configuration"
            if not -1.0 <= self._fixed_suspension_action <= 1.0:
                raise ValueError(
                    "Fixed suspension action must be within [-1, 1], "
                    f"got {self._fixed_suspension_action}."
                )
            print(
                "[Ranger] Fixed suspension action enabled from "
                f"{fixed_suspension_source}: all four hydraulic actions={self._fixed_suspension_action:.3f}"
            )
        suspension_action_scale_text = os.environ.get("RANGER_SUSPENSION_ACTION_SCALE", "").strip()
        self._suspension_action_scale = float(
            suspension_action_scale_text
            if suspension_action_scale_text
            else getattr(self.cfg, "suspension_action_scale", 1.0)
        )
        if not 0.0 <= self._suspension_action_scale <= 1.0:
            raise ValueError(
                "Suspension action scale must be within [0, 1], "
                f"got {self._suspension_action_scale}."
            )
        if self._suspension_action_scale != 1.0:
            print(f"[Ranger] Suspension action scale enabled: {self._suspension_action_scale:.3f}")

        stop_phase_suspension_action_text = os.environ.get(
            "RANGER_STOP_PHASE_SUSPENSION_ACTION", ""
        ).strip()
        configured_stop_phase_suspension_action = getattr(self.cfg, "stop_phase_suspension_action", None)
        self._stop_phase_suspension_mode = str(
            getattr(self.cfg, "stop_phase_suspension_mode", "legacy")
        ).strip().lower()
        if self._stop_phase_suspension_mode not in {"legacy", "policy"}:
            raise ValueError(
                "Stop-phase suspension mode must be 'legacy' or 'policy', "
                f"got {self._stop_phase_suspension_mode!r}."
            )
        if self._stop_phase_suspension_mode == "policy" and stop_phase_suspension_action_text:
            raise ValueError(
                "RANGER_STOP_PHASE_SUSPENSION_ACTION cannot be used when "
                "stop_phase_suspension_mode='policy'."
            )
        self._stop_phase_suspension_action_enabled = (
            self._stop_phase_suspension_mode == "legacy"
            and (
                bool(stop_phase_suspension_action_text)
                or configured_stop_phase_suspension_action is not None
            )
        )
        self._stop_phase_suspension_action = 0.0
        self._stop_phase_wheel_override_enabled = bool(
            getattr(self.cfg, "stop_phase_wheel_override_enabled", True)
        )
        if self._stop_phase_suspension_mode == "policy":
            wheel_mode = "forced to zero" if self._stop_phase_wheel_override_enabled else "policy output preserved"
            print(
                "[Ranger] Stop-phase control: suspension policy output preserved; "
                f"wheel actions {wheel_mode}."
            )
        elif self._stop_phase_suspension_action_enabled:
            if stop_phase_suspension_action_text:
                self._stop_phase_suspension_action = float(stop_phase_suspension_action_text)
                stop_phase_source = "environment override"
            else:
                self._stop_phase_suspension_action = float(configured_stop_phase_suspension_action)
                stop_phase_source = "task configuration"
            if not -1.0 <= self._stop_phase_suspension_action <= 1.0:
                raise ValueError(
                    "Stop-phase suspension action must be within [-1, 1], "
                    f"got {self._stop_phase_suspension_action}."
                )
            print(
                "[Ranger] Stop-phase suspension override enabled from "
                f"{stop_phase_source}: all four hydraulic actions={self._stop_phase_suspension_action:.3f}"
            )
        if self._fixed_suspension_action_enabled and (
            self._stop_phase_suspension_action_enabled or self._stop_phase_suspension_mode == "policy"
        ):
            raise ValueError(
                "Fixed suspension control cannot be combined with stop-phase suspension override "
                "or stop_phase_suspension_mode='policy'."
            )
        self._enable_reset_settle = bool(getattr(self.cfg, "enable_reset_settle", False))
        self._reset_settle_steps = max(int(getattr(self.cfg, "reset_settle_steps", 0)), 0)
        self._reset_settle_leg_action = float(getattr(self.cfg, "reset_settle_leg_action", -0.34))
        self._reset_settle_remaining_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._reset_settle_step_active = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._turn_sanity_semantic_cmd = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._turn_sanity_prev_yaw = torch.full((self.num_envs,), float("nan"), dtype=torch.float32, device=self.device)
        # support_hold leg order is always [lb, lf, rf, rb].
        self._support_hold_leg_order = ("lb", "lf", "rf", "rb")
        self._support_hold_action_raw_to_lb_lf_rf_rb = torch.tensor([0, 1, 2, 3], device=self.device, dtype=torch.long)
        self._support_hold_stroke_to_lb_lf_rf_rb = torch.tensor([0, 1, 2, 3], device=self.device, dtype=torch.long)
        self._support_hold_contact_to_lb_lf_rf_rb = torch.tensor([0, 1, 2, 3], device=self.device, dtype=torch.long)
        self._yaw_turn_support_hold_semantic_action = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._yaw_turn_support_hold_raw_action = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._yaw_turn_support_executed_raw_action = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._yaw_turn_support_contact_correction_semantic = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._yaw_turn_support_stroke_nominal_correction_semantic = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._yaw_turn_support_stroke_range_correction_semantic = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._yaw_turn_support_diag_correction_semantic = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._yaw_turn_support_stroke_correction_semantic = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._yaw_turn_support_roll_pitch_correction_semantic = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._yaw_turn_support_non_contact_correction_semantic = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._yaw_turn_support_contact_cancellation_semantic = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._yaw_turn_support_total_pre_filter_semantic = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._policy_hydraulic_action_raw = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._short_goal_policy_wheel_action_raw = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._short_goal_stop_latched_hydraulic_action_raw = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._short_goal_hydraulic_action_prev = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._short_goal_stop_phase_active = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._short_goal_stop_phase_stable_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._short_goal_initial_side_sign = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._short_goal_path_length_m = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._short_goal_prev_root_xy = robot.data.root_pos_w[:, :2].detach().clone()
        self._short_goal_initial_goal_distance_m = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._short_goal_min_goal_distance_m = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=self.device
        )
        self._short_goal_max_distance_rebound_after_min_m = torch.zeros(
            (self.num_envs,), dtype=torch.float32, device=self.device
        )
        self._short_goal_last_turn_sign = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._short_goal_turn_reversal_count = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._short_goal_turn_reversal_deadband = 0.03
        self._short_goal_stop_phase_enter_distance = float(getattr(self.cfg, "short_goal_stop_phase_enter_distance", 0.25))
        self._short_goal_stop_success_distance = float(getattr(self.cfg, "short_goal_stop_success_distance", 0.25))
        self._short_goal_stop_max_xy_speed = float(getattr(self.cfg, "short_goal_stop_max_xy_speed", 0.15))
        self._short_goal_stop_max_yaw_rate = float(getattr(self.cfg, "short_goal_stop_max_yaw_rate", 0.20))
        self._short_goal_stop_max_roll = float(getattr(self.cfg, "short_goal_stop_max_roll", float("inf")))
        self._short_goal_stop_max_pitch = float(getattr(self.cfg, "short_goal_stop_max_pitch", float("inf")))
        self._short_goal_stop_max_stroke_tracking_error = float(
            getattr(self.cfg, "short_goal_stop_max_stroke_tracking_error", float("inf"))
        )
        self._short_goal_stop_count_only_after_phase = bool(
            getattr(self.cfg, "short_goal_stop_count_only_after_phase", False)
        )
        self._short_goal_stop_required_hold_steps = int(getattr(self.cfg, "short_goal_stop_required_hold_steps", 24))
        self._executed_hydraulic_action_prev = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        nominal_stroke_cfg = getattr(getattr(getattr(self.cfg, "rewards", None), "actual_stroke_nominal", None), "params", None)
        nominal_stroke_value = None
        if isinstance(nominal_stroke_cfg, dict):
            nominal_stroke_value = nominal_stroke_cfg.get("stroke_nominal", None)
        if nominal_stroke_value is None:
            nominal_stroke_value = 0.5
        self._yaw_turn_support_nominal_stroke = float(nominal_stroke_value)
        if self._support_contact_sign_mode == "inverted":
            self._support_contact_correction_sign = -1.0
        else:
            self._support_contact_correction_sign = 1.0
        self._stand_reset_yaw = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._stand_recent_reset_happened = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._stand_reset_pattern_code = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._stand_reset_stroke_init_lf_lr_rf_rr = torch.full(
            (self.num_envs, 4), 0.5, dtype=torch.float32, device=self.device
        )
        self._stand_reset_root_z_init = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._stand_reset_roll_init = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._stand_reset_pitch_init = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._stand_reset_yaw_init = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._stand_first50_max_contact_over_weight = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._stand_first50_max_non_wheel_contact_over_weight = torch.zeros(
            (self.num_envs,), dtype=torch.float32, device=self.device
        )
        self._stand_first50_any_airborne = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._stand_trace_enabled = self._is_stand_training_task and os.environ.get("RANGER_STAND_TRACE", "0").strip() == "1"
        self._stand_trace_mode = os.environ.get("RANGER_STAND_TRACE_MODE", "policy_action").strip().lower()
        if self._stand_trace_mode not in {"zero_action", "policy_action"}:
            self._stand_trace_mode = "policy_action"
        self._stand_trace_zero_action_horizon = 300
        self._stand_trace_env_ids = torch.tensor([0], dtype=torch.long, device=self.device)
        self._stand_last50_length = 50
        self._stand_last50_index = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._stand_last50_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._stand_last50_root_height = torch.zeros((self.num_envs, self._stand_last50_length), dtype=torch.float32, device=self.device)
        self._stand_last50_root_lin_vel_z = torch.zeros((self.num_envs, self._stand_last50_length), dtype=torch.float32, device=self.device)
        self._stand_last50_wheel_contact_over_weight = torch.zeros((self.num_envs, self._stand_last50_length), dtype=torch.float32, device=self.device)
        self._stand_last50_non_wheel_contact_over_weight = torch.zeros((self.num_envs, self._stand_last50_length), dtype=torch.float32, device=self.device)
        self._stand_trace_csv_file = None
        self._stand_trace_csv_writer = None
        self._stand_trace_csv_path: Path | None = None
        if self._stand_trace_enabled:
            trace_root = Path("logs") / "stand_trace" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_root.mkdir(parents=True, exist_ok=True)
            self._stand_trace_csv_path = trace_root / f"stand_trace_{self._stand_trace_mode}_env0.csv"
            self._stand_trace_csv_file = self._stand_trace_csv_path.open("w", newline="", encoding="utf-8")
            self._stand_trace_csv_writer = csv.DictWriter(
                self._stand_trace_csv_file,
                fieldnames=[
                    "step",
                    "episode_step",
                    "reset_happened",
                    "termination_any",
                    "termination_reason",
                    "root_height",
                    "root_height_reward_used",
                    "root_world_z",
                    "root_world_z_minus_env_origin_z",
                    "root_world_z_minus_terrain_height",
                    "base_world_pos_x",
                    "base_world_pos_y",
                    "base_world_pos_z",
                    "base_link_world_z",
                    "root_lin_vel_z",
                    "roll",
                    "pitch",
                    "wheel_world_z_lf",
                    "wheel_world_z_lr",
                    "wheel_world_z_rf",
                    "wheel_world_z_rr",
                    "wheel_relative_z_lf",
                    "wheel_relative_z_lr",
                    "wheel_relative_z_rf",
                    "wheel_relative_z_rr",
                    "env_origin_x",
                    "env_origin_y",
                    "env_origin_z",
                    "terrain_height_under_robot",
                    "robot_body_names",
                    "wheel_contact_sensor_body_names",
                    "all_body_contact_sensor_body_names",
                    "reset_pattern_type",
                    "stroke_init_lf",
                    "stroke_init_lr",
                    "stroke_init_rf",
                    "stroke_init_rr",
                    "root_z_init",
                    "roll_init",
                    "pitch_init",
                    "yaw_init",
                    "stroke_lf",
                    "stroke_lr",
                    "stroke_rf",
                    "stroke_rr",
                    "target_stroke_lf",
                    "target_stroke_lr",
                    "target_stroke_rf",
                    "target_stroke_rr",
                    "raw_action_lf",
                    "raw_action_lr",
                    "raw_action_rf",
                    "raw_action_rr",
                    "clipped_action_lf",
                    "clipped_action_lr",
                    "clipped_action_rf",
                    "clipped_action_rr",
                    "max_abs_joint_pos_minus_target",
                    "wheel_contact_force_over_weight",
                    "non_wheel_contact_force_over_weight",
                    "contact_bool_lf",
                    "contact_bool_lr",
                    "contact_bool_rf",
                    "contact_bool_rr",
                    "first50_max_contact_force_over_weight",
                    "first50_max_non_wheel_contact_force_over_weight",
                    "first50_any_airborne",
                    "root_height_low_raw",
                    "root_height_low_after_grace",
                    "actual_root_height_low_term",
                    "grace_active",
                    "root_height_last50_mean",
                    "root_lin_vel_z_last50_mean",
                    "wheel_contact_over_weight_last50_mean",
                    "non_wheel_contact_over_weight_last50_mean",
                ],
            )
            self._stand_trace_csv_writer.writeheader()
            self._stand_trace_csv_file.flush()
        if self._short_goal_visualization_enabled:
            self._initialize_short_goal_visualizer()
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
        turn_to_target_metric_names = (
            "turn/heading_error",
            "turn/abs_heading_error",
            "turn/turn_progress",
            "turn/yaw_rate_cmd",
            "turn/base_ang_vel_z",
            "turn/goal_x_body",
            "turn/goal_y_body",
            "turn/goal_distance",
            "turn/target_lf",
            "turn/target_lr",
            "turn/target_rf",
            "turn/target_rr",
            "turn/actual_lf",
            "turn/actual_lr",
            "turn/actual_rf",
            "turn/actual_rr",
            "turn/torque_lf",
            "turn/torque_lr",
            "turn/torque_rf",
            "turn/torque_rr",
            "turn/contact_force_lf",
            "turn/contact_force_lr",
            "turn/contact_force_rf",
            "turn/contact_force_rr",
            "turn/contact_bool_lf",
            "turn/contact_bool_lr",
            "turn/contact_bool_rf",
            "turn/contact_bool_rr",
            "turn/contact_force_diag_diff",
            "turn/contact_force_front_rear_diff",
            "turn/contact_force_left_right_diff",
            "turn/base_roll",
            "turn/base_pitch",
            "turn/root_height",
            "turn/hydraulic_stroke_lf",
            "turn/hydraulic_stroke_lr",
            "turn/hydraulic_stroke_rf",
            "turn/hydraulic_stroke_rr",
            "turn/reset_settling",
            "turn/reset_settle_steps_remaining",
            "turn/wheel_vel_shape_dim",
            "turn/unloaded_mask_shape_dim",
            "turn/target_left_front_rear_diff",
            "turn/target_right_front_rear_diff",
            "turn/raw_target_left_mean",
            "turn/raw_target_right_mean",
            "turn/raw_target_diff",
            "turn/raw_target_common",
            "turn/semantic_target_left_mean",
            "turn/semantic_target_right_mean",
            "turn/semantic_target_diff",
            "turn/semantic_target_common",
            "turn/actual_left_front_rear_diff",
            "turn/actual_right_front_rear_diff",
            "turn/left_wheel_vel_mean",
            "turn/right_wheel_vel_mean",
            "turn/wheel_vel_diff",
            "turn/semantic_target_diff_times_yaw_cmd",
            "turn/semantic_target_diff_times_yaw_rate",
            "turn/yaw_cmd_times_yaw_rate",
            "turn/heading_sign_times_yaw_rate",
            "turn/yaw_rate_abs_error",
            "turn/wheel_diff_times_yaw_rate",
            "turn/reward_heading_alignment",
            "turn/reward_turn_progress",
            "turn/reward_yaw_tracking",
            "turn/reward_same_side_front_rear_penalty",
            "turn/reward_unloaded_wheel_spin_penalty",
            "turn/reward_wheel_target_overspeed_penalty",
            "turn/reward_wheel_target_common_mode_penalty",
            "turn/termination_any",
            "turn/termination_time_out",
            "turn/termination_bad_orientation",
            "turn/termination_root_height_low",
            "turn/termination_other",
            "turn/episode_length",
        )
        turn_to_target_metric_names = turn_to_target_metric_names + tuple(
            f"turn/termination_term/{name}" for name in self._turn_sanity_term_names
        )
        stand_metric_names = (
            "stand/contact_force_lf",
            "stand/contact_force_lr",
            "stand/contact_force_rf",
            "stand/contact_force_rr",
            "stand/contact_bool_lf",
            "stand/contact_bool_lr",
            "stand/contact_bool_rf",
            "stand/contact_bool_rr",
            "stand/contact_force_diag_diff",
            "stand/normalized_contact_force_diag_diff",
            "stand/contact_force_left_right_diff",
            "stand/contact_force_front_rear_diff",
            "stand/contact_force_range_balance",
            "stand/contact_force_total",
            "stand/contact_force_total_from_ratios_source",
            "stand/robot_total_mass",
            "stand/expected_weight_force",
            "stand/wheel_contact_force_total_raw",
            "stand/wheel_contact_force_total_raw_over_weight",
            "stand/all_body_contact_force_total_raw",
            "stand/all_body_contact_force_total_raw_over_weight",
            "stand/non_wheel_contact_force_total_raw",
            "stand/non_wheel_contact_force_total_raw_over_weight",
            "stand/contact_force_ratio_lf",
            "stand/contact_force_ratio_lr",
            "stand/contact_force_ratio_rf",
            "stand/contact_force_ratio_rr",
            "stand/contact_force_min_ratio",
            "stand/contact_force_ratio_sum",
            "stand/contact_force_ratio_valid_rate",
            "stand/min_contact_force",
            "stand/base_roll",
            "stand/base_pitch",
            "stand/base_yaw_rate",
            "stand/yaw_error_from_reset",
            "stand/base_lin_vel_x",
            "stand/base_lin_vel_y",
            "stand/base_xy_speed",
            "stand/root_height",
            "stand/root_height_reward_used",
            "stand/root_world_z",
            "stand/env_origin_z",
            "stand/terrain_height_under_robot",
            "stand/root_world_z_minus_env_origin_z",
            "stand/root_world_z_minus_terrain_height",
            "stand/base_link_world_z",
            "stand/wheel_world_z_lf",
            "stand/wheel_world_z_lr",
            "stand/wheel_world_z_rf",
            "stand/wheel_world_z_rr",
            "stand/wheel_relative_z_lf",
            "stand/wheel_relative_z_lr",
            "stand/wheel_relative_z_rf",
            "stand/wheel_relative_z_rr",
            "stand/root_height_target",
            "stand/root_height_band_score",
            "stand/root_height_error",
            "stand/root_height_abs_error",
            "stand/root_height_low",
            "stand/root_height_low_rate",
            "stand/root_height_low_raw_rate",
            "stand/root_height_high_rate",
            "stand/root_height_low_grace_active_rate",
            "stand/termination_root_height_low_after_grace_rate",
            "stand/episode_step_mean",
            "stand/episode_step_env0",
            "stand/hydraulic_stroke_lf",
            "stand/hydraulic_stroke_lr",
            "stand/hydraulic_stroke_rf",
            "stand/hydraulic_stroke_rr",
            "stand/hydraulic_action_lf",
            "stand/hydraulic_action_lr",
            "stand/hydraulic_action_rf",
            "stand/hydraulic_action_rr",
            "stand/raw_hydraulic_action_lf",
            "stand/raw_hydraulic_action_lr",
            "stand/raw_hydraulic_action_rf",
            "stand/raw_hydraulic_action_rr",
            "stand/clipped_hydraulic_action_lf",
            "stand/clipped_hydraulic_action_lr",
            "stand/clipped_hydraulic_action_rf",
            "stand/clipped_hydraulic_action_rr",
            "stand/processed_hydraulic_action_lf",
            "stand/processed_hydraulic_action_lr",
            "stand/processed_hydraulic_action_rf",
            "stand/processed_hydraulic_action_rr",
            "stand/target_stroke_lf",
            "stand/target_stroke_lr",
            "stand/target_stroke_rf",
            "stand/target_stroke_rr",
            "stand/joint_target_g_lf",
            "stand/joint_target_g_lb",
            "stand/joint_target_g_rf",
            "stand/joint_target_g_rb",
            "stand/joint_pos_g_lf",
            "stand/joint_pos_g_lb",
            "stand/joint_pos_g_rf",
            "stand/joint_pos_g_rb",
            "stand/joint_pos_minus_target_g_lf",
            "stand/joint_pos_minus_target_g_lb",
            "stand/joint_pos_minus_target_g_rf",
            "stand/joint_pos_minus_target_g_rb",
            "stand/max_abs_raw_hydraulic_action",
            "stand/max_abs_clipped_hydraulic_action",
            "stand/max_abs_joint_pos_minus_target",
            "stand/mean_abs_hydraulic_action",
            "stand/hydraulic_action_range",
            "stand/low_stroke_negative_action_penalty",
            "stand/height_gated_low_stroke_negative_action_penalty",
            "stand/low_stroke_negative_action_lf",
            "stand/low_stroke_negative_action_lr",
            "stand/low_stroke_negative_action_rf",
            "stand/low_stroke_negative_action_rr",
            "stand/height_gated_low_stroke_negative_action_lf",
            "stand/height_gated_low_stroke_negative_action_lr",
            "stand/height_gated_low_stroke_negative_action_rf",
            "stand/height_gated_low_stroke_negative_action_rr",
            "stand/low_stroke_negative_action_rate_lf",
            "stand/low_stroke_negative_action_rate_lr",
            "stand/low_stroke_negative_action_rate_rf",
            "stand/low_stroke_negative_action_rate_rr",
            "stand/low_stroke_negative_action_mask_rate_lf",
            "stand/low_stroke_negative_action_mask_rate_lr",
            "stand/low_stroke_negative_action_mask_rate_rf",
            "stand/low_stroke_negative_action_mask_rate_rr",
            "stand/low_stroke_negative_action_magnitude_lf",
            "stand/low_stroke_negative_action_magnitude_lr",
            "stand/low_stroke_negative_action_magnitude_rf",
            "stand/low_stroke_negative_action_magnitude_rr",
            "stand/height_gated_low_stroke_negative_action_rate_lf",
            "stand/height_gated_low_stroke_negative_action_rate_lr",
            "stand/height_gated_low_stroke_negative_action_rate_rf",
            "stand/height_gated_low_stroke_negative_action_rate_rr",
            "stand/high_height_low_stroke_negative_action_value",
            "stand/height_low_positive_action_rate_lf",
            "stand/height_low_positive_action_rate_lr",
            "stand/height_low_positive_action_rate_rf",
            "stand/height_low_positive_action_rate_rr",
            "stand/height_low_high_stroke_positive_action_penalty",
            "stand/height_low_high_stroke_positive_action_lf",
            "stand/height_low_high_stroke_positive_action_lr",
            "stand/height_low_high_stroke_positive_action_rf",
            "stand/height_low_high_stroke_positive_action_rr",
            "stand/high_stroke_positive_action_mask_rate_lf",
            "stand/high_stroke_positive_action_mask_rate_lr",
            "stand/high_stroke_positive_action_mask_rate_rf",
            "stand/high_stroke_positive_action_mask_rate_rr",
            "stand/high_stroke_positive_action_magnitude_lf",
            "stand/high_stroke_positive_action_magnitude_lr",
            "stand/high_stroke_positive_action_magnitude_rf",
            "stand/high_stroke_positive_action_magnitude_rr",
            "stand/stroke_away_from_nominal_action_penalty",
            "stand/stroke_away_from_nominal_action_lf",
            "stand/stroke_away_from_nominal_action_lr",
            "stand/stroke_away_from_nominal_action_rf",
            "stand/stroke_away_from_nominal_action_rr",
            "stand/stroke_toward_nominal_action_lf",
            "stand/stroke_toward_nominal_action_lr",
            "stand/stroke_toward_nominal_action_rf",
            "stand/stroke_toward_nominal_action_rr",
            "stand/front_mean_stroke",
            "stand/rear_mean_stroke",
            "stand/front_rear_stroke_diff",
            "stand/abs_front_rear_stroke_diff",
            "stand/front_rear_stroke_diff_over_threshold",
            "stand/front_rear_stroke_balance_penalty",
            "stand/front_rear_stroke_split_action_penalty",
            "stand/front_negative_hydraulic_action_mean",
            "stand/rear_positive_hydraulic_action_mean",
            "stand/front_positive_hydraulic_action_mean",
            "stand/rear_negative_hydraulic_action_mean",
            "stand/stroke_range_value",
            "stand/stroke_diagonal_balance_value",
            "stand/stroke_left_right_balance_value",
            "stand/stroke_front_rear_balance_value",
            "stand/stroke_variance_from_nominal_value",
            "stand/target_stroke_range_value",
            "stand/target_stroke_diagonal_balance_value",
            "stand/action_diagonal_split_value",
            "stand/stroke_soft_limit_margin_lf",
            "stand/stroke_soft_limit_margin_lr",
            "stand/stroke_soft_limit_margin_rf",
            "stand/stroke_soft_limit_margin_rr",
            "stand/min_stroke_soft_limit_margin",
            "stand/max_hydraulic_stroke",
            "stand/min_hydraulic_stroke",
            "stand/stroke_range",
            "stand/raw_wheel_target_lf",
            "stand/raw_wheel_target_lr",
            "stand/raw_wheel_target_rf",
            "stand/raw_wheel_target_rr",
            "stand/semantic_wheel_target_lf",
            "stand/semantic_wheel_target_lr",
            "stand/semantic_wheel_target_rf",
            "stand/semantic_wheel_target_rr",
            "stand/wheel_actual_vel_lf",
            "stand/wheel_actual_vel_lr",
            "stand/wheel_actual_vel_rf",
            "stand/wheel_actual_vel_rr",
            "stand/mean_abs_wheel_target",
            "stand/mean_abs_wheel_actual_velocity",
            "stand/env0_root_height",
            "stand/env0_root_height_high",
            "stand/env0_root_height_low_grace_active",
            "stand/root_height_last50_mean",
            "stand/root_lin_vel_z_last50_mean",
            "stand/wheel_contact_over_weight_last50_mean",
            "stand/non_wheel_contact_over_weight_last50_mean",
            "stand/env0_hydraulic_stroke_lf",
            "stand/env0_hydraulic_stroke_lr",
            "stand/env0_hydraulic_stroke_rf",
            "stand/env0_hydraulic_stroke_rr",
            "stand/env0_stroke_range_value",
            "stand/env0_stroke_diagonal_balance_value",
            "stand/env0_wheel_contact_force_lf",
            "stand/env0_wheel_contact_force_lr",
            "stand/env0_wheel_contact_force_rf",
            "stand/env0_wheel_contact_force_rr",
            "stand/env0_wheel_contact_force_total_raw",
            "stand/env0_all_body_contact_force_total_raw",
            "stand/env0_non_wheel_contact_force_total_raw",
            "stand/env0_contact_bool_lf",
            "stand/env0_contact_bool_lr",
            "stand/env0_contact_bool_rf",
            "stand/env0_contact_bool_rr",
            "stand/reset_happened_rate",
            "stand/env0_reset_happened",
            "stand/root_height_when_low",
            "stand/low_height_hydraulic_stroke_lf",
            "stand/low_height_hydraulic_stroke_lr",
            "stand/low_height_hydraulic_stroke_rf",
            "stand/low_height_hydraulic_stroke_rr",
            "stand/low_height_hydraulic_action_lf",
            "stand/low_height_hydraulic_action_lr",
            "stand/low_height_hydraulic_action_rf",
            "stand/low_height_hydraulic_action_rr",
            "stand/low_height_low_stroke_negative_action_penalty",
            "stand/low_height_height_gated_low_stroke_negative_action_penalty",
            "stand/low_height_low_stroke_negative_action_lf",
            "stand/low_height_low_stroke_negative_action_lr",
            "stand/low_height_low_stroke_negative_action_rf",
            "stand/low_height_low_stroke_negative_action_rr",
            "stand/low_height_height_gated_low_stroke_negative_action_lf",
            "stand/low_height_height_gated_low_stroke_negative_action_lr",
            "stand/low_height_height_gated_low_stroke_negative_action_rf",
            "stand/low_height_height_gated_low_stroke_negative_action_rr",
            "stand/low_height_positive_action_rate_lf",
            "stand/low_height_positive_action_rate_lr",
            "stand/low_height_positive_action_rate_rf",
            "stand/low_height_positive_action_rate_rr",
            "stand/low_height_height_low_high_stroke_positive_action_penalty",
            "stand/low_height_stroke_away_from_nominal_action_penalty",
            "stand/low_height_front_mean_stroke",
            "stand/low_height_rear_mean_stroke",
            "stand/low_height_front_rear_stroke_diff",
            "stand/low_height_abs_front_rear_stroke_diff",
            "stand/low_height_front_rear_stroke_diff_over_threshold",
            "stand/low_height_front_rear_stroke_balance_penalty",
            "stand/low_height_front_rear_stroke_split_action_penalty",
            "stand/low_height_front_negative_hydraulic_action_mean",
            "stand/low_height_rear_positive_hydraulic_action_mean",
            "stand/low_height_front_positive_hydraulic_action_mean",
            "stand/low_height_rear_negative_hydraulic_action_mean",
            "stand/low_height_contact_force_lf",
            "stand/low_height_contact_force_lr",
            "stand/low_height_contact_force_rf",
            "stand/low_height_contact_force_rr",
            "stand/low_height_contact_force_total",
            "stand/low_height_contact_force_ratio_lf",
            "stand/low_height_contact_force_ratio_lr",
            "stand/low_height_contact_force_ratio_rf",
            "stand/low_height_contact_force_ratio_rr",
            "stand/low_height_contact_force_min_ratio",
            "stand/low_height_contact_force_ratio_sum",
            "stand/low_height_base_xy_speed",
            "stand/low_height_base_roll",
            "stand/low_height_base_pitch",
            "stand/low_height_min_contact_force",
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
        short_goal_metric_names = (
            "goal_distance_mean",
            "path_length_m",
            "path_efficiency",
            "min_goal_distance_m",
            "max_distance_rebound_after_min_m",
            "turn_reversal_count",
            "progress_mean",
            "success_rate",
            "base_xy_vel_mean",
            "base_lin_vel_x",
            "base_lin_vel_y",
            "base_yaw_rate",
            "desired_speed_mean",
            "alignment_speed_gate_mean",
            "speed_profile_excess_mean",
            "speed_profile_penalty_mean",
            "turn_distance_gate_mean",
            "near_goal_rate",
            "near_goal_moving_rate",
            "stopped_success_rate",
            "wheel_target_abs_mean",
            "wheel_joint_vel_abs_mean",
            "raw_wheel_action_abs_mean",
            "raw_wheel_action_lb_mean",
            "raw_wheel_action_lf_mean",
            "raw_wheel_action_rf_mean",
            "raw_wheel_action_rb_mean",
            "wheel_target_lb_mean",
            "wheel_target_lf_mean",
            "wheel_target_rf_mean",
            "wheel_target_rb_mean",
            "semantic_wheel_target_lb_mean",
            "semantic_wheel_target_lf_mean",
            "semantic_wheel_target_rf_mean",
            "semantic_wheel_target_rb_mean",
            "wheel_joint_vel_lb_mean",
            "wheel_joint_vel_lf_mean",
            "wheel_joint_vel_rf_mean",
            "wheel_joint_vel_rb_mean",
            "semantic_wheel_joint_vel_lb_mean",
            "semantic_wheel_joint_vel_lf_mean",
            "semantic_wheel_joint_vel_rf_mean",
            "semantic_wheel_joint_vel_rb_mean",
            "policy_hydraulic_action_abs_mean",
            "hydraulic_action_abs_mean",
            "hydraulic_action_rate_abs_mean",
            "hydraulic_raw_action_abs_mean",
            "hydraulic_stroke_desired_mean",
            "hydraulic_stroke_desired_range",
            "hydraulic_stroke_actual_mean",
            "hydraulic_stroke_actual_range",
            "hydraulic_position_target_mean",
            "hydraulic_joint_position_target_mean",
            "hydraulic_joint_position_mean",
            "hydraulic_joint_velocity_abs_mean",
            "hydraulic_joint_position_tracking_error_mean",
            "hydraulic_action_lb_mean",
            "hydraulic_action_lf_mean",
            "hydraulic_action_rf_mean",
            "hydraulic_action_rb_mean",
            "stroke_desired_lb_mean",
            "stroke_desired_lf_mean",
            "stroke_desired_rf_mean",
            "stroke_desired_rb_mean",
            "stroke_actual_lb_mean",
            "stroke_actual_lf_mean",
            "stroke_actual_rf_mean",
            "stroke_actual_rb_mean",
            "short_goal_hydraulic_mode_debug",
            "support_hold_active_debug",
            "free_policy_control_active_debug",
            "hydraulic_action_diag_mode_mean",
            "hydraulic_action_diag_mode_abs_mean",
            "stroke_diag_mode_mean",
            "stroke_diag_mode_abs_mean",
            "contact_diag_force_mean",
            "contact_diag_force_abs_mean",
            "contact_diag_ratio_mean",
            "contact_diag_ratio_abs_mean",
            "action_stroke_diag_correlation",
            "action_contact_diag_correlation",
            "stroke_contact_diag_correlation",
            "support_hold_action_abs_mean",
            "support_hold_action_range",
            "support_hold_mode_debug",
            "executed_hydraulic_action_abs_mean",
            "support_hold_contact_correction_abs_mean",
            "support_hold_contact_correction_max_abs",
            "support_hold_stroke_correction_abs_mean",
            "support_hold_roll_pitch_correction_abs_mean",
            "support_hold_total_pre_filter_abs_mean",
            "support_hold_total_post_filter_abs_mean",
            "support_hold_stroke_lb_mean",
            "support_hold_stroke_lf_mean",
            "support_hold_stroke_rf_mean",
            "support_hold_stroke_rb_mean",
            "support_hold_stroke_nominal_correction_lb_mean",
            "support_hold_stroke_nominal_correction_lf_mean",
            "support_hold_stroke_nominal_correction_rf_mean",
            "support_hold_stroke_nominal_correction_rb_mean",
            "support_hold_stroke_range_correction_lb_mean",
            "support_hold_stroke_range_correction_lf_mean",
            "support_hold_stroke_range_correction_rf_mean",
            "support_hold_stroke_range_correction_rb_mean",
            "support_hold_diag_correction_lb_mean",
            "support_hold_diag_correction_lf_mean",
            "support_hold_diag_correction_rf_mean",
            "support_hold_diag_correction_rb_mean",
            "support_hold_contact_correction_lb_mean",
            "support_hold_contact_correction_lf_mean",
            "support_hold_contact_correction_rf_mean",
            "support_hold_contact_correction_rb_mean",
            "support_hold_roll_pitch_correction_lb_mean",
            "support_hold_roll_pitch_correction_lf_mean",
            "support_hold_roll_pitch_correction_rf_mean",
            "support_hold_roll_pitch_correction_rb_mean",
            "support_hold_non_contact_correction_lb_mean",
            "support_hold_non_contact_correction_lf_mean",
            "support_hold_non_contact_correction_rf_mean",
            "support_hold_non_contact_correction_rb_mean",
            "support_hold_contact_cancelled_lf_mean",
            "support_hold_contact_cancelled_rb_mean",
            "support_hold_contact_cancelled_abs_mean",
            "executed_hydraulic_action_lb_mean",
            "executed_hydraulic_action_lf_mean",
            "executed_hydraulic_action_rf_mean",
            "executed_hydraulic_action_rb_mean",
            "support_hold_total_pre_filter_lb_mean",
            "support_hold_total_pre_filter_lf_mean",
            "support_hold_total_pre_filter_rf_mean",
            "support_hold_total_pre_filter_rb_mean",
            "wrong_direction_yaw_mean",
            "forward_velocity_during_turn",
            "stroke_range_mean",
            "stroke_diagonal_balance",
            "contact_force_imbalance",
            "min_wheel_contact_force_mean",
            "zero_contact_ratio",
            "wheel_contact_force_lb_mean",
            "wheel_contact_force_lf_mean",
            "wheel_contact_force_rf_mean",
            "wheel_contact_force_rb_mean",
            "wheel_common_mode_target_abs_mean",
            "wheel_differential_target_mean",
            "wheel_differential_target_abs_mean",
            "hydraulic_mode_debug",
            "goal_x_body_mean",
            "goal_y_body_mean",
            "goal_angle_body_mean",
            "goal_angle_body_abs_mean",
            "goal_left_rate",
            "goal_right_rate",
            "goal_front_rate",
            "heading_error_abs_mean",
            "heading_error_reduction_mean",
            "turn_toward_goal_mean",
            "signed_yaw_rate_tracking_mean",
            "too_small_yaw_rate_when_error_large_mean",
            "desired_yaw_rate_mean",
            "desired_yaw_rate_abs_mean",
            "desired_yaw_sign_mean",
            "yaw_rate_error_mean",
            "yaw_rate_error_abs_mean",
            "yaw_active_rate",
            "yaw_correct_direction_rate",
            "yaw_too_small_rate",
            "left_yaw_correct_direction_rate",
            "right_yaw_correct_direction_rate",
            "left_yaw_too_small_rate",
            "right_yaw_too_small_rate",
            "left_signed_yaw_rate_mean",
            "right_signed_yaw_rate_mean",
            "wheel_velocity_target_mean",
            "semantic_wheel_velocity_target_mean",
            "wheel_joint_vel_mean",
            "semantic_wheel_joint_vel_mean",
            "wheel_left_target_mean",
            "wheel_right_target_mean",
            "left_turn_yaw_rate_mean",
            "right_turn_yaw_rate_mean",
            "left_goal_success_rate",
            "right_goal_success_rate",
            "bad_orientation_rate",
        )
        yaw_rate_command_metric_names = (
            "yaw_cmd_mean",
            "yaw_cmd_abs_mean",
            "yaw_cmd_positive_rate",
            "yaw_cmd_negative_rate",
            "yaw_cmd_sign_mode_debug",
            "yaw_rate_command_tracking_mean",
            "yaw_cmd_error_abs_mean",
            "yaw_cmd_correct_direction_rate",
            "yaw_cmd_too_small_rate",
            "positive_cmd_yaw_rate_mean",
            "negative_cmd_yaw_rate_mean",
            "positive_cmd_correct_direction_rate",
            "negative_cmd_correct_direction_rate",
            "base_yaw_rate",
            "base_lin_vel_x",
            "base_lin_vel_y",
            "base_xy_vel_mean",
            "wheel_target_abs_mean",
            "wheel_joint_vel_abs_mean",
            "raw_wheel_action_abs_mean",
            "semantic_forward_mode_target_abs_mean",
            "semantic_turn_mode_target_abs_mean",
            "wheel_forward_mode_penalty_mean",
            "wheel_turn_mode_soft_limit_mean",
            "wheel_target_abs_soft_limit_mean",
            "wheel_joint_vel_abs_soft_limit_mean",
            "wasted_turn_when_yaw_small_mean",
            "wheel_velocity_limit_debug",
            "wheel_left_target_mean",
            "wheel_right_target_mean",
            "raw_wheel_target_lb_mean",
            "raw_wheel_target_lf_mean",
            "raw_wheel_target_rf_mean",
            "raw_wheel_target_rb_mean",
            "raw_wheel_joint_vel_lb_mean",
            "raw_wheel_joint_vel_lf_mean",
            "raw_wheel_joint_vel_rf_mean",
            "raw_wheel_joint_vel_rb_mean",
            "semantic_wheel_target_lb_mean",
            "semantic_wheel_target_lf_mean",
            "semantic_wheel_target_rf_mean",
            "semantic_wheel_target_rb_mean",
            "semantic_wheel_joint_vel_lb_mean",
            "semantic_wheel_joint_vel_lf_mean",
            "semantic_wheel_joint_vel_rf_mean",
            "semantic_wheel_joint_vel_rb_mean",
            "hydraulic_mode_debug",
            "support_hold_mode_debug",
            "hydraulic_free_mode_gate_mean",
            "policy_hydraulic_action_abs_mean",
            "hydraulic_action_abs_mean",
            "hydraulic_action_rate_abs_mean",
            "stroke_range_mean",
            "stroke_diagonal_balance",
            "zero_contact_ratio",
            "contact_force_imbalance",
            "min_wheel_contact_force_mean",
            "bad_orientation_rate",
        )
        # Full diagnostic catalog retained for targeted investigations, but inactive by default.
        short_goal_metric_names_full_debug = (
            "goal_distance_mean",
            "progress_mean",
            "left_goal_progress_mean",
            "right_goal_progress_mean",
            "stopped_success_rate",
            "left_goal_stopped_success_rate",
            "right_goal_stopped_success_rate",
            "near_goal_rate",
            "near_goal_moving_rate",
            "stop_phase_active_rate",
            "stop_phase_stable_fraction",
            "base_xy_vel_mean",
            "base_yaw_rate",
            "desired_speed_mean",
            "speed_profile_excess_mean",
            "heading_error_abs_mean",
            "heading_error_signed_mean",
            "heading_error_reduction_mean",
            "desired_yaw_rate_mean",
            "desired_yaw_rate_abs_mean",
            "actual_yaw_rate_abs_mean",
            "left_goal_actual_yaw_rate_signed_mean",
            "right_goal_actual_yaw_rate_signed_mean",
            "yaw_rate_tracking_error_abs_mean",
            "yaw_rate_response_ratio_mean",
            "left_goal_yaw_rate_tracking_error_abs_mean",
            "right_goal_yaw_rate_tracking_error_abs_mean",
            "stage_far_rate",
            "stage_middle_rate",
            "stage_approach_rate",
            "stage_aligned_cruise_rate",
            "stage_far_turning_rate",
            "stage_far_other_rate",
            "stage_near_straight_rate",
            "stage_left_turn_rate",
            "stage_right_turn_rate",
            "far_heading_error_abs_mean",
            "middle_heading_error_abs_mean",
            "approach_heading_error_abs_mean",
            "far_heading_error_signed_mean",
            "middle_heading_error_signed_mean",
            "approach_heading_error_signed_mean",
            "left_goal_heading_error_signed_mean",
            "right_goal_heading_error_signed_mean",
            "far_velocity_toward_goal_mean",
            "middle_velocity_toward_goal_mean",
            "approach_velocity_toward_goal_mean",
            "far_wheel_common_mode_norm_mean",
            "middle_wheel_common_mode_norm_mean",
            "approach_wheel_common_mode_norm_mean",
            "far_wheel_turn_mode_norm_mean",
            "middle_wheel_turn_mode_norm_mean",
            "approach_wheel_turn_mode_norm_mean",
            "yaw_correct_direction_rate",
            "yaw_wrong_direction_rate",
            "yaw_too_small_rate",
            "semantic_wheel_target_lb_mean",
            "semantic_wheel_target_lf_mean",
            "semantic_wheel_target_rf_mean",
            "semantic_wheel_target_rb_mean",
            "semantic_wheel_joint_vel_lb_mean",
            "semantic_wheel_joint_vel_lf_mean",
            "semantic_wheel_joint_vel_rf_mean",
            "semantic_wheel_joint_vel_rb_mean",
            "wheel_slip_ratio_lb_mean",
            "wheel_slip_ratio_lf_mean",
            "wheel_slip_ratio_rf_mean",
            "wheel_slip_ratio_rb_mean",
            "wheel_slip_speed_lb_mean_mps",
            "wheel_slip_speed_lf_mean_mps",
            "wheel_slip_speed_rf_mean_mps",
            "wheel_slip_speed_rb_mean_mps",
            "wheel_surface_speed_abs_mean_mps",
            "wheel_hub_forward_speed_abs_mean_mps",
            "traction_speed_efficiency_mean",
            "left_goal_traction_speed_efficiency_mean",
            "right_goal_traction_speed_efficiency_mean",
            "left_goal_wheel_slip_mean",
            "right_goal_wheel_slip_mean",
            "far_wheel_slip_mean",
            "middle_wheel_slip_mean",
            "approach_wheel_slip_mean",
            "far_wheel_slip_speed_mean_mps",
            "middle_wheel_slip_speed_mean_mps",
            "approach_wheel_slip_speed_mean_mps",
            "far_wheel_slip_ratio_motion_gated_mean",
            "middle_wheel_slip_ratio_motion_gated_mean",
            "approach_wheel_slip_ratio_motion_gated_mean",
            "far_traction_overspeed_excess_mean_mps",
            "middle_traction_overspeed_excess_mean_mps",
            "approach_traction_overspeed_excess_mean_mps",
            "aligned_cruise_wheel_surface_speed_abs_mean_mps",
            "aligned_cruise_wheel_hub_forward_speed_abs_mean_mps",
            "aligned_cruise_traction_drive_efficiency_mean",
            "aligned_cruise_wheel_slip_ratio_motion_gated_mean",
            "aligned_cruise_wheel_slip_speed_mean_mps",
            "aligned_cruise_traction_overspeed_excess_mean_mps",
            "far_turning_wheel_surface_speed_abs_mean_mps",
            "far_turning_wheel_hub_forward_speed_abs_mean_mps",
            "far_turning_traction_drive_efficiency_mean",
            "far_turning_wheel_slip_ratio_motion_gated_mean",
            "far_turning_wheel_slip_speed_mean_mps",
            "far_turning_traction_overspeed_excess_mean_mps",
            "near_straight_traction_drive_efficiency_mean",
            "near_straight_wheel_slip_ratio_motion_gated_mean",
            "near_straight_wheel_slip_speed_mean_mps",
            "near_straight_traction_overspeed_excess_mean_mps",
            "left_turn_wheel_turn_mode_toward_goal_mean",
            "right_turn_wheel_turn_mode_toward_goal_mean",
            "left_turn_actual_yaw_rate_toward_goal_mean",
            "right_turn_actual_yaw_rate_toward_goal_mean",
            "left_turn_yaw_effectiveness_mean",
            "right_turn_yaw_effectiveness_mean",
            "left_turn_inner_left_wheel_slip_ratio_motion_gated_mean",
            "left_turn_outer_right_wheel_slip_ratio_motion_gated_mean",
            "right_turn_outer_left_wheel_slip_ratio_motion_gated_mean",
            "right_turn_inner_right_wheel_slip_ratio_motion_gated_mean",
            "left_turn_inner_left_wheel_slip_speed_mean_mps",
            "left_turn_outer_right_wheel_slip_speed_mean_mps",
            "right_turn_outer_left_wheel_slip_speed_mean_mps",
            "right_turn_inner_right_wheel_slip_speed_mean_mps",
            "left_turn_inner_left_traction_overspeed_excess_mean_mps",
            "left_turn_outer_right_traction_overspeed_excess_mean_mps",
            "right_turn_outer_left_traction_overspeed_excess_mean_mps",
            "right_turn_inner_right_traction_overspeed_excess_mean_mps",
            "left_turn_inner_left_traction_drive_efficiency_mean",
            "left_turn_outer_right_traction_drive_efficiency_mean",
            "right_turn_outer_left_traction_drive_efficiency_mean",
            "right_turn_inner_right_traction_drive_efficiency_mean",
            "wheel_target_tracking_error_lb_mean",
            "wheel_target_tracking_error_lf_mean",
            "wheel_target_tracking_error_rf_mean",
            "wheel_target_tracking_error_rb_mean",
            "wheel_common_mode_target_norm_mean",
            "wheel_turn_mode_target_norm_mean",
            "left_goal_wheel_common_mode_norm_mean",
            "right_goal_wheel_common_mode_norm_mean",
            "left_goal_wheel_turn_mode_norm_mean",
            "right_goal_wheel_turn_mode_norm_mean",
            "left_goal_velocity_toward_goal_mean",
            "right_goal_velocity_toward_goal_mean",
            "wheel_left_front_rear_target_diff_norm_mean",
            "wheel_right_front_rear_target_diff_norm_mean",
            "wheel_left_front_rear_opposite_sign_rate",
            "wheel_right_front_rear_opposite_sign_rate",
            "wheel_left_front_rear_antisymmetric_mode_mean",
            "wheel_right_front_rear_antisymmetric_mode_mean",
            "far_antisymmetric_mode_mean",
            "middle_antisymmetric_mode_mean",
            "approach_antisymmetric_mode_mean",
            "far_opposite_sign_rate",
            "middle_opposite_sign_rate",
            "approach_opposite_sign_rate",
            "wheel_target_rate_norm_mean",
            "wheel_action_saturation_rate",
            "root_height_mean",
            "roll_abs_mean",
            "pitch_abs_mean",
            "hydraulic_action_abs_mean",
            "hydraulic_action_rate_abs_mean",
            "stroke_actual_lb_mean",
            "stroke_actual_lf_mean",
            "stroke_actual_rf_mean",
            "stroke_actual_rb_mean",
            "stroke_actual_range",
            "stroke_max_mean",
            "stroke_above_0_55_rate",
            "stroke_above_0_60_rate",
            "wheel_contact_ratio_lb_mean",
            "wheel_contact_ratio_lf_mean",
            "wheel_contact_ratio_rf_mean",
            "wheel_contact_ratio_rb_mean",
            "wheel_contact_min_ratio_mean",
            "all_wheel_contact_rate",
            "non_wheel_contact_rate",
            "bad_orientation_rate",
        )

        # D-stage active metrics: compact curriculum, asymmetry, traction, and posture checks.
        short_goal_metric_names = (
            "goal_distance_mean",
            "path_length_m",
            "path_efficiency",
            "min_goal_distance_m",
            "max_distance_rebound_after_min_m",
            "turn_reversal_count",
            "progress_mean",
            "left_goal_progress_mean",
            "right_goal_progress_mean",
            "stopped_success_rate",
            "left_goal_stopped_success_rate",
            "right_goal_stopped_success_rate",
            "near_goal_moving_rate",
            "stop_phase_active_rate",
            "stop_phase_stable_fraction",
            "stop_posture_roll_signed_mean",
            "stop_posture_pitch_signed_mean",
            "stop_posture_stroke_lb_mean",
            "stop_posture_stroke_lf_mean",
            "stop_posture_stroke_rf_mean",
            "stop_posture_stroke_rb_mean",
            "stop_posture_stroke_range_mean",
            "stop_posture_stroke_tracking_error_max_mean",
            "stop_posture_ready_rate",
            "heading_error_abs_mean",
            "heading_error_signed_mean",
            "desired_yaw_rate_abs_mean",
            "actual_yaw_rate_abs_mean",
            "yaw_rate_tracking_error_abs_mean",
            "yaw_rate_response_ratio_mean",
            "stage_far_rate",
            "stage_middle_rate",
            "stage_approach_rate",
            "far_heading_error_abs_mean",
            "middle_heading_error_abs_mean",
            "approach_heading_error_abs_mean",
            "left_goal_heading_error_signed_mean",
            "right_goal_heading_error_signed_mean",
            "far_velocity_toward_goal_mean",
            "middle_velocity_toward_goal_mean",
            "approach_velocity_toward_goal_mean",
            "far_wheel_common_mode_norm_mean",
            "middle_wheel_common_mode_norm_mean",
            "approach_wheel_common_mode_norm_mean",
            "far_wheel_turn_mode_norm_mean",
            "middle_wheel_turn_mode_norm_mean",
            "approach_wheel_turn_mode_norm_mean",
            "yaw_correct_direction_rate",
            "yaw_wrong_direction_rate",
            "left_goal_wheel_turn_mode_norm_mean",
            "right_goal_wheel_turn_mode_norm_mean",
            "left_goal_velocity_toward_goal_mean",
            "right_goal_velocity_toward_goal_mean",
            "left_goal_wheel_slip_mean",
            "right_goal_wheel_slip_mean",
            "far_wheel_slip_speed_mean_mps",
            "far_wheel_slip_ratio_motion_gated_mean",
            "far_traction_overspeed_excess_mean_mps",
            "wheel_target_rate_norm_mean",
            "wheel_action_saturation_rate",
            "policy_hydraulic_action_abs_mean",
            "hydraulic_action_abs_mean",
            "hydraulic_action_rate_abs_mean",
            "hydraulic_raw_action_abs_mean",
            "hydraulic_action_lb_mean",
            "hydraulic_action_lf_mean",
            "hydraulic_action_rf_mean",
            "hydraulic_action_rb_mean",
            "hydraulic_stroke_desired_mean",
            "hydraulic_stroke_desired_range",
            "hydraulic_stroke_actual_mean",
            "hydraulic_stroke_actual_range",
            "stroke_desired_lb_mean",
            "stroke_desired_lf_mean",
            "stroke_desired_rf_mean",
            "stroke_desired_rb_mean",
            "stroke_actual_lb_mean",
            "stroke_actual_lf_mean",
            "stroke_actual_rf_mean",
            "stroke_actual_rb_mean",
            "hydraulic_action_diag_mode_abs_mean",
            "stroke_diag_mode_abs_mean",
            "contact_diag_ratio_abs_mean",
            "root_height_mean",
            "roll_abs_mean",
            "pitch_abs_mean",
            "stroke_actual_range",
            "stroke_max_mean",
            "wheel_contact_min_ratio_mean",
            "all_wheel_contact_rate",
            "non_wheel_contact_rate",
            "bad_orientation_rate",
        )

        if self._is_speed_command_task:
            self._forward_debug_metric_names = speed_command_metric_names
            self._debug_log_prefix = "forward_debug"
        elif self._is_turn_to_target_task:
            self._forward_debug_metric_names = turn_to_target_metric_names
            self._debug_log_prefix = "turn_to_target_debug"
        elif self._is_stand_training_task:
            self._forward_debug_metric_names = stand_metric_names
            self._debug_log_prefix = "stand_debug"
        elif self._is_yaw_rate_command_task:
            self._forward_debug_metric_names = yaw_rate_command_metric_names
            self._debug_log_prefix = "yaw_rate_command"
        elif self._is_short_goal_task:
            self._forward_debug_metric_names = short_goal_metric_names
            self._debug_log_prefix = "short_goal"
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
                policy_obs = obs_buf.get("policy_state", None)
            elif isinstance(obs_buf, torch.Tensor):
                policy_obs = obs_buf
            if (
                policy_obs is not None
                and policy_obs.ndim == 2
                and policy_obs.shape[1] >= self._command_obs_end
            ):
                policy_cmd_obs = policy_obs[:, self._command_obs_start : self._command_obs_end]
            else:
                policy_cmd_obs = torch.zeros((self.num_envs, 8), device=self.device, dtype=torch.float32)
            policy_obs_cmd_v_x_feature = policy_cmd_obs[:, 0]
            policy_obs_cmd_v_x_target_feature = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            policy_obs_cmd_time_left_feature = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
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

        if self._is_turn_to_target_task:
            target_vec_b, distance_to_target, heading_error = mdp.goal_heading_target_body(self)
            heading_error_abs = torch.abs(heading_error)
            wheel_contact_sensor = self.scene.sensors["wheel_contact_forces"]
            net_contact_forces = wheel_contact_sensor.data.net_forces_w_history[:, :, self._wheel_contact_body_ids, :]
            contact_force = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
            contacts = contact_force > 1.0
            wheel_torque = wheel_action_term.torque_actual
            stroke_actual = leg_action_term.stroke_actual
            roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
            root_height = robot.data.root_pos_w[:, 2]
            unloaded_spin_term = getattr(self.cfg.rewards, "turn_unloaded_wheel_spin", None)
            unloaded_force_threshold = (
                float(unloaded_spin_term.params.get("force_threshold", 20.0))
                if unloaded_spin_term is not None
                else 20.0
            )
            unloaded_mask = (contact_force < unloaded_force_threshold).to(torch.float32)
            wheel_vel_shape_dim = torch.full(
                (self.num_envs,),
                float(wheel_joint_vel.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            unloaded_mask_shape_dim = torch.full(
                (self.num_envs,),
                float(unloaded_mask.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            semantic_wheel_velocity_target = wheel_velocity_target * self._wheel_forward_sign
            semantic_wheel_joint_vel = wheel_joint_vel * self._wheel_forward_sign
            left_wheel_vel_mean = semantic_wheel_joint_vel[:, :2].mean(dim=1)
            right_wheel_vel_mean = semantic_wheel_joint_vel[:, 2:].mean(dim=1)
            wheel_vel_diff = right_wheel_vel_mean - left_wheel_vel_mean
            target_left_front_rear_diff = torch.abs(wheel_velocity_target[:, 1] - wheel_velocity_target[:, 0])
            target_right_front_rear_diff = torch.abs(wheel_velocity_target[:, 2] - wheel_velocity_target[:, 3])
            actual_left_front_rear_diff = torch.abs(wheel_joint_vel[:, 1] - wheel_joint_vel[:, 0])
            actual_right_front_rear_diff = torch.abs(wheel_joint_vel[:, 2] - wheel_joint_vel[:, 3])
            contact_force_diag_diff = (contact_force[:, 1] + contact_force[:, 3]) - (contact_force[:, 0] + contact_force[:, 2])
            contact_force_front_rear_diff = (contact_force[:, 1] + contact_force[:, 2]) - (
                contact_force[:, 0] + contact_force[:, 3]
            )
            contact_force_left_right_diff = (contact_force[:, 0] + contact_force[:, 1]) - (
                contact_force[:, 2] + contact_force[:, 3]
            )
            prev_heading_error_abs = self._goal_heading_debug_prev_abs_error
            turn_progress = torch.where(
                torch.isfinite(prev_heading_error_abs),
                prev_heading_error_abs - heading_error_abs,
                torch.zeros_like(heading_error_abs),
            )
            self._goal_heading_debug_prev_abs_error[:] = heading_error_abs

            heading_alignment_term = self.cfg.rewards.turn_heading_alignment
            turn_progress_term = self.cfg.rewards.turn_heading_progress
            yaw_tracking_term = self.cfg.rewards.turn_yaw_tracking
            reward_heading_alignment = heading_alignment_term.func(self, **heading_alignment_term.params)
            reward_turn_progress = turn_progress_term.func(self, **turn_progress_term.params)
            reward_yaw_tracking = yaw_tracking_term.func(self, **yaw_tracking_term.params)
            same_side_diff_term = self.cfg.rewards.turn_same_side_front_rear_diff
            reward_same_side_front_rear_penalty = same_side_diff_term.func(self, **same_side_diff_term.params)
            reward_unloaded_wheel_spin_penalty = (
                unloaded_spin_term.func(self, **unloaded_spin_term.params)
                if unloaded_spin_term is not None
                else torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            )
            wheel_target_overspeed_term = getattr(self.cfg.rewards, "turn_wheel_target_overspeed", None)
            reward_wheel_target_overspeed_penalty = (
                wheel_target_overspeed_term.func(self, **wheel_target_overspeed_term.params)
                if wheel_target_overspeed_term is not None
                else torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            )
            wheel_target_common_mode_term = getattr(self.cfg.rewards, "turn_wheel_target_common_mode", None)
            reward_wheel_target_common_mode_penalty = (
                wheel_target_common_mode_term.func(self, **wheel_target_common_mode_term.params)
                if wheel_target_common_mode_term is not None
                else torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            )
            yaw_rate_cmd = mdp.turn_to_target_yaw_rate_command(
                self,
                k_yaw=yaw_tracking_term.params.get("k_yaw", 1.5),
                yaw_rate_max=yaw_tracking_term.params.get("yaw_rate_max", 0.8),
                asset_cfg=yaw_tracking_term.params.get("asset_cfg"),
            )
            heading_sign = torch.sign(heading_error)
            raw_target_left_mean = 0.5 * (wheel_velocity_target[:, 0] + wheel_velocity_target[:, 1])
            raw_target_right_mean = 0.5 * (wheel_velocity_target[:, 2] + wheel_velocity_target[:, 3])
            raw_target_diff = raw_target_right_mean - raw_target_left_mean
            raw_target_common = 0.5 * (raw_target_left_mean + raw_target_right_mean)
            semantic_target_left_mean = 0.5 * (
                semantic_wheel_velocity_target[:, 0] + semantic_wheel_velocity_target[:, 1]
            )
            semantic_target_right_mean = 0.5 * (
                semantic_wheel_velocity_target[:, 2] + semantic_wheel_velocity_target[:, 3]
            )
            semantic_target_diff = semantic_target_right_mean - semantic_target_left_mean
            semantic_target_common = 0.5 * (semantic_target_left_mean + semantic_target_right_mean)
            semantic_target_diff_times_yaw_cmd = semantic_target_diff * yaw_rate_cmd
            semantic_target_diff_times_yaw_rate = semantic_target_diff * root_ang_vel_b_z
            yaw_cmd_times_yaw_rate = yaw_rate_cmd * root_ang_vel_b_z
            heading_sign_times_yaw_rate = heading_sign * root_ang_vel_b_z
            yaw_rate_abs_error = torch.abs(root_ang_vel_b_z - yaw_rate_cmd)
            wheel_diff_times_yaw_rate = wheel_vel_diff * root_ang_vel_b_z
            reset_settling = self._reset_settle_step_active.to(torch.float32)
            reset_settle_steps_remaining = self._reset_settle_remaining_steps.to(torch.float32)

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

            values = {
                "turn/heading_error": heading_error,
                "turn/abs_heading_error": heading_error_abs,
                "turn/turn_progress": turn_progress,
                "turn/yaw_rate_cmd": yaw_rate_cmd,
                "turn/base_ang_vel_z": root_ang_vel_b_z,
                "turn/goal_x_body": target_vec_b[:, 0],
                "turn/goal_y_body": target_vec_b[:, 1],
                "turn/goal_distance": distance_to_target,
                "turn/target_lf": wheel_velocity_target[:, 1],
                "turn/target_lr": wheel_velocity_target[:, 0],
                "turn/target_rf": wheel_velocity_target[:, 2],
                "turn/target_rr": wheel_velocity_target[:, 3],
                "turn/actual_lf": wheel_joint_vel[:, 1],
                "turn/actual_lr": wheel_joint_vel[:, 0],
                "turn/actual_rf": wheel_joint_vel[:, 2],
                "turn/actual_rr": wheel_joint_vel[:, 3],
                "turn/torque_lf": wheel_torque[:, 1],
                "turn/torque_lr": wheel_torque[:, 0],
                "turn/torque_rf": wheel_torque[:, 2],
                "turn/torque_rr": wheel_torque[:, 3],
                "turn/contact_force_lf": contact_force[:, 1],
                "turn/contact_force_lr": contact_force[:, 0],
                "turn/contact_force_rf": contact_force[:, 2],
                "turn/contact_force_rr": contact_force[:, 3],
                "turn/contact_bool_lf": contacts[:, 1].to(torch.float32),
                "turn/contact_bool_lr": contacts[:, 0].to(torch.float32),
                "turn/contact_bool_rf": contacts[:, 2].to(torch.float32),
                "turn/contact_bool_rr": contacts[:, 3].to(torch.float32),
                "turn/contact_force_diag_diff": contact_force_diag_diff,
                "turn/contact_force_front_rear_diff": contact_force_front_rear_diff,
                "turn/contact_force_left_right_diff": contact_force_left_right_diff,
                "turn/base_roll": roll,
                "turn/base_pitch": pitch,
                "turn/root_height": root_height,
                "turn/hydraulic_stroke_lf": stroke_actual[:, 1],
                "turn/hydraulic_stroke_lr": stroke_actual[:, 0],
                "turn/hydraulic_stroke_rf": stroke_actual[:, 2],
                "turn/hydraulic_stroke_rr": stroke_actual[:, 3],
                "turn/reset_settling": reset_settling,
                "turn/reset_settle_steps_remaining": reset_settle_steps_remaining,
                "turn/wheel_vel_shape_dim": wheel_vel_shape_dim,
                "turn/unloaded_mask_shape_dim": unloaded_mask_shape_dim,
                "turn/target_left_front_rear_diff": target_left_front_rear_diff,
                "turn/target_right_front_rear_diff": target_right_front_rear_diff,
                "turn/raw_target_left_mean": raw_target_left_mean,
                "turn/raw_target_right_mean": raw_target_right_mean,
                "turn/raw_target_diff": raw_target_diff,
                "turn/raw_target_common": raw_target_common,
                "turn/semantic_target_left_mean": semantic_target_left_mean,
                "turn/semantic_target_right_mean": semantic_target_right_mean,
                "turn/semantic_target_diff": semantic_target_diff,
                "turn/semantic_target_common": semantic_target_common,
                "turn/actual_left_front_rear_diff": actual_left_front_rear_diff,
                "turn/actual_right_front_rear_diff": actual_right_front_rear_diff,
                "turn/left_wheel_vel_mean": left_wheel_vel_mean,
                "turn/right_wheel_vel_mean": right_wheel_vel_mean,
                "turn/wheel_vel_diff": wheel_vel_diff,
                "turn/semantic_target_diff_times_yaw_cmd": semantic_target_diff_times_yaw_cmd,
                "turn/semantic_target_diff_times_yaw_rate": semantic_target_diff_times_yaw_rate,
                "turn/yaw_cmd_times_yaw_rate": yaw_cmd_times_yaw_rate,
                "turn/heading_sign_times_yaw_rate": heading_sign_times_yaw_rate,
                "turn/yaw_rate_abs_error": yaw_rate_abs_error,
                "turn/wheel_diff_times_yaw_rate": wheel_diff_times_yaw_rate,
                "turn/reward_heading_alignment": reward_heading_alignment,
                "turn/reward_turn_progress": reward_turn_progress,
                "turn/reward_yaw_tracking": reward_yaw_tracking,
                "turn/reward_same_side_front_rear_penalty": reward_same_side_front_rear_penalty,
                "turn/reward_unloaded_wheel_spin_penalty": reward_unloaded_wheel_spin_penalty,
                "turn/reward_wheel_target_overspeed_penalty": reward_wheel_target_overspeed_penalty,
                "turn/reward_wheel_target_common_mode_penalty": reward_wheel_target_common_mode_penalty,
                "turn/termination_any": termination_any,
                "turn/termination_time_out": termination_time_out,
                "turn/termination_bad_orientation": termination_bad_orientation,
                "turn/termination_root_height_low": termination_root_height_low,
                "turn/termination_other": termination_other_bool.to(torch.float32),
                "turn/episode_length": self.episode_length_buf.to(torch.float32),
            }
            for term_name, term_value in termination_term_values.items():
                values[f"turn/termination_term/{term_name}"] = term_value
            return values

        if self._is_stand_training_task:
            wheel_contact_sensor = self.scene.sensors["wheel_contact_forces"]
            net_contact_forces = wheel_contact_sensor.data.net_forces_w_history[:, :, self._wheel_contact_body_ids, :]
            contact_force = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
            contacts = contact_force > 1.0
            all_body_contact_sensor = self.scene.sensors["all_body_contact_forces"]
            all_body_net_contact_forces = all_body_contact_sensor.data.net_forces_w_history[
                :, :, self._all_body_contact_body_ids, :
            ]
            all_body_contact_force = torch.max(torch.norm(all_body_net_contact_forces, dim=-1), dim=1)[0]
            all_body_contact_force_total_raw = all_body_contact_force.sum(dim=1)
            if self._non_wheel_contact_body_ids.numel() > 0:
                non_wheel_net_contact_forces = all_body_contact_sensor.data.net_forces_w_history[
                    :, :, self._non_wheel_contact_body_ids, :
                ]
                non_wheel_contact_force = torch.max(torch.norm(non_wheel_net_contact_forces, dim=-1), dim=1)[0]
                non_wheel_contact_force_total_raw = non_wheel_contact_force.sum(dim=1)
            else:
                non_wheel_contact_force_total_raw = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            ratio_force_lf_lr_rf_rr, total_force_raw, contact_force_ratio_lf_lr_rf_rr, valid_contact_total_mask = (
                _wheel_contact_force_ratio_lf_lr_rf_rr(
                    self,
                    SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
                )
            )
            stroke_actual = leg_action_term.stroke_actual
            hydraulic_raw_action = leg_action_term.raw_actions
            hydraulic_clipped_action = leg_action_term.clipped_actions
            hydraulic_processed_action = leg_action_term.processed_actions
            target_stroke = leg_action_term.stroke_desired
            roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
            _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
            root_height = robot.data.root_pos_w[:, 2]
            root_height_target = float(self.cfg.rewards.root_height_tracking.params.get("target_height", 0.75))
            root_height_low_limit = float(self.cfg.terminations.root_height_low.params["minimum_height"])
            root_height_low_grace_time_s = float(self.cfg.terminations.root_height_low.params.get("grace_time_s", 0.0))
            root_height_low_grace_steps = max(
                int(math.ceil(root_height_low_grace_time_s / max(float(self.step_dt), 1.0e-6))),
                0,
            )
            stroke_soft_limit = float(self.cfg.rewards.actual_stroke_soft_limit.params.get("limit", 0.45))
            expected_weight_force = torch.full(
                (self.num_envs,),
                self._stand_debug_expected_weight_force,
                device=self.device,
                dtype=torch.float32,
            )
            robot_total_mass = torch.full(
                (self.num_envs,),
                self._stand_debug_robot_total_mass,
                device=self.device,
                dtype=torch.float32,
            )
            yaw_error_from_reset = torch.atan2(
                torch.sin(yaw - self._stand_reset_yaw),
                torch.cos(yaw - self._stand_reset_yaw),
            )
            contact_force_diag_diff = (
                ratio_force_lf_lr_rf_rr[:, 0] + ratio_force_lf_lr_rf_rr[:, 3]
            ) - (ratio_force_lf_lr_rf_rr[:, 1] + ratio_force_lf_lr_rf_rr[:, 2])
            safe_total_force = torch.clamp(total_force_raw, min=1.0e-6)
            normalized_contact_force_diag_diff = torch.abs(contact_force_diag_diff) / safe_total_force
            contact_force_left_right_diff = (ratio_force_lf_lr_rf_rr[:, 0] + ratio_force_lf_lr_rf_rr[:, 1]) - (
                ratio_force_lf_lr_rf_rr[:, 2] + ratio_force_lf_lr_rf_rr[:, 3]
            )
            contact_force_front_rear_diff = (ratio_force_lf_lr_rf_rr[:, 0] + ratio_force_lf_lr_rf_rr[:, 2]) - (
                ratio_force_lf_lr_rf_rr[:, 1] + ratio_force_lf_lr_rf_rr[:, 3]
            )
            contact_force_range_balance = (
                ratio_force_lf_lr_rf_rr.max(dim=1).values - ratio_force_lf_lr_rf_rr.min(dim=1).values
            ) / safe_total_force
            min_contact_force = ratio_force_lf_lr_rf_rr.min(dim=1).values
            min_contact_force_ratio = contact_force_ratio_lf_lr_rf_rr.min(dim=1).values
            contact_force_ratio_sum = contact_force_ratio_lf_lr_rf_rr.sum(dim=1)
            wheel_contact_force_total_raw_over_weight = total_force_raw / torch.clamp(expected_weight_force, min=1.0e-6)
            all_body_contact_force_total_raw_over_weight = all_body_contact_force_total_raw / torch.clamp(
                expected_weight_force, min=1.0e-6
            )
            non_wheel_contact_force_total_raw_over_weight = non_wheel_contact_force_total_raw / torch.clamp(
                expected_weight_force, min=1.0e-6
            )
            insert_idx = self._stand_last50_index
            env_arange = torch.arange(self.num_envs, device=self.device)
            self._stand_last50_root_height[env_arange, insert_idx] = root_height
            self._stand_last50_root_lin_vel_z[env_arange, insert_idx] = robot.data.root_lin_vel_b[:, 2]
            self._stand_last50_wheel_contact_over_weight[env_arange, insert_idx] = wheel_contact_force_total_raw_over_weight
            self._stand_last50_non_wheel_contact_over_weight[env_arange, insert_idx] = (
                non_wheel_contact_force_total_raw_over_weight
            )
            self._stand_last50_index = (self._stand_last50_index + 1) % self._stand_last50_length
            self._stand_last50_count = torch.clamp(self._stand_last50_count + 1, max=self._stand_last50_length)
            last50_valid = (
                torch.arange(self._stand_last50_length, device=self.device).unsqueeze(0)
                < self._stand_last50_count.unsqueeze(1)
            ).to(torch.float32)
            last50_denom = torch.clamp(self._stand_last50_count.to(torch.float32), min=1.0)
            root_height_last50_mean = (self._stand_last50_root_height * last50_valid).sum(dim=1) / last50_denom
            root_lin_vel_z_last50_mean = (self._stand_last50_root_lin_vel_z * last50_valid).sum(dim=1) / last50_denom
            wheel_contact_over_weight_last50_mean = (
                (self._stand_last50_wheel_contact_over_weight * last50_valid).sum(dim=1) / last50_denom
            )
            non_wheel_contact_over_weight_last50_mean = (
                (self._stand_last50_non_wheel_contact_over_weight * last50_valid).sum(dim=1) / last50_denom
            )
            root_height_error = root_height - root_height_target
            episode_steps = self.episode_length_buf.to(torch.float32)
            root_height_low_raw = (root_height < root_height_low_limit).to(torch.float32)
            root_height_high_rate = (root_height > 0.80).to(torch.float32)
            root_height_low_grace_active = (
                self.episode_length_buf < root_height_low_grace_steps
            ).to(torch.float32)
            termination_root_height_low_after_grace = root_height_low_raw * (1.0 - root_height_low_grace_active)
            root_height_band_score = mdp.root_height_band_piecewise(
                self,
                peak_min=float(self.cfg.rewards.root_height_band.params.get("peak_min", 0.74)),
                peak_max=float(self.cfg.rewards.root_height_band.params.get("peak_max", 0.76)),
                healthy_min=0.70,
                healthy_max=0.80,
                low_floor=root_height_low_limit,
                high_penalty_scale=6.0,
                asset_cfg=SceneEntityCfg("robot"),
            )
            base_lin_vel_y = robot.data.root_lin_vel_b[:, 1]
            base_xy_speed = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
            raw_wheel_target = wheel_velocity_target[:, [1, 0, 2, 3]]
            semantic_wheel_target = semantic_wheel_velocity_target[:, [1, 0, 2, 3]]
            wheel_actual_vel = wheel_joint_vel[:, [1, 0, 2, 3]]
            wheel_world_pos_lf_lr_rf_rr = robot.data.body_pos_w[:, self._wheel_robot_body_ids_lf_lr_rf_rr, :]
            wheel_relative_pos_lf_lr_rf_rr = wheel_world_pos_lf_lr_rf_rr - robot.data.root_pos_w.unsqueeze(1)
            base_link_world_z = robot.data.body_pos_w[:, self._base_link_body_ids[0], 2]
            env_origin_z = self.scene.env_origins[:, 2]
            root_world_z = robot.data.root_pos_w[:, 2]
            stroke_soft_limit_margin = stroke_soft_limit - stroke_actual[:, [1, 0, 2, 3]]
            stroke_max = stroke_actual.max(dim=1).values
            stroke_min = stroke_actual.min(dim=1).values
            stroke_range = stroke_max - stroke_min
            hydraulic_action_lf_lr_rf_rr = hydraulic_raw_action[:, [1, 0, 2, 3]]
            hydraulic_clipped_action_lf_lr_rf_rr = hydraulic_clipped_action[:, [1, 0, 2, 3]]
            hydraulic_processed_action_lf_lr_rf_rr = hydraulic_processed_action[:, [1, 0, 2, 3]]
            target_stroke_lf_lr_rf_rr = target_stroke[:, [1, 0, 2, 3]]
            stroke_actual_lf_lr_rf_rr = stroke_actual[:, [1, 0, 2, 3]]
            stroke_range_value = stroke_actual_lf_lr_rf_rr.max(dim=1).values - stroke_actual_lf_lr_rf_rr.min(dim=1).values
            stroke_diagonal_balance_value = torch.abs(
                (stroke_actual_lf_lr_rf_rr[:, 0] + stroke_actual_lf_lr_rf_rr[:, 3])
                - (stroke_actual_lf_lr_rf_rr[:, 1] + stroke_actual_lf_lr_rf_rr[:, 2])
            )
            stroke_left_right_balance_value = torch.abs(
                (stroke_actual_lf_lr_rf_rr[:, 0] + stroke_actual_lf_lr_rf_rr[:, 1])
                - (stroke_actual_lf_lr_rf_rr[:, 2] + stroke_actual_lf_lr_rf_rr[:, 3])
            )
            stroke_front_rear_balance_value = torch.abs(
                (stroke_actual_lf_lr_rf_rr[:, 0] + stroke_actual_lf_lr_rf_rr[:, 2])
                - (stroke_actual_lf_lr_rf_rr[:, 1] + stroke_actual_lf_lr_rf_rr[:, 3])
            )
            stroke_variance_from_nominal_value = torch.mean(torch.square(stroke_actual_lf_lr_rf_rr - 0.5), dim=1)
            target_stroke_range_value = (
                target_stroke_lf_lr_rf_rr.max(dim=1).values - target_stroke_lf_lr_rf_rr.min(dim=1).values
            )
            target_stroke_diagonal_balance_value = torch.abs(
                (target_stroke_lf_lr_rf_rr[:, 0] + target_stroke_lf_lr_rf_rr[:, 3])
                - (target_stroke_lf_lr_rf_rr[:, 1] + target_stroke_lf_lr_rf_rr[:, 2])
            )
            action_diagonal_split_value = torch.abs(
                (hydraulic_action_lf_lr_rf_rr[:, 0] + hydraulic_action_lf_lr_rf_rr[:, 3])
                - (hydraulic_action_lf_lr_rf_rr[:, 1] + hydraulic_action_lf_lr_rf_rr[:, 2])
            )
            joint_pos_g_lb_lf_rf_rb = robot.data.joint_pos[:, self._leg_joint_ids]
            joint_target_g_lb_lf_rf_rb = robot.data.joint_pos_target[:, self._leg_joint_ids]
            joint_pos_minus_target_g_lb_lf_rf_rb = joint_pos_g_lb_lf_rf_rb - joint_target_g_lb_lf_rf_rb
            max_abs_raw_hydraulic_action = torch.max(torch.abs(hydraulic_raw_action), dim=1).values
            max_abs_clipped_hydraulic_action = torch.max(torch.abs(hydraulic_clipped_action), dim=1).values
            max_abs_joint_pos_minus_target = torch.max(torch.abs(joint_pos_minus_target_g_lb_lf_rf_rb), dim=1).values
            env0_root_height = root_height[0].expand(self.num_envs)
            env0_root_height_high = root_height_high_rate[0].expand(self.num_envs)
            env0_root_height_low_grace_active = root_height_low_grace_active[0].expand(self.num_envs)
            env0_stroke_lf_lr_rf_rr = stroke_actual_lf_lr_rf_rr[0].unsqueeze(0).expand(self.num_envs, -1)
            env0_stroke_range_value = stroke_range_value[0].expand(self.num_envs)
            env0_stroke_diagonal_balance_value = stroke_diagonal_balance_value[0].expand(self.num_envs)
            env0_contact_force_lf_lr_rf_rr = ratio_force_lf_lr_rf_rr[0].unsqueeze(0).expand(self.num_envs, -1)
            env0_contact_bool_lf_lr_rf_rr = torch.stack(
                (contacts[:, 1], contacts[:, 0], contacts[:, 2], contacts[:, 3]),
                dim=1,
            )[0].to(torch.float32).unsqueeze(0).expand(self.num_envs, -1)
            env0_contact_force_total_raw = total_force_raw[0].expand(self.num_envs)
            env0_all_body_contact_force_total_raw = all_body_contact_force_total_raw[0].expand(self.num_envs)
            env0_non_wheel_contact_force_total_raw = non_wheel_contact_force_total_raw[0].expand(self.num_envs)
            reset_happened_rate = self._stand_recent_reset_happened.to(torch.float32)
            env0_reset_happened = reset_happened_rate[0].expand(self.num_envs)
            low_stroke_threshold = 0.35
            low_stroke_margin = 0.10
            low_height_threshold = 0.72
            low_height_margin = 0.07
            high_stroke_threshold = 0.55
            high_stroke_margin = 0.10
            nominal_stroke = 0.50
            stroke_low_factor = torch.relu(low_stroke_threshold - stroke_actual_lf_lr_rf_rr) / max(
                low_stroke_margin, 1.0e-6
            )
            height_low_factor = torch.relu(low_height_threshold - root_height) / max(low_height_margin, 1.0e-6)
            stroke_high_factor = torch.relu(stroke_actual_lf_lr_rf_rr - high_stroke_threshold) / max(
                high_stroke_margin, 1.0e-6
            )
            stroke_error_from_nominal = stroke_actual_lf_lr_rf_rr - nominal_stroke
            front_mean_stroke = 0.5 * (stroke_actual_lf_lr_rf_rr[:, 0] + stroke_actual_lf_lr_rf_rr[:, 2])
            rear_mean_stroke = 0.5 * (stroke_actual_lf_lr_rf_rr[:, 1] + stroke_actual_lf_lr_rf_rr[:, 3])
            front_rear_stroke_diff = rear_mean_stroke - front_mean_stroke
            abs_front_rear_stroke_diff = torch.abs(front_rear_stroke_diff)
            front_rear_stroke_diff_over_threshold = torch.relu(abs_front_rear_stroke_diff - 0.10)
            negative_action_rate = torch.relu(-hydraulic_action_lf_lr_rf_rr)
            positive_action_rate = torch.relu(hydraulic_action_lf_lr_rf_rr)
            low_stroke_negative_action_per_leg = stroke_low_factor * negative_action_rate
            height_gated_low_stroke_negative_action_per_leg = (
                height_low_factor.unsqueeze(1) * low_stroke_negative_action_per_leg
            )
            height_gated_negative_action_rate = height_low_factor.unsqueeze(1) * negative_action_rate
            height_high_factor = torch.relu(root_height - 0.80) / 0.10
            high_height_low_stroke_negative_action_per_leg = (
                height_high_factor.unsqueeze(1)
                * (torch.relu(0.45 - stroke_actual_lf_lr_rf_rr) / 0.10)
                * negative_action_rate
            )
            height_low_positive_action_rate = height_low_factor.unsqueeze(1) * positive_action_rate
            low_stroke_negative_action_mask = (
                (stroke_actual_lf_lr_rf_rr < low_stroke_threshold) & (hydraulic_action_lf_lr_rf_rr < 0.0)
            ).to(torch.float32)
            low_stroke_negative_action_magnitude = negative_action_rate
            high_stroke_positive_action_per_leg = (
                height_low_factor.unsqueeze(1) * stroke_high_factor * positive_action_rate
            )
            high_stroke_positive_action_mask = (
                (root_height.unsqueeze(1) < low_height_threshold)
                & (stroke_actual_lf_lr_rf_rr > high_stroke_threshold)
                & (hydraulic_action_lf_lr_rf_rr > 0.0)
            ).to(torch.float32)
            high_stroke_positive_action_magnitude = positive_action_rate
            stroke_away_from_nominal_action_per_leg = torch.relu(stroke_error_from_nominal * hydraulic_action_lf_lr_rf_rr)
            stroke_toward_nominal_action_per_leg = torch.relu(-stroke_error_from_nominal * hydraulic_action_lf_lr_rf_rr)
            front_negative_hydraulic_action_mean = 0.5 * (torch.relu(-hydraulic_action_lf_lr_rf_rr[:, 0]) + torch.relu(-hydraulic_action_lf_lr_rf_rr[:, 2]))
            rear_positive_hydraulic_action_mean = 0.5 * (torch.relu(hydraulic_action_lf_lr_rf_rr[:, 1]) + torch.relu(hydraulic_action_lf_lr_rf_rr[:, 3]))
            front_positive_hydraulic_action_mean = 0.5 * (torch.relu(hydraulic_action_lf_lr_rf_rr[:, 0]) + torch.relu(hydraulic_action_lf_lr_rf_rr[:, 2]))
            rear_negative_hydraulic_action_mean = 0.5 * (torch.relu(-hydraulic_action_lf_lr_rf_rr[:, 1]) + torch.relu(-hydraulic_action_lf_lr_rf_rr[:, 3]))
            front_rear_positive_factor = torch.clamp(
                torch.relu(front_rear_stroke_diff - 0.10) / 0.20,
                min=0.0,
                max=1.0,
            )
            front_rear_negative_factor = torch.clamp(
                torch.relu(-front_rear_stroke_diff - 0.10) / 0.20,
                min=0.0,
                max=1.0,
            )
            front_rear_stroke_split_action_penalty = (
                front_rear_positive_factor * (front_negative_hydraulic_action_mean + rear_positive_hydraulic_action_mean)
                + front_rear_negative_factor * (front_positive_hydraulic_action_mean + rear_negative_hydraulic_action_mean)
            )
            return {
                "stand/contact_force_lf": ratio_force_lf_lr_rf_rr[:, 0],
                "stand/contact_force_lr": ratio_force_lf_lr_rf_rr[:, 1],
                "stand/contact_force_rf": ratio_force_lf_lr_rf_rr[:, 2],
                "stand/contact_force_rr": ratio_force_lf_lr_rf_rr[:, 3],
                "stand/contact_bool_lf": contacts[:, 1].to(torch.float32),
                "stand/contact_bool_lr": contacts[:, 0].to(torch.float32),
                "stand/contact_bool_rf": contacts[:, 2].to(torch.float32),
                "stand/contact_bool_rr": contacts[:, 3].to(torch.float32),
                "stand/contact_force_diag_diff": contact_force_diag_diff,
                "stand/normalized_contact_force_diag_diff": normalized_contact_force_diag_diff,
                "stand/contact_force_left_right_diff": contact_force_left_right_diff,
                "stand/contact_force_front_rear_diff": contact_force_front_rear_diff,
                "stand/contact_force_range_balance": contact_force_range_balance,
                "stand/contact_force_total": total_force_raw,
                "stand/contact_force_total_from_ratios_source": total_force_raw,
                "stand/robot_total_mass": robot_total_mass,
                "stand/expected_weight_force": expected_weight_force,
                "stand/wheel_contact_force_total_raw": total_force_raw,
                "stand/wheel_contact_force_total_raw_over_weight": wheel_contact_force_total_raw_over_weight,
                "stand/all_body_contact_force_total_raw": all_body_contact_force_total_raw,
                "stand/all_body_contact_force_total_raw_over_weight": all_body_contact_force_total_raw_over_weight,
                "stand/non_wheel_contact_force_total_raw": non_wheel_contact_force_total_raw,
                "stand/non_wheel_contact_force_total_raw_over_weight": non_wheel_contact_force_total_raw_over_weight,
                "stand/contact_force_ratio_lf": contact_force_ratio_lf_lr_rf_rr[:, 0],
                "stand/contact_force_ratio_lr": contact_force_ratio_lf_lr_rf_rr[:, 1],
                "stand/contact_force_ratio_rf": contact_force_ratio_lf_lr_rf_rr[:, 2],
                "stand/contact_force_ratio_rr": contact_force_ratio_lf_lr_rf_rr[:, 3],
                "stand/contact_force_min_ratio": min_contact_force_ratio,
                "stand/contact_force_ratio_sum": contact_force_ratio_sum,
                "stand/contact_force_ratio_valid_rate": valid_contact_total_mask.to(torch.float32),
                "stand/min_contact_force": min_contact_force,
                "stand/base_roll": roll,
                "stand/base_pitch": pitch,
                "stand/base_yaw_rate": root_ang_vel_b_z,
                "stand/yaw_error_from_reset": yaw_error_from_reset,
                "stand/base_lin_vel_x": base_lin_vel_x,
                "stand/base_lin_vel_y": base_lin_vel_y,
                "stand/base_xy_speed": base_xy_speed,
                "stand/root_height": root_height,
                "stand/root_height_reward_used": root_height,
                "stand/root_world_z": root_world_z,
                "stand/env_origin_z": env_origin_z,
                "stand/terrain_height_under_robot": torch.full(
                    (self.num_envs,), self._stand_trace_ground_height, device=self.device, dtype=torch.float32
                ),
                "stand/root_world_z_minus_env_origin_z": root_world_z - env_origin_z,
                "stand/root_world_z_minus_terrain_height": root_world_z - self._stand_trace_ground_height,
                "stand/base_link_world_z": base_link_world_z,
                "stand/wheel_world_z_lf": wheel_world_pos_lf_lr_rf_rr[:, 0, 2],
                "stand/wheel_world_z_lr": wheel_world_pos_lf_lr_rf_rr[:, 1, 2],
                "stand/wheel_world_z_rf": wheel_world_pos_lf_lr_rf_rr[:, 2, 2],
                "stand/wheel_world_z_rr": wheel_world_pos_lf_lr_rf_rr[:, 3, 2],
                "stand/wheel_relative_z_lf": wheel_relative_pos_lf_lr_rf_rr[:, 0, 2],
                "stand/wheel_relative_z_lr": wheel_relative_pos_lf_lr_rf_rr[:, 1, 2],
                "stand/wheel_relative_z_rf": wheel_relative_pos_lf_lr_rf_rr[:, 2, 2],
                "stand/wheel_relative_z_rr": wheel_relative_pos_lf_lr_rf_rr[:, 3, 2],
                "stand/root_height_target": torch.full(
                    (self.num_envs,), root_height_target, device=self.device, dtype=torch.float32
                ),
                "stand/root_height_band_score": root_height_band_score,
                "stand/root_height_error": root_height_error,
                "stand/root_height_abs_error": torch.abs(root_height_error),
                "stand/root_height_low": root_height_low_raw,
                "stand/root_height_low_rate": root_height_low_raw,
                "stand/root_height_low_raw_rate": root_height_low_raw,
                "stand/root_height_high_rate": root_height_high_rate,
                "stand/root_height_low_grace_active_rate": root_height_low_grace_active,
                "stand/termination_root_height_low_after_grace_rate": termination_root_height_low_after_grace,
                "stand/episode_step_mean": episode_steps,
                "stand/episode_step_env0": self.episode_length_buf[0].to(torch.float32).expand(self.num_envs),
                "stand/hydraulic_stroke_lf": stroke_actual[:, 1],
                "stand/hydraulic_stroke_lr": stroke_actual[:, 0],
                "stand/hydraulic_stroke_rf": stroke_actual[:, 2],
                "stand/hydraulic_stroke_rr": stroke_actual[:, 3],
                "stand/hydraulic_action_lf": hydraulic_action_lf_lr_rf_rr[:, 0],
                "stand/hydraulic_action_lr": hydraulic_action_lf_lr_rf_rr[:, 1],
                "stand/hydraulic_action_rf": hydraulic_action_lf_lr_rf_rr[:, 2],
                "stand/hydraulic_action_rr": hydraulic_action_lf_lr_rf_rr[:, 3],
                "stand/raw_hydraulic_action_lf": hydraulic_action_lf_lr_rf_rr[:, 0],
                "stand/raw_hydraulic_action_lr": hydraulic_action_lf_lr_rf_rr[:, 1],
                "stand/raw_hydraulic_action_rf": hydraulic_action_lf_lr_rf_rr[:, 2],
                "stand/raw_hydraulic_action_rr": hydraulic_action_lf_lr_rf_rr[:, 3],
                "stand/clipped_hydraulic_action_lf": hydraulic_clipped_action_lf_lr_rf_rr[:, 0],
                "stand/clipped_hydraulic_action_lr": hydraulic_clipped_action_lf_lr_rf_rr[:, 1],
                "stand/clipped_hydraulic_action_rf": hydraulic_clipped_action_lf_lr_rf_rr[:, 2],
                "stand/clipped_hydraulic_action_rr": hydraulic_clipped_action_lf_lr_rf_rr[:, 3],
                "stand/processed_hydraulic_action_lf": hydraulic_processed_action_lf_lr_rf_rr[:, 0],
                "stand/processed_hydraulic_action_lr": hydraulic_processed_action_lf_lr_rf_rr[:, 1],
                "stand/processed_hydraulic_action_rf": hydraulic_processed_action_lf_lr_rf_rr[:, 2],
                "stand/processed_hydraulic_action_rr": hydraulic_processed_action_lf_lr_rf_rr[:, 3],
                "stand/target_stroke_lf": target_stroke_lf_lr_rf_rr[:, 0],
                "stand/target_stroke_lr": target_stroke_lf_lr_rf_rr[:, 1],
                "stand/target_stroke_rf": target_stroke_lf_lr_rf_rr[:, 2],
                "stand/target_stroke_rr": target_stroke_lf_lr_rf_rr[:, 3],
                "stand/joint_target_g_lf": joint_target_g_lb_lf_rf_rb[:, 1],
                "stand/joint_target_g_lb": joint_target_g_lb_lf_rf_rb[:, 0],
                "stand/joint_target_g_rf": joint_target_g_lb_lf_rf_rb[:, 2],
                "stand/joint_target_g_rb": joint_target_g_lb_lf_rf_rb[:, 3],
                "stand/joint_pos_g_lf": joint_pos_g_lb_lf_rf_rb[:, 1],
                "stand/joint_pos_g_lb": joint_pos_g_lb_lf_rf_rb[:, 0],
                "stand/joint_pos_g_rf": joint_pos_g_lb_lf_rf_rb[:, 2],
                "stand/joint_pos_g_rb": joint_pos_g_lb_lf_rf_rb[:, 3],
                "stand/joint_pos_minus_target_g_lf": joint_pos_minus_target_g_lb_lf_rf_rb[:, 1],
                "stand/joint_pos_minus_target_g_lb": joint_pos_minus_target_g_lb_lf_rf_rb[:, 0],
                "stand/joint_pos_minus_target_g_rf": joint_pos_minus_target_g_lb_lf_rf_rb[:, 2],
                "stand/joint_pos_minus_target_g_rb": joint_pos_minus_target_g_lb_lf_rf_rb[:, 3],
                "stand/max_abs_raw_hydraulic_action": max_abs_raw_hydraulic_action,
                "stand/max_abs_clipped_hydraulic_action": max_abs_clipped_hydraulic_action,
                "stand/max_abs_joint_pos_minus_target": max_abs_joint_pos_minus_target,
                "stand/mean_abs_hydraulic_action": torch.mean(torch.abs(hydraulic_action_lf_lr_rf_rr), dim=1),
                "stand/hydraulic_action_range": (
                    hydraulic_action_lf_lr_rf_rr.max(dim=1).values - hydraulic_action_lf_lr_rf_rr.min(dim=1).values
                ),
                "stand/low_stroke_negative_action_penalty": low_stroke_negative_action_per_leg.mean(dim=1),
                "stand/height_gated_low_stroke_negative_action_penalty": (
                    height_gated_low_stroke_negative_action_per_leg.mean(dim=1)
                ),
                "stand/low_stroke_negative_action_lf": low_stroke_negative_action_per_leg[:, 0],
                "stand/low_stroke_negative_action_lr": low_stroke_negative_action_per_leg[:, 1],
                "stand/low_stroke_negative_action_rf": low_stroke_negative_action_per_leg[:, 2],
                "stand/low_stroke_negative_action_rr": low_stroke_negative_action_per_leg[:, 3],
                "stand/height_gated_low_stroke_negative_action_lf": height_gated_low_stroke_negative_action_per_leg[:, 0],
                "stand/height_gated_low_stroke_negative_action_lr": height_gated_low_stroke_negative_action_per_leg[:, 1],
                "stand/height_gated_low_stroke_negative_action_rf": height_gated_low_stroke_negative_action_per_leg[:, 2],
                "stand/height_gated_low_stroke_negative_action_rr": height_gated_low_stroke_negative_action_per_leg[:, 3],
                "stand/low_stroke_negative_action_rate_lf": negative_action_rate[:, 0],
                "stand/low_stroke_negative_action_rate_lr": negative_action_rate[:, 1],
                "stand/low_stroke_negative_action_rate_rf": negative_action_rate[:, 2],
                "stand/low_stroke_negative_action_rate_rr": negative_action_rate[:, 3],
                "stand/low_stroke_negative_action_mask_rate_lf": low_stroke_negative_action_mask[:, 0],
                "stand/low_stroke_negative_action_mask_rate_lr": low_stroke_negative_action_mask[:, 1],
                "stand/low_stroke_negative_action_mask_rate_rf": low_stroke_negative_action_mask[:, 2],
                "stand/low_stroke_negative_action_mask_rate_rr": low_stroke_negative_action_mask[:, 3],
                "stand/low_stroke_negative_action_magnitude_lf": low_stroke_negative_action_magnitude[:, 0],
                "stand/low_stroke_negative_action_magnitude_lr": low_stroke_negative_action_magnitude[:, 1],
                "stand/low_stroke_negative_action_magnitude_rf": low_stroke_negative_action_magnitude[:, 2],
                "stand/low_stroke_negative_action_magnitude_rr": low_stroke_negative_action_magnitude[:, 3],
                "stand/height_gated_low_stroke_negative_action_rate_lf": height_gated_negative_action_rate[:, 0],
                "stand/height_gated_low_stroke_negative_action_rate_lr": height_gated_negative_action_rate[:, 1],
                "stand/height_gated_low_stroke_negative_action_rate_rf": height_gated_negative_action_rate[:, 2],
                "stand/height_gated_low_stroke_negative_action_rate_rr": height_gated_negative_action_rate[:, 3],
                "stand/high_height_low_stroke_negative_action_value": (
                    high_height_low_stroke_negative_action_per_leg.mean(dim=1)
                ),
                "stand/height_low_positive_action_rate_lf": height_low_positive_action_rate[:, 0],
                "stand/height_low_positive_action_rate_lr": height_low_positive_action_rate[:, 1],
                "stand/height_low_positive_action_rate_rf": height_low_positive_action_rate[:, 2],
                "stand/height_low_positive_action_rate_rr": height_low_positive_action_rate[:, 3],
                "stand/height_low_high_stroke_positive_action_penalty": high_stroke_positive_action_per_leg.mean(dim=1),
                "stand/height_low_high_stroke_positive_action_lf": high_stroke_positive_action_per_leg[:, 0],
                "stand/height_low_high_stroke_positive_action_lr": high_stroke_positive_action_per_leg[:, 1],
                "stand/height_low_high_stroke_positive_action_rf": high_stroke_positive_action_per_leg[:, 2],
                "stand/height_low_high_stroke_positive_action_rr": high_stroke_positive_action_per_leg[:, 3],
                "stand/high_stroke_positive_action_mask_rate_lf": high_stroke_positive_action_mask[:, 0],
                "stand/high_stroke_positive_action_mask_rate_lr": high_stroke_positive_action_mask[:, 1],
                "stand/high_stroke_positive_action_mask_rate_rf": high_stroke_positive_action_mask[:, 2],
                "stand/high_stroke_positive_action_mask_rate_rr": high_stroke_positive_action_mask[:, 3],
                "stand/high_stroke_positive_action_magnitude_lf": high_stroke_positive_action_magnitude[:, 0],
                "stand/high_stroke_positive_action_magnitude_lr": high_stroke_positive_action_magnitude[:, 1],
                "stand/high_stroke_positive_action_magnitude_rf": high_stroke_positive_action_magnitude[:, 2],
                "stand/high_stroke_positive_action_magnitude_rr": high_stroke_positive_action_magnitude[:, 3],
                "stand/stroke_away_from_nominal_action_penalty": stroke_away_from_nominal_action_per_leg.mean(dim=1),
                "stand/stroke_away_from_nominal_action_lf": stroke_away_from_nominal_action_per_leg[:, 0],
                "stand/stroke_away_from_nominal_action_lr": stroke_away_from_nominal_action_per_leg[:, 1],
                "stand/stroke_away_from_nominal_action_rf": stroke_away_from_nominal_action_per_leg[:, 2],
                "stand/stroke_away_from_nominal_action_rr": stroke_away_from_nominal_action_per_leg[:, 3],
                "stand/stroke_toward_nominal_action_lf": stroke_toward_nominal_action_per_leg[:, 0],
                "stand/stroke_toward_nominal_action_lr": stroke_toward_nominal_action_per_leg[:, 1],
                "stand/stroke_toward_nominal_action_rf": stroke_toward_nominal_action_per_leg[:, 2],
                "stand/stroke_toward_nominal_action_rr": stroke_toward_nominal_action_per_leg[:, 3],
                "stand/front_mean_stroke": front_mean_stroke,
                "stand/rear_mean_stroke": rear_mean_stroke,
                "stand/front_rear_stroke_diff": front_rear_stroke_diff,
                "stand/abs_front_rear_stroke_diff": abs_front_rear_stroke_diff,
                "stand/front_rear_stroke_diff_over_threshold": front_rear_stroke_diff_over_threshold,
                "stand/front_rear_stroke_balance_penalty": abs_front_rear_stroke_diff,
                "stand/front_rear_stroke_split_action_penalty": front_rear_stroke_split_action_penalty,
                "stand/front_negative_hydraulic_action_mean": front_negative_hydraulic_action_mean,
                "stand/rear_positive_hydraulic_action_mean": rear_positive_hydraulic_action_mean,
                "stand/front_positive_hydraulic_action_mean": front_positive_hydraulic_action_mean,
                "stand/rear_negative_hydraulic_action_mean": rear_negative_hydraulic_action_mean,
                "stand/stroke_range_value": stroke_range_value,
                "stand/stroke_diagonal_balance_value": stroke_diagonal_balance_value,
                "stand/stroke_left_right_balance_value": stroke_left_right_balance_value,
                "stand/stroke_front_rear_balance_value": stroke_front_rear_balance_value,
                "stand/stroke_variance_from_nominal_value": stroke_variance_from_nominal_value,
                "stand/target_stroke_range_value": target_stroke_range_value,
                "stand/target_stroke_diagonal_balance_value": target_stroke_diagonal_balance_value,
                "stand/action_diagonal_split_value": action_diagonal_split_value,
                "stand/stroke_soft_limit_margin_lf": stroke_soft_limit_margin[:, 0],
                "stand/stroke_soft_limit_margin_lr": stroke_soft_limit_margin[:, 1],
                "stand/stroke_soft_limit_margin_rf": stroke_soft_limit_margin[:, 2],
                "stand/stroke_soft_limit_margin_rr": stroke_soft_limit_margin[:, 3],
                "stand/min_stroke_soft_limit_margin": stroke_soft_limit_margin.min(dim=1).values,
                "stand/max_hydraulic_stroke": stroke_max,
                "stand/min_hydraulic_stroke": stroke_min,
                "stand/stroke_range": stroke_range,
                "stand/raw_wheel_target_lf": raw_wheel_target[:, 0],
                "stand/raw_wheel_target_lr": raw_wheel_target[:, 1],
                "stand/raw_wheel_target_rf": raw_wheel_target[:, 2],
                "stand/raw_wheel_target_rr": raw_wheel_target[:, 3],
                "stand/semantic_wheel_target_lf": semantic_wheel_target[:, 0],
                "stand/semantic_wheel_target_lr": semantic_wheel_target[:, 1],
                "stand/semantic_wheel_target_rf": semantic_wheel_target[:, 2],
                "stand/semantic_wheel_target_rr": semantic_wheel_target[:, 3],
                "stand/wheel_actual_vel_lf": wheel_actual_vel[:, 0],
                "stand/wheel_actual_vel_lr": wheel_actual_vel[:, 1],
                "stand/wheel_actual_vel_rf": wheel_actual_vel[:, 2],
                "stand/wheel_actual_vel_rr": wheel_actual_vel[:, 3],
                "stand/mean_abs_wheel_target": torch.mean(torch.abs(semantic_wheel_target), dim=1),
                "stand/mean_abs_wheel_actual_velocity": torch.mean(torch.abs(wheel_actual_vel), dim=1),
                "stand/env0_root_height": env0_root_height,
                "stand/env0_root_height_high": env0_root_height_high,
                "stand/env0_root_height_low_grace_active": env0_root_height_low_grace_active,
                "stand/root_height_last50_mean": root_height_last50_mean,
                "stand/root_lin_vel_z_last50_mean": root_lin_vel_z_last50_mean,
                "stand/wheel_contact_over_weight_last50_mean": wheel_contact_over_weight_last50_mean,
                "stand/non_wheel_contact_over_weight_last50_mean": non_wheel_contact_over_weight_last50_mean,
                "stand/env0_hydraulic_stroke_lf": env0_stroke_lf_lr_rf_rr[:, 0],
                "stand/env0_hydraulic_stroke_lr": env0_stroke_lf_lr_rf_rr[:, 1],
                "stand/env0_hydraulic_stroke_rf": env0_stroke_lf_lr_rf_rr[:, 2],
                "stand/env0_hydraulic_stroke_rr": env0_stroke_lf_lr_rf_rr[:, 3],
                "stand/env0_stroke_range_value": env0_stroke_range_value,
                "stand/env0_stroke_diagonal_balance_value": env0_stroke_diagonal_balance_value,
                "stand/env0_wheel_contact_force_lf": env0_contact_force_lf_lr_rf_rr[:, 0],
                "stand/env0_wheel_contact_force_lr": env0_contact_force_lf_lr_rf_rr[:, 1],
                "stand/env0_wheel_contact_force_rf": env0_contact_force_lf_lr_rf_rr[:, 2],
                "stand/env0_wheel_contact_force_rr": env0_contact_force_lf_lr_rf_rr[:, 3],
                "stand/env0_wheel_contact_force_total_raw": env0_contact_force_total_raw,
                "stand/env0_all_body_contact_force_total_raw": env0_all_body_contact_force_total_raw,
                "stand/env0_non_wheel_contact_force_total_raw": env0_non_wheel_contact_force_total_raw,
                "stand/env0_contact_bool_lf": env0_contact_bool_lf_lr_rf_rr[:, 0],
                "stand/env0_contact_bool_lr": env0_contact_bool_lf_lr_rf_rr[:, 1],
                "stand/env0_contact_bool_rf": env0_contact_bool_lf_lr_rf_rr[:, 2],
                "stand/env0_contact_bool_rr": env0_contact_bool_lf_lr_rf_rr[:, 3],
                "stand/reset_happened_rate": reset_happened_rate,
                "stand/env0_reset_happened": env0_reset_happened,
                "stand/root_height_when_low": root_height,
                "stand/low_height_hydraulic_stroke_lf": stroke_actual[:, 1],
                "stand/low_height_hydraulic_stroke_lr": stroke_actual[:, 0],
                "stand/low_height_hydraulic_stroke_rf": stroke_actual[:, 2],
                "stand/low_height_hydraulic_stroke_rr": stroke_actual[:, 3],
                "stand/low_height_hydraulic_action_lf": hydraulic_action_lf_lr_rf_rr[:, 0],
                "stand/low_height_hydraulic_action_lr": hydraulic_action_lf_lr_rf_rr[:, 1],
                "stand/low_height_hydraulic_action_rf": hydraulic_action_lf_lr_rf_rr[:, 2],
                "stand/low_height_hydraulic_action_rr": hydraulic_action_lf_lr_rf_rr[:, 3],
                "stand/low_height_low_stroke_negative_action_penalty": low_stroke_negative_action_per_leg.mean(dim=1),
                "stand/low_height_height_gated_low_stroke_negative_action_penalty": (
                    height_gated_low_stroke_negative_action_per_leg.mean(dim=1)
                ),
                "stand/low_height_low_stroke_negative_action_lf": low_stroke_negative_action_per_leg[:, 0],
                "stand/low_height_low_stroke_negative_action_lr": low_stroke_negative_action_per_leg[:, 1],
                "stand/low_height_low_stroke_negative_action_rf": low_stroke_negative_action_per_leg[:, 2],
                "stand/low_height_low_stroke_negative_action_rr": low_stroke_negative_action_per_leg[:, 3],
                "stand/low_height_height_gated_low_stroke_negative_action_lf": (
                    height_gated_low_stroke_negative_action_per_leg[:, 0]
                ),
                "stand/low_height_height_gated_low_stroke_negative_action_lr": (
                    height_gated_low_stroke_negative_action_per_leg[:, 1]
                ),
                "stand/low_height_height_gated_low_stroke_negative_action_rf": (
                    height_gated_low_stroke_negative_action_per_leg[:, 2]
                ),
                "stand/low_height_height_gated_low_stroke_negative_action_rr": (
                    height_gated_low_stroke_negative_action_per_leg[:, 3]
                ),
                "stand/low_height_positive_action_rate_lf": positive_action_rate[:, 0],
                "stand/low_height_positive_action_rate_lr": positive_action_rate[:, 1],
                "stand/low_height_positive_action_rate_rf": positive_action_rate[:, 2],
                "stand/low_height_positive_action_rate_rr": positive_action_rate[:, 3],
                "stand/low_height_height_low_high_stroke_positive_action_penalty": high_stroke_positive_action_per_leg.mean(dim=1),
                "stand/low_height_stroke_away_from_nominal_action_penalty": stroke_away_from_nominal_action_per_leg.mean(dim=1),
                "stand/low_height_front_mean_stroke": front_mean_stroke,
                "stand/low_height_rear_mean_stroke": rear_mean_stroke,
                "stand/low_height_front_rear_stroke_diff": front_rear_stroke_diff,
                "stand/low_height_abs_front_rear_stroke_diff": abs_front_rear_stroke_diff,
                "stand/low_height_front_rear_stroke_diff_over_threshold": front_rear_stroke_diff_over_threshold,
                "stand/low_height_front_rear_stroke_balance_penalty": abs_front_rear_stroke_diff,
                "stand/low_height_front_rear_stroke_split_action_penalty": front_rear_stroke_split_action_penalty,
                "stand/low_height_front_negative_hydraulic_action_mean": front_negative_hydraulic_action_mean,
                "stand/low_height_rear_positive_hydraulic_action_mean": rear_positive_hydraulic_action_mean,
                "stand/low_height_front_positive_hydraulic_action_mean": front_positive_hydraulic_action_mean,
                "stand/low_height_rear_negative_hydraulic_action_mean": rear_negative_hydraulic_action_mean,
                "stand/low_height_contact_force_lf": ratio_force_lf_lr_rf_rr[:, 0],
                "stand/low_height_contact_force_lr": ratio_force_lf_lr_rf_rr[:, 1],
                "stand/low_height_contact_force_rf": ratio_force_lf_lr_rf_rr[:, 2],
                "stand/low_height_contact_force_rr": ratio_force_lf_lr_rf_rr[:, 3],
                "stand/low_height_contact_force_total": total_force_raw,
                "stand/low_height_contact_force_ratio_lf": contact_force_ratio_lf_lr_rf_rr[:, 0],
                "stand/low_height_contact_force_ratio_lr": contact_force_ratio_lf_lr_rf_rr[:, 1],
                "stand/low_height_contact_force_ratio_rf": contact_force_ratio_lf_lr_rf_rr[:, 2],
                "stand/low_height_contact_force_ratio_rr": contact_force_ratio_lf_lr_rf_rr[:, 3],
                "stand/low_height_contact_force_min_ratio": min_contact_force_ratio,
                "stand/low_height_contact_force_ratio_sum": contact_force_ratio_sum,
                "stand/low_height_base_xy_speed": base_xy_speed,
                "stand/low_height_base_roll": roll,
                "stand/low_height_base_pitch": pitch,
                "stand/low_height_min_contact_force": min_contact_force,
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

        if self._is_yaw_rate_command_task:
            cmd_yaw_rate = mdp.yaw_rate_command(self)
            cmd_yaw_abs = torch.abs(cmd_yaw_rate)
            cmd_sign = torch.sign(cmd_yaw_rate)
            active_mask = cmd_yaw_abs > 0.02
            too_small_active_mask = cmd_yaw_abs > 0.05
            signed_yaw = cmd_sign * root_ang_vel_b_z
            yaw_cmd_error_abs = torch.abs(root_ang_vel_b_z - cmd_yaw_rate)
            yaw_cmd_correct_direction = ((signed_yaw > 0.0) & active_mask).to(torch.float32)
            min_yaw = torch.clamp(0.5 * cmd_yaw_abs, min=0.04, max=0.08)
            yaw_cmd_too_small = (
                torch.clamp(torch.relu(min_yaw - signed_yaw) / torch.clamp(min_yaw, min=1.0e-6), min=0.0, max=2.0)
                * too_small_active_mask.to(torch.float32)
            )
            base_lin_vel_y = robot.data.root_lin_vel_b[:, 1]
            base_xy_vel_mean = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
            wheel_target_abs_mean = torch.mean(torch.abs(wheel_velocity_target), dim=1)
            wheel_joint_vel_abs_mean = torch.mean(torch.abs(wheel_joint_vel), dim=1)
            raw_wheel_action_abs_mean = torch.mean(torch.abs(wheel_action_term.raw_actions), dim=1)
            semantic_left_target = semantic_wheel_velocity_target[:, :2].mean(dim=1)
            semantic_right_target = semantic_wheel_velocity_target[:, 2:].mean(dim=1)
            semantic_forward_mode_target = 0.5 * (semantic_left_target + semantic_right_target)
            semantic_turn_mode_target = 0.5 * (semantic_right_target - semantic_left_target)
            wheel_velocity_limit = max(float(getattr(wheel_action_term, "_velocity_limit", 1.0)), 1.0)
            wheel_forward_mode_penalty_mean = torch.abs(semantic_forward_mode_target) / wheel_velocity_limit
            wheel_turn_mode_soft_limit_mean = torch.relu(torch.abs(semantic_turn_mode_target) - 8.0) / wheel_velocity_limit
            wheel_target_abs_soft_limit_mean = torch.relu(wheel_target_abs_mean - 8.0) / wheel_velocity_limit
            wheel_joint_vel_abs_soft_limit_mean = torch.relu(wheel_joint_vel_abs_mean - 10.0) / wheel_velocity_limit
            yaw_deficit = torch.clamp(
                torch.relu(min_yaw - signed_yaw) / torch.clamp(min_yaw, min=1.0e-6),
                min=0.0,
                max=2.0,
            )
            wasted_turn_when_yaw_small_mean = (
                torch.abs(semantic_turn_mode_target) / wheel_velocity_limit
            ) * yaw_deficit * too_small_active_mask.to(torch.float32)
            wheel_left_target_mean = wheel_velocity_target[:, :2].mean(dim=1)
            wheel_right_target_mean = wheel_velocity_target[:, 2:].mean(dim=1)
            policy_hydraulic_action_abs_mean = torch.mean(torch.abs(self._policy_hydraulic_action_raw), dim=1)
            hydraulic_action_abs_mean = torch.mean(torch.abs(self._yaw_turn_support_executed_raw_action), dim=1)
            hydraulic_action_rate_abs_mean = torch.mean(
                torch.abs(self._yaw_turn_support_executed_raw_action - self._executed_hydraulic_action_prev), dim=1
            )
            stroke_actual_lb_lf_rf_rb = leg_action_term.stroke_actual[:, self._support_hold_stroke_to_lb_lf_rf_rb]
            stroke_range_mean = stroke_actual_lb_lf_rf_rb.max(dim=1).values - stroke_actual_lb_lf_rf_rb.min(dim=1).values
            stroke_diagonal_balance = torch.abs(
                (stroke_actual_lb_lf_rf_rb[:, 0] + stroke_actual_lb_lf_rf_rb[:, 2])
                - (stroke_actual_lb_lf_rf_rb[:, 1] + stroke_actual_lb_lf_rf_rb[:, 3])
            )
            contact_force_lb_lf_rf_rb, contact_force_total_raw, _, _ = _wheel_contact_force_ratio_lf_lr_rf_rr(
                self,
                SceneEntityCfg("wheel_contact_forces", body_names=["w_lb", "w_lf", "w_rf", "w_rb"]),
            )
            contact_force_imbalance = (
                contact_force_lb_lf_rf_rb.max(dim=1).values - contact_force_lb_lf_rf_rb.min(dim=1).values
            ) / torch.clamp(contact_force_total_raw, min=1.0e-6)
            min_wheel_contact_force_mean = contact_force_lb_lf_rf_rb.min(dim=1).values
            zero_contact_ratio = (contact_force_lb_lf_rf_rb < 1.0).to(torch.float32).mean(dim=1)
            bad_orientation_rate = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            for term_idx, term_name in enumerate(getattr(self.termination_manager, "active_terms", ())):
                if term_name == "bad_orientation":
                    bad_orientation_rate = self.termination_manager._term_dones[:, term_idx].to(torch.float32)
                    break

            def masked_mean_or_zero(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                mask_f = mask.to(value.dtype)
                denom = torch.clamp(mask_f.sum(), min=1.0)
                scalar = torch.sum(value * mask_f) / denom
                return torch.full_like(value, scalar)

            tracking_term_idx = None
            wrong_direction_term_idx = None
            too_small_term_idx = None
            for term_idx, term_name in enumerate(self.reward_manager.active_terms):
                if term_name == "signed_yaw_rate_tracking":
                    tracking_term_idx = term_idx
                elif term_name == "wrong_direction_yaw":
                    wrong_direction_term_idx = term_idx
                elif term_name == "too_small_yaw_rate_when_error_large":
                    too_small_term_idx = term_idx
            if tracking_term_idx is not None:
                tracking_weight = float(self.cfg.rewards.signed_yaw_rate_tracking.weight)
                yaw_rate_command_tracking_mean = self.reward_manager._step_reward[:, tracking_term_idx] / max(
                    abs(tracking_weight), 1.0e-6
                )
            else:
                yaw_rate_command_tracking_mean = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            hydraulic_free_mode_gate_mean = torch.ones(
                (self.num_envs,), dtype=torch.float32, device=self.device
            )
            hydraulic_mode_debug = torch.full(
                (self.num_envs,), 2.0, dtype=torch.float32, device=self.device
            )
            support_hold_mode_debug = hydraulic_mode_debug
            yaw_cmd_sign_mode = os.getenv("RANGER_YAW_CMD_SIGN_MODE", "balanced").strip().lower()
            yaw_cmd_sign_mode_debug = torch.full(
                (self.num_envs,),
                0.0 if yaw_cmd_sign_mode == "balanced" else 1.0 if yaw_cmd_sign_mode == "positive" else 2.0,
                dtype=torch.float32,
                device=self.device,
            )
            positive_mask = cmd_yaw_rate > 0.0
            negative_mask = cmd_yaw_rate < 0.0
            positive_cmd_yaw_rate_mean = masked_mean_or_zero(root_ang_vel_b_z, positive_mask)
            negative_cmd_yaw_rate_mean = masked_mean_or_zero(root_ang_vel_b_z, negative_mask)
            positive_cmd_correct_direction_rate = masked_mean_or_zero(yaw_cmd_correct_direction, positive_mask)
            negative_cmd_correct_direction_rate = masked_mean_or_zero(yaw_cmd_correct_direction, negative_mask)
            return {
                "yaw_cmd_mean": cmd_yaw_rate,
                "yaw_cmd_abs_mean": cmd_yaw_abs,
                "yaw_cmd_positive_rate": positive_mask.to(torch.float32),
                "yaw_cmd_negative_rate": negative_mask.to(torch.float32),
                "yaw_cmd_sign_mode_debug": yaw_cmd_sign_mode_debug,
                "yaw_rate_command_tracking_mean": yaw_rate_command_tracking_mean,
                "yaw_cmd_error_abs_mean": yaw_cmd_error_abs,
                "yaw_cmd_correct_direction_rate": yaw_cmd_correct_direction,
                "yaw_cmd_too_small_rate": yaw_cmd_too_small,
                "positive_cmd_yaw_rate_mean": positive_cmd_yaw_rate_mean,
                "negative_cmd_yaw_rate_mean": negative_cmd_yaw_rate_mean,
                "positive_cmd_correct_direction_rate": positive_cmd_correct_direction_rate,
                "negative_cmd_correct_direction_rate": negative_cmd_correct_direction_rate,
                "base_yaw_rate": root_ang_vel_b_z,
                "base_lin_vel_x": base_lin_vel_x,
                "base_lin_vel_y": base_lin_vel_y,
                "base_xy_vel_mean": base_xy_vel_mean,
                "wheel_target_abs_mean": wheel_target_abs_mean,
                "wheel_joint_vel_abs_mean": wheel_joint_vel_abs_mean,
                "raw_wheel_action_abs_mean": raw_wheel_action_abs_mean,
                "semantic_forward_mode_target_abs_mean": torch.abs(semantic_forward_mode_target),
                "semantic_turn_mode_target_abs_mean": torch.abs(semantic_turn_mode_target),
                "wheel_forward_mode_penalty_mean": wheel_forward_mode_penalty_mean,
                "wheel_turn_mode_soft_limit_mean": wheel_turn_mode_soft_limit_mean,
                "wheel_target_abs_soft_limit_mean": wheel_target_abs_soft_limit_mean,
                "wheel_joint_vel_abs_soft_limit_mean": wheel_joint_vel_abs_soft_limit_mean,
                "wasted_turn_when_yaw_small_mean": wasted_turn_when_yaw_small_mean,
                "wheel_velocity_limit_debug": torch.full(
                    (self.num_envs,), wheel_velocity_limit, dtype=torch.float32, device=self.device
                ),
                "wheel_left_target_mean": wheel_left_target_mean,
                "wheel_right_target_mean": wheel_right_target_mean,
                "raw_wheel_target_lb_mean": wheel_velocity_target[:, 0],
                "raw_wheel_target_lf_mean": wheel_velocity_target[:, 1],
                "raw_wheel_target_rf_mean": wheel_velocity_target[:, 2],
                "raw_wheel_target_rb_mean": wheel_velocity_target[:, 3],
                "raw_wheel_joint_vel_lb_mean": wheel_joint_vel[:, 0],
                "raw_wheel_joint_vel_lf_mean": wheel_joint_vel[:, 1],
                "raw_wheel_joint_vel_rf_mean": wheel_joint_vel[:, 2],
                "raw_wheel_joint_vel_rb_mean": wheel_joint_vel[:, 3],
                "semantic_wheel_target_lb_mean": semantic_wheel_velocity_target[:, 0],
                "semantic_wheel_target_lf_mean": semantic_wheel_velocity_target[:, 1],
                "semantic_wheel_target_rf_mean": semantic_wheel_velocity_target[:, 2],
                "semantic_wheel_target_rb_mean": semantic_wheel_velocity_target[:, 3],
                "semantic_wheel_joint_vel_lb_mean": semantic_wheel_joint_vel[:, 0],
                "semantic_wheel_joint_vel_lf_mean": semantic_wheel_joint_vel[:, 1],
                "semantic_wheel_joint_vel_rf_mean": semantic_wheel_joint_vel[:, 2],
                "semantic_wheel_joint_vel_rb_mean": semantic_wheel_joint_vel[:, 3],
                "hydraulic_mode_debug": hydraulic_mode_debug,
                "support_hold_mode_debug": support_hold_mode_debug,
                "hydraulic_free_mode_gate_mean": hydraulic_free_mode_gate_mean,
                "policy_hydraulic_action_abs_mean": policy_hydraulic_action_abs_mean,
                "hydraulic_action_abs_mean": hydraulic_action_abs_mean,
                "hydraulic_action_rate_abs_mean": hydraulic_action_rate_abs_mean,
                "stroke_range_mean": stroke_range_mean,
                "stroke_diagonal_balance": stroke_diagonal_balance,
                "zero_contact_ratio": zero_contact_ratio,
                "contact_force_imbalance": contact_force_imbalance,
                "min_wheel_contact_force_mean": min_wheel_contact_force_mean,
                "bad_orientation_rate": bad_orientation_rate,
            }

        if self._is_short_goal_task:
            target_vec_b, goal_distance, heading_error = mdp.short_goal_target_body(self)
            goal_angle_body = torch.atan2(target_vec_b[:, 1], target_vec_b[:, 0])
            base_lin_vel_y = robot.data.root_lin_vel_b[:, 1]
            base_xy_vel_mean = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
            heading_abs = torch.abs(heading_error)

            yaw_tracking_term = getattr(self.cfg.rewards, "yaw_rate_tracking", None)
            yaw_tracking_params = {} if yaw_tracking_term is None else yaw_tracking_term.params
            tracking_yaw_rate_max = float(yaw_tracking_params.get("yaw_rate_max", 0.45))
            tracking_heading_deadband = float(yaw_tracking_params.get("heading_deadband", 0.04))
            tracking_heading_scale = float(yaw_tracking_params.get("heading_scale", 0.30))
            tracking_effective_error = torch.sign(heading_error) * torch.relu(
                heading_abs - max(tracking_heading_deadband, 0.0)
            )
            desired_yaw_rate_continuous = tracking_yaw_rate_max * torch.tanh(
                tracking_effective_error / max(tracking_heading_scale, 1.0e-6)
            )
            desired_yaw_rate_abs = torch.abs(desired_yaw_rate_continuous)
            actual_yaw_rate_abs = torch.abs(root_ang_vel_b_z)
            yaw_rate_tracking_error_abs = torch.abs(root_ang_vel_b_z - desired_yaw_rate_continuous)
            yaw_response_valid = desired_yaw_rate_abs > 0.05
            yaw_rate_response_ratio = torch.where(
                yaw_response_valid,
                actual_yaw_rate_abs / torch.clamp(desired_yaw_rate_abs, min=0.05),
                torch.zeros_like(actual_yaw_rate_abs),
            )
            yaw_rate_response_ratio = torch.clamp(yaw_rate_response_ratio, max=2.0)

            speed_profile_params = self.cfg.rewards.short_goal_speed_profile.params
            braking_acceleration = float(speed_profile_params.get("braking_acceleration", 0.60))
            stop_distance = float(speed_profile_params.get("stop_distance", 0.30))
            reaction_time = float(speed_profile_params.get("reaction_time", 0.20))
            braking_margin = float(speed_profile_params.get("braking_margin", 0.08))
            near_distance = float(speed_profile_params.get("near_distance", 1.0))
            yaw_rate_ref = float(speed_profile_params.get("yaw_rate_ref", 0.80))
            yaw_component_weight = float(speed_profile_params.get("yaw_component_weight", 0.5))

            target_dir_b_for_speed = target_vec_b[:, :2] / torch.clamp(goal_distance.unsqueeze(1), min=1.0e-6)
            velocity_toward_goal_for_speed = torch.sum(
                robot.data.root_lin_vel_b[:, :2] * target_dir_b_for_speed, dim=1
            )
            positive_velocity_for_speed = torch.relu(velocity_toward_goal_for_speed)
            braking_accel_safe = max(braking_acceleration, 1.0e-6)
            required_braking_distance = (
                torch.square(positive_velocity_for_speed) / (2.0 * braking_accel_safe)
                + reaction_time * positive_velocity_for_speed
                + braking_margin
            )
            available_braking_distance = torch.relu(goal_distance - braking_margin)
            desired_speed = torch.relu(
                -braking_accel_safe * reaction_time
                + torch.sqrt(
                    (braking_accel_safe * reaction_time) ** 2
                    + 2.0 * braking_accel_safe * available_braking_distance
                )
            )

            cruise_params = self.cfg.rewards.cruise_underspeed.params
            heading_deadband = float(cruise_params.get("heading_deadband", 0.20))
            heading_full = float(cruise_params.get("heading_full", 0.80))
            alignment_floor = float(cruise_params.get("alignment_floor", 0.35))
            approach_full_distance = float(cruise_params.get("approach_full_distance", 0.60))
            cruise_full_distance = float(cruise_params.get("cruise_full_distance", 1.20))
            approach_speed = float(cruise_params.get("approach_speed", 0.35))
            cruise_speed = float(cruise_params.get("cruise_speed", 0.80))
            raw_alignment_gate = torch.clamp(
                (heading_full - heading_abs) / max(heading_full - heading_deadband, 1.0e-6),
                min=0.0,
                max=1.0,
            )
            alignment_speed_gate = alignment_floor + (1.0 - alignment_floor) * raw_alignment_gate
            approach_blend = torch.clamp(
                (goal_distance - approach_full_distance)
                / max(cruise_full_distance - approach_full_distance, 1.0e-6),
                min=0.0,
                max=1.0,
            )
            reference_speed = approach_speed + approach_blend * (cruise_speed - approach_speed)
            desired_speed = torch.minimum(desired_speed, reference_speed * alignment_speed_gate)
            speed_profile_excess = torch.relu(required_braking_distance - goal_distance)
            speed_profile_linear_penalty = torch.square(speed_profile_excess / max(near_distance, 1.0e-6))
            near_speed_profile_gate = torch.clamp(
                (near_distance - goal_distance) / max(near_distance - stop_distance, 1.0e-6),
                min=0.0,
                max=1.0,
            )
            speed_profile_yaw_penalty = near_speed_profile_gate * torch.square(
                torch.abs(root_ang_vel_b_z) / max(yaw_rate_ref, 1.0e-6)
            )
            speed_profile_penalty = speed_profile_linear_penalty + yaw_component_weight * speed_profile_yaw_penalty
            turn_distance_gate = torch.clamp((goal_distance - 0.50) / max(0.80 - 0.50, 1.0e-6), min=0.0, max=1.0)
            near_goal = goal_distance < self._short_goal_stop_success_distance
            near_goal_moving = near_goal & ((base_xy_vel_mean > 0.15) | (torch.abs(root_ang_vel_b_z) > 0.20))
            stopped_success = (
                (goal_distance < self._short_goal_stop_success_distance)
                & (base_xy_vel_mean < self._short_goal_stop_max_xy_speed)
                & (torch.abs(root_ang_vel_b_z) < self._short_goal_stop_max_yaw_rate)
                & (self._short_goal_stop_phase_stable_steps >= self._short_goal_stop_required_hold_steps)
            )
            wheel_target_abs_mean = torch.mean(torch.abs(wheel_velocity_target), dim=1)
            wheel_joint_vel_abs_mean = torch.mean(torch.abs(wheel_joint_vel), dim=1)
            raw_wheel_action_abs_mean = torch.mean(torch.abs(wheel_action_term.raw_actions), dim=1)
            policy_hydraulic_action_abs_mean = torch.mean(torch.abs(self._policy_hydraulic_action_raw), dim=1)
            final_hydraulic_action_lb_lf_rf_rb = leg_action_term.raw_actions[:, self._support_hold_action_raw_to_lb_lf_rf_rb]
            previous_hydraulic_action_lb_lf_rf_rb = self._short_goal_hydraulic_action_prev[
                :, self._support_hold_action_raw_to_lb_lf_rf_rb
            ]
            hydraulic_action_abs_mean = torch.mean(torch.abs(final_hydraulic_action_lb_lf_rf_rb), dim=1)
            hydraulic_action_rate_abs_mean = torch.mean(
                torch.abs(final_hydraulic_action_lb_lf_rf_rb - previous_hydraulic_action_lb_lf_rf_rb), dim=1
            )
            hydraulic_raw_action_abs_mean = hydraulic_action_abs_mean
            stroke_desired_lb_lf_rf_rb = leg_action_term.stroke_desired[:, self._support_hold_stroke_to_lb_lf_rf_rb]
            stroke_actual_lb_lf_rf_rb = leg_action_term.stroke_actual[:, self._support_hold_stroke_to_lb_lf_rf_rb]
            hydraulic_position_target_lb_lf_rf_rb = leg_action_term.position_target[:, self._support_hold_stroke_to_lb_lf_rf_rb]
            hydraulic_joint_position_target_lb_lf_rf_rb = robot.data.joint_pos_target[:, self._leg_joint_ids]
            hydraulic_joint_position_lb_lf_rf_rb = robot.data.joint_pos[:, self._leg_joint_ids]
            hydraulic_joint_velocity_lb_lf_rf_rb = robot.data.joint_vel[:, self._leg_joint_ids]
            hydraulic_joint_position_tracking_error = torch.abs(
                hydraulic_joint_position_target_lb_lf_rf_rb - hydraulic_joint_position_lb_lf_rf_rb
            )
            stroke_desired_range = stroke_desired_lb_lf_rf_rb.max(dim=1).values - stroke_desired_lb_lf_rf_rb.min(dim=1).values
            stroke_range_mean = stroke_actual_lb_lf_rf_rb.max(dim=1).values - stroke_actual_lb_lf_rf_rb.min(dim=1).values
            stroke_diagonal_balance = torch.abs(
                (stroke_actual_lb_lf_rf_rb[:, 0] + stroke_actual_lb_lf_rf_rb[:, 2])
                - (stroke_actual_lb_lf_rf_rb[:, 1] + stroke_actual_lb_lf_rf_rb[:, 3])
            )
            raw_contact_sensor = self.scene.sensors["wheel_contact_forces"]
            raw_contact_body_ids, _ = raw_contact_sensor.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
            raw_contact_body_ids = torch.as_tensor(raw_contact_body_ids, device=self.device, dtype=torch.long)
            contact_force_lb_lf_rf_rb = torch.max(
                torch.norm(raw_contact_sensor.data.net_forces_w_history[:, :, raw_contact_body_ids, :], dim=-1),
                dim=1,
            )[0]
            contact_force_total_raw = contact_force_lb_lf_rf_rb.sum(dim=1)
            contact_force_imbalance = (
                contact_force_lb_lf_rf_rb.max(dim=1).values - contact_force_lb_lf_rf_rb.min(dim=1).values
            ) / torch.clamp(contact_force_total_raw, min=1.0e-6)
            min_wheel_contact_force_mean = contact_force_lb_lf_rf_rb.min(dim=1).values
            zero_contact_ratio = (contact_force_lb_lf_rf_rb < 1.0).to(torch.float32).mean(dim=1)
            hydraulic_action_diag_mode = 0.5 * (
                final_hydraulic_action_lb_lf_rf_rb[:, 0]
                + final_hydraulic_action_lb_lf_rf_rb[:, 2]
                - final_hydraulic_action_lb_lf_rf_rb[:, 1]
                - final_hydraulic_action_lb_lf_rf_rb[:, 3]
            )
            stroke_diag_mode = 0.5 * (
                stroke_actual_lb_lf_rf_rb[:, 0]
                + stroke_actual_lb_lf_rf_rb[:, 2]
                - stroke_actual_lb_lf_rf_rb[:, 1]
                - stroke_actual_lb_lf_rf_rb[:, 3]
            )
            contact_diag_force = (
                contact_force_lb_lf_rf_rb[:, 0]
                + contact_force_lb_lf_rf_rb[:, 2]
                - contact_force_lb_lf_rf_rb[:, 1]
                - contact_force_lb_lf_rf_rb[:, 3]
            )
            contact_diag_ratio = contact_diag_force / torch.clamp(contact_force_total_raw, min=1.0e-6)

            def safe_batch_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x_centered = x - x.mean()
                y_centered = y - y.mean()
                cov = torch.mean(x_centered * y_centered)
                std_x = torch.sqrt(torch.mean(torch.square(x_centered)))
                std_y = torch.sqrt(torch.mean(torch.square(y_centered)))
                corr = cov / torch.clamp(std_x * std_y, min=1.0e-6)
                corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                return torch.full_like(x, corr)

            action_stroke_diag_correlation = safe_batch_correlation(hydraulic_action_diag_mode, stroke_diag_mode)
            action_contact_diag_correlation = safe_batch_correlation(hydraulic_action_diag_mode, contact_diag_ratio)
            stroke_contact_diag_correlation = safe_batch_correlation(stroke_diag_mode, contact_diag_ratio)
            # Short-goal tasks always execute the policy suspension output directly.
            hydraulic_mode_debug = torch.full((self.num_envs,), 2.0, dtype=torch.float32, device=self.device)
            short_goal_hydraulic_mode_debug = torch.full(
                (self.num_envs,), 2.0, dtype=torch.float32, device=self.device
            )
            support_hold_mode_debug = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            support_hold_active_debug = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            free_policy_control_active_debug = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
            support_hold_action_abs_mean = torch.mean(torch.abs(self._yaw_turn_support_hold_raw_action), dim=1)
            support_hold_action_range = (
                self._yaw_turn_support_hold_raw_action.max(dim=1).values - self._yaw_turn_support_hold_raw_action.min(dim=1).values
            )
            executed_hydraulic_action_abs_mean = hydraulic_action_abs_mean
            support_hold_contact_correction_abs_mean = torch.mean(
                torch.abs(self._yaw_turn_support_contact_correction_semantic), dim=1
            )
            support_hold_contact_correction_max_abs = torch.max(
                torch.abs(self._yaw_turn_support_contact_correction_semantic), dim=1
            ).values
            support_hold_stroke_correction_abs_mean = torch.mean(
                torch.abs(self._yaw_turn_support_stroke_correction_semantic), dim=1
            )
            support_hold_roll_pitch_correction_abs_mean = torch.mean(
                torch.abs(self._yaw_turn_support_roll_pitch_correction_semantic), dim=1
            )
            support_hold_total_pre_filter_abs_mean = torch.mean(
                torch.abs(self._yaw_turn_support_total_pre_filter_semantic), dim=1
            )
            support_hold_total_post_filter_abs_mean = torch.mean(
                torch.abs(self._yaw_turn_support_hold_semantic_action), dim=1
            )
            executed_hydraulic_action_lb_lf_rf_rb = final_hydraulic_action_lb_lf_rf_rb
            goal_reached_term_idx = None
            progress_term_idx = None
            heading_error_reduction_term_idx = None
            turn_toward_goal_term_idx = None
            wrong_direction_yaw_term_idx = None
            signed_yaw_rate_tracking_term_idx = None
            too_small_yaw_rate_when_error_large_term_idx = None
            for term_idx, term_name in enumerate(self.reward_manager.active_terms):
                if term_name == "progress_to_goal":
                    progress_term_idx = term_idx
                elif term_name == "goal_success":
                    goal_reached_term_idx = term_idx
                elif term_name == "heading_error_reduction":
                    heading_error_reduction_term_idx = term_idx
                elif term_name == "turn_toward_goal":
                    turn_toward_goal_term_idx = term_idx
                elif term_name == "wrong_direction_yaw":
                    wrong_direction_yaw_term_idx = term_idx
                elif term_name == "signed_yaw_rate_tracking":
                    signed_yaw_rate_tracking_term_idx = term_idx
                elif term_name == "too_small_yaw_rate_when_error_large":
                    too_small_yaw_rate_when_error_large_term_idx = term_idx
            if progress_term_idx is not None:
                progress_weight = float(self.cfg.rewards.progress_to_goal.weight)
                progress_mean = self.reward_manager._step_reward[:, progress_term_idx] / max(abs(progress_weight), 1.0e-6)
            else:
                progress_mean = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            if goal_reached_term_idx is not None:
                success_weight = float(self.cfg.rewards.goal_success.weight)
                success_rate = self.reward_manager._step_reward[:, goal_reached_term_idx] / max(abs(success_weight), 1.0e-6)
            else:
                success_rate = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            if heading_error_reduction_term_idx is not None:
                heading_error_reduction_weight = float(self.cfg.rewards.heading_error_reduction.weight)
                heading_error_reduction_mean = self.reward_manager._step_reward[:, heading_error_reduction_term_idx] / max(
                    abs(heading_error_reduction_weight), 1.0e-6
                )
            else:
                heading_error_reduction_mean = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            if turn_toward_goal_term_idx is not None:
                turn_toward_goal_weight = float(self.cfg.rewards.turn_toward_goal.weight)
                turn_toward_goal_mean = self.reward_manager._step_reward[:, turn_toward_goal_term_idx] / max(
                    abs(turn_toward_goal_weight), 1.0e-6
                )
            else:
                turn_toward_goal_mean = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            if wrong_direction_yaw_term_idx is not None:
                wrong_direction_yaw_weight = float(self.cfg.rewards.wrong_direction_yaw.weight)
                wrong_direction_yaw_mean = self.reward_manager._step_reward[:, wrong_direction_yaw_term_idx] / max(
                    abs(wrong_direction_yaw_weight), 1.0e-6
                )
            else:
                wrong_direction_yaw_mean = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            if signed_yaw_rate_tracking_term_idx is not None:
                signed_yaw_rate_tracking_weight = float(self.cfg.rewards.signed_yaw_rate_tracking.weight)
                signed_yaw_rate_tracking_mean = self.reward_manager._step_reward[
                    :, signed_yaw_rate_tracking_term_idx
                ] / max(abs(signed_yaw_rate_tracking_weight), 1.0e-6)
            else:
                signed_yaw_rate_tracking_mean = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            if too_small_yaw_rate_when_error_large_term_idx is not None:
                too_small_yaw_rate_when_error_large_weight = float(
                    self.cfg.rewards.too_small_yaw_rate_when_error_large.weight
                )
                too_small_yaw_rate_when_error_large_mean = self.reward_manager._step_reward[
                    :, too_small_yaw_rate_when_error_large_term_idx
                ] / max(abs(too_small_yaw_rate_when_error_large_weight), 1.0e-6)
            else:
                too_small_yaw_rate_when_error_large_mean = torch.zeros(
                    (self.num_envs,), dtype=torch.float32, device=self.device
                )
            desired_yaw_rate = torch.where(
                torch.abs(heading_error) > 0.10,
                torch.sign(target_vec_b[:, 1]) * 0.35,
                torch.zeros_like(root_ang_vel_b_z),
            )
            desired_yaw_sign = torch.sign(target_vec_b[:, 1])
            yaw_rate_error = root_ang_vel_b_z - desired_yaw_rate
            yaw_active_mask = (torch.abs(heading_error) > 0.10).to(torch.float32)
            yaw_correct_direction_rate = (
                ((desired_yaw_sign * root_ang_vel_b_z) > 0.0).to(torch.float32) * yaw_active_mask
            )
            yaw_too_small_rate = (
                (torch.abs(root_ang_vel_b_z) < 0.12).to(torch.float32) * (torch.abs(heading_error) > 0.25).to(torch.float32)
            )
            wheel_velocity_target_mean = wheel_velocity_target.mean(dim=1)
            wheel_joint_vel_mean = wheel_joint_vel.mean(dim=1)
            semantic_wheel_velocity_target_mean = semantic_wheel_velocity_target.mean(dim=1)
            semantic_wheel_joint_vel_mean = semantic_wheel_joint_vel.mean(dim=1)
            wheel_left_target_mean = semantic_wheel_velocity_target[:, 0:2].mean(dim=1)
            wheel_right_target_mean = semantic_wheel_velocity_target[:, 2:4].mean(dim=1)
            wheel_common_mode_target = 0.5 * (wheel_left_target_mean + wheel_right_target_mean)
            wheel_differential_target = wheel_right_target_mean - wheel_left_target_mean
            goal_left_mask = self._short_goal_initial_side_sign > 0.0
            goal_right_mask = self._short_goal_initial_side_sign < 0.0
            signed_yaw = desired_yaw_sign * root_ang_vel_b_z
            left_yaw_correct_direction_rate = yaw_correct_direction_rate * goal_left_mask.to(torch.float32)
            right_yaw_correct_direction_rate = yaw_correct_direction_rate * goal_right_mask.to(torch.float32)
            left_yaw_too_small_rate = yaw_too_small_rate * goal_left_mask.to(torch.float32)
            right_yaw_too_small_rate = yaw_too_small_rate * goal_right_mask.to(torch.float32)
            bad_orientation_rate = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            for term_idx, term_name in enumerate(getattr(self.termination_manager, "active_terms", ())):
                if term_name == "bad_orientation":
                    bad_orientation_rate = self.termination_manager._term_dones[:, term_idx].to(torch.float32)
                    break

            def masked_mean_or_zero(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                mask_f = mask.to(value.dtype)
                denom = torch.clamp(mask_f.sum(), min=1.0)
                scalar = torch.sum(value * mask_f) / denom
                return torch.full_like(value, scalar)

            wheel_velocity_limit = max(float(getattr(wheel_action_term, "_velocity_limit", 1.0)), 1.0e-6)
            semantic_previous_wheel_target = wheel_action_term.previous_velocity_target * self._wheel_forward_sign
            wheel_target_tracking_error = torch.abs(
                semantic_wheel_velocity_target - semantic_wheel_joint_vel
            ) / wheel_velocity_limit
            semantic_left_target = semantic_wheel_velocity_target[:, :2].mean(dim=1)
            semantic_right_target = semantic_wheel_velocity_target[:, 2:].mean(dim=1)
            wheel_common_mode_target_norm = 0.5 * (
                semantic_left_target + semantic_right_target
            ) / wheel_velocity_limit
            wheel_turn_mode_target_norm = 0.5 * (
                semantic_right_target - semantic_left_target
            ) / wheel_velocity_limit
            wheel_left_front_rear_target_diff_norm = torch.abs(
                semantic_wheel_velocity_target[:, 0] - semantic_wheel_velocity_target[:, 1]
            ) / wheel_velocity_limit
            wheel_right_front_rear_target_diff_norm = torch.abs(
                semantic_wheel_velocity_target[:, 2] - semantic_wheel_velocity_target[:, 3]
            ) / wheel_velocity_limit
            semantic_wheel_target_norm = semantic_wheel_velocity_target / wheel_velocity_limit
            left_front_target_norm = semantic_wheel_target_norm[:, 0]
            left_rear_target_norm = semantic_wheel_target_norm[:, 1]
            right_front_target_norm = semantic_wheel_target_norm[:, 2]
            right_rear_target_norm = semantic_wheel_target_norm[:, 3]
            opposite_sign_min_magnitude = 0.05
            wheel_left_front_rear_opposite_sign_rate = (
                (left_front_target_norm * left_rear_target_norm < 0.0)
                & (torch.minimum(torch.abs(left_front_target_norm), torch.abs(left_rear_target_norm)) > opposite_sign_min_magnitude)
            ).to(torch.float32)
            wheel_right_front_rear_opposite_sign_rate = (
                (right_front_target_norm * right_rear_target_norm < 0.0)
                & (torch.minimum(torch.abs(right_front_target_norm), torch.abs(right_rear_target_norm)) > opposite_sign_min_magnitude)
            ).to(torch.float32)
            wheel_left_front_rear_antisymmetric_mode = 0.5 * torch.abs(
                left_front_target_norm - left_rear_target_norm
            )
            wheel_right_front_rear_antisymmetric_mode = 0.5 * torch.abs(
                right_front_target_norm - right_rear_target_norm
            )
            wheel_target_rate_norm = torch.mean(
                torch.abs(semantic_wheel_velocity_target - semantic_previous_wheel_target), dim=1
            ) / wheel_velocity_limit
            wheel_action_saturation_rate = (
                torch.abs(wheel_action_term.raw_actions) >= 0.98
            ).to(torch.float32).mean(dim=1)

            target_dir_b = target_vec_b[:, :2] / torch.clamp(goal_distance.unsqueeze(1), min=1.0e-6)
            velocity_toward_goal = torch.sum(robot.data.root_lin_vel_b[:, :2] * target_dir_b, dim=1)
            stop_phase_active_rate = self._short_goal_stop_phase_active.to(torch.float32)
            stop_phase_stable_fraction = torch.clamp(
                self._short_goal_stop_phase_stable_steps.to(torch.float32)
                / max(float(self._short_goal_stop_required_hold_steps), 1.0),
                min=0.0,
                max=1.0,
            )

            left_goal_progress = masked_mean_or_zero(progress_mean, goal_left_mask)
            right_goal_progress = masked_mean_or_zero(progress_mean, goal_right_mask)
            left_goal_stopped_success = masked_mean_or_zero(stopped_success.to(torch.float32), goal_left_mask)
            right_goal_stopped_success = masked_mean_or_zero(stopped_success.to(torch.float32), goal_right_mask)
            left_goal_actual_yaw_rate_signed = masked_mean_or_zero(root_ang_vel_b_z, goal_left_mask)
            right_goal_actual_yaw_rate_signed = masked_mean_or_zero(root_ang_vel_b_z, goal_right_mask)
            left_goal_wheel_common_mode_norm = masked_mean_or_zero(wheel_common_mode_target_norm, goal_left_mask)
            right_goal_wheel_common_mode_norm = masked_mean_or_zero(wheel_common_mode_target_norm, goal_right_mask)
            left_goal_wheel_turn_mode_norm = masked_mean_or_zero(wheel_turn_mode_target_norm, goal_left_mask)
            right_goal_wheel_turn_mode_norm = masked_mean_or_zero(wheel_turn_mode_target_norm, goal_right_mask)
            left_goal_velocity_toward_goal = masked_mean_or_zero(velocity_toward_goal, goal_left_mask)
            right_goal_velocity_toward_goal = masked_mean_or_zero(velocity_toward_goal, goal_right_mask)
            left_goal_heading_error_signed = masked_mean_or_zero(heading_error, goal_left_mask)
            right_goal_heading_error_signed = masked_mean_or_zero(heading_error, goal_right_mask)
            left_goal_yaw_rate_tracking_error_abs = masked_mean_or_zero(
                yaw_rate_tracking_error_abs, goal_left_mask
            )
            right_goal_yaw_rate_tracking_error_abs = masked_mean_or_zero(
                yaw_rate_tracking_error_abs, goal_right_mask
            )

            wheel_radius = float(getattr(self.cfg, "short_goal_wheel_radius", 0.2024))
            root_forward_axis_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            root_forward_axis_b[:, 0] = 1.0
            root_forward_axis_w = quat_apply(robot.data.root_quat_w, root_forward_axis_b)
            wheel_hub_lin_vel_w = robot.data.body_lin_vel_w[:, self._wheel_robot_body_ids_lb_lf_rf_rb, :]
            wheel_hub_forward_speed = torch.sum(
                wheel_hub_lin_vel_w * root_forward_axis_w.unsqueeze(1), dim=-1
            )
            wheel_surface_speed = semantic_wheel_joint_vel * wheel_radius
            wheel_slip_speed = torch.abs(wheel_surface_speed - wheel_hub_forward_speed)
            wheel_motion_scale = torch.maximum(torch.abs(wheel_surface_speed), torch.abs(wheel_hub_forward_speed))
            wheel_slip_denominator = torch.maximum(
                wheel_motion_scale,
                torch.full_like(wheel_surface_speed, 0.10),
            )
            wheel_contact_valid = contact_force_lb_lf_rf_rb > 20.0
            wheel_motion_valid = wheel_contact_valid & (wheel_motion_scale > 0.20)
            wheel_slip_ratio = torch.where(
                wheel_contact_valid,
                wheel_slip_speed / wheel_slip_denominator,
                torch.zeros_like(wheel_slip_speed),
            )
            wheel_slip_ratio_motion_gated = torch.where(
                wheel_motion_valid,
                wheel_slip_speed / torch.clamp(wheel_motion_scale, min=0.20),
                torch.zeros_like(wheel_slip_speed),
            )
            wheel_slip_speed_contact = torch.where(
                wheel_contact_valid,
                wheel_slip_speed,
                torch.zeros_like(wheel_slip_speed),
            )
            wheel_contact_count = torch.clamp(wheel_contact_valid.to(torch.float32).sum(dim=1), min=1.0)
            wheel_motion_count = torch.clamp(wheel_motion_valid.to(torch.float32).sum(dim=1), min=1.0)
            wheel_slip_mean = torch.sum(wheel_slip_ratio, dim=1) / wheel_contact_count
            wheel_slip_speed_mean = torch.sum(wheel_slip_speed_contact, dim=1) / wheel_contact_count
            wheel_slip_ratio_motion_gated_mean = torch.sum(wheel_slip_ratio_motion_gated, dim=1) / wheel_motion_count

            traction_params = self.cfg.rewards.loaded_wheel_longitudinal_slip.params
            traction_absolute_margin = float(traction_params.get("absolute_margin", 0.10))
            traction_relative_margin = float(traction_params.get("relative_margin", 0.15))
            traction_min_contact_force = float(traction_params.get("min_contact_force", 20.0))
            traction_min_motion_speed = float(traction_params.get("min_motion_speed", 0.20))
            wheel_surface_speed_abs = torch.abs(wheel_surface_speed)
            wheel_hub_forward_speed_abs = torch.abs(wheel_hub_forward_speed)
            wheel_surface_speed_abs_mean = torch.mean(wheel_surface_speed_abs, dim=1)
            wheel_hub_forward_speed_abs_mean = torch.mean(wheel_hub_forward_speed_abs, dim=1)
            traction_efficiency_valid = wheel_contact_valid & (wheel_surface_speed_abs > 0.20)
            traction_efficiency_valid_f = traction_efficiency_valid.to(torch.float32)
            traction_speed_efficiency = torch.where(
                traction_efficiency_valid,
                wheel_hub_forward_speed_abs / torch.clamp(wheel_surface_speed_abs, min=0.20),
                torch.zeros_like(wheel_surface_speed_abs),
            )
            traction_speed_efficiency = torch.clamp(traction_speed_efficiency, max=2.0)
            traction_speed_efficiency_mean = torch.sum(
                traction_speed_efficiency * traction_efficiency_valid_f, dim=1
            ) / torch.clamp(traction_efficiency_valid_f.sum(dim=1), min=1.0)
            traction_drive_efficiency = torch.clamp(traction_speed_efficiency, max=1.0)
            traction_drive_efficiency_mean = torch.sum(
                traction_drive_efficiency * traction_efficiency_valid_f, dim=1
            ) / torch.clamp(traction_efficiency_valid_f.sum(dim=1), min=1.0)
            left_goal_traction_speed_efficiency = masked_mean_or_zero(
                traction_speed_efficiency_mean, goal_left_mask
            )
            right_goal_traction_speed_efficiency = masked_mean_or_zero(
                traction_speed_efficiency_mean, goal_right_mask
            )
            traction_allowed_overspeed = (
                traction_absolute_margin + traction_relative_margin * wheel_hub_forward_speed_abs
            )
            traction_overspeed_excess = torch.relu(
                wheel_surface_speed_abs - wheel_hub_forward_speed_abs - traction_allowed_overspeed
            )
            traction_valid = (
                (contact_force_lb_lf_rf_rb > traction_min_contact_force)
                & (wheel_motion_scale > traction_min_motion_speed)
            )
            traction_valid_f = traction_valid.to(torch.float32)
            traction_overspeed_excess_mean = torch.sum(
                traction_overspeed_excess * traction_valid_f, dim=1
            ) / torch.clamp(traction_valid_f.sum(dim=1), min=1.0)

            def wheel_side_valid_mean(
                value: torch.Tensor,
                valid: torch.Tensor,
                start: int,
                end: int,
            ) -> torch.Tensor:
                value_side = value[:, start:end]
                valid_side = valid[:, start:end].to(value.dtype)
                return torch.sum(value_side * valid_side, dim=1) / torch.clamp(
                    valid_side.sum(dim=1), min=1.0
                )

            left_wheel_slip_ratio_motion_gated_mean = wheel_side_valid_mean(
                wheel_slip_ratio_motion_gated, wheel_motion_valid, 0, 2
            )
            right_wheel_slip_ratio_motion_gated_mean = wheel_side_valid_mean(
                wheel_slip_ratio_motion_gated, wheel_motion_valid, 2, 4
            )
            left_wheel_slip_speed_motion_gated_mean = wheel_side_valid_mean(
                wheel_slip_speed, wheel_motion_valid, 0, 2
            )
            right_wheel_slip_speed_motion_gated_mean = wheel_side_valid_mean(
                wheel_slip_speed, wheel_motion_valid, 2, 4
            )
            left_traction_overspeed_excess_mean = wheel_side_valid_mean(
                traction_overspeed_excess, traction_valid, 0, 2
            )
            right_traction_overspeed_excess_mean = wheel_side_valid_mean(
                traction_overspeed_excess, traction_valid, 2, 4
            )
            left_traction_drive_efficiency_mean = wheel_side_valid_mean(
                traction_drive_efficiency, traction_efficiency_valid, 0, 2
            )
            right_traction_drive_efficiency_mean = wheel_side_valid_mean(
                traction_drive_efficiency, traction_efficiency_valid, 2, 4
            )
            left_goal_wheel_slip = masked_mean_or_zero(wheel_slip_mean, goal_left_mask)
            right_goal_wheel_slip = masked_mean_or_zero(wheel_slip_mean, goal_right_mask)

            far_stage_mask = goal_distance > 3.0
            middle_stage_mask = (goal_distance > 1.0) & (goal_distance <= 3.0)
            approach_stage_mask = goal_distance <= 1.0
            all_loaded_wheels_contact = torch.all(
                contact_force_lb_lf_rf_rb > traction_min_contact_force, dim=1
            )
            any_loaded_wheel_moving = torch.any(wheel_motion_valid, dim=1)
            heading_aligned = torch.abs(heading_error) <= 0.20
            yaw_rate_small = torch.abs(root_ang_vel_b_z) <= 0.20
            aligned_cruise_mask = (
                far_stage_mask
                & heading_aligned
                & yaw_rate_small
                & (velocity_toward_goal > 0.50)
                & all_loaded_wheels_contact
            )
            far_turning_mask = (
                far_stage_mask
                & ((~heading_aligned) | (~yaw_rate_small))
                & all_loaded_wheels_contact
                & any_loaded_wheel_moving
            )
            far_other_mask = far_stage_mask & (~aligned_cruise_mask) & (~far_turning_mask)

            turn_command_abs = torch.abs(wheel_turn_mode_target_norm)
            turn_motion_active = (turn_command_abs >= 0.05) | (torch.abs(root_ang_vel_b_z) >= 0.12)
            directional_turn_base_mask = (
                far_stage_mask & all_loaded_wheels_contact & any_loaded_wheel_moving
            )
            near_straight_mask = (
                directional_turn_base_mask
                & (~turn_motion_active)
                & (velocity_toward_goal > 0.40)
            )
            left_turn_mask = (
                directional_turn_base_mask
                & turn_motion_active
                & (heading_error > 0.05)
            )
            right_turn_mask = (
                directional_turn_base_mask
                & turn_motion_active
                & (heading_error < -0.05)
            )
            left_turn_command_toward_goal = torch.relu(wheel_turn_mode_target_norm)
            right_turn_command_toward_goal = torch.relu(-wheel_turn_mode_target_norm)
            left_turn_actual_yaw_toward_goal = torch.relu(root_ang_vel_b_z)
            right_turn_actual_yaw_toward_goal = torch.relu(-root_ang_vel_b_z)
            left_turn_yaw_effectiveness = torch.where(
                left_turn_command_toward_goal > 0.02,
                left_turn_actual_yaw_toward_goal
                / torch.clamp(left_turn_command_toward_goal, min=0.02),
                torch.zeros_like(root_ang_vel_b_z),
            )
            right_turn_yaw_effectiveness = torch.where(
                right_turn_command_toward_goal > 0.02,
                right_turn_actual_yaw_toward_goal
                / torch.clamp(right_turn_command_toward_goal, min=0.02),
                torch.zeros_like(root_ang_vel_b_z),
            )
            left_turn_yaw_effectiveness = torch.clamp(left_turn_yaw_effectiveness, max=10.0)
            right_turn_yaw_effectiveness = torch.clamp(right_turn_yaw_effectiveness, max=10.0)
            wheel_antisymmetric_mode_mean = 0.5 * (
                wheel_left_front_rear_antisymmetric_mode + wheel_right_front_rear_antisymmetric_mode
            )
            wheel_opposite_sign_rate = 0.5 * (
                wheel_left_front_rear_opposite_sign_rate + wheel_right_front_rear_opposite_sign_rate
            )

            roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
            root_height = robot.data.root_pos_w[:, 2]
            stroke_max = stroke_actual_lb_lf_rf_rb.max(dim=1).values
            stroke_tracking_error_max = torch.max(
                torch.abs(leg_action_term.stroke_desired - leg_action_term.stroke_actual), dim=1
            ).values
            stop_posture_ready = (
                (torch.abs(roll) <= self._short_goal_stop_max_roll)
                & (torch.abs(pitch) <= self._short_goal_stop_max_pitch)
                & (stroke_tracking_error_max <= self._short_goal_stop_max_stroke_tracking_error)
            )
            stroke_above_0_55_rate = (stroke_max > 0.55).to(torch.float32)
            stroke_above_0_60_rate = (stroke_max > 0.60).to(torch.float32)

            safe_contact_total = torch.clamp(contact_force_total_raw, min=1.0e-6)
            wheel_contact_ratio_lb_lf_rf_rb = contact_force_lb_lf_rf_rb / safe_contact_total.unsqueeze(1)
            wheel_contact_min_ratio = wheel_contact_ratio_lb_lf_rf_rb.min(dim=1).values
            all_wheel_contact_rate = (contact_force_lb_lf_rf_rb.min(dim=1).values > 1.0).to(torch.float32)
            all_body_contact_sensor = self.scene.sensors["all_body_contact_forces"]
            if self._non_wheel_contact_body_ids.numel() > 0:
                non_wheel_force = torch.max(
                    torch.norm(
                        all_body_contact_sensor.data.net_forces_w_history[
                            :, :, self._non_wheel_contact_body_ids, :
                        ],
                        dim=-1,
                    ),
                    dim=1,
                )[0]
                non_wheel_contact_rate = (non_wheel_force.max(dim=1).values > 1.0).to(torch.float32)
            else:
                non_wheel_contact_rate = torch.zeros(
                    (self.num_envs,), dtype=torch.float32, device=self.device
                )

            yaw_wrong_direction_rate = (
                ((desired_yaw_sign * root_ang_vel_b_z) < 0.0).to(torch.float32) * yaw_active_mask
            )

            self._short_goal_hydraulic_action_prev[:] = leg_action_term.raw_actions

            path_length = self._short_goal_path_length_m
            path_efficiency = torch.where(
                path_length > 1.0e-6,
                torch.clamp(
                    (self._short_goal_initial_goal_distance_m - goal_distance) / torch.clamp(path_length, min=1.0e-6),
                    min=0.0,
                    max=1.0,
                ),
                torch.zeros_like(path_length),
            )

            return {
                "goal_distance_mean": goal_distance,
                "path_length_m": path_length,
                "path_efficiency": path_efficiency,
                "min_goal_distance_m": self._short_goal_min_goal_distance_m,
                "max_distance_rebound_after_min_m": self._short_goal_max_distance_rebound_after_min_m,
                "turn_reversal_count": self._short_goal_turn_reversal_count,
                "progress_mean": progress_mean,
                "left_goal_progress_mean": left_goal_progress,
                "right_goal_progress_mean": right_goal_progress,
                "success_rate": success_rate,
                "base_xy_vel_mean": base_xy_vel_mean,
                "base_lin_vel_x": base_lin_vel_x,
                "base_lin_vel_y": base_lin_vel_y,
                "base_yaw_rate": root_ang_vel_b_z,
                "desired_speed_mean": desired_speed,
                "alignment_speed_gate_mean": alignment_speed_gate,
                "speed_profile_excess_mean": speed_profile_excess,
                "speed_profile_penalty_mean": speed_profile_penalty,
                "turn_distance_gate_mean": turn_distance_gate,
                "near_goal_rate": near_goal.to(torch.float32),
                "near_goal_moving_rate": near_goal_moving.to(torch.float32),
                "stopped_success_rate": stopped_success.to(torch.float32),
                "left_goal_stopped_success_rate": left_goal_stopped_success,
                "right_goal_stopped_success_rate": right_goal_stopped_success,
                "stop_phase_active_rate": stop_phase_active_rate,
                "stop_phase_stable_fraction": stop_phase_stable_fraction,
                "stop_posture_roll_signed_mean": roll,
                "stop_posture_pitch_signed_mean": pitch,
                "stop_posture_stroke_lb_mean": stroke_actual_lb_lf_rf_rb[:, 0],
                "stop_posture_stroke_lf_mean": stroke_actual_lb_lf_rf_rb[:, 1],
                "stop_posture_stroke_rf_mean": stroke_actual_lb_lf_rf_rb[:, 2],
                "stop_posture_stroke_rb_mean": stroke_actual_lb_lf_rf_rb[:, 3],
                "stop_posture_stroke_range_mean": stroke_range_mean,
                "stop_posture_stroke_tracking_error_max_mean": stroke_tracking_error_max,
                "stop_posture_ready_rate": stop_posture_ready.to(torch.float32),
                "yaw_wrong_direction_rate": yaw_wrong_direction_rate,
                "wheel_target_tracking_error_lb_mean": wheel_target_tracking_error[:, 0],
                "wheel_target_tracking_error_lf_mean": wheel_target_tracking_error[:, 1],
                "wheel_target_tracking_error_rf_mean": wheel_target_tracking_error[:, 2],
                "wheel_target_tracking_error_rb_mean": wheel_target_tracking_error[:, 3],
                "wheel_common_mode_target_norm_mean": wheel_common_mode_target_norm,
                "wheel_turn_mode_target_norm_mean": wheel_turn_mode_target_norm,
                "left_goal_wheel_common_mode_norm_mean": left_goal_wheel_common_mode_norm,
                "right_goal_wheel_common_mode_norm_mean": right_goal_wheel_common_mode_norm,
                "left_goal_wheel_turn_mode_norm_mean": left_goal_wheel_turn_mode_norm,
                "right_goal_wheel_turn_mode_norm_mean": right_goal_wheel_turn_mode_norm,
                "left_goal_velocity_toward_goal_mean": left_goal_velocity_toward_goal,
                "right_goal_velocity_toward_goal_mean": right_goal_velocity_toward_goal,
                "wheel_left_front_rear_target_diff_norm_mean": wheel_left_front_rear_target_diff_norm,
                "wheel_right_front_rear_target_diff_norm_mean": wheel_right_front_rear_target_diff_norm,
                "wheel_left_front_rear_opposite_sign_rate": wheel_left_front_rear_opposite_sign_rate,
                "wheel_right_front_rear_opposite_sign_rate": wheel_right_front_rear_opposite_sign_rate,
                "wheel_left_front_rear_antisymmetric_mode_mean": wheel_left_front_rear_antisymmetric_mode,
                "wheel_right_front_rear_antisymmetric_mode_mean": wheel_right_front_rear_antisymmetric_mode,
                "far_antisymmetric_mode_mean": wheel_antisymmetric_mode_mean,
                "middle_antisymmetric_mode_mean": wheel_antisymmetric_mode_mean,
                "approach_antisymmetric_mode_mean": wheel_antisymmetric_mode_mean,
                "far_opposite_sign_rate": wheel_opposite_sign_rate,
                "middle_opposite_sign_rate": wheel_opposite_sign_rate,
                "approach_opposite_sign_rate": wheel_opposite_sign_rate,
                "wheel_target_rate_norm_mean": wheel_target_rate_norm,
                "wheel_action_saturation_rate": wheel_action_saturation_rate,
                "root_height_mean": root_height,
                "roll_abs_mean": torch.abs(roll),
                "pitch_abs_mean": torch.abs(pitch),
                "stroke_actual_range": stroke_range_mean,
                "stroke_max_mean": stroke_max,
                "stroke_above_0_55_rate": stroke_above_0_55_rate,
                "stroke_above_0_60_rate": stroke_above_0_60_rate,
                "wheel_contact_ratio_lb_mean": wheel_contact_ratio_lb_lf_rf_rb[:, 0],
                "wheel_contact_ratio_lf_mean": wheel_contact_ratio_lb_lf_rf_rb[:, 1],
                "wheel_contact_ratio_rf_mean": wheel_contact_ratio_lb_lf_rf_rb[:, 2],
                "wheel_contact_ratio_rb_mean": wheel_contact_ratio_lb_lf_rf_rb[:, 3],
                "wheel_contact_min_ratio_mean": wheel_contact_min_ratio,
                "all_wheel_contact_rate": all_wheel_contact_rate,
                "non_wheel_contact_rate": non_wheel_contact_rate,
                "wheel_target_abs_mean": wheel_target_abs_mean,
                "wheel_joint_vel_abs_mean": wheel_joint_vel_abs_mean,
                "raw_wheel_action_abs_mean": raw_wheel_action_abs_mean,
                "raw_wheel_action_lb_mean": wheel_action_term.raw_actions[:, 0],
                "raw_wheel_action_lf_mean": wheel_action_term.raw_actions[:, 1],
                "raw_wheel_action_rf_mean": wheel_action_term.raw_actions[:, 2],
                "raw_wheel_action_rb_mean": wheel_action_term.raw_actions[:, 3],
                "wheel_target_lb_mean": wheel_velocity_target[:, 0],
                "wheel_target_lf_mean": wheel_velocity_target[:, 1],
                "wheel_target_rf_mean": wheel_velocity_target[:, 2],
                "wheel_target_rb_mean": wheel_velocity_target[:, 3],
                "semantic_wheel_target_lb_mean": semantic_wheel_velocity_target[:, 0],
                "semantic_wheel_target_lf_mean": semantic_wheel_velocity_target[:, 1],
                "semantic_wheel_target_rf_mean": semantic_wheel_velocity_target[:, 2],
                "semantic_wheel_target_rb_mean": semantic_wheel_velocity_target[:, 3],
                "wheel_joint_vel_lb_mean": wheel_joint_vel[:, 0],
                "wheel_joint_vel_lf_mean": wheel_joint_vel[:, 1],
                "wheel_joint_vel_rf_mean": wheel_joint_vel[:, 2],
                "wheel_joint_vel_rb_mean": wheel_joint_vel[:, 3],
                "semantic_wheel_joint_vel_lb_mean": semantic_wheel_joint_vel[:, 0],
                "semantic_wheel_joint_vel_lf_mean": semantic_wheel_joint_vel[:, 1],
                "semantic_wheel_joint_vel_rf_mean": semantic_wheel_joint_vel[:, 2],
                "semantic_wheel_joint_vel_rb_mean": semantic_wheel_joint_vel[:, 3],
                "wheel_slip_ratio_lb_mean": wheel_slip_ratio[:, 0],
                "wheel_slip_ratio_lf_mean": wheel_slip_ratio[:, 1],
                "wheel_slip_ratio_rf_mean": wheel_slip_ratio[:, 2],
                "wheel_slip_ratio_rb_mean": wheel_slip_ratio[:, 3],
                "wheel_slip_speed_lb_mean_mps": wheel_slip_speed_contact[:, 0],
                "wheel_slip_speed_lf_mean_mps": wheel_slip_speed_contact[:, 1],
                "wheel_slip_speed_rf_mean_mps": wheel_slip_speed_contact[:, 2],
                "wheel_slip_speed_rb_mean_mps": wheel_slip_speed_contact[:, 3],
                "wheel_surface_speed_abs_mean_mps": wheel_surface_speed_abs_mean,
                "wheel_hub_forward_speed_abs_mean_mps": wheel_hub_forward_speed_abs_mean,
                "traction_speed_efficiency_mean": traction_speed_efficiency_mean,
                "left_goal_traction_speed_efficiency_mean": left_goal_traction_speed_efficiency,
                "right_goal_traction_speed_efficiency_mean": right_goal_traction_speed_efficiency,
                "left_goal_wheel_slip_mean": left_goal_wheel_slip,
                "right_goal_wheel_slip_mean": right_goal_wheel_slip,
                "far_wheel_slip_mean": wheel_slip_mean,
                "middle_wheel_slip_mean": wheel_slip_mean,
                "approach_wheel_slip_mean": wheel_slip_mean,
                "far_wheel_slip_speed_mean_mps": wheel_slip_speed_mean,
                "middle_wheel_slip_speed_mean_mps": wheel_slip_speed_mean,
                "approach_wheel_slip_speed_mean_mps": wheel_slip_speed_mean,
                "far_wheel_slip_ratio_motion_gated_mean": wheel_slip_ratio_motion_gated_mean,
                "middle_wheel_slip_ratio_motion_gated_mean": wheel_slip_ratio_motion_gated_mean,
                "approach_wheel_slip_ratio_motion_gated_mean": wheel_slip_ratio_motion_gated_mean,
                "far_traction_overspeed_excess_mean_mps": traction_overspeed_excess_mean,
                "middle_traction_overspeed_excess_mean_mps": traction_overspeed_excess_mean,
                "approach_traction_overspeed_excess_mean_mps": traction_overspeed_excess_mean,
                "aligned_cruise_wheel_surface_speed_abs_mean_mps": wheel_surface_speed_abs_mean,
                "aligned_cruise_wheel_hub_forward_speed_abs_mean_mps": wheel_hub_forward_speed_abs_mean,
                "aligned_cruise_traction_drive_efficiency_mean": traction_drive_efficiency_mean,
                "aligned_cruise_wheel_slip_ratio_motion_gated_mean": wheel_slip_ratio_motion_gated_mean,
                "aligned_cruise_wheel_slip_speed_mean_mps": wheel_slip_speed_mean,
                "aligned_cruise_traction_overspeed_excess_mean_mps": traction_overspeed_excess_mean,
                "far_turning_wheel_surface_speed_abs_mean_mps": wheel_surface_speed_abs_mean,
                "far_turning_wheel_hub_forward_speed_abs_mean_mps": wheel_hub_forward_speed_abs_mean,
                "far_turning_traction_drive_efficiency_mean": traction_drive_efficiency_mean,
                "far_turning_wheel_slip_ratio_motion_gated_mean": wheel_slip_ratio_motion_gated_mean,
                "far_turning_wheel_slip_speed_mean_mps": wheel_slip_speed_mean,
                "far_turning_traction_overspeed_excess_mean_mps": traction_overspeed_excess_mean,
                "near_straight_traction_drive_efficiency_mean": traction_drive_efficiency_mean,
                "near_straight_wheel_slip_ratio_motion_gated_mean": wheel_slip_ratio_motion_gated_mean,
                "near_straight_wheel_slip_speed_mean_mps": wheel_slip_speed_mean,
                "near_straight_traction_overspeed_excess_mean_mps": traction_overspeed_excess_mean,
                "left_turn_wheel_turn_mode_toward_goal_mean": left_turn_command_toward_goal,
                "right_turn_wheel_turn_mode_toward_goal_mean": right_turn_command_toward_goal,
                "left_turn_actual_yaw_rate_toward_goal_mean": left_turn_actual_yaw_toward_goal,
                "right_turn_actual_yaw_rate_toward_goal_mean": right_turn_actual_yaw_toward_goal,
                "left_turn_yaw_effectiveness_mean": left_turn_yaw_effectiveness,
                "right_turn_yaw_effectiveness_mean": right_turn_yaw_effectiveness,
                "left_turn_inner_left_wheel_slip_ratio_motion_gated_mean": left_wheel_slip_ratio_motion_gated_mean,
                "left_turn_outer_right_wheel_slip_ratio_motion_gated_mean": right_wheel_slip_ratio_motion_gated_mean,
                "right_turn_outer_left_wheel_slip_ratio_motion_gated_mean": left_wheel_slip_ratio_motion_gated_mean,
                "right_turn_inner_right_wheel_slip_ratio_motion_gated_mean": right_wheel_slip_ratio_motion_gated_mean,
                "left_turn_inner_left_wheel_slip_speed_mean_mps": left_wheel_slip_speed_motion_gated_mean,
                "left_turn_outer_right_wheel_slip_speed_mean_mps": right_wheel_slip_speed_motion_gated_mean,
                "right_turn_outer_left_wheel_slip_speed_mean_mps": left_wheel_slip_speed_motion_gated_mean,
                "right_turn_inner_right_wheel_slip_speed_mean_mps": right_wheel_slip_speed_motion_gated_mean,
                "left_turn_inner_left_traction_overspeed_excess_mean_mps": left_traction_overspeed_excess_mean,
                "left_turn_outer_right_traction_overspeed_excess_mean_mps": right_traction_overspeed_excess_mean,
                "right_turn_outer_left_traction_overspeed_excess_mean_mps": left_traction_overspeed_excess_mean,
                "right_turn_inner_right_traction_overspeed_excess_mean_mps": right_traction_overspeed_excess_mean,
                "left_turn_inner_left_traction_drive_efficiency_mean": left_traction_drive_efficiency_mean,
                "left_turn_outer_right_traction_drive_efficiency_mean": right_traction_drive_efficiency_mean,
                "right_turn_outer_left_traction_drive_efficiency_mean": left_traction_drive_efficiency_mean,
                "right_turn_inner_right_traction_drive_efficiency_mean": right_traction_drive_efficiency_mean,
                "_aligned_cruise_mask": aligned_cruise_mask,
                "_far_turning_mask": far_turning_mask,
                "_near_straight_mask": near_straight_mask,
                "_left_turn_mask": left_turn_mask,
                "_right_turn_mask": right_turn_mask,
                "policy_hydraulic_action_abs_mean": policy_hydraulic_action_abs_mean,
                "hydraulic_action_abs_mean": hydraulic_action_abs_mean,
                "hydraulic_action_rate_abs_mean": hydraulic_action_rate_abs_mean,
                "hydraulic_raw_action_abs_mean": hydraulic_raw_action_abs_mean,
                "hydraulic_stroke_desired_mean": stroke_desired_lb_lf_rf_rb.mean(dim=1),
                "hydraulic_stroke_desired_range": stroke_desired_range,
                "hydraulic_stroke_actual_mean": stroke_actual_lb_lf_rf_rb.mean(dim=1),
                "hydraulic_stroke_actual_range": stroke_range_mean,
                "hydraulic_position_target_mean": hydraulic_position_target_lb_lf_rf_rb.mean(dim=1),
                "hydraulic_joint_position_target_mean": hydraulic_joint_position_target_lb_lf_rf_rb.mean(dim=1),
                "hydraulic_joint_position_mean": hydraulic_joint_position_lb_lf_rf_rb.mean(dim=1),
                "hydraulic_joint_velocity_abs_mean": torch.mean(torch.abs(hydraulic_joint_velocity_lb_lf_rf_rb), dim=1),
                "hydraulic_joint_position_tracking_error_mean": hydraulic_joint_position_tracking_error.mean(dim=1),
                "hydraulic_action_lb_mean": final_hydraulic_action_lb_lf_rf_rb[:, 0],
                "hydraulic_action_lf_mean": final_hydraulic_action_lb_lf_rf_rb[:, 1],
                "hydraulic_action_rf_mean": final_hydraulic_action_lb_lf_rf_rb[:, 2],
                "hydraulic_action_rb_mean": final_hydraulic_action_lb_lf_rf_rb[:, 3],
                "stroke_desired_lb_mean": stroke_desired_lb_lf_rf_rb[:, 0],
                "stroke_desired_lf_mean": stroke_desired_lb_lf_rf_rb[:, 1],
                "stroke_desired_rf_mean": stroke_desired_lb_lf_rf_rb[:, 2],
                "stroke_desired_rb_mean": stroke_desired_lb_lf_rf_rb[:, 3],
                "stroke_actual_lb_mean": stroke_actual_lb_lf_rf_rb[:, 0],
                "stroke_actual_lf_mean": stroke_actual_lb_lf_rf_rb[:, 1],
                "stroke_actual_rf_mean": stroke_actual_lb_lf_rf_rb[:, 2],
                "stroke_actual_rb_mean": stroke_actual_lb_lf_rf_rb[:, 3],
                "short_goal_hydraulic_mode_debug": short_goal_hydraulic_mode_debug,
                "support_hold_active_debug": support_hold_active_debug,
                "free_policy_control_active_debug": free_policy_control_active_debug,
                "hydraulic_action_diag_mode_mean": hydraulic_action_diag_mode,
                "hydraulic_action_diag_mode_abs_mean": torch.abs(hydraulic_action_diag_mode),
                "stroke_diag_mode_mean": stroke_diag_mode,
                "stroke_diag_mode_abs_mean": torch.abs(stroke_diag_mode),
                "contact_diag_force_mean": contact_diag_force,
                "contact_diag_force_abs_mean": torch.abs(contact_diag_force),
                "contact_diag_ratio_mean": contact_diag_ratio,
                "contact_diag_ratio_abs_mean": torch.abs(contact_diag_ratio),
                "action_stroke_diag_correlation": action_stroke_diag_correlation,
                "action_contact_diag_correlation": action_contact_diag_correlation,
                "stroke_contact_diag_correlation": stroke_contact_diag_correlation,
                "support_hold_action_abs_mean": support_hold_action_abs_mean,
                "support_hold_action_range": support_hold_action_range,
                "support_hold_mode_debug": support_hold_mode_debug,
                "executed_hydraulic_action_abs_mean": executed_hydraulic_action_abs_mean,
                "support_hold_contact_correction_abs_mean": support_hold_contact_correction_abs_mean,
                "support_hold_contact_correction_max_abs": support_hold_contact_correction_max_abs,
                "support_hold_stroke_correction_abs_mean": support_hold_stroke_correction_abs_mean,
                "support_hold_roll_pitch_correction_abs_mean": support_hold_roll_pitch_correction_abs_mean,
                "support_hold_total_pre_filter_abs_mean": support_hold_total_pre_filter_abs_mean,
                "support_hold_total_post_filter_abs_mean": support_hold_total_post_filter_abs_mean,
                "support_hold_stroke_lb_mean": stroke_actual_lb_lf_rf_rb[:, 0],
                "support_hold_stroke_lf_mean": stroke_actual_lb_lf_rf_rb[:, 1],
                "support_hold_stroke_rf_mean": stroke_actual_lb_lf_rf_rb[:, 2],
                "support_hold_stroke_rb_mean": stroke_actual_lb_lf_rf_rb[:, 3],
                "support_hold_stroke_nominal_correction_lb_mean": self._yaw_turn_support_stroke_nominal_correction_semantic[:, 0],
                "support_hold_stroke_nominal_correction_lf_mean": self._yaw_turn_support_stroke_nominal_correction_semantic[:, 1],
                "support_hold_stroke_nominal_correction_rf_mean": self._yaw_turn_support_stroke_nominal_correction_semantic[:, 2],
                "support_hold_stroke_nominal_correction_rb_mean": self._yaw_turn_support_stroke_nominal_correction_semantic[:, 3],
                "support_hold_stroke_range_correction_lb_mean": self._yaw_turn_support_stroke_range_correction_semantic[:, 0],
                "support_hold_stroke_range_correction_lf_mean": self._yaw_turn_support_stroke_range_correction_semantic[:, 1],
                "support_hold_stroke_range_correction_rf_mean": self._yaw_turn_support_stroke_range_correction_semantic[:, 2],
                "support_hold_stroke_range_correction_rb_mean": self._yaw_turn_support_stroke_range_correction_semantic[:, 3],
                "support_hold_diag_correction_lb_mean": self._yaw_turn_support_diag_correction_semantic[:, 0],
                "support_hold_diag_correction_lf_mean": self._yaw_turn_support_diag_correction_semantic[:, 1],
                "support_hold_diag_correction_rf_mean": self._yaw_turn_support_diag_correction_semantic[:, 2],
                "support_hold_diag_correction_rb_mean": self._yaw_turn_support_diag_correction_semantic[:, 3],
                "support_hold_contact_correction_lb_mean": self._yaw_turn_support_contact_correction_semantic[:, 0],
                "support_hold_contact_correction_lf_mean": self._yaw_turn_support_contact_correction_semantic[:, 1],
                "support_hold_contact_correction_rf_mean": self._yaw_turn_support_contact_correction_semantic[:, 2],
                "support_hold_contact_correction_rb_mean": self._yaw_turn_support_contact_correction_semantic[:, 3],
                "support_hold_roll_pitch_correction_lb_mean": self._yaw_turn_support_roll_pitch_correction_semantic[:, 0],
                "support_hold_roll_pitch_correction_lf_mean": self._yaw_turn_support_roll_pitch_correction_semantic[:, 1],
                "support_hold_roll_pitch_correction_rf_mean": self._yaw_turn_support_roll_pitch_correction_semantic[:, 2],
                "support_hold_roll_pitch_correction_rb_mean": self._yaw_turn_support_roll_pitch_correction_semantic[:, 3],
                "support_hold_non_contact_correction_lb_mean": self._yaw_turn_support_non_contact_correction_semantic[:, 0],
                "support_hold_non_contact_correction_lf_mean": self._yaw_turn_support_non_contact_correction_semantic[:, 1],
                "support_hold_non_contact_correction_rf_mean": self._yaw_turn_support_non_contact_correction_semantic[:, 2],
                "support_hold_non_contact_correction_rb_mean": self._yaw_turn_support_non_contact_correction_semantic[:, 3],
                "support_hold_contact_cancelled_lf_mean": self._yaw_turn_support_contact_cancellation_semantic[:, 1],
                "support_hold_contact_cancelled_rb_mean": self._yaw_turn_support_contact_cancellation_semantic[:, 3],
                "support_hold_contact_cancelled_abs_mean": torch.mean(
                    torch.abs(self._yaw_turn_support_contact_cancellation_semantic), dim=1
                ),
                "executed_hydraulic_action_lb_mean": executed_hydraulic_action_lb_lf_rf_rb[:, 0],
                "executed_hydraulic_action_lf_mean": executed_hydraulic_action_lb_lf_rf_rb[:, 1],
                "executed_hydraulic_action_rf_mean": executed_hydraulic_action_lb_lf_rf_rb[:, 2],
                "executed_hydraulic_action_rb_mean": executed_hydraulic_action_lb_lf_rf_rb[:, 3],
                "support_hold_total_pre_filter_lb_mean": self._yaw_turn_support_total_pre_filter_semantic[:, 0],
                "support_hold_total_pre_filter_lf_mean": self._yaw_turn_support_total_pre_filter_semantic[:, 1],
                "support_hold_total_pre_filter_rf_mean": self._yaw_turn_support_total_pre_filter_semantic[:, 2],
                "support_hold_total_pre_filter_rb_mean": self._yaw_turn_support_total_pre_filter_semantic[:, 3],
                "wrong_direction_yaw_mean": wrong_direction_yaw_mean,
                "forward_velocity_during_turn": torch.abs(base_lin_vel_x),
                "hydraulic_mode_debug": hydraulic_mode_debug,
                "stroke_range_mean": stroke_range_mean,
                "stroke_diagonal_balance": stroke_diagonal_balance,
                "contact_force_imbalance": contact_force_imbalance,
                "min_wheel_contact_force_mean": min_wheel_contact_force_mean,
                "zero_contact_ratio": zero_contact_ratio,
                "wheel_contact_force_lb_mean": contact_force_lb_lf_rf_rb[:, 0],
                "wheel_contact_force_lf_mean": contact_force_lb_lf_rf_rb[:, 1],
                "wheel_contact_force_rf_mean": contact_force_lb_lf_rf_rb[:, 2],
                "wheel_contact_force_rb_mean": contact_force_lb_lf_rf_rb[:, 3],
                "wheel_common_mode_target_abs_mean": torch.abs(wheel_common_mode_target),
                "wheel_differential_target_mean": wheel_differential_target,
                "wheel_differential_target_abs_mean": torch.abs(wheel_differential_target),
                "goal_x_body_mean": target_vec_b[:, 0],
                "goal_y_body_mean": target_vec_b[:, 1],
                "goal_angle_body_mean": goal_angle_body,
                "goal_angle_body_abs_mean": torch.abs(goal_angle_body),
                "goal_left_rate": (goal_angle_body > 0.0).to(torch.float32),
                "goal_right_rate": (goal_angle_body < 0.0).to(torch.float32),
                "goal_front_rate": (target_vec_b[:, 0] > 0.0).to(torch.float32),
                "heading_error_abs_mean": torch.abs(heading_error),
                "heading_error_signed_mean": heading_error,
                "heading_error_reduction_mean": heading_error_reduction_mean,
                "actual_yaw_rate_abs_mean": actual_yaw_rate_abs,
                "left_goal_actual_yaw_rate_signed_mean": left_goal_actual_yaw_rate_signed,
                "right_goal_actual_yaw_rate_signed_mean": right_goal_actual_yaw_rate_signed,
                "yaw_rate_tracking_error_abs_mean": yaw_rate_tracking_error_abs,
                "yaw_rate_response_ratio_mean": yaw_rate_response_ratio,
                "left_goal_yaw_rate_tracking_error_abs_mean": left_goal_yaw_rate_tracking_error_abs,
                "right_goal_yaw_rate_tracking_error_abs_mean": right_goal_yaw_rate_tracking_error_abs,
                "stage_far_rate": far_stage_mask.to(torch.float32),
                "stage_middle_rate": middle_stage_mask.to(torch.float32),
                "stage_approach_rate": approach_stage_mask.to(torch.float32),
                "stage_aligned_cruise_rate": aligned_cruise_mask.to(torch.float32),
                "stage_far_turning_rate": far_turning_mask.to(torch.float32),
                "stage_far_other_rate": far_other_mask.to(torch.float32),
                "stage_near_straight_rate": near_straight_mask.to(torch.float32),
                "stage_left_turn_rate": left_turn_mask.to(torch.float32),
                "stage_right_turn_rate": right_turn_mask.to(torch.float32),
                "far_heading_error_abs_mean": torch.abs(heading_error),
                "middle_heading_error_abs_mean": torch.abs(heading_error),
                "approach_heading_error_abs_mean": torch.abs(heading_error),
                "far_heading_error_signed_mean": heading_error,
                "middle_heading_error_signed_mean": heading_error,
                "approach_heading_error_signed_mean": heading_error,
                "left_goal_heading_error_signed_mean": left_goal_heading_error_signed,
                "right_goal_heading_error_signed_mean": right_goal_heading_error_signed,
                "far_velocity_toward_goal_mean": velocity_toward_goal,
                "middle_velocity_toward_goal_mean": velocity_toward_goal,
                "approach_velocity_toward_goal_mean": velocity_toward_goal,
                "far_wheel_common_mode_norm_mean": wheel_common_mode_target_norm,
                "middle_wheel_common_mode_norm_mean": wheel_common_mode_target_norm,
                "approach_wheel_common_mode_norm_mean": wheel_common_mode_target_norm,
                "far_wheel_turn_mode_norm_mean": wheel_turn_mode_target_norm,
                "middle_wheel_turn_mode_norm_mean": wheel_turn_mode_target_norm,
                "approach_wheel_turn_mode_norm_mean": wheel_turn_mode_target_norm,
                "turn_toward_goal_mean": turn_toward_goal_mean,
                "signed_yaw_rate_tracking_mean": signed_yaw_rate_tracking_mean,
                "too_small_yaw_rate_when_error_large_mean": too_small_yaw_rate_when_error_large_mean,
                "desired_yaw_rate_mean": desired_yaw_rate_continuous,
                "desired_yaw_rate_abs_mean": torch.abs(desired_yaw_rate_continuous),
                "desired_yaw_sign_mean": torch.sign(desired_yaw_rate_continuous),
                "yaw_rate_error_mean": root_ang_vel_b_z - desired_yaw_rate_continuous,
                "yaw_rate_error_abs_mean": yaw_rate_tracking_error_abs,
                "yaw_active_rate": yaw_active_mask,
                "yaw_correct_direction_rate": yaw_correct_direction_rate,
                "yaw_too_small_rate": yaw_too_small_rate,
                "left_yaw_correct_direction_rate": left_yaw_correct_direction_rate,
                "right_yaw_correct_direction_rate": right_yaw_correct_direction_rate,
                "left_yaw_too_small_rate": left_yaw_too_small_rate,
                "right_yaw_too_small_rate": right_yaw_too_small_rate,
                "left_signed_yaw_rate_mean": masked_mean_or_zero(signed_yaw, goal_left_mask),
                "right_signed_yaw_rate_mean": masked_mean_or_zero(signed_yaw, goal_right_mask),
                "wheel_velocity_target_mean": wheel_velocity_target_mean,
                "semantic_wheel_velocity_target_mean": semantic_wheel_velocity_target_mean,
                "wheel_joint_vel_mean": wheel_joint_vel_mean,
                "semantic_wheel_joint_vel_mean": semantic_wheel_joint_vel_mean,
                "wheel_left_target_mean": wheel_left_target_mean,
                "wheel_right_target_mean": wheel_right_target_mean,
                "left_turn_yaw_rate_mean": masked_mean_or_zero(root_ang_vel_b_z, goal_left_mask),
                "right_turn_yaw_rate_mean": masked_mean_or_zero(root_ang_vel_b_z, goal_right_mask),
                "left_goal_success_rate": masked_mean_or_zero(success_rate, goal_left_mask),
                "right_goal_success_rate": masked_mean_or_zero(success_rate, goal_right_mask),
                "bad_orientation_rate": bad_orientation_rate,
            }

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

    def _update_short_goal_trajectory_quality(self, active_mask: torch.Tensor | None = None) -> None:
        """Update per-episode XY path and turn-sequence diagnostics without affecting rewards."""

        if not self._is_short_goal_task:
            return
        if active_mask is None:
            active_mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        else:
            active_mask = active_mask.to(device=self.device, dtype=torch.bool)

        robot = self.scene["robot"]
        current_root_xy = robot.data.root_pos_w[:, :2].detach()
        step_distance = torch.linalg.vector_norm(current_root_xy - self._short_goal_prev_root_xy, dim=1)
        self._short_goal_path_length_m[active_mask] += step_distance[active_mask]
        self._short_goal_prev_root_xy[:] = current_root_xy

        _, goal_distance, _ = mdp.short_goal_target_body(self)
        uninitialized = self._short_goal_initial_goal_distance_m <= 0.0
        if torch.any(uninitialized):
            self._short_goal_initial_goal_distance_m[uninitialized] = goal_distance[uninitialized]
            self._short_goal_min_goal_distance_m[uninitialized] = goal_distance[uninitialized]

        updated_min = torch.minimum(self._short_goal_min_goal_distance_m, goal_distance)
        self._short_goal_min_goal_distance_m[active_mask] = updated_min[active_mask]
        rebound = torch.clamp(goal_distance - updated_min, min=0.0)
        self._short_goal_max_distance_rebound_after_min_m[active_mask] = torch.maximum(
            self._short_goal_max_distance_rebound_after_min_m[active_mask],
            rebound[active_mask],
        )

        wheel_action_term = self.action_manager.get_term("wheel_motor_csv")
        semantic_wheel_target = wheel_action_term.velocity_target * self._wheel_forward_sign
        semantic_left_target = semantic_wheel_target[:, :2].mean(dim=1)
        semantic_right_target = semantic_wheel_target[:, 2:].mean(dim=1)
        wheel_velocity_limit = max(float(getattr(wheel_action_term, "_velocity_limit", 1.0)), 1.0e-6)
        turn_mode = 0.5 * (semantic_right_target - semantic_left_target) / wheel_velocity_limit
        deadband = float(self._short_goal_turn_reversal_deadband)
        current_turn_sign = torch.where(
            turn_mode > deadband,
            torch.ones_like(turn_mode),
            torch.where(turn_mode < -deadband, -torch.ones_like(turn_mode), torch.zeros_like(turn_mode)),
        )
        significant_turn = active_mask & (current_turn_sign != 0.0)
        reversal = (
            significant_turn
            & (self._short_goal_last_turn_sign != 0.0)
            & (current_turn_sign != self._short_goal_last_turn_sign)
        )
        self._short_goal_turn_reversal_count[reversal] += 1.0
        self._short_goal_last_turn_sign[significant_turn] = current_turn_sign[significant_turn]

    def _accumulate_forward_debug_metrics(self) -> None:
        metric_values = self._compute_forward_debug_metric_values()
        if self._is_stand_training_task:
            root_height_low_mask = metric_values["stand/root_height_low"].to(torch.bool)
            low_height_names = {
                "stand/root_height_when_low",
                "stand/low_height_hydraulic_stroke_lf",
                "stand/low_height_hydraulic_stroke_lr",
                "stand/low_height_hydraulic_stroke_rf",
                "stand/low_height_hydraulic_stroke_rr",
                "stand/low_height_hydraulic_action_lf",
                "stand/low_height_hydraulic_action_lr",
                "stand/low_height_hydraulic_action_rf",
                "stand/low_height_hydraulic_action_rr",
                "stand/low_height_low_stroke_negative_action_penalty",
                "stand/low_height_height_gated_low_stroke_negative_action_penalty",
                "stand/low_height_low_stroke_negative_action_lf",
                "stand/low_height_low_stroke_negative_action_lr",
                "stand/low_height_low_stroke_negative_action_rf",
                "stand/low_height_low_stroke_negative_action_rr",
                "stand/low_height_height_gated_low_stroke_negative_action_lf",
                "stand/low_height_height_gated_low_stroke_negative_action_lr",
                "stand/low_height_height_gated_low_stroke_negative_action_rf",
                "stand/low_height_height_gated_low_stroke_negative_action_rr",
                "stand/low_height_positive_action_rate_lf",
                "stand/low_height_positive_action_rate_lr",
                "stand/low_height_positive_action_rate_rf",
                "stand/low_height_positive_action_rate_rr",
                "stand/low_height_height_low_high_stroke_positive_action_penalty",
                "stand/low_height_stroke_away_from_nominal_action_penalty",
                "stand/low_height_front_mean_stroke",
                "stand/low_height_rear_mean_stroke",
                "stand/low_height_front_rear_stroke_diff",
                "stand/low_height_abs_front_rear_stroke_diff",
                "stand/low_height_front_rear_stroke_diff_over_threshold",
                "stand/low_height_front_rear_stroke_balance_penalty",
                "stand/low_height_front_rear_stroke_split_action_penalty",
                "stand/low_height_front_negative_hydraulic_action_mean",
                "stand/low_height_rear_positive_hydraulic_action_mean",
                "stand/low_height_front_positive_hydraulic_action_mean",
                "stand/low_height_rear_negative_hydraulic_action_mean",
                "stand/low_height_contact_force_lf",
                "stand/low_height_contact_force_lr",
                "stand/low_height_contact_force_rf",
                "stand/low_height_contact_force_rr",
                "stand/low_height_contact_force_total",
                "stand/low_height_contact_force_ratio_lf",
                "stand/low_height_contact_force_ratio_lr",
                "stand/low_height_contact_force_ratio_rf",
                "stand/low_height_contact_force_ratio_rr",
                "stand/low_height_contact_force_min_ratio",
                "stand/low_height_contact_force_ratio_sum",
                "stand/low_height_base_xy_speed",
                "stand/low_height_base_roll",
                "stand/low_height_base_pitch",
                "stand/low_height_min_contact_force",
            }
            ratio_valid_names = {
                "stand/contact_force_total_from_ratios_source",
                "stand/contact_force_ratio_lf",
                "stand/contact_force_ratio_lr",
                "stand/contact_force_ratio_rf",
                "stand/contact_force_ratio_rr",
                "stand/contact_force_min_ratio",
                "stand/contact_force_ratio_sum",
            }
            low_height_ratio_valid_names = {
                "stand/low_height_contact_force_ratio_lf",
                "stand/low_height_contact_force_ratio_lr",
                "stand/low_height_contact_force_ratio_rf",
                "stand/low_height_contact_force_ratio_rr",
                "stand/low_height_contact_force_min_ratio",
                "stand/low_height_contact_force_ratio_sum",
            }
            ratio_valid_mask = metric_values["stand/contact_force_ratio_valid_rate"].to(torch.bool)
            for name, value in metric_values.items():
                mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
                if name in low_height_names:
                    mask = root_height_low_mask
                if name in ratio_valid_names:
                    mask = mask & ratio_valid_mask
                if name in low_height_ratio_valid_names:
                    mask = mask & ratio_valid_mask
                if torch.any(mask):
                    self._forward_debug_metric_sums[name][mask] += value[mask]
                    self._forward_debug_metric_counts_by_name[name][mask] += 1.0
            self._stand_recent_reset_happened.zero_()
            return
        if self._is_turn_to_target_task and self._enable_reset_settle and self._reset_settle_steps > 0:
            active_mask = ~self._reset_settle_step_active
            always_names = {
                "turn/contact_force_diag_diff",
                "turn/contact_force_front_rear_diff",
                "turn/contact_force_left_right_diff",
                "turn/base_roll",
                "turn/base_pitch",
                "turn/root_height",
                "turn/hydraulic_stroke_lf",
                "turn/hydraulic_stroke_lr",
                "turn/hydraulic_stroke_rf",
                "turn/hydraulic_stroke_rr",
                "turn/reset_settling",
                "turn/reset_settle_steps_remaining",
            }
            episode_end_names = {
                "turn/termination_any",
                "turn/termination_time_out",
                "turn/termination_bad_orientation",
                "turn/termination_root_height_low",
                "turn/termination_other",
            }
            reset_mask = self.reset_buf.to(torch.bool)
            for name, value in metric_values.items():
                if name in episode_end_names or name.startswith("turn/termination_term/"):
                    mask = reset_mask
                elif name in always_names:
                    mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
                else:
                    mask = active_mask
                if torch.any(mask):
                    self._forward_debug_metric_sums[name][mask] += value[mask]
                    self._forward_debug_metric_counts_by_name[name][mask] += 1.0
            return
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
        if self._is_short_goal_task:
            goal_distance = metric_values["goal_distance_mean"]
            episode_end_names = {
                "path_length_m",
                "path_efficiency",
                "min_goal_distance_m",
                "max_distance_rebound_after_min_m",
                "turn_reversal_count",
            }
            stage_masks = {
                # Check specific prefixes before the generic far_ prefix.
                "aligned_cruise_": metric_values["_aligned_cruise_mask"].to(torch.bool),
                "near_straight_": metric_values["_near_straight_mask"].to(torch.bool),
                "left_turn_": metric_values["_left_turn_mask"].to(torch.bool),
                "right_turn_": metric_values["_right_turn_mask"].to(torch.bool),
                "far_turning_": metric_values["_far_turning_mask"].to(torch.bool),
                "stop_posture_": self._short_goal_stop_phase_active,
                "far_": goal_distance > 3.0,
                "middle_": (goal_distance > 1.0) & (goal_distance <= 3.0),
                "approach_": goal_distance <= 1.0,
            }
            reset_mask = self.reset_buf.to(torch.bool)
            for name in self._forward_debug_metric_names:
                value = metric_values[name]
                if name in episode_end_names:
                    mask = reset_mask
                else:
                    mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
                    for prefix, stage_mask in stage_masks.items():
                        if name.startswith(prefix):
                            mask = stage_mask
                            break
                if torch.any(mask):
                    self._forward_debug_metric_sums[name][mask] += value[mask]
                    self._forward_debug_metric_counts_by_name[name][mask] += 1.0
            self._forward_debug_metric_counts += 1.0
            return
        for name in self._forward_debug_metric_names:
            value = metric_values[name]
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
            raw_counts = self._forward_debug_metric_counts_by_name[name][env_ids]
            valid_envs = raw_counts > 0.0
            if torch.any(valid_envs):
                per_env_mean = (
                    self._forward_debug_metric_sums[name][env_ids][valid_envs]
                    / raw_counts[valid_envs]
                )
                metric_mean = per_env_mean.mean()
            else:
                metric_mean = torch.zeros((), dtype=torch.float32, device=self.device)
            if name.startswith("turn_sanity/"):
                log_name = f"Metrics/{name}"
            elif name.startswith("turn/"):
                log_name = f"Metrics/{name}"
            else:
                log_name = f"Metrics/{self._debug_log_prefix}/{name}"
            logs[log_name] = float(metric_mean.item())
            self._forward_debug_metric_sums[name][env_ids] = 0.0
            self._forward_debug_metric_counts_by_name[name][env_ids] = 0.0
        self._forward_debug_metric_counts[env_ids] = 0.0
        return logs

    def _filter_forward_debug_logs_for_stdout(self, logs: dict[str, float]) -> dict[str, float]:
        full_stdout = bool(getattr(self.cfg, "debug_full_stdout_metrics", False))
        if self._is_short_goal_task and not full_stdout:
            allowlist = {
                "Metrics/short_goal/goal_distance_mean",
                "Metrics/short_goal/path_length_m",
                "Metrics/short_goal/path_efficiency",
                "Metrics/short_goal/min_goal_distance_m",
                "Metrics/short_goal/max_distance_rebound_after_min_m",
                "Metrics/short_goal/turn_reversal_count",
                "Metrics/short_goal/progress_mean",
                "Metrics/short_goal/stopped_success_rate",
                "Metrics/short_goal/near_goal_moving_rate",
                "Metrics/short_goal/stop_phase_active_rate",
                "Metrics/short_goal/stop_phase_stable_fraction",
                "Metrics/short_goal/heading_error_abs_mean",
                "Metrics/short_goal/heading_error_signed_mean",
                "Metrics/short_goal/desired_yaw_rate_mean",
                "Metrics/short_goal/desired_yaw_rate_abs_mean",
                "Metrics/short_goal/yaw_rate_tracking_error_abs_mean",
                "Metrics/short_goal/stage_far_rate",
                "Metrics/short_goal/stage_middle_rate",
                "Metrics/short_goal/stage_approach_rate",
                "Metrics/short_goal/stage_aligned_cruise_rate",
                "Metrics/short_goal/stage_far_turning_rate",
                "Metrics/short_goal/stage_far_other_rate",
                "Metrics/short_goal/stage_near_straight_rate",
                "Metrics/short_goal/stage_left_turn_rate",
                "Metrics/short_goal/stage_right_turn_rate",
                "Metrics/short_goal/far_heading_error_abs_mean",
                "Metrics/short_goal/middle_heading_error_abs_mean",
                "Metrics/short_goal/approach_heading_error_abs_mean",
                "Metrics/short_goal/far_heading_error_signed_mean",
                "Metrics/short_goal/middle_heading_error_signed_mean",
                "Metrics/short_goal/approach_heading_error_signed_mean",
                "Metrics/short_goal/left_goal_heading_error_signed_mean",
                "Metrics/short_goal/right_goal_heading_error_signed_mean",
                "Metrics/short_goal/far_velocity_toward_goal_mean",
                "Metrics/short_goal/middle_velocity_toward_goal_mean",
                "Metrics/short_goal/approach_velocity_toward_goal_mean",
                "Metrics/short_goal/far_wheel_common_mode_norm_mean",
                "Metrics/short_goal/middle_wheel_common_mode_norm_mean",
                "Metrics/short_goal/approach_wheel_common_mode_norm_mean",
                "Metrics/short_goal/far_wheel_turn_mode_norm_mean",
                "Metrics/short_goal/middle_wheel_turn_mode_norm_mean",
                "Metrics/short_goal/approach_wheel_turn_mode_norm_mean",
                "Metrics/short_goal/yaw_correct_direction_rate",
                "Metrics/short_goal/yaw_wrong_direction_rate",
                "Metrics/short_goal/left_goal_wheel_common_mode_norm_mean",
                "Metrics/short_goal/right_goal_wheel_common_mode_norm_mean",
                "Metrics/short_goal/left_goal_wheel_turn_mode_norm_mean",
                "Metrics/short_goal/right_goal_wheel_turn_mode_norm_mean",
                "Metrics/short_goal/left_goal_velocity_toward_goal_mean",
                "Metrics/short_goal/right_goal_velocity_toward_goal_mean",
                "Metrics/short_goal/left_goal_wheel_slip_mean",
                "Metrics/short_goal/right_goal_wheel_slip_mean",
                "Metrics/short_goal/far_wheel_slip_mean",
                "Metrics/short_goal/middle_wheel_slip_mean",
                "Metrics/short_goal/approach_wheel_slip_mean",
                "Metrics/short_goal/far_wheel_slip_speed_mean_mps",
                "Metrics/short_goal/middle_wheel_slip_speed_mean_mps",
                "Metrics/short_goal/approach_wheel_slip_speed_mean_mps",
                "Metrics/short_goal/far_wheel_slip_ratio_motion_gated_mean",
                "Metrics/short_goal/middle_wheel_slip_ratio_motion_gated_mean",
                "Metrics/short_goal/approach_wheel_slip_ratio_motion_gated_mean",
                "Metrics/short_goal/far_traction_overspeed_excess_mean_mps",
                "Metrics/short_goal/middle_traction_overspeed_excess_mean_mps",
                "Metrics/short_goal/approach_traction_overspeed_excess_mean_mps",
                "Metrics/short_goal/aligned_cruise_wheel_surface_speed_abs_mean_mps",
                "Metrics/short_goal/aligned_cruise_wheel_hub_forward_speed_abs_mean_mps",
                "Metrics/short_goal/aligned_cruise_traction_drive_efficiency_mean",
                "Metrics/short_goal/aligned_cruise_wheel_slip_ratio_motion_gated_mean",
                "Metrics/short_goal/aligned_cruise_wheel_slip_speed_mean_mps",
                "Metrics/short_goal/aligned_cruise_traction_overspeed_excess_mean_mps",
                "Metrics/short_goal/far_turning_wheel_surface_speed_abs_mean_mps",
                "Metrics/short_goal/far_turning_wheel_hub_forward_speed_abs_mean_mps",
                "Metrics/short_goal/far_turning_traction_drive_efficiency_mean",
                "Metrics/short_goal/far_turning_wheel_slip_ratio_motion_gated_mean",
                "Metrics/short_goal/far_turning_wheel_slip_speed_mean_mps",
                "Metrics/short_goal/far_turning_traction_overspeed_excess_mean_mps",
                "Metrics/short_goal/near_straight_traction_drive_efficiency_mean",
                "Metrics/short_goal/near_straight_wheel_slip_ratio_motion_gated_mean",
                "Metrics/short_goal/near_straight_wheel_slip_speed_mean_mps",
                "Metrics/short_goal/near_straight_traction_overspeed_excess_mean_mps",
                "Metrics/short_goal/left_turn_wheel_turn_mode_toward_goal_mean",
                "Metrics/short_goal/right_turn_wheel_turn_mode_toward_goal_mean",
                "Metrics/short_goal/left_turn_actual_yaw_rate_toward_goal_mean",
                "Metrics/short_goal/right_turn_actual_yaw_rate_toward_goal_mean",
                "Metrics/short_goal/left_turn_yaw_effectiveness_mean",
                "Metrics/short_goal/right_turn_yaw_effectiveness_mean",
                "Metrics/short_goal/left_turn_inner_left_wheel_slip_speed_mean_mps",
                "Metrics/short_goal/left_turn_outer_right_wheel_slip_speed_mean_mps",
                "Metrics/short_goal/right_turn_outer_left_wheel_slip_speed_mean_mps",
                "Metrics/short_goal/right_turn_inner_right_wheel_slip_speed_mean_mps",
                "Metrics/short_goal/left_turn_inner_left_traction_overspeed_excess_mean_mps",
                "Metrics/short_goal/left_turn_outer_right_traction_overspeed_excess_mean_mps",
                "Metrics/short_goal/right_turn_outer_left_traction_overspeed_excess_mean_mps",
                "Metrics/short_goal/right_turn_inner_right_traction_overspeed_excess_mean_mps",
                "Metrics/short_goal/left_turn_inner_left_traction_drive_efficiency_mean",
                "Metrics/short_goal/left_turn_outer_right_traction_drive_efficiency_mean",
                "Metrics/short_goal/right_turn_outer_left_traction_drive_efficiency_mean",
                "Metrics/short_goal/right_turn_inner_right_traction_drive_efficiency_mean",
                "Metrics/short_goal/wheel_left_front_rear_target_diff_norm_mean",
                "Metrics/short_goal/wheel_right_front_rear_target_diff_norm_mean",
                "Metrics/short_goal/wheel_left_front_rear_opposite_sign_rate",
                "Metrics/short_goal/wheel_right_front_rear_opposite_sign_rate",
                "Metrics/short_goal/wheel_left_front_rear_antisymmetric_mode_mean",
                "Metrics/short_goal/wheel_right_front_rear_antisymmetric_mode_mean",
                "Metrics/short_goal/far_antisymmetric_mode_mean",
                "Metrics/short_goal/middle_antisymmetric_mode_mean",
                "Metrics/short_goal/approach_antisymmetric_mode_mean",
                "Metrics/short_goal/far_opposite_sign_rate",
                "Metrics/short_goal/middle_opposite_sign_rate",
                "Metrics/short_goal/approach_opposite_sign_rate",
                "Metrics/short_goal/root_height_mean",
                "Metrics/short_goal/stroke_max_mean",
                "Metrics/short_goal/stroke_above_0_60_rate",
                "Metrics/short_goal/wheel_contact_min_ratio_mean",
                "Metrics/short_goal/non_wheel_contact_rate",
                "Metrics/short_goal/bad_orientation_rate",
            }
            return {name: value for name, value in logs.items() if name in allowlist}
        if self._is_stand_training_task and not self._stand_debug_metrics_enabled:
            return {}
        if not self._is_stand_training_task or full_stdout:
            return logs

        allowlist = {
            "Metrics/stand_debug/stand/root_height",
            "Metrics/stand_debug/stand/root_height_reward_used",
            "Metrics/stand_debug/stand/root_height_error",
            "Metrics/stand_debug/stand/root_height_low_rate",
            "Metrics/stand_debug/stand/root_height_low_raw_rate",
            "Metrics/stand_debug/stand/root_height_high_rate",
            "Metrics/stand_debug/stand/root_height_low_grace_active_rate",
            "Metrics/stand_debug/stand/termination_root_height_low_after_grace_rate",
            "Metrics/stand_debug/stand/root_height_last50_mean",
            "Metrics/stand_debug/stand/root_lin_vel_z_last50_mean",
            "Metrics/stand_debug/stand/wheel_contact_over_weight_last50_mean",
            "Metrics/stand_debug/stand/non_wheel_contact_over_weight_last50_mean",
            "Metrics/stand_debug/stand/episode_step_mean",
            "Metrics/stand_debug/stand/episode_step_env0",
            "Metrics/stand_debug/stand/hydraulic_stroke_lf",
            "Metrics/stand_debug/stand/hydraulic_stroke_lr",
            "Metrics/stand_debug/stand/hydraulic_stroke_rf",
            "Metrics/stand_debug/stand/hydraulic_stroke_rr",
            "Metrics/stand_debug/stand/hydraulic_action_lf",
            "Metrics/stand_debug/stand/hydraulic_action_lr",
            "Metrics/stand_debug/stand/hydraulic_action_rf",
            "Metrics/stand_debug/stand/hydraulic_action_rr",
            "Metrics/stand_debug/stand/raw_hydraulic_action_lf",
            "Metrics/stand_debug/stand/raw_hydraulic_action_lr",
            "Metrics/stand_debug/stand/raw_hydraulic_action_rf",
            "Metrics/stand_debug/stand/raw_hydraulic_action_rr",
            "Metrics/stand_debug/stand/clipped_hydraulic_action_lf",
            "Metrics/stand_debug/stand/clipped_hydraulic_action_lr",
            "Metrics/stand_debug/stand/clipped_hydraulic_action_rf",
            "Metrics/stand_debug/stand/clipped_hydraulic_action_rr",
            "Metrics/stand_debug/stand/target_stroke_lf",
            "Metrics/stand_debug/stand/target_stroke_lr",
            "Metrics/stand_debug/stand/target_stroke_rf",
            "Metrics/stand_debug/stand/target_stroke_rr",
            "Metrics/stand_debug/stand/stroke_range_value",
            "Metrics/stand_debug/stand/stroke_diagonal_balance_value",
            "Metrics/stand_debug/stand/stroke_left_right_balance_value",
            "Metrics/stand_debug/stand/stroke_front_rear_balance_value",
            "Metrics/stand_debug/stand/stroke_variance_from_nominal_value",
            "Metrics/stand_debug/stand/target_stroke_range_value",
            "Metrics/stand_debug/stand/target_stroke_diagonal_balance_value",
            "Metrics/stand_debug/stand/action_diagonal_split_value",
            "Metrics/stand_debug/stand/high_height_low_stroke_negative_action_value",
            "Metrics/stand_debug/stand/max_abs_raw_hydraulic_action",
            "Metrics/stand_debug/stand/max_abs_clipped_hydraulic_action",
            "Metrics/stand_debug/stand/max_abs_joint_pos_minus_target",
            "Metrics/stand_debug/stand/contact_force_total",
            "Metrics/stand_debug/stand/contact_force_min_ratio",
            "Metrics/stand_debug/stand/mean_abs_wheel_target",
            "Metrics/stand_debug/stand/robot_total_mass",
            "Metrics/stand_debug/stand/expected_weight_force",
            "Metrics/stand_debug/stand/contact_force_lf",
            "Metrics/stand_debug/stand/contact_force_lr",
            "Metrics/stand_debug/stand/contact_force_rf",
            "Metrics/stand_debug/stand/contact_force_rr",
            "Metrics/stand_debug/stand/wheel_contact_force_total_raw",
            "Metrics/stand_debug/stand/wheel_contact_force_total_raw_over_weight",
            "Metrics/stand_debug/stand/all_body_contact_force_total_raw",
            "Metrics/stand_debug/stand/all_body_contact_force_total_raw_over_weight",
            "Metrics/stand_debug/stand/non_wheel_contact_force_total_raw",
            "Metrics/stand_debug/stand/non_wheel_contact_force_total_raw_over_weight",
            "Metrics/stand_debug/stand/contact_bool_lf",
            "Metrics/stand_debug/stand/contact_bool_lr",
            "Metrics/stand_debug/stand/contact_bool_rf",
            "Metrics/stand_debug/stand/contact_bool_rr",
            "Metrics/stand_debug/stand/min_contact_force",
            "Metrics/stand_debug/stand/env0_root_height",
            "Metrics/stand_debug/stand/env0_root_height_high",
            "Metrics/stand_debug/stand/env0_root_height_low_grace_active",
            "Metrics/stand_debug/stand/env0_hydraulic_stroke_lf",
            "Metrics/stand_debug/stand/env0_hydraulic_stroke_lr",
            "Metrics/stand_debug/stand/env0_hydraulic_stroke_rf",
            "Metrics/stand_debug/stand/env0_hydraulic_stroke_rr",
            "Metrics/stand_debug/stand/env0_stroke_range_value",
            "Metrics/stand_debug/stand/env0_stroke_diagonal_balance_value",
            "Metrics/stand_debug/stand/env0_wheel_contact_force_lf",
            "Metrics/stand_debug/stand/env0_wheel_contact_force_lr",
            "Metrics/stand_debug/stand/env0_wheel_contact_force_rf",
            "Metrics/stand_debug/stand/env0_wheel_contact_force_rr",
            "Metrics/stand_debug/stand/env0_wheel_contact_force_total_raw",
            "Metrics/stand_debug/stand/env0_all_body_contact_force_total_raw",
            "Metrics/stand_debug/stand/env0_non_wheel_contact_force_total_raw",
            "Metrics/stand_debug/stand/env0_contact_bool_lf",
            "Metrics/stand_debug/stand/env0_contact_bool_lr",
            "Metrics/stand_debug/stand/env0_contact_bool_rf",
            "Metrics/stand_debug/stand/env0_contact_bool_rr",
            "Metrics/stand_debug/stand/reset_happened_rate",
            "Metrics/stand_debug/stand/env0_reset_happened",
        }
        return {name: value for name, value in logs.items() if name in allowlist}

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

    def _apply_reset_settle_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self._enable_reset_settle or self._reset_settle_steps <= 0:
            self._reset_settle_step_active.zero_()
            return action

        settling = self._reset_settle_remaining_steps > 0
        self._reset_settle_step_active[:] = settling
        if not torch.any(settling):
            return action

        action = action.clone()
        action[settling, 0:4] = self._reset_settle_leg_action
        action[settling, 4:8] = 0.0
        return action

    def _apply_stand_action_constraints(self, action: torch.Tensor) -> torch.Tensor:
        if not self._is_stand_training_task:
            return action
        action = action.clone()
        action[:, 4:8] = 0.0
        return action

    def _apply_stand_trace_action_override(self, action: torch.Tensor) -> torch.Tensor:
        if not self._stand_trace_enabled or self._stand_trace_mode != "zero_action":
            return action
        action = action.clone()
        zero_mask = self.episode_length_buf < self._stand_trace_zero_action_horizon
        if torch.any(zero_mask):
            action[zero_mask, 0:4] = 0.0
            action[zero_mask, 4:8] = 0.0
        return action

    def _append_stand_trace_row(self) -> None:
        if not self._stand_trace_enabled or self._stand_trace_csv_writer is None or self._stand_trace_csv_file is None:
            return
        if self._stand_trace_env_ids.numel() == 0:
            return

        env_id = int(self._stand_trace_env_ids[0].item())
        if self._stand_trace_mode == "zero_action":
            episode_step = int(self.episode_length_buf[env_id].item())
            if episode_step < 1 or episode_step > self._stand_trace_zero_action_horizon:
                return
        robot = self.scene["robot"]
        leg_action_term = self.action_manager.get_term("leg_hydraulic")
        wheel_contact_sensor = self.scene.sensors["wheel_contact_forces"]
        all_body_contact_sensor = self.scene.sensors["all_body_contact_forces"]

        net_contact_forces = wheel_contact_sensor.data.net_forces_w_history[:, :, self._wheel_contact_body_ids, :]
        contact_force = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
        contacts = contact_force > 1.0

        all_body_net_contact_forces = all_body_contact_sensor.data.net_forces_w_history[
            :, :, self._all_body_contact_body_ids, :
        ]
        all_body_contact_force = torch.max(torch.norm(all_body_net_contact_forces, dim=-1), dim=1)[0]
        all_body_contact_force_total_raw = all_body_contact_force.sum(dim=1)
        if self._non_wheel_contact_body_ids.numel() > 0:
            non_wheel_net_contact_forces = all_body_contact_sensor.data.net_forces_w_history[
                :, :, self._non_wheel_contact_body_ids, :
            ]
            non_wheel_contact_force = torch.max(torch.norm(non_wheel_net_contact_forces, dim=-1), dim=1)[0]
            non_wheel_contact_force_total_raw = non_wheel_contact_force.sum(dim=1)
        else:
            non_wheel_contact_force_total_raw = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        _, total_force_raw, _, _ = _wheel_contact_force_ratio_lf_lr_rf_rr(
            self,
            SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
        )

        wheel_body_pos_w = robot.data.body_pos_w[:, self._wheel_robot_body_ids_lf_lr_rf_rr, :]
        base_link_world_z = robot.data.body_pos_w[:, self._base_link_body_ids[0], 2]
        wheel_relative_pos = wheel_body_pos_w - robot.data.root_pos_w.unsqueeze(1)
        root_height_low_limit = float(self.cfg.terminations.root_height_low.params["minimum_height"])
        root_height_low_grace_time_s = float(self.cfg.terminations.root_height_low.params.get("grace_time_s", 0.0))
        root_height_low_grace_steps = max(
            int(math.ceil(root_height_low_grace_time_s / max(float(self.step_dt), 1.0e-6))),
            0,
        )
        grace_active = self.episode_length_buf < root_height_low_grace_steps
        root_height = robot.data.root_pos_w[:, 2]
        root_height_low_raw = root_height < root_height_low_limit
        root_height_low_after_grace = torch.logical_and(root_height_low_raw, ~grace_active)

        zeros = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        termination_term_values: dict[str, torch.Tensor] = {}
        for term_name in self._turn_sanity_term_names:
            termination_term_values[term_name] = self.termination_manager.get_term(term_name).to(torch.bool)
        actual_root_height_low_term = termination_term_values.get("root_height_low", zeros)
        termination_reason_names = [
            term_name for term_name, term_value in termination_term_values.items() if bool(term_value[env_id].item())
        ]
        termination_reason = "|".join(termination_reason_names)

        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        stroke_actual_lf_lr_rf_rr = leg_action_term.stroke_actual[:, [1, 0, 2, 3]]
        target_stroke_lf_lr_rf_rr = leg_action_term.stroke_desired[:, [1, 0, 2, 3]]
        raw_action_lf_lr_rf_rr = leg_action_term.raw_actions[:, [1, 0, 2, 3]]
        clipped_action_lf_lr_rf_rr = leg_action_term.clipped_actions[:, [1, 0, 2, 3]]
        joint_pos_g_lb_lf_rf_rb = robot.data.joint_pos[:, self._leg_joint_ids]
        joint_target_g_lb_lf_rf_rb = robot.data.joint_pos_target[:, self._leg_joint_ids]
        joint_pos_minus_target_g_lb_lf_rf_rb = joint_pos_g_lb_lf_rf_rb - joint_target_g_lb_lf_rf_rb
        max_abs_joint_pos_minus_target = torch.max(torch.abs(joint_pos_minus_target_g_lb_lf_rf_rb), dim=1).values
        expected_weight_force = torch.clamp(
            torch.full((self.num_envs,), self._stand_debug_expected_weight_force, device=self.device, dtype=torch.float32),
            min=1.0e-6,
        )
        wheel_contact_over_weight = total_force_raw / expected_weight_force
        non_wheel_contact_over_weight = non_wheel_contact_force_total_raw / expected_weight_force
        within_first50 = self.episode_length_buf < 50
        self._stand_first50_max_contact_over_weight = torch.where(
            within_first50,
            torch.maximum(self._stand_first50_max_contact_over_weight, wheel_contact_over_weight),
            self._stand_first50_max_contact_over_weight,
        )
        self._stand_first50_max_non_wheel_contact_over_weight = torch.where(
            within_first50,
            torch.maximum(self._stand_first50_max_non_wheel_contact_over_weight, non_wheel_contact_over_weight),
            self._stand_first50_max_non_wheel_contact_over_weight,
        )
        self._stand_first50_any_airborne = torch.where(
            within_first50,
            torch.logical_or(self._stand_first50_any_airborne, torch.logical_not(torch.all(contacts, dim=1))),
            self._stand_first50_any_airborne,
        )
        pattern_code = int(self._stand_reset_pattern_code[env_id].item())
        pattern_name = STAND_RESET_PATTERN_NAMES[pattern_code] if 0 <= pattern_code < len(STAND_RESET_PATTERN_NAMES) else "unknown"

        row = {
            "step": int(self.common_step_counter),
            "episode_step": int(self.episode_length_buf[env_id].item()),
            "reset_happened": int(self._stand_recent_reset_happened[env_id].item()),
            "termination_any": int(self.reset_buf[env_id].item()),
            "termination_reason": termination_reason,
            "root_height": float(root_height[env_id].item()),
            "root_height_reward_used": float(root_height[env_id].item()),
            "root_world_z": float(robot.data.root_pos_w[env_id, 2].item()),
            "root_world_z_minus_env_origin_z": float((robot.data.root_pos_w[env_id, 2] - self.scene.env_origins[env_id, 2]).item()),
            "root_world_z_minus_terrain_height": float((robot.data.root_pos_w[env_id, 2] - self._stand_trace_ground_height).item()),
            "base_world_pos_x": float(robot.data.root_pos_w[env_id, 0].item()),
            "base_world_pos_y": float(robot.data.root_pos_w[env_id, 1].item()),
            "base_world_pos_z": float(robot.data.root_pos_w[env_id, 2].item()),
            "base_link_world_z": float(base_link_world_z[env_id].item()),
            "root_lin_vel_z": float(robot.data.root_lin_vel_b[env_id, 2].item()),
            "roll": float(roll[env_id].item()),
            "pitch": float(pitch[env_id].item()),
            "wheel_world_z_lf": float(wheel_body_pos_w[env_id, 0, 2].item()),
            "wheel_world_z_lr": float(wheel_body_pos_w[env_id, 1, 2].item()),
            "wheel_world_z_rf": float(wheel_body_pos_w[env_id, 2, 2].item()),
            "wheel_world_z_rr": float(wheel_body_pos_w[env_id, 3, 2].item()),
            "wheel_relative_z_lf": float(wheel_relative_pos[env_id, 0, 2].item()),
            "wheel_relative_z_lr": float(wheel_relative_pos[env_id, 1, 2].item()),
            "wheel_relative_z_rf": float(wheel_relative_pos[env_id, 2, 2].item()),
            "wheel_relative_z_rr": float(wheel_relative_pos[env_id, 3, 2].item()),
            "env_origin_x": float(self.scene.env_origins[env_id, 0].item()),
            "env_origin_y": float(self.scene.env_origins[env_id, 1].item()),
            "env_origin_z": float(self.scene.env_origins[env_id, 2].item()),
            "terrain_height_under_robot": float(self._stand_trace_ground_height),
            "robot_body_names": self._robot_body_names_joined,
            "wheel_contact_sensor_body_names": self._wheel_contact_sensor_body_names_joined,
            "all_body_contact_sensor_body_names": self._all_body_contact_sensor_body_names_joined,
            "reset_pattern_type": pattern_name,
            "stroke_init_lf": float(self._stand_reset_stroke_init_lf_lr_rf_rr[env_id, 0].item()),
            "stroke_init_lr": float(self._stand_reset_stroke_init_lf_lr_rf_rr[env_id, 1].item()),
            "stroke_init_rf": float(self._stand_reset_stroke_init_lf_lr_rf_rr[env_id, 2].item()),
            "stroke_init_rr": float(self._stand_reset_stroke_init_lf_lr_rf_rr[env_id, 3].item()),
            "root_z_init": float(self._stand_reset_root_z_init[env_id].item()),
            "roll_init": float(self._stand_reset_roll_init[env_id].item()),
            "pitch_init": float(self._stand_reset_pitch_init[env_id].item()),
            "yaw_init": float(self._stand_reset_yaw_init[env_id].item()),
            "stroke_lf": float(stroke_actual_lf_lr_rf_rr[env_id, 0].item()),
            "stroke_lr": float(stroke_actual_lf_lr_rf_rr[env_id, 1].item()),
            "stroke_rf": float(stroke_actual_lf_lr_rf_rr[env_id, 2].item()),
            "stroke_rr": float(stroke_actual_lf_lr_rf_rr[env_id, 3].item()),
            "target_stroke_lf": float(target_stroke_lf_lr_rf_rr[env_id, 0].item()),
            "target_stroke_lr": float(target_stroke_lf_lr_rf_rr[env_id, 1].item()),
            "target_stroke_rf": float(target_stroke_lf_lr_rf_rr[env_id, 2].item()),
            "target_stroke_rr": float(target_stroke_lf_lr_rf_rr[env_id, 3].item()),
            "raw_action_lf": float(raw_action_lf_lr_rf_rr[env_id, 0].item()),
            "raw_action_lr": float(raw_action_lf_lr_rf_rr[env_id, 1].item()),
            "raw_action_rf": float(raw_action_lf_lr_rf_rr[env_id, 2].item()),
            "raw_action_rr": float(raw_action_lf_lr_rf_rr[env_id, 3].item()),
            "clipped_action_lf": float(clipped_action_lf_lr_rf_rr[env_id, 0].item()),
            "clipped_action_lr": float(clipped_action_lf_lr_rf_rr[env_id, 1].item()),
            "clipped_action_rf": float(clipped_action_lf_lr_rf_rr[env_id, 2].item()),
            "clipped_action_rr": float(clipped_action_lf_lr_rf_rr[env_id, 3].item()),
            "max_abs_joint_pos_minus_target": float(max_abs_joint_pos_minus_target[env_id].item()),
            "wheel_contact_force_over_weight": float(wheel_contact_over_weight[env_id].item()),
            "non_wheel_contact_force_over_weight": float(non_wheel_contact_over_weight[env_id].item()),
            "contact_bool_lf": int(contacts[env_id, 1].item()),
            "contact_bool_lr": int(contacts[env_id, 0].item()),
            "contact_bool_rf": int(contacts[env_id, 2].item()),
            "contact_bool_rr": int(contacts[env_id, 3].item()),
            "first50_max_contact_force_over_weight": float(self._stand_first50_max_contact_over_weight[env_id].item()),
            "first50_max_non_wheel_contact_force_over_weight": float(
                self._stand_first50_max_non_wheel_contact_over_weight[env_id].item()
            ),
            "first50_any_airborne": int(self._stand_first50_any_airborne[env_id].item()),
            "root_height_low_raw": int(root_height_low_raw[env_id].item()),
            "root_height_low_after_grace": int(root_height_low_after_grace[env_id].item()),
            "actual_root_height_low_term": int(actual_root_height_low_term[env_id].item()),
            "grace_active": int(grace_active[env_id].item()),
            "root_height_last50_mean": float(self._stand_last50_root_height[env_id, : max(int(self._stand_last50_count[env_id].item()), 1)].mean().item()),
            "root_lin_vel_z_last50_mean": float(self._stand_last50_root_lin_vel_z[env_id, : max(int(self._stand_last50_count[env_id].item()), 1)].mean().item()),
            "wheel_contact_over_weight_last50_mean": float(
                self._stand_last50_wheel_contact_over_weight[env_id, : max(int(self._stand_last50_count[env_id].item()), 1)].mean().item()
            ),
            "non_wheel_contact_over_weight_last50_mean": float(
                self._stand_last50_non_wheel_contact_over_weight[env_id, : max(int(self._stand_last50_count[env_id].item()), 1)].mean().item()
            ),
        }
        self._stand_trace_csv_writer.writerow(row)
        self._stand_trace_csv_file.flush()

    def _initialize_short_goal_visualizer(self) -> None:
        marker_cfg = POSITION_GOAL_MARKER_CFG.copy()
        marker_cfg.prim_path = "/World/Visuals/RangerShortGoalTargets"
        self._short_goal_goal_marker = VisualizationMarkers(marker_cfg)

    def _update_short_goal_stop_phase(self) -> None:
        """Latch the stop phase and count consecutive settled control steps."""

        if not self._is_short_goal_task:
            return
        robot = self.scene["robot"]
        _, goal_distance, _ = mdp.short_goal_target_body(self)
        was_active = self._short_goal_stop_phase_active.clone()
        self._short_goal_stop_phase_active |= goal_distance < self._short_goal_stop_phase_enter_distance
        newly_active = self._short_goal_stop_phase_active & (~was_active)
        if torch.any(newly_active):
            leg_action_term = self.action_manager.get_term("leg_hydraulic")
            self._short_goal_stop_latched_hydraulic_action_raw[newly_active] = leg_action_term.raw_actions[
                newly_active
            ]
        base_xy_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1)
        yaw_rate_abs = torch.abs(robot.data.root_ang_vel_b[:, 2])
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        leg_action_term = self.action_manager.get_term("leg_hydraulic")
        stroke_tracking_error_max = torch.max(
            torch.abs(leg_action_term.stroke_desired - leg_action_term.stroke_actual), dim=1
        ).values
        posture_ready = (
            (torch.abs(roll) <= self._short_goal_stop_max_roll)
            & (torch.abs(pitch) <= self._short_goal_stop_max_pitch)
            & (stroke_tracking_error_max <= self._short_goal_stop_max_stroke_tracking_error)
        )
        phase_count_ready = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        if self._short_goal_stop_count_only_after_phase:
            phase_count_ready = self._short_goal_stop_phase_active & (~newly_active)
        settled_now = (
            phase_count_ready
            & (goal_distance < self._short_goal_stop_success_distance)
            & (base_xy_speed < self._short_goal_stop_max_xy_speed)
            & (yaw_rate_abs < self._short_goal_stop_max_yaw_rate)
            & posture_ready
        )
        self._short_goal_stop_phase_stable_steps[:] = torch.where(
            settled_now,
            self._short_goal_stop_phase_stable_steps + 1,
            torch.zeros_like(self._short_goal_stop_phase_stable_steps),
        )

    def _apply_short_goal_stop_action_override(self, action: torch.Tensor) -> torch.Tensor:
        """Apply configured stop-phase suspension and optional wheel overrides."""

        if not self._is_short_goal_task or action.shape[1] < 8:
            return action
        stop_mask = self._short_goal_stop_phase_active
        if not torch.any(stop_mask):
            return action
        action = action.clone()
        if self._stop_phase_suspension_mode == "legacy":
            if self._stop_phase_suspension_action_enabled:
                action[stop_mask, 0:4] = self._stop_phase_suspension_action
            else:
                action[stop_mask, 0:4] = self._short_goal_stop_latched_hydraulic_action_raw[stop_mask]
        if self._stop_phase_wheel_override_enabled:
            action[stop_mask, 4:8] = 0.0
        return action

    def _apply_suspension_action_scale(self, action: torch.Tensor) -> torch.Tensor:
        """Scale active hydraulic policy commands while preserving the 8-D action interface."""

        if self._suspension_action_scale == 1.0 or action.shape[1] < 4:
            return action
        action = action.clone()
        action[:, 0:4] *= self._suspension_action_scale
        return action

    def _apply_fixed_suspension_action_override(self, action: torch.Tensor) -> torch.Tensor:
        """Override active suspension control with one normalized command."""

        if not self._fixed_suspension_action_enabled or action.shape[1] < 4:
            return action
        action = action.clone()
        action[:, 0:4] = self._fixed_suspension_action
        return action

    def _update_short_goal_visualizer(self) -> None:
        if not self._short_goal_visualization_enabled or self._short_goal_goal_marker is None:
            return
        goal_pos_w = mdp.short_goal_target_pos_w(self).clone()
        goal_pos_w[:, 2] = self._stand_trace_ground_height + 0.10
        marker_indices = torch.where(
            self._short_goal_stop_phase_active,
            torch.ones((self.num_envs,), dtype=torch.int32, device=self.device),
            torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device),
        )
        scales = torch.full((self.num_envs, 3), 6.0, dtype=torch.float32, device=self.device)
        self._short_goal_goal_marker.visualize(
            translations=goal_pos_w,
            marker_indices=marker_indices,
            scales=scales,
        )

    def step(self, action: torch.Tensor):
        # process actions
        action = action.to(self.device)
        if action.shape[1] >= 4:
            self._policy_hydraulic_action_raw[:] = action[:, 0:4]
        if action.shape[1] >= 8:
            self._short_goal_policy_wheel_action_raw[:] = action[:, 4:8]
        action = self._apply_reset_settle_action(action)
        action = self._apply_stand_action_constraints(action)
        action = self._apply_stand_trace_action_override(action)
        action = self._apply_suspension_action_scale(action)
        action = self._apply_short_goal_stop_action_override(action)
        action = self._apply_fixed_suspension_action_override(action)
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

        settling_mask = self._reset_settle_step_active.clone()
        if torch.any(settling_mask):
            self._reset_settle_remaining_steps[settling_mask] -= 1
        self.episode_length_buf += (~settling_mask).to(self.episode_length_buf.dtype)
        self.common_step_counter += 1
        self._update_short_goal_trajectory_quality(active_mask=~settling_mask)
        self._update_short_goal_stop_phase()
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if torch.any(settling_mask):
            self.reward_buf[settling_mask] = 0.0

        # Accumulate locomotion diagnosis metrics before any terminated env is reset.
        self._accumulate_forward_debug_metrics()
        self._append_stand_trace_row()

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
        self._update_short_goal_visualizer()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _sample_stand_reset_pattern_lf_lr_rf_rr(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_envs = int(env_ids.numel())
        dtype = torch.float32
        device = self.device
        base_stroke = float(getattr(self.cfg, "terrain_like_reset_base_stroke", 0.50))
        common_offset_range = getattr(self.cfg, "terrain_like_reset_common_offset_range", (-0.02, 0.04))
        amp_range = getattr(self.cfg, "terrain_like_reset_pattern_amplitude_range", (0.03, 0.06))
        stroke_clamp_range = getattr(self.cfg, "terrain_like_reset_stroke_clamp_range", (0.44, 0.58))
        slope_deg_range = getattr(self.cfg, "terrain_like_reset_slope_deg_range", (3.0, 5.0))
        twist_deg_range = getattr(self.cfg, "terrain_like_reset_twist_deg_range", (2.0, 5.0))
        yaw_deg_range = getattr(self.cfg, "terrain_like_reset_yaw_deg_range", (-3.0, 3.0))

        common_offset = torch.empty((num_envs, 1), device=device, dtype=dtype).uniform_(
            float(common_offset_range[0]), float(common_offset_range[1])
        )
        amplitude = torch.empty((num_envs, 1), device=device, dtype=dtype).uniform_(
            float(amp_range[0]), float(amp_range[1])
        )
        pattern_codes = torch.randint(len(STAND_RESET_PATTERN_NAMES), (num_envs,), device=device, dtype=torch.long)
        pattern_sign = torch.where(
            torch.rand((num_envs, 1), device=device, dtype=dtype) < 0.5,
            -torch.ones((num_envs, 1), device=device, dtype=dtype),
            torch.ones((num_envs, 1), device=device, dtype=dtype),
        )

        stroke_lf_lr_rf_rr = torch.full((num_envs, 4), base_stroke, device=device, dtype=dtype) + common_offset
        roll = torch.zeros((num_envs,), device=device, dtype=dtype)
        pitch = torch.zeros((num_envs,), device=device, dtype=dtype)
        yaw = torch.empty((num_envs,), device=device, dtype=dtype).uniform_(
            math.radians(float(yaw_deg_range[0])), math.radians(float(yaw_deg_range[1]))
        )

        slope_mag = torch.empty((num_envs,), device=device, dtype=dtype).uniform_(
            math.radians(float(slope_deg_range[0])), math.radians(float(slope_deg_range[1]))
        )
        twist_mag = torch.empty((num_envs,), device=device, dtype=dtype).uniform_(
            math.radians(float(twist_deg_range[0])), math.radians(float(twist_deg_range[1]))
        )

        left_right_mask = pattern_codes == 0
        if torch.any(left_right_mask):
            delta = amplitude[left_right_mask] * pattern_sign[left_right_mask]
            stroke_lf_lr_rf_rr[left_right_mask] += torch.cat((delta, delta, -delta, -delta), dim=1)
            roll[left_right_mask] = slope_mag[left_right_mask] * pattern_sign[left_right_mask, 0]

        front_back_mask = pattern_codes == 1
        if torch.any(front_back_mask):
            delta = amplitude[front_back_mask] * pattern_sign[front_back_mask]
            stroke_lf_lr_rf_rr[front_back_mask] += torch.cat((delta, -delta, delta, -delta), dim=1)
            pitch[front_back_mask] = twist_mag[front_back_mask] * pattern_sign[front_back_mask, 0]

        diagonal_mask = pattern_codes == 2
        if torch.any(diagonal_mask):
            delta = amplitude[diagonal_mask] * pattern_sign[diagonal_mask]
            stroke_lf_lr_rf_rr[diagonal_mask] += torch.cat((delta, -delta, -delta, delta), dim=1)
            roll[diagonal_mask] = twist_mag[diagonal_mask] * pattern_sign[diagonal_mask, 0]
            pitch[diagonal_mask] = -twist_mag[diagonal_mask] * pattern_sign[diagonal_mask, 0]

        single_bump_mask = pattern_codes == 3
        if torch.any(single_bump_mask):
            wheel_idx = torch.randint(4, (int(single_bump_mask.sum().item()),), device=device, dtype=torch.long)
            delta = amplitude[single_bump_mask, 0]
            stroke_lf_lr_rf_rr[single_bump_mask, wheel_idx] += delta
            roll_sign = torch.tensor([1.0, 1.0, -1.0, -1.0], device=device, dtype=dtype)[wheel_idx]
            pitch_sign = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device, dtype=dtype)[wheel_idx]
            roll[single_bump_mask] = twist_mag[single_bump_mask] * roll_sign
            pitch[single_bump_mask] = twist_mag[single_bump_mask] * pitch_sign

        single_dip_mask = pattern_codes == 4
        if torch.any(single_dip_mask):
            wheel_idx = torch.randint(4, (int(single_dip_mask.sum().item()),), device=device, dtype=torch.long)
            delta = amplitude[single_dip_mask, 0]
            stroke_lf_lr_rf_rr[single_dip_mask, wheel_idx] -= delta
            roll_sign = torch.tensor([1.0, 1.0, -1.0, -1.0], device=device, dtype=dtype)[wheel_idx]
            pitch_sign = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device, dtype=dtype)[wheel_idx]
            roll[single_dip_mask] = -twist_mag[single_dip_mask] * roll_sign
            pitch[single_dip_mask] = -twist_mag[single_dip_mask] * pitch_sign

        mixed_mask = pattern_codes == 5
        if torch.any(mixed_mask):
            mixed_delta = torch.empty((int(mixed_mask.sum().item()), 4), device=device, dtype=dtype).uniform_(-1.0, 1.0)
            mixed_delta = mixed_delta - mixed_delta.mean(dim=1, keepdim=True)
            mixed_delta = amplitude[mixed_mask] * mixed_delta / mixed_delta.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6)
            stroke_lf_lr_rf_rr[mixed_mask] += mixed_delta
            roll[mixed_mask] = torch.empty((int(mixed_mask.sum().item()),), device=device, dtype=dtype).uniform_(
                -math.radians(4.0), math.radians(4.0)
            )
            pitch[mixed_mask] = torch.empty((int(mixed_mask.sum().item()),), device=device, dtype=dtype).uniform_(
                -math.radians(4.0), math.radians(4.0)
            )

        stroke_lf_lr_rf_rr = torch.clamp(
            stroke_lf_lr_rf_rr,
            min=float(stroke_clamp_range[0]),
            max=float(stroke_clamp_range[1]),
        )
        return pattern_codes, stroke_lf_lr_rf_rr, roll, pitch, yaw

    def _apply_stand_terrain_like_reset(self, env_ids: torch.Tensor) -> None:
        leg_action_term = self.action_manager.get_term("leg_hydraulic")
        robot = self.scene["robot"]
        pattern_codes, stroke_lf_lr_rf_rr, roll, pitch, yaw = self._sample_stand_reset_pattern_lf_lr_rf_rr(env_ids)
        stroke_lr_lf_rf_rr = stroke_lf_lr_rf_rr[:, [1, 0, 2, 3]]
        position_target = leg_action_term._interp_stroke_to_joint_pos(stroke_lr_lf_rf_rr) * leg_action_term._joint_target_sign

        leg_action_term._stroke_actual[env_ids] = stroke_lr_lf_rf_rr
        leg_action_term._position_target[env_ids] = position_target
        leg_action_term._raw_actions[env_ids] = 0.0
        leg_action_term._processed_actions[env_ids] = 0.0
        leg_action_term._effort_actual[env_ids] = 0.0

        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
        joint_pos[:, self._leg_joint_ids] = position_target
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        mean_stroke = stroke_lf_lr_rf_rr.mean(dim=1)
        root_height_reference = float(getattr(self.cfg, "initial_stroke_root_height_reference", 0.884))
        nominal_stroke = float(getattr(self.cfg, "initial_stroke_root_height_nominal_stroke", 0.50))
        slope = float(getattr(self.cfg, "initial_stroke_root_height_slope", 0.38))
        safety_margin = float(getattr(self.cfg, "terrain_like_reset_root_height_margin", 0.015))
        target_root_height = root_height_reference - slope * (mean_stroke - nominal_stroke) + safety_margin

        root_pose = robot.data.root_pose_w[env_ids].clone()
        root_velocity = torch.zeros_like(robot.data.root_vel_w[env_ids])
        root_pose[:, 2] = target_root_height
        root_pose[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        lin_vel_xy_range = getattr(self.cfg, "terrain_like_reset_linear_xy_velocity_range", (-0.10, 0.10))
        ang_vel_range = getattr(self.cfg, "terrain_like_reset_angular_velocity_range", (-0.2, 0.2))
        root_velocity[:, 0] = torch.empty((env_ids.numel(),), device=self.device, dtype=torch.float32).uniform_(
            float(lin_vel_xy_range[0]), float(lin_vel_xy_range[1])
        )
        root_velocity[:, 1] = torch.empty((env_ids.numel(),), device=self.device, dtype=torch.float32).uniform_(
            float(lin_vel_xy_range[0]), float(lin_vel_xy_range[1])
        )
        root_velocity[:, 2] = 0.0
        root_velocity[:, 3] = torch.empty((env_ids.numel(),), device=self.device, dtype=torch.float32).uniform_(
            float(ang_vel_range[0]), float(ang_vel_range[1])
        )
        root_velocity[:, 4] = torch.empty((env_ids.numel(),), device=self.device, dtype=torch.float32).uniform_(
            float(ang_vel_range[0]), float(ang_vel_range[1])
        )
        root_velocity[:, 5] = torch.empty((env_ids.numel(),), device=self.device, dtype=torch.float32).uniform_(
            float(ang_vel_range[0]), float(ang_vel_range[1])
        )
        robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

        self._stand_reset_pattern_code[env_ids] = pattern_codes
        self._stand_reset_stroke_init_lf_lr_rf_rr[env_ids] = stroke_lf_lr_rf_rr
        self._stand_reset_root_z_init[env_ids] = target_root_height
        self._stand_reset_roll_init[env_ids] = roll
        self._stand_reset_pitch_init[env_ids] = pitch
        self._stand_reset_yaw_init[env_ids] = yaw
        self._stand_first50_max_contact_over_weight[env_ids] = 0.0
        self._stand_first50_max_non_wheel_contact_over_weight[env_ids] = 0.0
        self._stand_first50_any_airborne[env_ids] = False

    def _reset_idx(self, env_ids):
        debug_logs = self._consume_forward_debug_logs(env_ids)
        super()._reset_idx(env_ids)
        if self._enable_reset_settle and self._reset_settle_steps > 0:
            self._reset_settle_remaining_steps[env_ids] = self._reset_settle_steps
            self._reset_settle_step_active[env_ids] = False
        self._policy_hydraulic_action_raw[env_ids] = 0.0
        self._short_goal_policy_wheel_action_raw[env_ids] = 0.0
        self._short_goal_stop_latched_hydraulic_action_raw[env_ids] = 0.0
        self._short_goal_hydraulic_action_prev[env_ids] = 0.0
        self._short_goal_stop_phase_active[env_ids] = False
        self._short_goal_stop_phase_stable_steps[env_ids] = 0
        if self._is_short_goal_task:
            target_vec_b, goal_distance, _ = mdp.short_goal_target_body(self)
            robot = self.scene["robot"]
            self._short_goal_initial_side_sign[env_ids] = torch.sign(target_vec_b[env_ids, 1])
            self._short_goal_path_length_m[env_ids] = 0.0
            self._short_goal_prev_root_xy[env_ids] = robot.data.root_pos_w[env_ids, :2]
            self._short_goal_initial_goal_distance_m[env_ids] = goal_distance[env_ids]
            self._short_goal_min_goal_distance_m[env_ids] = goal_distance[env_ids]
            self._short_goal_max_distance_rebound_after_min_m[env_ids] = 0.0
            self._short_goal_last_turn_sign[env_ids] = 0.0
            self._short_goal_turn_reversal_count[env_ids] = 0.0
        self._yaw_turn_support_executed_raw_action[env_ids] = 0.0
        self._executed_hydraulic_action_prev[env_ids] = 0.0
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
        if bool(getattr(self.cfg, "enable_initial_stroke_randomization", False)):
            if bool(getattr(self.cfg, "terrain_like_reset_enabled", False)) and self._is_stand_training_task:
                self._apply_stand_terrain_like_reset(env_ids)
            else:
                leg_action_term = self.action_manager.get_term("leg_hydraulic")
                robot = self.scene["robot"]
                stroke_range = getattr(self.cfg, "initial_stroke_range", None)
                shared_across_legs = bool(getattr(self.cfg, "initial_stroke_shared_across_legs", True))
                randomized_stroke = None
                if stroke_range is not None:
                    stroke_min = float(stroke_range[0])
                    stroke_max = float(stroke_range[1])
                    if stroke_max < stroke_min:
                        stroke_min, stroke_max = stroke_max, stroke_min
                    if stroke_max > stroke_min:
                        if shared_across_legs:
                            shared_stroke = torch.empty(
                                (env_ids.numel(), 1),
                                device=self.device,
                                dtype=leg_action_term.stroke_actual.dtype,
                            ).uniform_(stroke_min, stroke_max)
                            randomized_stroke = shared_stroke.repeat(1, leg_action_term.stroke_actual.shape[1])
                        else:
                            randomized_stroke = torch.empty(
                                (env_ids.numel(), leg_action_term.stroke_actual.shape[1]),
                                device=self.device,
                                dtype=leg_action_term.stroke_actual.dtype,
                            ).uniform_(stroke_min, stroke_max)
                    else:
                        randomized_stroke = torch.full(
                            (env_ids.numel(), leg_action_term.stroke_actual.shape[1]),
                            fill_value=stroke_min,
                            device=self.device,
                            dtype=leg_action_term.stroke_actual.dtype,
                        )
                if randomized_stroke is None:
                    stroke_noise_range = float(getattr(self.cfg, "initial_stroke_noise_range", 0.0))
                    if stroke_noise_range > 0.0:
                        stroke_noise = torch.empty(
                            (env_ids.numel(), leg_action_term.stroke_actual.shape[1]),
                            device=self.device,
                            dtype=leg_action_term.stroke_actual.dtype,
                        ).uniform_(-stroke_noise_range, stroke_noise_range)
                        randomized_stroke = leg_action_term.stroke_actual[env_ids] + stroke_noise
                if randomized_stroke is not None:
                    randomized_stroke = torch.clamp(
                        randomized_stroke,
                        min=leg_action_term._stroke_min,
                        max=leg_action_term._stroke_max,
                    )
                    position_target = (
                        leg_action_term._interp_stroke_to_joint_pos(randomized_stroke) * leg_action_term._joint_target_sign
                    )
                    leg_action_term._stroke_actual[env_ids] = randomized_stroke
                    leg_action_term._position_target[env_ids] = position_target
                    leg_action_term._raw_actions[env_ids] = 0.0
                    leg_action_term._processed_actions[env_ids] = 0.0
                    leg_action_term._effort_actual[env_ids] = 0.0
                    joint_pos = robot.data.default_joint_pos[env_ids].clone()
                    joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
                    joint_pos[:, self._leg_joint_ids] = position_target
                    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
                    if bool(getattr(self.cfg, "sync_reset_root_height_to_initial_stroke", False)):
                        root_height_reference = float(getattr(self.cfg, "initial_stroke_root_height_reference", 0.884))
                        nominal_stroke = float(getattr(self.cfg, "initial_stroke_root_height_nominal_stroke", 0.50))
                        slope = float(getattr(self.cfg, "initial_stroke_root_height_slope", 0.38))
                        stroke_mean = randomized_stroke.mean(dim=1)
                        target_root_height = root_height_reference - slope * (stroke_mean - nominal_stroke)
                        root_pose = robot.data.root_pose_w[env_ids].clone()
                        root_velocity = torch.zeros_like(robot.data.root_vel_w[env_ids])
                        root_pose[:, 2] = target_root_height
                        robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
                        robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        if self._is_stand_training_task:
            robot = self.scene["robot"]
            _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[env_ids])
            self._stand_reset_yaw[env_ids] = yaw
            self._stand_recent_reset_happened[env_ids] = True
            self._stand_last50_index[env_ids] = 0
            self._stand_last50_count[env_ids] = 0
            self._stand_last50_root_height[env_ids] = 0.0
            self._stand_last50_root_lin_vel_z[env_ids] = 0.0
            self._stand_last50_wheel_contact_over_weight[env_ids] = 0.0
            self._stand_last50_non_wheel_contact_over_weight[env_ids] = 0.0
        if self._is_stand_training_task and not self._stand_debug_metrics_enabled:
            self._update_short_goal_visualizer()
            return
        self.extras.setdefault("full_log", {})
        self.extras["full_log"].update(debug_logs)
        self.extras["log"].update(self._filter_forward_debug_logs_for_stdout(debug_logs))
        self._update_short_goal_visualizer()

    def render(self, recompute: bool = False) -> np.ndarray | None:
        return super().render(recompute=recompute)
