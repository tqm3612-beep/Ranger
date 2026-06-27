from __future__ import annotations

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
            "local_ground_height_b",
            "local_ground_height_w",
            "base_clearance",
            "clearance_error",
            "root_height_w",
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


    def _compute_forward_debug_metric_values(self) -> dict[str, torch.Tensor]:
        robot = self.scene["robot"]
        wheel_action_term = self.action_manager.get_term("wheel_motor_csv")
        leg_action_term = self.action_manager.get_term("leg_hydraulic")
        flat_base_clearance_params = self.cfg.rewards.flat_base_clearance.params
        flat_stroke_nominal_params = self.cfg.rewards.flat_stroke_nominal.params
        flat_stroke_asset_cfg = flat_stroke_nominal_params.get("asset_cfg", SceneEntityCfg("robot"))
        flat_base_asset_cfg = flat_base_clearance_params.get("asset_cfg", SceneEntityCfg("robot"))
        flat_stroke_asset_name = flat_stroke_asset_cfg.name
        flat_base_asset_name = flat_base_asset_cfg.name

        base_lin_vel_x = robot.data.root_lin_vel_b[:, 0]
        root_height_w = robot.data.root_pos_w[:, 2]
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        roll_deg = torch.rad2deg(roll)
        pitch_deg = torch.rad2deg(pitch)
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

        front_rear_stroke_diff = stroke_command[:, 1:3].mean(dim=1) - stroke_command[:, [0, 3]].mean(dim=1)
        left_right_stroke_diff = stroke_command[:, :2].mean(dim=1) - stroke_command[:, 2:].mean(dim=1)
        stroke_min = torch.min(stroke_command, dim=1).values
        stroke_max = torch.max(stroke_command, dim=1).values
        stroke_mean = torch.mean(stroke_command, dim=1)
        stroke_abs_mean = torch.mean(torch.abs(stroke_command), dim=1)
        stroke_rate_mean = torch.mean(stroke_rate, dim=1)
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
            "local_ground_height_b": local_ground_height_b,
            "local_ground_height_w": local_ground_height_w,
            "base_clearance": base_clearance,
            "clearance_error": clearance_error,
            "root_height_w": root_height_w,
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
        metric_values = self._compute_forward_debug_metric_values()
        for name, value in metric_values.items():
            self._forward_debug_metric_sums[name] += value
        self._forward_debug_metric_counts += 1.0

    def _consume_forward_debug_logs(self, env_ids) -> dict[str, float]:
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
        leg_action_term = self.action_manager.get_term("leg_hydraulic")
        self._prev_stroke_command[env_ids] = leg_action_term.stroke_command[env_ids]
        self.extras["log"].update(debug_logs)

    def render(self, recompute: bool = False) -> np.ndarray | None:
        return super().render(recompute=recompute)
