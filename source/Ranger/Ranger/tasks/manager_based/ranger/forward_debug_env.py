from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from . import mdp


class RangerForwardDebugEnv(ManagerBasedRLEnv):
    """Forward-stage environment with episode-level debug metrics for locomotion diagnosis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        robot = self.scene["robot"]
        self._wheel_joint_ids, _ = robot.find_joints(
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

        self._forward_debug_metric_names = (
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

        base_lin_vel_x = robot.data.root_lin_vel_b[:, 0]

        wheel_velocity_target = wheel_action_term.velocity_target
        wheel_joint_vel = robot.data.joint_vel[:, self._wheel_joint_ids]
        semantic_wheel_velocity_target = wheel_velocity_target * self._wheel_forward_sign
        semantic_wheel_joint_vel = wheel_joint_vel * self._wheel_forward_sign

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
        self.extras["log"].update(debug_logs)

    def render(self, recompute: bool = False) -> np.ndarray | None:
        return super().render(recompute=recompute)
