from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def compute_distillation_losses(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    stop_phase_mask: torch.Tensor,
    *,
    suspension_loss_weight: float = 1.0,
    wheel_loss_weight: float = 1.0,
    stop_phase_wheel_weight: float = 0.0,
    student_features: torch.Tensor | None = None,
    teacher_features: torch.Tensor | None = None,
    feature_loss_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Compute deterministic teacher-student losses with stop-phase wheel masking."""

    if student_actions.shape != teacher_actions.shape:
        raise ValueError(
            f"Action shape mismatch: student={tuple(student_actions.shape)} teacher={tuple(teacher_actions.shape)}"
        )
    if student_actions.shape[-1] != 8:
        raise ValueError(f"Ranger distillation expects 8 actions, got {student_actions.shape[-1]}.")
    if not 0.0 <= stop_phase_wheel_weight <= 1.0:
        raise ValueError(f"stop_phase_wheel_weight must be in [0, 1], got {stop_phase_wheel_weight}.")

    stop_phase_mask = stop_phase_mask.to(device=student_actions.device, dtype=torch.bool)
    if stop_phase_mask.shape != student_actions.shape[:-1]:
        raise ValueError(
            f"stop_phase_mask shape must be {tuple(student_actions.shape[:-1])}, got {tuple(stop_phase_mask.shape)}"
        )

    squared_error = (student_actions - teacher_actions).square()
    suspension_mse = squared_error[..., :4].mean()
    wheel_mse_raw = squared_error[..., 4:].mean()
    wheel_mse_per_sample = squared_error[..., 4:].mean(dim=-1)
    wheel_sample_weight = torch.where(
        stop_phase_mask,
        torch.full_like(wheel_mse_per_sample, float(stop_phase_wheel_weight)),
        torch.ones_like(wheel_mse_per_sample),
    )
    wheel_mse_effective = (wheel_mse_per_sample * wheel_sample_weight).mean()
    action_mse_total = squared_error.mean()

    if (student_features is None) != (teacher_features is None):
        raise ValueError("student_features and teacher_features must be provided together.")
    if student_features is None:
        feature_mse = student_actions.new_zeros(())
    else:
        if student_features.shape != teacher_features.shape:
            raise ValueError(
                f"Feature shape mismatch: student={tuple(student_features.shape)} "
                f"teacher={tuple(teacher_features.shape)}"
            )
        feature_mse = F.mse_loss(student_features, teacher_features)

    loss = (
        float(suspension_loss_weight) * suspension_mse
        + float(wheel_loss_weight) * wheel_mse_effective
        + float(feature_loss_weight) * feature_mse
    )
    action_cosine = F.cosine_similarity(student_actions, teacher_actions, dim=-1, eps=1.0e-8).mean()

    return {
        "loss": loss,
        "action_mse_total": action_mse_total,
        "suspension_mse": suspension_mse,
        "wheel_mse": wheel_mse_raw,
        "wheel_mse_effective": wheel_mse_effective,
        "feature_mse": feature_mse,
        "action_cosine": action_cosine,
        "stop_phase_rate": stop_phase_mask.float().mean(),
    }


def _detach_hidden_state(hidden_state):
    if hidden_state is None:
        return None
    if isinstance(hidden_state, tuple):
        return tuple(item.detach() for item in hidden_state)
    return hidden_state.detach()


def detach_recurrent_memory(policy) -> None:
    """Truncate the graph while preserving recurrent hidden values across rollout steps."""

    for branch_name in ("actor", "critic"):
        branch = getattr(policy, branch_name, None)
        memory = getattr(branch, "memory", None)
        if memory is not None and hasattr(memory, "hidden_state"):
            memory.hidden_state = _detach_hidden_state(memory.hidden_state)


def _mask_done_hidden_state(hidden_state, done_mask: torch.Tensor):
    """Zero done-env hidden values out-of-place while preserving other env autograd paths."""

    if hidden_state is None:
        return None
    if isinstance(hidden_state, tuple):
        return tuple(_mask_done_hidden_state(state, done_mask) for state in hidden_state)
    keep_mask = (~done_mask).to(device=hidden_state.device, dtype=hidden_state.dtype)
    mask_shape = [1] * hidden_state.ndim
    mask_shape[-2] = keep_mask.numel()
    return hidden_state * keep_mask.reshape(mask_shape)


def reset_recurrent_memory(policy, dones: torch.Tensor) -> float:
    """Reset only done envs without truncating recurrent graphs for the remaining envs."""

    done_mask = dones.reshape(-1).to(dtype=torch.bool)
    if not torch.any(done_mask):
        return 0.0
    for branch_name in ("actor", "critic"):
        branch = getattr(policy, branch_name, None)
        memory = getattr(branch, "memory", None)
        if memory is not None and hasattr(memory, "hidden_state"):
            memory.hidden_state = _mask_done_hidden_state(memory.hidden_state, done_mask)

    actor_hidden, _ = policy.get_hidden_states()
    if actor_hidden is None:
        return 0.0
    if isinstance(actor_hidden, tuple):
        selected = [state[..., done_mask, :] for state in actor_hidden]
        return max(float(state.abs().max().item()) if state.numel() else 0.0 for state in selected)
    selected = actor_hidden[..., done_mask, :]
    return float(selected.abs().max().item()) if selected.numel() else 0.0


def _actor_hidden_norms(policy) -> tuple[float, float]:
    actor_hidden, _ = policy.get_hidden_states()
    if actor_hidden is None:
        return 0.0, 0.0
    if isinstance(actor_hidden, tuple):
        flattened = torch.cat([state.reshape(-1) for state in actor_hidden])
    else:
        flattened = actor_hidden.reshape(-1)
    if flattened.numel() == 0:
        return 0.0, 0.0
    return float(flattened.abs().mean().item()), float(flattened.abs().max().item())


def _module_grad_norm(module: torch.nn.Module) -> float:
    """Return the module's total L2 gradient norm before global clipping."""

    squared_norms = [parameter.grad.detach().norm(2).square() for parameter in module.parameters() if parameter.grad is not None]
    if not squared_norms:
        return 0.0
    return float(torch.stack(squared_norms).sum().sqrt().item())


def _stop_phase_mask(env, num_envs: int, device: torch.device | str) -> torch.Tensor:
    stop_state = getattr(env.unwrapped, "_short_goal_stop_phase_active", None)
    if isinstance(stop_state, torch.Tensor) and stop_state.shape[0] == num_envs:
        return stop_state.to(device=device, dtype=torch.bool)
    return torch.zeros(num_envs, dtype=torch.bool, device=device)


def _unwrap_observations(obs_result):
    return obs_result[0] if isinstance(obs_result, tuple) else obs_result


def _parse_step_result(step_result):
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        dones = terminated | truncated
    else:
        obs, reward, dones, info = step_result
    return obs, reward, dones, info


def _restore_distillation_optimizer(
    checkpoint_path: Path,
    optimizer: torch.optim.Optimizer,
    *,
    learning_rate: float,
    map_location: torch.device | str,
) -> bool:
    """Restore distillation Adam state when available while keeping the requested LR."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    optimizer_state = checkpoint.get("distillation_optimizer_state_dict") if isinstance(checkpoint, dict) else None
    if optimizer_state is None:
        return False
    optimizer.load_state_dict(optimizer_state)
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(learning_rate)
    return True


def _select_rollout_actions(
    teacher_actions: torch.Tensor,
    student_actions: torch.Tensor,
    *,
    rollout_mode: str,
    teacher_action_blend: float,
) -> torch.Tensor:
    """Select deterministic environment actions for teacher, student, or blended rollout."""

    if rollout_mode == "teacher_rollout":
        return teacher_actions
    if rollout_mode == "student_rollout":
        return student_actions.detach()
    if rollout_mode != "blended_rollout":
        raise ValueError(f"Unsupported rollout_mode: {rollout_mode!r}")
    blend = float(teacher_action_blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"teacher_action_blend must be in [0, 1], got {blend}.")
    return blend * teacher_actions + (1.0 - blend) * student_actions.detach()


def _save_student_checkpoint(
    path: Path,
    *,
    student_runner,
    distillation_optimizer: torch.optim.Optimizer,
    infos: dict[str, Any],
) -> None:
    """Save the standard RSL-RL fields plus distillation metadata/state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ppo_optimizer = student_runner.alg.optimizer
    # PPO after distillation should start with optimizer moments consistent with
    # the distilled parameters, not stale moments from a pre-distillation resume.
    ppo_optimizer.state.clear()
    checkpoint = {
        "model_state_dict": student_runner.alg.policy.state_dict(),
        "optimizer_state_dict": ppo_optimizer.state_dict(),
        "iter": 0,
        "infos": infos,
        "distillation_optimizer_state_dict": distillation_optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Distill a feedforward Ranger V10 teacher into the recurrent C-stage policy.")
    parser.add_argument(
        "--task",
        type=str,
        default="Template-Ranger-ShortGoalFlat-C-Recurrent",
        help="Recurrent Ranger task used for student observations/environment behavior.",
    )
    parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Frozen feedforward V10 teacher checkpoint.")
    parser.add_argument(
        "--student_checkpoint",
        type=str,
        default=None,
        help="Optional recurrent checkpoint to continue distillation from. If omitted, student_init is used.",
    )
    parser.add_argument(
        "--student_init",
        choices=("v10_compatible", "random"),
        default="v10_compatible",
        help="Student initialization when no recurrent student checkpoint is supplied.",
    )
    parser.add_argument(
        "--rollout_mode",
        choices=("teacher_rollout", "blended_rollout", "student_rollout"),
        default="teacher_rollout",
        help="Which deterministic action source controls the environment while the teacher labels every visited state.",
    )
    parser.add_argument(
        "--teacher_action_blend",
        type=float,
        default=0.5,
        help="Teacher fraction for blended_rollout: env_action=blend*teacher+(1-blend)*student.",
    )
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--distill_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--bptt_steps",
        type=int,
        default=16,
        help=(
            "Truncated backpropagation-through-time window. Done environments reset only their own hidden-state graph; "
            "the batch window is flushed only at this length, save boundaries, or the final step."
        ),
    )
    parser.add_argument("--suspension_loss_weight", type=float, default=1.0)
    parser.add_argument("--wheel_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--stop_phase_wheel_weight",
        type=float,
        default=0.0,
        help="Multiplier for raw V10 wheel imitation during latched stop phase; 0 masks it completely.",
    )
    parser.add_argument(
        "--feature_loss_weight",
        type=float,
        default=0.05,
        help="Weak MSE weight between teacher/student 128-D actor trunk features.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=0, help="Optional intermediate checkpoint interval in steps; 0 disables.")
    parser.add_argument("--output_dir", type=str, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    if args_cli.num_envs <= 0 or args_cli.distill_steps <= 0:
        raise ValueError("--num_envs and --distill_steps must be positive.")
    if args_cli.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be positive.")
    if args_cli.bptt_steps <= 0:
        raise ValueError("--bptt_steps must be positive.")
    if not 0.0 <= args_cli.stop_phase_wheel_weight <= 1.0:
        raise ValueError("--stop_phase_wheel_weight must be in [0, 1].")
    if args_cli.feature_loss_weight < 0.0:
        raise ValueError("--feature_loss_weight must be non-negative.")
    if args_cli.max_grad_norm <= 0.0:
        raise ValueError("--max_grad_norm must be positive.")
    if args_cli.log_interval <= 0:
        raise ValueError("--log_interval must be positive.")
    if args_cli.save_interval < 0:
        raise ValueError("--save_interval must be non-negative.")

    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import Ranger.tasks  # noqa: F401
    from Ranger.tasks.manager_based.ranger.agents import RangerTerrainActorCritic, RangerTerrainActorCriticRecurrent
    from Ranger.tasks.manager_based.ranger.agents.rsl_rl_ppo_cfg import ShortGoalFlatV10PPORunnerCfg
    from warm_start import warm_start_ranger_recurrent_from_feedforward

    rsl_on_policy_runner.RangerTerrainActorCritic = RangerTerrainActorCritic
    rsl_on_policy_runner.RangerTerrainActorCriticRecurrent = RangerTerrainActorCriticRecurrent

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def run(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        student_agent_cfg: RslRlOnPolicyRunnerCfg,
    ) -> None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)
        env_cfg.seed = int(args_cli.seed)
        student_agent_cfg.seed = int(args_cli.seed)
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
            student_agent_cfg.device = args_cli.device

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=student_agent_cfg.clip_actions)

        student_runner = OnPolicyRunner(env, student_agent_cfg.to_dict(), log_dir=None, device=student_agent_cfg.device)
        student = student_runner.alg.policy
        if not getattr(student, "is_recurrent", False):
            raise TypeError(f"Distillation student must be recurrent, got {type(student).__name__}.")

        teacher_checkpoint_path = Path(args_cli.teacher_checkpoint).expanduser().resolve()
        student_checkpoint_path: Path | None = None
        if args_cli.student_checkpoint is not None:
            student_checkpoint_path = Path(args_cli.student_checkpoint).expanduser().resolve()
            print(f"[Distill] Loading recurrent student checkpoint: {student_checkpoint_path}")
            student_runner.load(str(student_checkpoint_path))
            student_runner.current_learning_iteration = 0
        elif args_cli.student_init == "v10_compatible":
            warm_start_ranger_recurrent_from_feedforward(student_runner, teacher_checkpoint_path)
        else:
            print("[Distill] Student initialization: random recurrent policy.")

        obs = _unwrap_observations(env.get_observations())
        num_actions = getattr(env, "num_actions", None)
        if num_actions is None:
            num_actions = int(env.unwrapped.action_manager.total_action_dim)
        else:
            num_actions = int(num_actions)

        teacher_runner_cfg = ShortGoalFlatV10PPORunnerCfg()
        teacher_policy_kwargs = teacher_runner_cfg.policy.to_dict()
        teacher_policy_kwargs.pop("class_name", None)
        teacher = RangerTerrainActorCritic(
            obs=obs,
            obs_groups=student_agent_cfg.obs_groups,
            num_actions=num_actions,
            **teacher_policy_kwargs,
        ).to(student_agent_cfg.device)

        teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=student_agent_cfg.device, weights_only=False)
        if not isinstance(teacher_checkpoint, dict) or "model_state_dict" not in teacher_checkpoint:
            raise KeyError("Teacher checkpoint must contain model_state_dict.")
        teacher.load_state_dict(teacher_checkpoint["model_state_dict"], strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        student.train()
        actor_parameters = [parameter for parameter in student.actor.parameters() if parameter.requires_grad]
        if not actor_parameters:
            raise RuntimeError("Student actor has no trainable parameters.")
        distillation_optimizer = torch.optim.Adam(actor_parameters, lr=float(args_cli.learning_rate))
        if student_checkpoint_path is not None:
            restored_optimizer = _restore_distillation_optimizer(
                student_checkpoint_path,
                distillation_optimizer,
                learning_rate=float(args_cli.learning_rate),
                map_location=student_agent_cfg.device,
            )
            if restored_optimizer:
                print("[Distill] Restored distillation optimizer state; using current --learning_rate.")
            else:
                print("[Distill] Student checkpoint has no distillation optimizer state; starting a new Adam optimizer.")

        captured_features: dict[str, torch.Tensor] = {}

        def capture_feature(name: str):
            def hook(_module, _inputs, output):
                captured_features[name] = output

            return hook

        teacher_hook = teacher.actor.actor_trunk_activation.register_forward_hook(capture_feature("teacher"))
        student_hook = student.actor.actor_trunk_activation.register_forward_hook(capture_feature("student"))

        if args_cli.output_dir is None:
            output_dir = Path("logs") / "rsl_rl" / "ranger_direct" / "distill_recurrent" / datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
        else:
            output_dir = Path(args_cli.output_dir).expanduser()
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "distillation_metrics.csv"
        final_checkpoint_path = output_dir / "model_distilled.pt"

        grad_module_names = (
            "prop_encoder",
            "goal_encoder",
            "map_encoder",
            "fusion_norm",
            "memory",
            "hidden_norm",
            "actor_trunk",
            "suspension_head",
            "wheel_head",
        )
        metric_names = [
            "step",
            "loss",
            "action_mse_total",
            "suspension_mse",
            "wheel_mse",
            "wheel_mse_effective",
            "feature_mse",
            "action_cosine",
            "teacher_susp_abs_mean",
            "teacher_susp_abs_max",
            "teacher_wheel_abs_mean",
            "teacher_wheel_abs_max",
            "student_susp_abs_mean",
            "student_susp_abs_max",
            "student_wheel_abs_mean",
            "student_wheel_abs_max",
            "stop_phase_rate",
            "done_rate",
            "bptt_window",
            "actor_hidden_abs_mean",
            "actor_hidden_abs_max",
            "hidden_reset_max_abs",
            "grad_norm",
            *(f"grad_norm_{module_name}" for module_name in grad_module_names),
        ]
        metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metric_names)
        metrics_writer.writeheader()
        window_sums = {name: 0.0 for name in metric_names if name != "step"}
        window_count = 0

        print(f"[Distill] teacher: {teacher_checkpoint_path}")
        print(f"[Distill] rollout_mode: {args_cli.rollout_mode}")
        if args_cli.rollout_mode == "blended_rollout":
            print(f"[Distill] teacher_action_blend: {args_cli.teacher_action_blend}")
        print(f"[Distill] output_dir: {output_dir}")
        print(f"[Distill] feature_loss_weight: {args_cli.feature_loss_weight}")
        print(f"[Distill] stop_phase_wheel_weight: {args_cli.stop_phase_wheel_weight}")
        print(f"[Distill] bptt_steps: {args_cli.bptt_steps}")

        pending_loss: torch.Tensor | None = None
        bptt_count = 0
        distillation_optimizer.zero_grad(set_to_none=True)
        try:
            for step in range(1, int(args_cli.distill_steps) + 1):
                captured_features.clear()
                stop_mask = _stop_phase_mask(env, int(args_cli.num_envs), student_agent_cfg.device)

                with torch.no_grad():
                    teacher_actions = teacher.act_inference(obs)
                    teacher_features = captured_features["teacher"].detach()

                student_actions = student.act_inference(obs)
                student_features = captured_features["student"]
                losses = compute_distillation_losses(
                    student_actions,
                    teacher_actions,
                    stop_mask,
                    suspension_loss_weight=float(args_cli.suspension_loss_weight),
                    wheel_loss_weight=float(args_cli.wheel_loss_weight),
                    stop_phase_wheel_weight=float(args_cli.stop_phase_wheel_weight),
                    student_features=student_features,
                    teacher_features=teacher_features,
                    feature_loss_weight=float(args_cli.feature_loss_weight),
                )
                pending_loss = losses["loss"] if pending_loss is None else pending_loss + losses["loss"]
                bptt_count += 1

                env_actions = _select_rollout_actions(
                    teacher_actions,
                    student_actions,
                    rollout_mode=args_cli.rollout_mode,
                    teacher_action_blend=float(args_cli.teacher_action_blend),
                )
                step_result = env.step(env_actions)
                obs, _, dones, _ = _parse_step_result(step_result)

                done_any = bool(torch.any(dones.to(dtype=torch.bool)).item())
                save_due = int(args_cli.save_interval) > 0 and step % int(args_cli.save_interval) == 0
                should_flush = (
                    bptt_count >= int(args_cli.bptt_steps)
                    or step == int(args_cli.distill_steps)
                    or save_due
                )
                grad_norm_value = 0.0
                bptt_window_value = 0
                module_grad_norms = {f"grad_norm_{module_name}": 0.0 for module_name in grad_module_names}
                if should_flush:
                    assert pending_loss is not None and bptt_count > 0
                    bptt_window_value = bptt_count
                    (pending_loss / bptt_count).backward()
                    module_grad_norms = {
                        f"grad_norm_{module_name}": _module_grad_norm(getattr(student.actor, module_name))
                        for module_name in grad_module_names
                    }
                    grad_norm = torch.nn.utils.clip_grad_norm_(actor_parameters, float(args_cli.max_grad_norm))
                    distillation_optimizer.step()
                    grad_norm_value = float(
                        grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    )
                    distillation_optimizer.zero_grad(set_to_none=True)
                    detach_recurrent_memory(student)
                    pending_loss = None
                    bptt_count = 0

                reset_max = reset_recurrent_memory(student, dones) if done_any else 0.0
                hidden_mean, hidden_max = _actor_hidden_norms(student)

                row = {
                    "step": step,
                    "loss": float(losses["loss"].detach().item()),
                    "action_mse_total": float(losses["action_mse_total"].detach().item()),
                    "suspension_mse": float(losses["suspension_mse"].detach().item()),
                    "wheel_mse": float(losses["wheel_mse"].detach().item()),
                    "wheel_mse_effective": float(losses["wheel_mse_effective"].detach().item()),
                    "feature_mse": float(losses["feature_mse"].detach().item()),
                    "action_cosine": float(losses["action_cosine"].detach().item()),
                    "teacher_susp_abs_mean": float(teacher_actions[..., :4].abs().mean().item()),
                    "teacher_susp_abs_max": float(teacher_actions[..., :4].abs().max().item()),
                    "teacher_wheel_abs_mean": float(teacher_actions[..., 4:].abs().mean().item()),
                    "teacher_wheel_abs_max": float(teacher_actions[..., 4:].abs().max().item()),
                    "student_susp_abs_mean": float(student_actions[..., :4].detach().abs().mean().item()),
                    "student_susp_abs_max": float(student_actions[..., :4].detach().abs().max().item()),
                    "student_wheel_abs_mean": float(student_actions[..., 4:].detach().abs().mean().item()),
                    "student_wheel_abs_max": float(student_actions[..., 4:].detach().abs().max().item()),
                    "stop_phase_rate": float(losses["stop_phase_rate"].detach().item()),
                    "done_rate": float(dones.to(dtype=torch.bool).float().mean().item()),
                    "bptt_window": bptt_window_value,
                    "actor_hidden_abs_mean": hidden_mean,
                    "actor_hidden_abs_max": hidden_max,
                    "hidden_reset_max_abs": reset_max,
                    "grad_norm": grad_norm_value,
                    **module_grad_norms,
                }
                metrics_writer.writerow(row)
                window_count += 1
                for name in window_sums:
                    window_sums[name] += float(row[name])

                if step % int(args_cli.log_interval) == 0 or step == int(args_cli.distill_steps):
                    averages = {name: value / window_count for name, value in window_sums.items()}
                    print(
                        f"[Distill] step={step} loss={averages['loss']:.6f} "
                        f"action_mse={averages['action_mse_total']:.6f} "
                        f"susp={averages['suspension_mse']:.6f} wheel={averages['wheel_mse']:.6f} "
                        f"feature={averages['feature_mse']:.6f} cosine={averages['action_cosine']:.4f} "
                        f"hidden_max={averages['actor_hidden_abs_max']:.4f} reset_max={averages['hidden_reset_max_abs']:.2e}"
                    )
                    metrics_file.flush()
                    window_sums = {name: 0.0 for name in window_sums}
                    window_count = 0

                if int(args_cli.save_interval) > 0 and step % int(args_cli.save_interval) == 0:
                    _save_student_checkpoint(
                        output_dir / f"model_distilled_step_{step}.pt",
                        student_runner=student_runner,
                        distillation_optimizer=distillation_optimizer,
                        infos={
                            "distillation_step": step,
                            "rollout_mode": args_cli.rollout_mode,
                            "teacher_action_blend": float(args_cli.teacher_action_blend),
                            "teacher_checkpoint": str(teacher_checkpoint_path),
                        },
                    )

            _save_student_checkpoint(
                final_checkpoint_path,
                student_runner=student_runner,
                distillation_optimizer=distillation_optimizer,
                infos={
                    "distillation_step": int(args_cli.distill_steps),
                    "rollout_mode": args_cli.rollout_mode,
                    "teacher_action_blend": float(args_cli.teacher_action_blend),
                    "teacher_checkpoint": str(teacher_checkpoint_path),
                    "student_init": args_cli.student_init,
                    "stop_phase_wheel_weight": float(args_cli.stop_phase_wheel_weight),
                    "feature_loss_weight": float(args_cli.feature_loss_weight),
                },
            )
            print(f"[Distill] saved recurrent student checkpoint: {final_checkpoint_path}")
            print(f"[Distill] metrics: {metrics_path}")
        finally:
            teacher_hook.remove()
            student_hook.remove()
            metrics_file.close()
            env.close()

    try:
        run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
