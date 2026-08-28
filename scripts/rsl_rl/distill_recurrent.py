from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict


POLICY_STATE_GROUP = "policy_state"
TEACHER_COMMAND_GROUP = "teacher_command"
COMMAND_STATE_SLICE = slice(26, 34)


def ranger_wheel_semantic_modes(wheel_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return forward common-mode and signed turn-mode from raw ``[lr, lf, rf, rr]`` actions."""

    if wheel_actions.shape[-1] != 4:
        raise ValueError(f"Expected four Ranger wheel actions, got shape={tuple(wheel_actions.shape)}")
    semantic_wheel = wheel_actions * wheel_actions.new_tensor((-1.0, -1.0, 1.0, 1.0))
    left = semantic_wheel[..., :2].mean(dim=-1)
    right = semantic_wheel[..., 2:].mean(dim=-1)
    return 0.5 * (left + right), 0.5 * (right - left)


def bound_teacher_actions(teacher_actions: torch.Tensor) -> torch.Tensor:
    """Convert the legacy linear V10 output to the normalized command actually executed by the environment."""

    return torch.clamp(teacher_actions, min=-1.0, max=1.0)


class SequenceReplayAnchor:
    """Fixed-length recurrent sequences collected from the proven teacher policy."""

    def __init__(
        self,
        *,
        num_envs: int,
        sequence_length: int,
        capacity_sequences: int,
        observation_keys: tuple[str, ...],
        seed: int,
    ) -> None:
        if num_envs <= 0 or sequence_length <= 0 or capacity_sequences <= 0:
            raise ValueError("Replay dimensions and capacity must be positive.")
        if not observation_keys:
            raise ValueError("Replay requires at least one observation key.")
        self.num_envs = int(num_envs)
        self.sequence_length = int(sequence_length)
        self.observation_keys = tuple(observation_keys)
        self._sequences: deque[dict[str, torch.Tensor]] = deque(maxlen=int(capacity_sequences))
        self._partial: list[dict[str, list[torch.Tensor]]] = [
            {
                **{key: [] for key in self.observation_keys},
                "teacher_actions": [],
                "stop_mask": [],
                "goal_hidden_mask": [],
            }
            for _ in range(self.num_envs)
        ]
        self._random = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self._sequences)

    def reset_partial_sequences(self) -> None:
        for partial in self._partial:
            for values in partial.values():
                values.clear()

    def append_step(
        self,
        observations,
        teacher_actions: torch.Tensor,
        stop_mask: torch.Tensor,
        dones: torch.Tensor,
        goal_hidden_mask: torch.Tensor | None = None,
    ) -> None:
        if teacher_actions.shape != (self.num_envs, 8):
            raise ValueError(
                f"Replay expected teacher actions {(self.num_envs, 8)}, got {tuple(teacher_actions.shape)}."
            )
        done_mask = dones.reshape(-1).to(dtype=torch.bool)
        stop_mask = stop_mask.reshape(-1).to(dtype=torch.bool)
        if tuple(done_mask.shape) != (self.num_envs,) or tuple(stop_mask.shape) != (self.num_envs,):
            raise ValueError("Replay done and stop masks must contain one value per environment.")
        if goal_hidden_mask is None:
            goal_hidden_mask = torch.zeros_like(stop_mask)
        goal_hidden_mask = goal_hidden_mask.reshape(-1).to(dtype=torch.bool)
        if tuple(goal_hidden_mask.shape) != (self.num_envs,):
            raise ValueError("Replay goal-hidden mask must contain one value per environment.")

        for env_index in range(self.num_envs):
            partial = self._partial[env_index]
            for key in self.observation_keys:
                partial[key].append(observations[key][env_index].detach().to(device="cpu").clone())
            partial["teacher_actions"].append(
                teacher_actions[env_index].detach().to(device="cpu").clone()
            )
            partial["stop_mask"].append(stop_mask[env_index].detach().to(device="cpu").clone())
            partial["goal_hidden_mask"].append(
                goal_hidden_mask[env_index].detach().to(device="cpu").clone()
            )

            if len(partial["teacher_actions"]) == self.sequence_length:
                self._sequences.append(
                    {key: torch.stack(values, dim=0) for key, values in partial.items()}
                )
                for values in partial.values():
                    values.clear()

            if bool(done_mask[env_index].item()):
                for values in partial.values():
                    values.clear()

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        *,
        require_goal_transition: bool = False,
        burn_in_steps: int = 0,
    ) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("Replay batch size must be positive.")
        if not self._sequences:
            raise RuntimeError("Cannot sample an empty replay anchor.")
        if not 0 <= int(burn_in_steps) < self.sequence_length:
            raise ValueError("burn_in_steps must be inside the replay sequence.")
        eligible_indices = list(range(len(self._sequences)))
        if require_goal_transition:
            eligible_indices = []
            for index, sequence in enumerate(self._sequences):
                hidden = sequence["goal_hidden_mask"].to(dtype=torch.bool)
                seen_visible = torch.cumsum((~hidden).to(dtype=torch.long), dim=0) > 0
                valid_hidden = hidden & seen_visible
                if torch.any(valid_hidden[int(burn_in_steps) :]):
                    eligible_indices.append(index)
            if not eligible_indices:
                raise RuntimeError(
                    "Replay contains no visible-to-hidden goal transition with supervised hidden steps."
                )
        indices = [eligible_indices[self._random.randrange(len(eligible_indices))] for _ in range(int(batch_size))]
        samples = [self._sequences[index] for index in indices]
        keys = (*self.observation_keys, "teacher_actions", "stop_mask", "goal_hidden_mask")
        return {
            key: torch.stack([sample[key] for sample in samples], dim=1).to(device=device)
            for key in keys
        }


def recurrent_sequence_actions(policy, replay_batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Evaluate a replay sequence from a clean GRU state without mutating online rollout memory."""

    actor_obs = torch.cat([replay_batch[group] for group in policy.obs_groups["policy"]], dim=-1)
    actor_obs = policy.actor_obs_normalizer(actor_obs)
    time_steps, batch_size = actor_obs.shape[:2]
    masks = torch.ones(time_steps, batch_size, dtype=torch.bool, device=actor_obs.device)
    hidden_state = actor_obs.new_zeros(policy.rnn_num_layers, batch_size, policy.rnn_hidden_dim)
    raw_mean = policy._apply_action_output_mask(
        policy.actor(actor_obs, masks=masks, hidden_state=hidden_state)
    )
    latent_mean = policy._bound_latent_mean(raw_mean)
    return torch.tanh(latent_mean)


def build_teacher_observations(observations: TensorDict) -> TensorDict:
    """Replace only the student's masked command with the privileged teacher command."""

    if TEACHER_COMMAND_GROUP not in observations.keys():
        return observations
    policy_state = observations[POLICY_STATE_GROUP]
    teacher_command = observations[TEACHER_COMMAND_GROUP]
    if policy_state.shape[-1] != 42 or teacher_command.shape[-1] != 8:
        raise ValueError(
            f"Unexpected asymmetric observation shapes: policy_state={tuple(policy_state.shape)}, "
            f"teacher_command={tuple(teacher_command.shape)}."
        )
    teacher_observations = observations.clone(recurse=False)
    teacher_policy_state = policy_state.clone()
    teacher_policy_state[..., COMMAND_STATE_SLICE] = teacher_command
    teacher_observations.set(POLICY_STATE_GROUP, teacher_policy_state)
    return teacher_observations


def goal_hidden_mask(observations: TensorDict) -> torch.Tensor:
    """Return which student observations currently hide the goal descriptor."""

    if TEACHER_COMMAND_GROUP not in observations.keys():
        return torch.zeros(
            observations[POLICY_STATE_GROUP].shape[:-1],
            dtype=torch.bool,
            device=observations[POLICY_STATE_GROUP].device,
        )
    student_command = observations[POLICY_STATE_GROUP][..., COMMAND_STATE_SLICE]
    teacher_command = observations[TEACHER_COMMAND_GROUP]
    return torch.any((student_command[..., 3:8] - teacher_command[..., 3:8]).abs() > 1.0e-6, dim=-1)


def goal_hidden_sample_weights(hidden_mask: torch.Tensor, hidden_weight: float) -> torch.Tensor:
    """Build stable-mean sample weights that emphasize hidden-goal sequence steps."""

    if hidden_weight < 0.0:
        raise ValueError(f"hidden_weight must be non-negative, got {hidden_weight}.")
    weights = torch.where(
        hidden_mask.to(dtype=torch.bool),
        torch.full_like(hidden_mask, float(hidden_weight), dtype=torch.float32),
        torch.ones_like(hidden_mask, dtype=torch.float32),
    )
    return weights / weights.mean().clamp_min(1.0e-8)


def compute_distillation_losses(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    stop_phase_mask: torch.Tensor,
    *,
    suspension_loss_weight: float = 1.0,
    wheel_loss_weight: float = 1.0,
    wheel_common_loss_weight: float = 0.0,
    wheel_turn_loss_weight: float = 0.0,
    stop_phase_wheel_weight: float = 0.0,
    turn_sign_threshold: float = 0.05,
    student_features: torch.Tensor | None = None,
    teacher_features: torch.Tensor | None = None,
    feature_loss_weight: float = 0.05,
    sample_weights: torch.Tensor | None = None,
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
    if wheel_common_loss_weight < 0.0 or wheel_turn_loss_weight < 0.0:
        raise ValueError("Semantic wheel loss weights must be non-negative.")
    if turn_sign_threshold < 0.0:
        raise ValueError("turn_sign_threshold must be non-negative.")

    teacher_actions = bound_teacher_actions(teacher_actions)

    stop_phase_mask = stop_phase_mask.to(device=student_actions.device, dtype=torch.bool)
    if stop_phase_mask.shape != student_actions.shape[:-1]:
        raise ValueError(
            f"stop_phase_mask shape must be {tuple(student_actions.shape[:-1])}, got {tuple(stop_phase_mask.shape)}"
        )
    if sample_weights is None:
        sample_weights = torch.ones_like(stop_phase_mask, dtype=student_actions.dtype)
    else:
        sample_weights = sample_weights.to(device=student_actions.device, dtype=student_actions.dtype)
        if sample_weights.shape != student_actions.shape[:-1]:
            raise ValueError(
                f"sample_weights shape must be {tuple(student_actions.shape[:-1])}, "
                f"got {tuple(sample_weights.shape)}"
            )
        if torch.any(sample_weights < 0.0):
            raise ValueError("sample_weights must be non-negative.")
        if not torch.any(sample_weights > 0.0):
            raise ValueError("sample_weights must contain at least one positive value.")
    sample_weights = sample_weights / sample_weights.mean().clamp_min(1.0e-8)

    squared_error = (student_actions - teacher_actions).square()
    suspension_mse = (squared_error[..., :4].mean(dim=-1) * sample_weights).mean()
    wheel_mse_per_sample = squared_error[..., 4:].mean(dim=-1)
    wheel_mse_raw = (wheel_mse_per_sample * sample_weights).mean()
    wheel_sample_weight = torch.where(
        stop_phase_mask,
        torch.full_like(wheel_mse_per_sample, float(stop_phase_wheel_weight)),
        torch.ones_like(wheel_mse_per_sample),
    )
    effective_wheel_weight = wheel_sample_weight * sample_weights
    wheel_mse_effective = (wheel_mse_per_sample * effective_wheel_weight).mean()
    action_mse_total = (squared_error.mean(dim=-1) * sample_weights).mean()

    student_common, student_turn = ranger_wheel_semantic_modes(student_actions[..., 4:8])
    teacher_common, teacher_turn = ranger_wheel_semantic_modes(teacher_actions[..., 4:8])
    wheel_common_mse = ((student_common - teacher_common).square() * effective_wheel_weight).mean()
    wheel_turn_mse = ((student_turn - teacher_turn).square() * effective_wheel_weight).mean()
    turn_sign_valid = (~stop_phase_mask) & (teacher_turn.abs() >= float(turn_sign_threshold))
    wrong_turn_sign = turn_sign_valid & (student_turn * teacher_turn < 0.0)
    if torch.any(turn_sign_valid):
        wrong_turn_sign_rate = wrong_turn_sign.to(student_actions.dtype).sum() / turn_sign_valid.sum()
    else:
        wrong_turn_sign_rate = student_actions.new_zeros(())
    teacher_turn_abs_mean = teacher_turn[turn_sign_valid].abs().mean() if torch.any(turn_sign_valid) else student_actions.new_zeros(())
    student_turn_abs_mean = student_turn[turn_sign_valid].abs().mean() if torch.any(turn_sign_valid) else student_actions.new_zeros(())

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
        feature_mse_per_sample = (student_features - teacher_features).square().mean(dim=-1)
        feature_mse = (feature_mse_per_sample * sample_weights).mean()

    loss = (
        float(suspension_loss_weight) * suspension_mse
        + float(wheel_loss_weight) * wheel_mse_effective
        + float(wheel_common_loss_weight) * wheel_common_mse
        + float(wheel_turn_loss_weight) * wheel_turn_mse
        + float(feature_loss_weight) * feature_mse
    )
    action_cosine = (
        F.cosine_similarity(student_actions, teacher_actions, dim=-1, eps=1.0e-8) * sample_weights
    ).mean()

    return {
        "loss": loss,
        "action_mse_total": action_mse_total,
        "suspension_mse": suspension_mse,
        "wheel_mse": wheel_mse_raw,
        "wheel_mse_effective": wheel_mse_effective,
        "wheel_common_mse": wheel_common_mse,
        "wheel_turn_mse": wheel_turn_mse,
        "wrong_turn_sign_rate": wrong_turn_sign_rate,
        "teacher_turn_abs_mean": teacher_turn_abs_mean,
        "student_turn_abs_mean": student_turn_abs_mean,
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


def _build_teacher_anchor_mask(
    num_envs: int,
    teacher_anchor_fraction: float,
    *,
    device: torch.device | str,
    seed: int,
) -> torch.Tensor:
    """Create a fixed seeded split between teacher-anchor and student-rollout environments."""

    fraction = float(teacher_anchor_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"teacher_anchor_fraction must be in (0, 1), got {fraction}.")
    teacher_count = int(round(num_envs * fraction))
    teacher_count = max(1, min(num_envs - 1, teacher_count))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    teacher_indices = torch.randperm(num_envs, generator=generator)[:teacher_count]
    mask = torch.zeros(num_envs, dtype=torch.bool)
    mask[teacher_indices] = True
    return mask.to(device=device)


def _select_rollout_actions(
    teacher_actions: torch.Tensor,
    student_actions: torch.Tensor,
    *,
    rollout_mode: str,
    teacher_action_blend: float,
    teacher_anchor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select deterministic environment actions for teacher, student, blended, or dual-distribution rollout."""

    if rollout_mode == "teacher_rollout":
        return teacher_actions
    if rollout_mode == "student_rollout":
        return student_actions.detach()
    if rollout_mode == "dual_distribution":
        if teacher_anchor_mask is None:
            raise ValueError("teacher_anchor_mask is required for dual_distribution rollout.")
        mask = teacher_anchor_mask.to(device=teacher_actions.device, dtype=torch.bool).reshape(-1)
        if mask.shape[0] != teacher_actions.shape[0]:
            raise ValueError(
                f"teacher_anchor_mask must have {teacher_actions.shape[0]} entries, got {mask.shape[0]}."
            )
        return torch.where(mask.unsqueeze(-1), teacher_actions, student_actions.detach())
    if rollout_mode != "blended_rollout":
        raise ValueError(f"Unsupported rollout_mode: {rollout_mode!r}")
    blend = float(teacher_action_blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"teacher_action_blend must be in [0, 1], got {blend}.")
    return blend * teacher_actions + (1.0 - blend) * student_actions.detach()


def _masked_distillation_losses(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    stop_phase_mask: torch.Tensor,
    env_mask: torch.Tensor,
    *,
    suspension_loss_weight: float,
    wheel_loss_weight: float,
    wheel_common_loss_weight: float,
    wheel_turn_loss_weight: float,
    stop_phase_wheel_weight: float,
    turn_sign_threshold: float,
    student_features: torch.Tensor | None,
    teacher_features: torch.Tensor | None,
    feature_loss_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute the standard distillation loss on one selected environment group."""

    mask = env_mask.to(device=student_actions.device, dtype=torch.bool).reshape(-1)
    if mask.shape[0] != student_actions.shape[0] or not torch.any(mask):
        raise ValueError("env_mask must select at least one environment from the current batch.")
    selected_student_features = None if student_features is None else student_features[mask]
    selected_teacher_features = None if teacher_features is None else teacher_features[mask]
    return compute_distillation_losses(
        student_actions[mask],
        teacher_actions[mask],
        stop_phase_mask.to(device=student_actions.device, dtype=torch.bool).reshape(-1)[mask],
        suspension_loss_weight=float(suspension_loss_weight),
        wheel_loss_weight=float(wheel_loss_weight),
        wheel_common_loss_weight=float(wheel_common_loss_weight),
        wheel_turn_loss_weight=float(wheel_turn_loss_weight),
        stop_phase_wheel_weight=float(stop_phase_wheel_weight),
        turn_sign_threshold=float(turn_sign_threshold),
        student_features=selected_student_features,
        teacher_features=selected_teacher_features,
        feature_loss_weight=float(feature_loss_weight),
    )


def _group_action_diagnostics(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    stop_phase_mask: torch.Tensor,
    env_mask: torch.Tensor,
    *,
    stop_phase_wheel_weight: float,
) -> dict[str, float]:
    """Compute detached action diagnostics for one rollout-distribution group."""

    mask = env_mask.to(device=student_actions.device, dtype=torch.bool).reshape(-1)
    if mask.shape[0] != student_actions.shape[0] or not torch.any(mask):
        raise ValueError("env_mask must select at least one environment from the current batch.")
    with torch.no_grad():
        student_group = student_actions.detach()[mask]
        teacher_group = teacher_actions.detach()[mask]
        stop_group = stop_phase_mask.to(device=student_actions.device, dtype=torch.bool).reshape(-1)[mask]
        squared_error = (student_group - teacher_group).square()
        wheel_mse_per_sample = squared_error[..., 4:].mean(dim=-1)
        wheel_weights = torch.where(
            stop_group,
            torch.full_like(wheel_mse_per_sample, float(stop_phase_wheel_weight)),
            torch.ones_like(wheel_mse_per_sample),
        )
        return {
            "action_mse_total": float(squared_error.mean().item()),
            "wheel_mse_effective": float((wheel_mse_per_sample * wheel_weights).mean().item()),
            "action_cosine": float(
                F.cosine_similarity(student_group, teacher_group, dim=-1, eps=1.0e-8).mean().item()
            ),
            "stop_phase_rate": float(stop_group.float().mean().item()),
        }


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
        default="Template-Ranger-Stage2-ControlTransfer",
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
        choices=("teacher_rollout", "blended_rollout", "student_rollout", "dual_distribution"),
        default="student_rollout",
        help=(
            "Which deterministic action source controls the environment while the teacher labels every visited state. "
            "dual_distribution keeps separate teacher-anchor and pure-student env trajectories in the same batch."
        ),
    )
    parser.add_argument(
        "--teacher_action_blend",
        type=float,
        default=0.5,
        help="Teacher fraction for blended_rollout: env_action=blend*teacher+(1-blend)*student.",
    )
    parser.add_argument(
        "--teacher_anchor_fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of environments controlled by 100% teacher actions in dual_distribution mode. "
            "The remaining environments are controlled by 100% student actions."
        ),
    )
    parser.add_argument(
        "--teacher_anchor_loss_weight",
        type=float,
        default=0.5,
        help=(
            "Loss fraction assigned to teacher-anchor environments in dual_distribution mode. "
            "The student-rollout loss receives 1-weight. This is independent of teacher_anchor_fraction."
        ),
    )
    parser.add_argument(
        "--reset_distillation_optimizer",
        action="store_true",
        default=False,
        help="Do not restore Adam moments from --student_checkpoint when starting a new distillation regime.",
    )
    parser.add_argument(
        "--freeze_control_backbone",
        action="store_true",
        default=False,
        help="Freeze the V10-compatible map encoder, actor trunk, and action heads during control recovery.",
    )
    parser.add_argument(
        "--freeze_memory_control_path",
        action="store_true",
        default=False,
        help=(
            "Freeze map encoder, actor trunk, action heads, and direct wheel-goal residual while training memory."
        ),
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
        "--wheel_common_loss_weight",
        type=float,
        default=0.0,
        help="Additional imitation weight for average forward wheel command.",
    )
    parser.add_argument(
        "--wheel_turn_loss_weight",
        type=float,
        default=0.0,
        help="Additional imitation weight for left/right differential wheel command.",
    )
    parser.add_argument(
        "--turn_sign_threshold",
        type=float,
        default=0.05,
        help="Ignore near-zero teacher turn commands when reporting wrong turn direction.",
    )
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
    parser.add_argument(
        "--goal_hidden_loss_weight",
        type=float,
        default=1.0,
        help="Relative sequence imitation weight for timesteps where the student's goal is hidden.",
    )
    parser.add_argument(
        "--anchor_collect_steps",
        type=int,
        default=0,
        help="Teacher-controlled pre-collection steps for the fixed recurrent replay anchor; 0 disables replay.",
    )
    parser.add_argument(
        "--anchor_sequence_length",
        type=int,
        default=32,
        help="Length of every contiguous replay sequence.",
    )
    parser.add_argument(
        "--anchor_capacity_sequences",
        type=int,
        default=512,
        help="Maximum teacher sequences kept in the in-memory replay anchor.",
    )
    parser.add_argument(
        "--anchor_batch_sequences",
        type=int,
        default=8,
        help="Replay sequences sampled at each online BPTT update.",
    )
    parser.add_argument(
        "--anchor_loss_weight",
        type=float,
        default=0.0,
        help="Weight of replay imitation relative to current student-rollout imitation.",
    )
    parser.add_argument(
        "--dagger_rounds",
        type=int,
        default=0,
        help="Run fixed-policy student collection followed by offline replay updates; 0 uses the legacy online loop.",
    )
    parser.add_argument(
        "--dagger_collect_steps",
        type=int,
        default=1400,
        help="Student-controlled environment steps collected with frozen weights in every DAgger round.",
    )
    parser.add_argument(
        "--dagger_updates_per_round",
        type=int,
        default=256,
        help="Offline recurrent replay updates after each fixed-policy collection round.",
    )
    parser.add_argument(
        "--dagger_student_capacity_sequences",
        type=int,
        default=1024,
        help="Maximum aggregated student-state sequences retained across DAgger rounds.",
    )
    parser.add_argument(
        "--dagger_student_batch_sequences",
        type=int,
        default=8,
        help="Aggregated student sequences sampled per offline update.",
    )
    parser.add_argument(
        "--replay_burn_in_steps",
        type=int,
        default=8,
        help="Sequence prefix used only to reconstruct GRU state before imitation loss is evaluated.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=0, help="Optional intermediate checkpoint interval in steps; 0 disables.")
    parser.add_argument("--output_dir", type=str, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if args_cli.distill_steps <= 0 and args_cli.dagger_rounds <= 0:
        raise ValueError("--distill_steps must be positive unless offline DAgger is enabled.")
    if args_cli.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be positive.")
    if args_cli.bptt_steps <= 0:
        raise ValueError("--bptt_steps must be positive.")
    if args_cli.rollout_mode == "dual_distribution":
        if args_cli.num_envs < 2:
            raise ValueError("dual_distribution rollout requires --num_envs >= 2.")
        if not 0.0 < args_cli.teacher_anchor_fraction < 1.0:
            raise ValueError("--teacher_anchor_fraction must be in (0, 1) for dual_distribution rollout.")
        if not 0.0 <= args_cli.teacher_anchor_loss_weight <= 1.0:
            raise ValueError("--teacher_anchor_loss_weight must be in [0, 1] for dual_distribution rollout.")
    if not 0.0 <= args_cli.stop_phase_wheel_weight <= 1.0:
        raise ValueError("--stop_phase_wheel_weight must be in [0, 1].")
    if args_cli.feature_loss_weight < 0.0:
        raise ValueError("--feature_loss_weight must be non-negative.")
    if args_cli.goal_hidden_loss_weight < 0.0:
        raise ValueError("--goal_hidden_loss_weight must be non-negative.")
    if args_cli.wheel_common_loss_weight < 0.0 or args_cli.wheel_turn_loss_weight < 0.0:
        raise ValueError("Semantic wheel loss weights must be non-negative.")
    if args_cli.turn_sign_threshold < 0.0:
        raise ValueError("--turn_sign_threshold must be non-negative.")
    if args_cli.anchor_collect_steps < 0:
        raise ValueError("--anchor_collect_steps must be non-negative.")
    replay_values = (
        args_cli.anchor_sequence_length,
        args_cli.anchor_capacity_sequences,
        args_cli.anchor_batch_sequences,
    )
    if args_cli.anchor_collect_steps > 0 and any(value <= 0 for value in replay_values):
        raise ValueError("Replay sequence length, capacity, and batch size must be positive when replay is enabled.")
    if args_cli.anchor_loss_weight < 0.0:
        raise ValueError("--anchor_loss_weight must be non-negative.")
    if args_cli.anchor_loss_weight > 0.0 and args_cli.anchor_collect_steps == 0:
        raise ValueError("--anchor_loss_weight > 0 requires --anchor_collect_steps > 0.")
    if args_cli.dagger_rounds < 0:
        raise ValueError("--dagger_rounds must be non-negative.")
    dagger_values = (
        args_cli.dagger_collect_steps,
        args_cli.dagger_updates_per_round,
        args_cli.dagger_student_capacity_sequences,
        args_cli.dagger_student_batch_sequences,
    )
    if args_cli.dagger_rounds > 0 and any(value <= 0 for value in dagger_values):
        raise ValueError("DAgger collection, update, capacity, and batch values must be positive.")
    if args_cli.dagger_rounds > 0 and args_cli.anchor_collect_steps <= 0:
        raise ValueError("Offline DAgger requires a non-empty teacher replay anchor.")
    if not 0 <= args_cli.replay_burn_in_steps < args_cli.anchor_sequence_length:
        raise ValueError("--replay_burn_in_steps must be in [0, anchor_sequence_length).")
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
    from Ranger.tasks.manager_based.ranger.agents import (
        RangerTeacherRegularizedPPO,
        RangerTerrainActorCritic,
        RangerTerrainActorCriticRecurrent,
    )
    from Ranger.tasks.manager_based.ranger.agents.rsl_rl_ppo_cfg import ShortGoalFlatV10PPORunnerCfg
    from warm_start import warm_start_ranger_recurrent_from_feedforward

    rsl_on_policy_runner.RangerTerrainActorCritic = RangerTerrainActorCritic
    rsl_on_policy_runner.RangerTerrainActorCriticRecurrent = RangerTerrainActorCriticRecurrent
    rsl_on_policy_runner.RangerTeacherRegularizedPPO = RangerTeacherRegularizedPPO

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
        memory_task = TEACHER_COMMAND_GROUP in obs.keys()
        print(f"[Distill] asymmetric teacher goal observations: {memory_task}")
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
        if args_cli.freeze_control_backbone or args_cli.freeze_memory_control_path:
            frozen_backbone_modules = ["map_encoder", "actor_trunk", "suspension_head", "wheel_head"]
            if args_cli.freeze_memory_control_path:
                frozen_backbone_modules.append("wheel_goal_residual")
            for module_name in frozen_backbone_modules:
                for parameter in getattr(student.actor, module_name).parameters():
                    parameter.requires_grad_(False)
            print(f"[Distill] frozen control backbone: {', '.join(frozen_backbone_modules)}")
        actor_parameters = [parameter for parameter in student.actor.parameters() if parameter.requires_grad]
        if not actor_parameters:
            raise RuntimeError("Student actor has no trainable parameters.")
        trainable_actor_names = [
            name for name, parameter in student.actor.named_parameters() if parameter.requires_grad
        ]
        print(f"[Distill] trainable actor parameters: {', '.join(trainable_actor_names)}")
        distillation_optimizer = torch.optim.Adam(actor_parameters, lr=float(args_cli.learning_rate))
        if student_checkpoint_path is not None:
            if args_cli.reset_distillation_optimizer:
                print("[Distill] Starting a new Adam optimizer as requested by --reset_distillation_optimizer.")
            else:
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

        teacher_anchor_mask: torch.Tensor | None = None
        if args_cli.rollout_mode == "dual_distribution":
            teacher_anchor_mask = _build_teacher_anchor_mask(
                int(args_cli.num_envs),
                float(args_cli.teacher_anchor_fraction),
                device=student_agent_cfg.device,
                seed=int(args_cli.seed),
            )
            if int(teacher_anchor_mask.sum().item()) == int(args_cli.num_envs):
                raise RuntimeError("dual_distribution requires at least one student-rollout environment.")

        anchor_replay: SequenceReplayAnchor | None = None
        if int(args_cli.anchor_collect_steps) > 0:
            policy_observation_keys = tuple(student_agent_cfg.obs_groups["policy"])
            anchor_replay = SequenceReplayAnchor(
                num_envs=int(args_cli.num_envs),
                sequence_length=int(args_cli.anchor_sequence_length),
                capacity_sequences=int(args_cli.anchor_capacity_sequences),
                observation_keys=policy_observation_keys,
                seed=int(args_cli.seed) + 10_000,
            )
            print(
                f"[ReplayAnchor] collecting {args_cli.anchor_collect_steps} teacher steps "
                f"as length-{args_cli.anchor_sequence_length} sequences"
            )
            for _ in range(int(args_cli.anchor_collect_steps)):
                stop_mask = _stop_phase_mask(env, int(args_cli.num_envs), student_agent_cfg.device)
                teacher_obs = build_teacher_observations(obs)
                with torch.inference_mode():
                    teacher_actions = bound_teacher_actions(teacher.act_inference(teacher_obs))
                # The fixed anchor remains fully visible even when it is collected
                # inside a memory task whose student observation is masked.
                anchor_obs = teacher_obs
                step_result = env.step(teacher_actions)
                obs, _, dones, _ = _parse_step_result(step_result)
                anchor_replay.append_step(
                    anchor_obs,
                    teacher_actions,
                    stop_mask,
                    dones,
                    goal_hidden_mask=torch.zeros_like(stop_mask),
                )
            if len(anchor_replay) == 0:
                raise RuntimeError(
                    "Replay collection produced no complete sequences; increase --anchor_collect_steps "
                    "or reduce --anchor_sequence_length."
                )
            obs = _unwrap_observations(env.reset())
            student.reset_memory()
            print(f"[ReplayAnchor] ready: {len(anchor_replay)} sequences")

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

        if int(args_cli.dagger_rounds) > 0:
            assert anchor_replay is not None
            student_replay = SequenceReplayAnchor(
                num_envs=int(args_cli.num_envs),
                sequence_length=int(args_cli.anchor_sequence_length),
                capacity_sequences=int(args_cli.dagger_student_capacity_sequences),
                observation_keys=tuple(student_agent_cfg.obs_groups["policy"]),
                seed=int(args_cli.seed) + 20_000,
            )
            dagger_metrics_path = output_dir / "dagger_metrics.csv"
            dagger_metric_names = (
                "round",
                "update",
                "student_replay_size",
                "collection_action_mse",
                "collection_action_cosine",
                "collection_wrong_turn_sign_rate",
                "collection_goal_hidden_rate",
                "student_replay_loss",
                "student_action_mse",
                "student_action_cosine",
                "student_wrong_turn_sign_rate",
                "student_goal_hidden_rate",
                "anchor_replay_loss",
                "anchor_action_mse",
                "anchor_action_cosine",
                "anchor_wrong_turn_sign_rate",
                "total_loss",
                "grad_norm",
            )
            dagger_file = dagger_metrics_path.open("w", newline="", encoding="utf-8")
            dagger_writer = csv.DictWriter(dagger_file, fieldnames=dagger_metric_names)
            dagger_writer.writeheader()
            burn_in = int(args_cli.replay_burn_in_steps)
            print(
                f"[DAgger] rounds={args_cli.dagger_rounds} collect_steps={args_cli.dagger_collect_steps} "
                f"updates_per_round={args_cli.dagger_updates_per_round} burn_in={burn_in}"
            )
            try:
                for dagger_round in range(1, int(args_cli.dagger_rounds) + 1):
                    obs = _unwrap_observations(env.reset())
                    student.reset_memory()
                    student_replay.reset_partial_sequences()
                    student.eval()
                    collection_sums = {
                        "action_mse_total": 0.0,
                        "action_cosine": 0.0,
                        "wrong_turn_sign_rate": 0.0,
                        "goal_hidden_rate": 0.0,
                    }
                    # IsaacLab mutates simulation state tensors during reset, so the environment
                    # step must not run under inference_mode (which creates immutable tensors).
                    with torch.no_grad():
                        for _ in range(int(args_cli.dagger_collect_steps)):
                            stop_mask = _stop_phase_mask(env, int(args_cli.num_envs), student_agent_cfg.device)
                            hidden_mask = goal_hidden_mask(obs)
                            teacher_obs = build_teacher_observations(obs)
                            teacher_actions = bound_teacher_actions(teacher.act_inference(teacher_obs))
                            student_actions = student.act_inference(obs)
                            collection_losses = compute_distillation_losses(
                                student_actions,
                                teacher_actions,
                                stop_mask,
                                suspension_loss_weight=float(args_cli.suspension_loss_weight),
                                wheel_loss_weight=float(args_cli.wheel_loss_weight),
                                wheel_common_loss_weight=float(args_cli.wheel_common_loss_weight),
                                wheel_turn_loss_weight=float(args_cli.wheel_turn_loss_weight),
                                stop_phase_wheel_weight=float(args_cli.stop_phase_wheel_weight),
                                turn_sign_threshold=float(args_cli.turn_sign_threshold),
                                feature_loss_weight=0.0,
                            )
                            for name in ("action_mse_total", "action_cosine", "wrong_turn_sign_rate"):
                                collection_sums[name] += float(collection_losses[name].item())
                            collection_sums["goal_hidden_rate"] += float(hidden_mask.float().mean().item())
                            collected_obs = obs
                            step_result = env.step(student_actions)
                            obs, _, dones, _ = _parse_step_result(step_result)
                            student_replay.append_step(
                                collected_obs,
                                teacher_actions,
                                stop_mask,
                                dones,
                                goal_hidden_mask=hidden_mask,
                            )
                            if torch.any(dones.to(dtype=torch.bool)):
                                student.reset(dones)

                    if len(student_replay) == 0:
                        raise RuntimeError("DAgger collection produced no complete student sequences.")
                    collection_averages = {
                        name: value / int(args_cli.dagger_collect_steps)
                        for name, value in collection_sums.items()
                    }
                    print(
                        f"[DAgger] round={dagger_round} collected={len(student_replay)} "
                        f"mse={collection_averages['action_mse_total']:.6f} "
                        f"cosine={collection_averages['action_cosine']:.4f} "
                        f"wrong_sign={collection_averages['wrong_turn_sign_rate']:.4f} "
                        f"hidden={collection_averages['goal_hidden_rate']:.4f}"
                    )

                    student.train()
                    for update in range(1, int(args_cli.dagger_updates_per_round) + 1):
                        student_batch = student_replay.sample(
                            int(args_cli.dagger_student_batch_sequences),
                            student_agent_cfg.device,
                            require_goal_transition=memory_task,
                            burn_in_steps=burn_in,
                        )
                        anchor_batch = anchor_replay.sample(
                            int(args_cli.anchor_batch_sequences),
                            student_agent_cfg.device,
                        )
                        student_sequence = recurrent_sequence_actions(student, student_batch)
                        anchor_sequence = recurrent_sequence_actions(student, anchor_batch)
                        student_hidden_mask = student_batch["goal_hidden_mask"][burn_in:]
                        student_sample_weights = goal_hidden_sample_weights(
                            student_hidden_mask,
                            float(args_cli.goal_hidden_loss_weight),
                        )
                        student_losses = compute_distillation_losses(
                            student_sequence[burn_in:],
                            student_batch["teacher_actions"][burn_in:],
                            student_batch["stop_mask"][burn_in:],
                            suspension_loss_weight=float(args_cli.suspension_loss_weight),
                            wheel_loss_weight=float(args_cli.wheel_loss_weight),
                            wheel_common_loss_weight=float(args_cli.wheel_common_loss_weight),
                            wheel_turn_loss_weight=float(args_cli.wheel_turn_loss_weight),
                            stop_phase_wheel_weight=float(args_cli.stop_phase_wheel_weight),
                            turn_sign_threshold=float(args_cli.turn_sign_threshold),
                            feature_loss_weight=0.0,
                            sample_weights=student_sample_weights,
                        )
                        anchor_losses = compute_distillation_losses(
                            anchor_sequence[burn_in:],
                            anchor_batch["teacher_actions"][burn_in:],
                            anchor_batch["stop_mask"][burn_in:],
                            suspension_loss_weight=float(args_cli.suspension_loss_weight),
                            wheel_loss_weight=float(args_cli.wheel_loss_weight),
                            wheel_common_loss_weight=float(args_cli.wheel_common_loss_weight),
                            wheel_turn_loss_weight=float(args_cli.wheel_turn_loss_weight),
                            stop_phase_wheel_weight=float(args_cli.stop_phase_wheel_weight),
                            turn_sign_threshold=float(args_cli.turn_sign_threshold),
                            feature_loss_weight=0.0,
                        )
                        total_loss = student_losses["loss"] + (
                            float(args_cli.anchor_loss_weight) * anchor_losses["loss"]
                        )
                        distillation_optimizer.zero_grad(set_to_none=True)
                        total_loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            actor_parameters,
                            float(args_cli.max_grad_norm),
                        )
                        distillation_optimizer.step()

                        dagger_writer.writerow(
                            {
                                "round": dagger_round,
                                "update": update,
                                "student_replay_size": len(student_replay),
                                "collection_action_mse": collection_averages["action_mse_total"],
                                "collection_action_cosine": collection_averages["action_cosine"],
                                "collection_wrong_turn_sign_rate": collection_averages["wrong_turn_sign_rate"],
                                "collection_goal_hidden_rate": collection_averages["goal_hidden_rate"],
                                "student_replay_loss": float(student_losses["loss"].detach().item()),
                                "student_action_mse": float(student_losses["action_mse_total"].detach().item()),
                                "student_action_cosine": float(student_losses["action_cosine"].detach().item()),
                                "student_wrong_turn_sign_rate": float(
                                    student_losses["wrong_turn_sign_rate"].detach().item()
                                ),
                                "student_goal_hidden_rate": float(student_hidden_mask.float().mean().item()),
                                "anchor_replay_loss": float(anchor_losses["loss"].detach().item()),
                                "anchor_action_mse": float(anchor_losses["action_mse_total"].detach().item()),
                                "anchor_action_cosine": float(anchor_losses["action_cosine"].detach().item()),
                                "anchor_wrong_turn_sign_rate": float(
                                    anchor_losses["wrong_turn_sign_rate"].detach().item()
                                ),
                                "total_loss": float(total_loss.detach().item()),
                                "grad_norm": float(grad_norm.detach().item()),
                            }
                        )
                        if update % int(args_cli.log_interval) == 0 or update == int(
                            args_cli.dagger_updates_per_round
                        ):
                            print(
                                f"[DAgger] round={dagger_round} update={update} "
                                f"student_loss={student_losses['loss'].item():.6f} "
                                f"anchor_loss={anchor_losses['loss'].item():.6f} "
                                f"student_cos={student_losses['action_cosine'].item():.4f} "
                                f"anchor_cos={anchor_losses['action_cosine'].item():.4f}"
                            )
                            dagger_file.flush()

                    _save_student_checkpoint(
                        output_dir / f"model_dagger_round_{dagger_round}.pt",
                        student_runner=student_runner,
                        distillation_optimizer=distillation_optimizer,
                        infos={
                            "dagger_round": dagger_round,
                            "dagger_student_replay_size": len(student_replay),
                            "teacher_checkpoint": str(teacher_checkpoint_path),
                            "anchor_loss_weight": float(args_cli.anchor_loss_weight),
                            "freeze_control_backbone": bool(args_cli.freeze_control_backbone),
                            "freeze_memory_control_path": bool(args_cli.freeze_memory_control_path),
                            "goal_hidden_loss_weight": float(args_cli.goal_hidden_loss_weight),
                        },
                    )

                _save_student_checkpoint(
                    final_checkpoint_path,
                    student_runner=student_runner,
                    distillation_optimizer=distillation_optimizer,
                    infos={
                        "dagger_rounds": int(args_cli.dagger_rounds),
                        "dagger_student_replay_size": len(student_replay),
                        "teacher_checkpoint": str(teacher_checkpoint_path),
                        "anchor_loss_weight": float(args_cli.anchor_loss_weight),
                        "freeze_control_backbone": bool(args_cli.freeze_control_backbone),
                        "freeze_memory_control_path": bool(args_cli.freeze_memory_control_path),
                        "goal_hidden_loss_weight": float(args_cli.goal_hidden_loss_weight),
                        "replay_burn_in_steps": burn_in,
                    },
                )
                print(f"[DAgger] saved recurrent student checkpoint: {final_checkpoint_path}")
                print(f"[DAgger] metrics: {dagger_metrics_path}")
            finally:
                dagger_file.close()
                teacher_hook.remove()
                student_hook.remove()
                env.close()
            return

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
            "wheel_common_mse",
            "wheel_turn_mse",
            "wrong_turn_sign_rate",
            "teacher_turn_abs_mean",
            "student_turn_abs_mean",
            "feature_mse",
            "action_cosine",
            "anchor_replay_loss",
            "anchor_replay_size",
            "total_optimization_loss",
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
        if args_cli.rollout_mode == "dual_distribution":
            metric_names.extend(
                [
                    "teacher_anchor_loss",
                    "student_rollout_loss",
                    "teacher_anchor_action_mse_total",
                    "teacher_anchor_wheel_mse_effective",
                    "teacher_anchor_action_cosine",
                    "teacher_anchor_stop_phase_rate",
                    "student_rollout_action_mse_total",
                    "student_rollout_wheel_mse_effective",
                    "student_rollout_action_cosine",
                    "student_rollout_stop_phase_rate",
                ]
            )
        metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metric_names)
        metrics_writer.writeheader()
        window_sums = {name: 0.0 for name in metric_names if name != "step"}
        window_count = 0

        print(f"[Distill] teacher: {teacher_checkpoint_path}")
        print(f"[Distill] rollout_mode: {args_cli.rollout_mode}")
        if args_cli.rollout_mode == "blended_rollout":
            print(f"[Distill] teacher_action_blend: {args_cli.teacher_action_blend}")
        elif args_cli.rollout_mode == "dual_distribution":
            assert teacher_anchor_mask is not None
            teacher_anchor_count = int(teacher_anchor_mask.sum().item())
            student_rollout_count = int(args_cli.num_envs) - teacher_anchor_count
            print(
                f"[Distill] dual_distribution: teacher_anchor={teacher_anchor_count} "
                f"student_rollout={student_rollout_count} fraction={args_cli.teacher_anchor_fraction:.4f} "
                f"anchor_loss_weight={args_cli.teacher_anchor_loss_weight:.4f} "
                f"student_loss_weight={1.0 - args_cli.teacher_anchor_loss_weight:.4f}"
            )
        print(f"[Distill] output_dir: {output_dir}")
        print(f"[Distill] feature_loss_weight: {args_cli.feature_loss_weight}")
        print(f"[Distill] goal_hidden_loss_weight: {args_cli.goal_hidden_loss_weight}")
        print(f"[Distill] stop_phase_wheel_weight: {args_cli.stop_phase_wheel_weight}")
        print(
            f"[Distill] semantic wheel weights: common={args_cli.wheel_common_loss_weight} "
            f"turn={args_cli.wheel_turn_loss_weight}"
        )
        print(
            f"[Distill] replay anchor: sequences={0 if anchor_replay is None else len(anchor_replay)} "
            f"batch={args_cli.anchor_batch_sequences} weight={args_cli.anchor_loss_weight}"
        )
        print(f"[Distill] bptt_steps: {args_cli.bptt_steps}")

        pending_loss: torch.Tensor | None = None
        bptt_count = 0
        distillation_optimizer.zero_grad(set_to_none=True)
        try:
            for step in range(1, int(args_cli.distill_steps) + 1):
                captured_features.clear()
                stop_mask = _stop_phase_mask(env, int(args_cli.num_envs), student_agent_cfg.device)
                hidden_mask = goal_hidden_mask(obs)
                teacher_obs = build_teacher_observations(obs)

                with torch.no_grad():
                    teacher_actions = bound_teacher_actions(teacher.act_inference(teacher_obs))
                    teacher_features = captured_features["teacher"].detach()

                student_actions = student.act_inference(obs)
                student_features = captured_features["student"]
                losses = compute_distillation_losses(
                    student_actions,
                    teacher_actions,
                    stop_mask,
                    suspension_loss_weight=float(args_cli.suspension_loss_weight),
                    wheel_loss_weight=float(args_cli.wheel_loss_weight),
                    wheel_common_loss_weight=float(args_cli.wheel_common_loss_weight),
                    wheel_turn_loss_weight=float(args_cli.wheel_turn_loss_weight),
                    stop_phase_wheel_weight=float(args_cli.stop_phase_wheel_weight),
                    turn_sign_threshold=float(args_cli.turn_sign_threshold),
                    student_features=student_features,
                    teacher_features=teacher_features,
                    feature_loss_weight=float(args_cli.feature_loss_weight),
                    sample_weights=goal_hidden_sample_weights(
                        hidden_mask,
                        float(args_cli.goal_hidden_loss_weight),
                    ),
                )

                dual_diagnostics: dict[str, float] = {}
                optimized_loss = losses["loss"]
                if args_cli.rollout_mode == "dual_distribution":
                    assert teacher_anchor_mask is not None
                    teacher_anchor_losses = _masked_distillation_losses(
                        student_actions,
                        teacher_actions,
                        stop_mask,
                        teacher_anchor_mask,
                        suspension_loss_weight=float(args_cli.suspension_loss_weight),
                        wheel_loss_weight=float(args_cli.wheel_loss_weight),
                        wheel_common_loss_weight=float(args_cli.wheel_common_loss_weight),
                        wheel_turn_loss_weight=float(args_cli.wheel_turn_loss_weight),
                        stop_phase_wheel_weight=float(args_cli.stop_phase_wheel_weight),
                        turn_sign_threshold=float(args_cli.turn_sign_threshold),
                        student_features=student_features,
                        teacher_features=teacher_features,
                        feature_loss_weight=float(args_cli.feature_loss_weight),
                    )
                    student_rollout_losses = _masked_distillation_losses(
                        student_actions,
                        teacher_actions,
                        stop_mask,
                        ~teacher_anchor_mask,
                        suspension_loss_weight=float(args_cli.suspension_loss_weight),
                        wheel_loss_weight=float(args_cli.wheel_loss_weight),
                        wheel_common_loss_weight=float(args_cli.wheel_common_loss_weight),
                        wheel_turn_loss_weight=float(args_cli.wheel_turn_loss_weight),
                        stop_phase_wheel_weight=float(args_cli.stop_phase_wheel_weight),
                        turn_sign_threshold=float(args_cli.turn_sign_threshold),
                        student_features=student_features,
                        teacher_features=teacher_features,
                        feature_loss_weight=float(args_cli.feature_loss_weight),
                    )
                    anchor_loss_weight = float(args_cli.teacher_anchor_loss_weight)
                    optimized_loss = (
                        anchor_loss_weight * teacher_anchor_losses["loss"]
                        + (1.0 - anchor_loss_weight) * student_rollout_losses["loss"]
                    )
                    dual_diagnostics = {
                        "teacher_anchor_loss": float(teacher_anchor_losses["loss"].detach().item()),
                        "student_rollout_loss": float(student_rollout_losses["loss"].detach().item()),
                        **{
                            f"teacher_anchor_{name}": float(teacher_anchor_losses[name].detach().item())
                            for name in (
                                "action_mse_total",
                                "wheel_mse_effective",
                                "action_cosine",
                                "stop_phase_rate",
                            )
                        },
                        **{
                            f"student_rollout_{name}": float(student_rollout_losses[name].detach().item())
                            for name in (
                                "action_mse_total",
                                "wheel_mse_effective",
                                "action_cosine",
                                "stop_phase_rate",
                            )
                        },
                    }

                losses["loss"] = optimized_loss
                pending_loss = optimized_loss if pending_loss is None else pending_loss + optimized_loss
                bptt_count += 1

                env_actions = _select_rollout_actions(
                    teacher_actions,
                    student_actions,
                    rollout_mode=args_cli.rollout_mode,
                    teacher_action_blend=float(args_cli.teacher_action_blend),
                    teacher_anchor_mask=teacher_anchor_mask,
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
                anchor_replay_loss_value = 0.0
                total_optimization_loss_value = float(optimized_loss.detach().item())
                module_grad_norms = {f"grad_norm_{module_name}": 0.0 for module_name in grad_module_names}
                if should_flush:
                    assert pending_loss is not None and bptt_count > 0
                    bptt_window_value = bptt_count
                    online_loss = pending_loss / bptt_count
                    total_optimization_loss = online_loss
                    if anchor_replay is not None and float(args_cli.anchor_loss_weight) > 0.0:
                        replay_batch = anchor_replay.sample(
                            int(args_cli.anchor_batch_sequences),
                            student_agent_cfg.device,
                        )
                        replay_student_actions = recurrent_sequence_actions(student, replay_batch)
                        replay_losses = compute_distillation_losses(
                            replay_student_actions,
                            replay_batch["teacher_actions"],
                            replay_batch["stop_mask"],
                            suspension_loss_weight=float(args_cli.suspension_loss_weight),
                            wheel_loss_weight=float(args_cli.wheel_loss_weight),
                            wheel_common_loss_weight=float(args_cli.wheel_common_loss_weight),
                            wheel_turn_loss_weight=float(args_cli.wheel_turn_loss_weight),
                            stop_phase_wheel_weight=float(args_cli.stop_phase_wheel_weight),
                            turn_sign_threshold=float(args_cli.turn_sign_threshold),
                            feature_loss_weight=0.0,
                        )
                        total_optimization_loss = total_optimization_loss + (
                            float(args_cli.anchor_loss_weight) * replay_losses["loss"]
                        )
                        anchor_replay_loss_value = float(replay_losses["loss"].detach().item())
                    total_optimization_loss_value = float(total_optimization_loss.detach().item())
                    total_optimization_loss.backward()
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
                    "wheel_common_mse": float(losses["wheel_common_mse"].detach().item()),
                    "wheel_turn_mse": float(losses["wheel_turn_mse"].detach().item()),
                    "wrong_turn_sign_rate": float(losses["wrong_turn_sign_rate"].detach().item()),
                    "teacher_turn_abs_mean": float(losses["teacher_turn_abs_mean"].detach().item()),
                    "student_turn_abs_mean": float(losses["student_turn_abs_mean"].detach().item()),
                    "feature_mse": float(losses["feature_mse"].detach().item()),
                    "action_cosine": float(losses["action_cosine"].detach().item()),
                    "anchor_replay_loss": anchor_replay_loss_value,
                    "anchor_replay_size": 0 if anchor_replay is None else len(anchor_replay),
                    "total_optimization_loss": total_optimization_loss_value,
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
                    **dual_diagnostics,
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
                        f"common={averages['wheel_common_mse']:.6f} turn={averages['wheel_turn_mse']:.6f} "
                        f"wrong_sign={averages['wrong_turn_sign_rate']:.4f} "
                        f"feature={averages['feature_mse']:.6f} cosine={averages['action_cosine']:.4f} "
                        f"hidden_max={averages['actor_hidden_abs_max']:.4f} reset_max={averages['hidden_reset_max_abs']:.2e}"
                    )
                    if args_cli.rollout_mode == "dual_distribution":
                        print(
                            "[Distill] dual "
                            f"anchor_mse={averages['teacher_anchor_action_mse_total']:.6f} "
                            f"anchor_eff_wheel={averages['teacher_anchor_wheel_mse_effective']:.6f} "
                            f"anchor_cos={averages['teacher_anchor_action_cosine']:.4f} "
                            f"student_mse={averages['student_rollout_action_mse_total']:.6f} "
                            f"student_eff_wheel={averages['student_rollout_wheel_mse_effective']:.6f} "
                            f"student_cos={averages['student_rollout_action_cosine']:.4f}"
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
                            "teacher_anchor_fraction": float(args_cli.teacher_anchor_fraction),
                            "teacher_anchor_loss_weight": float(args_cli.teacher_anchor_loss_weight),
                            "anchor_collect_steps": int(args_cli.anchor_collect_steps),
                            "anchor_sequence_length": int(args_cli.anchor_sequence_length),
                            "anchor_loss_weight": float(args_cli.anchor_loss_weight),
                            "reset_distillation_optimizer": bool(args_cli.reset_distillation_optimizer),
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
                    "teacher_anchor_fraction": float(args_cli.teacher_anchor_fraction),
                    "teacher_anchor_loss_weight": float(args_cli.teacher_anchor_loss_weight),
                    "reset_distillation_optimizer": bool(args_cli.reset_distillation_optimizer),
                    "teacher_checkpoint": str(teacher_checkpoint_path),
                    "student_init": args_cli.student_init,
                    "stop_phase_wheel_weight": float(args_cli.stop_phase_wheel_weight),
                    "feature_loss_weight": float(args_cli.feature_loss_weight),
                    "wheel_common_loss_weight": float(args_cli.wheel_common_loss_weight),
                    "wheel_turn_loss_weight": float(args_cli.wheel_turn_loss_weight),
                    "turn_sign_threshold": float(args_cli.turn_sign_threshold),
                    "anchor_collect_steps": int(args_cli.anchor_collect_steps),
                    "anchor_sequence_length": int(args_cli.anchor_sequence_length),
                    "anchor_capacity_sequences": int(args_cli.anchor_capacity_sequences),
                    "anchor_batch_sequences": int(args_cli.anchor_batch_sequences),
                    "anchor_loss_weight": float(args_cli.anchor_loss_weight),
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
