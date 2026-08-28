# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from export_iteration_metrics import export_tensorboard_scalars  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--disable_iteration_metrics_csv",
    action="store_true",
    default=False,
    help="Disable automatic TensorBoard scalar export to per-iteration CSV files.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--warm_start_checkpoint",
    type=str,
    default=None,
    help="Checkpoint used only to initialize actor parameters.",
)
parser.add_argument(
    "--teacher_checkpoint",
    type=str,
    default=None,
    help="Frozen V10 teacher checkpoint for Ranger Teacher-PPO. The teacher never controls rollout actions.",
)
parser.add_argument(
    "--teacher_loss_coef",
    type=float,
    default=None,
    help="Override Teacher-PPO action regularization coefficient. Leave unset to use the task config.",
)
parser.add_argument(
    "--latent_loss_coef",
    type=float,
    default=None,
    help="Latent state-encoder alignment coefficient. This is independent from teacher action loss.",
)
parser.add_argument(
    "--teacher_ppo_actor_loss_scale",
    type=float,
    default=None,
    help="Scale actor-side PPO/entropy/teacher losses while leaving critic value loss unchanged.",
)
parser.add_argument(
    "--teacher_ppo_surrogate_scale",
    type=float,
    default=None,
    help="Scale only the PPO surrogate term; set to zero for auxiliary-only actor repair.",
)
parser.add_argument(
    "--teacher_ppo_actor_learning_rate",
    type=float,
    default=None,
    help="Optional split-optimizer actor/log-std learning rate; critic keeps the PPO learning rate.",
)
parser.add_argument(
    "--teacher_ppo_actor_train_scope",
    choices=("all", "wheel_head", "wheel_residual"),
    default=None,
    help="Optionally restrict actor optimization to the wheel action head during local repair.",
)
parser.add_argument(
    "--teacher_ppo_large_heading_action_prior_coef",
    type=float,
    default=None,
    help="Student-only large-heading wheel-action auxiliary coefficient; zero disables the repair prior.",
)
parser.add_argument(
    "--teacher_ppo_lam",
    type=float,
    default=None,
    help="Optional Teacher-PPO GAE lambda override. Used to probe longer-horizon actor credit assignment.",
)
parser.add_argument(
    "--model_only_resume",
    action="store_true",
    default=False,
    help="Load only model_state_dict from the resume checkpoint, leaving the newly configured optimizer intact.",
)
parser.add_argument(
    "--partial_transfer",
    action="store_true",
    default=False,
    help=(
        "Load only selected compatible actor modules from --checkpoint into the current policy. "
        "Optimizer and runner state are intentionally not restored."
    ),
)
parser.add_argument(
    "--aligned_latent_old_trunk",
    action="store_true",
    default=False,
    help=(
        "Experiment 1-E: bypass the recurrent actor GRU and drive the V10 actor trunk/heads from "
        "map latent plus aligned state latent."
    ),
)
parser.add_argument(
    "--freeze_old_actor_trunk",
    action="store_true",
    default=False,
    help="Experiment 1-E first-stage mode: freeze the transferred old actor trunk and train only latent projection.",
)
parser.add_argument(
    "--gru_residual",
    action="store_true",
    default=False,
    help="Stage 1: keep the aligned-latent old actor path as base action and train a zero-initialized GRU residual.",
)
parser.add_argument(
    "--residual_action_scale",
    type=float,
    default=None,
    help="Stage1A residual action scale used in final_action = base_action + scale * delta_action.",
)
parser.add_argument(
    "--residual_loss_coef",
    type=float,
    default=None,
    help="Stage1A coefficient for mean(delta_action^2).",
)
parser.add_argument(
    "--anchor_loss_coef",
    type=float,
    default=None,
    help="Stage1A coefficient for mean((final_action - base_action)^2).",
)
parser.add_argument(
    "--teacher_ppo_critic_only",
    action="store_true",
    default=False,
    help="Teacher-PPO TP0 mode: update only the recurrent critic and leave actor/std unchanged.",
)
parser.add_argument(
    "--teacher_ppo_diagnostic_only",
    action="store_true",
    default=False,
    help=(
        "Teacher-PPO diagnostic mode: compute PPO/teacher gradient diagnostics and advantage bins "
        "without applying optimizer updates."
    ),
)
parser.add_argument(
    "--teacher_ppo_diagnostic_rollout_steps",
    type=int,
    default=None,
    help="Override num_steps_per_env only for Teacher-PPO diagnostic-only runs.",
)
parser.add_argument(
    "--teacher_ppo_joint_probe",
    action="store_true",
    default=False,
    help=(
        "Conservative continuation mode for J-series joint actor-critic probes: 192-step rollout, "
        "1 epoch, 4 minibatches, clip=0.1, fixed schedule, learning_rate=1e-6."
    ),
)
parser.add_argument(
    "--teacher_ppo_critic_probe",
    action="store_true",
    default=False,
    help=(
        "Conservative critic-only continuation mode: 192-step rollout, 1 epoch, 4 minibatches, "
        "clip=0.1, fixed schedule, learning_rate=1e-6, save_interval=1."
    ),
)
parser.add_argument(
    "--teacher_ppo_long_return_critic_probe",
    action="store_true",
    default=False,
    help=(
        "Conservative actor-frozen critic continuation using lambda=1.0 long-return targets: "
        "192-step rollout, 1 epoch, 4 minibatches, clip=0.1, fixed schedule, learning_rate=1e-6."
    ),
)
parser.add_argument(
    "--critic_relearning_actor_checkpoint",
    type=str,
    default=None,
    help="Bounded recurrent actor checkpoint used only to initialize critic-relearning runs.",
)
parser.add_argument(
    "--critic_relearning_v10_checkpoint",
    type=str,
    default=None,
    help="Feedforward V10 checkpoint providing critic map/privileged representation weights.",
)
parser.add_argument(
    "--critic_relearning_rollout_steps",
    type=int,
    default=192,
    help="Rollout length for critic relearning. Used with lambda=1 long-return targets.",
)
parser.add_argument(
    "--critic_relearning_perturb_burst",
    type=int,
    default=0,
    help=(
        "Number of steps per perturbation period that use local wheel-action perturbations during critic relearning. "
        "Default 0 keeps C2-A on clean bounded-R6 rollouts; use a positive value only for the later coverage phase."
    ),
)
parser.add_argument(
    "--critic_relearning_learning_rate",
    type=float,
    default=None,
    help="Optional critic-only learning-rate override for critic relearning experiments.",
)
parser.add_argument(
    "--critic_relearning_joint_actor",
    action="store_true",
    default=False,
    help=(
        "Initialize from the same clean critic-relearning sources but keep actor/log-std trainable for a "
        "conservative one-iteration joint actor-critic probe."
    ),
)
parser.add_argument(
    "--warm_start_mode",
    type=str,
    choices=(
        "actor_suspension",
        "actor_wheel",
        "actor_suspension_only",
        "actor_wheel_reset_suspension",
        "actor_wheel_reset_final",
        "actor_reset_heads",
        "actor_heads",
        "actor_full",
        "recurrent_v10_actor",
    ),
    default=None,
    help=(
        "Actor warm-start mode. "
        "'actor_suspension' loads the shared encoders, actor trunk, "
        "and suspension head while keeping the wheel head random. "
        "'actor_wheel' loads the complete actor and trains only the wheel head. "
        "'actor_suspension_only' loads the complete actor and trains only the suspension head. "
        "'actor_wheel_reset_suspension' loads the shared actor and wheel head, resets the suspension head, "
        "and trains only the suspension head. "
        "'actor_wheel_reset_final' loads the complete actor, resets only the wheel head's final Linear, "
        "and trains only that final Linear. "
        "'actor_reset_heads' loads only the shared encoders/trunk, resets both action heads, "
        "and trains both action heads. "
        "'actor_heads' loads the complete actor and trains both action heads while freezing encoders/trunk. "
        "'actor_full' loads the complete actor. "
        "'recurrent_v10_actor' is recurrent-only and copies only the V10-compatible map encoder, actor trunk, "
        "suspension head, and wheel head; recurrent representation/critic/std stay newly initialized."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.resume and args_cli.warm_start_checkpoint is not None:
    raise ValueError("--resume and --warm_start_checkpoint cannot be used together.")
if args_cli.warm_start_checkpoint is not None and args_cli.warm_start_mode is None:
    raise ValueError("--warm_start_mode is required when --warm_start_checkpoint is provided.")
if args_cli.teacher_loss_coef is not None and args_cli.teacher_loss_coef < 0.0:
    raise ValueError("--teacher_loss_coef must be non-negative.")
if args_cli.latent_loss_coef is not None and args_cli.latent_loss_coef < 0.0:
    raise ValueError("--latent_loss_coef must be non-negative.")
if args_cli.teacher_ppo_actor_loss_scale is not None and args_cli.teacher_ppo_actor_loss_scale <= 0.0:
    raise ValueError("--teacher_ppo_actor_loss_scale must be positive.")
if args_cli.teacher_ppo_surrogate_scale is not None and args_cli.teacher_ppo_surrogate_scale < 0.0:
    raise ValueError("--teacher_ppo_surrogate_scale must be non-negative.")
if args_cli.teacher_ppo_actor_learning_rate is not None and args_cli.teacher_ppo_actor_learning_rate <= 0.0:
    raise ValueError("--teacher_ppo_actor_learning_rate must be positive.")
if args_cli.teacher_ppo_actor_train_scope is not None and args_cli.teacher_ppo_actor_learning_rate is None:
    raise ValueError("--teacher_ppo_actor_train_scope requires --teacher_ppo_actor_learning_rate.")
if args_cli.teacher_ppo_large_heading_action_prior_coef is not None and args_cli.teacher_ppo_large_heading_action_prior_coef < 0.0:
    raise ValueError("--teacher_ppo_large_heading_action_prior_coef must be non-negative.")
if args_cli.teacher_ppo_lam is not None and not (0.0 <= args_cli.teacher_ppo_lam <= 1.0):
    raise ValueError("--teacher_ppo_lam must be in [0, 1].")
if args_cli.model_only_resume and not args_cli.resume:
    raise ValueError("--model_only_resume requires --resume.")
if args_cli.partial_transfer:
    if args_cli.resume:
        raise ValueError("--partial_transfer is a fresh initialization path and cannot be combined with --resume.")
    if args_cli.warm_start_checkpoint is not None:
        raise ValueError("--partial_transfer cannot be combined with --warm_start_checkpoint.")
    if args_cli.checkpoint is None:
        raise ValueError("--partial_transfer requires --checkpoint.")
if args_cli.freeze_old_actor_trunk and not args_cli.aligned_latent_old_trunk:
    raise ValueError("--freeze_old_actor_trunk requires --aligned_latent_old_trunk.")
if args_cli.gru_residual and not args_cli.aligned_latent_old_trunk:
    raise ValueError("--gru_residual requires --aligned_latent_old_trunk.")
if args_cli.residual_action_scale is not None and args_cli.residual_action_scale < 0.0:
    raise ValueError("--residual_action_scale must be non-negative.")
if args_cli.residual_loss_coef is not None and args_cli.residual_loss_coef < 0.0:
    raise ValueError("--residual_loss_coef must be non-negative.")
if args_cli.anchor_loss_coef is not None and args_cli.anchor_loss_coef < 0.0:
    raise ValueError("--anchor_loss_coef must be non-negative.")
if args_cli.teacher_ppo_diagnostic_rollout_steps is not None:
    if not args_cli.teacher_ppo_diagnostic_only:
        raise ValueError("--teacher_ppo_diagnostic_rollout_steps requires --teacher_ppo_diagnostic_only.")
    if args_cli.teacher_ppo_diagnostic_rollout_steps < 2:
        raise ValueError("--teacher_ppo_diagnostic_rollout_steps must be at least 2.")
critic_relearning_requested = (
    args_cli.critic_relearning_actor_checkpoint is not None
    or args_cli.critic_relearning_v10_checkpoint is not None
)
if critic_relearning_requested:
    if args_cli.critic_relearning_actor_checkpoint is None or args_cli.critic_relearning_v10_checkpoint is None:
        raise ValueError(
            "critic relearning requires both --critic_relearning_actor_checkpoint and "
            "--critic_relearning_v10_checkpoint."
        )
    if args_cli.resume or args_cli.warm_start_checkpoint is not None:
        raise ValueError("critic relearning uses a clean initialization and cannot be combined with --resume/warm-start.")
    if args_cli.critic_relearning_rollout_steps < 32:
        raise ValueError("--critic_relearning_rollout_steps must be at least 32.")
    if args_cli.critic_relearning_perturb_burst < 0:
        raise ValueError("--critic_relearning_perturb_burst must be non-negative.")
    if args_cli.critic_relearning_learning_rate is not None and args_cli.critic_relearning_learning_rate <= 0.0:
        raise ValueError("--critic_relearning_learning_rate must be positive when provided.")
elif args_cli.critic_relearning_joint_actor:
    raise ValueError("--critic_relearning_joint_actor requires the critic-relearning actor and V10 checkpoints.")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

try:
    from isaaclab.utils.io import dump_pickle
except ImportError:
    try:
        import cloudpickle as pickle
    except ImportError:
        import pickle

    def dump_pickle(filename, data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Ranger.tasks  # noqa: F401
from Ranger.tasks.manager_based.ranger.agents import (
    RangerTeacherRegularizedPPO,
    RangerTerrainActorCritic,
    RangerTerrainActorCriticRecurrent,
)
from Ranger.tasks.manager_based.ranger.agents.rsl_rl_ppo_cfg import ShortGoalFlatV10PPORunnerCfg
from warm_start import (
    warm_start_ranger_actor,
    warm_start_ranger_recurrent_critic_relearning,
    warm_start_ranger_recurrent_from_feedforward,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

rsl_on_policy_runner.RangerTerrainActorCritic = RangerTerrainActorCritic
rsl_on_policy_runner.RangerTerrainActorCriticRecurrent = RangerTerrainActorCriticRecurrent
rsl_on_policy_runner.RangerTeacherRegularizedPPO = RangerTeacherRegularizedPPO


def _resolve_resume_path(log_root_path: str, load_run: str, load_checkpoint: str) -> str:
    """Resolve a checkpoint path from either an explicit file path or the standard run/checkpoint selectors."""

    expanded_checkpoint = os.path.abspath(os.path.expanduser(load_checkpoint))
    if os.path.isfile(expanded_checkpoint):
        return expanded_checkpoint
    return get_checkpoint_path(log_root_path, load_run, load_checkpoint)


def _reset_action_std_from_env(runner: OnPolicyRunner) -> None:
    """Optionally reset policy exploration std after checkpoint loading."""

    reset_std = os.environ.get("RANGER_RESET_ACTION_STD")
    if reset_std is None:
        return

    try:
        reset_std_value = float(reset_std)
    except ValueError as exc:
        raise ValueError(f"RANGER_RESET_ACTION_STD must be a positive float, got: {reset_std!r}") from exc
    if reset_std_value <= 0.0:
        raise ValueError(f"RANGER_RESET_ACTION_STD must be positive, got: {reset_std_value}")

    actor_critic = runner.alg.policy
    with torch.no_grad():
        if hasattr(actor_critic, "std"):
            actor_critic.std.fill_(reset_std_value)
        elif hasattr(actor_critic, "log_std"):
            log_std_value = torch.log(
                torch.tensor(reset_std_value, device=actor_critic.log_std.device, dtype=actor_critic.log_std.dtype)
            )
            actor_critic.log_std.fill_(log_std_value)
        else:
            print("[WARN] RANGER_RESET_ACTION_STD was set, but actor_critic has no std/log_std attribute.")
            return

    print(f"[INFO] Reset action std to {reset_std_value} from RANGER_RESET_ACTION_STD.")


def _is_state_key_under_module(key: str, modules: tuple[str, ...]) -> bool:
    return any(key == module or key.startswith(f"{module}.") for module in modules)


def transfer_state_encoder_to_split_encoders(
    old_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Split legacy actor.state_encoder weights into recurrent prop/goal encoders."""

    load_state: dict[str, torch.Tensor] = {}
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    prop_indices = list(range(0, 26)) + list(range(34, 42))
    goal_indices = list(range(26, 34))
    mappings = {
        "actor.prop_encoder.0.weight": ("actor.state_encoder.0.weight", prop_indices),
        "actor.goal_encoder.0.weight": ("actor.state_encoder.0.weight", goal_indices),
        "actor.prop_encoder.0.bias": ("actor.state_encoder.0.bias", None),
        "actor.goal_encoder.0.bias": ("actor.state_encoder.0.bias", None),
        "actor.prop_encoder.2.weight": ("actor.state_encoder.2.weight", None),
        "actor.goal_encoder.2.weight": ("actor.state_encoder.2.weight", None),
        "actor.prop_encoder.2.bias": ("actor.state_encoder.2.bias", None),
        "actor.goal_encoder.2.bias": ("actor.state_encoder.2.bias", None),
    }

    for new_key, (old_key, indices) in mappings.items():
        old_tensor = old_state.get(old_key)
        new_tensor = current_state.get(new_key)
        if old_tensor is None:
            skipped_keys.append(f"{new_key} <- {old_key} | missing in checkpoint")
            continue
        if new_tensor is None:
            skipped_keys.append(f"{new_key} <- {old_key} | missing in current policy")
            continue
        source_tensor = old_tensor[:, indices] if indices is not None else old_tensor
        if tuple(source_tensor.shape) != tuple(new_tensor.shape):
            skipped_keys.append(
                f"{new_key} <- {old_key} | shape mismatch source={tuple(source_tensor.shape)} "
                f"current={tuple(new_tensor.shape)}"
            )
            continue
        load_state[new_key] = source_tensor.to(dtype=new_tensor.dtype)
        loaded_keys.append(new_key)

    return load_state, loaded_keys, skipped_keys


def load_partial_transfer_state_dict(
    policy: torch.nn.Module,
    checkpoint_path: str | Path,
    aligned_latent_old_trunk: bool = False,
    freeze_old_actor_trunk: bool = False,
    gru_residual: bool = False,
    transfer_modules: tuple[str, ...] = (
        "actor.map_encoder",
        "actor.prop_encoder",
        "actor.goal_encoder",
        "actor.wheel_head",
        "actor.suspension_head",
    ),
    optional_transfer_modules: tuple[str, ...] = (),
) -> None:
    """Load a whitelist of compatible checkpoint tensors into the current policy.

    This path intentionally does not restore optimizer state, runner iteration,
    observation normalizers, action std, or any recurrent-only modules.
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError(f"Partial transfer checkpoint must contain model_state_dict: {checkpoint_path}")
    old_state = checkpoint["model_state_dict"]
    if not isinstance(old_state, dict):
        raise TypeError(f"model_state_dict must be a dict, got {type(old_state).__name__}")

    current_state = policy.state_dict()
    if aligned_latent_old_trunk and "actor.actor_trunk" not in transfer_modules:
        transfer_modules = transfer_modules + ("actor.actor_trunk",)
    candidate_modules = transfer_modules + optional_transfer_modules
    load_state, loaded_keys, skipped_keys = transfer_state_encoder_to_split_encoders(old_state, current_state)

    for key in sorted(current_state):
        if key in load_state:
            continue
        if key.startswith("actor.prop_encoder.") or key.startswith("actor.goal_encoder."):
            continue
        if not _is_state_key_under_module(key, candidate_modules):
            continue
        old_tensor = old_state.get(key)
        if old_tensor is None:
            skipped_keys.append(f"{key} | missing in checkpoint")
            continue
        if tuple(old_tensor.shape) != tuple(current_state[key].shape):
            skipped_keys.append(
                f"{key} | shape mismatch checkpoint={tuple(old_tensor.shape)} current={tuple(current_state[key].shape)}"
            )
            continue
        load_state[key] = old_tensor.to(dtype=current_state[key].dtype)
        loaded_keys.append(key)

    if not loaded_keys:
        raise RuntimeError(
            "Partial transfer found no compatible tensors. "
            f"checkpoint={checkpoint_path} modules={candidate_modules}"
        )

    load_result = policy.load_state_dict(load_state, strict=False)

    frozen_transfer_modules = (
        "actor.map_encoder",
        "actor.prop_encoder",
        "actor.goal_encoder",
        "actor.wheel_head",
        "actor.suspension_head",
    )
    if gru_residual:
        frozen_transfer_modules = frozen_transfer_modules + (
            "actor.latent_alignment_projection",
            "actor.actor_trunk",
        )
    if gru_residual:
        trainable_actor_modules = (
            "actor.residual_memory",
            "actor.residual_hidden_norm",
            "actor.residual_head",
        )
    elif aligned_latent_old_trunk:
        trainable_actor_modules = ("actor.latent_alignment_projection",)
        if not freeze_old_actor_trunk:
            trainable_actor_modules = trainable_actor_modules + ("actor.actor_trunk",)
    else:
        trainable_actor_modules = (
            "actor.fusion_norm",
            "actor.latent_alignment_projection",
            "actor.memory",
            "actor.hidden_norm",
            "actor.actor_trunk",
        )
    for name, parameter in policy.named_parameters():
        if name.startswith("actor."):
            parameter.requires_grad = any(
                name == module or name.startswith(f"{module}.") for module in trainable_actor_modules
            )

    unexpected = getattr(load_result, "unexpected_keys", ())
    unexpected_loaded = [key for key in unexpected if key in load_state]
    if unexpected_loaded:
        raise RuntimeError(f"Partial transfer produced unexpected loaded keys: {unexpected_loaded}")

    loaded_modules = sorted({key.rsplit(".", 1)[0] for key in loaded_keys})
    core_module_loaded = {
        module: any(_is_state_key_under_module(key, (module,)) for key in loaded_keys)
        for module in transfer_modules
    }
    missing_core_modules = [module for module, loaded in core_module_loaded.items() if not loaded]
    if missing_core_modules:
        raise RuntimeError(f"Partial transfer failed to load required modules: {missing_core_modules}")

    random_initialized_modules = (
        "actor.memory",
        "actor.residual_memory",
        "actor.residual_hidden_norm",
        "actor.residual_head",
        "actor.fusion_norm",
        "actor.hidden_norm",
        "actor.latent_alignment_projection",
    )
    print(f"[PartialTransfer] source: {checkpoint_path}")
    print("[PartialTransfer] Loaded transfer modules:")
    loaded_root_modules = sorted(
        module for module in transfer_modules if any(_is_state_key_under_module(key, (module,)) for key in loaded_keys)
    )
    for module in loaded_root_modules:
        print(f"  - {module}")
    print(f"[PartialTransfer] loaded tensors: {len(loaded_keys)}")
    for key in loaded_keys:
        print(f"[PartialTransfer] loaded: {key} shape={tuple(current_state[key].shape)}")
    print("[PartialTransfer] Frozen:")
    for module in frozen_transfer_modules:
        print(f"  - {module}")
    print("[PartialTransfer] Trainable:")
    for module in trainable_actor_modules:
        print(f"  - {module}")
    if aligned_latent_old_trunk:
        if gru_residual:
            print("[PartialTransfer] actor forward: old trunk/head base action + GRU residual delta")
        else:
            print("[PartialTransfer] actor forward: aligned latent + old trunk/head bypass; actor.memory/GRU skipped")
        print(f"[PartialTransfer] old actor_trunk frozen: {bool(freeze_old_actor_trunk or gru_residual)}")
        if gru_residual:
            residual_scale = float(getattr(policy.actor, "residual_action_scale", 0.0))
            print("[PartialTransfer] Stage1A freeze: encoders, latent projection, old trunk, and old heads")
            print("[PartialTransfer] Stage1A train: actor.residual_memory, actor.residual_hidden_norm, actor.residual_head")
            print(f"[PartialTransfer] Stage1A residual_action_scale: {residual_scale:g}")
        trunk_weight = current_state.get("actor.actor_trunk.0.weight")
        projection_weight = current_state.get("actor.latent_alignment_projection.weight")
        residual_weight = current_state.get("actor.residual_head.weight")
        if trunk_weight is not None:
            print(f"[PartialTransfer] old actor_trunk input dim: {tuple(trunk_weight.shape)}")
        if projection_weight is not None:
            print(f"[PartialTransfer] student latent projection shape: {tuple(projection_weight.shape)}")
        if residual_weight is not None:
            print(f"[PartialTransfer] residual head shape: {tuple(residual_weight.shape)}")
    print("[PartialTransfer] Random initialized:")
    for module in random_initialized_modules:
        print(f"  - {module}")
    if skipped_keys:
        print(f"[PartialTransfer] skipped tensors: {len(skipped_keys)}")
        for skipped in skipped_keys[:80]:
            print(f"[PartialTransfer] skipped: {skipped}")
        if len(skipped_keys) > 80:
            print(f"[PartialTransfer] skipped: ... {len(skipped_keys) - 80} more")
    print("[PartialTransfer] optimizer: fresh")
    print("[PartialTransfer] iteration: 0")


def _configure_teacher_regularized_ppo(runner: OnPolicyRunner) -> None:
    """Attach the frozen V10 teacher only when Teacher-PPO regularization is active."""

    algorithm = runner.alg
    if not isinstance(algorithm, RangerTeacherRegularizedPPO):
        return
    if algorithm.critic_only:
        if algorithm.teacher_loss_coef > 0.0:
            raise ValueError("Teacher-PPO critic_only mode requires teacher_loss_coef=0.")
        print("[TeacherPPO] critic_only=True: actor/std are unchanged and no teacher is instantiated.")
        return
    latent_loss_coef = float(getattr(algorithm, "latent_loss_coef", 0.0))
    if algorithm.teacher_loss_coef <= 0.0 and latent_loss_coef <= 0.0:
        print("[TeacherPPO] teacher_loss_coef=0 and latent_loss_coef=0: running pure recurrent PPO.")
        return
    if algorithm.teacher_checkpoint is None:
        raise ValueError("Teacher-PPO teacher/latent loss requires --teacher_checkpoint or --checkpoint.")

    teacher_checkpoint_path = Path(algorithm.teacher_checkpoint).expanduser().resolve()
    if not teacher_checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")

    checkpoint = torch.load(teacher_checkpoint_path, map_location=runner.device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError("Teacher checkpoint must contain model_state_dict.")

    if algorithm.teacher_loss_coef > 0.0:
        # Use the same recurrent architecture as the student for action regularization.
        # Latent-only alignment below uses the legacy V10 feed-forward teacher instead.
        import copy
        teacher = copy.deepcopy(runner.alg.policy).to(runner.device)
        teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        algorithm.configure_teacher(teacher)
        print(
            "[TeacherPPO] Frozen recurrent teacher configured: "
            f"checkpoint={teacher_checkpoint_path} coef={algorithm.teacher_loss_coef:.6g} "
            f"stop_phase_wheel_weight={algorithm.teacher_stop_phase_wheel_weight:.3f}"
        )
    else:
        print("[TeacherPPO] teacher_loss_coef=0: action teacher disabled.")

    if latent_loss_coef > 0.0:
        obs = runner.env.get_observations()
        teacher_runner_cfg = ShortGoalFlatV10PPORunnerCfg()
        teacher_policy_kwargs = teacher_runner_cfg.policy.to_dict()
        teacher_policy_kwargs.pop("class_name", None)
        latent_teacher = RangerTerrainActorCritic(
            obs=obs,
            obs_groups=runner.cfg["obs_groups"],
            num_actions=int(runner.env.num_actions),
            **teacher_policy_kwargs,
        ).to(runner.device)
        latent_teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
        algorithm.configure_latent_teacher(latent_teacher)
        state_group = getattr(runner.alg.policy, "state_obs_group", "policy_state")
        with torch.no_grad():
            student_latent = runner.alg.policy.alignment_latent(obs)
            teacher_latent = latent_teacher.actor.state_encoder(obs[state_group])
            latent_smoke_loss = torch.nn.functional.mse_loss(student_latent, teacher_latent)
        print(
            "[LatentAlignment] Frozen V10 teacher state_encoder configured: "
            f"checkpoint={teacher_checkpoint_path} latent_loss_coef={algorithm.latent_loss_coef:g}"
        )
        print(
            "[LatentAlignment] smoke: "
            f"student_latent_shape={tuple(student_latent.shape)} "
            f"teacher_latent_shape={tuple(teacher_latent.shape)} "
            f"mse={float(latent_smoke_loss.item()):.6g}"
        )


def _print_gru_residual_smoke(runner: OnPolicyRunner) -> None:
    """Print initial residual-action statistics without changing the rollout start state."""

    policy = runner.alg.policy
    if not bool(getattr(policy, "use_gru_residual", False)):
        return
    obs = runner.env.get_observations()
    with torch.no_grad():
        base_action = policy.base_action(obs)
        delta_action = policy.residual_delta(obs)
        actor_obs = policy.actor_obs_normalizer(policy.get_actor_obs(obs))
        final_action = policy.actor(actor_obs)
        residual_error = final_action - base_action
    policy.reset_memory()
    residual_scale = float(getattr(policy.actor, "residual_action_scale", 0.0))
    print(
        "[GRUResidual] delta_action stats: "
        f"scale={residual_scale:g} "
        f"mean={float(delta_action.mean().item()):.6g} "
        f"abs_mean={float(delta_action.abs().mean().item()):.6g} "
        f"abs_max={float(delta_action.abs().max().item()):.6g}"
    )
    print(
        "[GRUResidual] initial behavior check: "
        f"final_minus_base_abs_max={float(residual_error.abs().max().item()):.6g} "
        f"base_shape={tuple(base_action.shape)} final_shape={tuple(final_action.shape)}"
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    teacher_ppo_selected = getattr(agent_cfg.algorithm, "class_name", "") == "RangerTeacherRegularizedPPO"
    teacher_ppo_args_used = (
        args_cli.teacher_checkpoint is not None
        or args_cli.teacher_loss_coef is not None
        or args_cli.latent_loss_coef is not None
        or args_cli.residual_loss_coef is not None
        or args_cli.anchor_loss_coef is not None
        or args_cli.teacher_ppo_actor_loss_scale is not None
        or args_cli.teacher_ppo_surrogate_scale is not None
        or args_cli.teacher_ppo_actor_learning_rate is not None
        or args_cli.teacher_ppo_actor_train_scope is not None
        or args_cli.teacher_ppo_large_heading_action_prior_coef is not None
        or args_cli.teacher_ppo_lam is not None
        or args_cli.teacher_ppo_critic_only
        or args_cli.teacher_ppo_diagnostic_only
        or args_cli.teacher_ppo_joint_probe
        or args_cli.teacher_ppo_critic_probe
        or args_cli.teacher_ppo_long_return_critic_probe
        or critic_relearning_requested
        or args_cli.aligned_latent_old_trunk
        or args_cli.gru_residual
    )
    if teacher_ppo_args_used and not teacher_ppo_selected:
        raise ValueError(
            "Teacher-PPO CLI options require a RangerTeacherRegularizedPPO task."
        )
    if teacher_ppo_selected:
        if args_cli.aligned_latent_old_trunk:
            agent_cfg.policy.use_recurrent_actor = False
            print("[Exp1E] actor forward: use_recurrent_actor=False")
        if args_cli.gru_residual:
            agent_cfg.policy.use_gru_residual = True
            print("[Stage1A] residual-constrained GRU branch enabled.")
        if args_cli.residual_action_scale is not None:
            agent_cfg.policy.residual_action_scale = float(args_cli.residual_action_scale)
        if args_cli.teacher_checkpoint is not None:
            agent_cfg.algorithm.teacher_checkpoint = str(Path(args_cli.teacher_checkpoint).expanduser().resolve())
        elif args_cli.latent_loss_coef is not None and args_cli.latent_loss_coef > 0.0 and args_cli.checkpoint is not None:
            agent_cfg.algorithm.teacher_checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
        if args_cli.teacher_loss_coef is not None:
            agent_cfg.algorithm.teacher_loss_coef = float(args_cli.teacher_loss_coef)
        if args_cli.latent_loss_coef is not None:
            agent_cfg.algorithm.latent_loss_coef = float(args_cli.latent_loss_coef)
            print(f"[LatentAlignment] latent loss coefficient: {agent_cfg.algorithm.latent_loss_coef:g}")
        if args_cli.residual_loss_coef is not None:
            agent_cfg.algorithm.residual_loss_coef = float(args_cli.residual_loss_coef)
        if args_cli.anchor_loss_coef is not None:
            agent_cfg.algorithm.anchor_loss_coef = float(args_cli.anchor_loss_coef)
        if args_cli.gru_residual:
            print(
                "[Stage1A] residual constraints: "
                f"residual_action_scale={agent_cfg.policy.residual_action_scale:g} "
                f"residual_loss_coef={agent_cfg.algorithm.residual_loss_coef:g} "
                f"anchor_loss_coef={agent_cfg.algorithm.anchor_loss_coef:g}"
            )
        if args_cli.teacher_ppo_actor_loss_scale is not None:
            agent_cfg.algorithm.actor_loss_scale = float(args_cli.teacher_ppo_actor_loss_scale)
        if args_cli.teacher_ppo_surrogate_scale is not None:
            agent_cfg.algorithm.ppo_surrogate_scale = float(args_cli.teacher_ppo_surrogate_scale)
            print(f"[TeacherPPO] PPO surrogate scale override: {agent_cfg.algorithm.ppo_surrogate_scale:g}")
        if args_cli.teacher_ppo_actor_learning_rate is not None:
            agent_cfg.algorithm.actor_learning_rate = float(args_cli.teacher_ppo_actor_learning_rate)
        if args_cli.teacher_ppo_actor_train_scope is not None:
            agent_cfg.algorithm.actor_train_scope = str(args_cli.teacher_ppo_actor_train_scope)
            print(f"[TeacherPPO] actor train scope override: {agent_cfg.algorithm.actor_train_scope}")
        if args_cli.teacher_ppo_large_heading_action_prior_coef is not None:
            agent_cfg.algorithm.large_heading_action_prior_coef = float(args_cli.teacher_ppo_large_heading_action_prior_coef)
            print(
                "[TeacherPPO] student-only large-heading action prior coefficient: "
                f"{agent_cfg.algorithm.large_heading_action_prior_coef:g}"
            )
        if args_cli.teacher_ppo_lam is not None:
            agent_cfg.algorithm.lam = float(args_cli.teacher_ppo_lam)
            print(f"[TeacherPPO] GAE lambda override: lam={agent_cfg.algorithm.lam:g}")
        if args_cli.teacher_ppo_critic_only:
            agent_cfg.algorithm.critic_only = True
        if args_cli.teacher_ppo_diagnostic_only:
            agent_cfg.algorithm.diagnostic_only = True
        if args_cli.teacher_ppo_diagnostic_rollout_steps is not None:
            agent_cfg.num_steps_per_env = int(args_cli.teacher_ppo_diagnostic_rollout_steps)
            print(
                "[TeacherPPO] diagnostic-only rollout steps override: "
                f"num_steps_per_env={agent_cfg.num_steps_per_env}"
            )
        if args_cli.teacher_ppo_joint_probe:
            agent_cfg.algorithm.critic_only = False
            agent_cfg.algorithm.diagnostic_only = False
            agent_cfg.algorithm.critic_relearning = False
            agent_cfg.algorithm.critic_relearning_perturb_burst = 0
            agent_cfg.algorithm.learning_rate = 1.0e-6
            agent_cfg.algorithm.num_learning_epochs = 1
            agent_cfg.algorithm.num_mini_batches = 4
            agent_cfg.algorithm.clip_param = 0.1
            agent_cfg.algorithm.schedule = "fixed"
            agent_cfg.algorithm.use_clipped_value_loss = True
            agent_cfg.num_steps_per_env = 192
            agent_cfg.save_interval = 1
            print(
                "[JointProbe] continuation configured: num_steps_per_env=192, epochs=1, minibatches=4, "
                "clip=0.1, learning_rate=1e-6, fixed schedule, save_interval=1"
            )
        if args_cli.teacher_ppo_critic_probe:
            agent_cfg.algorithm.critic_only = True
            agent_cfg.algorithm.diagnostic_only = False
            agent_cfg.algorithm.critic_relearning = False
            agent_cfg.algorithm.critic_relearning_perturb_burst = 0
            agent_cfg.algorithm.learning_rate = 1.0e-6
            agent_cfg.algorithm.num_learning_epochs = 1
            agent_cfg.algorithm.num_mini_batches = 4
            agent_cfg.algorithm.clip_param = 0.1
            agent_cfg.algorithm.schedule = "fixed"
            agent_cfg.algorithm.use_clipped_value_loss = True
            agent_cfg.num_steps_per_env = 192
            agent_cfg.save_interval = 1
            print(
                "[CriticProbe] continuation configured: critic_only=True, num_steps_per_env=192, epochs=1, "
                "minibatches=4, clip=0.1, learning_rate=1e-6, fixed schedule, save_interval=1"
            )
        if args_cli.teacher_ppo_long_return_critic_probe:
            agent_cfg.algorithm.critic_only = True
            agent_cfg.algorithm.diagnostic_only = False
            agent_cfg.algorithm.critic_relearning = True
            agent_cfg.algorithm.critic_relearning_return_lam = 1.0
            agent_cfg.algorithm.critic_relearning_perturb_burst = 0
            agent_cfg.algorithm.learning_rate = (
                float(args_cli.critic_relearning_learning_rate)
                if args_cli.critic_relearning_learning_rate is not None
                else 1.0e-6
            )
            agent_cfg.algorithm.num_learning_epochs = 1
            agent_cfg.algorithm.num_mini_batches = 4
            agent_cfg.algorithm.clip_param = 0.1
            agent_cfg.algorithm.schedule = "fixed"
            agent_cfg.algorithm.use_clipped_value_loss = True
            agent_cfg.num_steps_per_env = 192
            agent_cfg.save_interval = 1
            print(
                "[LongReturnCriticProbe] continuation configured: critic_only=True, lambda_return=1.0, "
                "num_steps_per_env=192, epochs=1, minibatches=4, clip=0.1, "
                f"learning_rate={agent_cfg.algorithm.learning_rate:g}, fixed schedule, save_interval=1"
            )
        if critic_relearning_requested:
            agent_cfg.num_steps_per_env = int(args_cli.critic_relearning_rollout_steps)
            if args_cli.critic_relearning_joint_actor:
                agent_cfg.algorithm.critic_only = False
                agent_cfg.algorithm.critic_relearning = False
                agent_cfg.algorithm.critic_relearning_perturb_burst = 0
                agent_cfg.algorithm.learning_rate = (
                    float(args_cli.critic_relearning_learning_rate)
                    if args_cli.critic_relearning_learning_rate is not None
                    else 1.0e-6
                )
                agent_cfg.algorithm.num_learning_epochs = 1
                agent_cfg.algorithm.num_mini_batches = 4
                agent_cfg.algorithm.clip_param = 0.1
                agent_cfg.algorithm.schedule = "fixed"
                agent_cfg.algorithm.use_clipped_value_loss = True
                print(
                    "[JointProbe] configured: fresh V10-representation recurrent critic + trainable bounded actor, "
                    f"num_steps_per_env={agent_cfg.num_steps_per_env}, epochs=1, minibatches=4, "
                    f"clip=0.1, learning_rate={agent_cfg.algorithm.learning_rate:.3g}, "
                    f"teacher_loss_coef={agent_cfg.algorithm.teacher_loss_coef:.6g}"
                )
            else:
                agent_cfg.algorithm.critic_only = True
                agent_cfg.algorithm.critic_relearning = True
                agent_cfg.algorithm.critic_relearning_return_lam = 1.0
                agent_cfg.algorithm.critic_relearning_perturb_burst = int(args_cli.critic_relearning_perturb_burst)
                if args_cli.critic_relearning_learning_rate is not None:
                    agent_cfg.algorithm.learning_rate = float(args_cli.critic_relearning_learning_rate)
                agent_cfg.algorithm.teacher_loss_coef = 0.0
                agent_cfg.algorithm.use_clipped_value_loss = False
                print(
                    "[CriticRelearn] configured: actor frozen, teacher disabled, unclipped critic loss, "
                    f"num_steps_per_env={agent_cfg.num_steps_per_env}, return_lambda=1.0, "
                    f"perturb_burst={agent_cfg.algorithm.critic_relearning_perturb_burst}, "
                    f"learning_rate={agent_cfg.algorithm.learning_rate:.3g}"
                )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if args_cli.partial_transfer:
        partial_transfer_path = _resolve_resume_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = _resolve_resume_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if args_cli.partial_transfer:
        load_partial_transfer_state_dict(
            policy=runner.alg.policy,
            checkpoint_path=partial_transfer_path,
            aligned_latent_old_trunk=args_cli.aligned_latent_old_trunk,
            freeze_old_actor_trunk=args_cli.freeze_old_actor_trunk,
            gru_residual=args_cli.gru_residual,
        )
    elif agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.model_only_resume:
            model_only_checkpoint = torch.load(resume_path, map_location=runner.device, weights_only=False)
            if not isinstance(model_only_checkpoint, dict) or "model_state_dict" not in model_only_checkpoint:
                raise KeyError("Model-only resume requires checkpoint['model_state_dict'].")
            runner.alg.policy.load_state_dict(model_only_checkpoint["model_state_dict"], strict=True)
            runner.current_learning_iteration = int(model_only_checkpoint.get("iter", 0))
            print(
                "Model-only resume: loaded model_state_dict; "
                "optimizer state intentionally skipped."
            )
        else:
            runner.load(resume_path)
        if args_cli.teacher_ppo_long_return_critic_probe and args_cli.critic_relearning_learning_rate is not None:
            resumed_lr = float(args_cli.critic_relearning_learning_rate)
            runner.alg.learning_rate = resumed_lr
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = resumed_lr
            print(f"[LongReturnCriticProbe] post-resume optimizer learning_rate reset to {resumed_lr:g}")
        _reset_action_std_from_env(runner)
    elif args_cli.warm_start_checkpoint is not None:
        is_recurrent_policy = bool(getattr(runner.alg.policy, "is_recurrent", False))
        if is_recurrent_policy:
            if args_cli.warm_start_mode != "recurrent_v10_actor":
                raise ValueError(
                    "RangerTerrainActorCriticRecurrent requires --warm_start_mode recurrent_v10_actor when "
                    "initializing from a feedforward V10 checkpoint. Use --resume for recurrent checkpoints."
                )
            warm_start_ranger_recurrent_from_feedforward(
                runner=runner,
                checkpoint_path=args_cli.warm_start_checkpoint,
            )
        else:
            if args_cli.warm_start_mode == "recurrent_v10_actor":
                raise ValueError("--warm_start_mode recurrent_v10_actor is valid only for the recurrent Ranger task.")
            warm_start_ranger_actor(
                runner=runner,
                checkpoint_path=args_cli.warm_start_checkpoint,
                mode=args_cli.warm_start_mode,
            )
    elif critic_relearning_requested:
        warm_start_ranger_recurrent_critic_relearning(
            runner=runner,
            actor_checkpoint_path=args_cli.critic_relearning_actor_checkpoint,
            v10_checkpoint_path=args_cli.critic_relearning_v10_checkpoint,
            freeze_actor=not args_cli.critic_relearning_joint_actor,
        )

    _configure_teacher_regularized_ppo(runner)
    _print_gru_residual_smoke(runner)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training. Always flush and export scalar history, including on KeyboardInterrupt.
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        writer = getattr(runner, "writer", None)
        if writer is not None and hasattr(writer, "flush"):
            writer.flush()
        if not args_cli.disable_iteration_metrics_csv:
            try:
                export_tensorboard_scalars(log_dir)
            except Exception as error:
                print(f"[WARN] Failed to export per-iteration CSV metrics: {error}")
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
