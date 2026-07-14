from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


ACTOR_SUSPENSION_MODULES = ("map_encoder", "state_encoder", "actor_trunk", "suspension_head")
ACTOR_FULL_MODULES = ACTOR_SUSPENSION_MODULES + ("wheel_head",)


def _describe_public_attrs(obj: object) -> str:
    attrs = [name for name in dir(obj) if not name.startswith("__")]
    return ", ".join(attrs[:80])


def get_ranger_actor(runner) -> nn.Module:
    """Return the Ranger dual-head actor module from the current runner."""

    alg = getattr(runner, "alg", None)
    if alg is None:
        raise AttributeError(f"Runner has no 'alg' attribute. Runner type={type(runner).__name__}")

    candidate_paths = (
        ("runner.alg.policy", getattr(alg, "policy", None)),
        ("runner.alg.actor_critic", getattr(alg, "actor_critic", None)),
    )
    diagnostics: list[str] = []
    for path, policy in candidate_paths:
        if policy is None:
            diagnostics.append(f"{path}: missing")
            continue
        actor = getattr(policy, "actor", None)
        if actor is None:
            diagnostics.append(f"{path}: type={type(policy).__name__}, no actor attr")
            continue
        required = ACTOR_FULL_MODULES
        missing = [name for name in required if not hasattr(actor, name)]
        if missing:
            diagnostics.append(f"{path}.actor: type={type(actor).__name__}, missing={missing}")
            continue
        return actor

    alg_attrs = _describe_public_attrs(alg)
    raise AttributeError(
        "Could not locate Ranger dual-head actor. "
        f"Diagnostics={diagnostics}. alg type={type(alg).__name__}; attrs={alg_attrs}"
    )


def _get_ranger_policy(runner) -> nn.Module:
    alg = getattr(runner, "alg", None)
    for policy in (getattr(alg, "policy", None), getattr(alg, "actor_critic", None)):
        if policy is not None and getattr(policy, "actor", None) is get_ranger_actor(runner):
            return policy
    raise AttributeError("Could not locate policy object that owns the Ranger actor.")


def _load_checkpoint_model_state(checkpoint_path: str | Path) -> tuple[dict, dict[str, torch.Tensor]]:
    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}: {path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'model_state_dict'. Available keys: {sorted(checkpoint.keys())}")
    model_state = checkpoint["model_state_dict"]
    if not isinstance(model_state, dict):
        raise TypeError(f"model_state_dict must be a dict, got {type(model_state).__name__}")
    return checkpoint, model_state


def _validate_no_forbidden_loaded_keys(model_state: dict[str, torch.Tensor], modules: Iterable[str]) -> None:
    del modules
    forbidden_prefixes = ("critic.", "actor_obs_normalizer.", "critic_obs_normalizer.")
    forbidden_exact = {"std", "log_std"}
    forbidden = [
        key
        for key in model_state
        if key in forbidden_exact or any(key.startswith(prefix) for prefix in forbidden_prefixes)
    ]
    if forbidden:
        preview = ", ".join(forbidden[:12])
        print(f"[WarmStart] checkpoint contains excluded keys that will not be loaded: {preview}")


def _load_actor_module(actor: nn.Module, model_state: dict[str, torch.Tensor], module_name: str) -> int:
    module = getattr(actor, module_name, None)
    if module is None:
        raise AttributeError(f"Target actor is missing module '{module_name}'. Actor type={type(actor).__name__}")
    if not isinstance(module, nn.Module):
        raise TypeError(f"Target actor attribute '{module_name}' is not an nn.Module: {type(module).__name__}")

    target_state = module.state_dict()
    source_prefix = f"actor.{module_name}."
    source_keys = {key for key in model_state if key.startswith(source_prefix)}
    expected_source_keys = {source_prefix + key for key in target_state}
    missing = sorted(expected_source_keys - source_keys)
    extra = sorted(source_keys - expected_source_keys)
    if missing:
        raise KeyError(f"Warm-start checkpoint missing keys for {module_name}: {missing}")
    if extra:
        raise KeyError(f"Warm-start checkpoint has unexpected keys for {module_name}: {extra}")

    load_state: dict[str, torch.Tensor] = {}
    for target_key, target_tensor in target_state.items():
        source_key = source_prefix + target_key
        source_tensor = model_state[source_key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"Shape mismatch for {source_key}: checkpoint {tuple(source_tensor.shape)} "
                f"vs target {tuple(target_tensor.shape)}"
            )
        load_state[target_key] = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)

    module.load_state_dict(load_state, strict=True)
    return len(load_state)


def _reset_actor_suspension_action_std(policy: nn.Module) -> str:
    if hasattr(policy, "std"):
        std = getattr(policy, "std")
        if tuple(std.shape) != (8,):
            raise ValueError(f"Expected 8-dim std for Ranger action space, got shape {tuple(std.shape)}")
        with torch.no_grad():
            std[:4].fill_(0.30)
            std[4:8].fill_(0.55)
        return "std[:4]=0.30 std[4:8]=0.55"
    if hasattr(policy, "log_std"):
        log_std = getattr(policy, "log_std")
        if tuple(log_std.shape) != (8,):
            raise ValueError(f"Expected 8-dim log_std for Ranger action space, got shape {tuple(log_std.shape)}")
        with torch.no_grad():
            log_std[:4].fill_(math.log(0.30))
            log_std[4:8].fill_(math.log(0.55))
        return "log_std[:4]=log(0.30) log_std[4:8]=log(0.55)"
    return "policy has no std/log_std; unchanged"


def warm_start_ranger_actor(runner, checkpoint_path: str | Path, mode: str) -> None:
    if mode not in {"actor_suspension", "actor_full"}:
        raise ValueError(f"Unsupported warm-start mode: {mode!r}. Expected 'actor_suspension' or 'actor_full'.")

    actor = get_ranger_actor(runner)
    policy = _get_ranger_policy(runner)
    checkpoint, model_state = _load_checkpoint_model_state(checkpoint_path)
    modules = ACTOR_SUSPENSION_MODULES if mode == "actor_suspension" else ACTOR_FULL_MODULES
    _validate_no_forbidden_loaded_keys(model_state, modules)

    loaded_counts: dict[str, int] = {}
    for module_name in modules:
        loaded_counts[module_name] = _load_actor_module(actor, model_state, module_name)

    for parameter in actor.parameters():
        parameter.requires_grad = True

    runner.current_learning_iteration = 0

    print(f"[WarmStart] source: {Path(checkpoint_path).expanduser()}")
    print(f"[WarmStart] checkpoint keys: {sorted(checkpoint.keys())}")
    print(f"[WarmStart] model_state_dict tensors: {len(model_state)}")
    print(f"[WarmStart] mode: {mode}")
    if mode == "actor_suspension":
        for module_name in modules:
            print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
        print("[WarmStart] random: actor.wheel_head")
        std_msg = _reset_actor_suspension_action_std(policy)
    else:
        print("[WarmStart] loaded: complete actor")
        for module_name in modules:
            print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
        std_msg = "kept freshly initialized policy std/log_std"
    print("[WarmStart] new: critic")
    print("[WarmStart] new: optimizer")
    print(f"[WarmStart] new: action std ({std_msg})")
    print("[WarmStart] iteration: 0")
    print(f"[WarmStart] all actor parameters trainable: {all(parameter.requires_grad for parameter in actor.parameters())}")
