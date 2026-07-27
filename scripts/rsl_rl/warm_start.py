from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


ACTOR_SHARED_MODULES = ("map_encoder", "state_encoder", "actor_trunk")
ACTOR_SUSPENSION_MODULES = ACTOR_SHARED_MODULES + ("suspension_head",)
ACTOR_WHEEL_MODULES = ACTOR_SHARED_MODULES + ("wheel_head",)
ACTOR_FULL_MODULES = ACTOR_SHARED_MODULES + ("suspension_head", "wheel_head")


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


def _reset_suspension_head(actor: nn.Module) -> str:
    """Reinitialize the suspension branch with an exactly neutral initial output."""

    suspension_head = getattr(actor, "suspension_head", None)
    if suspension_head is None:
        raise AttributeError(f"Target actor has no suspension_head. Actor type={type(actor).__name__}")
    if not isinstance(suspension_head, nn.Module):
        raise TypeError(f"actor.suspension_head is not an nn.Module: {type(suspension_head).__name__}")

    for module in suspension_head.modules():
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()

    linear_layers = [module for module in suspension_head.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise TypeError("actor.suspension_head must contain at least one nn.Linear layer.")
    output_layer = linear_layers[-1]
    with torch.no_grad():
        output_layer.weight.zero_()
        if output_layer.bias is not None:
            output_layer.bias.zero_()

    return (
        f"reset {len(linear_layers)} linear layers; final weight/bias exactly zero "
        f"(output_dim={output_layer.out_features})"
    )


def _reset_wheel_head(actor: nn.Module) -> str:
    """Reinitialize the wheel branch with an exactly neutral initial output."""

    wheel_head = getattr(actor, "wheel_head", None)
    if wheel_head is None:
        raise AttributeError(f"Target actor has no wheel_head. Actor type={type(actor).__name__}")
    if not isinstance(wheel_head, nn.Module):
        raise TypeError(f"actor.wheel_head is not an nn.Module: {type(wheel_head).__name__}")

    for module in wheel_head.modules():
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()

    linear_layers = [module for module in wheel_head.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise TypeError("actor.wheel_head must contain at least one nn.Linear layer.")
    output_layer = linear_layers[-1]
    with torch.no_grad():
        output_layer.weight.zero_()
        if output_layer.bias is not None:
            output_layer.bias.zero_()

    return (
        f"reset {len(linear_layers)} linear layers; final weight/bias exactly zero "
        f"(output_dim={output_layer.out_features})"
    )


def _reset_wheel_output_layer(actor: nn.Module) -> tuple[nn.Linear, str]:
    """Reset only the final wheel-action mapping while preserving wheel hidden features."""

    wheel_head = getattr(actor, "wheel_head", None)
    if wheel_head is None:
        raise AttributeError(f"Target actor has no wheel_head. Actor type={type(actor).__name__}")
    if not isinstance(wheel_head, nn.Module):
        raise TypeError(f"actor.wheel_head is not an nn.Module: {type(wheel_head).__name__}")

    linear_layers = [module for module in wheel_head.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise TypeError("actor.wheel_head must contain at least one nn.Linear layer.")
    output_layer = linear_layers[-1]
    with torch.no_grad():
        output_layer.weight.zero_()
        if output_layer.bias is not None:
            output_layer.bias.zero_()

    return (
        output_layer,
        f"final weight/bias exactly zero (input_dim={output_layer.in_features}, output_dim={output_layer.out_features})",
    )


def _reset_warm_start_action_std(policy: nn.Module) -> str:
    """Restore the action-noise profile declared by the current runner configuration."""

    configured = getattr(policy, "_configured_initial_action_std", None)
    if configured is None:
        configured = torch.tensor([0.15] * 4 + [0.25] * 4, dtype=torch.float32)
    configured = configured.detach()
    if tuple(configured.shape) != (8,):
        raise ValueError(f"Expected 8-dim configured action std, got shape {tuple(configured.shape)}")
    if hasattr(policy, "std"):
        std = getattr(policy, "std")
        if tuple(std.shape) != (8,):
            raise ValueError(f"Expected 8-dim std for Ranger action space, got shape {tuple(std.shape)}")
        with torch.no_grad():
            std.copy_(configured.to(device=std.device, dtype=std.dtype))
        std.requires_grad_(False)
        return f"fixed std={configured.tolist()}"
    if hasattr(policy, "log_std"):
        log_std = getattr(policy, "log_std")
        if tuple(log_std.shape) != (8,):
            raise ValueError(f"Expected 8-dim log_std for Ranger action space, got shape {tuple(log_std.shape)}")
        with torch.no_grad():
            log_std.copy_(torch.log(configured).to(device=log_std.device, dtype=log_std.dtype))
        log_std.requires_grad_(False)
        return f"fixed log_std=log({configured.tolist()})"
    return "policy has no std/log_std; unchanged"


def _reset_suspension_action_std(policy: nn.Module) -> str:
    """Reset only suspension exploration dimensions from the current runner configuration."""

    configured = getattr(policy, "_configured_initial_action_std", None)
    if configured is None:
        configured = torch.tensor([0.003] * 4 + [0.01] * 4, dtype=torch.float32)
    configured = configured.detach()
    if tuple(configured.shape) != (8,):
        raise ValueError(f"Expected 8-dim configured action std, got shape {tuple(configured.shape)}")

    if hasattr(policy, "std"):
        std = getattr(policy, "std")
        if tuple(std.shape) != (8,):
            raise ValueError(f"Expected 8-dim std for Ranger action space, got shape {tuple(std.shape)}")
        with torch.no_grad():
            std[:4].copy_(configured[:4].to(device=std.device, dtype=std.dtype))
        std.requires_grad_(False)
        return f"fixed suspension std={std[:4].detach().cpu().tolist()}; wheel std preserved"
    if hasattr(policy, "log_std"):
        log_std = getattr(policy, "log_std")
        if tuple(log_std.shape) != (8,):
            raise ValueError(f"Expected 8-dim log_std for Ranger action space, got shape {tuple(log_std.shape)}")
        with torch.no_grad():
            log_std[:4].copy_(torch.log(configured[:4]).to(device=log_std.device, dtype=log_std.dtype))
        log_std.requires_grad_(False)
        return f"fixed suspension log_std=log({configured[:4].tolist()}); wheel log_std preserved"
    return "policy has no std/log_std; unchanged"


def warm_start_ranger_actor(runner, checkpoint_path: str | Path, mode: str) -> None:
    valid_modes = {
        "actor_suspension",
        "actor_wheel",
        "actor_suspension_only",
        "actor_wheel_reset_suspension",
        "actor_wheel_reset_final",
        "actor_reset_heads",
        "actor_heads",
        "actor_full",
    }
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported warm-start mode: {mode!r}. Expected one of {sorted(valid_modes)}."
        )

    actor = get_ranger_actor(runner)
    policy = _get_ranger_policy(runner)
    checkpoint, model_state = _load_checkpoint_model_state(checkpoint_path)
    if mode == "actor_suspension":
        modules = ACTOR_SUSPENSION_MODULES
    elif mode == "actor_wheel_reset_suspension":
        modules = ACTOR_WHEEL_MODULES
    elif mode == "actor_reset_heads":
        modules = ACTOR_SHARED_MODULES
    else:
        # actor_wheel also loads the complete actor so the existing suspension branch
        # remains available for later staged reintroduction.
        modules = ACTOR_FULL_MODULES
    _validate_no_forbidden_loaded_keys(model_state, modules)

    loaded_counts: dict[str, int] = {}
    for module_name in modules:
        loaded_counts[module_name] = _load_actor_module(actor, model_state, module_name)
    suspension_reset_msg = None
    wheel_reset_msg = None
    wheel_output_layer = None
    if mode == "actor_wheel_reset_suspension":
        suspension_reset_msg = _reset_suspension_head(actor)
    elif mode == "actor_reset_heads":
        suspension_reset_msg = _reset_suspension_head(actor)
        wheel_reset_msg = _reset_wheel_head(actor)
    elif mode == "actor_wheel_reset_final":
        wheel_output_layer, wheel_reset_msg = _reset_wheel_output_layer(actor)

    for parameter in actor.parameters():
        parameter.requires_grad = True
    if mode in {
        "actor_wheel",
        "actor_suspension_only",
        "actor_wheel_reset_suspension",
        "actor_wheel_reset_final",
        "actor_reset_heads",
        "actor_heads",
    }:
        for parameter in actor.parameters():
            parameter.requires_grad = False
    if mode == "actor_wheel":
        for parameter in actor.wheel_head.parameters():
            parameter.requires_grad = True
    elif mode in {"actor_suspension_only", "actor_wheel_reset_suspension"}:
        for parameter in actor.suspension_head.parameters():
            parameter.requires_grad = True
    elif mode == "actor_wheel_reset_final":
        assert wheel_output_layer is not None
        for parameter in wheel_output_layer.parameters():
            parameter.requires_grad = True
    elif mode in {"actor_reset_heads", "actor_heads"}:
        for head in (actor.suspension_head, actor.wheel_head):
            for parameter in head.parameters():
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
        std_msg = _reset_warm_start_action_std(policy)
    elif mode == "actor_wheel_reset_suspension":
        print("[WarmStart] loaded: navigation encoders/trunk + wheel head")
        for module_name in modules:
            print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
        print(f"[WarmStart] reset: actor.suspension_head ({suspension_reset_msg})")
        print("[WarmStart] trainable: actor.suspension_head only")
        print("[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk/wheel_head")
        std_msg = _reset_suspension_action_std(policy)
    elif mode == "actor_wheel_reset_final":
        print("[WarmStart] loaded: complete actor")
        for module_name in modules:
            print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
        print(f"[WarmStart] reset: actor.wheel_head final Linear ({wheel_reset_msg})")
        print("[WarmStart] trainable: actor.wheel_head final Linear only")
        print(
            "[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk/"
            "suspension_head/wheel_head hidden layers"
        )
        std_msg = _reset_warm_start_action_std(policy)
    elif mode == "actor_reset_heads":
        print("[WarmStart] loaded: navigation encoders/trunk only")
        for module_name in modules:
            print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
        print(f"[WarmStart] reset: actor.suspension_head ({suspension_reset_msg})")
        print(f"[WarmStart] reset: actor.wheel_head ({wheel_reset_msg})")
        print("[WarmStart] trainable: actor.suspension_head + actor.wheel_head")
        print("[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk")
        std_msg = _reset_warm_start_action_std(policy)
    elif mode == "actor_wheel":
        print("[WarmStart] loaded: complete actor")
        for module_name in modules:
            print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
        print("[WarmStart] trainable: actor.wheel_head only")
        print("[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk/suspension_head")
        std_msg = _reset_warm_start_action_std(policy)
    elif mode == "actor_suspension_only":
        print("[WarmStart] loaded: complete actor")
        for module_name in modules:
            print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
        print("[WarmStart] trainable: actor.suspension_head only")
        print("[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk/wheel_head")
        std_msg = _reset_warm_start_action_std(policy)
    elif mode == "actor_heads":
        print("[WarmStart] loaded: complete actor")
        for module_name in modules:
            print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
        print("[WarmStart] trainable: actor.suspension_head + actor.wheel_head")
        print("[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk")
        std_msg = _reset_warm_start_action_std(policy)
    else:
        print("[WarmStart] loaded: complete actor")
        for module_name in modules:
            print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
        std_msg = _reset_warm_start_action_std(policy)
    print("[WarmStart] new: critic")
    print("[WarmStart] new: optimizer")
    print(f"[WarmStart] new: action std ({std_msg})")
    print("[WarmStart] iteration: 0")
    trainable_names = [name for name, parameter in actor.named_parameters() if parameter.requires_grad]
    frozen_names = [name for name, parameter in actor.named_parameters() if not parameter.requires_grad]
    if mode == "actor_wheel_reset_suspension":
        expected_trainable_names = {
            f"suspension_head.{name}" for name, _ in actor.suspension_head.named_parameters()
        }
        if set(trainable_names) != expected_trainable_names:
            raise RuntimeError(
                "Reset-suspension warm start exposed an unexpected actor parameter set. "
                f"Expected={sorted(expected_trainable_names)}, actual={sorted(trainable_names)}"
            )
        print("[WarmStart] validated: only actor.suspension_head is trainable")
    elif mode == "actor_reset_heads":
        expected_trainable_names = {
            f"suspension_head.{name}" for name, _ in actor.suspension_head.named_parameters()
        } | {
            f"wheel_head.{name}" for name, _ in actor.wheel_head.named_parameters()
        }
        if set(trainable_names) != expected_trainable_names:
            raise RuntimeError(
                "Reset-heads warm start exposed an unexpected actor parameter set. "
                f"Expected={sorted(expected_trainable_names)}, actual={sorted(trainable_names)}"
            )
        print("[WarmStart] validated: only actor.suspension_head and actor.wheel_head are trainable")
    elif mode == "actor_wheel_reset_final":
        assert wheel_output_layer is not None
        output_parameter_ids = {id(parameter) for parameter in wheel_output_layer.parameters()}
        expected_trainable_names = {
            name for name, parameter in actor.named_parameters() if id(parameter) in output_parameter_ids
        }
        if set(trainable_names) != expected_trainable_names:
            raise RuntimeError(
                "Reset-wheel-final warm start exposed an unexpected actor parameter set. "
                f"Expected={sorted(expected_trainable_names)}, actual={sorted(trainable_names)}"
            )
        print("[WarmStart] validated: only actor.wheel_head final Linear is trainable")
    print(f"[WarmStart] trainable actor parameter tensors: {len(trainable_names)}")
    print(f"[WarmStart] frozen actor parameter tensors: {len(frozen_names)}")
    if frozen_names:
        print(f"[WarmStart] frozen actor parameters: {frozen_names}")
