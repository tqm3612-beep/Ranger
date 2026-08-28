from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch
from tensordict import TensorDict


CHECKPOINT_KEYWORDS = (
    "map_encoder",
    "wheel_head",
    "suspension_head",
    "actor",
    "trunk",
    "mlp",
)
DEFAULT_TRANSFER_MODULES = (
    "actor.map_encoder",
    "actor.prop_encoder",
    "actor.goal_encoder",
    "actor.wheel_head",
    "actor.suspension_head",
)
OPTIONAL_TRANSFER_MODULES = ()
NEW_RANDOM_MODULES = (
    "actor.memory",
    "actor.fusion_norm",
    "actor.hidden_norm",
    "actor.actor_trunk",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_recurrent_policy_class():
    agents_dir = (
        _repo_root()
        / "source"
        / "Ranger"
        / "Ranger"
        / "tasks"
        / "manager_based"
        / "ranger"
        / "agents"
    )
    package_name = "_partial_transfer_agents"
    package = types.ModuleType(package_name)
    package.__path__ = [str(agents_dir)]
    sys.modules[package_name] = package

    for module_name in ("rsl_rl_custom_policy", "rsl_rl_recurrent_policy"):
        module_path = agents_dir / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(f"{package_name}.{module_name}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package_name}.{module_name}"] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}.rsl_rl_recurrent_policy"].RangerTerrainActorCriticRecurrent


def _load_checkpoint_state(checkpoint_path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}: {checkpoint_path}")
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise KeyError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")
    return checkpoint, model_state


def _make_recurrent_policy():
    policy_cls = _load_recurrent_policy_class()
    obs = TensorDict(
        {
            "policy_state": torch.zeros(1, 42),
            "policy_map": torch.zeros(1, 8 * 21 * 13),
            "critic_privileged": torch.zeros(1, 12),
        },
        batch_size=[1],
    )
    return policy_cls(
        obs=obs,
        obs_groups={
            "policy": ["policy_state", "policy_map"],
            "critic": ["policy_state", "policy_map", "critic_privileged"],
        },
        num_actions=8,
        actor_hidden_dims=[],
        critic_hidden_dims=[],
        noise_std_type="log",
    )


def _shape_text(tensor: torch.Tensor) -> str:
    return "x".join(str(dim) for dim in tuple(tensor.shape)) or "scalar"


def _is_under_module(key: str, modules: tuple[str, ...]) -> bool:
    return any(key == module or key.startswith(f"{module}.") for module in modules)


def write_checkpoint_key_analysis(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint, model_state = _load_checkpoint_state(checkpoint_path)
    lines: list[str] = []
    lines.append(f"checkpoint: {checkpoint_path}")
    lines.append(f"top_level_keys: {sorted(checkpoint.keys())}")
    lines.append(f"model_state_dict_key_count: {len(model_state)}")
    lines.append("")
    lines.append("filtered_keys:")
    for key in sorted(model_state):
        if any(keyword in key for keyword in CHECKPOINT_KEYWORDS):
            tensor = model_state[key]
            shape = _shape_text(tensor) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
            lines.append(f"{key} | shape={shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_key_mapping(checkpoint_path: Path, output_path: Path) -> None:
    _, old_state = _load_checkpoint_state(checkpoint_path)
    new_state = _make_recurrent_policy().state_dict()
    transfer_modules = DEFAULT_TRANSFER_MODULES
    optional_modules = OPTIONAL_TRANSFER_MODULES
    random_modules = NEW_RANDOM_MODULES
    loaded_modules = sorted(set(key.rsplit(".", 1)[0] for key in new_state if _is_under_module(key, transfer_modules)))

    lines: list[str] = []
    lines.append(f"checkpoint: {checkpoint_path}")
    lines.append(f"old_key_count: {len(old_state)}")
    lines.append(f"new_recurrent_key_count: {len(new_state)}")
    lines.append("")
    lines.append("current_recurrent_keys:")
    for key in sorted(new_state):
        lines.append(f"{key} | shape={_shape_text(new_state[key])}")

    lines.append("")
    lines.append("direct_transfer_candidates:")
    candidate_modules = transfer_modules + optional_modules
    for key in sorted(new_state):
        if not _is_under_module(key, candidate_modules):
            continue
        old_tensor = old_state.get(key)
        new_tensor = new_state[key]
        if old_tensor is None:
            lines.append(f"SKIP missing | {key}")
        elif tuple(old_tensor.shape) != tuple(new_tensor.shape):
            lines.append(
                f"SKIP shape mismatch | {key} | old={_shape_text(old_tensor)} new={_shape_text(new_tensor)}"
            )
        else:
            tag = "LOAD optional" if _is_under_module(key, optional_modules) else "LOAD"
            lines.append(f"{tag} | {key} -> {key} | shape={_shape_text(new_tensor)}")

    lines.append("")
    lines.append("explicit_random_initialized_modules:")
    for module in random_modules:
        for key in sorted(new_state):
            if _is_under_module(key, (module,)):
                reason = "architecture mismatch/new recurrent path"
                if module in ("actor.prop_encoder", "actor.goal_encoder"):
                    reason = "architecture mismatch/new perception state split"
                elif module == "actor.memory":
                    reason = "architecture mismatch/new GRU recurrent state"
                elif module == "actor.fusion_norm":
                    reason = "architecture mismatch/new recurrent fusion normalization"
                lines.append(f"RANDOM | {key} | {reason}")

    lines.append("")
    lines.append("skipped_old_keys_in_keywords:")
    for key in sorted(old_state):
        if not any(keyword in key for keyword in CHECKPOINT_KEYWORDS):
            continue
        if key in new_state and _is_under_module(key, candidate_modules):
            continue
        reason = "missing"
        if key in new_state and tuple(old_state[key].shape) != tuple(new_state[key].shape):
            reason = f"shape mismatch old={_shape_text(old_state[key])} new={_shape_text(new_state[key])}"
        elif key.startswith("actor.state_encoder."):
            reason = "architecture mismatch; old monolithic state_encoder splits into prop_encoder/goal_encoder"
        elif key.startswith("critic."):
            reason = "not requested for actor partial transfer"
        lines.append(f"SKIP old | {key} | {reason}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_state_encoder_mapping(checkpoint_path: Path, output_path: Path) -> None:
    _, old_state = _load_checkpoint_state(checkpoint_path)
    new_state = _make_recurrent_policy().state_dict()
    prop_indices = list(range(0, 26)) + list(range(34, 42))
    goal_indices = list(range(26, 34))
    mappings = (
        ("actor.state_encoder.0.weight", "actor.prop_encoder.0.weight", prop_indices),
        ("actor.state_encoder.0.weight", "actor.goal_encoder.0.weight", goal_indices),
        ("actor.state_encoder.0.bias", "actor.prop_encoder.0.bias", None),
        ("actor.state_encoder.0.bias", "actor.goal_encoder.0.bias", None),
        ("actor.state_encoder.2.weight", "actor.prop_encoder.2.weight", None),
        ("actor.state_encoder.2.weight", "actor.goal_encoder.2.weight", None),
        ("actor.state_encoder.2.bias", "actor.prop_encoder.2.bias", None),
        ("actor.state_encoder.2.bias", "actor.goal_encoder.2.bias", None),
    )

    lines: list[str] = []
    lines.append(f"checkpoint: {checkpoint_path}")
    lines.append("old_state_encoder_keys:")
    for key in sorted(old_state):
        if key.startswith("actor.state_encoder."):
            lines.append(f"{key} | shape={_shape_text(old_state[key])}")
    lines.append("")
    lines.append("current_split_encoder_keys:")
    for key in sorted(new_state):
        if key.startswith("actor.prop_encoder.") or key.startswith("actor.goal_encoder."):
            lines.append(f"{key} | shape={_shape_text(new_state[key])}")
    lines.append("")
    lines.append(f"prop_indices: {prop_indices}")
    lines.append(f"goal_indices: {goal_indices}")
    lines.append("")
    lines.append("state_encoder_split_mapping:")
    for old_key, new_key, indices in mappings:
        old_tensor = old_state.get(old_key)
        new_tensor = new_state.get(new_key)
        if old_tensor is None:
            lines.append(f"SKIP | {old_key} -> {new_key} | missing old key")
            continue
        if new_tensor is None:
            lines.append(f"SKIP | {old_key} -> {new_key} | missing new key")
            continue
        source_tensor = old_tensor[:, indices] if indices is not None else old_tensor
        if tuple(source_tensor.shape) == tuple(new_tensor.shape):
            lines.append(f"LOAD | {old_key} -> {new_key} | shape={_shape_text(source_tensor)}")
        else:
            lines.append(
                f"SKIP shape mismatch | {old_key} -> {new_key} | "
                f"source={_shape_text(source_tensor)} new={_shape_text(new_tensor)}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Ranger feedforward-to-recurrent partial transfer keys.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--analysis_dir", type=Path, default=Path("logs/analysis"))
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    key_output = args.analysis_dir / "partial_transfer_checkpoint_keys.txt"
    mapping_output = args.analysis_dir / "partial_transfer_key_mapping.txt"
    state_mapping_output = args.analysis_dir / "state_encoder_transfer_mapping.txt"
    write_checkpoint_key_analysis(checkpoint_path, key_output)
    write_key_mapping(checkpoint_path, mapping_output)
    write_state_encoder_mapping(checkpoint_path, state_mapping_output)
    print(f"Wrote checkpoint key analysis: {key_output}")
    print(f"Wrote partial transfer mapping: {mapping_output}")
    print(f"Wrote state encoder transfer mapping: {state_mapping_output}")


if __name__ == "__main__":
    main()
