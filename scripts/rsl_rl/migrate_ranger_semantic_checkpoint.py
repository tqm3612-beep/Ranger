#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch


PROP_WHEEL_COLUMNS = (14, 15, 16, 17)
LEFT_WHEEL_COLUMNS = (14, 15)
MIGRATED_WEIGHT_KEYS = (
    "actor.prop_encoder.0.weight",
    "critic.prop_encoder.0.weight",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate a legacy Ranger checkpoint to semantic wheel-velocity observations."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _verify_equivalent_first_layer(old_weight: torch.Tensor, new_weight: torch.Tensor) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(29)
    semantic_prop = torch.randn((256, old_weight.shape[1]), generator=generator, dtype=old_weight.dtype)
    legacy_prop = semantic_prop.clone()
    legacy_prop[:, LEFT_WHEEL_COLUMNS] *= -1.0

    old_output = legacy_prop @ old_weight.T
    new_output = semantic_prop @ new_weight.T
    return float(torch.max(torch.abs(old_output - new_output)).item())


def main() -> None:
    args = _parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict")

    source_state = checkpoint["model_state_dict"]
    migrated_state = {key: value.clone() if torch.is_tensor(value) else value for key, value in source_state.items()}

    verification_errors: dict[str, float] = {}
    for key in MIGRATED_WEIGHT_KEYS:
        if key not in source_state:
            raise KeyError(f"Required checkpoint tensor is missing: {key}")
        old_weight = source_state[key]
        if old_weight.ndim != 2 or old_weight.shape[1] <= max(PROP_WHEEL_COLUMNS):
            raise ValueError(f"Unexpected shape for {key}: {tuple(old_weight.shape)}")
        new_weight = old_weight.clone()
        new_weight[:, LEFT_WHEEL_COLUMNS] *= -1.0
        migrated_state[key] = new_weight
        verification_errors[key] = _verify_equivalent_first_layer(old_weight, new_weight)

    checkpoint["model_state_dict"] = migrated_state
    infos = dict(checkpoint.get("infos") or {})
    infos.update(
        {
            "ranger_wheel_policy_space": "semantic_lb_lf_rf_rb",
            "ranger_wheel_observation_space": "semantic_lb_lf_rf_rb",
            "ranger_semantic_checkpoint_migrated_from": str(args.input),
            "ranger_semantic_migrated_prop_columns": list(PROP_WHEEL_COLUMNS),
            "ranger_semantic_sign_flipped_prop_columns": list(LEFT_WHEEL_COLUMNS),
        }
    )
    checkpoint["infos"] = infos

    # Optimizer moments were accumulated in the legacy observation coordinates.
    # Semantic baselines are intentionally resumed with --model_only_resume.
    checkpoint["optimizer_state_dict"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    changed_keys = [
        key
        for key in migrated_state
        if torch.is_tensor(migrated_state[key]) and not torch.equal(migrated_state[key], source_state[key])
    ]
    print(f"[OK] wrote semantic-compatible checkpoint: {args.output}")
    print(f"[OK] changed model tensors: {changed_keys}")
    for key, error in verification_errors.items():
        print(f"[VERIFY] {key}: max_first_layer_equivalence_error={error:.9g}")
    if set(changed_keys) != set(MIGRATED_WEIGHT_KEYS):
        raise RuntimeError(f"Unexpected changed model tensors: {changed_keys}")
    if any(error > 1.0e-5 for error in verification_errors.values()):
        raise RuntimeError(f"First-layer equivalence verification failed: {verification_errors}")


if __name__ == "__main__":
    main()

