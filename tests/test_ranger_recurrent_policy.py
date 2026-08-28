from __future__ import annotations

import os
import sys
import types
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_ROOT = os.path.join(REPO_ROOT, "source", "Ranger")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)


def _load_agent_module(module_name: str, file_name: str):
    package_names = [
        "Ranger",
        "Ranger.tasks",
        "Ranger.tasks.manager_based",
        "Ranger.tasks.manager_based.ranger",
        "Ranger.tasks.manager_based.ranger.agents",
    ]
    package_paths = [
        os.path.join(SOURCE_ROOT, "Ranger"),
        os.path.join(SOURCE_ROOT, "Ranger", "tasks"),
        os.path.join(SOURCE_ROOT, "Ranger", "tasks", "manager_based"),
        os.path.join(SOURCE_ROOT, "Ranger", "tasks", "manager_based", "ranger"),
        os.path.join(SOURCE_ROOT, "Ranger", "tasks", "manager_based", "ranger", "agents"),
    ]
    for package_name, package_path in zip(package_names, package_paths, strict=True):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [package_path]
            sys.modules[package_name] = package

    full_name = f"Ranger.tasks.manager_based.ranger.agents.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    module_path = os.path.join(package_paths[-1], file_name)
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


feedforward_policy_module = _load_agent_module("rsl_rl_custom_policy", "rsl_rl_custom_policy.py")
RangerTerrainActorCritic = feedforward_policy_module.RangerTerrainActorCritic
recurrent_policy_module = _load_agent_module("rsl_rl_recurrent_policy", "rsl_rl_recurrent_policy.py")
RangerTerrainActorCriticRecurrent = recurrent_policy_module.RangerTerrainActorCriticRecurrent
NETWORK_SPEC = recurrent_policy_module._RANGER_RECURRENT_NETWORK_SPEC
teacher_ppo_module = _load_agent_module("rsl_rl_teacher_regularized_ppo", "rsl_rl_teacher_regularized_ppo.py")
RangerTeacherRegularizedPPO = teacher_ppo_module.RangerTeacherRegularizedPPO
teacher_wheel_sample_weights = teacher_ppo_module.teacher_wheel_sample_weights
near_goal_brake_common_loss = teacher_ppo_module.near_goal_brake_common_loss
ranger_wheel_raw_to_semantic = teacher_ppo_module.ranger_wheel_raw_to_semantic
ranger_wheel_semantic_modes = teacher_ppo_module.ranger_wheel_semantic_modes
warm_start_path = os.path.join(REPO_ROOT, "scripts", "rsl_rl", "warm_start.py")
warm_start_spec = importlib.util.spec_from_file_location("ranger_recurrent_warm_start", warm_start_path)
warm_start_module = importlib.util.module_from_spec(warm_start_spec)
assert warm_start_spec.loader is not None
warm_start_spec.loader.exec_module(warm_start_module)
warm_start_ranger_recurrent_from_feedforward = warm_start_module.warm_start_ranger_recurrent_from_feedforward
RECURRENT_V10_COMPATIBLE_MODULES = warm_start_module.RECURRENT_V10_COMPATIBLE_MODULES
RECURRENT_NEW_ACTOR_MODULES = warm_start_module.RECURRENT_NEW_ACTOR_MODULES
distill_path = os.path.join(REPO_ROOT, "scripts", "rsl_rl", "distill_recurrent.py")
distill_spec = importlib.util.spec_from_file_location("ranger_recurrent_distillation", distill_path)
distill_module = importlib.util.module_from_spec(distill_spec)
assert distill_spec.loader is not None
distill_spec.loader.exec_module(distill_module)
compute_distillation_losses = distill_module.compute_distillation_losses
bound_teacher_actions = distill_module.bound_teacher_actions
SequenceReplayAnchor = distill_module.SequenceReplayAnchor
recurrent_sequence_actions = distill_module.recurrent_sequence_actions
build_teacher_observations = distill_module.build_teacher_observations
goal_hidden_mask = distill_module.goal_hidden_mask
goal_hidden_sample_weights = distill_module.goal_hidden_sample_weights
reset_recurrent_memory = distill_module.reset_recurrent_memory
restore_distillation_optimizer = distill_module._restore_distillation_optimizer
select_rollout_actions = distill_module._select_rollout_actions


OBS_GROUPS = {
    "policy": ["policy_state", "policy_map"],
    "critic": ["policy_state", "policy_map", "critic_privileged"],
}


def _make_obs(batch_shape: tuple[int, ...], privileged_dim: int = 17) -> TensorDict:
    return TensorDict(
        {
            "policy_state": torch.randn(*batch_shape, 42),
            "policy_map": torch.randn(*batch_shape, 8 * 21 * 13),
            "critic_privileged": torch.randn(*batch_shape, privileged_dim),
        },
        batch_size=batch_shape,
    )


def _make_policy(num_envs: int = 4, privileged_dim: int = 17) -> RangerTerrainActorCriticRecurrent:
    torch.manual_seed(7)
    obs = _make_obs((num_envs,), privileged_dim=privileged_dim)
    policy = RangerTerrainActorCriticRecurrent(
        obs=obs,
        obs_groups=OBS_GROUPS,
        num_actions=8,
        init_noise_std=0.1,
        initial_action_std=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    )
    policy.eval()
    return policy


def _make_feedforward_policy(num_envs: int = 4, privileged_dim: int = 17) -> RangerTerrainActorCritic:
    torch.manual_seed(7)
    obs = _make_obs((num_envs,), privileged_dim=privileged_dim)
    policy = RangerTerrainActorCritic(
        obs=obs,
        obs_groups=OBS_GROUPS,
        num_actions=8,
        init_noise_std=0.1,
        initial_action_std=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    )
    policy.eval()
    return policy


def test_recurrent_policy_shapes() -> None:
    num_envs = 5
    policy = _make_policy(num_envs=num_envs, privileged_dim=19)
    obs = _make_obs((num_envs,), privileged_dim=19)

    actions = policy.act(obs)
    values = policy.evaluate(obs)
    log_prob = policy.get_actions_log_prob(actions)
    actor_hidden, critic_hidden = policy.get_hidden_states()

    assert policy.is_recurrent is True
    assert policy.state_dim == 42
    assert policy.map_dim == 2184
    assert actions.shape == (num_envs, 8)
    assert policy.action_mean.shape == (num_envs, 8)
    assert policy.action_std.shape == (num_envs, 8)
    assert log_prob.shape == (num_envs,)
    assert values.shape == (num_envs, 1)
    assert actor_hidden.shape == (1, num_envs, 256)
    assert critic_hidden.shape == (1, num_envs, 256)


def test_action_prior_heading_deadband_zeroes_closeout_yaw_target() -> None:
    algorithm = object.__new__(RangerTeacherRegularizedPPO)
    algorithm.large_heading_action_prior_heading_deadband = 0.04
    algorithm.large_heading_action_prior_heading_scale = 0.40
    algorithm.large_heading_action_prior_yaw_rate_reference = 0.35

    heading = torch.tensor([-0.03, 0.0, 0.03, 0.20])
    effective_heading, desired_yaw_rate = algorithm._prior_heading_targets(heading)

    assert torch.equal(effective_heading[:3], torch.zeros(3))
    assert torch.equal(desired_yaw_rate[:3], torch.zeros(3))
    assert effective_heading[3] > 0.0
    assert desired_yaw_rate[3] > 0.0


def test_hidden_goal_residual_preserves_visible_actions_and_changes_only_hidden_wheels() -> None:
    base = _make_policy(num_envs=2)
    obs = _make_obs((2,))
    residual = RangerTerrainActorCriticRecurrent(
        obs=obs,
        obs_groups=OBS_GROUPS,
        num_actions=8,
        init_noise_std=0.1,
        initial_action_std=[0.01] * 8,
        use_hidden_goal_residual=True,
    )
    residual.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        residual.actor.residual_head.bias[4:] = 0.5
    obs["policy_state"][:, 26] = torch.tensor([0.0, 1.0])

    base_actions = base.act_inference(obs)
    residual_actions = residual.act_inference(obs)

    assert torch.equal(residual_actions[0], base_actions[0])
    assert torch.equal(residual_actions[1, :4], base_actions[1, :4])
    assert not torch.equal(residual_actions[1, 4:], base_actions[1, 4:])


def test_recurrent_teacher_targets_use_matching_rollout_hidden_state() -> None:
    teacher = _make_policy(num_envs=2)
    with torch.no_grad():
        teacher.actor.suspension_head[-1].weight.normal_(std=0.05)
        teacher.actor.wheel_head[-1].weight.normal_(std=0.05)

    obs = _make_obs((4, 2))
    masks = torch.ones((4, 2), dtype=torch.bool)
    hidden_state = torch.randn(1, 2, NETWORK_SPEC.rnn_hidden_dim)
    actor_obs = teacher.actor_obs_normalizer(teacher.get_actor_obs(obs))
    with torch.inference_mode():
        raw_mean = teacher._apply_action_output_mask(teacher.actor(actor_obs, masks, hidden_state))
        expected = torch.tanh(teacher._bound_latent_mean(raw_mean))

    algorithm = types.SimpleNamespace(teacher=teacher)
    targets = RangerTeacherRegularizedPPO._teacher_targets(
        algorithm,
        obs,
        masks,
        actor_hidden_state=hidden_state,
    )
    zero_state_targets = RangerTeacherRegularizedPPO._teacher_targets(
        algorithm,
        obs,
        masks,
        actor_hidden_state=torch.zeros_like(hidden_state),
    )

    assert torch.allclose(targets, expected)
    assert not torch.allclose(targets, zero_state_targets)


def test_network_structure_has_one_derived_dimension_source() -> None:
    policy = _make_policy(num_envs=2, privileged_dim=19)

    assert NETWORK_SPEC.prop_dim == 34
    assert NETWORK_SPEC.goal_dim == 8
    assert NETWORK_SPEC.map_dim == 2184
    assert NETWORK_SPEC.actor_fusion_dim == 384
    assert NETWORK_SPEC.critic_fusion_dim == 448
    assert NETWORK_SPEC.num_actions == 8

    assert policy.actor.fusion_norm.normalized_shape == (NETWORK_SPEC.actor_fusion_dim,)
    assert policy.critic.fusion_norm.normalized_shape == (NETWORK_SPEC.critic_fusion_dim,)
    assert policy.actor.prop_encoder[-1].out_features == NETWORK_SPEC.prop_latent_dim
    assert policy.actor.goal_encoder[-1].out_features == NETWORK_SPEC.goal_latent_dim
    assert policy.actor.actor_trunk[-1].out_features == NETWORK_SPEC.actor_trunk_output_dim
    assert policy.actor.suspension_head[0].in_features == NETWORK_SPEC.actor_trunk_output_dim
    assert policy.actor.wheel_head[0].in_features == NETWORK_SPEC.actor_trunk_output_dim
    assert policy.actor.wheel_control_residual[0].in_features == NETWORK_SPEC.goal_dim + 1


def test_policy_yaw_rate_is_restored_to_physical_units_for_control_prior() -> None:
    fake_algorithm = SimpleNamespace(policy=SimpleNamespace(network_spec=NETWORK_SPEC))
    normalized_yaw_rate = torch.tensor([-0.20, 0.0, 0.20])

    physical_yaw_rate = RangerTeacherRegularizedPPO._physical_yaw_rate_from_policy_state(
        fake_algorithm, normalized_yaw_rate
    )

    assert NETWORK_SPEC.base_ang_vel_observation_scale == pytest.approx(3.0)
    assert torch.allclose(physical_yaw_rate, torch.tensor([-0.60, 0.0, 0.60]))


def test_legacy_goal_residual_migrates_into_unified_wheel_control_residual() -> None:
    source = _make_policy(num_envs=2)
    with torch.no_grad():
        for parameter in source.actor.wheel_control_residual.parameters():
            parameter.uniform_(-0.2, 0.2)
        source.actor.wheel_control_residual[0].weight[:, -1].zero_()

    source_state = source.state_dict()
    legacy_state = {
        key: value.clone()
        for key, value in source_state.items()
        if not key.startswith("actor.wheel_control_residual.")
    }
    legacy_prefix = "actor.wheel_goal_residual."
    unified_prefix = "actor.wheel_control_residual."
    for key, value in source_state.items():
        if key.startswith(unified_prefix):
            suffix = key[len(unified_prefix) :]
            legacy_key = f"{legacy_prefix}{suffix}"
            legacy_state[legacy_key] = value[:, :-1].clone() if suffix == "0.weight" else value.clone()

    restored = _make_policy(num_envs=2)
    restored.load_state_dict(legacy_state, strict=True)
    restored_state = restored.state_dict()

    restored_first = restored_state[f"{unified_prefix}0.weight"]
    legacy_first = legacy_state[f"{legacy_prefix}0.weight"]
    heading_rate_slot = NETWORK_SPEC.goal_heading_rate_slot
    copied_columns = [index for index in range(NETWORK_SPEC.goal_dim) if index != heading_rate_slot]
    assert torch.allclose(restored_first[:, copied_columns], legacy_first[:, copied_columns])
    assert torch.count_nonzero(restored_first[:, heading_rate_slot]) == 0
    assert torch.count_nonzero(restored_first[:, -1]) == 0
    for suffix in ("0.bias", "2.weight", "2.bias"):
        assert torch.allclose(restored_state[f"{unified_prefix}{suffix}"], legacy_state[f"{legacy_prefix}{suffix}"])


def test_rsl_rl_hidden_dim_compatibility_fields_cannot_redefine_network() -> None:
    obs = _make_obs((2,))
    with pytest.raises(ValueError, match="actor_hidden_dims"):
        RangerTerrainActorCriticRecurrent(
            obs=obs,
            obs_groups=OBS_GROUPS,
            num_actions=8,
            actor_hidden_dims=[256, 128],
        )
    with pytest.raises(ValueError, match="critic_hidden_dims"):
        RangerTerrainActorCriticRecurrent(
            obs=obs,
            obs_groups=OBS_GROUPS,
            num_actions=8,
            critic_hidden_dims=[256, 128],
        )


def test_distillation_loss_masks_stop_phase_wheels_and_supports_feature_loss() -> None:
    teacher_actions = torch.zeros(2, 8)
    student_actions = torch.zeros(2, 8)
    student_actions[:, 4:] = 1.0
    stop_phase_mask = torch.tensor([False, True])
    teacher_features = torch.zeros(2, 128)
    student_features = torch.ones(2, 128)

    losses = compute_distillation_losses(
        student_actions,
        teacher_actions,
        stop_phase_mask,
        stop_phase_wheel_weight=0.0,
        student_features=student_features,
        teacher_features=teacher_features,
        feature_loss_weight=0.05,
    )

    assert torch.isclose(losses["suspension_mse"], torch.tensor(0.0))
    assert torch.isclose(losses["wheel_mse"], torch.tensor(1.0))
    assert torch.isclose(losses["wheel_mse_effective"], torch.tensor(0.5))
    assert torch.isclose(losses["feature_mse"], torch.tensor(1.0))
    assert torch.isclose(losses["stop_phase_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["loss"], torch.tensor(0.55))


def test_teacher_wheel_weights_relax_near_goal_and_disable_in_stop_phase() -> None:
    weights = teacher_wheel_sample_weights(
        torch.tensor([False, False, True, True]),
        torch.tensor([False, True, False, True]),
        stop_phase_weight=0.0,
        near_goal_weight=0.1,
    )

    assert torch.equal(weights, torch.tensor([1.0, 0.1, 0.0, 0.0]))


def test_teacher_wheel_weights_require_matching_masks() -> None:
    with pytest.raises(ValueError, match="same shape"):
        teacher_wheel_sample_weights(
            torch.tensor([False, True]),
            torch.tensor([True]),
            stop_phase_weight=0.0,
            near_goal_weight=0.1,
        )


def test_near_goal_brake_common_loss_scales_only_forward_mode() -> None:
    teacher = torch.zeros(3, 8)
    teacher[:, 4:8] = torch.tensor([-0.4, -0.6, 0.8, 1.0])
    student = teacher.clone().requires_grad_(True)
    distance = torch.tensor([1.8, 1.15, 0.5])
    stop_phase = torch.tensor([False, False, True])

    loss, active_fraction, error_abs, target_abs = near_goal_brake_common_loss(
        student,
        teacher,
        distance,
        stop_phase,
        stop_distance=0.5,
        full_distance=1.8,
    )

    # Teacher common mode is 0.7; desired scales are 1, 0.5, and 0.
    expected_error = torch.tensor([0.0, 0.35, 0.7])
    assert torch.isclose(loss, expected_error.square().mean())
    assert torch.isclose(active_fraction, torch.tensor(1.0))
    assert torch.isclose(error_abs, expected_error.mean())
    assert torch.isclose(target_abs, torch.tensor([0.7, 0.35, 0.0]).mean())
    loss.backward()
    assert student.grad is not None
    assert torch.allclose(student.grad[:, :4], torch.zeros_like(student.grad[:, :4]))


def test_ranger_wheel_semantic_modes_distinguish_forward_and_turn() -> None:
    raw_wheel = torch.tensor(
        [
            [-0.6, -0.6, 0.6, 0.6],
            [0.4, 0.4, 0.4, 0.4],
        ]
    )

    semantic = ranger_wheel_raw_to_semantic(raw_wheel)
    common, turn = ranger_wheel_semantic_modes(raw_wheel)

    assert torch.equal(semantic[0], torch.full((4,), 0.6))
    assert torch.equal(semantic[1], torch.tensor([-0.4, -0.4, 0.4, 0.4]))
    assert torch.allclose(common, torch.tensor([0.6, 0.0]))
    assert torch.allclose(turn, torch.tensor([0.0, 0.4]))


def test_near_goal_brake_common_loss_ignores_far_samples() -> None:
    actions = torch.ones(2, 8, requires_grad=True)
    loss, active_fraction, _, _ = near_goal_brake_common_loss(
        actions,
        torch.zeros_like(actions),
        torch.tensor([2.0, 3.0]),
        torch.tensor([False, False]),
        stop_distance=0.5,
        full_distance=1.8,
    )

    assert loss.item() == 0.0
    assert active_fraction.item() == 0.0


def test_distillation_bounds_legacy_teacher_actions_and_reports_semantic_wheel_error() -> None:
    bounded = bound_teacher_actions(torch.tensor([[-4.0, -1.0, 0.5, 3.0]]))
    assert torch.equal(bounded, torch.tensor([[-1.0, -1.0, 0.5, 1.0]]))

    teacher_actions = torch.zeros(1, 8)
    teacher_actions[:, 6:8] = 1.0
    student_actions = torch.zeros(1, 8)
    student_actions[:, 4:6] = -1.0
    losses = compute_distillation_losses(
        student_actions,
        teacher_actions,
        torch.tensor([False]),
        suspension_loss_weight=0.0,
        wheel_loss_weight=0.0,
        wheel_common_loss_weight=1.0,
        wheel_turn_loss_weight=2.0,
    )

    assert torch.isclose(losses["wheel_common_mse"], torch.tensor(0.0))
    assert torch.isclose(losses["wheel_turn_mse"], torch.tensor(1.0))
    assert torch.isclose(losses["wrong_turn_sign_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["loss"], torch.tensor(2.0))


def test_sequence_replay_anchor_samples_contiguous_time_major_batches() -> None:
    replay = SequenceReplayAnchor(
        num_envs=2,
        sequence_length=3,
        capacity_sequences=4,
        observation_keys=("policy_state", "policy_map"),
        seed=11,
    )
    for step in range(3):
        replay.append_step(
            {
                "policy_state": torch.full((2, 42), float(step)),
                "policy_map": torch.full((2, 8 * 21 * 13), float(step)),
            },
            torch.full((2, 8), float(step)),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )

    assert len(replay) == 2
    batch = replay.sample(batch_size=2, device="cpu")
    assert batch["policy_state"].shape == (3, 2, 42)
    assert batch["policy_map"].shape == (3, 2, 8 * 21 * 13)
    assert batch["teacher_actions"].shape == (3, 2, 8)
    assert batch["goal_hidden_mask"].shape == (3, 2)
    assert torch.equal(batch["policy_state"][:, 0, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_asymmetric_teacher_observations_restore_hidden_goal_without_mutating_student() -> None:
    obs = _make_obs((2,))
    original_state = obs["policy_state"].clone()
    teacher_command = original_state[:, 26:34].clone()
    obs["teacher_command"] = teacher_command
    obs["policy_state"][:, 26] = 1.0
    obs["policy_state"][:, 29:34] = 0.0

    hidden = goal_hidden_mask(obs)
    teacher_obs = build_teacher_observations(obs)

    assert torch.equal(hidden, torch.ones(2, dtype=torch.bool))
    assert torch.equal(teacher_obs["policy_state"][:, 26:34], teacher_command)
    assert torch.equal(obs["policy_state"][:, 29:34], torch.zeros(2, 5))
    assert teacher_obs["policy_map"].data_ptr() == obs["policy_map"].data_ptr()


def test_memory_replay_sampling_requires_visible_context_before_hidden_loss() -> None:
    replay = SequenceReplayAnchor(
        num_envs=1,
        sequence_length=4,
        capacity_sequences=4,
        observation_keys=("policy_state", "policy_map"),
        seed=5,
    )
    hidden_pattern = (False, False, True, True, True, True, True, True)
    for step, hidden in enumerate(hidden_pattern):
        replay.append_step(
            {
                "policy_state": torch.full((1, 42), float(step)),
                "policy_map": torch.zeros(1, 8 * 21 * 13),
            },
            torch.zeros(1, 8),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            goal_hidden_mask=torch.tensor([hidden]),
        )

    batch = replay.sample(
        batch_size=3,
        device="cpu",
        require_goal_transition=True,
        burn_in_steps=1,
    )
    assert torch.equal(batch["policy_state"][:, :, 0], torch.tensor([[0.0], [1.0], [2.0], [3.0]]).expand(-1, 3))


def test_hidden_goal_sample_weights_emphasize_hidden_steps_with_stable_mean() -> None:
    hidden = torch.tensor([[False, True], [False, False]])
    weights = goal_hidden_sample_weights(hidden, hidden_weight=7.0)

    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0, 1] > weights[0, 0]


def test_replay_sequence_forward_does_not_modify_online_hidden_state() -> None:
    policy = _make_policy(num_envs=2)
    replay_batch = {
        "policy_state": torch.randn(4, 2, 42),
        "policy_map": torch.randn(4, 2, 8 * 21 * 13),
    }
    actor_hidden_before, _ = policy.get_hidden_states()
    actions = recurrent_sequence_actions(policy, replay_batch)
    actor_hidden_after, _ = policy.get_hidden_states()

    assert actor_hidden_before is None
    assert actor_hidden_after is None
    assert actions.shape == (4, 2, 8)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


def test_distillation_rollout_action_selection_supports_smooth_teacher_blend() -> None:
    teacher_actions = torch.full((2, 8), 4.0)
    student_actions = torch.full((2, 8), -2.0, requires_grad=True)

    teacher_rollout = select_rollout_actions(
        teacher_actions, student_actions, rollout_mode="teacher_rollout", teacher_action_blend=0.75
    )
    student_rollout = select_rollout_actions(
        teacher_actions, student_actions, rollout_mode="student_rollout", teacher_action_blend=0.75
    )
    blended_rollout = select_rollout_actions(
        teacher_actions, student_actions, rollout_mode="blended_rollout", teacher_action_blend=0.75
    )

    assert torch.equal(teacher_rollout, teacher_actions)
    assert torch.equal(student_rollout, student_actions.detach())
    assert torch.allclose(blended_rollout, torch.full((2, 8), 2.5))
    assert blended_rollout.requires_grad is False
    with pytest.raises(ValueError, match="teacher_action_blend"):
        select_rollout_actions(
            teacher_actions, student_actions, rollout_mode="blended_rollout", teacher_action_blend=1.1
        )


def test_distillation_optimizer_restore_preserves_moments_and_current_lr(tmp_path) -> None:
    source_model = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=3.0e-4)
    source_loss = source_model(torch.randn(5, 3)).square().mean()
    source_loss.backward()
    source_optimizer.step()

    checkpoint_path = tmp_path / "distilled.pt"
    torch.save({"distillation_optimizer_state_dict": source_optimizer.state_dict()}, checkpoint_path)

    target_model = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=9.0e-4)
    restored = restore_distillation_optimizer(
        checkpoint_path,
        target_optimizer,
        learning_rate=1.0e-4,
        map_location="cpu",
    )

    assert restored is True
    assert all(group["lr"] == pytest.approx(1.0e-4) for group in target_optimizer.param_groups)
    source_state = source_optimizer.state_dict()["state"]
    target_state = target_optimizer.state_dict()["state"]
    assert source_state.keys() == target_state.keys()
    for parameter_id in source_state:
        assert source_state[parameter_id].keys() == target_state[parameter_id].keys()
        for state_name, source_value in source_state[parameter_id].items():
            target_value = target_state[parameter_id][state_name]
            if isinstance(source_value, torch.Tensor):
                assert torch.equal(source_value, target_value), state_name
            else:
                assert source_value == target_value


def test_recurrent_v10_warm_start_copies_only_compatible_modules(tmp_path) -> None:
    feedforward = _make_feedforward_policy(num_envs=2, privileged_dim=19)
    recurrent = _make_policy(num_envs=2, privileged_dim=19)

    with torch.no_grad():
        for module_index, module_name in enumerate(RECURRENT_V10_COMPATIBLE_MODULES, start=1):
            for parameter in getattr(feedforward.actor, module_name).parameters():
                parameter.fill_(0.01 * module_index)

    def clone_state(module):
        return {key: value.detach().clone() for key, value in module.state_dict().items()}

    recurrent_new_before = {
        module_name: clone_state(getattr(recurrent.actor, module_name))
        for module_name in RECURRENT_NEW_ACTOR_MODULES
    }
    recurrent_critic_before = clone_state(recurrent.critic)
    recurrent_std_before = recurrent.std.detach().clone()

    checkpoint_path = tmp_path / "feedforward_v10.pt"
    torch.save(
        {
            "model_state_dict": feedforward.state_dict(),
            "optimizer_state_dict": {"sentinel": "feedforward-only"},
            "iter": 99,
        },
        checkpoint_path,
    )
    runner = types.SimpleNamespace(
        alg=types.SimpleNamespace(policy=recurrent),
        current_learning_iteration=123,
    )

    warm_start_ranger_recurrent_from_feedforward(runner, checkpoint_path)

    for module_name in RECURRENT_V10_COMPATIBLE_MODULES:
        source_state = getattr(feedforward.actor, module_name).state_dict()
        target_state = getattr(recurrent.actor, module_name).state_dict()
        assert source_state.keys() == target_state.keys()
        for key in source_state:
            assert torch.equal(target_state[key], source_state[key]), f"{module_name}.{key}"

    for module_name, before_state in recurrent_new_before.items():
        after_state = getattr(recurrent.actor, module_name).state_dict()
        assert before_state.keys() == after_state.keys()
        for key in before_state:
            assert torch.equal(after_state[key], before_state[key]), f"{module_name}.{key}"

    recurrent_critic_after = recurrent.critic.state_dict()
    for key in recurrent_critic_before:
        assert torch.equal(recurrent_critic_after[key], recurrent_critic_before[key]), f"critic.{key}"
    assert torch.equal(recurrent.std, recurrent_std_before)
    assert all(parameter.requires_grad for parameter in recurrent.parameters())
    assert runner.current_learning_iteration == 0


def test_v10_compatible_actor_modules_keep_exact_state_dict_shapes() -> None:
    feedforward = _make_feedforward_policy(num_envs=2, privileged_dim=19)
    recurrent = _make_policy(num_envs=2, privileged_dim=19)

    for module_name in ("map_encoder", "actor_trunk", "suspension_head", "wheel_head"):
        feedforward_state = getattr(feedforward.actor, module_name).state_dict()
        recurrent_state = getattr(recurrent.actor, module_name).state_dict()

        assert feedforward_state.keys() == recurrent_state.keys(), module_name
        assert {key: tuple(value.shape) for key, value in feedforward_state.items()} == {
            key: tuple(value.shape) for key, value in recurrent_state.items()
        }, module_name


def test_partial_hidden_reset_clears_only_done_envs() -> None:
    num_envs = 4
    policy = _make_policy(num_envs=num_envs)
    obs = _make_obs((num_envs,))

    policy.act_inference(obs)
    policy.evaluate(obs)
    actor_before, critic_before = policy.get_hidden_states()
    actor_before = actor_before.clone()
    critic_before = critic_before.clone()

    dones = torch.tensor([False, True, False, True])
    policy.reset(dones)
    actor_after, critic_after = policy.get_hidden_states()

    assert torch.equal(actor_after[:, dones, :], torch.zeros_like(actor_after[:, dones, :]))
    assert torch.equal(critic_after[:, dones, :], torch.zeros_like(critic_after[:, dones, :]))
    assert torch.equal(actor_after[:, ~dones, :], actor_before[:, ~dones, :])
    assert torch.equal(critic_after[:, ~dones, :], critic_before[:, ~dones, :])


def test_distillation_partial_hidden_reset_preserves_other_env_autograd_paths() -> None:
    num_envs = 4
    policy = _make_policy(num_envs=num_envs)
    policy.train()
    obs = _make_obs((num_envs,))

    policy.act_inference(obs)
    actor_before, _ = policy.get_hidden_states()
    assert actor_before is not None
    assert actor_before.grad_fn is not None

    dones = torch.tensor([False, True, False, True])
    reset_max = reset_recurrent_memory(policy, dones)
    actor_after, _ = policy.get_hidden_states()

    assert reset_max == 0.0
    assert torch.equal(actor_after[:, dones, :], torch.zeros_like(actor_after[:, dones, :]))
    assert torch.equal(actor_after[:, ~dones, :], actor_before[:, ~dones, :])

    hidden_grad = torch.autograd.grad(actor_after.sum(), actor_before, retain_graph=True)[0]
    assert torch.equal(hidden_grad[:, dones, :], torch.zeros_like(hidden_grad[:, dones, :]))
    assert torch.equal(hidden_grad[:, ~dones, :], torch.ones_like(hidden_grad[:, ~dones, :]))


def test_sequence_and_step_outputs_match_with_native_layout() -> None:
    torch.manual_seed(11)
    time_steps = 4
    num_envs = 3
    policy = _make_policy(num_envs=num_envs)
    obs_seq = _make_obs((time_steps, num_envs))

    # The Phase-1 actor initializes final heads to zero for V10-compatible warm-starts.
    # Randomize them here so this consistency test exercises the recurrent path itself.
    with torch.no_grad():
        for head in (policy.actor.suspension_head[-1], policy.actor.wheel_head[-1]):
            head.weight.normal_(mean=0.0, std=0.05)
            head.bias.normal_(mean=0.0, std=0.05)

    masks = torch.ones(time_steps, num_envs, dtype=torch.bool)
    initial_actor_hidden = torch.zeros(1, num_envs, 256)
    initial_critic_hidden = torch.zeros(1, num_envs, 256)

    policy.act(obs_seq, masks=masks, hidden_state=initial_actor_hidden)
    sequence_mean = policy.action_mean.detach()
    sequence_values = policy.evaluate(obs_seq, masks=masks, hidden_state=initial_critic_hidden).detach()

    policy.reset()
    step_means = []
    step_values = []
    for step in range(time_steps):
        step_obs = obs_seq[step]
        step_means.append(policy.act_inference(step_obs).detach())
        step_values.append(policy.evaluate(step_obs).detach())

    assert sequence_mean.shape == (time_steps, num_envs, 8)
    assert sequence_values.shape == (time_steps, num_envs, 1)
    assert torch.allclose(torch.tanh(sequence_mean), torch.stack(step_means), atol=1.0e-5, rtol=1.0e-5)
    assert torch.allclose(sequence_values, torch.stack(step_values), atol=1.0e-5, rtol=1.0e-5)


def test_advantage_diagnostics_preserve_singleton_recurrent_minibatch_axis() -> None:
    time_steps = 6
    state = torch.zeros(time_steps, 1, 42)
    goal_start = int(NETWORK_SPEC.goal_state_range[0])
    state[..., goal_start + 3] = 1.0
    state[..., goal_start + 4] = 0.2
    obs = TensorDict({"policy_state": state}, batch_size=(time_steps, 1))
    actions = torch.zeros(time_steps, 1, 8)
    actions[..., 4:6] = -0.2
    actions[..., 6:8] = 0.2
    advantages = torch.linspace(-1.0, 1.0, time_steps).reshape(time_steps, 1, 1)
    masks = torch.ones(time_steps, 1, dtype=torch.bool)

    algorithm = object.__new__(RangerTeacherRegularizedPPO)
    algorithm.policy = SimpleNamespace(state_obs_group="policy_state", network_spec=NETWORK_SPEC)
    result = algorithm._advantage_diagnostics(obs, actions, advantages, masks)

    assert result["fraction_goal_left"] == pytest.approx(1.0)
    assert result["advantage_goal_left"] == pytest.approx(float(advantages.mean()))


def test_action_masks_match_feedforward_semantics() -> None:
    policy = RangerTerrainActorCriticRecurrent(
        obs=_make_obs((2,)),
        obs_groups=OBS_GROUPS,
        num_actions=8,
        init_noise_std=0.5,
        action_training_mask=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        action_output_mask=[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        action_exploration_mask=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        inactive_action_std=1.0e-6,
    )
    obs = _make_obs((2,))
    with torch.no_grad():
        for head in (policy.actor.suspension_head[-1], policy.actor.wheel_head[-1]):
            head.weight.normal_(mean=0.0, std=0.05)
            head.bias.normal_(mean=0.0, std=0.05)
    actions = policy.act(obs)
    log_prob = policy.get_actions_log_prob(actions)

    assert torch.equal(policy.action_mean[:, [2, 3, 6, 7]], torch.zeros_like(policy.action_mean[:, [2, 3, 6, 7]]))
    assert torch.any(policy.action_mean[:, [0, 1, 4, 5]].abs() > 0.0)
    assert torch.allclose(policy.action_std[:, [1, 3, 5, 7]], torch.full((2, 4), 1.0e-6))
    assert log_prob.shape == (2,)
