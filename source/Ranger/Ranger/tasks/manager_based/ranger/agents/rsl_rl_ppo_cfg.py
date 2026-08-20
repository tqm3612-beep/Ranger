# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RangerTerrainActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "RangerTerrainActorCritic"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list[int] = [256, 128]
    critic_hidden_dims: list[int] = [256, 128]
    activation: str = "elu"

    terrain_obs_group: str = "policy_map"
    state_obs_group: str = "policy_state"
    privileged_obs_group: str = "critic_privileged"
    terrain_channels: int = 8
    terrain_grid_shape: tuple[int, int] = (21, 13)
    terrain_cnn_channels: list[int] = [16, 32, 32]
    terrain_latent_dim: int = 128
    state_hidden_dims: list[int] = [128]
    state_latent_dim: int = 128
    privileged_hidden_dims: list[int] = [64]
    wheel_head_hidden_dims: list[int] = [128]
    suspension_head_hidden_dims: list[int] = [128]
    action_training_mask: list[float] | None = None
    action_output_mask: list[float] | None = None
    action_exploration_mask: list[float] | None = None
    initial_action_std: list[float] | None = None
    inactive_action_std: float = 1.0e-6


@configclass
class RangerRecurrentActorCriticCfg(RslRlPpoActorCriticCfg):
    """RSL-RL config surface for the fixed Ranger recurrent architecture."""

    class_name: str = "RangerTerrainActorCriticRecurrent"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    # Required by the generic RSL-RL actor-critic config schema. These are
    # intentionally empty: internal recurrent network widths live only in
    # _RangerRecurrentNetworkSpec and non-empty values are rejected by policy.
    actor_hidden_dims: list[int] = []
    critic_hidden_dims: list[int] = []
    activation: str = "elu"

    action_training_mask: list[float] | None = None
    action_output_mask: list[float] | None = None
    action_exploration_mask: list[float] | None = None
    initial_action_std: list[float] | None = None
    inactive_action_std: float = 1.0e-6


@configclass
class RangerTeacherRegularizedPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration with optional frozen V10 action regularization."""

    class_name: str = "RangerTeacherRegularizedPPO"
    teacher_loss_coef: float = 0.0
    teacher_checkpoint: str | None = None
    teacher_suspension_loss_weight: float = 1.0
    teacher_wheel_loss_weight: float = 1.0
    teacher_stop_phase_wheel_weight: float = 0.0
    actor_loss_scale: float = 1.0
    ppo_surrogate_scale: float = 1.0
    actor_learning_rate: float | None = None
    actor_train_scope: str = "all"
    large_heading_action_prior_coef: float = 0.0
    large_heading_action_prior_start: float = 0.70
    large_heading_action_prior_full: float = 1.05
    large_heading_action_prior_common_target: float = 0.45
    large_heading_action_prior_turn_gain: float = 0.55
    critic_only: bool = False
    diagnostic_only: bool = False
    critic_relearning: bool = False
    critic_relearning_return_lam: float = 1.0
    critic_relearning_common_delta: float = 0.25
    critic_relearning_turn_scale: float = 0.35
    critic_relearning_perturb_period: int = 48
    critic_relearning_perturb_burst: int = 12


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "ranger_direct"
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy_state", "policy_map"],
        "critic": ["policy_state", "policy_map", "critic_privileged"],
    }
    policy = RangerTerrainActorCriticCfg(
        init_noise_std=1.0,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class StandPPORunnerCfg(PPORunnerCfg):
    policy = RangerTerrainActorCriticCfg(
        init_noise_std=0.5,
    )


@configclass
class ShortGoalFlatPPORunnerCfg(PPORunnerCfg):
    """悬架车轮完全参与"""
    policy = RangerTerrainActorCriticCfg(
        init_noise_std=0.5,
    )

    def __post_init__(self) -> None:
        post_init = getattr(super(), "__post_init__", None)
        if post_init is not None:
            post_init()
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 1.0e-3


@configclass
class ShortGoalFlatV2PPORunnerCfg(ShortGoalFlatPPORunnerCfg):
    """Wheel-only PPO loss while preserving the Ranger 8-D actor/checkpoint layout."""
    """悬架置0,不参与训练和探索，不输出动作，车轮正常"""

    policy = RangerTerrainActorCriticCfg(
        init_noise_std=0.5,
        action_training_mask=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        action_output_mask=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        action_exploration_mask=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        initial_action_std=[0.05, 0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 0.25],
        inactive_action_std=1.0e-6,
    )


@configclass
class ShortGoalFlatFrozenSuspensionWheelPPORunnerCfg(ShortGoalFlatPPORunnerCfg):
    """Execute the preserved suspension policy deterministically and train wheel actions only."""

    policy = RangerTerrainActorCriticCfg(
        init_noise_std=0.10,
        action_training_mask=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_exploration_mask=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        initial_action_std=[0.04, 0.04, 0.04, 0.04, 0.10, 0.10, 0.10, 0.10],
        inactive_action_std=1.0e-6,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 5.0e-4


@configclass
class ShortGoalFlatLimitedSuspensionPPORunnerCfg(ShortGoalFlatPPORunnerCfg):
    """Train only the suspension dimensions while executing a deterministic wheel policy."""
    """只训练悬架，车轮执行确定性策略，只执行5%悬架动作"""

    policy = RangerTerrainActorCriticCfg(
        init_noise_std=0.05,
        action_training_mask=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_exploration_mask=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        initial_action_std=[0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10],
        inactive_action_std=1.0e-6,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 5.0e-4


@configclass
class ShortGoalFlatLimitedJointPPORunnerCfg(ShortGoalFlatPPORunnerCfg):
    """Low-noise limited joint fine-tuning of both Ranger action heads."""
    """悬架和车轮都输出动作并探索，悬架只执行5%动作"""

    policy = RangerTerrainActorCriticCfg(
        init_noise_std=0.10,
        action_training_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_exploration_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        initial_action_std=[0.04, 0.04, 0.04, 0.04, 0.10, 0.10, 0.10, 0.10],
        inactive_action_std=1.0e-6,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 7.5e-5
        self.algorithm.entropy_coef = 5.0e-4


@configclass
class ShortGoalFlatV7PPORunnerCfg(ShortGoalFlatPPORunnerCfg):
    """Stage V7: low-noise full-authority fine-tuning of both Ranger action heads."""

    max_iterations = 150
    save_interval = 25
    policy = RangerTerrainActorCriticCfg(
        init_noise_std=0.05,
        action_training_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_exploration_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        initial_action_std=[0.005, 0.005, 0.005, 0.005, 0.05, 0.05, 0.05, 0.05],
        inactive_action_std=1.0e-6,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 1.0e-4


@configclass
class ShortGoalFlatV8PPORunnerCfg(ShortGoalFlatV7PPORunnerCfg):
    """Stage V8: fine-tune both action heads against the balanced-stop success gate."""

    max_iterations = 500
    save_interval = 25


@configclass
class ShortGoalFlatV9PPORunnerCfg(ShortGoalFlatV7PPORunnerCfg):
    """Stage V9: jointly fine-tune both action heads for policy-controlled balanced stopping."""

    max_iterations = 500
    save_interval = 25
    policy = RangerTerrainActorCriticCfg(
        init_noise_std=0.01,
        action_training_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_exploration_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        initial_action_std=[0.003, 0.003, 0.003, 0.003, 0.01, 0.01, 0.01, 0.01],
        inactive_action_std=1.0e-6,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.entropy_coef = 1.0e-4


@configclass
class ShortGoalFlatV10PPORunnerCfg(ShortGoalFlatV9PPORunnerCfg):
    """Stage V10: low-noise adaptation to 0.50 m capture with a precision bonus."""

    max_iterations = 300
    save_interval = 25


@configclass
class ShortGoalFlatCRecurrentPPORunnerCfg(ShortGoalFlatV10PPORunnerCfg):
    """C-stage recurrent PPO wiring with the unchanged V10 environment/reward behavior."""

    num_steps_per_env = 16
    policy = RangerRecurrentActorCriticCfg(
        init_noise_std=0.01,
        noise_std_type="log",
        action_training_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_exploration_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        initial_action_std=[0.003, 0.003, 0.003, 0.003, 0.01, 0.01, 0.01, 0.01],
        inactive_action_std=1.0e-6,
    )


@configclass
class ShortGoalFlatCRecurrentTeacherPPORunnerCfg(ShortGoalFlatCRecurrentPPORunnerCfg):
    """Recurrent PPO with a frozen V10 teacher used only during PPO updates."""

    max_iterations = 25
    save_interval = 5
    algorithm = RangerTeacherRegularizedPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1.0e-4,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        teacher_loss_coef=0.0,
        teacher_suspension_loss_weight=1.0,
        teacher_wheel_loss_weight=1.0,
        teacher_stop_phase_wheel_weight=0.0,
        critic_only=False,
    )


@configclass
class ShortGoalFlatV11PPORunnerCfg(ShortGoalFlatV10PPORunnerCfg):
    """Stage V11: jointly adapt both actor heads for policy-controlled braking."""

    max_iterations = 200
    save_interval = 25
