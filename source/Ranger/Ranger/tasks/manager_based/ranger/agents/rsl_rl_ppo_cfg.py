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
    trainable_modules: list[str] | None = None
    frozen_modules: list[str] | None = None
    actor_trainable_modules: list[str] | None = None
    actor_frozen_modules: list[str] | None = None
    critic_trainable_modules: list[str] | None = None
    critic_frozen_modules: list[str] | None = None
    use_recurrent_actor: bool = True
    use_gru_residual: bool = False
    residual_action_scale: float = 0.1
    use_hidden_goal_residual: bool = False
    hidden_goal_residual_scale: float = 1.0


@configclass
class RangerTeacherRegularizedPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration with optional frozen V10 action regularization."""

    class_name: str = "RangerTeacherRegularizedPPO"
    teacher_loss_coef: float = 0.0
    teacher_checkpoint: str | None = None
    latent_loss_coef: float = 0.0
    residual_loss_coef: float = 1.0
    anchor_loss_coef: float = 1.0
    teacher_suspension_loss_weight: float = 1.0
    teacher_wheel_loss_weight: float = 1.0
    teacher_stop_phase_wheel_weight: float = 0.0
    teacher_near_goal_distance: float | None = None
    teacher_near_goal_wheel_weight: float = 1.0
    teacher_goal_distance_unit: float = 1.0
    brake_common_loss_coef: float = 0.0
    brake_common_stop_distance: float = 0.5
    brake_common_full_distance: float = 1.8
    actor_loss_scale: float = 1.0
    ppo_surrogate_scale: float = 1.0
    actor_learning_rate: float | None = None
    actor_train_scope: str = "all"
    large_heading_action_prior_coef: float = 0.0
    large_heading_action_prior_start: float = 0.70
    large_heading_action_prior_full: float = 1.05
    large_heading_action_prior_common_target: float = 0.45
    large_heading_action_prior_taper_common: bool = False
    large_heading_action_prior_common_stop_distance: float = 0.45
    large_heading_action_prior_common_full_distance: float = 2.0
    large_heading_action_prior_turn_gain: float = 0.55
    large_heading_action_prior_yaw_damping: float = 0.0
    large_heading_action_prior_yaw_rate_reference: float = 0.35
    large_heading_action_prior_heading_deadband: float = 0.0
    large_heading_action_prior_heading_scale: float = 0.40
    large_heading_action_prior_use_heading_rate_feedback: bool = False
    large_heading_action_prior_heading_rate_gain: float = 0.60
    large_heading_action_prior_balance_turn_targets: bool = False
    large_heading_action_prior_same_side_coef: float = 0.0
    large_heading_action_prior_onset_boost: float = 0.0
    large_heading_action_prior_onset_heading_max: float = 0.60
    large_heading_action_prior_onset_indicator_slot: int | None = None
    large_heading_action_prior_use_full_wheel_targets: bool = False
    large_heading_action_prior_stop_phase_zero_target_weight: float = 0.0
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
    algorithm = RangerTeacherRegularizedPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1.0e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        teacher_loss_coef=0.0,
        latent_loss_coef=0.0,
    )
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
class PerceptionP0RecurrentPPORunnerCfg(ShortGoalFlatCRecurrentPPORunnerCfg):
    """P0 wiring for the clean V10-based perception branch using the J4 recurrent architecture."""

    pass


@configclass
class Stage2ControlTransferPPORunnerCfg(PerceptionP0RecurrentPPORunnerCfg):
    """Current perception + GRU student used by the Stage2 imitation pipeline."""

    pass


@configclass
class Stage2ObstacleP0PPORunnerCfg(Stage2ControlTransferPPORunnerCfg):
    """Conservative map-conditioned PPO adaptation for one flat-ground obstacle."""

    num_steps_per_env = 192
    # Starting from the P1.5 model_53 transfer checkpoint, the guided P0
    # policy peaks around model_90 and then begins to over-adapt the shared GRU.
    max_iterations = 40
    save_interval = 5
    experiment_name = "ranger_direct/stage2_obstacle_p0"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.clip_param = 0.10
        self.algorithm.entropy_coef = 5.0e-5
        self.algorithm.num_learning_epochs = 1
        self.algorithm.num_mini_batches = 1
        self.algorithm.learning_rate = 5.0e-6
        self.algorithm.schedule = "fixed"
        self.algorithm.teacher_loss_coef = 0.5
        self.algorithm.teacher_suspension_loss_weight = 1.0
        self.algorithm.teacher_wheel_loss_weight = 0.0
        self.algorithm.latent_loss_coef = 0.0
        self.algorithm.ppo_surrogate_scale = 0.20

        # The transferred control and recurrent state remain the anchor. Only
        # perception-to-control adaptation and wheel actions learn in Obstacle P0.
        self.policy.action_training_mask = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        self.policy.action_output_mask = [1.0] * 8
        self.policy.action_exploration_mask = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        self.policy.initial_action_std = [0.003, 0.003, 0.003, 0.003, 0.15, 0.15, 0.15, 0.15]
        self.policy.actor_trainable_modules = [
            "map_encoder",
            "fusion_norm",
            "memory",
            "actor_trunk",
            "wheel_head",
        ]
        self.policy.critic_trainable_modules = [
            "prop_encoder",
            "goal_encoder",
            "map_encoder",
            "privileged_encoder",
            "fusion_norm",
            "memory",
            "hidden_norm",
            "value_head",
        ]


@configclass
class Stage2ObstacleP05PPORunnerCfg(Stage2ObstacleP0PPORunnerCfg):
    """Near-goal repair while anchoring the proven model_90 obstacle policy."""

    max_iterations = 30
    save_interval = 5
    experiment_name = "ranger_direct/stage2_obstacle_p05"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 2.0e-6
        self.algorithm.ppo_surrogate_scale = 0.10
        self.algorithm.teacher_loss_coef = 1.0
        self.algorithm.teacher_suspension_loss_weight = 1.0
        self.algorithm.teacher_wheel_loss_weight = 1.0
        self.algorithm.teacher_near_goal_distance = 2.50
        self.algorithm.teacher_near_goal_wheel_weight = 0.0
        self.algorithm.teacher_stop_phase_wheel_weight = 0.0

        # The obstacle representation and recurrent controller are already
        # validated by the P0 map ablation. Repair only the final wheel mapping.
        self.policy.initial_action_std = [0.003, 0.003, 0.003, 0.003, 0.05, 0.05, 0.05, 0.05]
        self.policy.actor_trainable_modules = ["wheel_head"]


@configclass
class Stage2ObstacleP05ExitPPORunnerCfg(Stage2ObstacleP05PPORunnerCfg):
    """Recurrent post-obstacle heading recovery with the proven P0 policy as anchor."""

    max_iterations = 24
    save_interval = 4
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 2.0e-6
        self.algorithm.ppo_surrogate_scale = 0.15
        self.algorithm.teacher_loss_coef = 0.50
        self.algorithm.teacher_suspension_loss_weight = 1.0
        self.algorithm.teacher_wheel_loss_weight = 0.10
        self.algorithm.teacher_near_goal_distance = 2.5
        self.algorithm.teacher_near_goal_wheel_weight = 0.0
        self.algorithm.teacher_stop_phase_wheel_weight = 0.0
        self.algorithm.brake_common_loss_coef = 0.0
        self.policy.actor_trainable_modules = ["memory", "actor_trunk", "wheel_head"]


@configclass
class Stage2ObstacleP05ExitV2PPORunnerCfg(Stage2ObstacleP05ExitPPORunnerCfg):
    """Longer dynamic-exit curriculum covering realistic post-obstacle momentum."""

    max_iterations = 120
    save_interval = 10
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.5e-6
        self.algorithm.ppo_surrogate_scale = 0.20
        self.algorithm.teacher_loss_coef = 0.35
        self.algorithm.teacher_wheel_loss_weight = 0.05
        self.algorithm.large_heading_action_prior_coef = 0.03
        self.algorithm.large_heading_action_prior_start = 0.10
        self.algorithm.large_heading_action_prior_full = 0.70
        self.algorithm.large_heading_action_prior_common_target = 0.15
        self.algorithm.large_heading_action_prior_turn_gain = 0.65
        self.algorithm.large_heading_action_prior_yaw_damping = 0.40
        self.algorithm.large_heading_action_prior_yaw_rate_reference = 0.35
        self.policy.initial_action_std = [0.003, 0.003, 0.003, 0.003, 0.06, 0.06, 0.06, 0.06]


@configclass
class Stage2ObstacleP05ExitV2BootstrapPPORunnerCfg(Stage2ObstacleP05ExitV2PPORunnerCfg):
    """Short symmetric bootstrap that teaches turn onset and taper before PPO refinement."""

    max_iterations = 40
    save_interval = 5
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.num_learning_epochs = 3
        self.algorithm.num_mini_batches = 4
        self.algorithm.learning_rate = 3.0e-6
        self.algorithm.actor_learning_rate = 2.0e-6
        self.algorithm.actor_train_scope = "wheel_head"
        self.algorithm.ppo_surrogate_scale = 0.0
        self.algorithm.entropy_coef = 0.0
        self.algorithm.teacher_loss_coef = 0.20
        self.algorithm.teacher_wheel_loss_weight = 0.0
        self.algorithm.large_heading_action_prior_coef = 0.10
        self.algorithm.large_heading_action_prior_start = 0.05
        self.algorithm.large_heading_action_prior_full = 0.60
        self.algorithm.large_heading_action_prior_common_target = 0.12
        self.algorithm.large_heading_action_prior_turn_gain = 0.65
        self.algorithm.large_heading_action_prior_yaw_damping = 0.80
        self.algorithm.large_heading_action_prior_yaw_rate_reference = 0.35
        self.policy.initial_action_std = [0.003, 0.003, 0.003, 0.003, 0.04, 0.04, 0.04, 0.04]


@configclass
class Stage2ObstacleP05ExitV2WheelControlBootstrapPPORunnerCfg(Stage2ObstacleP05ExitV2BootstrapPPORunnerCfg):
    """B5A: repair wheel control while keeping shared recurrent features frozen.

    Keep the B4 heading-to-desired-yaw-rate prior unchanged so the effect of unfreezing the
    direct goal-to-wheel residual can be isolated. Only the two wheel-specific actor branches
    are trainable; encoders, GRU, shared actor trunk, and suspension head remain frozen.
    """

    max_iterations = 20
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b5"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_learning_rate = 1.0e-6
        self.algorithm.actor_train_scope = "wheel_control"
        # Suspension is frozen and wheel teacher weight is zero in this repair stage, so
        # keeping the teacher loss enabled only adds checkpoint/loading overhead without
        # providing gradients to the trainable wheel-specific parameters.
        self.algorithm.teacher_loss_coef = 0.0


@configclass
class Stage2ObstacleP05ExitV2CloseoutBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2WheelControlBootstrapPPORunnerCfg
):
    """B5B: reduce heading overshoot with deadbanded, more strongly damped yaw tracking."""

    max_iterations = 20
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b5_closeout"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.large_heading_action_prior_heading_deadband = 0.04
        self.algorithm.large_heading_action_prior_heading_scale = 0.40
        self.algorithm.large_heading_action_prior_turn_gain = 0.45
        self.algorithm.large_heading_action_prior_yaw_damping = 1.20


@configclass
class Stage2ObstacleP05ExitV2HeadingRateResidualBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2CloseoutBootstrapPPORunnerCfg
):
    """B6: learn a zero-initialized wheel-only heading/yaw-rate feedback correction."""

    max_iterations = 20
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b6_rate_residual"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_learning_rate = 2.0e-6
        self.algorithm.actor_train_scope = "wheel_control_residual"


@configclass
class Stage2ObstacleP05ExitV2BalancedRateResidualBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2HeadingRateResidualBootstrapPPORunnerCfg
):
    """B7: prevent one-sided rollout collapse from dominating the wheel-control prior."""

    max_iterations = 20
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b7_balanced_rate_residual"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.large_heading_action_prior_balance_turn_targets = True


@configclass
class Stage2ObstacleP05ExitV2FastBalancedRateResidualBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2BalancedRateResidualBootstrapPPORunnerCfg
):
    """B8: use a supervised-scale learning rate for the isolated zero-init wheel adapter."""

    max_iterations = 12
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b8_fast_rate_residual"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_learning_rate = 1.0e-4


@configclass
class Stage2ObstacleP05ExitV2UnifiedWheelResidualBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2BalancedRateResidualBootstrapPPORunnerCfg
):
    """B9: train one unified goal+yaw-rate wheel-control residual from model108."""

    max_iterations = 30
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b9_unified_wheel_residual"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_learning_rate = 2.0e-5
        self.algorithm.actor_train_scope = "wheel_control_residual"


@configclass
class Stage2ObstacleP05ExitV2TaperedCommonResidualBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2UnifiedWheelResidualBootstrapPPORunnerCfg
):
    """B11: taper supervised forward common-mode near the point goal and at large heading error."""

    max_iterations = 30
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b11_tapered_common_residual"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.large_heading_action_prior_taper_common = True
        self.algorithm.large_heading_action_prior_common_stop_distance = 0.45
        self.algorithm.large_heading_action_prior_common_full_distance = 2.0


@configclass
class Stage2ObstacleP05ExitV2HeadingRateFeedbackBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2TaperedCommonResidualBootstrapPPORunnerCfg
):
    """B12: use true point-goal heading-error-rate feedback for wheel-turn closeout."""

    max_iterations = 30
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b12_heading_rate_feedback"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.large_heading_action_prior_use_heading_rate_feedback = True
        self.algorithm.large_heading_action_prior_heading_rate_gain = 0.60


@configclass
class Stage2ObstacleP05ExitV2FixedRolloutIntensiveBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2HeadingRateFeedbackBootstrapPPORunnerCfg
):
    """B13: fit the B12 wheel residual intensively on one fixed rollout.

    One runner iteration collects exactly one rollout. Increasing the PPO update
    schedule from B12's 3 x 4 = 12 minibatch updates to 30 x 4 = 120 therefore
    reuses the same rollout throughout this diagnostic instead of repeatedly
    recollecting from the drifting on-policy state distribution.

    Keep the B12 actor learning rate, action prior, trainable scope, and disabled
    PPO actor objective unchanged so this experiment isolates update count only.
    """

    max_iterations = 1
    save_interval = 1
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b13_fixed_rollout_intensive"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.num_learning_epochs = 30
        self.algorithm.num_mini_batches = 4


@configclass
class Stage2ObstacleP05ExitV2FixedRolloutFastLRBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2FixedRolloutIntensiveBootstrapPPORunnerCfg
):
    """B14: isolate whether B13's supervised residual fitting is learning-rate limited."""

    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b14_fixed_rollout_fast_lr"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_learning_rate = 1.0e-4


@configclass
class Stage2ObstacleP05ExitV2OnsetNullspaceBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2FixedRolloutFastLRBootstrapPPORunnerCfg
):
    """B15: cancel inherited steering bias early while closing the four-wheel prior nullspace."""

    max_iterations = 6
    save_interval = 1
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b15_onset_nullspace"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.large_heading_action_prior_same_side_coef = 0.50
        self.algorithm.large_heading_action_prior_onset_boost = 2.0
        self.algorithm.large_heading_action_prior_onset_heading_max = 0.60


@configclass
class Stage2ObstacleP05ExitV2ResetOnsetNullspaceBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2OnsetNullspaceBootstrapPPORunnerCfg
):
    """B17: prioritize the first two seconds after reset using a loss-only onset indicator."""

    max_iterations = 6
    save_interval = 1
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b17_reset_onset_nullspace"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.large_heading_action_prior_onset_indicator_slot = 0
        self.algorithm.large_heading_action_prior_onset_boost = 8.0


@configclass
class Stage2ObstacleP05ExitV2FullWheelBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2ResetOnsetNullspaceBootstrapPPORunnerCfg
):
    """B18: supervise all four semantic wheel actions directly to eliminate the wheel nullspace."""

    max_iterations = 8
    save_interval = 1
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b18_full_wheel"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.large_heading_action_prior_use_full_wheel_targets = True
        self.algorithm.large_heading_action_prior_same_side_coef = 0.0
        self.algorithm.large_heading_action_prior_onset_boost = 2.0
        self.algorithm.actor_learning_rate = 1.0e-5
        self.algorithm.actor_train_scope = "wheel_control_residual"
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4


@configclass
class Stage2ObstacleP05ExitV2FullWheelHeadIntensiveBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2FullWheelBootstrapPPORunnerCfg
):
    """B20: intensively fit only the state-dependent wheel head to full-wheel targets on one rollout."""

    max_iterations = 1
    save_interval = 1
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b20_full_wheel_head_intensive"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_train_scope = "wheel_head"
        self.algorithm.actor_learning_rate = 1.0e-4
        self.algorithm.num_learning_epochs = 30
        self.algorithm.num_mini_batches = 4


@configclass
class Stage2ObstacleP05ExitV2SemanticFullWheelHeadBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2FullWheelBootstrapPPORunnerCfg
):
    """B21: recalibrate model108 wheel-head control in the canonical semantic wheel space."""

    max_iterations = 4
    save_interval = 1
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_b21_semantic_full_wheel_head"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_train_scope = "wheel_head"
        self.algorithm.actor_learning_rate = 2.0e-5
        self.algorithm.num_learning_epochs = 10
        self.algorithm.num_mini_batches = 4
        self.algorithm.large_heading_action_prior_use_full_wheel_targets = True
        self.algorithm.large_heading_action_prior_same_side_coef = 0.0
        self.algorithm.large_heading_action_prior_onset_boost = 2.0


@configclass
class Stage2ObstacleP05ExitV2SemanticTerminalFullWheelHeadPPORunnerCfg(
    Stage2ObstacleP05ExitV2SemanticFullWheelHeadBootstrapPPORunnerCfg
):
    """B22: keep semantic full-wheel navigation supervision active and learn zero-wheel terminal braking."""

    max_iterations = 4
    save_interval = 1
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_b22_semantic_terminal_full_wheel_head"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_learning_rate = 1.0e-5
        self.algorithm.large_heading_action_prior_stop_phase_zero_target_weight = 1.0


@configclass
class Stage2ObstacleP05SemanticStopFullWheelHeadPPORunnerCfg(
    Stage2ObstacleP05ExitV2SemanticTerminalFullWheelHeadPPORunnerCfg
):
    """B23: terminal-heavy calibration for autonomous zero-wheel stopping and same-side consistency."""

    max_iterations = 4
    save_interval = 1
    experiment_name = "ranger_direct/stage2_obstacle_p05_b23_semantic_stop_full_wheel_head"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_learning_rate = 1.0e-5
        self.algorithm.large_heading_action_prior_stop_phase_zero_target_weight = 1.0


@configclass
class Stage2ObstacleP05SemanticStopCaptureFullWheelHeadPPORunnerCfg(
    Stage2ObstacleP05SemanticStopFullWheelHeadPPORunnerCfg
):
    """B24: concentrated stop-latch calibration around 0.30-0.70 m."""

    max_iterations = 4
    save_interval = 1
    experiment_name = "ranger_direct/stage2_obstacle_p05_b24_semantic_stop_capture_full_wheel_head"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_learning_rate = 1.0e-5
        self.algorithm.large_heading_action_prior_stop_phase_zero_target_weight = 1.0


@configclass
class Stage2ObstacleP05SemanticFullV1PPORunnerCfg(
    Stage2ObstacleP05ExitV2SemanticTerminalFullWheelHeadPPORunnerCfg
):
    """Full-route V1 with weak navigation supervision and protected autonomous stopping."""

    max_iterations = 24
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v1"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_train_scope = "wheel_control"
        self.algorithm.actor_learning_rate = 2.0e-6
        self.algorithm.learning_rate = 1.5e-6
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
        self.algorithm.ppo_surrogate_scale = 0.05
        self.algorithm.entropy_coef = 0.0
        self.algorithm.teacher_loss_coef = 0.0

        # Effective navigation prior coefficient is 0.10 * 0.10 = 0.01,
        # while the latched terminal zero-wheel target retains coefficient 0.10.
        self.algorithm.large_heading_action_prior_coef = 0.10
        self.algorithm.large_heading_action_prior_non_stop_weight = 0.10
        self.algorithm.large_heading_action_prior_stop_phase_zero_target_weight = 1.0


@configclass
class Stage2ObstacleP05SemanticFullV2PPORunnerCfg(Stage2ObstacleP05SemanticFullV1PPORunnerCfg):
    """Full-route V2: PPO-only map-conditioned navigation with protected terminal stopping."""

    max_iterations = 24
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v2"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_train_scope = "wheel_head"
        self.algorithm.ppo_surrogate_scale = 0.20
        self.algorithm.actor_learning_rate = 2.0e-6
        self.algorithm.large_heading_action_prior_non_stop_weight = 0.0
        self.algorithm.large_heading_action_prior_stop_phase_zero_target_weight = 1.0


@configclass
class Stage2ObstacleP05SemanticFullV3PPORunnerCfg(Stage2ObstacleP05SemanticFullV2PPORunnerCfg):
    """Full-route V3: adapt perception map encoder and wheel head while protecting recurrent/goal control."""

    max_iterations = 24
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v3"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_train_scope = "map_wheel_head"
        self.algorithm.actor_learning_rate = 1.0e-6


@configclass
class Stage2ObstacleP05SemanticFullV4WaypointPPORunnerCfg(Stage2ObstacleP05SemanticFullV1PPORunnerCfg):
    """Waypoint-conditioned integration while preserving the learned point-goal controller."""

    max_iterations = 24
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v4_waypoint"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_train_scope = "wheel_head"
        self.algorithm.actor_learning_rate = 1.0e-6
        self.algorithm.ppo_surrogate_scale = 0.05
        self.algorithm.large_heading_action_prior_coef = 0.10
        self.algorithm.large_heading_action_prior_non_stop_weight = 0.50
        self.algorithm.large_heading_action_prior_stop_phase_zero_target_weight = 1.0


@configclass
class Stage2ObstacleP05SemanticFullV5RouteFramePPORunnerCfg(Stage2ObstacleP05SemanticFullV4WaypointPPORunnerCfg):
    """Generalized navigation PPO with joint perception/recurrent/control adaptation."""

    max_iterations = 80
    save_interval = 5
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v5_route_frame"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_train_scope = "navigation_stack"
        self.algorithm.actor_learning_rate = 1.5e-6
        self.algorithm.learning_rate = 1.5e-6
        self.algorithm.ppo_surrogate_scale = 0.10
        self.algorithm.num_learning_epochs = 1
        self.algorithm.num_mini_batches = 1
        self.algorithm.entropy_coef = 5.0e-5
        # Waypoints are now a curriculum hint rather than the primary objective.
        self.algorithm.large_heading_action_prior_coef = 0.05
        self.algorithm.large_heading_action_prior_non_stop_weight = 0.20
        self.algorithm.large_heading_action_prior_stop_phase_zero_target_weight = 1.0


@configclass
class Stage2ObstacleP05SemanticFullV6PhaseGatedRewardPPORunnerCfg(
    Stage2ObstacleP05SemanticFullV5RouteFramePPORunnerCfg
):
    """V6 reward-semantic experiment; keep V5 navigation-stack optimization unchanged for isolation."""

    max_iterations = 100
    save_interval = 5
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v6_phase_gated_reward"


@configclass
class Stage2ObstacleP05SemanticFullV7SafetyFirstRoutePPORunnerCfg(
    Stage2ObstacleP05SemanticFullV6PhaseGatedRewardPPORunnerCfg
):
    """Safety-first full-route acquisition with no direct-goal non-stop actor prior."""

    max_iterations = 100
    save_interval = 5
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v7_safety_first_route"

    def __post_init__(self) -> None:
        super().__post_init__()
        # During route phases the policy command is the active waypoint, while future
        # no-teacher navigation may expose the final goal. Avoid an actor-side prior that
        # can re-introduce direct-goal pressure outside RewardManager. Keep terminal zero-wheel
        # supervision active through the dedicated stop-phase weight.
        self.algorithm.large_heading_action_prior_non_stop_weight = 0.0
        self.algorithm.large_heading_action_prior_stop_phase_zero_target_weight = 1.0


@configclass
class Stage2ObstacleP05SemanticFullV8TurnAwareRoutePPORunnerCfg(
    Stage2ObstacleP05SemanticFullV7SafetyFirstRoutePPORunnerCfg
):
    """Turn-aware route acquisition experiment retaining the V7 navigation-stack train scope."""

    max_iterations = 100
    save_interval = 5
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v8_turn_aware_route"


@configclass
class Stage2ObstacleP05SemanticFullV9FastRoutePPORunnerCfg(
    Stage2ObstacleP05SemanticFullV8TurnAwareRoutePPORunnerCfg
):
    """Faster route-speed experiment retaining the V8 navigation-stack optimization scope."""

    max_iterations = 100
    save_interval = 5
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v9_fast_route"


@configclass
class Stage2ObstacleP05SemanticFullV10CorridorRoutePPORunnerCfg(
    Stage2ObstacleP05SemanticFullV9FastRoutePPORunnerCfg
):
    """Corridor-route objective cleanup retaining the navigation-stack optimization scope."""

    max_iterations = 100
    save_interval = 5
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v10_corridor_route"


@configclass
class Stage2ObstacleP05SemanticFullV10EntryRelaxedPPORunnerCfg(
    Stage2ObstacleP05SemanticFullV10CorridorRoutePPORunnerCfg
):
    """Entry-transition relaxation experiment retaining all V10 PPO settings."""

    max_iterations = 60
    save_interval = 5
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v10_entry_relaxed"


@configclass
class Stage2ObstacleP05SemanticFullV10SafeCorridorPPORunnerCfg(
    Stage2ObstacleP05SemanticFullV10EntryRelaxedPPORunnerCfg
):
    """Safe-corridor geometry and aligned exit-control experiment."""

    max_iterations = 60
    save_interval = 5
    experiment_name = "ranger_direct/stage2_obstacle_p05_semantic_full_v10_safe_corridor"


@configclass
class Stage2ObstacleP05ExitV2FullWheelControlBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2FullWheelBootstrapPPORunnerCfg
):
    """B19: full-wheel supervision while adapting the wheel head and wheel residual together."""

    max_iterations = 8
    save_interval = 1
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b19_full_wheel_control"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_train_scope = "wheel_control"
        self.algorithm.actor_learning_rate = 1.0e-5


@configclass
class Stage2ObstacleP05ExitV2OnsetNullspaceWheelControlBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2OnsetNullspaceBootstrapPPORunnerCfg
):
    """B16: unfreeze the final wheel head plus residual after nullspace repair."""

    max_iterations = 6
    save_interval = 1
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b16_onset_nullspace_wheel_control"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_learning_rate = 2.0e-5
        self.algorithm.actor_train_scope = "wheel_control"
        self.algorithm.num_learning_epochs = 10
        self.algorithm.num_mini_batches = 4


@configclass
class Stage2ObstacleP05ExitV2UnifiedWheelControlBootstrapPPORunnerCfg(
    Stage2ObstacleP05ExitV2UnifiedWheelResidualBootstrapPPORunnerCfg
):
    """B10: train wheel head plus unified physical-yaw residual while shared recurrent features stay frozen."""

    max_iterations = 30
    save_interval = 2
    experiment_name = "ranger_direct/stage2_obstacle_p05_exit_v2_bootstrap_b10_unified_wheel_control"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.actor_learning_rate = 1.0e-5
        self.algorithm.actor_train_scope = "wheel_control"


@configclass
class Stage2ObstacleP05StopPPORunnerCfg(Stage2ObstacleP05ExitPPORunnerCfg):
    """Near-goal recurrent braking lesson without a hard wheel-action override."""

    max_iterations = 24
    experiment_name = "ranger_direct/stage2_obstacle_p05_stop"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.5e-6
        self.algorithm.ppo_surrogate_scale = 0.20
        self.algorithm.teacher_loss_coef = 0.50
        self.algorithm.teacher_wheel_loss_weight = 0.0
        self.algorithm.brake_common_loss_coef = 0.0
        self.policy.initial_action_std = [0.003, 0.003, 0.003, 0.003, 0.04, 0.04, 0.04, 0.04]


@configclass
class Stage2ObstacleP05FullPPORunnerCfg(Stage2ObstacleP05ExitPPORunnerCfg):
    """Mixed-distribution consolidation with a frozen recurrent sequence anchor."""

    max_iterations = 36
    save_interval = 4
    experiment_name = "ranger_direct/stage2_obstacle_p05_full"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-6
        self.algorithm.ppo_surrogate_scale = 0.10
        self.algorithm.teacher_loss_coef = 0.75
        self.algorithm.teacher_suspension_loss_weight = 1.0
        self.algorithm.teacher_wheel_loss_weight = 0.35
        self.algorithm.teacher_near_goal_distance = 2.5
        self.algorithm.teacher_near_goal_wheel_weight = 0.0
        self.algorithm.teacher_stop_phase_wheel_weight = 0.0
        self.policy.initial_action_std = [0.003, 0.003, 0.003, 0.003, 0.035, 0.035, 0.035, 0.035]


@configclass
class Stage2MemoryM0PPORunnerCfg(Stage2ControlTransferPPORunnerCfg):
    """M0 memory probe using the proven Stage2 recurrent policy unchanged."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.policy.use_hidden_goal_residual = True
        self.policy.hidden_goal_residual_scale = 0.25
        self.policy.actor_trainable_modules = ["residual_head"]


@configclass
class Stage2MemoryM1PPORunnerCfg(Stage2MemoryM0PPORunnerCfg):
    """One-second memory probe runner."""

    pass


@configclass
class Stage2TerrainP0PPORunnerCfg(Stage2ControlTransferPPORunnerCfg):
    """Protected main-control adaptation on gentle slopes."""

    num_steps_per_env = 192
    max_iterations = 200
    save_interval = 5
    experiment_name = "ranger_direct/stage2_terrain_p0"
    algorithm = RangerTeacherRegularizedPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=5.0e-5,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=1.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        teacher_loss_coef=1.0,
        teacher_suspension_loss_weight=0.0,
        teacher_wheel_loss_weight=1.0,
        teacher_stop_phase_wheel_weight=0.0,
        latent_loss_coef=0.0,
        ppo_surrogate_scale=0.05,
    )
    policy = RangerRecurrentActorCriticCfg(
        init_noise_std=0.01,
        noise_std_type="log",
        action_training_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_exploration_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        initial_action_std=[0.003, 0.003, 0.003, 0.003, 0.01, 0.01, 0.01, 0.01],
        inactive_action_std=1.0e-6,
        use_hidden_goal_residual=True,
        hidden_goal_residual_scale=0.25,
        actor_trainable_modules=[
            "map_encoder",
            "actor_trunk",
            "suspension_head",
            "wheel_head",
        ],
        critic_trainable_modules=[
            "prop_encoder",
            "goal_encoder",
            "map_encoder",
            "privileged_encoder",
            "fusion_norm",
            "memory",
            "hidden_norm",
            "value_head",
        ],
    )


@configclass
class Stage2TerrainP1PPORunnerCfg(Stage2TerrainP0PPORunnerCfg):
    """Protected perception/control adaptation for moderate slopes and stairs."""

    max_iterations = 300
    experiment_name = "ranger_direct/stage2_terrain_p1"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-6
        self.algorithm.teacher_loss_coef = 1.0
        self.policy.actor_trainable_modules = [
            "map_encoder",
            "fusion_norm",
            "actor_trunk",
            "suspension_head",
            "wheel_head",
        ]


@configclass
class Stage2TerrainP15BrakeAssistPPORunnerCfg(Stage2TerrainP1PPORunnerCfg):
    """Protected near-goal braking adaptation while the legacy stop override remains active."""

    max_iterations = 80
    save_interval = 5
    experiment_name = "ranger_direct/stage2_terrain_p15_brake_assist"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.entropy_coef = 0.0
        self.algorithm.ppo_surrogate_scale = 0.0
        self.algorithm.teacher_near_goal_distance = 1.8
        self.algorithm.teacher_near_goal_wheel_weight = 0.0
        self.algorithm.teacher_stop_phase_wheel_weight = 0.0
        self.algorithm.brake_common_loss_coef = 2.0
        self.algorithm.brake_common_stop_distance = 0.5
        self.algorithm.brake_common_full_distance = 1.8
        # Keep perception, posture, and steering features fixed during the explicit braking lesson.
        self.policy.actor_trainable_modules = ["wheel_head"]


@configclass
class Stage2TerrainP15PPORunnerCfg(Stage2TerrainP15BrakeAssistPPORunnerCfg):
    """Autonomous near-goal braking after removing the environment wheel override."""

    max_iterations = 120
    experiment_name = "ranger_direct/stage2_terrain_p15"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 5.0e-7
        self.algorithm.ppo_surrogate_scale = 0.05
        self.algorithm.brake_common_loss_coef = 0.5
        self.policy.actor_trainable_modules = ["wheel_head"]


@configclass
class WaveAdaptationPPORunnerCfg(ShortGoalFlatCRecurrentPPORunnerCfg):
    """Fine-tune the J4 recurrent policy on wave terrain while freezing goal encoders."""

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
        teacher_loss_coef=1.0,
        teacher_suspension_loss_weight=1.0,
        teacher_wheel_loss_weight=1.0,
        teacher_stop_phase_wheel_weight=0.0,
    )

    policy = RangerRecurrentActorCriticCfg(
        init_noise_std=0.01,
        noise_std_type="log",
        action_training_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        action_exploration_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        initial_action_std=[0.003, 0.003, 0.003, 0.003, 0.01, 0.01, 0.01, 0.01],
        inactive_action_std=1.0e-6,
        trainable_modules=[
            "prop_encoder",
            "map_encoder",
            "fusion_norm",
            "memory",
            "hidden_norm",
            "actor_trunk",
            "suspension_head",
            "wheel_head",
            "wheel_control_residual",
        ],
        frozen_modules=["goal_encoder"],
        actor_trainable_modules=[
            "prop_encoder",
            "map_encoder",
            "fusion_norm",
            "memory",
            "hidden_norm",
            "actor_trunk",
            "suspension_head",
            "wheel_head",
            "wheel_control_residual",
        ],
        actor_frozen_modules=["goal_encoder"],
        critic_trainable_modules=[
            "prop_encoder",
            "map_encoder",
            "privileged_encoder",
            "fusion_norm",
            "memory",
            "hidden_norm",
            "value_head",
        ],
        critic_frozen_modules=["goal_encoder"],
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.entropy_coef = 1.0e-4


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
