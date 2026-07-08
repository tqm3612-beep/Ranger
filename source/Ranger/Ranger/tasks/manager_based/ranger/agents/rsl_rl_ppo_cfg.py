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
