# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, HiddenState, Memory
from rsl_rl.utils import unpad_trajectories

from .rsl_rl_custom_policy import _TerrainMapEncoder, _activation, _build_mlp


@dataclass(frozen=True)
class _RangerRecurrentNetworkSpec:
    """Single source of truth for the fixed Phase-1 recurrent network structure.

    Environment-facing dimensions are still read from the TensorDict/RSL-RL API and
    validated against this specification. Internal layer inputs are derived from the
    preceding module output dimensions declared here instead of repeating numeric
    literals throughout the actor and critic implementations.
    """

    state_dim: int = 42
    base_ang_vel_observation_scale: float = 3.0
    goal_heading_rate_slot: int = 1
    goal_heading_rate_observation_scale: float = 2.0
    prop_state_ranges: tuple[tuple[int, int], ...] = ((0, 26), (34, 42))
    goal_state_range: tuple[int, int] = (26, 34)

    prop_hidden_dims: tuple[int, ...] = (128,)
    prop_latent_dim: int = 128
    goal_hidden_dims: tuple[int, ...] = (128,)
    goal_latent_dim: int = 128

    terrain_channels: int = 8
    terrain_grid_shape: tuple[int, int] = (21, 13)
    terrain_cnn_channels: tuple[int, ...] = (16, 32, 32)
    terrain_latent_dim: int = 128

    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1

    actor_trunk_hidden_dims: tuple[int, ...] = (256,)
    actor_trunk_output_dim: int = 128
    suspension_head_hidden_dims: tuple[int, ...] = (128,)
    suspension_action_dim: int = 4
    wheel_head_hidden_dims: tuple[int, ...] = (128,)
    wheel_control_residual_hidden_dims: tuple[int, ...] = (64,)
    wheel_action_dim: int = 4

    privileged_hidden_dims: tuple[int, ...] = (64,)
    privileged_latent_dim: int = 64
    critic_value_hidden_dims: tuple[int, ...] = (256, 128)
    value_dim: int = 1

    @property
    def prop_dim(self) -> int:
        return sum(end - start for start, end in self.prop_state_ranges)

    @property
    def goal_dim(self) -> int:
        start, end = self.goal_state_range
        return end - start

    @property
    def map_dim(self) -> int:
        return self.terrain_channels * self.terrain_grid_shape[0] * self.terrain_grid_shape[1]

    @property
    def num_actions(self) -> int:
        return self.suspension_action_dim + self.wheel_action_dim

    @property
    def actor_fusion_dim(self) -> int:
        return self.prop_latent_dim + self.goal_latent_dim + self.terrain_latent_dim

    @property
    def critic_fusion_dim(self) -> int:
        return self.actor_fusion_dim + self.privileged_latent_dim


_RANGER_RECURRENT_NETWORK_SPEC = _RangerRecurrentNetworkSpec()


class _RangerRecurrentActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        map_dim: int,
        num_actions: int,
        activation: str,
        network_spec: _RangerRecurrentNetworkSpec,
        use_recurrent: bool = True,
        use_gru_residual: bool = False,
        residual_action_scale: float = 0.1,
        use_hidden_goal_residual: bool = False,
        hidden_goal_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if state_dim != network_spec.state_dim:
            raise ValueError(
                f"Ranger recurrent actor expects policy_state dim {network_spec.state_dim}, got {state_dim}."
            )
        if num_actions != network_spec.num_actions:
            raise ValueError(f"Ranger recurrent actor expects {network_spec.num_actions} actions, got {num_actions}.")

        self.network_spec = network_spec
        self.state_dim = int(state_dim)
        self.map_dim = int(map_dim)
        self.num_actions = int(num_actions)
        self.use_recurrent = bool(use_recurrent)
        self.use_gru_residual = bool(use_gru_residual)
        self.use_hidden_goal_residual = bool(use_hidden_goal_residual)
        if self.use_recurrent and self.use_gru_residual:
            raise ValueError("GRU residual mode is only valid for the aligned-latent old-trunk actor path.")
        if residual_action_scale < 0.0:
            raise ValueError(f"residual_action_scale must be non-negative, got {residual_action_scale}.")
        self.residual_action_scale = float(residual_action_scale)
        if hidden_goal_residual_scale < 0.0:
            raise ValueError(
                f"hidden_goal_residual_scale must be non-negative, got {hidden_goal_residual_scale}."
            )
        self.hidden_goal_residual_scale = float(hidden_goal_residual_scale)
        if self.map_dim != network_spec.map_dim:
            raise ValueError(
                f"Terrain map dim mismatch: expected {network_spec.map_dim} from "
                f"{network_spec.terrain_channels}x{network_spec.terrain_grid_shape}, got {self.map_dim}."
            )

        self.prop_encoder = _build_mlp(
            network_spec.prop_dim, list(network_spec.prop_hidden_dims), network_spec.prop_latent_dim, activation
        )
        self.goal_encoder = _build_mlp(
            network_spec.goal_dim, list(network_spec.goal_hidden_dims), network_spec.goal_latent_dim, activation
        )
        self.map_encoder = _TerrainMapEncoder(
            input_channels=network_spec.terrain_channels,
            grid_shape=network_spec.terrain_grid_shape,
            conv_channels=list(network_spec.terrain_cnn_channels),
            latent_dim=network_spec.terrain_latent_dim,
            activation=activation,
        )
        self.fusion_norm = nn.LayerNorm(network_spec.actor_fusion_dim)
        self.latent_alignment_projection = nn.Linear(network_spec.actor_fusion_dim, 128)
        self.memory = Memory(
            network_spec.actor_fusion_dim,
            network_spec.rnn_hidden_dim,
            network_spec.rnn_num_layers,
            network_spec.rnn_type,
        )
        self.hidden_norm = nn.LayerNorm(network_spec.rnn_hidden_dim)
        self.actor_trunk = _build_mlp(
            network_spec.rnn_hidden_dim,
            list(network_spec.actor_trunk_hidden_dims),
            network_spec.actor_trunk_output_dim,
            activation,
        )
        self.actor_trunk_activation = _activation(activation)
        self.suspension_head = _build_mlp(
            network_spec.actor_trunk_output_dim,
            list(network_spec.suspension_head_hidden_dims),
            network_spec.suspension_action_dim,
            activation,
        )
        self.wheel_head = _build_mlp(
            network_spec.actor_trunk_output_dim,
            list(network_spec.wheel_head_hidden_dims),
            network_spec.wheel_action_dim,
            activation,
        )
        self.wheel_control_residual = _build_mlp(
            network_spec.goal_dim + 1,
            list(network_spec.wheel_control_residual_hidden_dims),
            network_spec.wheel_action_dim,
            activation,
        )
        self.residual_memory = Memory(
            128,
            network_spec.rnn_hidden_dim,
            network_spec.rnn_num_layers,
            network_spec.rnn_type,
        )
        self.residual_hidden_norm = nn.LayerNorm(network_spec.rnn_hidden_dim)
        self.residual_head = nn.Linear(network_spec.rnn_hidden_dim, num_actions)

        nn.init.zeros_(self.suspension_head[-1].weight)
        nn.init.zeros_(self.suspension_head[-1].bias)
        nn.init.zeros_(self.wheel_head[-1].weight)
        nn.init.zeros_(self.wheel_head[-1].bias)
        nn.init.zeros_(self.wheel_control_residual[-1].weight)
        nn.init.zeros_(self.wheel_control_residual[-1].bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _split_encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_shape = obs.shape[:-1]
        flat_obs = obs.reshape(-1, obs.shape[-1])
        state_obs = flat_obs[:, : self.state_dim]
        map_obs = flat_obs[:, self.state_dim : self.state_dim + self.map_dim]
        prop_obs = torch.cat(
            tuple(state_obs[:, start:end] for start, end in self.network_spec.prop_state_ranges), dim=-1
        )
        goal_start, goal_end = self.network_spec.goal_state_range
        goal_obs = state_obs[:, goal_start:goal_end]
        goal_encoder_obs = goal_obs.clone()
        goal_encoder_obs[:, self.network_spec.goal_heading_rate_slot] = 0.0
        map_obs = map_obs.view(
            flat_obs.shape[0],
            self.network_spec.terrain_channels,
            self.network_spec.terrain_grid_shape[0],
            self.network_spec.terrain_grid_shape[1],
        )

        prop_latent = self.prop_encoder(prop_obs)
        goal_latent = self.goal_encoder(goal_encoder_obs)
        map_latent = self.map_encoder(map_obs)
        return (
            prop_latent.reshape(*prefix_shape, prop_latent.shape[-1]),
            goal_latent.reshape(*prefix_shape, goal_latent.shape[-1]),
            map_latent.reshape(*prefix_shape, map_latent.shape[-1]),
            goal_obs.reshape(*prefix_shape, goal_obs.shape[-1]),
        )

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        prefix_shape = obs.shape[:-1]
        prop_latent, goal_latent, map_latent, _ = self._split_encode(obs)
        encoded = torch.cat((prop_latent, goal_latent, map_latent), dim=-1)
        return self.fusion_norm(encoded).reshape(*prefix_shape, encoded.shape[-1])

    def alignment_latent(self, obs: torch.Tensor) -> torch.Tensor:
        return self.latent_alignment_projection(self._encode(obs))

    def _set_bypass_hidden_state(self, obs: torch.Tensor) -> None:
        batch_size = int(obs.shape[0])
        weight = self.latent_alignment_projection.weight
        target_memory = self.residual_memory if self.use_gru_residual else self.memory
        target_memory.hidden_state = weight.new_zeros(
            self.network_spec.rnn_num_layers,
            batch_size,
            self.network_spec.rnn_hidden_dim,
        )

    def _old_trunk_action(
        self,
        map_latent: torch.Tensor,
        state_latent: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trunk_input = torch.cat((map_latent, state_latent), dim=-1)
        if masks is not None:
            trunk_input = unpad_trajectories(trunk_input, masks)
        trunk = self.actor_trunk_activation(self.actor_trunk(trunk_input))
        return torch.cat((self.suspension_head(trunk), self.wheel_head(trunk)), dim=-1)

    def residual_delta(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        state_latent = self.alignment_latent(obs)
        out = self.residual_memory(state_latent, masks, hidden_state)
        if masks is None:
            out = out.squeeze(0)
        out = self.residual_hidden_norm(out)
        return self.residual_head(out)

    def base_action(self, obs: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        _, _, map_latent, _ = self._split_encode(obs)
        state_latent = self.alignment_latent(obs)
        return self._old_trunk_action(map_latent, state_latent, masks)

    def forward(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        if self.use_recurrent:
            fused = self._encode(obs)
            out = self.memory(fused, masks, hidden_state)
            if masks is None:
                out = out.squeeze(0)
            out = self.hidden_norm(out)
            trunk_input = out
            state_obs = obs[..., : self.state_dim]
            goal_start, goal_end = self.network_spec.goal_state_range
            goal_obs = state_obs[..., goal_start:goal_end]
            yaw_rate_obs = state_obs[..., 2:3] * self.network_spec.base_ang_vel_observation_scale
        else:
            _, _, map_latent, goal_obs = self._split_encode(obs)
            state_latent = self.alignment_latent(obs)
            if masks is None:
                self._set_bypass_hidden_state(obs)
            base_action = self._old_trunk_action(map_latent, state_latent, masks)
            if self.use_gru_residual:
                return base_action + self.residual_action_scale * self.residual_delta(obs, masks, hidden_state)
            return base_action

        trunk = self.actor_trunk_activation(self.actor_trunk(trunk_input))
        if masks is not None:
            goal_obs = unpad_trajectories(goal_obs, masks)
            yaw_rate_obs = unpad_trajectories(yaw_rate_obs, masks)
        wheel_output = self.wheel_head(trunk)
        if self.use_recurrent:
            wheel_feedback_obs = torch.cat((goal_obs, yaw_rate_obs), dim=-1)
            wheel_output = wheel_output + self.wheel_control_residual(wheel_feedback_obs)
        action = torch.cat((self.suspension_head(trunk), wheel_output), dim=-1)
        if self.use_hidden_goal_residual:
            hidden_indicator = goal_obs[..., 0:1].clamp(min=0.0, max=1.0)
            residual = self.residual_head(out)
            bounded_residual = torch.tanh(residual)
            wheel_residual = torch.cat(
                (torch.zeros_like(bounded_residual[..., :4]), bounded_residual[..., 4:]), dim=-1
            )
            action = action + self.hidden_goal_residual_scale * hidden_indicator * wheel_residual
        return action

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory.reset(dones)
        self.residual_memory.reset(dones)

    @property
    def hidden_state(self) -> HiddenState:
        if self.use_gru_residual:
            return self.residual_memory.hidden_state
        return self.memory.hidden_state


class _RangerRecurrentCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        map_dim: int,
        privileged_dim: int,
        activation: str,
        network_spec: _RangerRecurrentNetworkSpec,
    ) -> None:
        super().__init__()
        if state_dim != network_spec.state_dim:
            raise ValueError(
                f"Ranger recurrent critic expects policy_state dim {network_spec.state_dim}, got {state_dim}."
            )

        self.network_spec = network_spec
        self.state_dim = int(state_dim)
        self.map_dim = int(map_dim)
        self.privileged_dim = int(privileged_dim)
        if self.map_dim != network_spec.map_dim:
            raise ValueError(
                f"Terrain map dim mismatch: expected {network_spec.map_dim} from "
                f"{network_spec.terrain_channels}x{network_spec.terrain_grid_shape}, got {self.map_dim}."
            )

        self.prop_encoder = _build_mlp(
            network_spec.prop_dim, list(network_spec.prop_hidden_dims), network_spec.prop_latent_dim, activation
        )
        self.goal_encoder = _build_mlp(
            network_spec.goal_dim, list(network_spec.goal_hidden_dims), network_spec.goal_latent_dim, activation
        )
        self.map_encoder = _TerrainMapEncoder(
            input_channels=network_spec.terrain_channels,
            grid_shape=network_spec.terrain_grid_shape,
            conv_channels=list(network_spec.terrain_cnn_channels),
            latent_dim=network_spec.terrain_latent_dim,
            activation=activation,
        )
        self.privileged_encoder = _build_mlp(
            self.privileged_dim,
            list(network_spec.privileged_hidden_dims),
            network_spec.privileged_latent_dim,
            activation,
        )
        self.fusion_norm = nn.LayerNorm(network_spec.critic_fusion_dim)
        self.memory = Memory(
            network_spec.critic_fusion_dim,
            network_spec.rnn_hidden_dim,
            network_spec.rnn_num_layers,
            network_spec.rnn_type,
        )
        self.hidden_norm = nn.LayerNorm(network_spec.rnn_hidden_dim)
        self.value_head = _build_mlp(
            network_spec.rnn_hidden_dim,
            list(network_spec.critic_value_hidden_dims),
            network_spec.value_dim,
            activation,
        )

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        prefix_shape = obs.shape[:-1]
        flat_obs = obs.reshape(-1, obs.shape[-1])
        state_obs = flat_obs[:, : self.state_dim]
        map_start = self.state_dim
        map_end = map_start + self.map_dim
        map_obs = flat_obs[:, map_start:map_end]
        priv_obs = flat_obs[:, map_end : map_end + self.privileged_dim]
        prop_obs = torch.cat(
            tuple(state_obs[:, start:end] for start, end in self.network_spec.prop_state_ranges), dim=-1
        )
        goal_start, goal_end = self.network_spec.goal_state_range
        goal_obs = state_obs[:, goal_start:goal_end]
        goal_encoder_obs = goal_obs.clone()
        goal_encoder_obs[:, self.network_spec.goal_heading_rate_slot] = 0.0
        map_obs = map_obs.view(
            flat_obs.shape[0],
            self.network_spec.terrain_channels,
            self.network_spec.terrain_grid_shape[0],
            self.network_spec.terrain_grid_shape[1],
        )

        encoded = torch.cat(
            (
                self.prop_encoder(prop_obs),
                self.goal_encoder(goal_encoder_obs),
                self.map_encoder(map_obs),
                self.privileged_encoder(priv_obs),
            ),
            dim=-1,
        )
        return self.fusion_norm(encoded).reshape(*prefix_shape, encoded.shape[-1])

    def forward(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        fused = self._encode(obs)
        out = self.memory(fused, masks, hidden_state)
        if masks is None:
            out = out.squeeze(0)
        out = self.hidden_norm(out)
        return self.value_head(out)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory.reset(dones)

    @property
    def hidden_state(self) -> HiddenState:
        return self.memory.hidden_state


class RangerTerrainActorCriticRecurrent(nn.Module):
    """Ranger recurrent actor-critic following RSL-RL's native recurrent policy API.

    Current Isaac Lab env inspection found rsl-rl-lib 3.1.2 installed. Its recurrent
    PPO path calls act(obs, masks=None, hidden_state=None), evaluate(...), reset(dones),
    and get_hidden_states(). In rollout updates, obs is padded as [time, trajectories, dim],
    masks is [time, trajectories], and GRU hidden states are [layers, trajectories, hidden].
    """

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        activation: str = "elu",
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        terrain_obs_group: str = "policy_map",
        state_obs_group: str = "policy_state",
        privileged_obs_group: str = "critic_privileged",
        terrain_channels: int = _RANGER_RECURRENT_NETWORK_SPEC.terrain_channels,
        terrain_grid_shape: tuple[int, int] = _RANGER_RECURRENT_NETWORK_SPEC.terrain_grid_shape,
        terrain_cnn_channels: list[int] | tuple[int, ...] = _RANGER_RECURRENT_NETWORK_SPEC.terrain_cnn_channels,
        action_training_mask: list[float] | None = None,
        action_output_mask: list[float] | None = None,
        action_exploration_mask: list[float] | None = None,
        initial_action_std: list[float] | None = None,
        inactive_action_std: float = 1.0e-6,
        trainable_modules: list[str] | None = None,
        frozen_modules: list[str] | None = None,
        actor_trainable_modules: list[str] | None = None,
        actor_frozen_modules: list[str] | None = None,
        critic_trainable_modules: list[str] | None = None,
        critic_frozen_modules: list[str] | None = None,
        use_recurrent_actor: bool = True,
        use_gru_residual: bool = False,
        residual_action_scale: float = 0.1,
        use_hidden_goal_residual: bool = False,
        hidden_goal_residual_scale: float = 1.0,
        rnn_type: str = _RANGER_RECURRENT_NETWORK_SPEC.rnn_type,
        rnn_hidden_dim: int = _RANGER_RECURRENT_NETWORK_SPEC.rnn_hidden_dim,
        rnn_num_layers: int = _RANGER_RECURRENT_NETWORK_SPEC.rnn_num_layers,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "RangerTerrainActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        if state_dependent_std:
            raise NotImplementedError("RangerTerrainActorCriticRecurrent does not implement state-dependent std.")
        if actor_hidden_dims not in (None, []):
            raise ValueError(
                "actor_hidden_dims is an RSL-RL compatibility field only for the fixed recurrent architecture; "
                f"expected [] or None, got {actor_hidden_dims}."
            )
        if critic_hidden_dims not in (None, []):
            raise ValueError(
                "critic_hidden_dims is an RSL-RL compatibility field only for the fixed recurrent architecture; "
                f"expected [] or None, got {critic_hidden_dims}."
            )
        super().__init__()

        network_spec = _RANGER_RECURRENT_NETWORK_SPEC
        requested_architecture = {
            "terrain_channels": int(terrain_channels),
            "terrain_grid_shape": tuple(int(v) for v in terrain_grid_shape),
            "terrain_cnn_channels": tuple(int(v) for v in terrain_cnn_channels),
            "rnn_type": rnn_type.lower(),
            "rnn_hidden_dim": int(rnn_hidden_dim),
            "rnn_num_layers": int(rnn_num_layers),
        }
        fixed_architecture = {
            "terrain_channels": network_spec.terrain_channels,
            "terrain_grid_shape": network_spec.terrain_grid_shape,
            "terrain_cnn_channels": network_spec.terrain_cnn_channels,
            "rnn_type": network_spec.rnn_type,
            "rnn_hidden_dim": network_spec.rnn_hidden_dim,
            "rnn_num_layers": network_spec.rnn_num_layers,
        }
        if requested_architecture != fixed_architecture:
            raise ValueError(
                "RangerTerrainActorCriticRecurrent uses a fixed network specification in Phase 1. "
                f"Expected {fixed_architecture}, got {requested_architecture}."
            )

        self.obs_groups = obs_groups
        self.terrain_obs_group = terrain_obs_group
        self.state_obs_group = state_obs_group
        self.privileged_obs_group = privileged_obs_group
        self.state_dependent_std = state_dependent_std
        self.network_spec = network_spec
        self.use_recurrent_actor = bool(use_recurrent_actor)
        self.use_gru_residual = bool(use_gru_residual)
        if self.use_gru_residual and self.use_recurrent_actor:
            raise ValueError("use_gru_residual requires use_recurrent_actor=False.")
        if residual_action_scale < 0.0:
            raise ValueError(f"residual_action_scale must be non-negative, got {residual_action_scale}.")
        self.residual_action_scale = float(residual_action_scale)
        self.use_hidden_goal_residual = bool(use_hidden_goal_residual)
        self.hidden_goal_residual_scale = float(hidden_goal_residual_scale)
        self.rnn_hidden_dim = network_spec.rnn_hidden_dim
        self.rnn_num_layers = network_spec.rnn_num_layers
        if inactive_action_std <= 0.0:
            raise ValueError(f"inactive_action_std must be positive, got {inactive_action_std}.")

        def _binary_mask(name: str, values: list[float] | None, default: list[float]) -> torch.Tensor:
            mask_values = default if values is None else values
            if len(mask_values) != num_actions:
                raise ValueError(f"{name} must have {num_actions} entries, got {len(mask_values)}.")
            mask = torch.tensor(mask_values, dtype=torch.float32)
            if not torch.all((mask == 0.0) | (mask == 1.0)):
                raise ValueError(f"{name} entries must be exactly 0.0 or 1.0.")
            return mask

        action_training_mask_tensor = _binary_mask("action_training_mask", action_training_mask, [1.0] * num_actions)
        if not torch.any(action_training_mask_tensor > 0.0):
            raise ValueError("action_training_mask must keep at least one trainable action dimension.")
        action_output_mask_tensor = _binary_mask("action_output_mask", action_output_mask, [1.0] * num_actions)
        action_exploration_mask_tensor = _binary_mask(
            "action_exploration_mask", action_exploration_mask, action_training_mask_tensor.tolist()
        )
        if initial_action_std is None:
            initial_action_std_tensor = init_noise_std * torch.ones(num_actions, dtype=torch.float32)
        else:
            if len(initial_action_std) != num_actions:
                raise ValueError(f"initial_action_std must have {num_actions} entries, got {len(initial_action_std)}.")
            initial_action_std_tensor = torch.tensor(initial_action_std, dtype=torch.float32)
            if torch.any(initial_action_std_tensor <= 0.0):
                raise ValueError("initial_action_std entries must all be positive.")

        self.register_buffer("_action_training_mask", action_training_mask_tensor, persistent=False)
        self.register_buffer("_action_output_mask", action_output_mask_tensor, persistent=False)
        self.register_buffer("_action_exploration_mask", action_exploration_mask_tensor, persistent=False)
        self.register_buffer("_configured_initial_action_std", initial_action_std_tensor, persistent=False)
        self._inactive_action_std = float(inactive_action_std)

        self.num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        self.num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        self.state_dim = int(obs[self.state_obs_group].shape[-1])
        self.map_dim = int(obs[self.terrain_obs_group].shape[-1])
        self.privileged_dim = int(obs[self.privileged_obs_group].shape[-1])

        self.actor = _RangerRecurrentActor(
            state_dim=self.state_dim,
            map_dim=self.map_dim,
            num_actions=num_actions,
            activation=activation,
            network_spec=network_spec,
            use_recurrent=self.use_recurrent_actor,
            use_gru_residual=self.use_gru_residual,
            residual_action_scale=self.residual_action_scale,
            use_hidden_goal_residual=self.use_hidden_goal_residual,
            hidden_goal_residual_scale=self.hidden_goal_residual_scale,
        )
        self.critic = _RangerRecurrentCritic(
            state_dim=self.state_dim,
            map_dim=self.map_dim,
            privileged_dim=self.privileged_dim,
            activation=activation,
            network_spec=network_spec,
        )
        self._configure_module_trainability(
            self.actor,
            trainable_modules if actor_trainable_modules is None else actor_trainable_modules,
            frozen_modules if actor_frozen_modules is None else actor_frozen_modules,
            owner_name="actor",
        )
        self._configure_module_trainability(
            self.critic,
            critic_trainable_modules,
            frozen_modules if critic_frozen_modules is None else critic_frozen_modules,
            owner_name="critic",
        )

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(self.num_actor_obs) if actor_obs_normalization else torch.nn.Identity()
        )
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(self.num_critic_obs) if critic_obs_normalization else torch.nn.Identity()
        )

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(self._configured_initial_action_std.clone())
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(self._configured_initial_action_std.clone()))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'.")

        self.distribution = None
        # Keep the pre-tanh Gaussian mean in a numerically invertible range. Legacy
        # Ranger heads often emit magnitudes above 10; direct tanh would round those
        # actions to exactly +/-1 in float32, making atanh(action) unusable for PPO
        # likelihood ratios. A smooth cap at +/-5 leaves the executed command almost
        # unchanged while preserving a stable one-to-one tanh transform.
        self._latent_action_mean_limit = 5.0
        Normal.set_default_validate_args(False)

    @staticmethod
    def _configure_module_trainability(
        module_owner: nn.Module,
        trainable_modules: list[str] | None,
        frozen_modules: list[str] | None,
        owner_name: str,
    ) -> None:
        """Apply config-driven module freezing without changing checkpoint keys."""

        child_modules = dict(module_owner.named_children())
        valid_names = set(child_modules)

        def _validate_list(field_name: str, module_names: list[str] | None) -> set[str]:
            if module_names is None:
                return set()
            requested = set(module_names)
            unknown = sorted(requested - valid_names)
            if unknown:
                raise ValueError(
                    f"Unknown {owner_name} {field_name}: {unknown}. "
                    f"Valid modules are: {sorted(valid_names)}."
                )
            return requested

        trainable = _validate_list("trainable_modules", trainable_modules)
        frozen = _validate_list("frozen_modules", frozen_modules)

        if trainable_modules is not None:
            for name, child in child_modules.items():
                requires_grad = name in trainable
                for parameter in child.parameters():
                    parameter.requires_grad = requires_grad

        for name in frozen:
            for parameter in child_modules[name].parameters():
                parameter.requires_grad = False

    @property
    def action_mean(self) -> torch.Tensor:
        """Latent Gaussian mean used by RSL-RL storage and adaptive-KL bookkeeping."""

        return self.distribution.mean

    @property
    def bounded_action_mean(self) -> torch.Tensor:
        """Deterministic environment-space action corresponding to the latent Gaussian mean."""

        return torch.tanh(self.distribution.mean)

    @property
    def action_std(self) -> torch.Tensor:
        """Latent Gaussian standard deviation used by RSL-RL adaptive-KL bookkeeping."""

        return self.distribution.stddev

    @staticmethod
    def _tanh_log_det_jacobian(latent_action: torch.Tensor) -> torch.Tensor:
        """Stable ``log(1 - tanh(z)^2)`` for the tanh action transform."""

        return 2.0 * (math.log(2.0) - latent_action - F.softplus(-2.0 * latent_action))

    @property
    def entropy(self) -> torch.Tensor:
        """Monte-Carlo entropy estimate for the tanh-squashed Gaussian policy."""

        latent_sample = self.distribution.rsample()
        entropy = self.distribution.entropy() + self._tanh_log_det_jacobian(latent_sample)
        mask = self._action_training_mask.to(device=entropy.device, dtype=entropy.dtype)
        return (entropy * mask).sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.actor.reset(dones)
        self.critic.reset(dones)

    def reset_memory(self) -> None:
        """Reset recurrent memory state for inference-only recurrent policy usage."""
        self.actor.memory.hidden_state = None
        self.actor.residual_memory.hidden_state = None
        self.critic.memory.hidden_state = None

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def _apply_action_output_mask(self, mean: torch.Tensor) -> torch.Tensor:
        mask = self._action_output_mask.to(device=mean.device, dtype=mean.dtype)
        return mean * mask

    def _bound_latent_mean(self, raw_mean: torch.Tensor) -> torch.Tensor:
        limit = float(self._latent_action_mean_limit)
        return limit * torch.tanh(raw_mean / limit)

    def _update_distribution(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> None:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        raw_mean = self._apply_action_output_mask(self.actor(actor_obs, masks, hidden_state))
        mean = self._bound_latent_mean(raw_mean)
        exploration_mask = self._action_exploration_mask.to(device=mean.device, dtype=mean.dtype)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        std = std * exploration_mask + self._inactive_action_std * (1.0 - exploration_mask)
        self.distribution = Normal(mean, std)

    def act(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        self._update_distribution(obs, masks, hidden_state)
        latent_action = self.distribution.sample()
        return torch.tanh(latent_action)

    def act_inference(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, TensorDict):
            actor_obs = self.get_actor_obs(obs)
        else:
            actor_obs = obs
        actor_obs = self.actor_obs_normalizer(actor_obs)
        raw_mean = self._apply_action_output_mask(self.actor(actor_obs))
        latent_mean = self._bound_latent_mean(raw_mean)
        return torch.tanh(latent_mean)

    def evaluate(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs, masks, hidden_state)

    def alignment_latent(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor.alignment_latent(actor_obs)

    def residual_delta(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor.residual_delta(actor_obs)

    def base_action(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor.base_action(actor_obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return log-probability of bounded actions under the tanh-squashed Gaussian."""

        eps = 1.0e-6
        bounded_actions = torch.clamp(actions, min=-1.0 + eps, max=1.0 - eps)
        latent_actions = torch.atanh(bounded_actions)
        log_prob = self.distribution.log_prob(latent_actions) - self._tanh_log_det_jacobian(latent_actions)
        mask = self._action_training_mask.to(device=log_prob.device, dtype=log_prob.dtype)
        return (log_prob * mask).sum(dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.actor.hidden_state, self.critic.hidden_state

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load recurrent checkpoints, migrating legacy positive ``std`` to ``log_std`` when needed."""

        migrated_state = state_dict.copy()
        if self.noise_std_type == "log" and "std" in migrated_state and "log_std" not in migrated_state:
            legacy_std = migrated_state.pop("std")
            migrated_state["log_std"] = torch.log(torch.clamp(legacy_std, min=1.0e-8))
        elif self.noise_std_type == "scalar" and "log_std" in migrated_state and "std" not in migrated_state:
            legacy_log_std = migrated_state.pop("log_std")
            migrated_state["std"] = torch.exp(legacy_log_std)

        # Legacy recurrent checkpoints used an 8-D goal-only wheel residual. Embed it into the
        # current 9-D goal+yaw-rate wheel-control residual by copying the old goal columns and
        # leaving the new yaw-rate column at zero. This preserves the checkpoint's initial wheel
        # action while exposing yaw rate for subsequent residual training.
        current_state = self.state_dict()
        new_prefix = "actor.wheel_control_residual."
        old_goal_prefix = "actor.wheel_goal_residual."
        old_rate_prefix = "actor.wheel_heading_rate_residual."
        if not any(key.startswith(new_prefix) for key in migrated_state):
            for key, value in current_state.items():
                if key.startswith(new_prefix):
                    migrated_state[key] = value.clone()

            old_first_key = f"{old_goal_prefix}0.weight"
            new_first_key = f"{new_prefix}0.weight"
            if old_first_key in migrated_state and new_first_key in migrated_state:
                old_weight = migrated_state[old_first_key]
                new_weight = migrated_state[new_first_key].clone()
                if old_weight.shape[0] == new_weight.shape[0] and old_weight.shape[1] + 1 == new_weight.shape[1]:
                    new_weight[:, : old_weight.shape[1]] = old_weight
                    # ShortGoal slot 1 was identically zero in legacy checkpoints and is
                    # now reserved for normalized heading-error rate.  Zero its migrated
                    # column so enabling the new feedback observation does not change the
                    # model108 action before B12 training starts.
                    new_weight[:, self.network_spec.goal_heading_rate_slot] = 0.0
                    new_weight[:, old_weight.shape[1] :] = 0.0
                    migrated_state[new_first_key] = new_weight
                    for old_key, old_value in list(migrated_state.items()):
                        if old_key.startswith(old_goal_prefix) and old_key != old_first_key:
                            suffix = old_key[len(old_goal_prefix) :]
                            new_key = f"{new_prefix}{suffix}"
                            if new_key in migrated_state and migrated_state[new_key].shape == old_value.shape:
                                migrated_state[new_key] = old_value

        for key in list(migrated_state):
            if key.startswith(old_goal_prefix) or key.startswith(old_rate_prefix):
                migrated_state.pop(key)
        super().load_state_dict(migrated_state, strict=strict)
        return True
