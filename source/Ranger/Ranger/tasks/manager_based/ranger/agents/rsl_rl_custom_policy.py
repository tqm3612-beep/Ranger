# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import EmpiricalNormalization


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "selu":
        return nn.SELU()
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(_activation(activation))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class _TerrainMapEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        grid_shape: tuple[int, int],
        conv_channels: list[int],
        latent_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        conv_layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in conv_channels:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(_activation(activation))
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, grid_shape[0], grid_shape[1], dtype=torch.float32)
            conv_out_dim = int(self.conv(dummy).reshape(1, -1).shape[-1])

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, latent_dim),
            _activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class _RangerDualHeadActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        map_dim: int,
        num_actions: int,
        terrain_channels: int,
        terrain_grid_shape: tuple[int, int],
        terrain_cnn_channels: list[int],
        terrain_latent_dim: int,
        state_hidden_dims: list[int],
        state_latent_dim: int,
        actor_trunk_hidden_dims: list[int],
        wheel_head_hidden_dims: list[int],
        suspension_head_hidden_dims: list[int],
        activation: str,
    ) -> None:
        super().__init__()
        if num_actions != 8:
            raise ValueError(f"RangerDualHeadActor expects 8 actions, got {num_actions}.")

        self.state_dim = int(state_dim)
        self.map_dim = int(map_dim)
        self.num_actions = int(num_actions)
        self.terrain_channels = int(terrain_channels)
        self.terrain_grid_shape = tuple(int(v) for v in terrain_grid_shape)
        expected_map_dim = self.terrain_channels * self.terrain_grid_shape[0] * self.terrain_grid_shape[1]
        if self.map_dim != expected_map_dim:
            raise ValueError(
                f"Terrain map dim mismatch: expected {expected_map_dim} from "
                f"{self.terrain_channels}x{self.terrain_grid_shape}, got {self.map_dim}."
            )

        self.map_encoder = _TerrainMapEncoder(
            input_channels=self.terrain_channels,
            grid_shape=self.terrain_grid_shape,
            conv_channels=terrain_cnn_channels,
            latent_dim=terrain_latent_dim,
            activation=activation,
        )
        self.state_encoder = _build_mlp(
            input_dim=self.state_dim,
            hidden_dims=state_hidden_dims,
            output_dim=state_latent_dim,
            activation=activation,
        )
        fusion_input_dim = terrain_latent_dim + state_latent_dim
        trunk_output_dim = actor_trunk_hidden_dims[-1] if actor_trunk_hidden_dims else fusion_input_dim
        self.actor_trunk = _build_mlp(
            input_dim=fusion_input_dim,
            hidden_dims=actor_trunk_hidden_dims[:-1] if actor_trunk_hidden_dims else [],
            output_dim=trunk_output_dim,
            activation=activation,
        )
        self.actor_trunk_activation = _activation(activation)
        self.suspension_head = _build_mlp(
            input_dim=trunk_output_dim,
            hidden_dims=suspension_head_hidden_dims,
            output_dim=4,
            activation=activation,
        )
        self.wheel_head = _build_mlp(
            input_dim=trunk_output_dim,
            hidden_dims=wheel_head_hidden_dims,
            output_dim=4,
            activation=activation,
        )

        nn.init.zeros_(self.suspension_head[-1].weight)
        nn.init.zeros_(self.suspension_head[-1].bias)
        nn.init.zeros_(self.wheel_head[-1].weight)
        nn.init.zeros_(self.wheel_head[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state_obs = obs[:, : self.state_dim]
        map_obs = obs[:, self.state_dim : self.state_dim + self.map_dim]
        map_obs = map_obs.view(
            obs.shape[0], self.terrain_channels, self.terrain_grid_shape[0], self.terrain_grid_shape[1]
        )

        z_state = self.state_encoder(state_obs)
        z_map = self.map_encoder(map_obs)
        fused = torch.cat((z_map, z_state), dim=-1)
        trunk = self.actor_trunk_activation(self.actor_trunk(fused))

        suspension_action = self.suspension_head(trunk)
        wheel_action = self.wheel_head(trunk)
        return torch.cat((suspension_action, wheel_action), dim=-1)


class _RangerCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        map_dim: int,
        privileged_dim: int,
        terrain_channels: int,
        terrain_grid_shape: tuple[int, int],
        terrain_cnn_channels: list[int],
        terrain_latent_dim: int,
        state_hidden_dims: list[int],
        state_latent_dim: int,
        privileged_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.map_dim = int(map_dim)
        self.privileged_dim = int(privileged_dim)
        self.terrain_channels = int(terrain_channels)
        self.terrain_grid_shape = tuple(int(v) for v in terrain_grid_shape)

        self.map_encoder = _TerrainMapEncoder(
            input_channels=self.terrain_channels,
            grid_shape=self.terrain_grid_shape,
            conv_channels=terrain_cnn_channels,
            latent_dim=terrain_latent_dim,
            activation=activation,
        )
        self.state_encoder = _build_mlp(
            input_dim=self.state_dim,
            hidden_dims=state_hidden_dims,
            output_dim=state_latent_dim,
            activation=activation,
        )
        self.privileged_encoder = _build_mlp(
            input_dim=self.privileged_dim,
            hidden_dims=privileged_hidden_dims,
            output_dim=privileged_hidden_dims[-1] if privileged_hidden_dims else self.privileged_dim,
            activation=activation,
        )
        privileged_latent_dim = privileged_hidden_dims[-1] if privileged_hidden_dims else self.privileged_dim
        self.value_head = _build_mlp(
            input_dim=terrain_latent_dim + state_latent_dim + privileged_latent_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state_obs = obs[:, : self.state_dim]
        map_start = self.state_dim
        map_end = map_start + self.map_dim
        map_obs = obs[:, map_start:map_end]
        priv_obs = obs[:, map_end : map_end + self.privileged_dim]
        map_obs = map_obs.view(
            obs.shape[0], self.terrain_channels, self.terrain_grid_shape[0], self.terrain_grid_shape[1]
        )

        z_state = self.state_encoder(state_obs)
        z_map = self.map_encoder(map_obs)
        z_priv = self.privileged_encoder(priv_obs)
        return self.value_head(torch.cat((z_map, z_state, z_priv), dim=-1))


class RangerTerrainActorCritic(ActorCritic):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        terrain_obs_group: str = "policy_map",
        state_obs_group: str = "policy_state",
        privileged_obs_group: str = "critic_privileged",
        terrain_channels: int = 8,
        terrain_grid_shape: tuple[int, int] = (21, 13),
        terrain_cnn_channels: list[int] = [16, 32, 32],
        terrain_latent_dim: int = 128,
        state_hidden_dims: list[int] = [128],
        state_latent_dim: int = 128,
        actor_hidden_dims: list[int] | None = None,
        actor_trunk_hidden_dims: list[int] = [256, 128],
        critic_hidden_dims: list[int] = [256, 128],
        privileged_hidden_dims: list[int] = [64],
        wheel_head_hidden_dims: list[int] = [128],
        suspension_head_hidden_dims: list[int] = [128],
        action_training_mask: list[float] | None = None,
        action_output_mask: list[float] | None = None,
        action_exploration_mask: list[float] | None = None,
        initial_action_std: list[float] | None = None,
        inactive_action_std: float = 1.0e-6,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "RangerTerrainActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        if state_dependent_std:
            raise NotImplementedError("RangerTerrainActorCritic does not implement state-dependent std in V1.")
        nn.Module.__init__(self)

        self.obs_groups = obs_groups
        self.terrain_obs_group = terrain_obs_group
        self.state_obs_group = state_obs_group
        self.privileged_obs_group = privileged_obs_group
        self.state_dependent_std = state_dependent_std
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

        action_training_mask_tensor = _binary_mask(
            "action_training_mask", action_training_mask, [1.0] * num_actions
        )
        if not torch.any(action_training_mask_tensor > 0.0):
            raise ValueError("action_training_mask must keep at least one trainable action dimension.")
        action_output_mask_tensor = _binary_mask(
            "action_output_mask", action_output_mask, [1.0] * num_actions
        )
        action_exploration_mask_tensor = _binary_mask(
            "action_exploration_mask", action_exploration_mask, action_training_mask_tensor.tolist()
        )
        if initial_action_std is None:
            initial_action_std_tensor = init_noise_std * torch.ones(num_actions, dtype=torch.float32)
        else:
            if len(initial_action_std) != num_actions:
                raise ValueError(
                    f"initial_action_std must have {num_actions} entries, got {len(initial_action_std)}."
                )
            initial_action_std_tensor = torch.tensor(initial_action_std, dtype=torch.float32)
            if torch.any(initial_action_std_tensor <= 0.0):
                raise ValueError("initial_action_std entries must all be positive.")

        self.register_buffer("_action_training_mask", action_training_mask_tensor, persistent=False)
        self.register_buffer("_action_output_mask", action_output_mask_tensor, persistent=False)
        self.register_buffer("_action_exploration_mask", action_exploration_mask_tensor, persistent=False)
        self.register_buffer("_configured_initial_action_std", initial_action_std_tensor, persistent=False)
        self._inactive_action_std = float(inactive_action_std)
        if not torch.all(action_training_mask_tensor == 1.0) or not torch.all(action_output_mask_tensor == 1.0):
            print(
                "[RangerPolicy] action masks enabled: "
                f"training={action_training_mask_tensor.tolist()} "
                f"output={action_output_mask_tensor.tolist()} "
                f"exploration={action_exploration_mask_tensor.tolist()} "
                f"inactive_std={self._inactive_action_std:.1e}"
            )

        self.num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        self.num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        self.state_dim = int(obs[self.state_obs_group].shape[-1])
        self.map_dim = int(obs[self.terrain_obs_group].shape[-1])
        self.privileged_dim = int(obs[self.privileged_obs_group].shape[-1])
        if actor_hidden_dims is not None:
            actor_trunk_hidden_dims = actor_hidden_dims

        self.actor = _RangerDualHeadActor(
            state_dim=self.state_dim,
            map_dim=self.map_dim,
            num_actions=num_actions,
            terrain_channels=terrain_channels,
            terrain_grid_shape=terrain_grid_shape,
            terrain_cnn_channels=terrain_cnn_channels,
            terrain_latent_dim=terrain_latent_dim,
            state_hidden_dims=state_hidden_dims,
            state_latent_dim=state_latent_dim,
            actor_trunk_hidden_dims=actor_trunk_hidden_dims,
            wheel_head_hidden_dims=wheel_head_hidden_dims,
            suspension_head_hidden_dims=suspension_head_hidden_dims,
            activation=activation,
        )
        print(f"Ranger actor: {self.actor}")

        self.critic = _RangerCritic(
            state_dim=self.state_dim,
            map_dim=self.map_dim,
            privileged_dim=self.privileged_dim,
            terrain_channels=terrain_channels,
            terrain_grid_shape=terrain_grid_shape,
            terrain_cnn_channels=terrain_cnn_channels,
            terrain_latent_dim=terrain_latent_dim,
            state_hidden_dims=state_hidden_dims,
            state_latent_dim=state_latent_dim,
            privileged_hidden_dims=privileged_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        print(f"Ranger critic: {self.critic}")

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
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'."
            )

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        entropy = self.distribution.entropy()
        mask = self._action_training_mask.to(device=entropy.device, dtype=entropy.dtype)
        return (entropy * mask).sum(dim=-1)

    def _apply_action_output_mask(self, mean: torch.Tensor) -> torch.Tensor:
        mask = self._action_output_mask.to(device=mean.device, dtype=mean.dtype)
        return mean * mask

    def _update_distribution(self, obs: TensorDict) -> None:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        mean = self._apply_action_output_mask(self.actor(actor_obs))
        exploration_mask = self._action_exploration_mask.to(device=mean.device, dtype=mean.dtype)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        std = std * exploration_mask + self._inactive_action_std * (1.0 - exploration_mask)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, TensorDict):
            actor_obs = self.get_actor_obs(obs)
        else:
            actor_obs = obs
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self._apply_action_output_mask(self.actor(actor_obs))

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution.log_prob(actions)
        mask = self._action_training_mask.to(device=log_prob.device, dtype=log_prob.dtype)
        return (log_prob * mask).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
