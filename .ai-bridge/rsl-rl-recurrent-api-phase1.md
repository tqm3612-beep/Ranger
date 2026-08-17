# RSL-RL Recurrent API Inspection - Phase 1

Inspected in `/home/tqm/miniconda3/envs/isaaclab` with:

```bash
source /home/tqm/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
python -c "import importlib.metadata as md; print(md.version('rsl-rl-lib'))"
```

Actual installed package version: `rsl-rl-lib 3.1.2`.

The plan text expected `2.3.1`, but Phase 1 implementation follows the currently active Isaac Lab Python environment.

## Policy API

Native recurrent policy surface in `rsl_rl.modules.actor_critic_recurrent.ActorCriticRecurrent`:

- `is_recurrent = True`
- `act(obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor`
- `evaluate(obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor`
- `act_inference(obs: TensorDict) -> torch.Tensor`
- `get_hidden_states() -> tuple[HiddenState, HiddenState]`
- `reset(dones: torch.Tensor | None = None) -> None`
- `get_actions_log_prob(actions)`, `action_mean`, `action_std`, `entropy`

`HiddenState` is `torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None`. GRU uses a single tensor.

## Memory API

`rsl_rl.networks.Memory(input_size, hidden_dim=256, num_layers=1, type="lstm")`
wraps `nn.GRU` or `nn.LSTM`.

- Step/inference mode is selected by `masks is None`.
- Step input layout: `[num_envs, input_size]`.
- Step output layout from `Memory.forward`: `[1, num_envs, hidden_dim]`.
- Stored GRU hidden state layout: `[num_layers, num_envs, hidden_dim]`.
- Batch/update mode requires explicit saved hidden state.
- Batch input layout: `[time, trajectories, input_size]`.
- Batch masks layout: `[time, trajectories]`.
- Batch hidden layout for GRU: `[num_layers, trajectories, hidden_dim]`.
- Batch output is unpadded back to `[time, trajectories, hidden_dim]` when all masks are valid, or compacted through `unpad_trajectories` for split trajectories.

## Reset Semantics

`Memory.reset(None)` clears the whole stored hidden state by setting it to `None`.

`Memory.reset(dones)` only operates when an internal hidden state exists. For GRU it zeros:

```python
self.hidden_state[..., dones == 1, :] = 0.0
```

So done environments are cleared exactly; non-done hidden slots remain unchanged. RSL-RL PPO calls `policy.reset(dones)` after adding a transition in `PPO.process_env_step`.

## Rollout Storage Sequence Layout

`RolloutStorage.recurrent_mini_batch_generator()` uses `split_and_pad_trajectories(self.observations, self.dones)`.

- Stored observations start as `[time, num_envs, ...]`.
- `split_and_pad_trajectories` returns padded trajectories with `[time, trajectories, ...]`.
- `trajectory_masks` is `[time, trajectories]`.
- Saved hidden states are stored as `[time, num_layers, num_envs, hidden_dim]`.
- Mini-batch hidden states are selected at trajectory starts and reshaped back to GRU `[num_layers, trajectories, hidden_dim]`.

The Phase 1 `RangerTerrainActorCriticRecurrent` reuses these native layouts and does not implement custom PPO/storage flattening.
