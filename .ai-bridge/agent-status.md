# Agent Status

## 2026-07-28 Phase 1 Complete

Scope completed:
- Inspected active Isaac Lab Python environment RSL-RL recurrent policy, Memory, PPO, and rollout storage APIs.
- Implemented independent `RangerTerrainActorCriticRecurrent` policy.
- Added focused synthetic tests for shape, partial hidden reset, sequence/step consistency, and action mask behavior.
- Did not modify environment, reward, stop phase, V10/V11 config, task registration, train/play integration, warm-start, distillation, or training scripts.

Files touched:
- `source/Ranger/Ranger/tasks/manager_based/ranger/agents/rsl_rl_recurrent_policy.py`
- `tests/test_ranger_recurrent_policy.py` (`tests/` is ignored by repository `.gitignore`)
- `.ai-bridge/rsl-rl-recurrent-api-phase1.md`
- `.ai-bridge/agent-status.md`

Checks run:
- `source /home/tqm/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && PYTHONPATH=source/Ranger python -m pytest -q tests/test_ranger_recurrent_policy.py`
  - Result: `4 passed in 1.53s`
- `source /home/tqm/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && PYTHONPATH=source/Ranger python -m py_compile source/Ranger/Ranger/tasks/manager_based/ranger/agents/rsl_rl_recurrent_policy.py tests/test_ranger_recurrent_policy.py`
  - Result: pass

Notable finding:
- The active `isaaclab` environment has `rsl-rl-lib 3.1.2`, not the plan's expected `2.3.1`. The implementation follows the active environment API.

Blockers/open items:
- No Phase 2 config/export/task/train/play integration done yet by request.
- No V10 warm-start compatibility loader done yet; that is Phase 3.
- No distillation script done yet; that is Phase 4.

## 2026-08-17 Phase 1 Network-Spec Cleanup

Scope completed:
- Centralized the fixed recurrent actor/critic architecture in `_RangerRecurrentNetworkSpec`.
- Removed repeated internal literals for state splits, encoder outputs, fusion widths, GRU width/layers/type, trunk/head connections, map geometry, and action-head widths.
- Downstream dimensions are now derived from the single spec (for example fusion width from encoder latent widths and head input from actor trunk output).
- Environment/RSL-RL supplied dimensions remain runtime contracts and are validated against the fixed spec instead of redefining the internal architecture.
- Kept Phase 1 behavior and topology unchanged; no Phase 2 integration, rewards, environment, stop-phase, V10/V11, warm-start, checkpoint, or distillation changes were made.
- Added a focused structure regression test in `tests/test_ranger_recurrent_policy.py`.

Verification:
- Manual source review confirms the former duplicated `34:42` / `26:34` slicing and repeated MLP/fusion numeric literals were replaced by spec-derived values.
- After CodexPro bash access was restored, `tests/test_ranger_recurrent_policy.py` passed with `6 passed in 1.74s`.
- `py_compile` passed for both the recurrent policy and its test module.
- Added a regression test that instantiates the existing feedforward policy and confirms exact state-dict key/shape compatibility for `map_encoder`, `actor_trunk`, `suspension_head`, and `wheel_head`.
- Removed the repository-wide `tests/` ignore rule so the recurrent regression test is now visible to Git; cache directories remain ignored by existing rules.
- Phase 1 is now closed and verified. Phase 2 has not yet modified runtime integration at this checkpoint.

## 2026-08-17 Phases 2-4 Complete (Short Verification Only)

Phase 2 integration:
- Exported both feedforward and recurrent Ranger policy classes from `agents/__init__.py`.
- Added `RangerRecurrentActorCriticCfg` and `ShortGoalFlatCRecurrentPPORunnerCfg` without exposing internal recurrent network widths in the runner config; generic RSL-RL hidden-dim fields are empty compatibility placeholders and non-empty values are rejected by the policy.
- Added `RangerShortGoalFlatCRecurrentEnvCfg` as a behavior-identical alias of V10 and registered `Template-Ranger-ShortGoalFlat-C-Recurrent`.
- Registered the recurrent policy class in `train.py` and `play.py`.
- Added per-environment recurrent hidden reset after deterministic `play.py` env steps. Recurrent JIT/ONNX export is explicitly skipped because Isaac Lab's generic recurrent exporter assumes the stock `memory_a`/`memory_c` layout; feedforward export behavior is unchanged.
- Runtime verification: recurrent train completed 1 PPO iteration / 128 timesteps; recurrent play ran 1400 steps and crossed 8 timeout resets at 1320 steps; existing V10 feedforward train also completed a 1-iteration regression smoke.

Phase 3 V10 initialization:
- Added a distinct `recurrent_v10_actor` warm-start mode and a separate loader `warm_start_ranger_recurrent_from_feedforward`.
- It strictly copies only `actor.map_encoder`, `actor.actor_trunk`, `actor.suspension_head`, and `actor.wheel_head` from a feedforward V10 checkpoint.
- `prop_encoder`, `goal_encoder`, `fusion_norm`, GRU memory, `hidden_norm`, recurrent critic, action std/normalizers, and optimizer are not inherited from V10.
- Existing feedforward warm-start modes retain their previous code path and reject the recurrent-only mode.
- Unit tests verify copied tensors exactly match the feedforward source while recurrent-only modules, critic, and action std remain unchanged and all recurrent policy parameters stay trainable.
- Real-checkpoint smoke used `logs/rsl_rl/ranger_direct/2026-07-27_17-26-14_ShortGoalFlatV10_F1_JointHeads_WheelOnlyM149_100/model_99.pt`; strict loading reported 8 map tensors + 4 trunk + 4 suspension-head + 4 wheel-head tensors and completed one recurrent PPO iteration.

Phase 4 distillation baseline:
- Added separate `scripts/rsl_rl/distill_recurrent.py`; standard PPO is not modified with a teacher loss.
- Frozen V10 teacher always supplies deterministic `act_inference` means, never sampled noisy actions.
- Supports `teacher_rollout` and `student_rollout` modes.
- Logs total action MSE, suspension MSE, raw/effective wheel MSE, optional 128-D trunk feature MSE, teacher/student action cosine, stop-phase rate, actor hidden mean/max, hidden reset max, and gradient norm to CSV.
- Stop-phase wheel imitation is configurable with `--stop_phase_wheel_weight` and defaults to 0.0 because V10's raw wheel output is not the physically executed stop action.
- Weak shared-feature loss is configurable and defaults to 0.05.
- Recurrent hidden values remain continuous across rollout steps and training now uses configurable truncated BPTT (`--bptt_steps`, default 16), flushing before any episode reset, save boundary, final step, or full BPTT window.
- Saves a complete recurrent checkpoint with standard RSL-RL `model_state_dict`, PPO `optimizer_state_dict`, `iter`, and `infos`, plus distillation optimizer state. PPO optimizer moments are cleared so post-distillation PPO starts with optimizer state consistent with the distilled parameters.
- Both teacher-rollout and student-rollout dry runs succeeded. A distilled checkpoint was successfully loaded by both `play.py` and standard recurrent `train.py --resume`, and the resumed PPO completed one iteration.

Final verification:
- `tests/test_ranger_recurrent_policy.py`: `9 passed` after all Phase 1-4 changes.
- `py_compile` passed for all modified Ranger recurrent policy/config/task/train/play/warm-start/distillation/test modules.
- No long distillation or long recurrent PPO training was run.

Remaining non-blocking items:
- Actual control quality is not established by these smoke tests; the next work should be a deliberately sized distillation experiment followed by evaluation before committing to long recurrent PPO.
- Custom recurrent JIT/ONNX export is not implemented; deterministic Python inference/play is working.
- The active environment remains `rsl-rl-lib 3.1.2`; recurrent API compatibility should be rechecked if that package is upgraded.
