# Implement Ranger Recurrent C-Stage

Updated: 2026-07-27T20:51:36.573Z
Workspace: /home/tqm/Isaaclab_projects/Ranger
Target agent: Codex (codex)

## Plan

Implement the Ranger C-stage recurrent policy as a separate, backward-compatible architecture. Do not modify V10/V11 behavior, rewards, observations, action dimensions, stop-phase logic, or existing feedforward checkpoints. Work in small verifiable phases and stop after code/tests; do not launch long RL training.

Repository root: /home/tqm/Isaaclab_projects/Ranger
Context bundle: .ai-bridge/pro-context.md
Teacher checkpoint for later runtime use only (do not require it for static tests): logs/rsl_rl/ranger_direct/2026-07-27_17-26-14_ShortGoalFlatV10_F1_JointHeads_WheelOnlyM149_100/model_99.pt

ARCHITECTURE TO IMPLEMENT
1. Preserve the existing RangerTerrainActorCritic feedforward class unchanged in behavior and state_dict layout.
2. Add a separate RangerTerrainActorCriticRecurrent class, preferably in agents/rsl_rl_recurrent_policy.py.
3. Actor input remains the existing TensorDict groups policy_state (42) and policy_map (8*21*13). Split policy_state internally:
   - prop = concat(policy_state[..., 0:26], policy_state[..., 34:42]) => 34 dims.
   - goal = policy_state[..., 26:34] => 8 dims.
4. Actor modules:
   - prop_encoder: MLP 34 -> 128 -> 128, ELU.
   - goal_encoder: MLP 8 -> 64 -> 64, ELU.
   - map_encoder: same architecture and parameter names/shapes as existing V10 _TerrainMapEncoder (8 channels, 21x13, conv [16,32,32], latent 128), so V10 weights can be copied.
   - concatenate 128+64+128 = 320.
   - fusion_norm = LayerNorm(320).
   - one-layer GRU, input_size=320, hidden_size=256.
   - hidden_norm = LayerNorm(256).
   - actor_trunk: same structure and parameter shapes as V10 trunk, 256 -> 256 -> 128 using ELU.
   - suspension_head and wheel_head: same structures and parameter shapes as V10, 128 -> 128 -> 4.
   - output concat suspension first, wheel second. No action residual, no parallel V10 actor, no perception adapters, no head bypasses, no second actor GRU.
5. Recurrent critic is independent from actor hidden state:
   - prop encoder 34 -> 128 -> 128.
   - goal encoder 8 -> 64 -> 64.
   - map encoder 8x21x13 -> 128.
   - privileged encoder using current critic_privileged dim -> 64 -> 64.
   - concat 384 -> LayerNorm(384) -> GRU(input 384, hidden 256, 1 layer) -> LayerNorm(256) -> trunk 256 -> 256 -> 128 -> value 1.
6. Keep existing action masks, exploration masks, action std handling, normalizers, entropy masking, action log-prob logic and TensorDict observation grouping semantics equivalent to RangerTerrainActorCritic.

RSL-RL API REQUIREMENT
Before implementing recurrent methods, inspect the installed rsl-rl-lib 2.3.1 source in the active Isaac Lab Python environment. Identify the exact expected recurrent policy API (ActorCriticRecurrent/Memory, act/evaluate signatures, get_hidden_states, reset, masks, sequence tensor layout, rollout storage generator). Reuse the native recurrent PPO/storage path. Do not implement a custom PPO or manually flatten recurrent sequences. Document the inspected API in code comments or a short .ai-bridge note.

HIDDEN STATE REQUIREMENTS
- is_recurrent = True.
- Actor and critic hidden states remain separate.
- reset(dones) must clear only done environments.
- Support both rollout training with explicit hidden states/masks and step-wise deterministic inference according to native RSL-RL conventions.
- Ensure sequence forward and one-step forward use consistent dimensions.
- Do not clear hidden state on stop-phase entry; only episode termination/reset.

CONFIG AND REGISTRATION
1. agents/__init__.py: export both feedforward and recurrent policy classes.
2. rsl_rl_ppo_cfg.py:
   - add RangerRecurrentActorCriticCfg with class_name RangerTerrainActorCriticRecurrent and explicit dimensions/settings.
   - add a first recurrent runner config derived from V10 PPO settings, with num_steps_per_env=16 initially, low exploration matching V10, and no reward/environment changes.
   - do not overwrite any existing configs.
3. ranger_env_cfg.py: add a no-behavior-change config inheriting RangerShortGoalFlatV10EnvCfg, only for experiment separation.
4. ranger/__init__.py: register a new task, preferably Template-Ranger-ShortGoalFlat-C-Recurrent, pointing to the new env/runner config.
5. train.py: register the recurrent class with the runner namespace without changing feedforward registration or behavior.
6. play.py: make recurrent inference correctly reset hidden state per environment after done. Preserve existing feedforward behavior and all current trace/video features.

V10 INITIALIZATION / WARM START
Do not force the existing generic warm_start function to treat recurrent actors as feedforward actors. Add a separate recurrent initialization path/function.
- From V10 checkpoint copy only compatible actor modules by exact shape/key mapping:
  map_encoder, actor_trunk, suspension_head, wheel_head.
- Optionally copy action std/normalizer only when semantically and structurally compatible; log exactly what is copied and skipped.
- New prop_encoder, goal_encoder, fusion_norm, GRU, hidden_norm and recurrent critic remain new.
- Add a distinct CLI warm-start mode/name; do not change semantics of existing modes.
- Add strict diagnostics for missing keys, shape mismatches, trainable/frozen modules.

TEACHER-STUDENT DISTILLATION
Implement as a separate script, not inside standard train.py, preferably scripts/rsl_rl/distill_recurrent.py.
Required first version:
1. Load frozen V10 feedforward teacher and recurrent student.
2. Same environment/task observation interface for both.
3. Teacher deterministic mean is the target; never regress to sampled noisy actions.
4. Maintain student hidden state over continuous trajectories and reset per-env on done.
5. Support two modes:
   - teacher_rollout: teacher controls env; student observes and imitates.
   - student_rollout: student controls env; teacher labels visited observations.
6. Losses:
   - action mean MSE separated/logged for suspension and wheel.
   - optional weak shared-feature MSE between the 128-d trunk outputs; configurable and default lower-weight than action loss.
   - stop phase: suspension imitation remains active; wheel imitation is maskable/down-weighted because V10 raw wheel output is not the physically executed stop action.
7. Save a complete recurrent student checkpoint that can later be resumed/warm-started for PPO.
8. No teacher-guided PPO loss in this first implementation. Pure distillation followed by standard recurrent PPO is the baseline. Leave clean extension points/config only; do not modify PPO algorithm now.

DO NOT ADD IN FIRST VERSION
- parallel V10 actor at inference
- action residual heads
- perception adapters
- head-specific recurrent memories
- active policy braking / removal of stop wheel override
- new rewards, obstacles, terrain, observation dimensions
- custom PPO implementation
- long training runs

DIAGNOSTICS
Expose/log where practical:
- actor/critic hidden norm mean/max
- hidden reset max absolute value
- distillation action MSE total/suspension/wheel
- shared-feature MSE if enabled
- teacher/student action cosine similarity
- phase-specific or side-specific metrics only if easy to obtain without invasive environment changes
Use existing metrics infrastructure where possible; avoid bloating standard PPO logs with large tensors.

TESTS / VERIFICATION
Add focused tests or small smoke scripts that do not require long training:
1. Import/registration smoke test for both policy classes and new task.
2. Shape tests:
   - policy_state 42, map 2184, action 8, actor hidden 256, critic hidden 256, value 1.
3. Partial reset test:
   - set nonzero hidden for multiple envs, reset selected dones, verify done env hidden exactly zero and others unchanged.
4. Sequence-vs-step consistency test using eval mode and fixed hidden/inputs, matching native RSL-RL tensor layout.
5. State-dict compatibility test confirming V10 map_encoder/trunk/heads load exactly by shape while incompatible modules are skipped with diagnostics.
6. Feedforward regression: existing RangerTerrainActorCritic import, config class_name, and current V10 task registration unchanged.
7. Run syntax/import/type or available project verification commands. Do not start Isaac Sim long jobs unless a short headless smoke test is already supported and safe.

IMPLEMENTATION PHASES
Phase 1: inspect RSL-RL recurrent API and implement policy + unit shape/reset tests.
Phase 2: configs, exports, task registration, train/play integration, feedforward regression checks.
Phase 3: recurrent V10 initializer/warm-start path and compatibility tests.
Phase 4: separate distillation script and dry-run/synthetic tests.
Stop and report after all phases with changed-file list, design decisions, test outputs, unresolved runtime dependencies and exact commands for short smoke tests. Do not run actual distillation or PPO training without a later explicit request.

ACCEPTANCE CRITERIA
- Existing V10/V11 task configurations and feedforward policy remain loadable with unchanged behavior/code path.
- New recurrent policy uses native RSL-RL recurrent sequence handling.
- Hidden state reset is per-env and tested.
- New task/config imports successfully.
- V10 compatible modules can be selectively initialized into recurrent student with strict logging.
- Distillation script supports both rollout modes and stop-phase wheel masking.
- No unrelated refactors or environment/reward changes.

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update .ai-bridge/agent-status.md with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to .ai-bridge/implementation-diff.patch when practical.
- Append notable execution events to .ai-bridge/execution-log.jsonl when the implementation agent supports logging.
