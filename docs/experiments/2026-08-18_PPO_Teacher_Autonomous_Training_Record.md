# PPO + Teacher 自主训练与排查记录

日期：2026-08-18
项目：Ranger / Isaac Lab
工作区：`/home/tqm/Isaaclab_projects/Ranger`

## 目标

在当前 ShortGoalFlat recurrent 任务上，自主推进 PPO + Teacher 联合训练与排查，直到满足：

1. 当前任务由 student/recurrent policy 稳定完成；Teacher 不参与 rollout 控制，只保留训练正则化时也不得成为主要驱动力。
2. 多 seed deterministic evaluation 稳定，无明显 R6 式严重过度转向、长时间 genuine-worsening turn 或 M20 式 high-common/low-turn collapse。
3. critic 对中长期 realized return 的排序/相关性足以支撑 PPO，不出现持续性 value collapse。
4. 关闭 Teacher regularization 后进行短期 PPO-only continuation，策略仍能维持当前任务能力且不发生明显行为/critic 退化。
5. 达到上述条件后，认定“可以去掉 Teacher，进入后续激活感知信息等训练环节”。

## 当前已知起点

- J3：`logs/rsl_rl/ranger_direct/2026-08-18_21-27-36_J3_JointProbe_Continue_Lambda100_LR1e6_s2_1/model_0.pt`
- J4：`logs/rsl_rl/ranger_direct/2026-08-18_22-20-52_J4_JointProbe_Continue_Lambda100_LR1e6_s2_1/model_0.pt`
- Teacher V10：`logs/rsl_rl/ranger_direct/2026-07-27_17-26-14_ShortGoalFlatV10_F1_JointHeads_WheelOnlyM149_100/model_99.pt`

J1→J3 期间 critic RTG120/180 correlation、heading、path efficiency、rebound 总体改善；J4 post diagnostic 中 RTG correlation 回落，但未重新出现 M20 式 high-common/low-turn collapse。wrong-turn 已拆分为：

- `yaw_braking_counter_turn`
- `premature_counter_turn`（更准确理解为 early counter-steer，未必有害）
- `genuinely_worsening_turn`

## 自主阶段的停止/推进原则

- 不再因为单个 iteration 的单个 seed 指标波动立即回退。
- 优先用多 seed characterization 区分真实退化与 sampling/on-policy variance。
- 若 actor 行为仍改善而 critic 退化，优先修 critic target/credit assignment，不盲目回退 actor。
- 若 J3/J4 差异落在方差范围内，则进入 3–5 iteration short block，并逐 checkpoint 验收。
- 若确认 long-horizon critic 是瓶颈，优先增加 critic long-return anchor，避免直接惩罚所有 counter-steer，也避免一次性改变 actor GAE 与 reward。
- 最终必须执行 Teacher-off PPO continuation 验收；未通过则不能宣布可以去掉 Teacher。
- 所有录制视频的运行命令必须在最前面设置 `RANGER_VISUALIZE_GOAL=1`，确保目标可视化。

---

## Phase A — J3/J4 Characterization

### A0. 计划

对 J3/J4 分别执行 seed 2/3/4：

- deterministic evaluation：64 env，1400 steps；
- diagnostic-only：64 env，192 steps；

比较 actor 行为与 critic correlation/turn credit 的跨 seed 稳定性，再决定是否进入 short-block 或先修改 critic。

### A1. 执行记录

- J3 deterministic：seed 2/3/4，64 env，1400 steps。
- J4 deterministic：seed 2/3/4，64 env，1400 steps。
- J3 diagnostic-only：seed 2/3/4，64 env，192 steps。
- J4 diagnostic-only：seed 2/3/4，64 env，192 steps。
- 所有 diagnostic-only 均不进行 optimizer update。
- 首次批量 deterministic 封装调用发生一次连接层 502；未修改模型。之后改为逐 seed 执行并全部成功。

### A2. Deterministic 多 seed 汇总

| 指标 | J3 mean ± std | J4 mean ± std |
|---|---:|---:|
| episode success | 0.9944 | 0.9980 |
| path efficiency | 0.8601 ± 0.0071 | 0.8593 ± 0.0032 |
| path length (m) | 7.384 ± 0.213 | 7.445 ± 0.150 |
| heading abs | 0.5833 ± 0.0117 | 0.5767 ± 0.0095 |
| rebound (m) | 0.0996 ± 0.1088 | 0.0302 ± 0.0064 |
| turn reversal | 0.9881 ± 0.0684 | 0.8663 ± 0.0569 |
| yaw wrong direction | 0.2512 ± 0.0119 | 0.2435 ± 0.0024 |

结论：J4 actor 没有系统性退化；相反 success、heading、reversal、yaw-wrong 和 rebound 稳定性略好。J3 seed3 rebound=0.253 显示单 seed 方差明显。

### A3. Critic 多 seed汇总

| 指标 | J3 mean ± std | J4 mean ± std |
|---|---:|---:|
| corr(V,RTG30) | 0.2497 ± 0.1135 | 0.2624 ± 0.0588 |
| corr(V,RTG60) | 0.3125 ± 0.1457 | 0.3264 ± 0.0752 |
| corr(V,RTG120) | 0.3124 ± 0.1575 | 0.3208 ± 0.0779 |
| corr(V,RTG180) | 0.2488 ± 0.1071 | 0.2683 ± 0.0828 |
| genuine-worsening advantage | 0.1946 ± 0.0484 | 0.2522 ± 0.0823 |
| early-counter advantage | 0.4860 ± 0.1183 | 0.5071 ± 0.0601 |
| correct-turn advantage | 0.0427 ± 0.0392 | 0.0413 ± 0.0242 |
| high-common advantage | -0.1225 ± 0.0910 | -0.1389 ± 0.0593 |
| low-turn advantage | -0.0457 ± 0.0727 | -0.0913 ± 0.0688 |

关键判断：seed2 的 J3→J4 RTG correlation 下跌不是系统性现象。J4 的多 seed correlation 平均值略高且方差更低。genuine-worsening advantage 在 J4 平均略高，需要继续监控，但 high-common/low-turn 保持负值，没有 M20 collapse。

### A4. Phase A 决策

- 不回退 J3。
- 暂不修改 reward、GAE 或 critic long-return target，因为目前没有跨 seed 证据证明 J4 critic 系统性退化。
- 结束逐 iteration probe，进入 4-iteration conservative joint block。
- block 仍保持 rollout=192、LR=1e-6、1 epoch、4 minibatches、clip=0.1、fixed schedule、Teacher λ=1；每 iteration 保存 checkpoint。

---

## Phase B — Conservative Joint Short Block

### B1. 训练

从 J4 `model_0.pt` 恢复，固定 seed2，连续 4 iterations：

- rollout=192
- LR=1e-6
- epochs=1
- minibatches=4
- clip=0.1
- fixed schedule
- Teacher λ=1
- save_interval=1

运行目录：`logs/rsl_rl/ranger_direct/2026-08-18_23-09-37_B1_JointBlock4_FromJ4_Lambda100_LR1e6_s2`

保存：`model_0.pt` ~ `model_3.pt`。

### B2. Block 内 seed2 post-diagnostic

| checkpoint | corr RTG120 | corr RTG180 | genuine-worsening adv | high-common adv | low-turn adv |
|---|---:|---:|---:|---:|---:|
| model0 | 0.284 | 0.254 | 0.217 | -0.036 | -0.017 |
| model1 | 0.429 | 0.348 | 0.152 | -0.076 | -0.008 |
| model2 | 0.497 | 0.346 | 0.232 | -0.255 | -0.172 |
| model3 | 0.541 | 0.383 | 0.253 | -0.274 | -0.186 |

seed2 内 critic 明显改善，但必须检查跨 seed 泛化。

### B3. model3 三 seed diagnostic

- corr RTG120：`[0.541, 0.045, 0.198]`，mean=0.261，std=0.207。
- corr RTG180：`[0.383, 0.068, 0.131]`，mean=0.194，std=0.136。
- genuine-worsening advantage mean=0.185。
- high-common / low-turn 在 seed3 出现轻微正值。

对比 J4 三 seed：RTG120 mean=0.321±0.078，RTG180 mean=0.268±0.083。说明 B1 固定 seed2 连续更新出现明显 seed specialization。

### B4. model3 deterministic 三 seed

| 指标 | model3 mean ± std | J4 mean ± std |
|---|---:|---:|
| success | 0.9923 | 0.9980 |
| path efficiency | 0.8615 ± 0.0024 | 0.8593 ± 0.0032 |
| path length (m) | 7.361 ± 0.194 | 7.445 ± 0.150 |
| heading abs | 0.5736 ± 0.0037 | 0.5767 ± 0.0095 |
| rebound (m) | 0.1061 ± 0.1046 | 0.0302 ± 0.0064 |
| turn reversal | 0.8630 ± 0.0408 | 0.8663 ± 0.0569 |
| yaw wrong | 0.2510 ± 0.0030 | 0.2435 ± 0.0024 |

actor 的平均路径/heading 略改善，但 success、yaw-wrong 与 rebound robustness 变差，尤其 seed3 rebound 再次约 0.254 m。

### B5. 决策

- 不继续固定 seed2 block。
- 不选 model3 作为下一阶段起点。
- 返回多 seed 更稳的 J4 actor/critic 状态。
- 下一阶段采用 sequential multi-seed on-policy updates：seed3 → seed4 → seed2，各 1 iteration，保持同样 conservative PPO+Teacher 超参数；目标是覆盖不同随机轨迹分布，减少 seed specialization。

---

## Phase C — Sequential Multi-Seed Joint Updates

### C1. 训练链

从 J4 开始，保持 conservative joint PPO+Teacher 参数，依次进行：

1. seed3：`2026-08-18_23-19-47_C1_SeqSeed3_FromJ4_Lambda100_LR1e6/model_0.pt`
2. seed4：`2026-08-18_23-20-18_C2_SeqSeed4_FromC1_Lambda100_LR1e6/model_0.pt`
3. seed2：`2026-08-18_23-20-49_C3_SeqSeed2_FromC2_Lambda100_LR1e6/model_0.pt`

三轮训练 batch 均没有 M20 式 collapse；high-common/low-turn 保持总体负值，Teacher/PPO actor grad ratio 约 1%–1.5%，Teacher 仍是弱正则而非主要更新来源。

### C2. C3 三 seed critic diagnostic

- corr RTG30：0.309 ± 0.061
- corr RTG60：0.384 ± 0.068
- corr RTG120：0.420 ± 0.098，三 seed `[0.531, 0.294, 0.436]`
- corr RTG180：0.290 ± 0.058
- genuine-worsening advantage：0.118 ± 0.087，seed4 已为 -0.005
- early-counter advantage：0.516 ± 0.085
- high-common advantage：-0.178 ± 0.052
- low-turn advantage：-0.119 ± 0.022

与 J4 相比，critic 跨 seed correlation 明显改善，且没有 B1 的 seed2 specialization。

### C3. C3 deterministic 三 seed

- success：`[0.98895, 0.98864, 1.00000]`，mean≈0.9925
- path efficiency：mean≈0.8559
- path length：mean≈7.418 m
- heading abs：mean≈0.5929
- rebound：mean≈0.295 m；seed2≈0.363 m，seed3≈0.499 m，seed4≈0.023 m
- turn reversal：mean≈0.931
- yaw wrong：mean≈0.2575

结论：**critic 变好，但 actor 行为变差**。尤其 seed2/3 rebound 和 heading/path 指标说明已有好 actor 在 joint update 期间被 advantage 推向更明显的过冲/后期修正。

### C4. 目标可视化视频与 trajectory trace

按用户要求，所有视频命令前显式设置 `RANGER_VISUALIZE_GOAL=1`。

- J4 seed3：`logs/rsl_rl/ranger_direct/2026-08-18_22-20-52_J4_JointProbe_Continue_Lambda100_LR1e6_s2_1/videos/Autonomous_J4_seed3_goalvis/rl-video-step-0.mp4`
- C3 seed3：`logs/rsl_rl/ranger_direct/2026-08-18_23-20-49_C3_SeqSeed2_FromC2_Lambda100_LR1e6/videos/Autonomous_C3_seed3_goalvis/rl-video-step-0.mp4`

8-env trace 事件分析：

| 指标 | J4 | C3 |
|---|---:|---:|
| heading zero-cross rate | 0.619 | 0.500 |
| mean max overshoot (rad) | 0.529 | 0.538 |
| p90 overshoot (rad) | 1.014 | 1.157 |
| mean correction latency (steps) | 1.92 | 1.82 |
| genuine-worsening fraction | 0.0520 | 0.0566 |
| mean longest genuine segment | 5.48 | 5.77 |
| max longest genuine segment | 49 | 55 |

尾部 overshoot / genuine-worsening 在 C3 更差，与 64-env rebound 恶化一致。

### C5. 决策

- 不选择 C3 actor 作为下一阶段起点。
- 关键矛盾已变成：多 seed joint update 可以改善 critic，但在 critic 稳定前 actor 已经被 PPO advantage 推坏。
- 回到 J4，先冻结 actor，只做 multi-seed critic refinement；若标准 GAE critic-only 无法改善，则测试 long-return critic target。

---

## Phase D — Actor-Frozen Multi-Seed Critic Refinement

### D1. 基础设施

新增 `--teacher_ppo_critic_probe`：critic-only continuation，固定 `rollout=192 / LR=1e-6 / 1 epoch / 4 minibatches / clip=0.1 / fixed / save_interval=1`。该模式不实例化 Teacher，不计算 actor loss。

### D2. 标准 GAE critic-only cycle

从 J4 依次：

1. seed3：`2026-08-18_23-33-37_D1_CriticOnly_SeqSeed3_FromJ4_LR1e6/model_0.pt`
2. seed4：`2026-08-18_23-34-06_D2_CriticOnly_SeqSeed4_FromD1_LR1e6/model_0.pt`
3. seed2：`2026-08-18_23-34-34_D3_CriticOnly_SeqSeed2_FromD2_LR1e6/model_0.pt`

参数级验证 J4→D3：37 个 `actor.* + log_std` tensor 的 `max_abs_diff=0.0`，全部 bitwise 不变；critic 最大变化约 `1.16e-5`。

### D3. D3 三 seed diagnostic

- corr RTG30：0.260 ± 0.059
- corr RTG60：0.324 ± 0.075
- corr RTG120：0.319 ± 0.077
- corr RTG180：0.254 ± 0.078
- genuine-worsening advantage：0.250 ± 0.082
- high-common：-0.138 ± 0.059
- low-turn：-0.092 ± 0.069

与 J4 基本相同，没有可重复的改善。

### D4. 结论

标准 `λ=0.95` value target 在 actor 完全固定、多 seed覆盖的情况下仍不能持续改善中长期 value structure。因此瓶颈不是“critic训练次数不足”，而是当前 value target 对 1–3 秒 delayed steering consequences 的约束不足。

下一步：保持 actor 冻结，测试 `λ=1.0` long-return critic target；只有 critic 跨 seed明显改善后才再次释放 actor。

---

## Phase E — Long-Return Critic Target

### E1. 基础设施

新增 `--teacher_ppo_long_return_critic_probe`：actor 冻结、critic-only、`critic_relearning=True`、`critic_relearning_return_lam=1.0`，无 action perturbation；其余保持 `192 / LR1e-6 / 1 epoch / 4 minibatches / clip0.1 / fixed / save1`。

### E2. λ=1 multi-seed cycle

从 J4 依次：

1. seed3：`2026-08-18_23-39-04_E1_LongReturnCritic_SeqSeed3_FromJ4_LR1e6/model_0.pt`
2. seed4：`2026-08-18_23-39-41_E2_LongReturnCritic_SeqSeed4_FromE1_LR1e6/model_0.pt`
3. seed2：`2026-08-18_23-40-10_E3_LongReturnCritic_SeqSeed2_FromE2_LR1e6/model_0.pt`

训练中 `critic_relearning_return_lam=1.0` 明确生效；value loss 约 0.35–0.49，显著高于标准 GAE target 的 ~0.03，这是 target horizon 变长后的正常尺度变化。actor grad 始终为 0。

参数验证 J4→E3：actor/log_std `max_abs_diff=0.0`，critic max abs diff≈`1.16e-5`。

### E3. E3 三 seed diagnostic

- corr RTG30：0.260 ± 0.059
- corr RTG60：0.324 ± 0.075
- corr RTG120：0.319 ± 0.077
- corr RTG180：0.254 ± 0.078
- genuine-worsening advantage：0.250 ± 0.082

结果与 D3（标准 λ=.95 critic-only）几乎逐项相同，也与 J4 非常接近。

### E4. 结论

在固定 J4 actor distribution 下，`LR=1e-6 × 每 seed 1 epoch` 的 critic 参数改变量不足以显著重塑 value mapping；仅把 target λ 改为 1.0 但维持这一更新尺度无法产生可测改善。C3 的 correlation 提升因此至少有相当部分来自 actor 改变后的 state distribution，而不能简单归因于 critic 权重改善。

下一步保持 actor bitwise 冻结，提高 long-return critic 的更新强度；优先小幅提高 critic LR，而不是增加 actor 更新或修改 reward。

---

## Phase F — Stronger Long-Return Critic Refinement

### F0. 首次 LR=3e-6 尝试发现 resume 优先级问题

原计划从 J4 以 `λ=1.0`、critic-only、seed3→4→2、`critic LR=3e-6` 做一轮 multi-seed cycle。首次运行生成：

- `2026-08-18_23-44-54_F1_LongReturnCritic_SeqSeed3_FromJ4_LR3e6`
- `2026-08-18_23-45-29_F2_LongReturnCritic_SeqSeed4_FromF1_LR3e6`
- `2026-08-18_23-46-01_F3_LongReturnCritic_SeqSeed2_FromF2_LR3e6`

但参数 diff 显示 critic 更新量与 E3 几乎完全相同。进一步读取 F3 checkpoint 的 `optimizer_state_dict.param_groups[*].lr`，实际为 **`1e-6`**，而非命令要求的 `3e-6`。

根因：`train.py` 先根据 CLI 配置 algorithm LR，随后 `runner.load(resume_path)` 恢复 checkpoint optimizer state，从而覆盖了 CLI LR。故上述首次 F1–F3 实际仍是 LR=1e-6，不能用于评价 3e-6 long-return critic。

修复：在 `runner.load()` 之后，若启用 `--teacher_ppo_long_return_critic_probe` 且显式提供 `--critic_relearning_learning_rate`，重新设置 `runner.alg.learning_rate` 和所有 optimizer param group 的 `lr`。后续重新执行真正的 LR=3e-6 cycle，并检查保存 checkpoint 内 optimizer LR。

### F1. 修复后真正的 LR=3e-6 cycle

重新从 J4 执行：

- seed3：`2026-08-18_23-48-17_F1b_LongReturnCritic_SeqSeed3_FromJ4_LR3e6/model_0.pt`
- seed4：`2026-08-18_23-48-50_F2b_LongReturnCritic_SeqSeed4_FromF1b_LR3e6/model_0.pt`
- seed2：`2026-08-18_23-49-22_F3b_LongReturnCritic_SeqSeed2_FromF2b_LR3e6/model_0.pt`

检查 F3b checkpoint：optimizer LR=`3e-6`，J4→F3b actor max abs diff=`0.0`，critic max abs diff≈`3.50e-5`，确认本轮实验强度真正生效。

### F2. F3b 三 seed diagnostic

- corr RTG30：0.253 ± 0.058
- corr RTG60：0.318 ± 0.074
- corr RTG120：0.312 ± 0.076
- corr RTG180：0.226 ± 0.069
- genuine-worsening advantage：0.245 ± 0.082
- early-counter：0.498 ± 0.060
- high-common：-0.137 ± 0.060
- low-turn：-0.092 ± 0.069

相比 J4 / D3 / E3，没有形成可重复的中长期 critic 改善；RTG180 反而略差。说明把 λ=1 target 与更高 critic LR组合起来也不足以解决当前问题。

### F3. 决策

停止继续提高 long-return critic LR。J4 actor 已经具备很高的任务完成率，而 C3 证明正常 joint PPO 会在 critic 不确定时把已有好行为推向更大 overshoot。下一阶段不再以“先把 critic correlation 单独做高”为前提，而是**解耦 actor 与 critic 的更新尺度**：critic 继续正常学习，actor PPO step 显著缩小，并直接执行 Teacher-off continuation。若 Teacher-off 在多 seed 下稳定，则 Teacher 可移除；若仍退化，再考虑更复杂的 critic auxiliary/replay。

---

## Phase G — Reduced-Actor-Step Teacher-Off PPO

### G1. actor loss scaling 尝试

新增 `actor_loss_scale`，仅缩放 actor-side PPO surrogate / entropy / Teacher regularizer，不缩放 critic value loss。Teacher-off 设 `actor_loss_scale=0.1`，从 J4 依次 seed3→4→2：

- `2026-08-18_23-55-59_G1_TeacherOff_ActorScale010_Seed3_FromJ4/model_0.pt`
- `2026-08-18_23-56-33_G2_TeacherOff_ActorScale010_Seed4_FromG1/model_0.pt`
- `2026-08-18_23-57-05_G3_TeacherOff_ActorScale010_Seed2_FromG2/model_0.pt`

全程 `teacher_loss_coef=0`，Teacher 未实例化。

J4→G3 actor max abs drift≈`7.10e-6`，仅略低于普通 multi-seed joint C3 的 `7.91e-6`。原因：原始 actor gradient 很大，loss scaling 后仍进入 global grad clipping，再加 Adam 尺度归一化，loss 缩放并不等价于参数步长缩小。

G3 三 seed diagnostic：RTG120≈`0.388±0.087`，genuine-worsening advantage≈`0.209±0.079`，high-common/low-turn 稳定为负。

但 deterministic seed3 rebound≈`0.535 m`，复现 C3 式长尾过冲，因此 **G3 未通过 Teacher-off 行为验收**。

### G2. 决策

放弃单纯 actor loss scaling；改用 actor/critic 分离 optimizer LR 与分组 grad clipping，直接控制实际 actor 参数步长。

---

## Phase H — Teacher-Off Split Actor/Critic Learning Rate

### H1. 基础设施

新增：

- `actor_learning_rate`；
- actor param group：`actor.* + log_std`；
- critic param group：`critic.*`；
- split optimizer 模式下 actor/critic 分别执行 `clip_grad_norm_`，不再使用同一个 global grad norm；
- metrics：`applied_actor_grad_norm`、`applied_critic_grad_norm`；
- `--teacher_ppo_model_only_resume`，用于从旧单-group optimizer checkpoint 只加载模型权重；
- `play.py` 推理改为 `load_optimizer=False`，兼容 split-optimizer checkpoint。

### H2. actor LR=1e-7 Teacher-off cycle

从 J4：

- H1 seed3：`2026-08-19_00-10-22_H1_TeacherOff_SplitLR_A1e7_C1e6_Seed3_FromJ4/model_0.pt`
- H2 seed4：`2026-08-19_00-11-14_H2_TeacherOff_SplitLR_A1e7_C1e6_Seed4_FromH1/model_0.pt`
- H3 seed2：`2026-08-19_00-11-43_H3_TeacherOff_SplitLR_A1e7_C1e6_Seed2_FromH2/model_0.pt`

H1 optimizer 明确为 actor LR=`1e-7`、critic LR=`1e-6`；J4→H1 actor max abs drift=`4.77e-7`，critic≈`4.05e-6`。训练时 pre-clip actor grad norm约 `1e3`，critic约 `2`，证实旧 global clipping 强烈耦合二者。

J4→H3：actor max abs≈`1.08e-6`，critic max abs≈`1.22e-5`，actor drift 比 C3/G3 缩小约 7 倍。

H3 diagnostic：RTG120≈`0.430±0.099`，high-common≈`-0.186`，low-turn≈`-0.143`，整体健康。

H3 deterministic：

| seed | success | rebound m | path eff | heading | reversal | yaw wrong |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.9944 | 0.0222 | 0.8673 | 0.5686 | 0.7710 | 0.2412 |
| 3 | 0.9944 | **0.2385** | 0.8585 | 0.5865 | 1.0225 | 0.2602 |
| 4 | 1.0000 | 0.0246 | 0.8650 | 0.5705 | 0.8644 | 0.2473 |

虽然明显好于 G/C，但 seed3 长尾 rebound 仍重新出现，因此 H3 仍未通过最终 Teacher-off 行为门槛。

---

## Phase I — Teacher-Off Actor LR=3e-8

### I1. 训练

从 J4 重新开始，critic LR保持 `1e-6`，actor LR降为 `3e-8`：

- I1 seed3：`2026-08-19_00-19-22_I1_TeacherOff_SplitLR_A3e8_C1e6_Seed3_FromJ4/model_0.pt`
- I2 seed4：`2026-08-19_00-19-54_I2_TeacherOff_SplitLR_A3e8_C1e6_Seed4_FromI1/model_0.pt`
- I3 seed2：`2026-08-19_00-20-25_I3_TeacherOff_SplitLR_A3e8_C1e6_Seed2_FromI2/model_0.pt`

J4→I3：actor max abs drift=`3.43e-7`，RMS≈`2.03e-7`；critic max abs≈`1.22e-5`。

### I2. I3 diagnostic

- RTG120≈`0.235±0.117`
- RTG180≈`0.187±0.074`
- genuine-worsening advantage≈`0.163±0.024`
- high-common≈`-0.092`，low-turn≈`-0.053`（seed4 low-turn轻微正值）

### I3. I3 deterministic

| seed | success | rebound m | path eff | heading | reversal | yaw wrong |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.9947 | 0.0318 | 0.8626 | 0.5720 | 0.8617 | 0.2473 |
| 3 | 0.9944 | **0.2174** | 0.8558 | 0.5824 | 0.9944 | 0.2582 |
| 4 | 0.9943 | 0.0297 | 0.8623 | 0.5799 | 0.9261 | 0.2369 |

即使 actor 三轮累计移动只有 `3.43e-7`，seed3 仍从 J4 的低 rebound 重新发展出约 `0.217 m` 长尾过冲。继续单纯缩小 actor LR只会延迟问题，而不会改变 PPO 更新方向。

### I4. 决策

停止继续缩 actor LR。下一步直接检验 delayed-credit 假设：在 J4 固定模型上，不更新参数，仅把 PPO/GAE `lambda` 从 `.95` 提到 `1.0` 做多 seed diagnostic，观察 genuine-worsening advantage 是否得到根本性修正。若有效，再进行 λ=1 Teacher-off PPO continuation。

---

## Phase J — Long-Horizon Actor Credit (GAE λ Probe)

### J1. J4 固定模型 GAE λ=1.0 对照（update-free）

为了验证 genuine-worsening 的正 advantage 是否主要由 λ=.95 的短视野导致，在 J4 固定模型、同一 64-env / 192-step / seed2/3/4 diagnostic-only 条件下，仅将 GAE λ 改为 1.0。

| 指标 | λ=.95 | λ=1.0 |
|---|---:|---:|
| genuine-worsening advantage | +0.252 | **+0.404** |
| premature/early counter-turn advantage | +0.507 | **+0.576** |
| all wrong-turn advantage | +0.409 | **+0.511** |
| correct-turn advantage | +0.041 | +0.015 |
| high-common advantage | -0.139 | **-0.204** |
| low-turn advantage | -0.091 | **-0.154** |
| corr(V,RTG120) | 0.321 | 0.321 |
| corr(V,RTG180) | 0.268 | 0.268 |

RTG correlation 不变是预期的，因为模型权重与 rollout 相同，只改变 advantage 计算。λ=1 虽进一步打压 high-common / low-turn，但同时显著提高 genuine-worsening / wrong-turn 的 normalized advantage。因此“GAE 视野太短”不是单一根因，直接将 actor λ 提到 1.0 反而会加重当前 steering credit 问题。

### J2. 决策

不进行 λ=1 PPO 更新。下一步转向拆分 actor update 本身：分别测 PPO surrogate、entropy、log_std 与 actor mean-network 各层的梯度/参数漂移，并检查 closed-loop recurrent policy 对不同层微小更新的敏感性。

### J3. PPO surrogate / entropy 梯度分解

在 `rsl_rl_teacher_regularized_ppo.py` 增加 update-free 梯度分解 diagnostic，分别统计 PPO surrogate 与 squashed-entropy 对 actor mean-network 和 `log_std` 的梯度范数/夹角。J4 三 seed 结果：

- surrogate → actor mean-network grad norm：约 `487.6 ± 155.5`；
- entropy → actor mean-network grad norm：约 `0.0128`；
- surrogate → log_std grad norm：约 `0.0649`；
- entropy → log_std grad norm：约 `0.0003`。

因此 deterministic closed-loop 漂移几乎完全由 **PPO surrogate** 驱动；entropy 不是当前问题主因。

---

## Phase K — Rare Catastrophic Branch 定位与 Turn-First Reward

### K1. I3 seed3 全轨迹复查：平均退化主要来自单个极端分支

对 J4 与 I3 重新运行 `64 env / seed3 / 1400 steps` 并输出完整 trajectory trace；只比较 `episode_id=0`，保证 64 个 env 的初始目标逐一完全相同。

发现 I3 的平均 rebound 恶化几乎完全来自 `env48`：

- 初始目标：`distance=11.0077 m`，`heading=+72.927°`；
- J4：932 steps 成功，min distance≈`0.353 m`，rebound≈`0.0037 m`；
- I3：1320 steps timeout，约 step489 时 heading 穿过 `90°`，之后持续负 progress，最终 distance≈`42.58 m`，rebound≈`34.05 m`。

J4 在该状态本身也不是高裕度解：前 400 steps heading 长期维持 `70–75°`，actual yaw 只有约 `0.08–0.20 rad/s`，显著低于 desired≈`0.35 rad/s`，同时 common-mode 仍约 `0.27–0.83`。因此 J4 实际依赖一条**边前进边慢转的临界弧线**，并没有稳健学会“先建立足够 yaw authority 再前进”。极小 PPO 参数变化只是把该临界解推过失稳边界。

### K2. 原 reward 的结构性薄弱点

检查 V10/C recurrent reward：

- `progress_to_goal` 在 `|heading|>0.8 rad` 时仍保留 `alignment_floor=0.15`；
- `cruise_underspeed` 同样保留 `alignment_floor=0.15`；
- `yaw_rate_tracking` 权重仅 `-0.12`；
- `short_goal_speed_profile` 主要是近目标 braking feasibility，并不约束远距离大 heading 时的前进速度。

这允许大 heading 状态继续获得部分前进收益/速度压力，形成 drive-while-turning 临界解。

### K3. recurrent-only turn-first reward override

只修改 `RangerShortGoalFlatCRecurrentEnvCfg`，不改 V10 基础任务：

- progress alignment floor：`0.15 → 0.0`；
- cruise underspeed alignment floor：`0.15 → 0.0`；
- yaw-rate tracking weight：`-0.12 → -0.30`。

从干净 J4、Teacher 完全关闭、actor LR=`3e-7`、critic LR=`1e-6` 做 seed3→4→2 三轮 K1/K2/K3。

K3 seed3 全分布结果：success≈`98.27%`、rebound≈`0.539 m`，episode0 中 `env44/env48/env3` 出现 timeout/明显恶化；env48 仍在约 step400 穿过 90° 后跑离目标。结论：**仅通过状态 reward 改权重仍不足以修复该薄弱分支。**

---

## Phase L — Large-Heading Repair Curriculum

增加仅由环境变量启用的 repair curriculum：`RANGER_RECURRENT_TURN_REPAIR=1`。启用时：

- heading 只采样 `[-75,-45]` 与 `[45,75]`；
- distance weights 调成 `(3–5:0.10, 5–8:0.30, 8–12:0.60)`；
- 正常评估不设置该环境变量，因此仍使用原始全分布。

从干净 J4 运行 L1（seed3，Teacher off，actor LR=`3e-7`）。repair rollout 中 heading-large fraction 提高到约 `0.352`，但 `genuine-worsening advantage≈+0.505`、early-counter≈`+0.888`，显示 state-bucket credit 问题仍存在。

恢复正常全分布评估 L1：success≈`98.88%`、rebound≈`0.253 m`；episode0 的 env48 仍 timeout，最终约 `48.58 m`，另有 `-66.2°` 目标 timeout。结论：**单纯 oversampling 也不能修复 action credit。**

---

## Phase M — Immediate Action Credit

### M1. large-heading common-mode action penalty

新增 recurrent-only action-level reward：当 `|heading|` 从 `0.70 rad` 增大到 `1.05 rad` 时，平滑增大 gate，直接惩罚当前 wheel semantic common-mode；stop phase 不生效。初始权重 `-0.50`。

update-free repair diagnostic（J4/seed3）显示该 term 确实直接打压目标动作模式：

- high-common advantage 约 `-0.199`；
- turning-high-common advantage 约 `-0.175`；
- high-common 即时 reward / TD / raw advantage 均更差。

但 `genuine-worsening` normalized advantage 仍约 `+0.601`，进一步说明该 state bucket 存在 selection bias，不能等价解释为 action 本身被奖励。

从干净 J4 做 1 次 M1 repair PPO update（Teacher off，actor LR=`3e-7`，critic LR=`1e-6`）后恢复全分布评估：success≈`98.30%`、rebound≈`0.259 m`。env48 仍在约 step400 穿过 `90°` 后 timeout；同时出现一个近直线小 heading episode timeout。结论：**只压 common-mode 不够，还必须同时保证大 heading 时有足够、方向正确且持续的 turn differential / yaw authority。**

### M2. 下一步

检查现有 `short_goal_wheel_diff_prior` 后确认其实际是 `short_goal_wheel_turn_mode_prior_l1`，当前 `turn_gain=0.5`、weight=`-0.05`，即时约束过弱。下一步在 recurrent repair 阶段把“低 common + 足够 turn”成对约束，而不是继续增加 PPO iteration。

---

## 2026-08-20 — J4 最佳参考策略视频复查与阶段推进讨论

按“跨 seed 稳定性 + 低 rebound + 高成功率”的标准，J4 仍是当前整体最好的 robust behavior reference，而不是单 seed 峰值最高的 J3。

重新录制 J4 可视化视频，严格使用：

```bash
RANGER_VISUALIZE_GOAL=1 /home/tqm/miniconda3/envs/isaaclab/bin/python scripts/rsl_rl/play.py \
  --task Template-Ranger-ShortGoalFlat-C-Recurrent \
  --num_envs 4 --seed 2 --max_steps 900 \
  --video --video_length 900 \
  --camera_mode overview_fixed --overview_fixed_padding 3.0 \
  --evaluation_summary \
  --evaluation_metrics_csv /tmp/J4_best_visual_metrics.csv \
  --trajectory_trace_csv /tmp/J4_best_visual_trace.csv \
  --video_subdir J4_best_visual_20260820 \
  --checkpoint /home/tqm/Isaaclab_projects/Ranger/logs/rsl_rl/ranger_direct/2026-08-18_22-20-52_J4_JointProbe_Continue_Lambda100_LR1e6_s2_1/model_0.pt \
  --headless
```

视频：

`logs/rsl_rl/ranger_direct/2026-08-18_22-20-52_J4_JointProbe_Continue_Lambda100_LR1e6_s2_1/videos/J4_best_visual_20260820/rl-video-step-0.mp4`

本次 4-env / seed2 / 900-step 小样本：7/7 completed episodes 成功，success=1.0，mean success steps≈406.9，path efficiency≈0.8669，rebound≈0.00356 m。该视频用于直观看当前最好参考策略的实际完成能力，不替代之前 64-env、多 seed 稳健性结论。

阶段讨论：可以考虑提前启动后续感知分支实验，但不应直接把 large-heading rare catastrophic branch 视为“未来复杂环境会自然消失”。更合适的路线是保留 J4 作为运动核心参考，在初期冻结或强锚定已有 actor/trunk/head，只训练新增 perception encoder / fusion 或受控 residual，使“激活感知”与“重学基础运动”分离。这样可以验证复杂环境是否确实诱导新的 action structure，同时避免把当前 PPO update instability 混入地形/障碍学习。

---

## 2026-08-20 — P0 Teacher-Free Perception Activation smoke test

建立独立 perception 分支，不修改旧 `RangerShortGoalFlatCRecurrentEnvCfg`：

- `RangerPerceptionP0EnvCfg` 直接继承 `RangerShortGoalFlatV10EnvCfg`，仅把 `policy_map.local_navigation_map.use_neutral_map=False`；
- `RangerPerceptionP0NeutralEnvCfg` 作为同任务 neutral-map 控制组；
- 两者统一使用 `PerceptionP0RecurrentPPORunnerCfg`，保持 J4 的 recurrent actor 架构；
- P0 全程仅 play/inference，没有 PPO update，没有 Teacher，也没有 partial freeze。

首先以 4 env / seed2 / 900 steps 做 smoke test：neutral 7/7 success，real-map 6/6 success，均无 timeout 或其他失败，说明 real map 接入没有造成即时策略崩溃。

随后以 64 env / seed2 / 900 steps 做扩大评估：

| Metric | Neutral map | Real map |
| --- | ---: | ---: |
| completed episodes | 110 | 107 |
| stopped-goal success | 100% | 100% |
| mean success steps | 406.582 | 405.869 |
| path efficiency | 0.863881 | 0.861027 |
| max rebound after min | 0.027265 m | 0.023437 m |
| heading error abs mean | 0.557768 | 0.586702 |
| approach heading error abs mean | 0.635095 | 0.702911 |
| yaw correct direction rate | 0.698337 | 0.713247 |
| yaw wrong direction rate | 0.243705 | 0.240624 |
| far velocity toward goal | 1.042616 m/s | 1.121676 m/s |

初步结论：在 flat ground 上，仅把 neutral map 替换成真实 8-channel map 后，J4 的主要运动能力保持稳定，没有出现明显 regression。行为并非数值完全一致：real-map 组 heading error 略高、far forward speed 略高，但 path efficiency、完成时间、rebound、yaw-direction 指标总体仍处于同一行为区间。P0 因而通过“不会立即摧毁 J4”的第一层 smoke-test，可以继续进入 map 数值/语义验证和 simple-terrain P1-A；当前结果仍不能证明 J4 已经真正利用 perception，因为 flat real-map 本身缺少有意义的 terrain variation。

