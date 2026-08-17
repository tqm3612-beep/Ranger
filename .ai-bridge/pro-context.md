# Ranger Recurrent Policy C-Stage Context

Generated: 2026-07-27T20:50:43.881Z
Workspace: /home/tqm/Isaaclab_projects/Ranger
Workspace ID: ws_c74fe79c2f8669eb0ff61b1f
Write mode: workspace
Bash mode: safe
Tool mode: standard

Purpose: paste this bundle into a high-context ChatGPT model when that model cannot call the CodexPro MCP tools directly.
Instruction for ChatGPT: use this as repository context, produce a narrow Codex execution plan, and avoid inventing files or runtime facts not shown here.

## Repository Tree

.
├── goal_logs/
│   ├── GoalHeading-Flat_stand199_s1_stage1_smoke100_v1_debug.log
│   ├── GoalHeading-Flat_stand199_s1_stage1_smoke100_v1.log
│   ├── GoalHeading-Flat_stand199_s1_stage1_smoke100_v2.log
│   ├── GoalHeading-Flat_stand199_s1_stage1_smoke100_v3.log
│   ├── GoalHeading-Flat_turn_sanity_forward_standlegs.log
│   ├── GoalHeading-Flat_turn_sanity_forward_zerolegs.log
│   ├── GoalHeading-Flat_turn_sanity_forward.log
│   ├── GoalHeading-Flat_turn_sanity_left_standlegs.log
│   ├── GoalHeading-Flat_turn_sanity_left.log
│   ├── GoalHeading-Flat_turn_sanity_right_standlegs.log
│   └── GoalHeading-Flat_turn_sanity_right.log
├── logs/
│   ├── debug_wheel_turn_mirror/
│   │   └── 20260716_214646/
│   │       ├── mirror_comparison.csv
│   │       ├── summary_by_side.csv
│   │       └── timeseries.csv
│   └── rsl_rl/
│       └── ranger_direct/
│           ├── 2026-07-14_16-17-45_NewURDF_Stand-v0_4096_seed2_Stand-model499_actorfull_500/
│           ├── 2026-07-14_22-20-43_ShortGoalFlatV1_4096_seed2_2026-07-14_16-17-45_actorsuspension_2000/
│           ├── 2026-07-15_09-05-18_ShortGoalFlatV1_4096_s2_2026-07-14_22-20-43_actorfull_1000_15s_Faster/
│           ├── 2026-07-15_10-49-11_ShortGoalFlatV1_s2_2026-07-15_09-05-18_actorfull_1000_StopPhase_SlipMetrics/
│           ├── 2026-07-15_12-09-59_ShortGoalFlatV1_s2_2026-07-15_10-49-11_actorfull_1000_5to8m_Braking/
│           ├── 2026-07-15_18-17-39_ShortGoalFlatV1_s2_2026-07-15_10-49-11_actorsuspension_1000_5to8m_LongTermGoal_AntiOppose/
│           ├── 2026-07-15_20-36-17_ShortGoalFlatV1_s2_2026-07-15_18-17-39_actorsuspension_500_5to8m_LongTerm_SlipAntiSymHeading/
│           ├── 2026-07-15_22-16-03_ShortGoalFlatV1_s2_2026-07-15_18-17-39_actorfull_5000_5to8m_DynamicBrake_NonConservative/
│           ├── 2026-07-16_09-45-06_ShortGoalFlatV1_s2_2026-07-15_18-17-39_actorfull_1000_5to8m_CaptureZeroWheel/
│           ├── 2026-07-16_12-05-42_ShortGoalFlatV1_s2_2026-07-16_09-45-06_actorfull_1000_5to8m_Capture03_Success05_FixedStd_SlipMetrics/
│           ├── 2026-07-16_14-12-11_ShortGoalFlatV1_s2_2026-07-16_12-05-42_actorfull_500_5to8m_TractionOverspeed_w005/
│           ├── 2026-07-16_17-02-46_ShortGoalFlatV1_s2_2026-07-16_12-05-42_actorfull_500_5to8m_ContinuousYawTracking_Traction_w005/
│           ├── 2026-07-16_22-26-58_CaptureZeroWheel_s2_2026-07-16_09-45-06_actorfull_5000_Yaw035_w012_AlignFloor015_WheelStd025/
│           ├── 2026-07-17_15-38-42_ShortGoalFlatV2_s2_2026-07-16_22-26-58_actorwheel_200_FixedSuspension_WheelHeadOnly/
│           ├── 2026-07-17_19-31-21_ShortGoalFlatV3_A1_SuspHeadOnly_Scale005_200/
│           ├── 2026-07-18_00-01-27_ShortGoalFlatV4_B1_Heading60_FrozenSuspension_WheelHead_150/
│           ├── 2026-07-18_00-50-07_ShortGoalFlatV4_B1_Heading60_FrozenSuspension_WheelHead_15000/
│           ├── 2026-07-21_20-46-33_ShortGoalFlatV5_C1_Distance3to12_FrozenSuspension_WheelHead_TractionMonitor_3000/
│           ├── 2026-07-22_18-20-12_ShortGoalFlatV5_TractionMetricsSmokeTest/
│           ├── 2026-07-22_18-24-58_ShortGoalFlatV5_TractionMetricsValidation_50/
│           ├── 2026-07-22_18-52-41_ShortGoalFlatV6_D1_Heading75_Distance3to12_FrozenSuspension_WheelHead_CompactMetrics_1000/
│           ├── 2026-07-22_23-12-10_ShortGoalFlatV7_E1_FullSuspension_ActorHeads_LowNoise_8000/
│           ├── 2026-07-23_11-13-15_ShortGoalFlatV8_E1_BalancedStop_RollPitch035_StrokeTrack002_Hold60_500/
│           ├── 2026-07-23_12-24-36_ShortGoalFlatV9_E1_StopPolicySuspension_RollPitchDense_SuspHeadOnly_V7M6100_500/
│           ├── 2026-07-23_13-28-10_ShortGoalFlatV9_E2_StopPolicySuspension_ActorHeads_LowNoise_V7M6100_3000/
│           ├── 2026-07-24_10-24-55_ShortGoalFlatV9_E3_NoHydraulicRegularizers_ActorHeads_V7M6100_2000/
│           ├── 2026-07-26_16-36-14_ShortGoalFlatV9_E4_ResetSuspHead_V6M600_300/
│           ├── 2026-07-26_17-42-06_ShortGoalFlatV9_E5_ResetSuspStable_JointHeads_M100_200/
│           ├── 2026-07-26_19-53-07_ShortGoalFlatV9_E6_ResetHeads_V6M600_300/
│           ├── 2026-07-26_20-50-12_ShortGoalFlatV9_E7_WheelFinalReset_E4M100_800/
│           ├── 2026-07-27_16-57-17_ShortGoalFlatV10_StopR50_PrecisionR30_WheelOnly_E4M100_150/
│           ├── 2026-07-27_17-26-14_ShortGoalFlatV10_F1_JointHeads_WheelOnlyM149_100/
│           ├── 2026-07-27_18-17-46_ShortGoalFlatV11_G1_PolicyStop_DistanceEnvelope_JointHeads_V10F1M99_500/
│           └── Stand/
├── outputs/
│   ├── 2026-06-13/
│   │   ├── 03-23-45/
│   │   │   └── hydra.log
│   │   ├── 03-25-13/
│   │   │   └── hydra.log
│   │   ├── 03-27-01/
│   │   │   └── hydra.log
│   │   ├── 03-27-50/
│   │   │   └── hydra.log
│   │   └── 03-57-08/
│   │       └── hydra.log
│   ├── 2026-06-17/
│   │   ├── 16-58-45/
│   │   │   └── hydra.log
│   │   └── 23-29-32/
│   │       └── hydra.log
│   ├── 2026-06-18/
│   │   ├── 01-12-28/
│   │   │   └── hydra.log
│   │   ├── 01-13-09/
│   │   │   └── hydra.log
│   │   ├── 14-26-12/
│   │   │   └── hydra.log
│   │   ├── 15-43-34/
│   │   │   └── hydra.log
│   │   ├── 15-46-43/
│   │   │   └── hydra.log
│   │   └── 15-47-40/
│   │       └── hydra.log
│   ├── 2026-06-23/
│   │   └── 22-57-16/
│   │       └── hydra.log
│   ├── 2026-06-24/
│   │   ├── 00-22-20/
│   │   │   └── hydra.log
│   │   ├── 00-31-54/
│   │   │   └── hydra.log
│   │   ├── 00-38-30/
│   │   │   └── hydra.log
│   │   └── 00-48-42/
│   │       └── hydra.log
│   ├── 2026-06-25/
│   │   ├── 13-31-47/
│   │   │   └── hydra.log
│   │   ├── 14-44-49/
│   │   │   └── hydra.log
│   │   ├── 14-54-16/
│   │   │   └── hydra.log
│   │   ├── 14-57-40/
│   │   │   └── hydra.log
│   │   ├── 15-02-52/
│   │   │   └── hydra.log
│   │   ├── 15-20-02/
│   │   │   └── hydra.log
│   │   ├── 15-40-28/
│   │   │   └── hydra.log
│   │   ├── 15-41-24/
│   │   │   └── hydra.log
│   │   ├── 15-55-21/
│   │   │   └── hydra.log
│   │   ├── 16-08-43/
│   │   │   └── hydra.log
│   │   ├── 20-10-56/
│   │   │   └── hydra.log
│   │   ├── 20-13-50/
│   │   │   └── hydra.log
│   │   ├── 20-15-35/
│   │   │   └── hydra.log
│   │   ├── 20-16-31/
│   │   │   └── hydra.log
│   │   ├── 20-28-32/
│   │   │   └── hydra.log
│   │   ├── 20-29-05/
│   │   │   └── hydra.log
│   │   ├── 20-30-14/
│   │   │   └── hydra.log
│   │   ├── 20-55-33/
│   │   │   └── hydra.log
│   │   ├── 20-59-21/
│   │   │   └── hydra.log
│   │   ├── 21-08-03/
│   │   │   └── hydra.log
│   │   ├── 21-09-43/
│   │   │   └── hydra.log
│   │   ├── 21-20-54/
│   │   │   └── hydra.log
│   │   ├── 21-21-58/
│   │   │   └── hydra.log
│   │   ├── 21-22-33/
│   │   │   └── hydra.log
│   │   ├── 21-23-13/
│   │   │   └── hydra.log
│   │   └── 21-26-26/
│   │       └── hydra.log
│   ├── 2026-06-26/
│   │   ├── 00-58-57/
│   │   │   └── hydra.log
│   │   ├── 01-06-58/
│   │   │   └── hydra.log
│   │   ├── 01-08-15/
│   │   │   └── hydra.log
│   │   ├── 01-10-24/
│   │   │   └── hydra.log
│   │   ├── 01-13-00/
│   │   │   └── hydra.log
│   │   ├── 01-13-36/
│   │   │   └── hydra.log
│   │   ├── 01-14-11/
│   │   │   └── hydra.log
│   │   ├── 01-19-44/
│   │   │   └── hydra.log
│   │   ├── 01-23-06/
│   │   │   └── hydra.log
│   │   ├── 01-46-37/
│   │   │   └── hydra.log
│   │   ├── 01-51-46/
│   │   │   └── hydra.log
│   │   ├── 01-53-46/
│   │   │   └── hydra.log
│   │   ├── 02-25-44/
│   │   │   └── hydra.log
│   │   ├── 02-37-25/
│   │   │   └── hydra.log
│   │   ├── 02-38-34/
│   │   │   └── hydra.log
│   │   ├── 02-43-16/
│   │   │   └── hydra.log
│   │   ├── 02-46-58/
│   │   │   └── hydra.log
│   │   ├── 02-49-12/
│   │   │   └── hydra.log
│   │   ├── 02-53-21/
│   │   │   └── hydra.log
│   │   ├── 03-00-48/
│   │   │   └── hydra.log
│   │   ├── 03-06-47/
│   │   │   └── hydra.log
│   │   ├── 03-08-27/
│   │   │   └── hydra.log
│   │   ├── 03-09-36/
│   │   │   └── hydra.log
│   │   ├── 04-13-35/
│   │   │   └── hydra.log
│   │   ├── 04-14-08/
│   │   │   └── hydra.log
│   │   ├── 04-18-45/
│   │   │   └── hydra.log
│   │   ├── 04-19-36/
│   │   │   └── hydra.log
│   │   ├── 04-20-09/
│   │   │   └── hydra.log
│   │   ├── 16-41-15/
│   │   │   └── hydra.log
│   │   ├── 16-42-24/
│   │   │   └── hydra.log
│   │   ├── 16-49-29/
│   │   │   └── hydra.log
│   │   ├── 16-50-15/
│   │   │   └── hydra.log
│   │   ├── 17-19-41/
│   │   │   └── hydra.log
│   │   ├── 17-22-58/
│   │   │   └── hydra.log
│   │   ├── 17-25-49/
│   │   │   └── hydra.log
│   │   ├── 17-31-47/
│   │   │   └── hydra.log
│   │   ├── 20-42-42/
│   │   │   └── hydra.log
│   │   ├── 20-43-06/
│   │   │   └── hydra.log
│   │   ├── 20-48-43/
│   │   │   └── hydra.log
│   │   ├── 23-12-18/
│   │   │   └── hydra.log
│   │   ├── 23-17-46/
│   │   │   └── hydra.log
│   │   ├── 23-20-07/
│   │   │   └── hydra.log
│   │   ├── 23-26-53/
│   │   │   └── hydra.log
│   │   ├── 23-31-02/
│   │   │   └── hydra.log
│   │   ├── 23-39-31/
│   │   │   └── hydra.log
│   │   ├── 23-46-38/
│   │   │   └── hydra.log
│   │   └── 23-54-24/
│   │       └── hydra.log
│   ├── 2026-06-27/
│   │   ├── 00-07-21/
│   │   │   └── hydra.log
│   │   ├── 00-09-24/
│   │   │   └── hydra.log
│   │   ├── 14-39-18/
│   │   │   └── hydra.log
│   │   ├── 14-44-36/
│   │   │   └── hydra.log
│   │   ├── 14-46-19/
│   │   │   └── hydra.log
│   │   ├── 14-46-42/
│   │   │   └── hydra.log
│   │   ├── 14-47-17/
│   │   │   └── hydra.log
│   │   ├── 14-47-39/
│   │   │   └── hydra.log
│   │   ├── 14-51-06/
│   │   │   └── hydra.log
│   │   ├── 14-59-06/
│   │   │   └── hydra.log
│   │   ├── 15-03-40/
│   │   │   └── hydra.log
│   │   ├── 15-09-19/
│   │   │   └── hydra.log
│   │   ├── 15-15-40/
│   │   │   └── hydra.log
│   │   ├── 15-29-54/
│   │   │   └── hydra.log
│   │   ├── 15-33-45/
│   │   │   └── hydra.log
│   │   ├── 15-37-08/
│   │   │   └── hydra.log
│   │   ├── 15-45-54/
│   │   │   └── hydra.log
│   │   ├── 15-49-35/
│   │   │   └── hydra.log
│   │   ├── 15-54-39/
│   │   │   └── hydra.log
│   │   ├── 15-58-41/
│   │   │   └── hydra.log
│   │   ├── 16-03-32/
│   │   │   └── hydra.log
│   │   ├── 16-04-28/
│   │   │   └── hydra.log
│   │   ├── 16-12-24/
│   │   │   └── hydra.log
│   │   ├── 16-17-34/
│   │   │   └── hydra.log
│   │   ├── 16-22-07/
│   │   │   └── hydra.log
│   │   ├── 16-32-39/
│   │   │   └── hydra.log
│   │   ├── 16-41-31/
│   │   │   └── hydra.log
│   │   ├── 16-46-16/
│   │   │   └── hydra.log
│   │   ├── 16-46-31/
│   │   │   └── hydra.log
│   │   ├── 16-47-37/
│   │   │   └── hydra.log
│   │   ├── 17-06-20/
│   │   │   └── hydra.log
│   │   ├── 17-09-28/
│   │   │   └── hydra.log
│   │   ├── 17-12-35/
│   │   │   └── hydra.log
│   │   ├── 17-33-41/
│   │   │   └── hydra.log
│   │   ├── 18-38-08/
│   │   │   └── hydra.log
│   │   ├── 18-57-46/
│   │   │   └── hydra.log
│   │   ├── 19-12-50/
│   │   │   └── hydra.log
│   │   ├── 19-21-42/
│   │   │   └── hydra.log
│   │   ├── 19-25-53/
│   │   │   └── hydra.log
│   │   ├── 19-29-43/
│   │   │   └── hydra.log
│   │   ├── 19-34-28/
│   │   │   └── hydra.log
│   │   ├── 20-05-49/
│   │   │   └── hydra.log
│   │   ├── 20-56-31/
│   │   │   └── hydra.log
│   │   ├── 21-09-37/
│   │   │   └── hydra.log
│   │   ├── 22-23-07/
│   │   │   └── hydra.log
│   │   ├── 22-23-20/
│   │   │   └── hydra.log
│   │   ├── 22-42-41/
│   │   │   └── hydra.log
│   │   ├── 22-47-11/
│   │   │   └── hydra.log
│   │   ├── 22-51-23/
│   │   │   └── hydra.log
│   │   ├── 22-55-08/
│   │   │   └── hydra.log
│   │   ├── 23-01-39/
│   │   │   └── hydra.log
│   │   ├── 23-18-45/
│   │   │   └── hydra.log
│   │   ├── 23-25-42/
│   │   │   └── hydra.log
│   │   ├── 23-29-43/
│   │   │   └── hydra.log
│   │   ├── 23-33-13/
│   │   │   └── hydra.log
│   │   ├── 23-37-26/
│   │   │   └── hydra.log
│   │   ├── 23-41-15/
│   │   │   └── hydra.log
│   │   ├── 23-47-38/
│   │   │   └── hydra.log
│   │   └── 23-59-44/
│   │       └── hydra.log
│   ├── 2026-06-28/
│   │   ├── 00-00-01/
│   │   │   └── hydra.log
│   │   ├── 00-09-06/
│   │   │   └── hydra.log
│   │   ├── 00-15-23/
│   │   │   └── hydra.log
│   │   └── 00-20-28/
│   │       └── hydra.log
│   ├── 2026-06-29/
│   │   └── 14-26-59/
│   │       └── hydra.log
│   ├── 2026-06-30/
│   │   ├── 23-26-21/
│   │   │   └── hydra.log
│   │   └── 23-28-01/
│   │       └── hydra.log
│   ├── 2026-07-01/
│   │   ├── 14-36-32/
│   │   │   └── hydra.log
│   │   ├── 14-37-37/
│   │   │   └── hydra.log
│   │   ├── 14-39-35/
│   │   │   └── hydra.log
│   │   ├── 14-40-58/
│   │   │   └── hydra.log
│   │   ├── 16-09-08/
│   │   │   └── hydra.log
│   │   └── 23-28-46/
│   │       └── hydra.log
│   ├── 2026-07-02/
│   │   ├── 00-06-17/
│   │   │   └── hydra.log
│   │   ├── 00-14-58/
│   │   │   └── hydra.log
│   │   ├── 00-27-12/
│   │   │   └── hydra.log
│   │   ├── 00-31-03/
│   │   │   └── hydra.log
│   │   ├── 00-32-43/
│   │   │   └── hydra.log
│   │   ├── 00-38-51/
│   │   │   └── hydra.log
│   │   ├── 00-43-20/
│   │   │   └── hydra.log
│   │   ├── 01-15-56/
│   │   │   └── hydra.log
│   │   ├── 02-14-44/
│   │   │   └── hydra.log
│   │   ├── 02-20-50/
│   │   │   └── hydra.log
│   │   ├── 02-21-12/
│   │   │   └── hydra.log
│   │   ├── 02-28-36/
│   │   │   └── hydra.log
│   │   ├── 02-39-18/
│   │   │   └── hydra.log
│   │   ├── 02-41-16/
│   │   │   └── hydra.log
│   │   ├── 02-44-26/
│   │   │   └── hydra.log
│   │   ├── 02-50-24/
│   │   │   └── hydra.log
│   │   ├── 02-53-46/
│   │   │   └── hydra.log
│   │   ├── 02-55-49/
│   │   │   └── hydra.log
│   │   ├── 02-57-48/
│   │   │   └── hydra.log
│   │   ├── 02-59-18/
│   │   │   └── hydra.log
│   │   ├── 03-12-32/
│   │   │   └── hydra.log
│   │   ├── 14-51-48/
│   │   │   └── hydra.log
│   │   ├── 16-34-34/
│   │   │   └── hydra.log
│   │   ├── 16-34-51/
│   │   │   └── hydra.log
│   │   ├── 16-35-13/
│   │   │   └── hydra.log
│   │   ├── 16-35-59/
│   │   │   └── hydra.log
│   │   ├── 16-36-30/
│   │   │   └── hydra.log
│   │   ├── 16-37-07/
│   │   │   └── hydra.log
│   │   ├── 16-40-46/
│   │   │   └── hydra.log
│   │   ├── 17-09-09/
│   │   │   └── hydra.log
│   │   ├── 17-09-55/
│   │   │   └── hydra.log
│   │   ├── 17-14-30/
│   │   │   └── hydra.log
│   │   ├── 17-39-38/
│   │   │   └── hydra.log
│   │   ├── 17-42-36/
│   │   │   └── hydra.log
│   │   ├── 18-05-17/
│   │   │   └── hydra.log
│   │   ├── 18-06-15/
│   │   │   └── hydra.log
│   │   ├── 18-07-27/
│   │   │   └── hydra.log
│   │   ├── 20-27-13/
│   │   │   └── hydra.log
│   │   ├── 20-29-37/
│   │   │   └── hydra.log
│   │   ├── 20-31-13/
│   │   │   └── hydra.log
│   │   ├── 20-45-12/
│   │   │   └── hydra.log
│   │   ├── 20-46-45/
│   │   │   └── hydra.log
│   │   ├── 20-49-37/
│   │   │   └── hydra.log
│   │   ├── 21-15-45/
│   │   │   └── hydra.log
│   │   ├── 21-16-03/
│   │   │   └── hydra.log
│   │   ├── 21-37-55/
│   │   │   └── hydra.log
│   │   ├── 21-52-57/
│   │   │   └── hydra.log
│   │   ├── 21-55-51/
│   │   │   └── hydra.log
│   │   └── 22-16-25/
│   │       └── hydra.log
│   ├── 2026-07-03/
│   │   ├── 13-20-16/
│   │   │   └── hydra.log
│   │   ├── 13-49-42/
│   │   │   └── hydra.log
│   │   ├── 13-58-08/
│   │   │   └── hydra.log
│   │   ├── 14-33-17/
│   │   │   └── hydra.log
│   │   ├── 15-03-10/
│   │   │   └── hydra.log
│   │   ├── 15-08-55/
│   │   │   └── hydra.log
│   │   ├── 16-23-33/
│   │   │   └── hydra.log
│   │   ├── 16-24-35/
│   │   │   └── hydra.log
│   │   ├── 16-27-35/
│   │   │   └── hydra.log
│   │   ├── 16-29-48/
│   │   │   └── hydra.log
│   │   ├── 16-30-25/
│   │   │   └── hydra.log
│   │   ├── 16-30-43/
│   │   │   └── hydra.log
│   │   ├── 16-38-44/
│   │   │   └── hydra.log
│   │   ├── 17-14-08/
│   │   │   └── hydra.log
│   │   ├── 17-14-25/
│   │   │   └── hydra.log
│   │   ├── 17-32-24/
│   │   │   └── hydra.log
│   │   ├── 17-59-44/
│   │   │   └── hydra.log
│   │   ├── 18-00-58/
│   │   │   └── hydra.log
│   │   ├── 18-52-40/
│   │   │   └── hydra.log
│   │   ├── 18-52-59/
│   │   │   └── hydra.log
│   │   ├── 18-53-26/
│   │   │   └── hydra.log
│   │   ├── 18-56-16/
│   │   │   └── hydra.log
│   │   ├── 18-58-33/
│   │   │   └── hydra.log
│   │   ├── 19-25-00/
│   │   │   └── hydra.log
│   │   ├── 19-41-08/
│   │   │   └── hydra.log
│   │   ├── 19-42-10/
│   │   │   └── hydra.log
│   │   ├── 20-25-15/
│   │   │   └── hydra.log
│   │   ├── 20-26-26/
│   │   │   └── hydra.log
│   │   ├── 20-30-20/
│   │   │   └── hydra.log
│   │   ├── 20-33-28/
│   │   │   └── hydra.log
│   │   ├── 20-34-13/
│   │   │   └── hydra.log
│   │   └── 20-49-13/
│   │       └── hydra.log
│   ├── 2026-07-04/
│   │   ├── 14-20-14/
│   │   │   └── hydra.log
│   │   ├── 14-29-49/
│   │   │   └── hydra.log
│   │   ├── 14-34-34/
│   │   │   └── hydra.log
│   │   ├── 14-39-33/
│   │   │   └── hydra.log
│   │   ├── 14-42-47/
│   │   │   └── hydra.log
│   │   ├── 14-59-09/
│   │   │   └── hydra.log
│   │   ├── 15-01-20/
│   │   │   └── hydra.log
│   │   ├── 15-03-23/
│   │   │   └── hydra.log
│   │   ├── 15-05-09/
│   │   │   └── hydra.log
│   │   ├── 15-25-45/
│   │   │   └── hydra.log
│   │   ├── 15-29-02/
│   │   │   └── hydra.log
│   │   ├── 16-03-14/
│   │   │   └── hydra.log
│   │   ├── 16-14-21/
│   │   │   └── hydra.log
│   │   ├── 16-26-06/
│   │   │   └── hydra.log
│   │   ├── 16-49-37/
│   │   │   └── hydra.log
│   │   ├── 16-53-37/
│   │   │   └── hydra.log
│   │   ├── 17-03-42/
│   │   │   └── hydra.log
│   │   ├── 17-10-33/
│   │   │   └── hydra.log
│   │   ├── 17-16-29/
│   │   │   └── hydra.log
│   │   ├── 17-22-59/
│   │   │   └── hydra.log
│   │   ├── 17-26-34/
│   │   │   └── hydra.log
│   │   ├── 17-34-59/
│   │   │   └── hydra.log
│   │   ├── 17-37-20/
│   │   │   └── hydra.log
│   │   ├── 19-09-58/
│   │   │   └── hydra.log
│   │   ├── 19-16-18/
│   │   │   └── hydra.log
│   │   ├── 19-26-18/
│   │   │   └── hydra.log
│   │   ├── 19-37-45/
│   │   │   └── hydra.log
│   │   ├── 20-00-29/
│   │   │   └── hydra.log
│   │   ├── 20-01-10/
│   │   │   └── hydra.log
│   │   ├── 20-02-09/
│   │   │   └── hydra.log
│   │   ├── 20-09-11/
│   │   │   └── hydra.log
│   │   ├── 20-09-44/
│   │   │   └── hydra.log
│   │   ├── 20-10-11/
│   │   │   └── hydra.log
│   │   └── 20-18-44/
│   │       └── hydra.log
│   ├── 2026-07-05/
│   │   ├── 15-09-41/
│   │   │   └── hydra.log
│   │   ├── 15-13-05/
│   │   │   └── hydra.log
│   │   ├── 15-14-16/
│   │   │   └── hydra.log
│   │   ├── 15-20-28/
│   │   │   └── hydra.log
│   │   ├── 15-26-56/
│   │   │   └── hydra.log
│   │   ├── 15-29-13/
│   │   │   └── hydra.log
│   │   ├── 15-30-21/
│   │   │   └── hydra.log
│   │   ├── 15-39-28/
│   │   │   └── hydra.log
│   │   ├── 15-39-49/
│   │   │   └── hydra.log
│   │   ├── 15-40-41/
│   │   │   └── hydra.log
│   │   ├── 15-51-14/
│   │   │   └── hydra.log
│   │   ├── 15-52-02/
│   │   │   └── hydra.log
│   │   ├── 15-52-33/
│   │   │   └── hydra.log
│   │   ├── 16-05-13/
│   │   │   └── hydra.log
│   │   ├── 16-05-48/
│   │   │   └── hydra.log
│   │   ├── 16-06-39/
│   │   │   └── hydra.log
│   │   ├── 16-38-58/
│   │   │   └── hydra.log
│   │   ├── 16-45-10/
│   │   │   └── hydra.log
│   │   ├── 16-46-17/
│   │   │   └── hydra.log
│   │   ├── 16-57-07/
│   │   │   └── hydra.log
│   │   ├── 17-02-40/
│   │   │   └── hydra.log
│   │   ├── 17-24-32/
│   │   │   └── hydra.log
│   │   ├── 17-29-17/
│   │   │   └── hydra.log
│   │   ├── 17-39-21/
│   │   │   └── hydra.log
│   │   ├── 17-39-36/
│   │   │   └── hydra.log
│   │   ├── 17-40-52/
│   │   │   └── hydra.log
│   │   ├── 17-45-15/
│   │   │   └── hydra.log
│   │   ├── 17-53-03/
│   │   │   └── hydra.log
│   │   ├── 17-54-10/
│   │   │   └── hydra.log
│   │   ├── 17-59-29/
│   │   │   └── hydra.log
│   │   ├── 22-21-02/
│   │   │   └── hydra.log
│   │   ├── 22-24-27/
│   │   │   └── hydra.log
│   │   ├── 22-34-36/
│   │   │   └── hydra.log
│   │   ├── 22-56-45/
│   │   │   └── hydra.log
│   │   ├── 23-02-01/
│   │   │   └── hydra.log
│   │   ├── 23-10-16/
│   │   │   └── hydra.log
│   │   ├── 23-13-02/
...[tree truncated after 700 entries]

## Git Status

```text
## 20260704_13-44_Turn...origin/20260704_13-44_Turn [领先 2]
?? .ai-bridge/
```

## Recent Commits

```text
dde5174 (HEAD -> 20260704_13-44_Turn) 20260727_20-33_B随机距离和随机左右角度目标
55cf50d 20260714_16-00_修正urdf100mm偏置
c9628f9 (origin/20260704_13-44_Turn) 20260710_00-14_Turn_失败
72df537 20260709_02-39_Stand
641b25d 20260708_16-37_修改urdf+感知链路
bdb843e 20260708_02-02_Stand
91bc4ed 2026060704_20-47
b0360d2 20260626_02-00_forward
```

## Git Diff

```diff
(no output)
```

## Existing AI Bridge Context

--- .ai-bridge/current-plan.md ---
1 | # Current Plan
2 | 
3 | No plan written yet.
4 | 

--- .ai-bridge/agent-status.md ---
1 | # Agent Status
2 | 
3 | No implementation agent status written yet.
4 | 

--- .ai-bridge/implementation-diff.patch ---
1 | 

--- .ai-bridge/codex-status.md ---
1 | # Codex Status
2 | 
3 | No Codex status written yet.
4 | 

--- .ai-bridge/decisions.md ---
1 | # Decisions
2 | 
3 | 

--- .ai-bridge/open-questions.md ---
1 | # Open Questions
2 | 
3 | 

--- .ai-bridge/execution-log.jsonl ---
1 |

## Selected Files

Changed files detected: .ai-bridge/
Auto-include important root files: yes
Auto-include changed files: yes
Explicit selected paths: source/Ranger/Ranger/tasks/manager_based/ranger/agents/rsl_rl_custom_policy.py, source/Ranger/Ranger/tasks/manager_based/ranger/agents/rsl_rl_ppo_cfg.py, source/Ranger/Ranger/tasks/manager_based/ranger/agents/__init__.py, source/Ranger/Ranger/tasks/manager_based/ranger/ranger_env_cfg.py, source/Ranger/Ranger/tasks/manager_based/ranger/__init__.py, source/Ranger/Ranger/tasks/manager_based/ranger/mdp/observations.py, source/Ranger/Ranger/tasks/manager_based/ranger/forward_debug_env.py, scripts/rsl_rl/train.py, scripts/rsl_rl/play.py, scripts/rsl_rl/warm_start.py
Extra globs: none
Files included below: scripts/rsl_rl/play.py, scripts/rsl_rl/train.py, scripts/rsl_rl/warm_start.py, source/Ranger/Ranger/tasks/manager_based/ranger/__init__.py, source/Ranger/Ranger/tasks/manager_based/ranger/agents/__init__.py, source/Ranger/Ranger/tasks/manager_based/ranger/agents/rsl_rl_custom_policy.py, source/Ranger/Ranger/tasks/manager_based/ranger/agents/rsl_rl_ppo_cfg.py, source/Ranger/Ranger/tasks/manager_based/ranger/forward_debug_env.py, source/Ranger/Ranger/tasks/manager_based/ranger/mdp/observations.py, source/Ranger/Ranger/tasks/manager_based/ranger/ranger_env_cfg.py, README.md, .ai-bridge/

## File Contents

### scripts/rsl_rl/play.py

Bytes: 37461
SHA-256: 0e52031d2292b1a8d27fc5f3ecb1494bf4c51c04ce517f54a24f3aca008b253f
Lines: 1-787 of 787

```python
  1 | # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
  2 | # All rights reserved.
  3 | #
  4 | # SPDX-License-Identifier: BSD-3-Clause
  5 | 
  6 | """Script to play a checkpoint if an RL agent from RSL-RL."""
  7 | 
  8 | """Launch Isaac Sim Simulator first."""
  9 | 
 10 | import argparse
 11 | import csv
 12 | import os
 13 | import sys
 14 | 
 15 | from isaaclab.app import AppLauncher
 16 | 
 17 | # local imports
 18 | import cli_args  # isort: skip
 19 | 
 20 | # add argparse arguments
 21 | parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
 22 | parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
 23 | parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
 24 | parser.add_argument("--video_width", type=int, default=1920, help="Recorded viewport width in pixels.")
 25 | parser.add_argument("--video_height", type=int, default=1080, help="Recorded viewport height in pixels.")
 26 | parser.add_argument(
 27 |     "--show_sensor_debug",
 28 |     action="store_true",
 29 |     default=False,
 30 |     help="Keep lidar and ray-caster debug points visible in recorded videos.",
 31 | )
 32 | parser.add_argument(
 33 |     "--camera_mode",
 34 |     type=str,
 35 |     choices=("fixed", "follow", "overview", "overview_fixed"),
 36 |     default="follow",
 37 |     help=(
 38 |         "Recording camera mode. 'follow' tracks environment zero from behind; "
 39 |         "'overview' dynamically reframes all robots; 'overview_fixed' computes one overview pose "
 40 |         "from the initial robots and goals, then keeps it fixed."
 41 |     ),
 42 | )
 43 | parser.add_argument(
 44 |     "--overview_fixed_scale",
 45 |     type=float,
 46 |     default=None,
 47 |     help="Optional fixed overview camera scale. Larger values zoom farther out.",
 48 | )
 49 | parser.add_argument(
 50 |     "--overview_fixed_padding",
 51 |     type=float,
 52 |     default=2.0,
 53 |     help="Extra XY padding in meters when inferring the fixed overview framing.",
 54 | )
 55 | parser.add_argument(
 56 |     "--fixed_suspension_action",
 57 |     type=float,
 58 |     default=None,
 59 |     help="Evaluation-only override for all four suspension actions. Use 0.0 for nominal mid-stroke.",
 60 | )
 61 | parser.add_argument(
 62 |     "--stop_phase_suspension_action",
 63 |     type=float,
 64 |     default=None,
 65 |     help="Evaluation-only override applied to all four suspension actions after stop phase begins.",
 66 | )
 67 | parser.add_argument(
 68 |     "--suspension_action_scale",
 69 |     type=float,
 70 |     default=None,
 71 |     help="Evaluation-only multiplier for the four policy suspension actions. Use 1.0 for full authority.",
 72 | )
 73 | parser.add_argument(
 74 |     "--max_steps",
 75 |     type=int,
 76 |     default=None,
 77 |     help="Stop evaluation after this many policy steps even when video recording is disabled.",
 78 | )
 79 | parser.add_argument(
 80 |     "--evaluation_summary",
 81 |     action="store_true",
 82 |     default=False,
 83 |     help="Print termination counts and stopped-goal success rate at the end of play.",
 84 | )
 85 | parser.add_argument(
 86 |     "--evaluation_metrics_csv",
 87 |     type=str,
 88 |     default=None,
 89 |     help="Optional CSV path for episode-weighted full debug metrics collected during deterministic play.",
 90 | )
 91 | parser.add_argument(
 92 |     "--success_gate_trace_csv",
 93 |     type=str,
 94 |     default=None,
 95 |     help=(
 96 |         "Optional per-policy-step CSV trace for every stopped-goal success gate, including goal distance, "
 97 |         "base speed, yaw rate, roll/pitch, suspension tracking error, stop-phase state, and stable-step count."
 98 |     ),
 99 | )
100 | parser.add_argument(
101 |     "--video_subdir",
102 |     type=str,
103 |     default="play",
104 |     help="Subdirectory below the checkpoint run's videos directory.",
105 | )
106 | parser.add_argument(
107 |     "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
108 | )
109 | parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
110 | parser.add_argument("--task", type=str, default=None, help="Name of the task.")
111 | parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
112 | parser.add_argument(
113 |     "--use_pretrained_checkpoint",
114 |     action="store_true",
115 |     help="Use the pre-trained checkpoint from Nucleus.",
116 | )
117 | parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
118 | # append RSL-RL cli arguments
119 | cli_args.add_rsl_rl_args(parser)
120 | # append AppLauncher cli args
121 | AppLauncher.add_app_launcher_args(parser)
122 | # parse the arguments
123 | args_cli, hydra_args = parser.parse_known_args()
124 | if args_cli.fixed_suspension_action is not None and args_cli.stop_phase_suspension_action is not None:
125 |     raise ValueError(
126 |         "Use either --fixed_suspension_action or --stop_phase_suspension_action, not both."
127 |     )
128 | if args_cli.fixed_suspension_action is not None:
129 |     if not -1.0 <= float(args_cli.fixed_suspension_action) <= 1.0:
130 |         raise ValueError("--fixed_suspension_action must be within [-1, 1].")
131 |     os.environ["RANGER_FIXED_SUSPENSION_ACTION"] = str(float(args_cli.fixed_suspension_action))
132 | if args_cli.stop_phase_suspension_action is not None:
133 |     if not -1.0 <= float(args_cli.stop_phase_suspension_action) <= 1.0:
134 |         raise ValueError("--stop_phase_suspension_action must be within [-1, 1].")
135 |     os.environ["RANGER_STOP_PHASE_SUSPENSION_ACTION"] = str(
136 |         float(args_cli.stop_phase_suspension_action)
137 |     )
138 | if args_cli.suspension_action_scale is not None:
139 |     if not 0.0 <= float(args_cli.suspension_action_scale) <= 1.0:
140 |         raise ValueError("--suspension_action_scale must be within [0, 1].")
141 |     os.environ["RANGER_SUSPENSION_ACTION_SCALE"] = str(float(args_cli.suspension_action_scale))
142 | if args_cli.max_steps is not None and int(args_cli.max_steps) <= 0:
143 |     raise ValueError("--max_steps must be positive when provided.")
144 | # Always enable cameras to record video and pass the requested viewport resolution
145 | # through to SimulationApp before AppLauncher is constructed.
146 | if args_cli.video:
147 |     args_cli.enable_cameras = True
148 |     args_cli.width = max(int(args_cli.video_width), 320)
149 |     args_cli.height = max(int(args_cli.video_height), 240)
150 | 
151 | # clear out sys.argv for Hydra
152 | sys.argv = [sys.argv[0]] + hydra_args
153 | 
154 | # launch omniverse app
155 | app_launcher = AppLauncher(args_cli)
156 | simulation_app = app_launcher.app
157 | 
158 | """Rest everything follows."""
159 | 
160 | import gymnasium as gym
161 | import math
162 | import time
163 | import torch
164 | 
165 | from rsl_rl.runners import OnPolicyRunner
166 | import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner
167 | 
168 | from isaaclab.envs import (
169 |     DirectMARLEnv,
170 |     DirectMARLEnvCfg,
171 |     DirectRLEnvCfg,
172 |     ManagerBasedRLEnvCfg,
173 |     multi_agent_to_single_agent,
174 | )
175 | from isaaclab.utils.assets import retrieve_file_path
176 | from isaaclab.utils.dict import print_dict
177 | from isaaclab.utils.math import euler_xyz_from_quat
178 | try:
179 |     from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
180 | except ModuleNotFoundError:
181 |     def get_published_pretrained_checkpoint(*args, **kwargs):
182 |         return None
183 | 
184 | from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
185 | 
186 | import isaaclab_tasks  # noqa: F401
187 | from isaaclab_tasks.utils import get_checkpoint_path
188 | from isaaclab_tasks.utils.hydra import hydra_task_config
189 | 
190 | import Ranger.tasks  # noqa: F401
191 | from Ranger.tasks.manager_based.ranger import mdp as ranger_mdp
192 | from Ranger.tasks.manager_based.ranger.agents import RangerTerrainActorCritic
193 | 
194 | rsl_on_policy_runner.RangerTerrainActorCritic = RangerTerrainActorCritic
195 | 
196 | 
197 | @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
198 | def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
199 |     """Play with RSL-RL agent."""
200 |     task_name = args_cli.task.split(":")[-1]
201 |     # override configurations with non-hydra CLI arguments
202 |     agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
203 |     env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
204 | 
205 |     # set the environment seed
206 |     # note: certain randomizations occur in the environment initialization so we set the seed here
207 |     env_cfg.seed = agent_cfg.seed
208 |     env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
209 |     agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
210 | 
211 |     if args_cli.video and not args_cli.show_sensor_debug:
212 |         for sensor_name in ("mid360_lidar", "avia_lidar", "d435i_camera"):
213 |             sensor_cfg = getattr(env_cfg.scene, sensor_name, None)
214 |             if sensor_cfg is not None and hasattr(sensor_cfg, "debug_vis"):
215 |                 sensor_cfg.debug_vis = False
216 | 
217 |     # specify directory for logging experiments
218 |     log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
219 |     log_root_path = os.path.abspath(log_root_path)
220 |     print(f"[INFO] Loading experiment from directory: {log_root_path}")
221 |     if args_cli.use_pretrained_checkpoint:
222 |         resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
223 |         if not resume_path:
224 |             print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
225 |             return
226 |     elif args_cli.checkpoint:
227 |         resume_path = retrieve_file_path(args_cli.checkpoint)
228 |     else:
229 |         resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
230 | 
231 |     log_dir = os.path.dirname(resume_path)
232 | 
233 |     # create isaac environment
234 |     env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
235 | 
236 |     # convert to single-agent instance if required by the RL algorithm
237 |     if isinstance(env.unwrapped, DirectMARLEnv):
238 |         env = multi_agent_to_single_agent(env)
239 | 
240 |     # wrap for video recording
241 |     if args_cli.video:
242 |         video_kwargs = {
243 |             "video_folder": os.path.join(log_dir, "videos", os.path.basename(args_cli.video_subdir)),
244 |             "step_trigger": lambda step: step == 0,
245 |             "video_length": args_cli.video_length,
246 |             "disable_logger": True,
247 |         }
248 |         print("[INFO] Recording videos during training.")
249 |         print_dict(video_kwargs, nesting=4)
250 |         env = gym.wrappers.RecordVideo(env, **video_kwargs)
251 | 
252 |     # wrap around environment for rsl-rl
253 |     env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
254 | 
255 |     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
256 |     # load previously trained model
257 |     ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
258 |     ppo_runner.load(resume_path)
259 | 
260 |     # obtain the trained policy for inference
261 |     policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
262 | 
263 |     # extract the neural network module
264 |     # we do this in a try-except to maintain backwards compatibility.
265 |     try:
266 |         # version 2.3 onwards
267 |         policy_nn = ppo_runner.alg.policy
268 |     except AttributeError:
269 |         # version 2.2 and below
270 |         policy_nn = ppo_runner.alg.actor_critic
271 | 
272 |     # export policy to onnx/jit
273 |     export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
274 |     obs_normalizer = getattr(ppo_runner, "obs_normalizer", None)
275 |     if obs_normalizer is None:
276 |         obs_normalizers = getattr(ppo_runner, "obs_normalizers", None)
277 |         if isinstance(obs_normalizers, dict):
278 |             obs_normalizer = obs_normalizers.get("policy", None)
279 |         else:
280 |             obs_normalizer = obs_normalizers
281 |     try:
282 |         export_policy_as_jit(policy_nn, obs_normalizer, path=export_model_dir, filename="policy.pt")
283 |     except Exception as e:
284 |         print(f"[WARN] Failed to export JIT policy, continue play without export: {e}")
285 |     try:
286 |         export_policy_as_onnx(policy_nn, normalizer=obs_normalizer, path=export_model_dir, filename="policy.onnx")
287 |     except Exception as e:
288 |         print(f"[WARN] Failed to export ONNX policy, continue play without export: {e}")
289 | 
290 |     dt = env.unwrapped.step_dt
291 | 
292 |     # reset environment
293 |     obs_result = env.get_observations()
294 |     obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result
295 |     timestep = 0
296 |     termination_counts: dict[str, int] = {
297 |         str(name): 0 for name in getattr(env.unwrapped.termination_manager, "active_terms", ())
298 |     }
299 |     completed_episodes = 0
300 |     completed_episode_steps_sum = 0
301 |     success_episode_steps_sum = 0
302 |     success_stop_entry_steps_sum = 0
303 |     success_stop_phase_steps_sum = 0
304 |     success_with_stop_entry_count = 0
305 |     evaluation_metric_weighted_sums: dict[str, float] = {}
306 |     evaluation_metric_weights: dict[str, int] = {}
307 |     episode_age_steps = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
308 |     stop_entry_age_steps = torch.full(
309 |         (env.unwrapped.num_envs,), -1, dtype=torch.long, device=env.unwrapped.device
310 |     )
311 |     episode_ids = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
312 | 
313 |     success_gate_trace_file = None
314 |     success_gate_trace_writer = None
315 |     success_gate_trace_fields = [
316 |         "policy_step",
317 |         "env_id",
318 |         "episode_id",
319 |         "episode_step_before_action",
320 |         "goal_side",
321 |         "goal_distance_m",
322 |         "heading_error_rad",
323 |         "stop_phase_enter_distance_m",
324 |         "success_distance_m",
325 |         "distance_lt_stop_enter",
326 |         "distance_ok",
327 |         "velocity_toward_goal_mps",
328 |         "velocity_tangential_to_goal_mps",
329 |         "base_xy_speed_mps",
330 |         "max_xy_speed_mps",
331 |         "xy_speed_ok",
332 |         "yaw_rate_signed_radps",
333 |         "yaw_rate_abs_radps",
334 |         "max_yaw_rate_radps",
335 |         "yaw_rate_ok",
336 |         "roll_abs_rad",
337 |         "max_roll_rad",
338 |         "roll_ok",
339 |         "pitch_abs_rad",
340 |         "max_pitch_rad",
341 |         "pitch_ok",
342 |         "stroke_tracking_error_max_m",
343 |         "max_stroke_tracking_error_m",
344 |         "stroke_tracking_ok",
345 |         "posture_ok",
346 |         "stop_phase_active",
347 |         "stop_phase_first_observed_before_action",
348 |         "stable_steps_before_action",
349 |         "required_hold_steps",
350 |         "stable_fraction_before_action",
351 |         "settled_now_before_action",
352 |         "success_ready_before_action",
353 |         "failed_gates_before_action",
354 |         "policy_wheel_raw_lb",
355 |         "policy_wheel_raw_lf",
356 |         "policy_wheel_raw_rf",
357 |         "policy_wheel_raw_rb",
358 |         "current_wheel_target_semantic_lb_radps",
359 |         "current_wheel_target_semantic_lf_radps",
360 |         "current_wheel_target_semantic_rf_radps",
361 |         "current_wheel_target_semantic_rb_radps",
362 |         "termination_any_after_step",
363 |         "stopped_goal_reached_after_step",
364 |         "time_out_after_step",
365 |         "termination_terms_after_step",
366 |     ]
367 |     if args_cli.success_gate_trace_csv is not None:
368 |         trace_path = os.path.abspath(os.path.expanduser(args_cli.success_gate_trace_csv))
369 |         trace_dir = os.path.dirname(trace_path)
370 |         if trace_dir:
371 |             os.makedirs(trace_dir, exist_ok=True)
372 |         success_gate_trace_file = open(trace_path, "w", newline="", encoding="utf-8")
373 |         success_gate_trace_writer = csv.DictWriter(
374 |             success_gate_trace_file,
375 |             fieldnames=success_gate_trace_fields,
376 |             extrasaction="ignore",
377 |         )
378 |         success_gate_trace_writer.writeheader()
379 |         print(f"[SuccessGateTrace] CSV writing to: {trace_path}")
380 | 
381 |     camera_eye_state: torch.Tensor | None = None
382 |     camera_target_state: torch.Tensor | None = None
383 | 
384 |     def update_recording_camera() -> None:
385 |         """Track environment zero during video recording without affecting policy observations."""
386 | 
387 |         nonlocal camera_eye_state, camera_target_state
388 |         if not args_cli.video or args_cli.camera_mode == "fixed":
389 |             return
390 |         if args_cli.camera_mode == "overview_fixed" and camera_eye_state is not None:
391 |             return
392 |         robot = env.unwrapped.scene["robot"]
393 |         root_positions = robot.data.root_pos_w.detach().cpu()
394 |         root_pos = root_positions[0]
395 |         if args_cli.camera_mode in {"overview", "overview_fixed"}:
396 |             framing_positions = root_positions
397 |             if args_cli.camera_mode == "overview_fixed":
398 |                 goal_positions = getattr(env.unwrapped, "_ranger_short_goal_pos_w", None)
399 |                 if isinstance(goal_positions, torch.Tensor) and goal_positions.shape[0] == root_positions.shape[0]:
400 |                     framing_positions = torch.cat((root_positions, goal_positions.detach().cpu()), dim=0)
401 |             xy_min = framing_positions[:, :2].min(dim=0).values
402 |             xy_max = framing_positions[:, :2].max(dim=0).values
403 |             center_xy = 0.5 * (xy_min + xy_max)
404 |             horizontal_extent = float(torch.max(xy_max - xy_min).item())
405 |             if args_cli.camera_mode == "overview_fixed":
406 |                 horizontal_extent += 2.0 * max(float(args_cli.overview_fixed_padding), 0.0)
407 |                 if args_cli.overview_fixed_scale is None:
408 |                     camera_scale = max(9.0, 1.35 * horizontal_extent + 6.0)
409 |                 else:
410 |                     camera_scale = max(float(args_cli.overview_fixed_scale), 1.0)
411 |             else:
412 |                 # Keep less empty margin than the original fleet view so each robot occupies
413 |                 # more pixels while still scaling out when the environments separate.
414 |                 camera_scale = max(9.0, 1.35 * horizontal_extent + 6.0)
415 |             desired_target = torch.tensor([float(center_xy[0]), float(center_xy[1]), 0.7])
416 |             # Oblique fleet view: keep all environments framed while exposing the wheel sidewalls
417 |             # well enough to distinguish front/rear wheel rotation direction.
418 |             desired_eye = desired_target + torch.tensor(
419 |                 [0.55 * camera_scale, -0.80 * camera_scale, 0.65 * camera_scale]
420 |             )
421 |         else:
422 |             _, _, yaw_tensor = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
423 |             yaw = float(yaw_tensor[0].item())
424 |             cos_yaw = math.cos(yaw)
425 |             sin_yaw = math.sin(yaw)
426 |             offset_body = (-4.5, -3.5)
427 |             lookahead_body = (1.0, 0.0)
428 |             offset_world = torch.tensor(
429 |                 [
430 |                     cos_yaw * offset_body[0] - sin_yaw * offset_body[1],
431 |                     sin_yaw * offset_body[0] + cos_yaw * offset_body[1],
432 |                     2.8,
433 |                 ]
434 |             )
435 |             lookahead_world = torch.tensor(
436 |                 [
437 |                     cos_yaw * lookahead_body[0] - sin_yaw * lookahead_body[1],
438 |                     sin_yaw * lookahead_body[0] + cos_yaw * lookahead_body[1],
439 |                     0.7,
440 |                 ]
441 |             )
442 |             desired_eye = root_pos + offset_world
443 |             desired_target = root_pos + lookahead_world
444 | 
445 |         smoothing = 0.12
446 |         if camera_eye_state is None:
447 |             camera_eye_state = desired_eye
448 |             camera_target_state = desired_target
449 |         else:
450 |             camera_eye_state = torch.lerp(camera_eye_state, desired_eye, smoothing)
451 |             camera_target_state = torch.lerp(camera_target_state, desired_target, smoothing)
452 |         env.unwrapped.sim.set_camera_view(camera_eye_state.tolist(), camera_target_state.tolist())
453 | 
454 |     def collect_success_gate_rows(
455 |         actions: torch.Tensor,
456 |         stop_phase_active: torch.Tensor,
457 |         first_observed_stop: torch.Tensor,
458 |     ) -> list[dict[str, object]]:
459 |         """Collect every stopped-goal success gate before the current env.step()."""
460 | 
461 |         unwrapped = env.unwrapped
462 |         robot = unwrapped.scene["robot"]
463 |         target_vec_b, goal_distance, heading_error = ranger_mdp.short_goal_target_body(unwrapped)
464 |         base_xy_velocity = robot.data.root_lin_vel_b[:, :2]
465 |         base_xy_speed = torch.linalg.vector_norm(base_xy_velocity, dim=1)
466 |         target_dir_b = target_vec_b[:, :2] / torch.clamp(goal_distance.unsqueeze(1), min=1.0e-6)
467 |         velocity_toward_goal = torch.sum(base_xy_velocity * target_dir_b, dim=1)
468 |         velocity_tangential_to_goal = (
469 |             target_dir_b[:, 0] * base_xy_velocity[:, 1]
470 |             - target_dir_b[:, 1] * base_xy_velocity[:, 0]
471 |         )
472 |         yaw_rate_signed = robot.data.root_ang_vel_b[:, 2]
473 |         yaw_rate_abs = torch.abs(yaw_rate_signed)
474 |         roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
475 |         roll_abs = torch.abs(roll)
476 |         pitch_abs = torch.abs(pitch)
477 | 
478 |         leg_term = unwrapped.action_manager.get_term("leg_hydraulic")
479 |         stroke_error = torch.abs(leg_term.stroke_desired - leg_term.stroke_actual)
480 |         stroke_tracking_error_max = torch.max(stroke_error, dim=1).values
481 | 
482 |         wheel_term = unwrapped.action_manager.get_term("wheel_motor_csv")
483 |         wheel_forward_sign = getattr(unwrapped, "_wheel_forward_sign", None)
484 |         if not isinstance(wheel_forward_sign, torch.Tensor):
485 |             wheel_forward_sign = torch.ones(4, device=unwrapped.device, dtype=wheel_term.velocity_target.dtype)
486 |         wheel_forward_sign = wheel_forward_sign.to(
487 |             device=unwrapped.device,
488 |             dtype=wheel_term.velocity_target.dtype,
489 |         )
490 |         current_wheel_target_semantic = wheel_term.velocity_target * wheel_forward_sign
491 | 
492 |         stop_enter_distance = float(getattr(unwrapped, "_short_goal_stop_phase_enter_distance", 0.30))
493 |         success_distance = float(getattr(unwrapped, "_short_goal_stop_success_distance", 0.50))
494 |         max_xy_speed = float(getattr(unwrapped, "_short_goal_stop_max_xy_speed", 0.15))
495 |         max_yaw_rate = float(getattr(unwrapped, "_short_goal_stop_max_yaw_rate", 0.20))
496 |         max_roll = float(getattr(unwrapped, "_short_goal_stop_max_roll", 0.035))
497 |         max_pitch = float(getattr(unwrapped, "_short_goal_stop_max_pitch", 0.035))
498 |         max_stroke_tracking_error = float(
499 |             getattr(unwrapped, "_short_goal_stop_max_stroke_tracking_error", 0.02)
500 |         )
501 |         required_hold_steps = int(getattr(unwrapped, "_short_goal_stop_required_hold_steps", 60))
502 |         stable_steps = getattr(unwrapped, "_short_goal_stop_phase_stable_steps", None)
503 |         if not isinstance(stable_steps, torch.Tensor):
504 |             stable_steps = torch.zeros(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
505 | 
506 |         distance_lt_stop_enter = goal_distance < stop_enter_distance
507 |         distance_ok = goal_distance < success_distance
508 |         xy_speed_ok = base_xy_speed < max_xy_speed
509 |         yaw_rate_ok = yaw_rate_abs < max_yaw_rate
510 |         roll_ok = roll_abs <= max_roll
511 |         pitch_ok = pitch_abs <= max_pitch
512 |         stroke_tracking_ok = stroke_tracking_error_max <= max_stroke_tracking_error
513 |         posture_ok = roll_ok & pitch_ok & stroke_tracking_ok
514 |         settled_now = stop_phase_active & distance_ok & xy_speed_ok & yaw_rate_ok & posture_ok
515 |         success_ready = settled_now & (stable_steps >= required_hold_steps)
516 | 
517 |         initial_side_sign = getattr(unwrapped, "_short_goal_initial_side_sign", None)
518 |         if not isinstance(initial_side_sign, torch.Tensor):
519 |             initial_side_sign = torch.sign(heading_error)
520 | 
521 |         rows: list[dict[str, object]] = []
522 |         wheel_names = ("lb", "lf", "rf", "rb")
523 |         for env_id in range(unwrapped.num_envs):
524 |             side_sign = float(initial_side_sign[env_id].item())
525 |             goal_side = "left" if side_sign > 0.0 else ("right" if side_sign < 0.0 else "straight")
526 |             failed_gates: list[str] = []
527 |             if not bool(stop_phase_active[env_id].item()):
528 |                 failed_gates.append("stop_phase")
529 |             if not bool(distance_ok[env_id].item()):
530 |                 failed_gates.append("distance")
531 |             if not bool(xy_speed_ok[env_id].item()):
532 |                 failed_gates.append("xy_speed")
533 |             if not bool(yaw_rate_ok[env_id].item()):
534 |                 failed_gates.append("yaw_rate")
535 |             if not bool(roll_ok[env_id].item()):
536 |                 failed_gates.append("roll")
537 |             if not bool(pitch_ok[env_id].item()):
538 |                 failed_gates.append("pitch")
539 |             if not bool(stroke_tracking_ok[env_id].item()):
540 |                 failed_gates.append("stroke_tracking")
541 |             if bool(settled_now[env_id].item()) and int(stable_steps[env_id].item()) < required_hold_steps:
542 |                 failed_gates.append("hold_steps")
543 |             row: dict[str, object] = {
544 |                 "policy_step": timestep,
545 |                 "env_id": env_id,
546 |                 "episode_id": int(episode_ids[env_id].item()),
547 |                 "episode_step_before_action": int(episode_age_steps[env_id].item()),
548 |                 "goal_side": goal_side,
549 |                 "goal_distance_m": float(goal_distance[env_id].item()),
550 |                 "heading_error_rad": float(heading_error[env_id].item()),
551 |                 "stop_phase_enter_distance_m": stop_enter_distance,
552 |                 "success_distance_m": success_distance,
553 |                 "distance_lt_stop_enter": int(distance_lt_stop_enter[env_id].item()),
554 |                 "distance_ok": int(distance_ok[env_id].item()),
555 |                 "velocity_toward_goal_mps": float(velocity_toward_goal[env_id].item()),
556 |                 "velocity_tangential_to_goal_mps": float(velocity_tangential_to_goal[env_id].item()),
557 |                 "base_xy_speed_mps": float(base_xy_speed[env_id].item()),
558 |                 "max_xy_speed_mps": max_xy_speed,
559 |                 "xy_speed_ok": int(xy_speed_ok[env_id].item()),
560 |                 "yaw_rate_signed_radps": float(yaw_rate_signed[env_id].item()),
561 |                 "yaw_rate_abs_radps": float(yaw_rate_abs[env_id].item()),
562 |                 "max_yaw_rate_radps": max_yaw_rate,
563 |                 "yaw_rate_ok": int(yaw_rate_ok[env_id].item()),
564 |                 "roll_abs_rad": float(roll_abs[env_id].item()),
565 |                 "max_roll_rad": max_roll,
566 |                 "roll_ok": int(roll_ok[env_id].item()),
567 |                 "pitch_abs_rad": float(pitch_abs[env_id].item()),
568 |                 "max_pitch_rad": max_pitch,
569 |                 "pitch_ok": int(pitch_ok[env_id].item()),
570 |                 "stroke_tracking_error_max_m": float(stroke_tracking_error_max[env_id].item()),
571 |                 "max_stroke_tracking_error_m": max_stroke_tracking_error,
572 |                 "stroke_tracking_ok": int(stroke_tracking_ok[env_id].item()),
573 |                 "posture_ok": int(posture_ok[env_id].item()),
574 |                 "stop_phase_active": int(stop_phase_active[env_id].item()),
575 |                 "stop_phase_first_observed_before_action": int(first_observed_stop[env_id].item()),
576 |                 "stable_steps_before_action": int(stable_steps[env_id].item()),
577 |                 "required_hold_steps": required_hold_steps,
578 |                 "stable_fraction_before_action": min(
579 |                     float(stable_steps[env_id].item()) / max(float(required_hold_steps), 1.0),
580 |                     1.0,
581 |                 ),
582 |                 "settled_now_before_action": int(settled_now[env_id].item()),
583 |                 "success_ready_before_action": int(success_ready[env_id].item()),
584 |                 "failed_gates_before_action": ";".join(failed_gates),
585 |                 "termination_any_after_step": 0,
586 |                 "stopped_goal_reached_after_step": 0,
587 |                 "time_out_after_step": 0,
588 |                 "termination_terms_after_step": "",
589 |             }
590 |             for wheel_idx, wheel_name in enumerate(wheel_names):
591 |                 row[f"policy_wheel_raw_{wheel_name}"] = float(actions[env_id, wheel_idx + 4].item())
592 |                 row[f"current_wheel_target_semantic_{wheel_name}_radps"] = float(
593 |                     current_wheel_target_semantic[env_id, wheel_idx].item()
594 |                 )
595 |             rows.append(row)
596 |         return rows
597 | 
598 |     update_recording_camera()
599 | 
600 |     # simulate environment
601 |     while simulation_app.is_running():
602 |         start_time = time.time()
603 |         # run everything in inference mode
604 |         with torch.inference_mode():
605 |             collect_episode_state = args_cli.evaluation_summary or success_gate_trace_writer is not None
606 |             stop_phase_active = torch.zeros(
607 |                 env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device
608 |             )
609 |             first_observed_stop = torch.zeros_like(stop_phase_active)
610 |             if collect_episode_state:
611 |                 # Sample the latched stop-phase state before env.step(). Successful
612 |                 # environments are reset inside env.step(), so post-step sampling loses it.
613 |                 stop_phase_state = getattr(env.unwrapped, "_short_goal_stop_phase_active", None)
614 |                 if isinstance(stop_phase_state, torch.Tensor):
615 |                     stop_phase_active = stop_phase_state.to(dtype=torch.bool)
616 |                 newly_entered_stop = stop_phase_active & (stop_entry_age_steps < 0)
617 |                 first_observed_stop = newly_entered_stop.clone()
618 |                 stop_entry_age_steps[newly_entered_stop] = episode_age_steps[newly_entered_stop]
619 | 
620 |             # agent stepping
621 |             actions = policy(obs)
622 |             success_gate_rows = None
623 |             if success_gate_trace_writer is not None:
624 |                 success_gate_rows = collect_success_gate_rows(
625 |                     actions,
626 |                     stop_phase_active,
627 |                     first_observed_stop,
628 |                 )
629 | 
630 |             # env stepping
631 |             step_result = env.step(actions)
632 |             if len(step_result) == 5:
633 |                 obs, _, _, _, step_info = step_result
634 |             else:
635 |                 obs, _, _, step_info = step_result
636 | 
637 |             if collect_episode_state:
638 |                 episode_age_steps += 1
639 |                 term_manager = env.unwrapped.termination_manager
640 |                 term_any = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
641 |                 success_mask = torch.zeros_like(term_any)
642 |                 timeout_mask = torch.zeros_like(term_any)
643 |                 termination_names_by_env: list[list[str]] = [
644 |                     [] for _ in range(env.unwrapped.num_envs)
645 |                 ]
646 |                 for term_idx, term_name in enumerate(getattr(term_manager, "active_terms", ())):
647 |                     term_done = term_manager._term_dones[:, term_idx]
648 |                     term_name_str = str(term_name)
649 |                     if args_cli.evaluation_summary:
650 |                         termination_counts[term_name_str] = termination_counts.get(term_name_str, 0) + int(
651 |                             term_done.sum().item()
652 |                         )
653 |                     for env_id in torch.nonzero(term_done, as_tuple=False).flatten().tolist():
654 |                         termination_names_by_env[int(env_id)].append(term_name_str)
655 |                     if term_name_str == "stopped_goal_reached":
656 |                         success_mask |= term_done
657 |                     if term_name_str == "time_out":
658 |                         timeout_mask |= term_done
659 |                     term_any |= term_done
660 | 
661 |                 if success_gate_rows is not None:
662 |                     for env_id, row in enumerate(success_gate_rows):
663 |                         row["termination_any_after_step"] = int(term_any[env_id].item())
664 |                         row["stopped_goal_reached_after_step"] = int(success_mask[env_id].item())
665 |                         row["time_out_after_step"] = int(timeout_mask[env_id].item())
666 |                         row["termination_terms_after_step"] = ";".join(termination_names_by_env[env_id])
667 |                     success_gate_trace_writer.writerows(success_gate_rows)
668 |                     if timestep % 60 == 0:
669 |                         success_gate_trace_file.flush()
670 | 
671 |                 if args_cli.evaluation_summary:
672 |                     completed_episode_steps_sum += int(episode_age_steps[term_any].sum().item())
673 |                     success_episode_steps_sum += int(episode_age_steps[success_mask].sum().item())
674 |                     valid_success_stop = success_mask & (stop_entry_age_steps >= 0)
675 |                     success_stop_entry_steps_sum += int(stop_entry_age_steps[valid_success_stop].sum().item())
676 |                     success_stop_phase_steps_sum += int(
677 |                         (episode_age_steps[valid_success_stop] - stop_entry_age_steps[valid_success_stop]).sum().item()
678 |                     )
679 |                     success_with_stop_entry_count += int(valid_success_stop.sum().item())
680 |                     completed_this_step = int(term_any.sum().item())
681 |                     completed_episodes += completed_this_step
682 |                     if completed_this_step > 0:
683 |                         full_log = step_info.get("full_log", {}) if isinstance(step_info, dict) else {}
684 |                         if not full_log:
685 |                             env_extras = getattr(env.unwrapped, "extras", {})
686 |                             full_log = env_extras.get("full_log", {}) if isinstance(env_extras, dict) else {}
687 |                         if isinstance(full_log, dict):
688 |                             for metric_name, metric_value in full_log.items():
689 |                                 if not str(metric_name).startswith("Metrics/short_goal/"):
690 |                                     continue
691 |                                 try:
692 |                                     scalar_value = float(metric_value)
693 |                                 except (TypeError, ValueError):
694 |                                     continue
695 |                                 evaluation_metric_weighted_sums[metric_name] = (
696 |                                     evaluation_metric_weighted_sums.get(metric_name, 0.0)
697 |                                     + scalar_value * completed_this_step
698 |                                 )
699 |                                 evaluation_metric_weights[metric_name] = (
700 |                                     evaluation_metric_weights.get(metric_name, 0) + completed_this_step
701 |                                 )
702 |                 episode_ids[term_any] += 1
703 |                 episode_age_steps[term_any] = 0
704 |                 stop_entry_age_steps[term_any] = -1
705 |         update_recording_camera()
706 |         timestep += 1
707 |         if args_cli.video:
708 |             # Exit the play loop after recording one video
709 |             if timestep >= args_cli.video_length:
710 |                 break
711 |         if args_cli.max_steps is not None and timestep >= int(args_cli.max_steps):
712 |             break
713 | 
714 |         # time delay for real-time evaluation
715 |         sleep_time = dt - (time.time() - start_time)
716 |         if args_cli.real_time and sleep_time > 0:
717 |             time.sleep(sleep_time)
718 | 
719 |     if success_gate_trace_file is not None:
720 |         success_gate_trace_file.flush()
721 |         success_gate_trace_file.close()
722 | 
723 |     if args_cli.evaluation_summary:
724 |         success_count = int(termination_counts.get("stopped_goal_reached", 0))
725 |         timeout_count = int(termination_counts.get("time_out", 0))
726 |         other_failure_count = max(completed_episodes - success_count - timeout_count, 0)
727 |         success_rate = success_count / max(completed_episodes, 1)
728 |         mean_completed_steps = completed_episode_steps_sum / max(completed_episodes, 1)
729 |         mean_success_steps = success_episode_steps_sum / max(success_count, 1)
730 |         mean_stop_entry_steps = success_stop_entry_steps_sum / max(success_with_stop_entry_count, 1)
731 |         mean_stop_phase_steps = success_stop_phase_steps_sum / max(success_with_stop_entry_count, 1)
732 |         print("[EvaluationSummary]")
733 |         print(f"  policy_steps: {timestep}")
734 |         print(f"  completed_episodes: {completed_episodes}")
735 |         print(f"  stopped_goal_reached: {success_count}")
736 |         print(f"  time_out: {timeout_count}")
737 |         print(f"  other_failures: {other_failure_count}")
738 |         print(f"  stopped_goal_success_rate: {success_rate:.6f}")
739 |         print(f"  mean_completed_episode_steps: {mean_completed_steps:.3f}")
740 |         print(f"  mean_success_episode_steps: {mean_success_steps:.3f}")
741 |         print(f"  success_with_stop_entry_count: {success_with_stop_entry_count}")
742 |         print(f"  mean_steps_to_stop_phase: {mean_stop_entry_steps:.3f}")
743 |         print(f"  mean_steps_stop_phase_to_success: {mean_stop_phase_steps:.3f}")
744 |         print(f"  termination_counts: {termination_counts}")
745 | 
746 |         evaluation_metric_means = {
747 |             name: evaluation_metric_weighted_sums[name] / max(evaluation_metric_weights[name], 1)
748 |             for name in evaluation_metric_weighted_sums
749 |         }
750 |         if evaluation_metric_means:
751 |             print("[EvaluationMetrics]")
752 |             for metric_name in sorted(evaluation_metric_means):
753 |                 print(
754 |                     f"  {metric_name}: {evaluation_metric_means[metric_name]:.8f} "
755 |                     f"(episodes={evaluation_metric_weights[metric_name]})"
756 |                 )
757 |         else:
758 |             print("[EvaluationMetrics] No completed-episode full_log metrics were collected.")
759 | 
760 |         if args_cli.evaluation_metrics_csv is not None:
761 |             metrics_csv_path = os.path.abspath(os.path.expanduser(args_cli.evaluation_metrics_csv))
762 |             metrics_csv_dir = os.path.dirname(metrics_csv_path)
763 |             if metrics_csv_dir:
764 |                 os.makedirs(metrics_csv_dir, exist_ok=True)
765 |             with open(metrics_csv_path, "w", newline="", encoding="utf-8") as csv_file:
766 |                 writer = csv.writer(csv_file)
767 |                 writer.writerow(("metric", "value", "episode_weight"))
768 |                 for metric_name in sorted(evaluation_metric_means):
769 |                     writer.writerow(
770 |                         (
771 |                             metric_name,
772 |                             f"{evaluation_metric_means[metric_name]:.10g}",
773 |                             evaluation_metric_weights[metric_name],
774 |                         )
775 |                     )
776 |             print(f"[EvaluationMetrics] CSV written to: {metrics_csv_path}")
777 | 
778 |     # close the simulator
779 |     env.close()
780 | 
781 | 
782 | if __name__ == "__main__":
783 |     # run the main function
784 |     main()
785 |     # close sim app
786 |     simulation_app.close()
787 | 
```

### scripts/rsl_rl/train.py

Bytes: 12667
SHA-256: e63afbd835253a50b35693541929548bc48c0ac44569dfe452397566c73f2cfb
Lines: 1-314 of 314

```python
  1 | # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
  2 | # All rights reserved.
  3 | #
  4 | # SPDX-License-Identifier: BSD-3-Clause
  5 | 
  6 | """Script to train RL agent with RSL-RL."""
  7 | 
  8 | """Launch Isaac Sim Simulator first."""
  9 | 
 10 | import argparse
 11 | import sys
 12 | 
 13 | from isaaclab.app import AppLauncher
 14 | 
 15 | # local imports
 16 | import cli_args  # isort: skip
 17 | from export_iteration_metrics import export_tensorboard_scalars  # isort: skip
 18 | 
 19 | 
 20 | # add argparse arguments
 21 | parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
 22 | parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
 23 | parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
 24 | parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
 25 | parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
 26 | parser.add_argument("--task", type=str, default=None, help="Name of the task.")
 27 | parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
 28 | parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
 29 | parser.add_argument(
 30 |     "--disable_iteration_metrics_csv",
 31 |     action="store_true",
 32 |     default=False,
 33 |     help="Disable automatic TensorBoard scalar export to per-iteration CSV files.",
 34 | )
 35 | parser.add_argument(
 36 |     "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
 37 | )
 38 | parser.add_argument(
 39 |     "--warm_start_checkpoint",
 40 |     type=str,
 41 |     default=None,
 42 |     help="Checkpoint used only to initialize actor parameters.",
 43 | )
 44 | parser.add_argument(
 45 |     "--warm_start_mode",
 46 |     type=str,
 47 |     choices=(
 48 |         "actor_suspension",
 49 |         "actor_wheel",
 50 |         "actor_suspension_only",
 51 |         "actor_wheel_reset_suspension",
 52 |         "actor_wheel_reset_final",
 53 |         "actor_reset_heads",
 54 |         "actor_heads",
 55 |         "actor_full",
 56 |     ),
 57 |     default=None,
 58 |     help=(
 59 |         "Actor warm-start mode. "
 60 |         "'actor_suspension' loads the shared encoders, actor trunk, "
 61 |         "and suspension head while keeping the wheel head random. "
 62 |         "'actor_wheel' loads the complete actor and trains only the wheel head. "
 63 |         "'actor_suspension_only' loads the complete actor and trains only the suspension head. "
 64 |         "'actor_wheel_reset_suspension' loads the shared actor and wheel head, resets the suspension head, "
 65 |         "and trains only the suspension head. "
 66 |         "'actor_wheel_reset_final' loads the complete actor, resets only the wheel head's final Linear, "
 67 |         "and trains only that final Linear. "
 68 |         "'actor_reset_heads' loads only the shared encoders/trunk, resets both action heads, "
 69 |         "and trains both action heads. "
 70 |         "'actor_heads' loads the complete actor and trains both action heads while freezing encoders/trunk. "
 71 |         "'actor_full' loads the complete actor."
 72 |     ),
 73 | )
 74 | # append RSL-RL cli arguments
 75 | cli_args.add_rsl_rl_args(parser)
 76 | # append AppLauncher cli args
 77 | AppLauncher.add_app_launcher_args(parser)
 78 | args_cli, hydra_args = parser.parse_known_args()
 79 | 
 80 | if args_cli.resume and args_cli.warm_start_checkpoint is not None:
 81 |     raise ValueError("--resume and --warm_start_checkpoint cannot be used together.")
 82 | if args_cli.warm_start_checkpoint is not None and args_cli.warm_start_mode is None:
 83 |     raise ValueError("--warm_start_mode is required when --warm_start_checkpoint is provided.")
 84 | 
 85 | # always enable cameras to record video
 86 | if args_cli.video:
 87 |     args_cli.enable_cameras = True
 88 | 
 89 | # clear out sys.argv for Hydra
 90 | sys.argv = [sys.argv[0]] + hydra_args
 91 | 
 92 | # launch omniverse app
 93 | app_launcher = AppLauncher(args_cli)
 94 | simulation_app = app_launcher.app
 95 | 
 96 | """Check for minimum supported RSL-RL version."""
 97 | 
 98 | import importlib.metadata as metadata
 99 | import platform
100 | 
101 | from packaging import version
102 | 
103 | # for distributed training, check minimum supported rsl-rl version
104 | RSL_RL_VERSION = "2.3.1"
105 | installed_version = metadata.version("rsl-rl-lib")
106 | if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
107 |     if platform.system() == "Windows":
108 |         cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
109 |     else:
110 |         cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
111 |     print(
112 |         f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
113 |         f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
114 |         f"\n\n\t{' '.join(cmd)}\n"
115 |     )
116 |     exit(1)
117 | 
118 | """Rest everything follows."""
119 | 
120 | import gymnasium as gym
121 | import os
122 | import torch
123 | from datetime import datetime
124 | 
125 | from rsl_rl.runners import OnPolicyRunner
126 | import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner
127 | 
128 | from isaaclab.envs import (
129 |     DirectMARLEnv,
130 |     DirectMARLEnvCfg,
131 |     DirectRLEnvCfg,
132 |     ManagerBasedRLEnvCfg,
133 |     multi_agent_to_single_agent,
134 | )
135 | from isaaclab.utils.dict import print_dict
136 | from isaaclab.utils.io import dump_yaml
137 | 
138 | try:
139 |     from isaaclab.utils.io import dump_pickle
140 | except ImportError:
141 |     try:
142 |         import cloudpickle as pickle
143 |     except ImportError:
144 |         import pickle
145 | 
146 |     def dump_pickle(filename, data):
147 |         os.makedirs(os.path.dirname(filename), exist_ok=True)
148 |         with open(filename, "wb") as f:
149 |             pickle.dump(data, f)
150 | 
151 | from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
152 | 
153 | import isaaclab_tasks  # noqa: F401
154 | from isaaclab_tasks.utils import get_checkpoint_path
155 | from isaaclab_tasks.utils.hydra import hydra_task_config
156 | 
157 | import Ranger.tasks  # noqa: F401
158 | from Ranger.tasks.manager_based.ranger.agents import RangerTerrainActorCritic
159 | from warm_start import warm_start_ranger_actor
160 | 
161 | torch.backends.cuda.matmul.allow_tf32 = True
162 | torch.backends.cudnn.allow_tf32 = True
163 | torch.backends.cudnn.deterministic = False
164 | torch.backends.cudnn.benchmark = False
165 | 
166 | rsl_on_policy_runner.RangerTerrainActorCritic = RangerTerrainActorCritic
167 | 
168 | 
169 | def _resolve_resume_path(log_root_path: str, load_run: str, load_checkpoint: str) -> str:
170 |     """Resolve a checkpoint path from either an explicit file path or the standard run/checkpoint selectors."""
171 | 
172 |     expanded_checkpoint = os.path.abspath(os.path.expanduser(load_checkpoint))
173 |     if os.path.isfile(expanded_checkpoint):
174 |         return expanded_checkpoint
175 |     return get_checkpoint_path(log_root_path, load_run, load_checkpoint)
176 | 
177 | 
178 | def _reset_action_std_from_env(runner: OnPolicyRunner) -> None:
179 |     """Optionally reset policy exploration std after checkpoint loading."""
180 | 
181 |     reset_std = os.environ.get("RANGER_RESET_ACTION_STD")
182 |     if reset_std is None:
183 |         return
184 | 
185 |     try:
186 |         reset_std_value = float(reset_std)
187 |     except ValueError as exc:
188 |         raise ValueError(f"RANGER_RESET_ACTION_STD must be a positive float, got: {reset_std!r}") from exc
189 |     if reset_std_value <= 0.0:
190 |         raise ValueError(f"RANGER_RESET_ACTION_STD must be positive, got: {reset_std_value}")
191 | 
192 |     actor_critic = runner.alg.policy
193 |     with torch.no_grad():
194 |         if hasattr(actor_critic, "std"):
195 |             actor_critic.std.fill_(reset_std_value)
196 |         elif hasattr(actor_critic, "log_std"):
197 |             log_std_value = torch.log(
198 |                 torch.tensor(reset_std_value, device=actor_critic.log_std.device, dtype=actor_critic.log_std.dtype)
199 |             )
200 |             actor_critic.log_std.fill_(log_std_value)
201 |         else:
202 |             print("[WARN] RANGER_RESET_ACTION_STD was set, but actor_critic has no std/log_std attribute.")
203 |             return
204 | 
205 |     print(f"[INFO] Reset action std to {reset_std_value} from RANGER_RESET_ACTION_STD.")
206 | 
207 | 
208 | @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
209 | def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
210 |     """Train with RSL-RL agent."""
211 |     # override configurations with non-hydra CLI arguments
212 |     agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
213 |     env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
214 |     agent_cfg.max_iterations = (
215 |         args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
216 |     )
217 | 
218 |     # set the environment seed
219 |     # note: certain randomizations occur in the environment initialization so we set the seed here
220 |     env_cfg.seed = agent_cfg.seed
221 |     env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
222 |     agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
223 | 
224 |     # multi-gpu training configuration
225 |     if args_cli.distributed:
226 |         env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
227 |         agent_cfg.device = f"cuda:{app_launcher.local_rank}"
228 | 
229 |         # set seed to have diversity in different threads
230 |         seed = agent_cfg.seed + app_launcher.local_rank
231 |         env_cfg.seed = seed
232 |         agent_cfg.seed = seed
233 | 
234 |     # specify directory for logging experiments
235 |     log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
236 |     log_root_path = os.path.abspath(log_root_path)
237 |     print(f"[INFO] Logging experiment in directory: {log_root_path}")
238 |     # specify directory for logging runs: {time-stamp}_{run_name}
239 |     log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
240 |     # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
241 |     print(f"Exact experiment name requested from command line: {log_dir}")
242 |     if agent_cfg.run_name:
243 |         log_dir += f"_{agent_cfg.run_name}"
244 |     log_dir = os.path.join(log_root_path, log_dir)
245 | 
246 |     # create isaac environment
247 |     env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
248 | 
249 |     # convert to single-agent instance if required by the RL algorithm
250 |     if isinstance(env.unwrapped, DirectMARLEnv):
251 |         env = multi_agent_to_single_agent(env)
252 | 
253 |     # save resume path before creating a new log_dir
254 |     if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
255 |         resume_path = _resolve_resume_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
256 | 
257 |     # wrap for video recording
258 |     if args_cli.video:
259 |         video_kwargs = {
260 |             "video_folder": os.path.join(log_dir, "videos", "train"),
261 |             "step_trigger": lambda step: step % args_cli.video_interval == 0,
262 |             "video_length": args_cli.video_length,
263 |             "disable_logger": True,
264 |         }
265 |         print("[INFO] Recording videos during training.")
266 |         print_dict(video_kwargs, nesting=4)
267 |         env = gym.wrappers.RecordVideo(env, **video_kwargs)
268 | 
269 |     # wrap around environment for rsl-rl
270 |     env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
271 | 
272 |     # create runner from rsl-rl
273 |     runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
274 |     # write git state to logs
275 |     runner.add_git_repo_to_log(__file__)
276 |     # load the checkpoint
277 |     if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
278 |         print(f"[INFO]: Loading model checkpoint from: {resume_path}")
279 |         runner.load(resume_path)
280 |         _reset_action_std_from_env(runner)
281 |     elif args_cli.warm_start_checkpoint is not None:
282 |         warm_start_ranger_actor(
283 |             runner=runner,
284 |             checkpoint_path=args_cli.warm_start_checkpoint,
285 |             mode=args_cli.warm_start_mode,
286 |         )
287 | 
288 |     # dump the configuration into log-directory
289 |     dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
290 |     dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
291 |     dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
292 |     dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
293 | 
294 |     # run training. Always flush and export scalar history, including on KeyboardInterrupt.
295 |     try:
296 |         runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
297 |     finally:
298 |         writer = getattr(runner, "writer", None)
299 |         if writer is not None and hasattr(writer, "flush"):
300 |             writer.flush()
301 |         if not args_cli.disable_iteration_metrics_csv:
302 |             try:
303 |                 export_tensorboard_scalars(log_dir)
304 |             except Exception as error:
305 |                 print(f"[WARN] Failed to export per-iteration CSV metrics: {error}")
306 |         env.close()
307 | 
308 | 
309 | if __name__ == "__main__":
310 |     # run the main function
311 |     main()
312 |     # close sim app
313 |     simulation_app.close()
314 | 
```

### scripts/rsl_rl/warm_start.py

Bytes: 21087
SHA-256: d3e866cd9e9b657a669d6424ec36e6cff369672f202265e93d1af3e96e6dea42
Lines: 1-443 of 443

```python
  1 | from __future__ import annotations
  2 | 
  3 | from pathlib import Path
  4 | from typing import Iterable
  5 | 
  6 | import torch
  7 | import torch.nn as nn
  8 | 
  9 | 
 10 | ACTOR_SHARED_MODULES = ("map_encoder", "state_encoder", "actor_trunk")
 11 | ACTOR_SUSPENSION_MODULES = ACTOR_SHARED_MODULES + ("suspension_head",)
 12 | ACTOR_WHEEL_MODULES = ACTOR_SHARED_MODULES + ("wheel_head",)
 13 | ACTOR_FULL_MODULES = ACTOR_SHARED_MODULES + ("suspension_head", "wheel_head")
 14 | 
 15 | 
 16 | def _describe_public_attrs(obj: object) -> str:
 17 |     attrs = [name for name in dir(obj) if not name.startswith("__")]
 18 |     return ", ".join(attrs[:80])
 19 | 
 20 | 
 21 | def get_ranger_actor(runner) -> nn.Module:
 22 |     """Return the Ranger dual-head actor module from the current runner."""
 23 | 
 24 |     alg = getattr(runner, "alg", None)
 25 |     if alg is None:
 26 |         raise AttributeError(f"Runner has no 'alg' attribute. Runner type={type(runner).__name__}")
 27 | 
 28 |     candidate_paths = (
 29 |         ("runner.alg.policy", getattr(alg, "policy", None)),
 30 |         ("runner.alg.actor_critic", getattr(alg, "actor_critic", None)),
 31 |     )
 32 |     diagnostics: list[str] = []
 33 |     for path, policy in candidate_paths:
 34 |         if policy is None:
 35 |             diagnostics.append(f"{path}: missing")
 36 |             continue
 37 |         actor = getattr(policy, "actor", None)
 38 |         if actor is None:
 39 |             diagnostics.append(f"{path}: type={type(policy).__name__}, no actor attr")
 40 |             continue
 41 |         required = ACTOR_FULL_MODULES
 42 |         missing = [name for name in required if not hasattr(actor, name)]
 43 |         if missing:
 44 |             diagnostics.append(f"{path}.actor: type={type(actor).__name__}, missing={missing}")
 45 |             continue
 46 |         return actor
 47 | 
 48 |     alg_attrs = _describe_public_attrs(alg)
 49 |     raise AttributeError(
 50 |         "Could not locate Ranger dual-head actor. "
 51 |         f"Diagnostics={diagnostics}. alg type={type(alg).__name__}; attrs={alg_attrs}"
 52 |     )
 53 | 
 54 | 
 55 | def _get_ranger_policy(runner) -> nn.Module:
 56 |     alg = getattr(runner, "alg", None)
 57 |     for policy in (getattr(alg, "policy", None), getattr(alg, "actor_critic", None)):
 58 |         if policy is not None and getattr(policy, "actor", None) is get_ranger_actor(runner):
 59 |             return policy
 60 |     raise AttributeError("Could not locate policy object that owns the Ranger actor.")
 61 | 
 62 | 
 63 | def _load_checkpoint_model_state(checkpoint_path: str | Path) -> tuple[dict, dict[str, torch.Tensor]]:
 64 |     path = Path(checkpoint_path).expanduser()
 65 |     checkpoint = torch.load(path, map_location="cpu", weights_only=False)
 66 |     if not isinstance(checkpoint, dict):
 67 |         raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}: {path}")
 68 |     if "model_state_dict" not in checkpoint:
 69 |         raise KeyError(f"Checkpoint is missing 'model_state_dict'. Available keys: {sorted(checkpoint.keys())}")
 70 |     model_state = checkpoint["model_state_dict"]
 71 |     if not isinstance(model_state, dict):
 72 |         raise TypeError(f"model_state_dict must be a dict, got {type(model_state).__name__}")
 73 |     return checkpoint, model_state
 74 | 
 75 | 
 76 | def _validate_no_forbidden_loaded_keys(model_state: dict[str, torch.Tensor], modules: Iterable[str]) -> None:
 77 |     del modules
 78 |     forbidden_prefixes = ("critic.", "actor_obs_normalizer.", "critic_obs_normalizer.")
 79 |     forbidden_exact = {"std", "log_std"}
 80 |     forbidden = [
 81 |         key
 82 |         for key in model_state
 83 |         if key in forbidden_exact or any(key.startswith(prefix) for prefix in forbidden_prefixes)
 84 |     ]
 85 |     if forbidden:
 86 |         preview = ", ".join(forbidden[:12])
 87 |         print(f"[WarmStart] checkpoint contains excluded keys that will not be loaded: {preview}")
 88 | 
 89 | 
 90 | def _load_actor_module(actor: nn.Module, model_state: dict[str, torch.Tensor], module_name: str) -> int:
 91 |     module = getattr(actor, module_name, None)
 92 |     if module is None:
 93 |         raise AttributeError(f"Target actor is missing module '{module_name}'. Actor type={type(actor).__name__}")
 94 |     if not isinstance(module, nn.Module):
 95 |         raise TypeError(f"Target actor attribute '{module_name}' is not an nn.Module: {type(module).__name__}")
 96 | 
 97 |     target_state = module.state_dict()
 98 |     source_prefix = f"actor.{module_name}."
 99 |     source_keys = {key for key in model_state if key.startswith(source_prefix)}
100 |     expected_source_keys = {source_prefix + key for key in target_state}
101 |     missing = sorted(expected_source_keys - source_keys)
102 |     extra = sorted(source_keys - expected_source_keys)
103 |     if missing:
104 |         raise KeyError(f"Warm-start checkpoint missing keys for {module_name}: {missing}")
105 |     if extra:
106 |         raise KeyError(f"Warm-start checkpoint has unexpected keys for {module_name}: {extra}")
107 | 
108 |     load_state: dict[str, torch.Tensor] = {}
109 |     for target_key, target_tensor in target_state.items():
110 |         source_key = source_prefix + target_key
111 |         source_tensor = model_state[source_key]
112 |         if tuple(source_tensor.shape) != tuple(target_tensor.shape):
113 |             raise ValueError(
114 |                 f"Shape mismatch for {source_key}: checkpoint {tuple(source_tensor.shape)} "
115 |                 f"vs target {tuple(target_tensor.shape)}"
116 |             )
117 |         load_state[target_key] = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
118 | 
119 |     module.load_state_dict(load_state, strict=True)
120 |     return len(load_state)
121 | 
122 | 
123 | def _reset_suspension_head(actor: nn.Module) -> str:
124 |     """Reinitialize the suspension branch with an exactly neutral initial output."""
125 | 
126 |     suspension_head = getattr(actor, "suspension_head", None)
127 |     if suspension_head is None:
128 |         raise AttributeError(f"Target actor has no suspension_head. Actor type={type(actor).__name__}")
129 |     if not isinstance(suspension_head, nn.Module):
130 |         raise TypeError(f"actor.suspension_head is not an nn.Module: {type(suspension_head).__name__}")
131 | 
132 |     for module in suspension_head.modules():
133 |         reset_parameters = getattr(module, "reset_parameters", None)
134 |         if callable(reset_parameters):
135 |             reset_parameters()
136 | 
137 |     linear_layers = [module for module in suspension_head.modules() if isinstance(module, nn.Linear)]
138 |     if not linear_layers:
139 |         raise TypeError("actor.suspension_head must contain at least one nn.Linear layer.")
140 |     output_layer = linear_layers[-1]
141 |     with torch.no_grad():
142 |         output_layer.weight.zero_()
143 |         if output_layer.bias is not None:
144 |             output_layer.bias.zero_()
145 | 
146 |     return (
147 |         f"reset {len(linear_layers)} linear layers; final weight/bias exactly zero "
148 |         f"(output_dim={output_layer.out_features})"
149 |     )
150 | 
151 | 
152 | def _reset_wheel_head(actor: nn.Module) -> str:
153 |     """Reinitialize the wheel branch with an exactly neutral initial output."""
154 | 
155 |     wheel_head = getattr(actor, "wheel_head", None)
156 |     if wheel_head is None:
157 |         raise AttributeError(f"Target actor has no wheel_head. Actor type={type(actor).__name__}")
158 |     if not isinstance(wheel_head, nn.Module):
159 |         raise TypeError(f"actor.wheel_head is not an nn.Module: {type(wheel_head).__name__}")
160 | 
161 |     for module in wheel_head.modules():
162 |         reset_parameters = getattr(module, "reset_parameters", None)
163 |         if callable(reset_parameters):
164 |             reset_parameters()
165 | 
166 |     linear_layers = [module for module in wheel_head.modules() if isinstance(module, nn.Linear)]
167 |     if not linear_layers:
168 |         raise TypeError("actor.wheel_head must contain at least one nn.Linear layer.")
169 |     output_layer = linear_layers[-1]
170 |     with torch.no_grad():
171 |         output_layer.weight.zero_()
172 |         if output_layer.bias is not None:
173 |             output_layer.bias.zero_()
174 | 
175 |     return (
176 |         f"reset {len(linear_layers)} linear layers; final weight/bias exactly zero "
177 |         f"(output_dim={output_layer.out_features})"
178 |     )
179 | 
180 | 
181 | def _reset_wheel_output_layer(actor: nn.Module) -> tuple[nn.Linear, str]:
182 |     """Reset only the final wheel-action mapping while preserving wheel hidden features."""
183 | 
184 |     wheel_head = getattr(actor, "wheel_head", None)
185 |     if wheel_head is None:
186 |         raise AttributeError(f"Target actor has no wheel_head. Actor type={type(actor).__name__}")
187 |     if not isinstance(wheel_head, nn.Module):
188 |         raise TypeError(f"actor.wheel_head is not an nn.Module: {type(wheel_head).__name__}")
189 | 
190 |     linear_layers = [module for module in wheel_head.modules() if isinstance(module, nn.Linear)]
191 |     if not linear_layers:
192 |         raise TypeError("actor.wheel_head must contain at least one nn.Linear layer.")
193 |     output_layer = linear_layers[-1]
194 |     with torch.no_grad():
195 |         output_layer.weight.zero_()
196 |         if output_layer.bias is not None:
197 |             output_layer.bias.zero_()
198 | 
199 |     return (
200 |         output_layer,
201 |         f"final weight/bias exactly zero (input_dim={output_layer.in_features}, output_dim={output_layer.out_features})",
202 |     )
203 | 
204 | 
205 | def _reset_warm_start_action_std(policy: nn.Module) -> str:
206 |     """Restore the action-noise profile declared by the current runner configuration."""
207 | 
208 |     configured = getattr(policy, "_configured_initial_action_std", None)
209 |     if configured is None:
210 |         configured = torch.tensor([0.15] * 4 + [0.25] * 4, dtype=torch.float32)
211 |     configured = configured.detach()
212 |     if tuple(configured.shape) != (8,):
213 |         raise ValueError(f"Expected 8-dim configured action std, got shape {tuple(configured.shape)}")
214 |     if hasattr(policy, "std"):
215 |         std = getattr(policy, "std")
216 |         if tuple(std.shape) != (8,):
217 |             raise ValueError(f"Expected 8-dim std for Ranger action space, got shape {tuple(std.shape)}")
218 |         with torch.no_grad():
219 |             std.copy_(configured.to(device=std.device, dtype=std.dtype))
220 |         std.requires_grad_(False)
221 |         return f"fixed std={configured.tolist()}"
222 |     if hasattr(policy, "log_std"):
223 |         log_std = getattr(policy, "log_std")
224 |         if tuple(log_std.shape) != (8,):
225 |             raise ValueError(f"Expected 8-dim log_std for Ranger action space, got shape {tuple(log_std.shape)}")
226 |         with torch.no_grad():
227 |             log_std.copy_(torch.log(configured).to(device=log_std.device, dtype=log_std.dtype))
228 |         log_std.requires_grad_(False)
229 |         return f"fixed log_std=log({configured.tolist()})"
230 |     return "policy has no std/log_std; unchanged"
231 | 
232 | 
233 | def _reset_suspension_action_std(policy: nn.Module) -> str:
234 |     """Reset only suspension exploration dimensions from the current runner configuration."""
235 | 
236 |     configured = getattr(policy, "_configured_initial_action_std", None)
237 |     if configured is None:
238 |         configured = torch.tensor([0.003] * 4 + [0.01] * 4, dtype=torch.float32)
239 |     configured = configured.detach()
240 |     if tuple(configured.shape) != (8,):
241 |         raise ValueError(f"Expected 8-dim configured action std, got shape {tuple(configured.shape)}")
242 | 
243 |     if hasattr(policy, "std"):
244 |         std = getattr(policy, "std")
245 |         if tuple(std.shape) != (8,):
246 |             raise ValueError(f"Expected 8-dim std for Ranger action space, got shape {tuple(std.shape)}")
247 |         with torch.no_grad():
248 |             std[:4].copy_(configured[:4].to(device=std.device, dtype=std.dtype))
249 |         std.requires_grad_(False)
250 |         return f"fixed suspension std={std[:4].detach().cpu().tolist()}; wheel std preserved"
251 |     if hasattr(policy, "log_std"):
252 |         log_std = getattr(policy, "log_std")
253 |         if tuple(log_std.shape) != (8,):
254 |             raise ValueError(f"Expected 8-dim log_std for Ranger action space, got shape {tuple(log_std.shape)}")
255 |         with torch.no_grad():
256 |             log_std[:4].copy_(torch.log(configured[:4]).to(device=log_std.device, dtype=log_std.dtype))
257 |         log_std.requires_grad_(False)
258 |         return f"fixed suspension log_std=log({configured[:4].tolist()}); wheel log_std preserved"
259 |     return "policy has no std/log_std; unchanged"
260 | 
261 | 
262 | def warm_start_ranger_actor(runner, checkpoint_path: str | Path, mode: str) -> None:
263 |     valid_modes = {
264 |         "actor_suspension",
265 |         "actor_wheel",
266 |         "actor_suspension_only",
267 |         "actor_wheel_reset_suspension",
268 |         "actor_wheel_reset_final",
269 |         "actor_reset_heads",
270 |         "actor_heads",
271 |         "actor_full",
272 |     }
273 |     if mode not in valid_modes:
274 |         raise ValueError(
275 |             f"Unsupported warm-start mode: {mode!r}. Expected one of {sorted(valid_modes)}."
276 |         )
277 | 
278 |     actor = get_ranger_actor(runner)
279 |     policy = _get_ranger_policy(runner)
280 |     checkpoint, model_state = _load_checkpoint_model_state(checkpoint_path)
281 |     if mode == "actor_suspension":
282 |         modules = ACTOR_SUSPENSION_MODULES
283 |     elif mode == "actor_wheel_reset_suspension":
284 |         modules = ACTOR_WHEEL_MODULES
285 |     elif mode == "actor_reset_heads":
286 |         modules = ACTOR_SHARED_MODULES
287 |     else:
288 |         # actor_wheel also loads the complete actor so the existing suspension branch
289 |         # remains available for later staged reintroduction.
290 |         modules = ACTOR_FULL_MODULES
291 |     _validate_no_forbidden_loaded_keys(model_state, modules)
292 | 
293 |     loaded_counts: dict[str, int] = {}
294 |     for module_name in modules:
295 |         loaded_counts[module_name] = _load_actor_module(actor, model_state, module_name)
296 |     suspension_reset_msg = None
297 |     wheel_reset_msg = None
298 |     wheel_output_layer = None
299 |     if mode == "actor_wheel_reset_suspension":
300 |         suspension_reset_msg = _reset_suspension_head(actor)
301 |     elif mode == "actor_reset_heads":
302 |         suspension_reset_msg = _reset_suspension_head(actor)
303 |         wheel_reset_msg = _reset_wheel_head(actor)
304 |     elif mode == "actor_wheel_reset_final":
305 |         wheel_output_layer, wheel_reset_msg = _reset_wheel_output_layer(actor)
306 | 
307 |     for parameter in actor.parameters():
308 |         parameter.requires_grad = True
309 |     if mode in {
310 |         "actor_wheel",
311 |         "actor_suspension_only",
312 |         "actor_wheel_reset_suspension",
313 |         "actor_wheel_reset_final",
314 |         "actor_reset_heads",
315 |         "actor_heads",
316 |     }:
317 |         for parameter in actor.parameters():
318 |             parameter.requires_grad = False
319 |     if mode == "actor_wheel":
320 |         for parameter in actor.wheel_head.parameters():
321 |             parameter.requires_grad = True
322 |     elif mode in {"actor_suspension_only", "actor_wheel_reset_suspension"}:
323 |         for parameter in actor.suspension_head.parameters():
324 |             parameter.requires_grad = True
325 |     elif mode == "actor_wheel_reset_final":
326 |         assert wheel_output_layer is not None
327 |         for parameter in wheel_output_layer.parameters():
328 |             parameter.requires_grad = True
329 |     elif mode in {"actor_reset_heads", "actor_heads"}:
330 |         for head in (actor.suspension_head, actor.wheel_head):
331 |             for parameter in head.parameters():
332 |                 parameter.requires_grad = True
333 | 
334 |     runner.current_learning_iteration = 0
335 | 
336 |     print(f"[WarmStart] source: {Path(checkpoint_path).expanduser()}")
337 |     print(f"[WarmStart] checkpoint keys: {sorted(checkpoint.keys())}")
338 |     print(f"[WarmStart] model_state_dict tensors: {len(model_state)}")
339 |     print(f"[WarmStart] mode: {mode}")
340 |     if mode == "actor_suspension":
341 |         for module_name in modules:
342 |             print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
343 |         print("[WarmStart] random: actor.wheel_head")
344 |         std_msg = _reset_warm_start_action_std(policy)
345 |     elif mode == "actor_wheel_reset_suspension":
346 |         print("[WarmStart] loaded: navigation encoders/trunk + wheel head")
347 |         for module_name in modules:
348 |             print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
349 |         print(f"[WarmStart] reset: actor.suspension_head ({suspension_reset_msg})")
350 |         print("[WarmStart] trainable: actor.suspension_head only")
351 |         print("[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk/wheel_head")
352 |         std_msg = _reset_suspension_action_std(policy)
353 |     elif mode == "actor_wheel_reset_final":
354 |         print("[WarmStart] loaded: complete actor")
355 |         for module_name in modules:
356 |             print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
357 |         print(f"[WarmStart] reset: actor.wheel_head final Linear ({wheel_reset_msg})")
358 |         print("[WarmStart] trainable: actor.wheel_head final Linear only")
359 |         print(
360 |             "[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk/"
361 |             "suspension_head/wheel_head hidden layers"
362 |         )
363 |         std_msg = _reset_warm_start_action_std(policy)
364 |     elif mode == "actor_reset_heads":
365 |         print("[WarmStart] loaded: navigation encoders/trunk only")
366 |         for module_name in modules:
367 |             print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
368 |         print(f"[WarmStart] reset: actor.suspension_head ({suspension_reset_msg})")
369 |         print(f"[WarmStart] reset: actor.wheel_head ({wheel_reset_msg})")
370 |         print("[WarmStart] trainable: actor.suspension_head + actor.wheel_head")
371 |         print("[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk")
372 |         std_msg = _reset_warm_start_action_std(policy)
373 |     elif mode == "actor_wheel":
374 |         print("[WarmStart] loaded: complete actor")
375 |         for module_name in modules:
376 |             print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
377 |         print("[WarmStart] trainable: actor.wheel_head only")
378 |         print("[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk/suspension_head")
379 |         std_msg = _reset_warm_start_action_std(policy)
380 |     elif mode == "actor_suspension_only":
381 |         print("[WarmStart] loaded: complete actor")
382 |         for module_name in modules:
383 |             print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
384 |         print("[WarmStart] trainable: actor.suspension_head only")
385 |         print("[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk/wheel_head")
386 |         std_msg = _reset_warm_start_action_std(policy)
387 |     elif mode == "actor_heads":
388 |         print("[WarmStart] loaded: complete actor")
389 |         for module_name in modules:
390 |             print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
391 |         print("[WarmStart] trainable: actor.suspension_head + actor.wheel_head")
392 |         print("[WarmStart] frozen and preserved: actor.map_encoder/state_encoder/actor_trunk")
393 |         std_msg = _reset_warm_start_action_std(policy)
394 |     else:
395 |         print("[WarmStart] loaded: complete actor")
396 |         for module_name in modules:
397 |             print(f"[WarmStart] loaded: actor.{module_name} ({loaded_counts[module_name]} tensors)")
398 |         std_msg = _reset_warm_start_action_std(policy)
399 |     print("[WarmStart] new: critic")
400 |     print("[WarmStart] new: optimizer")
401 |     print(f"[WarmStart] new: action std ({std_msg})")
402 |     print("[WarmStart] iteration: 0")
403 |     trainable_names = [name for name, parameter in actor.named_parameters() if parameter.requires_grad]
404 |     frozen_names = [name for name, parameter in actor.named_parameters() if not parameter.requires_grad]
405 |     if mode == "actor_wheel_reset_suspension":
406 |         expected_trainable_names = {
407 |             f"suspension_head.{name}" for name, _ in actor.suspension_head.named_parameters()
408 |         }
409 |         if set(trainable_names) != expected_trainable_names:
410 |             raise RuntimeError(
411 |                 "Reset-suspension warm start exposed an unexpected actor parameter set. "
412 |                 f"Expected={sorted(expected_trainable_names)}, actual={sorted(trainable_names)}"
413 |             )
414 |         print("[WarmStart] validated: only actor.suspension_head is trainable")
415 |     elif mode == "actor_reset_heads":
416 |         expected_trainable_names = {
417 |             f"suspension_head.{name}" for name, _ in actor.suspension_head.named_parameters()
418 |         } | {
419 |             f"wheel_head.{name}" for name, _ in actor.wheel_head.named_parameters()
420 |         }
421 |         if set(trainable_names) != expected_trainable_names:
422 |             raise RuntimeError(
423 |                 "Reset-heads warm start exposed an unexpected actor parameter set. "
424 |                 f"Expected={sorted(expected_trainable_names)}, actual={sorted(trainable_names)}"
425 |             )
426 |         print("[WarmStart] validated: only actor.suspension_head and actor.wheel_head are trainable")
427 |     elif mode == "actor_wheel_reset_final":
428 |         assert wheel_output_layer is not None
429 |         output_parameter_ids = {id(parameter) for parameter in wheel_output_layer.parameters()}
430 |         expected_trainable_names = {
431 |             name for name, parameter in actor.named_parameters() if id(parameter) in output_parameter_ids
432 |         }
433 |         if set(trainable_names) != expected_trainable_names:
434 |             raise RuntimeError(
435 |                 "Reset-wheel-final warm start exposed an unexpected actor parameter set. "
436 |                 f"Expected={sorted(expected_trainable_names)}, actual={sorted(trainable_names)}"
437 |             )
438 |         print("[WarmStart] validated: only actor.wheel_head final Linear is trainable")
439 |     print(f"[WarmStart] trainable actor parameter tensors: {len(trainable_names)}")
440 |     print(f"[WarmStart] frozen actor parameter tensors: {len(frozen_names)}")
441 |     if frozen_names:
442 |         print(f"[WarmStart] frozen actor parameters: {frozen_names}")
443 | 
```

### source/Ranger/Ranger/tasks/manager_based/ranger/__init__.py

Bytes: 5889
SHA-256: f7d793919a37779401d99aaee35fe15177291b48f1d2dce9193c30be7adb0c8c
Lines: 1-175 of 175

```python
  1 | # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
  2 | # All rights reserved.
  3 | #
  4 | # SPDX-License-Identifier: BSD-3-Clause
  5 | 
  6 | import gymnasium as gym
  7 | 
  8 | from . import agents
  9 | 
 10 | ##
 11 | # Register Gym environments.
 12 | ##
 13 | 
 14 | 
 15 | gym.register(
 16 |     id="Template-Ranger-Debug-v0",
 17 |     entry_point="isaaclab.envs:ManagerBasedRLEnv",
 18 |     disable_env_checker=True,
 19 |     kwargs={
 20 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerEnvCfg",
 21 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
 22 |     },
 23 | )
 24 | 
 25 | gym.register(
 26 |     id="Template-Ranger-Stand-v0",
 27 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
 28 |     disable_env_checker=True,
 29 |     kwargs={
 30 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStandEnvCfg",
 31 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StandPPORunnerCfg",
 32 |     },
 33 | )
 34 | 
 35 | gym.register(
 36 |     id="Template-Ranger-ShortGoalFlat-v0",
 37 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
 38 |     disable_env_checker=True,
 39 |     kwargs={
 40 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatEnvCfg",
 41 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatPPORunnerCfg",
 42 |     },
 43 | )
 44 | 
 45 | gym.register(
 46 |     id="Template-Ranger-ShortGoalFlat-v1",
 47 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
 48 |     disable_env_checker=True,
 49 |     kwargs={
 50 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV1EnvCfg",
 51 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatPPORunnerCfg",
 52 |     },
 53 | )
 54 | 
 55 | gym.register(
 56 |     id="Template-Ranger-ShortGoalFlat-v2",
 57 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
 58 |     disable_env_checker=True,
 59 |     kwargs={
 60 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV2EnvCfg",
 61 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV2PPORunnerCfg",
 62 |     },
 63 | )
 64 | 
 65 | gym.register(
 66 |     id="Template-Ranger-ShortGoalFlat-v3",
 67 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
 68 |     disable_env_checker=True,
 69 |     kwargs={
 70 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV3EnvCfg",
 71 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatLimitedSuspensionPPORunnerCfg",
 72 |     },
 73 | )
 74 | 
 75 | gym.register(
 76 |     id="Template-Ranger-ShortGoalFlatJoint-v3",
 77 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
 78 |     disable_env_checker=True,
 79 |     kwargs={
 80 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV3EnvCfg",
 81 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatLimitedJointPPORunnerCfg",
 82 |     },
 83 | )
 84 | 
 85 | gym.register(
 86 |     id="Template-Ranger-ShortGoalFlat-v4",
 87 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
 88 |     disable_env_checker=True,
 89 |     kwargs={
 90 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV4EnvCfg",
 91 |         "rsl_rl_cfg_entry_point": (
 92 |             f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatFrozenSuspensionWheelPPORunnerCfg"
 93 |         ),
 94 |     },
 95 | )
 96 | 
 97 | gym.register(
 98 |     id="Template-Ranger-ShortGoalFlat-v5",
 99 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
100 |     disable_env_checker=True,
101 |     kwargs={
102 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV5EnvCfg",
103 |         "rsl_rl_cfg_entry_point": (
104 |             f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatFrozenSuspensionWheelPPORunnerCfg"
105 |         ),
106 |     },
107 | )
108 | 
109 | gym.register(
110 |     id="Template-Ranger-ShortGoalFlat-v6",
111 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
112 |     disable_env_checker=True,
113 |     kwargs={
114 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV6EnvCfg",
115 |         "rsl_rl_cfg_entry_point": (
116 |             f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatFrozenSuspensionWheelPPORunnerCfg"
117 |         ),
118 |     },
119 | )
120 | 
121 | 
122 | gym.register(
123 |     id="Template-Ranger-ShortGoalFlat-v7",
124 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
125 |     disable_env_checker=True,
126 |     kwargs={
127 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV7EnvCfg",
128 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV7PPORunnerCfg",
129 |     },
130 | )
131 | 
132 | 
133 | gym.register(
134 |     id="Template-Ranger-ShortGoalFlat-v8",
135 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
136 |     disable_env_checker=True,
137 |     kwargs={
138 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV8EnvCfg",
139 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV8PPORunnerCfg",
140 |     },
141 | )
142 | 
143 | 
144 | gym.register(
145 |     id="Template-Ranger-ShortGoalFlat-v9",
146 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
147 |     disable_env_checker=True,
148 |     kwargs={
149 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV9EnvCfg",
150 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV9PPORunnerCfg",
151 |     },
152 | )
153 | 
154 | 
155 | gym.register(
156 |     id="Template-Ranger-ShortGoalFlat-v10",
157 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
158 |     disable_env_checker=True,
159 |     kwargs={
160 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV10EnvCfg",
161 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV10PPORunnerCfg",
162 |     },
163 | )
164 | 
165 | 
166 | gym.register(
167 |     id="Template-Ranger-ShortGoalFlat-v11",
168 |     entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
169 |     disable_env_checker=True,
170 |     kwargs={
171 |         "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV11EnvCfg",
172 |         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV11PPORunnerCfg",
173 |     },
174 | )
175 | 
```

### source/Ranger/Ranger/tasks/manager_based/ranger/agents/__init__.py

Bytes: 292
SHA-256: 88e1a3e09e9e290c5a4bfc3fcefd8c3cf33259c1a55dabfbb52c193f29688f46
Lines: 1-9 of 9

```python
1 | # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
2 | # All rights reserved.
3 | #
4 | # SPDX-License-Identifier: BSD-3-Clause
5 | 
6 | from .rsl_rl_custom_policy import RangerTerrainActorCritic
7 | 
8 | __all__ = ["RangerTerrainActorCritic"]
9 | 
```

### source/Ranger/Ranger/tasks/manager_based/ranger/agents/rsl_rl_custom_policy.py

Bytes: 19437
SHA-256: e4d1c7d08ea754c5bb6ce0857c40edc2ab791acc70f4aef05853172c0437acad
Lines: 1-469 of 469

```python
  1 | # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
  2 | # All rights reserved.
  3 | #
  4 | # SPDX-License-Identifier: BSD-3-Clause
  5 | 
  6 | from __future__ import annotations
  7 | 
  8 | from typing import Any
  9 | 
 10 | import torch
 11 | import torch.nn as nn
 12 | from tensordict import TensorDict
 13 | from torch.distributions import Normal
 14 | 
 15 | from rsl_rl.modules.actor_critic import ActorCritic
 16 | from rsl_rl.networks import EmpiricalNormalization
 17 | 
 18 | 
 19 | def _activation(name: str) -> nn.Module:
 20 |     name = name.lower()
 21 |     if name == "elu":
 22 |         return nn.ELU()
 23 |     if name == "relu":
 24 |         return nn.ReLU()
 25 |     if name == "leaky_relu":
 26 |         return nn.LeakyReLU()
 27 |     if name == "tanh":
 28 |         return nn.Tanh()
 29 |     if name == "sigmoid":
 30 |         return nn.Sigmoid()
 31 |     if name == "selu":
 32 |         return nn.SELU()
 33 |     raise ValueError(f"Unsupported activation: {name}")
 34 | 
 35 | 
 36 | def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
 37 |     layers: list[nn.Module] = []
 38 |     last_dim = input_dim
 39 |     for hidden_dim in hidden_dims:
 40 |         layers.append(nn.Linear(last_dim, hidden_dim))
 41 |         layers.append(_activation(activation))
 42 |         last_dim = hidden_dim
 43 |     layers.append(nn.Linear(last_dim, output_dim))
 44 |     return nn.Sequential(*layers)
 45 | 
 46 | 
 47 | class _TerrainMapEncoder(nn.Module):
 48 |     def __init__(
 49 |         self,
 50 |         input_channels: int,
 51 |         grid_shape: tuple[int, int],
 52 |         conv_channels: list[int],
 53 |         latent_dim: int,
 54 |         activation: str,
 55 |     ) -> None:
 56 |         super().__init__()
 57 |         conv_layers: list[nn.Module] = []
 58 |         in_channels = input_channels
 59 |         for out_channels in conv_channels:
 60 |             conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
 61 |             conv_layers.append(_activation(activation))
 62 |             in_channels = out_channels
 63 |         self.conv = nn.Sequential(*conv_layers)
 64 | 
 65 |         with torch.no_grad():
 66 |             dummy = torch.zeros(1, input_channels, grid_shape[0], grid_shape[1], dtype=torch.float32)
 67 |             conv_out_dim = int(self.conv(dummy).reshape(1, -1).shape[-1])
 68 | 
 69 |         self.proj = nn.Sequential(
 70 |             nn.Flatten(),
 71 |             nn.Linear(conv_out_dim, latent_dim),
 72 |             _activation(activation),
 73 |         )
 74 | 
 75 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 76 |         return self.proj(self.conv(x))
 77 | 
 78 | 
 79 | class _RangerDualHeadActor(nn.Module):
 80 |     def __init__(
 81 |         self,
 82 |         state_dim: int,
 83 |         map_dim: int,
 84 |         num_actions: int,
 85 |         terrain_channels: int,
 86 |         terrain_grid_shape: tuple[int, int],
 87 |         terrain_cnn_channels: list[int],
 88 |         terrain_latent_dim: int,
 89 |         state_hidden_dims: list[int],
 90 |         state_latent_dim: int,
 91 |         actor_trunk_hidden_dims: list[int],
 92 |         wheel_head_hidden_dims: list[int],
 93 |         suspension_head_hidden_dims: list[int],
 94 |         activation: str,
 95 |     ) -> None:
 96 |         super().__init__()
 97 |         if num_actions != 8:
 98 |             raise ValueError(f"RangerDualHeadActor expects 8 actions, got {num_actions}.")
 99 | 
100 |         self.state_dim = int(state_dim)
101 |         self.map_dim = int(map_dim)
102 |         self.num_actions = int(num_actions)
103 |         self.terrain_channels = int(terrain_channels)
104 |         self.terrain_grid_shape = tuple(int(v) for v in terrain_grid_shape)
105 |         expected_map_dim = self.terrain_channels * self.terrain_grid_shape[0] * self.terrain_grid_shape[1]
106 |         if self.map_dim != expected_map_dim:
107 |             raise ValueError(
108 |                 f"Terrain map dim mismatch: expected {expected_map_dim} from "
109 |                 f"{self.terrain_channels}x{self.terrain_grid_shape}, got {self.map_dim}."
110 |             )
111 | 
112 |         self.map_encoder = _TerrainMapEncoder(
113 |             input_channels=self.terrain_channels,
114 |             grid_shape=self.terrain_grid_shape,
115 |             conv_channels=terrain_cnn_channels,
116 |             latent_dim=terrain_latent_dim,
117 |             activation=activation,
118 |         )
119 |         self.state_encoder = _build_mlp(
120 |             input_dim=self.state_dim,
121 |             hidden_dims=state_hidden_dims,
122 |             output_dim=state_latent_dim,
123 |             activation=activation,
124 |         )
125 |         fusion_input_dim = terrain_latent_dim + state_latent_dim
126 |         trunk_output_dim = actor_trunk_hidden_dims[-1] if actor_trunk_hidden_dims else fusion_input_dim
127 |         self.actor_trunk = _build_mlp(
128 |             input_dim=fusion_input_dim,
129 |             hidden_dims=actor_trunk_hidden_dims[:-1] if actor_trunk_hidden_dims else [],
130 |             output_dim=trunk_output_dim,
131 |             activation=activation,
132 |         )
133 |         self.actor_trunk_activation = _activation(activation)
134 |         self.suspension_head = _build_mlp(
135 |             input_dim=trunk_output_dim,
136 |             hidden_dims=suspension_head_hidden_dims,
137 |             output_dim=4,
138 |             activation=activation,
139 |         )
140 |         self.wheel_head = _build_mlp(
141 |             input_dim=trunk_output_dim,
142 |             hidden_dims=wheel_head_hidden_dims,
143 |             output_dim=4,
144 |             activation=activation,
145 |         )
146 | 
147 |         nn.init.zeros_(self.suspension_head[-1].weight)
148 |         nn.init.zeros_(self.suspension_head[-1].bias)
149 |         nn.init.zeros_(self.wheel_head[-1].weight)
150 |         nn.init.zeros_(self.wheel_head[-1].bias)
151 | 
152 |     def forward(self, obs: torch.Tensor) -> torch.Tensor:
153 |         state_obs = obs[:, : self.state_dim]
154 |         map_obs = obs[:, self.state_dim : self.state_dim + self.map_dim]
155 |         map_obs = map_obs.view(
156 |             obs.shape[0], self.terrain_channels, self.terrain_grid_shape[0], self.terrain_grid_shape[1]
157 |         )
158 | 
159 |         z_state = self.state_encoder(state_obs)
160 |         z_map = self.map_encoder(map_obs)
161 |         fused = torch.cat((z_map, z_state), dim=-1)
162 |         trunk = self.actor_trunk_activation(self.actor_trunk(fused))
163 | 
164 |         suspension_action = self.suspension_head(trunk)
165 |         wheel_action = self.wheel_head(trunk)
166 |         return torch.cat((suspension_action, wheel_action), dim=-1)
167 | 
168 | 
169 | class _RangerCritic(nn.Module):
170 |     def __init__(
171 |         self,
172 |         state_dim: int,
173 |         map_dim: int,
174 |         privileged_dim: int,
175 |         terrain_channels: int,
176 |         terrain_grid_shape: tuple[int, int],
177 |         terrain_cnn_channels: list[int],
178 |         terrain_latent_dim: int,
179 |         state_hidden_dims: list[int],
180 |         state_latent_dim: int,
181 |         privileged_hidden_dims: list[int],
182 |         critic_hidden_dims: list[int],
183 |         activation: str,
184 |     ) -> None:
185 |         super().__init__()
186 |         self.state_dim = int(state_dim)
187 |         self.map_dim = int(map_dim)
188 |         self.privileged_dim = int(privileged_dim)
189 |         self.terrain_channels = int(terrain_channels)
190 |         self.terrain_grid_shape = tuple(int(v) for v in terrain_grid_shape)
191 | 
192 |         self.map_encoder = _TerrainMapEncoder(
193 |             input_channels=self.terrain_channels,
194 |             grid_shape=self.terrain_grid_shape,
195 |             conv_channels=terrain_cnn_channels,
196 |             latent_dim=terrain_latent_dim,
197 |             activation=activation,
198 |         )
199 |         self.state_encoder = _build_mlp(
200 |             input_dim=self.state_dim,
201 |             hidden_dims=state_hidden_dims,
202 |             output_dim=state_latent_dim,
203 |             activation=activation,
204 |         )
205 |         self.privileged_encoder = _build_mlp(
206 |             input_dim=self.privileged_dim,
207 |             hidden_dims=privileged_hidden_dims,
208 |             output_dim=privileged_hidden_dims[-1] if privileged_hidden_dims else self.privileged_dim,
209 |             activation=activation,
210 |         )
211 |         privileged_latent_dim = privileged_hidden_dims[-1] if privileged_hidden_dims else self.privileged_dim
212 |         self.value_head = _build_mlp(
213 |             input_dim=terrain_latent_dim + state_latent_dim + privileged_latent_dim,
214 |             hidden_dims=critic_hidden_dims,
215 |             output_dim=1,
216 |             activation=activation,
217 |         )
218 | 
219 |     def forward(self, obs: torch.Tensor) -> torch.Tensor:
220 |         state_obs = obs[:, : self.state_dim]
221 |         map_start = self.state_dim
222 |         map_end = map_start + self.map_dim
223 |         map_obs = obs[:, map_start:map_end]
224 |         priv_obs = obs[:, map_end : map_end + self.privileged_dim]
225 |         map_obs = map_obs.view(
226 |             obs.shape[0], self.terrain_channels, self.terrain_grid_shape[0], self.terrain_grid_shape[1]
227 |         )
228 | 
229 |         z_state = self.state_encoder(state_obs)
230 |         z_map = self.map_encoder(map_obs)
231 |         z_priv = self.privileged_encoder(priv_obs)
232 |         return self.value_head(torch.cat((z_map, z_state, z_priv), dim=-1))
233 | 
234 | 
235 | class RangerTerrainActorCritic(ActorCritic):
236 |     is_recurrent: bool = False
237 | 
238 |     def __init__(
239 |         self,
240 |         obs: TensorDict,
241 |         obs_groups: dict[str, list[str]],
242 |         num_actions: int,
243 |         actor_obs_normalization: bool = False,
244 |         critic_obs_normalization: bool = False,
245 |         activation: str = "elu",
246 |         init_noise_std: float = 1.0,
247 |         noise_std_type: str = "scalar",
248 |         state_dependent_std: bool = False,
249 |         terrain_obs_group: str = "policy_map",
250 |         state_obs_group: str = "policy_state",
251 |         privileged_obs_group: str = "critic_privileged",
252 |         terrain_channels: int = 8,
253 |         terrain_grid_shape: tuple[int, int] = (21, 13),
254 |         terrain_cnn_channels: list[int] = [16, 32, 32],
255 |         terrain_latent_dim: int = 128,
256 |         state_hidden_dims: list[int] = [128],
257 |         state_latent_dim: int = 128,
258 |         actor_hidden_dims: list[int] | None = None,
259 |         actor_trunk_hidden_dims: list[int] = [256, 128],
260 |         critic_hidden_dims: list[int] = [256, 128],
261 |         privileged_hidden_dims: list[int] = [64],
262 |         wheel_head_hidden_dims: list[int] = [128],
263 |         suspension_head_hidden_dims: list[int] = [128],
264 |         action_training_mask: list[float] | None = None,
265 |         action_output_mask: list[float] | None = None,
266 |         action_exploration_mask: list[float] | None = None,
267 |         initial_action_std: list[float] | None = None,
268 |         inactive_action_std: float = 1.0e-6,
269 |         **kwargs: dict[str, Any],
270 |     ) -> None:
271 |         if kwargs:
272 |             print(
273 |                 "RangerTerrainActorCritic.__init__ got unexpected arguments, which will be ignored: "
274 |                 + str([key for key in kwargs])
275 |             )
276 |         if state_dependent_std:
277 |             raise NotImplementedError("RangerTerrainActorCritic does not implement state-dependent std in V1.")
278 |         nn.Module.__init__(self)
279 | 
280 |         self.obs_groups = obs_groups
281 |         self.terrain_obs_group = terrain_obs_group
282 |         self.state_obs_group = state_obs_group
283 |         self.privileged_obs_group = privileged_obs_group
284 |         self.state_dependent_std = state_dependent_std
285 |         if inactive_action_std <= 0.0:
286 |             raise ValueError(f"inactive_action_std must be positive, got {inactive_action_std}.")
287 | 
288 |         def _binary_mask(name: str, values: list[float] | None, default: list[float]) -> torch.Tensor:
289 |             mask_values = default if values is None else values
290 |             if len(mask_values) != num_actions:
291 |                 raise ValueError(f"{name} must have {num_actions} entries, got {len(mask_values)}.")
292 |             mask = torch.tensor(mask_values, dtype=torch.float32)
293 |             if not torch.all((mask == 0.0) | (mask == 1.0)):
294 |                 raise ValueError(f"{name} entries must be exactly 0.0 or 1.0.")
295 |             return mask
296 | 
297 |         action_training_mask_tensor = _binary_mask(
298 |             "action_training_mask", action_training_mask, [1.0] * num_actions
299 |         )
300 |         if not torch.any(action_training_mask_tensor > 0.0):
301 |             raise ValueError("action_training_mask must keep at least one trainable action dimension.")
302 |         action_output_mask_tensor = _binary_mask(
303 |             "action_output_mask", action_output_mask, [1.0] * num_actions
304 |         )
305 |         action_exploration_mask_tensor = _binary_mask(
306 |             "action_exploration_mask", action_exploration_mask, action_training_mask_tensor.tolist()
307 |         )
308 |         if initial_action_std is None:
309 |             initial_action_std_tensor = init_noise_std * torch.ones(num_actions, dtype=torch.float32)
310 |         else:
311 |             if len(initial_action_std) != num_actions:
312 |                 raise ValueError(
313 |                     f"initial_action_std must have {num_actions} entries, got {len(initial_action_std)}."
314 |                 )
315 |             initial_action_std_tensor = torch.tensor(initial_action_std, dtype=torch.float32)
316 |             if torch.any(initial_action_std_tensor <= 0.0):
317 |                 raise ValueError("initial_action_std entries must all be positive.")
318 | 
319 |         self.register_buffer("_action_training_mask", action_training_mask_tensor, persistent=False)
320 |         self.register_buffer("_action_output_mask", action_output_mask_tensor, persistent=False)
321 |         self.register_buffer("_action_exploration_mask", action_exploration_mask_tensor, persistent=False)
322 |         self.register_buffer("_configured_initial_action_std", initial_action_std_tensor, persistent=False)
323 |         self._inactive_action_std = float(inactive_action_std)
324 |         if not torch.all(action_training_mask_tensor == 1.0) or not torch.all(action_output_mask_tensor == 1.0):
325 |             print(
326 |                 "[RangerPolicy] action masks enabled: "
327 |                 f"training={action_training_mask_tensor.tolist()} "
328 |                 f"output={action_output_mask_tensor.tolist()} "
329 |                 f"exploration={action_exploration_mask_tensor.tolist()} "
330 |                 f"inactive_std={self._inactive_action_std:.1e}"
331 |             )
332 | 
333 |         self.num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
334 |         self.num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])
335 |         self.state_dim = int(obs[self.state_obs_group].shape[-1])
336 |         self.map_dim = int(obs[self.terrain_obs_group].shape[-1])
337 |         self.privileged_dim = int(obs[self.privileged_obs_group].shape[-1])
338 |         if actor_hidden_dims is not None:
339 |             actor_trunk_hidden_dims = actor_hidden_dims
340 | 
341 |         self.actor = _RangerDualHeadActor(
342 |             state_dim=self.state_dim,
343 |             map_dim=self.map_dim,
344 |             num_actions=num_actions,
345 |             terrain_channels=terrain_channels,
346 |             terrain_grid_shape=terrain_grid_shape,
347 |             terrain_cnn_channels=terrain_cnn_channels,
348 |             terrain_latent_dim=terrain_latent_dim,
349 |             state_hidden_dims=state_hidden_dims,
350 |             state_latent_dim=state_latent_dim,
351 |             actor_trunk_hidden_dims=actor_trunk_hidden_dims,
352 |             wheel_head_hidden_dims=wheel_head_hidden_dims,
353 |             suspension_head_hidden_dims=suspension_head_hidden_dims,
354 |             activation=activation,
355 |         )
356 |         print(f"Ranger actor: {self.actor}")
357 | 
358 |         self.critic = _RangerCritic(
359 |             state_dim=self.state_dim,
360 |             map_dim=self.map_dim,
361 |             privileged_dim=self.privileged_dim,
362 |             terrain_channels=terrain_channels,
363 |             terrain_grid_shape=terrain_grid_shape,
364 |             terrain_cnn_channels=terrain_cnn_channels,
365 |             terrain_latent_dim=terrain_latent_dim,
366 |             state_hidden_dims=state_hidden_dims,
367 |             state_latent_dim=state_latent_dim,
368 |             privileged_hidden_dims=privileged_hidden_dims,
369 |             critic_hidden_dims=critic_hidden_dims,
370 |             activation=activation,
371 |         )
372 |         print(f"Ranger critic: {self.critic}")
373 | 
374 |         self.actor_obs_normalization = actor_obs_normalization
375 |         self.actor_obs_normalizer = (
376 |             EmpiricalNormalization(self.num_actor_obs) if actor_obs_normalization else torch.nn.Identity()
377 |         )
378 |         self.critic_obs_normalization = critic_obs_normalization
379 |         self.critic_obs_normalizer = (
380 |             EmpiricalNormalization(self.num_critic_obs) if critic_obs_normalization else torch.nn.Identity()
381 |         )
382 | 
383 |         self.noise_std_type = noise_std_type
384 |         if self.noise_std_type == "scalar":
385 |             self.std = nn.Parameter(self._configured_initial_action_std.clone())
386 |         elif self.noise_std_type == "log":
387 |             self.log_std = nn.Parameter(torch.log(self._configured_initial_action_std.clone()))
388 |         else:
389 |             raise ValueError(
390 |                 f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'."
391 |             )
392 | 
393 |         self.distribution = None
394 |         Normal.set_default_validate_args(False)
395 | 
396 |     def reset(self, dones: torch.Tensor | None = None) -> None:
397 |         pass
398 | 
399 |     def forward(self):
400 |         raise NotImplementedError
401 | 
402 |     @property
403 |     def action_mean(self) -> torch.Tensor:
404 |         return self.distribution.mean
405 | 
406 |     @property
407 |     def action_std(self) -> torch.Tensor:
408 |         return self.distribution.stddev
409 | 
410 |     @property
411 |     def entropy(self) -> torch.Tensor:
412 |         entropy = self.distribution.entropy()
413 |         mask = self._action_training_mask.to(device=entropy.device, dtype=entropy.dtype)
414 |         return (entropy * mask).sum(dim=-1)
415 | 
416 |     def _apply_action_output_mask(self, mean: torch.Tensor) -> torch.Tensor:
417 |         mask = self._action_output_mask.to(device=mean.device, dtype=mean.dtype)
418 |         return mean * mask
419 | 
420 |     def _update_distribution(self, obs: TensorDict) -> None:
421 |         actor_obs = self.get_actor_obs(obs)
422 |         actor_obs = self.actor_obs_normalizer(actor_obs)
423 |         mean = self._apply_action_output_mask(self.actor(actor_obs))
424 |         exploration_mask = self._action_exploration_mask.to(device=mean.device, dtype=mean.dtype)
425 |         if self.noise_std_type == "scalar":
426 |             std = self.std.expand_as(mean)
427 |         else:
428 |             std = torch.exp(self.log_std).expand_as(mean)
429 |         std = std * exploration_mask + self._inactive_action_std * (1.0 - exploration_mask)
430 |         self.distribution = Normal(mean, std)
431 | 
432 |     def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
433 |         self._update_distribution(obs)
434 |         return self.distribution.sample()
435 | 
436 |     def act_inference(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
437 |         if isinstance(obs, TensorDict):
438 |             actor_obs = self.get_actor_obs(obs)
439 |         else:
440 |             actor_obs = obs
441 |         actor_obs = self.actor_obs_normalizer(actor_obs)
442 |         return self._apply_action_output_mask(self.actor(actor_obs))
443 | 
444 |     def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
445 |         critic_obs = self.get_critic_obs(obs)
446 |         critic_obs = self.critic_obs_normalizer(critic_obs)
447 |         return self.critic(critic_obs)
448 | 
449 |     def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
450 |         return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)
451 | 
452 |     def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
453 |         return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)
454 | 
455 |     def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
456 |         log_prob = self.distribution.log_prob(actions)
457 |         mask = self._action_training_mask.to(device=log_prob.device, dtype=log_prob.dtype)
458 |         return (log_prob * mask).sum(dim=-1)
459 | 
460 |     def update_normalization(self, obs: TensorDict) -> None:
461 |         if self.actor_obs_normalization:
462 |             self.actor_obs_normalizer.update(self.get_actor_obs(obs))
463 |         if self.critic_obs_normalization:
464 |             self.critic_obs_normalizer.update(self.get_critic_obs(obs))
465 | 
466 |     def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
467 |         super().load_state_dict(state_dict, strict=strict)
468 |         return True
469 | 
```

### source/Ranger/Ranger/tasks/manager_based/ranger/agents/rsl_rl_ppo_cfg.py

Bytes: 8038
SHA-256: 643f8ae99e7ceb68b1b3545d2d4274a8b161d91b7751523742b9ba16fc1ed980
Lines: 1-229 of 229

```python
  1 | # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
  2 | # All rights reserved.
  3 | #
  4 | # SPDX-License-Identifier: BSD-3-Clause
  5 | 
  6 | from isaaclab.utils import configclass
  7 | 
  8 | from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
  9 | 
 10 | 
 11 | @configclass
 12 | class RangerTerrainActorCriticCfg(RslRlPpoActorCriticCfg):
 13 |     class_name: str = "RangerTerrainActorCritic"
 14 |     actor_obs_normalization: bool = False
 15 |     critic_obs_normalization: bool = False
 16 |     actor_hidden_dims: list[int] = [256, 128]
 17 |     critic_hidden_dims: list[int] = [256, 128]
 18 |     activation: str = "elu"
 19 | 
 20 |     terrain_obs_group: str = "policy_map"
 21 |     state_obs_group: str = "policy_state"
 22 |     privileged_obs_group: str = "critic_privileged"
 23 |     terrain_channels: int = 8
 24 |     terrain_grid_shape: tuple[int, int] = (21, 13)
 25 |     terrain_cnn_channels: list[int] = [16, 32, 32]
 26 |     terrain_latent_dim: int = 128
 27 |     state_hidden_dims: list[int] = [128]
 28 |     state_latent_dim: int = 128
 29 |     privileged_hidden_dims: list[int] = [64]
 30 |     wheel_head_hidden_dims: list[int] = [128]
 31 |     suspension_head_hidden_dims: list[int] = [128]
 32 |     action_training_mask: list[float] | None = None
 33 |     action_output_mask: list[float] | None = None
 34 |     action_exploration_mask: list[float] | None = None
 35 |     initial_action_std: list[float] | None = None
 36 |     inactive_action_std: float = 1.0e-6
 37 | 
 38 | 
 39 | @configclass
 40 | class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
 41 |     num_steps_per_env = 16
 42 |     max_iterations = 150
 43 |     save_interval = 50
 44 |     experiment_name = "ranger_direct"
 45 |     empirical_normalization = False
 46 |     obs_groups = {
 47 |         "policy": ["policy_state", "policy_map"],
 48 |         "critic": ["policy_state", "policy_map", "critic_privileged"],
 49 |     }
 50 |     policy = RangerTerrainActorCriticCfg(
 51 |         init_noise_std=1.0,
 52 |     )
 53 |     algorithm = RslRlPpoAlgorithmCfg(
 54 |         value_loss_coef=1.0,
 55 |         use_clipped_value_loss=True,
 56 |         clip_param=0.2,
 57 |         entropy_coef=0.005,
 58 |         num_learning_epochs=5,
 59 |         num_mini_batches=4,
 60 |         learning_rate=1.0e-3,
 61 |         schedule="adaptive",
 62 |         gamma=0.99,
 63 |         lam=0.95,
 64 |         desired_kl=0.01,
 65 |         max_grad_norm=1.0,
 66 |     )
 67 | 
 68 | 
 69 | @configclass
 70 | class StandPPORunnerCfg(PPORunnerCfg):
 71 |     policy = RangerTerrainActorCriticCfg(
 72 |         init_noise_std=0.5,
 73 |     )
 74 | 
 75 | 
 76 | @configclass
 77 | class ShortGoalFlatPPORunnerCfg(PPORunnerCfg):
 78 |     """悬架车轮完全参与"""
 79 |     policy = RangerTerrainActorCriticCfg(
 80 |         init_noise_std=0.5,
 81 |     )
 82 | 
 83 |     def __post_init__(self) -> None:
 84 |         post_init = getattr(super(), "__post_init__", None)
 85 |         if post_init is not None:
 86 |             post_init()
 87 |         self.algorithm.learning_rate = 3.0e-4
 88 |         self.algorithm.entropy_coef = 1.0e-3
 89 | 
 90 | 
 91 | @configclass
 92 | class ShortGoalFlatV2PPORunnerCfg(ShortGoalFlatPPORunnerCfg):
 93 |     """Wheel-only PPO loss while preserving the Ranger 8-D actor/checkpoint layout."""
 94 |     """悬架置0,不参与训练和探索，不输出动作，车轮正常"""
 95 | 
 96 |     policy = RangerTerrainActorCriticCfg(
 97 |         init_noise_std=0.5,
 98 |         action_training_mask=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
 99 |         action_output_mask=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
100 |         action_exploration_mask=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
101 |         initial_action_std=[0.05, 0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 0.25],
102 |         inactive_action_std=1.0e-6,
103 |     )
104 | 
105 | 
106 | @configclass
107 | class ShortGoalFlatFrozenSuspensionWheelPPORunnerCfg(ShortGoalFlatPPORunnerCfg):
108 |     """Execute the preserved suspension policy deterministically and train wheel actions only."""
109 | 
110 |     policy = RangerTerrainActorCriticCfg(
111 |         init_noise_std=0.10,
112 |         action_training_mask=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
113 |         action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
114 |         action_exploration_mask=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
115 |         initial_action_std=[0.04, 0.04, 0.04, 0.04, 0.10, 0.10, 0.10, 0.10],
116 |         inactive_action_std=1.0e-6,
117 |     )
118 | 
119 |     def __post_init__(self) -> None:
120 |         super().__post_init__()
121 |         self.algorithm.learning_rate = 1.0e-4
122 |         self.algorithm.entropy_coef = 5.0e-4
123 | 
124 | 
125 | @configclass
126 | class ShortGoalFlatLimitedSuspensionPPORunnerCfg(ShortGoalFlatPPORunnerCfg):
127 |     """Train only the suspension dimensions while executing a deterministic wheel policy."""
128 |     """只训练悬架，车轮执行确定性策略，只执行5%悬架动作"""
129 | 
130 |     policy = RangerTerrainActorCriticCfg(
131 |         init_noise_std=0.05,
132 |         action_training_mask=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
133 |         action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
134 |         action_exploration_mask=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
135 |         initial_action_std=[0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10],
136 |         inactive_action_std=1.0e-6,
137 |     )
138 | 
139 |     def __post_init__(self) -> None:
140 |         super().__post_init__()
141 |         self.algorithm.learning_rate = 1.0e-4
142 |         self.algorithm.entropy_coef = 5.0e-4
143 | 
144 | 
145 | @configclass
146 | class ShortGoalFlatLimitedJointPPORunnerCfg(ShortGoalFlatPPORunnerCfg):
147 |     """Low-noise limited joint fine-tuning of both Ranger action heads."""
148 |     """悬架和车轮都输出动作并探索，悬架只执行5%动作"""
149 | 
150 |     policy = RangerTerrainActorCriticCfg(
151 |         init_noise_std=0.10,
152 |         action_training_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
153 |         action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
154 |         action_exploration_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
155 |         initial_action_std=[0.04, 0.04, 0.04, 0.04, 0.10, 0.10, 0.10, 0.10],
156 |         inactive_action_std=1.0e-6,
157 |     )
158 | 
159 |     def __post_init__(self) -> None:
160 |         super().__post_init__()
161 |         self.algorithm.learning_rate = 7.5e-5
162 |         self.algorithm.entropy_coef = 5.0e-4
163 | 
164 | 
165 | @configclass
166 | class ShortGoalFlatV7PPORunnerCfg(ShortGoalFlatPPORunnerCfg):
167 |     """Stage V7: low-noise full-authority fine-tuning of both Ranger action heads."""
168 | 
169 |     max_iterations = 150
170 |     save_interval = 25
171 |     policy = RangerTerrainActorCriticCfg(
172 |         init_noise_std=0.05,
173 |         action_training_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
174 |         action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
175 |         action_exploration_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
176 |         initial_action_std=[0.005, 0.005, 0.005, 0.005, 0.05, 0.05, 0.05, 0.05],
177 |         inactive_action_std=1.0e-6,
178 |     )
179 | 
180 |     def __post_init__(self) -> None:
181 |         super().__post_init__()
182 |         self.algorithm.learning_rate = 5.0e-5
183 |         self.algorithm.entropy_coef = 1.0e-4
184 | 
185 | 
186 | @configclass
187 | class ShortGoalFlatV8PPORunnerCfg(ShortGoalFlatV7PPORunnerCfg):
188 |     """Stage V8: fine-tune both action heads against the balanced-stop success gate."""
189 | 
190 |     max_iterations = 500
191 |     save_interval = 25
192 | 
193 | 
194 | @configclass
195 | class ShortGoalFlatV9PPORunnerCfg(ShortGoalFlatV7PPORunnerCfg):
196 |     """Stage V9: jointly fine-tune both action heads for policy-controlled balanced stopping."""
197 | 
198 |     max_iterations = 500
199 |     save_interval = 25
200 |     policy = RangerTerrainActorCriticCfg(
201 |         init_noise_std=0.01,
202 |         action_training_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
203 |         action_output_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
204 |         action_exploration_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
205 |         initial_action_std=[0.003, 0.003, 0.003, 0.003, 0.01, 0.01, 0.01, 0.01],
206 |         inactive_action_std=1.0e-6,
207 |     )
208 | 
209 |     def __post_init__(self) -> None:
210 |         super().__post_init__()
211 |         self.algorithm.learning_rate = 1.0e-5
212 |         self.algorithm.entropy_coef = 1.0e-4
213 | 
214 | 
215 | @configclass
216 | class ShortGoalFlatV10PPORunnerCfg(ShortGoalFlatV9PPORunnerCfg):
217 |     """Stage V10: low-noise adaptation to 0.50 m capture with a precision bonus."""
218 | 
219 |     max_iterations = 300
220 |     save_interval = 25
221 | 
222 | 
223 | @configclass
224 | class ShortGoalFlatV11PPORunnerCfg(ShortGoalFlatV10PPORunnerCfg):
225 |     """Stage V11: jointly adapt both actor heads for policy-controlled braking."""
226 | 
227 |     max_iterations = 200
228 |     save_interval = 25
229 | 
```

### source/Ranger/Ranger/tasks/manager_based/ranger/mdp/observations.py

Bytes: 74626
SHA-256: 33a5673a9b77d486e20840cd1237263ab48db444bcccd4615a462f9e3c2108b5
Lines: 1-1794 of 1794

```python
   1 | # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
   2 | # All rights reserved.
   3 | #
   4 | # SPDX-License-Identifier: BSD-3-Clause
   5 | 
   6 | """Observation terms for Ranger perception."""
   7 | 
   8 | from __future__ import annotations
   9 | 
  10 | import math
  11 | import os
  12 | import torch
  13 | import torch.nn.functional as F
  14 | 
  15 | import isaaclab.utils.math as math_utils
  16 | from isaaclab.assets import Articulation
  17 | from isaaclab.envs import ManagerBasedEnv
  18 | from isaaclab.managers import SceneEntityCfg
  19 | from isaaclab.sensors import RayCaster
  20 | 
  21 | 
  22 | SPEED_COMMAND_ATTR = "_ranger_speed_command"
  23 | SPEED_COMMAND_TARGET_ATTR = "_ranger_speed_command_target"
  24 | SPEED_COMMAND_TIMER_ATTR = "_ranger_speed_command_timer"
  25 | SPEED_COMMAND_DURATION_ATTR = "_ranger_speed_command_duration"
  26 | GOAL_HEADING_TARGET_ATTR = "_ranger_goal_heading_target_pos_w"
  27 | GOAL_HEADING_PREV_HEADING_ERROR_ATTR = "_ranger_goal_heading_prev_heading_error"
  28 | SHORT_GOAL_TARGET_ATTR = "_ranger_short_goal_pos_w"
  29 | SHORT_GOAL_PREV_DISTANCE_ATTR = "_ranger_short_goal_prev_distance"
  30 | SHORT_GOAL_REACHED_ATTR = "_ranger_short_goal_reached"
  31 | SHORT_GOAL_PREV_HEADING_ERROR_ATTR = "_ranger_short_goal_prev_heading_error"
  32 | YAW_RATE_COMMAND_ATTR = "_ranger_yaw_rate_command"
  33 | SUSPENSION_STROKE_PREV_ATTR = "_ranger_suspension_stroke_prev"
  34 | SUSPENSION_STROKE_RATE_ATTR = "_ranger_suspension_stroke_rate"
  35 | SUSPENSION_STROKE_RATE_STEP_ATTR = "_ranger_suspension_stroke_rate_step"
  36 | COMMAND_OBS_CACHE_ATTR = "_ranger_command_obs_cache"
  37 | COMMAND_OBS_CACHE_STEP_ATTR = "_ranger_command_obs_cache_step"
  38 | WHEEL_CONTACT_SENSOR_BODY_IDS_ATTR = "_ranger_wheel_contact_sensor_body_ids"
  39 | 
  40 | 
  41 | def _obs_debug_enabled() -> bool:
  42 |     return os.getenv("RANGER_OBS_DEBUG", "0") == "1"
  43 | 
  44 | 
  45 | def _obs_debug(msg: str) -> None:
  46 |     if _obs_debug_enabled():
  47 |         print(f"[OBS_DEBUG] {msg}", flush=True)
  48 | 
  49 | 
  50 | def _policy_state_disabled() -> bool:
  51 |     return os.getenv("RANGER_DISABLE_POLICY_STATE", "0") == "1"
  52 | 
  53 | 
  54 | def _policy_map_disabled() -> bool:
  55 |     return os.getenv("RANGER_DISABLE_POLICY_MAP", "0") == "1"
  56 | 
  57 | 
  58 | def _critic_privileged_disabled() -> bool:
  59 |     return os.getenv("RANGER_DISABLE_CRITIC_PRIVILEGED", "0") == "1"
  60 | 
  61 | 
  62 | def _as_env_ids(env: ManagerBasedEnv, env_ids) -> torch.Tensor:
  63 |     if env_ids is None or isinstance(env_ids, slice):
  64 |         return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  65 |     if isinstance(env_ids, torch.Tensor):
  66 |         return env_ids.to(device=env.device, dtype=torch.long)
  67 |     return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
  68 | 
  69 | 
  70 | def _env_float(name: str, default: float) -> float:
  71 |     raw = os.getenv(name)
  72 |     if raw is None or raw == "":
  73 |         return float(default)
  74 |     return float(raw)
  75 | 
  76 | 
  77 | def _ensure_speed_command_buffers(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  78 |     command = getattr(env, SPEED_COMMAND_ATTR, None)
  79 |     command_target = getattr(env, SPEED_COMMAND_TARGET_ATTR, None)
  80 |     command_timer = getattr(env, SPEED_COMMAND_TIMER_ATTR, None)
  81 |     command_duration = getattr(env, SPEED_COMMAND_DURATION_ATTR, None)
  82 |     needs_init = (
  83 |         command is None
  84 |         or command_target is None
  85 |         or command_timer is None
  86 |         or command_duration is None
  87 |         or command.shape != (env.num_envs, 2)
  88 |         or command_target.shape != (env.num_envs, 2)
  89 |         or command_timer.shape != (env.num_envs,)
  90 |         or command_duration.shape != (env.num_envs,)
  91 |     )
  92 |     if needs_init:
  93 |         command = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
  94 |         command_target = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
  95 |         command_timer = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  96 |         command_duration = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
  97 |         setattr(env, SPEED_COMMAND_ATTR, command)
  98 |         setattr(env, SPEED_COMMAND_TARGET_ATTR, command_target)
  99 |         setattr(env, SPEED_COMMAND_TIMER_ATTR, command_timer)
 100 |         setattr(env, SPEED_COMMAND_DURATION_ATTR, command_duration)
 101 |     return command, command_target, command_timer, command_duration
 102 | 
 103 | 
 104 | def _sample_speed_command_targets(
 105 |     env: ManagerBasedEnv,
 106 |     env_ids: torch.Tensor,
 107 |     stage: str,
 108 |     v_x_range: tuple[float, float],
 109 |     yaw_rate_range: tuple[float, float],
 110 |     command_duration_range: tuple[float, float],
 111 |     small_yaw_prob: float = 0.0,
 112 |     small_yaw_range: tuple[float, float] = (0.03, 0.08),
 113 |     v_x_bins: tuple[tuple[float, float], ...] | None = None,
 114 | ) -> None:
 115 |     command, command_target, command_timer, command_duration = _ensure_speed_command_buffers(env)
 116 |     if env_ids.numel() == 0:
 117 |         return
 118 | 
 119 |     num = env_ids.numel()
 120 |     if v_x_bins:
 121 |         selector = torch.randint(len(v_x_bins), (num,), device=env.device)
 122 |         v_x_target = torch.empty(num, device=env.device, dtype=torch.float32)
 123 |         for bin_id, speed_range in enumerate(v_x_bins):
 124 |             mask = selector == bin_id
 125 |             if torch.any(mask):
 126 |                 v_x_target[mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 127 |                     float(speed_range[0]), float(speed_range[1])
 128 |                 )[mask]
 129 |         command_target[env_ids, 0] = v_x_target
 130 |     else:
 131 |         command_target[env_ids, 0] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 132 |             float(v_x_range[0]), float(v_x_range[1])
 133 |         )
 134 |     if stage.upper() == "A":
 135 |         selector = torch.rand(num, device=env.device)
 136 |         yaw_target = torch.zeros(num, device=env.device, dtype=torch.float32)
 137 |         half_prob = 0.5 * max(float(small_yaw_prob), 0.0)
 138 |         neg_mask = selector < half_prob
 139 |         pos_mask = (selector >= half_prob) & (selector < 2.0 * half_prob)
 140 |         if torch.any(neg_mask):
 141 |             yaw_target[neg_mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 142 |                 -float(small_yaw_range[1]), -float(small_yaw_range[0])
 143 |             )[neg_mask]
 144 |         if torch.any(pos_mask):
 145 |             yaw_target[pos_mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 146 |                 float(small_yaw_range[0]), float(small_yaw_range[1])
 147 |             )[pos_mask]
 148 |         command_target[env_ids, 1] = yaw_target
 149 |     else:
 150 |         selector = torch.rand(num, device=env.device)
 151 |         yaw_target = torch.zeros(num, device=env.device, dtype=torch.float32)
 152 |         neg_mask = (selector >= 0.4) & (selector < 0.7)
 153 |         pos_mask = selector >= 0.7
 154 |         if torch.any(neg_mask):
 155 |             yaw_target[neg_mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 156 |                 float(yaw_rate_range[0]), -0.12
 157 |             )[neg_mask]
 158 |         if torch.any(pos_mask):
 159 |             yaw_target[pos_mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 160 |                 0.12, float(yaw_rate_range[1])
 161 |             )[pos_mask]
 162 |         command_target[env_ids, 1] = yaw_target
 163 | 
 164 |     command_timer[env_ids] = 0.0
 165 |     command_duration[env_ids] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 166 |         float(command_duration_range[0]), float(command_duration_range[1])
 167 |     )
 168 |     just_initialized = torch.all(command[env_ids] == 0.0, dim=1)
 169 |     if torch.any(just_initialized):
 170 |         command[env_ids[just_initialized]] = command_target[env_ids[just_initialized]]
 171 | 
 172 | 
 173 | def reset_speed_command(
 174 |     env: ManagerBasedEnv,
 175 |     env_ids,
 176 |     stage: str = "A",
 177 |     v_x_range: tuple[float, float] = (0.05, 0.4),
 178 |     yaw_rate_range: tuple[float, float] = (-0.4, 0.4),
 179 |     command_duration_range: tuple[float, float] = (2.0, 5.0),
 180 |     small_yaw_prob: float = 0.0,
 181 |     small_yaw_range: tuple[float, float] = (0.03, 0.08),
 182 |     v_x_bins: tuple[tuple[float, float], ...] | None = None,
 183 | ) -> None:
 184 |     """Initialize per-env speed-command buffers at reset."""
 185 | 
 186 |     env_ids = _as_env_ids(env, env_ids)
 187 |     command, _, _, _ = _ensure_speed_command_buffers(env)
 188 |     command[env_ids] = 0.0
 189 |     _sample_speed_command_targets(
 190 |         env=env,
 191 |         env_ids=env_ids,
 192 |         stage=stage,
 193 |         v_x_range=v_x_range,
 194 |         yaw_rate_range=yaw_rate_range,
 195 |         command_duration_range=command_duration_range,
 196 |         small_yaw_prob=small_yaw_prob,
 197 |         small_yaw_range=small_yaw_range,
 198 |         v_x_bins=v_x_bins,
 199 |     )
 200 | 
 201 | 
 202 | def update_speed_command(
 203 |     env: ManagerBasedEnv,
 204 |     stage: str = "A",
 205 |     v_x_range: tuple[float, float] = (0.05, 0.4),
 206 |     yaw_rate_range: tuple[float, float] = (-0.4, 0.4),
 207 |     command_duration_range: tuple[float, float] = (2.0, 5.0),
 208 |     smoothing_alpha: float = 0.1,
 209 |     max_command_duration: float = 5.0,
 210 |     small_yaw_prob: float = 0.0,
 211 |     small_yaw_range: tuple[float, float] = (0.03, 0.08),
 212 |     v_x_bins: tuple[tuple[float, float], ...] | None = None,
 213 | ) -> torch.Tensor:
 214 |     """Update and return the smoothed dynamic speed command."""
 215 | 
 216 |     command, command_target, command_timer, command_duration = _ensure_speed_command_buffers(env)
 217 |     dt = float(getattr(env, "step_dt", 1.0 / 60.0))
 218 |     command_timer += dt
 219 |     expired = command_timer >= command_duration
 220 |     if torch.any(expired):
 221 |         _sample_speed_command_targets(
 222 |             env=env,
 223 |             env_ids=expired.nonzero(as_tuple=False).squeeze(-1),
 224 |             stage=stage,
 225 |             v_x_range=v_x_range,
 226 |             yaw_rate_range=yaw_rate_range,
 227 |             command_duration_range=command_duration_range,
 228 |             small_yaw_prob=small_yaw_prob,
 229 |             small_yaw_range=small_yaw_range,
 230 |             v_x_bins=v_x_bins,
 231 |         )
 232 |     alpha = float(smoothing_alpha)
 233 |     command[:] = command + alpha * (command_target - command)
 234 |     return command
 235 | 
 236 | 
 237 | def speed_command(env: ManagerBasedEnv) -> torch.Tensor:
 238 |     """Return current smoothed ``[v_x_cmd, yaw_rate_cmd]`` without updating it."""
 239 | 
 240 |     command, _, _, _ = _ensure_speed_command_buffers(env)
 241 |     return command
 242 | 
 243 | 
 244 | def speed_command_target(env: ManagerBasedEnv) -> torch.Tensor:
 245 |     """Return current command target ``[v_x_cmd_target, yaw_rate_cmd_target]``."""
 246 | 
 247 |     _, command_target, _, _ = _ensure_speed_command_buffers(env)
 248 |     return command_target
 249 | 
 250 | 
 251 | def speed_command_time_left(env: ManagerBasedEnv) -> torch.Tensor:
 252 |     """Return seconds until each command target is resampled."""
 253 | 
 254 |     _, _, command_timer, command_duration = _ensure_speed_command_buffers(env)
 255 |     return torch.clamp(command_duration - command_timer, min=0.0)
 256 | 
 257 | 
 258 | def speed_command_state(
 259 |     env: ManagerBasedEnv,
 260 |     stage: str = "A",
 261 |     v_x_range: tuple[float, float] = (0.05, 0.4),
 262 |     yaw_rate_range: tuple[float, float] = (-0.4, 0.4),
 263 |     command_duration_range: tuple[float, float] = (2.0, 5.0),
 264 |     smoothing_alpha: float = 0.1,
 265 |     max_command_duration: float = 5.0,
 266 |     small_yaw_prob: float = 0.0,
 267 |     small_yaw_range: tuple[float, float] = (0.03, 0.08),
 268 |     v_x_bins: tuple[tuple[float, float], ...] | None = None,
 269 | ) -> torch.Tensor:
 270 |     """Return six-dimensional dynamic velocity-command observation."""
 271 | 
 272 |     command = update_speed_command(
 273 |         env=env,
 274 |         stage=stage,
 275 |         v_x_range=v_x_range,
 276 |         yaw_rate_range=yaw_rate_range,
 277 |         command_duration_range=command_duration_range,
 278 |         smoothing_alpha=smoothing_alpha,
 279 |         small_yaw_prob=small_yaw_prob,
 280 |         small_yaw_range=small_yaw_range,
 281 |         v_x_bins=v_x_bins,
 282 |     )
 283 |     _, command_target, _, _ = _ensure_speed_command_buffers(env)
 284 |     time_left = speed_command_time_left(env)
 285 |     obs = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
 286 |     obs[:, 0] = torch.clamp(command[:, 0] / 0.45, min=-1.0, max=1.0)
 287 |     obs[:, 1] = torch.clamp(command[:, 1], min=-1.0, max=1.0)
 288 |     obs[:, 2] = torch.sin(command[:, 1])
 289 |     obs[:, 3] = torch.cos(command[:, 1])
 290 |     obs[:, 4] = torch.clamp(command_target[:, 0] / 0.45, min=-1.0, max=1.0)
 291 |     obs[:, 5] = torch.clamp(time_left / max(float(max_command_duration), 1.0e-6), min=0.0, max=1.0)
 292 |     return obs
 293 | 
 294 | 
 295 | def _ensure_goal_heading_target(env: ManagerBasedEnv) -> torch.Tensor:
 296 |     target_pos_w = getattr(env, GOAL_HEADING_TARGET_ATTR, None)
 297 |     if target_pos_w is None or target_pos_w.shape != (env.num_envs, 3):
 298 |         target_pos_w = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
 299 |         setattr(env, GOAL_HEADING_TARGET_ATTR, target_pos_w)
 300 |     return target_pos_w
 301 | 
 302 | 
 303 | def _ensure_short_goal_target(env: ManagerBasedEnv) -> torch.Tensor:
 304 |     target_pos_w = getattr(env, SHORT_GOAL_TARGET_ATTR, None)
 305 |     if target_pos_w is None or target_pos_w.shape != (env.num_envs, 3):
 306 |         target_pos_w = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
 307 |         setattr(env, SHORT_GOAL_TARGET_ATTR, target_pos_w)
 308 |     return target_pos_w
 309 | 
 310 | 
 311 | def _ensure_short_goal_buffers(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
 312 |     target_pos_w = _ensure_short_goal_target(env)
 313 |     prev_goal_distance = getattr(env, SHORT_GOAL_PREV_DISTANCE_ATTR, None)
 314 |     goal_reached = getattr(env, SHORT_GOAL_REACHED_ATTR, None)
 315 |     needs_init = (
 316 |         prev_goal_distance is None
 317 |         or goal_reached is None
 318 |         or prev_goal_distance.shape != (env.num_envs,)
 319 |         or goal_reached.shape != (env.num_envs,)
 320 |     )
 321 |     if needs_init:
 322 |         prev_goal_distance = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
 323 |         goal_reached = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
 324 |         setattr(env, SHORT_GOAL_PREV_DISTANCE_ATTR, prev_goal_distance)
 325 |         setattr(env, SHORT_GOAL_REACHED_ATTR, goal_reached)
 326 |     return target_pos_w, prev_goal_distance, goal_reached
 327 | 
 328 | 
 329 | def _ensure_yaw_rate_command_buffer(env: ManagerBasedEnv) -> torch.Tensor:
 330 |     command = getattr(env, YAW_RATE_COMMAND_ATTR, None)
 331 |     if command is None or command.shape != (env.num_envs,):
 332 |         command = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
 333 |         setattr(env, YAW_RATE_COMMAND_ATTR, command)
 334 |     return command
 335 | 
 336 | 
 337 | def reset_yaw_rate_command(
 338 |     env: ManagerBasedEnv,
 339 |     env_ids,
 340 |     min_abs: float = 0.08,
 341 |     max_abs: float = 0.15,
 342 |     zero_rate: float = 0.0,
 343 |     sign_mode: str = "balanced",
 344 | ) -> None:
 345 |     """Sample a per-episode constant yaw-rate command."""
 346 | 
 347 |     env_ids = _as_env_ids(env, env_ids)
 348 |     if env_ids.numel() == 0:
 349 |         return
 350 | 
 351 |     min_abs = _env_float("RANGER_YAW_CMD_MIN_ABS", min_abs)
 352 |     max_abs = _env_float("RANGER_YAW_CMD_MAX_ABS", max_abs)
 353 |     zero_rate = _env_float("RANGER_YAW_CMD_ZERO_RATE", zero_rate)
 354 |     sign_mode = os.getenv("RANGER_YAW_CMD_SIGN_MODE", sign_mode).strip().lower()
 355 |     if sign_mode not in {"balanced", "positive", "negative"}:
 356 |         sign_mode = "balanced"
 357 |     if max_abs < min_abs:
 358 |         min_abs, max_abs = max_abs, min_abs
 359 |     zero_rate = min(max(float(zero_rate), 0.0), 1.0)
 360 | 
 361 |     command = _ensure_yaw_rate_command_buffer(env)
 362 |     num = env_ids.numel()
 363 |     abs_cmd = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(float(min_abs), float(max_abs))
 364 |     if sign_mode == "positive":
 365 |         sign = torch.ones(num, device=env.device, dtype=torch.float32)
 366 |     elif sign_mode == "negative":
 367 |         sign = -torch.ones(num, device=env.device, dtype=torch.float32)
 368 |     else:
 369 |         sign = torch.where(
 370 |             torch.rand(num, device=env.device) < 0.5,
 371 |             -torch.ones(num, device=env.device, dtype=torch.float32),
 372 |             torch.ones(num, device=env.device, dtype=torch.float32),
 373 |         )
 374 |     cmd = sign * abs_cmd
 375 |     if zero_rate > 0.0:
 376 |         zero_mask = torch.rand(num, device=env.device) < float(zero_rate)
 377 |         cmd = torch.where(zero_mask, torch.zeros_like(cmd), cmd)
 378 |     command[env_ids] = cmd
 379 | 
 380 | 
 381 | def yaw_rate_command(env: ManagerBasedEnv) -> torch.Tensor:
 382 |     """Return per-env constant yaw-rate command."""
 383 | 
 384 |     return _ensure_yaw_rate_command_buffer(env)
 385 | 
 386 | 
 387 | def reset_goal_heading_target(
 388 |     env: ManagerBasedEnv,
 389 |     env_ids,
 390 |     distance_range: tuple[float, float] = (2.0, 5.0),
 391 |     heading_range: tuple[float, float] = (-0.7853981633974483, 0.7853981633974483),
 392 |     heading_bins: tuple[tuple[float, float], ...] | None = None,
 393 |     heading_bin_probs: tuple[float, ...] | None = None,
 394 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 395 | ) -> None:
 396 |     """Sample a per-episode target point in front of the robot's reset heading."""
 397 | 
 398 |     env_ids = _as_env_ids(env, env_ids)
 399 |     if env_ids.numel() == 0:
 400 |         return
 401 | 
 402 |     asset: Articulation = env.scene[asset_cfg.name]
 403 |     target_pos_w = _ensure_goal_heading_target(env)
 404 |     num = env_ids.numel()
 405 |     distance = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 406 |         float(distance_range[0]), float(distance_range[1])
 407 |     )
 408 |     if heading_bins:
 409 |         if heading_bin_probs is None:
 410 |             probabilities = torch.ones(len(heading_bins), device=env.device, dtype=torch.float32)
 411 |         else:
 412 |             if len(heading_bin_probs) != len(heading_bins):
 413 |                 raise ValueError("heading_bin_probs must have the same length as heading_bins.")
 414 |             probabilities = torch.tensor(heading_bin_probs, device=env.device, dtype=torch.float32)
 415 |         probabilities = probabilities / torch.clamp(probabilities.sum(), min=1.0e-6)
 416 |         selector = torch.multinomial(probabilities, num, replacement=True)
 417 |         heading = torch.empty(num, device=env.device, dtype=torch.float32)
 418 |         for bin_id, angle_range in enumerate(heading_bins):
 419 |             mask = selector == bin_id
 420 |             if torch.any(mask):
 421 |                 heading[mask] = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 422 |                     float(angle_range[0]), float(angle_range[1])
 423 |                 )[mask]
 424 |     else:
 425 |         heading = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 426 |             float(heading_range[0]), float(heading_range[1])
 427 |         )
 428 |     target_vec_b = torch.zeros((num, 3), device=env.device, dtype=torch.float32)
 429 |     target_vec_b[:, 0] = distance * torch.cos(heading)
 430 |     target_vec_b[:, 1] = distance * torch.sin(heading)
 431 |     target_vec_w = math_utils.quat_apply_yaw(asset.data.root_quat_w[env_ids], target_vec_b)
 432 |     target_pos_w[env_ids] = asset.data.root_pos_w[env_ids] + target_vec_w
 433 |     prev_heading_error = getattr(env, GOAL_HEADING_PREV_HEADING_ERROR_ATTR, None)
 434 |     if prev_heading_error is not None and prev_heading_error.shape == (env.num_envs,):
 435 |         prev_heading_error[env_ids] = float("nan")
 436 | 
 437 | 
 438 | def reset_short_goal_target(
 439 |     env: ManagerBasedEnv,
 440 |     env_ids,
 441 |     distance_range: tuple[float, float] = (0.5, 2.0),
 442 |     heading_range: tuple[float, float] = (-0.7853981633974483, 0.7853981633974483),
 443 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 444 | ) -> None:
 445 |     """Sample a short-range flat-goal target and initialize per-env progress buffers."""
 446 | 
 447 |     env_ids = _as_env_ids(env, env_ids)
 448 |     if env_ids.numel() == 0:
 449 |         return
 450 | 
 451 |     asset: Articulation = env.scene[asset_cfg.name]
 452 |     target_pos_w, prev_goal_distance, goal_reached = _ensure_short_goal_buffers(env)
 453 |     num = env_ids.numel()
 454 |     distance_min = _env_float("RANGER_GOAL_DISTANCE_MIN", float(distance_range[0]))
 455 |     distance_max = _env_float("RANGER_GOAL_DISTANCE_MAX", float(distance_range[1]))
 456 |     angle_min_deg = _env_float("RANGER_GOAL_ANGLE_MIN_DEG", float(torch.rad2deg(torch.tensor(float(heading_range[0]))).item()))
 457 |     angle_max_deg = _env_float("RANGER_GOAL_ANGLE_MAX_DEG", float(torch.rad2deg(torch.tensor(float(heading_range[1]))).item()))
 458 |     if distance_max < distance_min:
 459 |         distance_min, distance_max = distance_max, distance_min
 460 |     if angle_max_deg < angle_min_deg:
 461 |         angle_min_deg, angle_max_deg = angle_max_deg, angle_min_deg
 462 |     heading_min = math.radians(angle_min_deg)
 463 |     heading_max = math.radians(angle_max_deg)
 464 |     distance = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 465 |         float(distance_min), float(distance_max)
 466 |     )
 467 |     heading = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 468 |         float(heading_min), float(heading_max)
 469 |     )
 470 |     target_vec_b = torch.zeros((num, 3), device=env.device, dtype=torch.float32)
 471 |     target_vec_b[:, 0] = distance * torch.cos(heading)
 472 |     target_vec_b[:, 1] = distance * torch.sin(heading)
 473 |     target_vec_w = math_utils.quat_apply_yaw(asset.data.root_quat_w[env_ids], target_vec_b)
 474 |     target_pos_w[env_ids] = asset.data.root_pos_w[env_ids] + target_vec_w
 475 |     prev_goal_distance[env_ids] = distance
 476 |     goal_reached[env_ids] = False
 477 |     prev_heading_error = getattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, None)
 478 |     if prev_heading_error is not None and prev_heading_error.shape == (env.num_envs,):
 479 |         prev_heading_error[env_ids] = float("nan")
 480 | 
 481 | 
 482 | def reset_short_goal_turn_target(
 483 |     env: ManagerBasedEnv,
 484 |     env_ids,
 485 |     distance_range: tuple[float, float] = (1.0, 1.5),
 486 |     left_heading_range_deg: tuple[float, float] = (25.0, 45.0),
 487 |     right_heading_range_deg: tuple[float, float] = (-45.0, -25.0),
 488 |     paired_sides: bool = False,
 489 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 490 | ) -> None:
 491 |     """Sample a side-only short-goal target for differential turning practice."""
 492 | 
 493 |     env_ids = _as_env_ids(env, env_ids)
 494 |     if env_ids.numel() == 0:
 495 |         return
 496 | 
 497 |     asset: Articulation = env.scene[asset_cfg.name]
 498 |     target_pos_w, prev_goal_distance, goal_reached = _ensure_short_goal_buffers(env)
 499 |     num = env_ids.numel()
 500 | 
 501 |     distance = torch.empty(num, device=env.device, dtype=torch.float32).uniform_(
 502 |         float(distance_range[0]), float(distance_range[1])
 503 |     )
 504 |     if paired_sides:
 505 |         left_mask = torch.remainder(env_ids, 2) == 0
 506 |     else:
 507 |         left_mask = torch.rand((num,), device=env.device) < 0.5
 508 |     heading = torch.empty(num, device=env.device, dtype=torch.float32)
 509 | 
 510 |     if torch.any(left_mask):
 511 |         heading[left_mask] = torch.empty(int(left_mask.sum().item()), device=env.device, dtype=torch.float32).uniform_(
 512 |             math.radians(float(left_heading_range_deg[0])),
 513 |             math.radians(float(left_heading_range_deg[1])),
 514 |         )
 515 |     right_mask = ~left_mask
 516 |     if torch.any(right_mask):
 517 |         heading[right_mask] = torch.empty(
 518 |             int(right_mask.sum().item()), device=env.device, dtype=torch.float32
 519 |         ).uniform_(
 520 |             math.radians(float(right_heading_range_deg[0])),
 521 |             math.radians(float(right_heading_range_deg[1])),
 522 |         )
 523 | 
 524 |     target_vec_b = torch.zeros((num, 3), device=env.device, dtype=torch.float32)
 525 |     target_vec_b[:, 0] = distance * torch.cos(heading)
 526 |     target_vec_b[:, 1] = distance * torch.sin(heading)
 527 |     target_vec_w = math_utils.quat_apply_yaw(asset.data.root_quat_w[env_ids], target_vec_b)
 528 |     target_pos_w[env_ids] = asset.data.root_pos_w[env_ids] + target_vec_w
 529 |     prev_goal_distance[env_ids] = distance
 530 |     goal_reached[env_ids] = False
 531 | 
 532 |     prev_heading_error = getattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, None)
 533 |     if prev_heading_error is None or prev_heading_error.shape != (env.num_envs,):
 534 |         prev_heading_error = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
 535 |         setattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, prev_heading_error)
 536 |     prev_heading_error[env_ids] = heading
 537 | 
 538 | 
 539 | def _sample_weighted_uniform_bands(
 540 |     num: int,
 541 |     bands: tuple[tuple[float, float], ...],
 542 |     weights: tuple[float, ...],
 543 |     device: str,
 544 | ) -> torch.Tensor:
 545 |     """Sample uniformly inside one of several weighted scalar bands."""
 546 | 
 547 |     if not bands:
 548 |         raise ValueError("At least one sampling band is required.")
 549 |     if len(weights) != len(bands):
 550 |         raise ValueError(f"Expected {len(bands)} band weights, got {len(weights)}.")
 551 |     bounds = torch.tensor(bands, device=device, dtype=torch.float32)
 552 |     if torch.any(bounds[:, 1] <= bounds[:, 0]):
 553 |         raise ValueError(f"Each sampling band must satisfy high > low, got {bands}.")
 554 |     probabilities = torch.tensor(weights, device=device, dtype=torch.float32)
 555 |     if torch.any(probabilities < 0.0) or float(probabilities.sum().item()) <= 0.0:
 556 |         raise ValueError(f"Band weights must be non-negative with positive sum, got {weights}.")
 557 |     probabilities = probabilities / probabilities.sum()
 558 |     band_ids = torch.multinomial(probabilities, num_samples=num, replacement=True)
 559 |     selected = bounds[band_ids]
 560 |     return selected[:, 0] + torch.rand((num,), device=device) * (selected[:, 1] - selected[:, 0])
 561 | 
 562 | 
 563 | def reset_short_goal_stratified_target(
 564 |     env: ManagerBasedEnv,
 565 |     env_ids,
 566 |     distance_bands: tuple[tuple[float, float], ...] = ((5.0, 8.0),),
 567 |     distance_weights: tuple[float, ...] = (1.0,),
 568 |     heading_bands_deg: tuple[tuple[float, float], ...] = ((-45.0, -25.0), (25.0, 45.0)),
 569 |     heading_weights: tuple[float, ...] = (0.5, 0.5),
 570 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 571 | ) -> None:
 572 |     """Sample short-goal distance and heading from weighted curriculum bands."""
 573 | 
 574 |     env_ids = _as_env_ids(env, env_ids)
 575 |     if env_ids.numel() == 0:
 576 |         return
 577 | 
 578 |     asset: Articulation = env.scene[asset_cfg.name]
 579 |     target_pos_w, prev_goal_distance, goal_reached = _ensure_short_goal_buffers(env)
 580 |     num = env_ids.numel()
 581 |     distance = _sample_weighted_uniform_bands(num, distance_bands, distance_weights, env.device)
 582 |     heading_deg = _sample_weighted_uniform_bands(num, heading_bands_deg, heading_weights, env.device)
 583 |     heading = torch.deg2rad(heading_deg)
 584 | 
 585 |     target_vec_b = torch.zeros((num, 3), device=env.device, dtype=torch.float32)
 586 |     target_vec_b[:, 0] = distance * torch.cos(heading)
 587 |     target_vec_b[:, 1] = distance * torch.sin(heading)
 588 |     target_vec_w = math_utils.quat_apply_yaw(asset.data.root_quat_w[env_ids], target_vec_b)
 589 |     target_pos_w[env_ids] = asset.data.root_pos_w[env_ids] + target_vec_w
 590 |     prev_goal_distance[env_ids] = distance
 591 |     goal_reached[env_ids] = False
 592 | 
 593 |     prev_heading_error = getattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, None)
 594 |     if prev_heading_error is None or prev_heading_error.shape != (env.num_envs,):
 595 |         prev_heading_error = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
 596 |         setattr(env, SHORT_GOAL_PREV_HEADING_ERROR_ATTR, prev_heading_error)
 597 |     prev_heading_error[env_ids] = heading
 598 | 
 599 | 
 600 | def goal_heading_target_pos_w(env: ManagerBasedEnv) -> torch.Tensor:
 601 |     """Return sampled goal-heading target positions in world frame."""
 602 | 
 603 |     return _ensure_goal_heading_target(env)
 604 | 
 605 | 
 606 | def goal_heading_target_body(
 607 |     env: ManagerBasedEnv,
 608 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 609 | ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
 610 |     """Return target vector, distance, and heading error in the robot body frame."""
 611 | 
 612 |     asset: Articulation = env.scene[asset_cfg.name]
 613 |     target_pos_w = _ensure_goal_heading_target(env)
 614 |     target_vec_w = target_pos_w - asset.data.root_pos_w
 615 |     target_vec_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, target_vec_w)
 616 |     target_xy_b = target_vec_b[:, :2]
 617 |     distance = torch.norm(target_xy_b, dim=1)
 618 |     heading_error = torch.atan2(target_xy_b[:, 1], target_xy_b[:, 0])
 619 |     return target_vec_b, distance, heading_error
 620 | 
 621 | 
 622 | def short_goal_target_pos_w(env: ManagerBasedEnv) -> torch.Tensor:
 623 |     """Return sampled short-goal target positions in world frame."""
 624 | 
 625 |     return _ensure_short_goal_target(env)
 626 | 
 627 | 
 628 | def short_goal_target_body(
 629 |     env: ManagerBasedEnv,
 630 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 631 | ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
 632 |     """Return short-goal vector, distance, and heading error in the robot body frame."""
 633 | 
 634 |     asset: Articulation = env.scene[asset_cfg.name]
 635 |     target_pos_w = _ensure_short_goal_target(env)
 636 |     target_vec_w = target_pos_w - asset.data.root_pos_w
 637 |     target_vec_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, target_vec_w)
 638 |     target_xy_b = target_vec_b[:, :2]
 639 |     distance = torch.norm(target_xy_b, dim=1)
 640 |     heading_error = torch.atan2(target_xy_b[:, 1], target_xy_b[:, 0])
 641 |     return target_vec_b, distance, heading_error
 642 | 
 643 | 
 644 | def goal_heading_state(
 645 |     env: ManagerBasedEnv,
 646 |     goal_range: float = 5.0,
 647 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 648 | ) -> torch.Tensor:
 649 |     """Return six-dimensional target-heading descriptor in the body frame."""
 650 | 
 651 |     if goal_range <= 0.0:
 652 |         raise ValueError(f"goal_range must be positive. Received: {goal_range}")
 653 | 
 654 |     target_vec_b, distance, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
 655 |     obs = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
 656 |     obs[:, 0] = torch.clamp(target_vec_b[:, 0] / float(goal_range), min=-1.0, max=1.0)
 657 |     obs[:, 1] = torch.clamp(target_vec_b[:, 1] / float(goal_range), min=-1.0, max=1.0)
 658 |     obs[:, 2] = torch.clamp(distance / float(goal_range), min=0.0, max=1.0)
 659 |     obs[:, 3] = torch.sin(heading_error)
 660 |     obs[:, 4] = torch.cos(heading_error)
 661 |     obs[:, 5] = 1.0
 662 |     return obs
 663 | 
 664 | 
 665 | def turn_to_target_goal_state(
 666 |     env: ManagerBasedEnv,
 667 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 668 | ) -> torch.Tensor:
 669 |     """Return the raw six-dimensional turn-to-target descriptor in the body frame."""
 670 | 
 671 |     target_vec_b, distance, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
 672 |     obs = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
 673 |     obs[:, 0] = target_vec_b[:, 0]
 674 |     obs[:, 1] = target_vec_b[:, 1]
 675 |     obs[:, 2] = distance
 676 |     obs[:, 3] = torch.sin(heading_error)
 677 |     obs[:, 4] = torch.cos(heading_error)
 678 |     obs[:, 5] = 1.0
 679 |     return obs
 680 | 
 681 | 
 682 | def base_lin_vel_normalized(
 683 |     env: ManagerBasedEnv,
 684 |     scale: float = 2.0,
 685 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 686 | ) -> torch.Tensor:
 687 |     """Return normalized base linear velocity in the body frame."""
 688 | 
 689 |     _obs_debug("enter critic_privileged/base_lin_vel")
 690 |     if _critic_privileged_disabled():
 691 |         out = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
 692 |         _obs_debug(f"exit critic_privileged/base_lin_vel disabled shape={tuple(out.shape)}")
 693 |         return out
 694 |     asset: Articulation = env.scene[asset_cfg.name]
 695 |     out = torch.clamp(asset.data.root_lin_vel_b / scale, min=-1.0, max=1.0)
 696 |     _obs_debug(f"exit critic_privileged/base_lin_vel shape={tuple(out.shape)}")
 697 |     return out
 698 | 
 699 | 
 700 | def base_ang_vel_normalized(
 701 |     env: ManagerBasedEnv,
 702 |     scale: float = 3.0,
 703 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 704 | ) -> torch.Tensor:
 705 |     """Return normalized base angular velocity in the body frame."""
 706 | 
 707 |     _obs_debug("enter policy_state/base_ang_vel")
 708 |     if _policy_state_disabled():
 709 |         out = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
 710 |         _obs_debug(f"exit policy_state/base_ang_vel disabled shape={tuple(out.shape)}")
 711 |         return out
 712 |     asset: Articulation = env.scene[asset_cfg.name]
 713 |     out = torch.clamp(asset.data.root_ang_vel_b / scale, min=-1.0, max=1.0)
 714 |     _obs_debug(f"exit policy_state/base_ang_vel shape={tuple(out.shape)}")
 715 |     return out
 716 | 
 717 | 
 718 | def projected_gravity_normalized(
 719 |     env: ManagerBasedEnv,
 720 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 721 | ) -> torch.Tensor:
 722 |     """Return clipped projected gravity vector."""
 723 | 
 724 |     _obs_debug("enter policy_state/projected_gravity")
 725 |     if _policy_state_disabled():
 726 |         out = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
 727 |         _obs_debug(f"exit policy_state/projected_gravity disabled shape={tuple(out.shape)}")
 728 |         return out
 729 |     asset: Articulation = env.scene[asset_cfg.name]
 730 |     out = torch.clamp(asset.data.projected_gravity_b, min=-1.0, max=1.0)
 731 |     _obs_debug(f"exit policy_state/projected_gravity shape={tuple(out.shape)}")
 732 |     return out
 733 | 
 734 | 
 735 | def joint_pos_rel_normalized(
 736 |     env: ManagerBasedEnv,
 737 |     scale: float = 1.0,
 738 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 739 | ) -> torch.Tensor:
 740 |     """Return normalized relative joint positions for the selected joints."""
 741 | 
 742 |     _obs_debug(f"enter policy_state/joint_pos_rel joint_count={len(asset_cfg.joint_ids)}")
 743 |     if _policy_state_disabled():
 744 |         out = torch.zeros((env.num_envs, len(asset_cfg.joint_ids)), device=env.device, dtype=torch.float32)
 745 |         _obs_debug(f"exit policy_state/joint_pos_rel disabled shape={tuple(out.shape)}")
 746 |         return out
 747 |     asset: Articulation = env.scene[asset_cfg.name]
 748 |     joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
 749 |     out = torch.clamp(joint_pos_rel / scale, min=-1.0, max=1.0)
 750 |     _obs_debug(f"exit policy_state/joint_pos_rel shape={tuple(out.shape)}")
 751 |     return out
 752 | 
 753 | 
 754 | def joint_vel_rel_normalized(
 755 |     env: ManagerBasedEnv,
 756 |     scale: float = 5.0,
 757 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 758 | ) -> torch.Tensor:
 759 |     """Return normalized relative joint velocities for the selected joints."""
 760 | 
 761 |     _obs_debug(f"enter policy_state/joint_vel_rel joint_count={len(asset_cfg.joint_ids)}")
 762 |     if _policy_state_disabled():
 763 |         out = torch.zeros((env.num_envs, len(asset_cfg.joint_ids)), device=env.device, dtype=torch.float32)
 764 |         _obs_debug(f"exit policy_state/joint_vel_rel disabled shape={tuple(out.shape)}")
 765 |         return out
 766 |     asset: Articulation = env.scene[asset_cfg.name]
 767 |     joint_vel_rel = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
 768 |     out = torch.clamp(joint_vel_rel / scale, min=-1.0, max=1.0)
 769 |     _obs_debug(f"exit policy_state/joint_vel_rel shape={tuple(out.shape)}")
 770 |     return out
 771 | 
 772 | 
 773 | def hydraulic_stroke_state(env: ManagerBasedEnv, action_name: str = "leg_hydraulic") -> torch.Tensor:
 774 |     """Return the current equivalent hydraulic-cylinder stroke state."""
 775 | 
 776 |     action_term = env.action_manager.get_term(action_name)
 777 |     if not hasattr(action_term, "stroke_actual"):
 778 |         raise AttributeError(f"Action term '{action_name}' does not expose 'stroke_actual'.")
 779 |     return action_term.stroke_actual
 780 | 
 781 | 
 782 | def suspension_stroke_state(env: ManagerBasedEnv, action_name: str = "leg_hydraulic") -> torch.Tensor:
 783 |     """Return the measured suspension/EHA stroke in normalized physical stroke units."""
 784 | 
 785 |     _obs_debug("enter policy_state/suspension_stroke")
 786 |     if _policy_state_disabled():
 787 |         out = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
 788 |         _obs_debug(f"exit policy_state/suspension_stroke disabled shape={tuple(out.shape)}")
 789 |         return out
 790 |     out = hydraulic_stroke_state(env=env, action_name=action_name)
 791 |     _obs_debug(f"exit policy_state/suspension_stroke shape={tuple(out.shape)}")
 792 |     return out
 793 | 
 794 | 
 795 | def suspension_stroke_rate_state(
 796 |     env: ManagerBasedEnv,
 797 |     action_name: str = "leg_hydraulic",
 798 |     clip: float | None = 5.0,
 799 | ) -> torch.Tensor:
 800 |     """Return per-step suspension stroke rate using a per-env cached finite difference."""
 801 | 
 802 |     _obs_debug("enter policy_state/suspension_stroke_rate")
 803 |     if _policy_state_disabled():
 804 |         out = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
 805 |         _obs_debug(f"exit policy_state/suspension_stroke_rate disabled shape={tuple(out.shape)}")
 806 |         return out
 807 |     stroke = suspension_stroke_state(env=env, action_name=action_name)
 808 |     prev_stroke = getattr(env, SUSPENSION_STROKE_PREV_ATTR, None)
 809 |     stroke_rate = getattr(env, SUSPENSION_STROKE_RATE_ATTR, None)
 810 |     last_step = getattr(env, SUSPENSION_STROKE_RATE_STEP_ATTR, None)
 811 | 
 812 |     if (
 813 |         prev_stroke is None
 814 |         or stroke_rate is None
 815 |         or prev_stroke.shape != stroke.shape
 816 |         or stroke_rate.shape != stroke.shape
 817 |         or last_step is None
 818 |     ):
 819 |         prev_stroke = stroke.clone()
 820 |         stroke_rate = torch.zeros_like(stroke)
 821 |         setattr(env, SUSPENSION_STROKE_PREV_ATTR, prev_stroke)
 822 |         setattr(env, SUSPENSION_STROKE_RATE_ATTR, stroke_rate)
 823 |         setattr(env, SUSPENSION_STROKE_RATE_STEP_ATTR, int(env.common_step_counter))
 824 |         _obs_debug(f"exit policy_state/suspension_stroke_rate init shape={tuple(stroke_rate.shape)}")
 825 |         return stroke_rate
 826 | 
 827 |     current_step = int(env.common_step_counter)
 828 |     if int(last_step) != current_step:
 829 |         dt = max(float(getattr(env, "step_dt", 1.0 / 60.0)), 1.0e-6)
 830 |         stroke_rate[:] = (stroke - prev_stroke) / dt
 831 |         reset_mask = env.episode_length_buf == 0
 832 |         if torch.any(reset_mask):
 833 |             stroke_rate[reset_mask] = 0.0
 834 |         if clip is not None:
 835 |             stroke_rate[:] = torch.clamp(stroke_rate, min=-float(clip), max=float(clip))
 836 |         prev_stroke[:] = stroke
 837 |         setattr(env, SUSPENSION_STROKE_RATE_STEP_ATTR, current_step)
 838 |     else:
 839 |         reset_mask = env.episode_length_buf == 0
 840 |         if torch.any(reset_mask):
 841 |             stroke_rate[reset_mask] = 0.0
 842 |             prev_stroke[reset_mask] = stroke[reset_mask]
 843 | 
 844 |     _obs_debug(f"exit policy_state/suspension_stroke_rate shape={tuple(stroke_rate.shape)}")
 845 |     return stroke_rate
 846 | 
 847 | 
 848 | def hydraulic_stroke_state_normalized(
 849 |     env: ManagerBasedEnv,
 850 |     action_name: str = "leg_hydraulic",
 851 |     stroke_min: float = 0.0,
 852 |     stroke_max: float = 1.0,
 853 | ) -> torch.Tensor:
 854 |     """Return normalized equivalent hydraulic-cylinder stroke state."""
 855 | 
 856 |     stroke_actual = hydraulic_stroke_state(env=env, action_name=action_name)
 857 |     normalized = (stroke_actual - stroke_min) / max(stroke_max - stroke_min, 1.0e-6)
 858 |     return torch.clamp(normalized, min=0.0, max=1.0)
 859 | 
 860 | 
 861 | def hydraulic_effort_state_normalized(
 862 |     env: ManagerBasedEnv,
 863 |     action_name: str = "leg_hydraulic",
 864 |     effort_limit: float = 300.0,
 865 | ) -> torch.Tensor:
 866 |     """Return normalized equivalent hydraulic effort state."""
 867 | 
 868 |     action_term = env.action_manager.get_term(action_name)
 869 |     if not hasattr(action_term, "effort_actual"):
 870 |         raise AttributeError(f"Action term '{action_name}' does not expose 'effort_actual'.")
 871 |     return torch.clamp(action_term.effort_actual / max(effort_limit, 1.0e-6), min=-1.0, max=1.0)
 872 | 
 873 | 
 874 | def wheel_velocity_target_normalized(
 875 |     env: ManagerBasedEnv,
 876 |     action_name: str = "wheel_motor_csv",
 877 |     velocity_limit: float = 20.0,
 878 | ) -> torch.Tensor:
 879 |     """Return normalized wheel CSV target velocity state."""
 880 | 
 881 |     action_term = env.action_manager.get_term(action_name)
 882 |     if not hasattr(action_term, "velocity_target"):
 883 |         raise AttributeError(f"Action term '{action_name}' does not expose 'velocity_target'.")
 884 |     return torch.clamp(action_term.velocity_target / max(velocity_limit, 1.0e-6), min=-1.0, max=1.0)
 885 | 
 886 | 
 887 | def wheel_torque_state_normalized(
 888 |     env: ManagerBasedEnv,
 889 |     action_name: str = "wheel_motor_csv",
 890 |     effort_limit: float = 100.0,
 891 | ) -> torch.Tensor:
 892 |     """Return normalized wheel torque state from the local CSV loop."""
 893 | 
 894 |     action_term = env.action_manager.get_term(action_name)
 895 |     if not hasattr(action_term, "torque_actual"):
 896 |         raise AttributeError(f"Action term '{action_name}' does not expose 'torque_actual'.")
 897 |     return torch.clamp(action_term.torque_actual / max(effort_limit, 1.0e-6), min=-1.0, max=1.0)
 898 | 
 899 | 
 900 | def goal_state(
 901 |     env: ManagerBasedEnv,
 902 |     goal_x_body: float = 0.0,
 903 |     goal_y_body: float = 0.0,
 904 |     goal_range: float = 5.0,
 905 |     goal_enabled: bool = False,
 906 | ) -> torch.Tensor:
 907 |     """Return a fixed six-dimensional goal descriptor in the body frame.
 908 | 
 909 |     This keeps the policy interface stable for later staged training. When
 910 |     ``goal_enabled`` is False, the term returns a neutral placeholder:
 911 |     ``[0, 0, 0, 0, 1, 0]``.
 912 |     """
 913 | 
 914 |     if goal_range <= 0.0:
 915 |         raise ValueError(f"goal_range must be positive. Received: {goal_range}")
 916 | 
 917 |     goal_obs = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
 918 |     if not goal_enabled:
 919 |         goal_obs[:, 4] = 1.0
 920 |         return goal_obs
 921 | 
 922 |     goal_x_body_tensor = torch.full((env.num_envs,), float(goal_x_body), device=env.device)
 923 |     goal_y_body_tensor = torch.full((env.num_envs,), float(goal_y_body), device=env.device)
 924 |     distance = torch.sqrt(goal_x_body_tensor.square() + goal_y_body_tensor.square())
 925 |     bearing = torch.atan2(goal_y_body_tensor, goal_x_body_tensor)
 926 | 
 927 |     goal_obs[:, 0] = torch.clamp(goal_x_body_tensor / goal_range, min=-1.0, max=1.0)
 928 |     goal_obs[:, 1] = torch.clamp(goal_y_body_tensor / goal_range, min=-1.0, max=1.0)
 929 |     goal_obs[:, 2] = torch.clamp(distance / goal_range, min=0.0, max=1.0)
 930 |     goal_obs[:, 3] = torch.sin(bearing)
 931 |     goal_obs[:, 4] = torch.cos(bearing)
 932 |     goal_obs[:, 5] = 1.0
 933 |     return goal_obs
 934 | 
 935 | 
 936 | def command_observation(
 937 |     env: ManagerBasedEnv,
 938 |     command_mode: str = "zero",
 939 |     goal_source: str = "none",
 940 |     stage: str = "A",
 941 |     v_x_range: tuple[float, float] = (0.05, 0.4),
 942 |     yaw_rate_range: tuple[float, float] = (-0.4, 0.4),
 943 |     command_duration_range: tuple[float, float] = (2.0, 5.0),
 944 |     smoothing_alpha: float = 0.1,
 945 |     max_command_duration: float = 5.0,
 946 |     small_yaw_prob: float = 0.0,
 947 |     small_yaw_range: tuple[float, float] = (0.03, 0.08),
 948 |     v_x_bins: tuple[tuple[float, float], ...] | None = None,
 949 |     goal_x_body: float = 0.0,
 950 |     goal_y_body: float = 0.0,
 951 |     short_goal_encoding_mode: str = "legacy",
 952 |     short_goal_observation_max_distance: float | None = None,
 953 |     short_goal_near_distance_range: float = 3.0,
 954 |     short_goal_global_distance_unit: float = 1.0,
 955 |     short_goal_velocity_reference: float = 1.5,
 956 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
 957 | ) -> torch.Tensor:
 958 |     """Return the fixed eight-dimensional command/goal observation for actor state."""
 959 | 
 960 |     _obs_debug("enter policy_state/command_state")
 961 |     if _policy_state_disabled():
 962 |         out = torch.zeros((env.num_envs, 8), device=env.device, dtype=torch.float32)
 963 |         _obs_debug(f"exit policy_state/command_state disabled shape={tuple(out.shape)}")
 964 |         return out
 965 | 
 966 |     del max_command_duration
 967 | 
 968 |     current_step = int(env.common_step_counter)
 969 |     cached_obs = getattr(env, COMMAND_OBS_CACHE_ATTR, None)
 970 |     cached_step = getattr(env, COMMAND_OBS_CACHE_STEP_ATTR, None)
 971 |     if (
 972 |         cached_obs is not None
 973 |         and cached_obs.shape == (env.num_envs, 8)
 974 |         and cached_step == current_step
 975 |         and not torch.any(env.episode_length_buf == 0)
 976 |     ):
 977 |         _obs_debug(f"exit policy_state/command_state cached shape={tuple(cached_obs.shape)}")
 978 |         return cached_obs
 979 | 
 980 |     obs = torch.zeros((env.num_envs, 8), device=env.device, dtype=torch.float32)
 981 | 
 982 |     command_mode = command_mode.lower()
 983 |     if command_mode == "speed_command":
 984 |         command = update_speed_command(
 985 |             env=env,
 986 |             stage=stage,
 987 |             v_x_range=v_x_range,
 988 |             yaw_rate_range=yaw_rate_range,
 989 |             command_duration_range=command_duration_range,
 990 |             smoothing_alpha=smoothing_alpha,
 991 |             small_yaw_prob=small_yaw_prob,
 992 |             small_yaw_range=small_yaw_range,
 993 |             v_x_bins=v_x_bins,
 994 |         )
 995 |         obs[:, 0] = command[:, 0]
 996 |         obs[:, 1] = 0.0
 997 |         obs[:, 2] = command[:, 1]
 998 |     elif command_mode == "yaw_rate_command":
 999 |         command = yaw_rate_command(env)
1000 |         obs[:, 0] = 0.0
1001 |         obs[:, 1] = 0.0
1002 |         obs[:, 2] = command
1003 |     elif command_mode != "zero":
1004 |         raise ValueError(f"Unsupported command_mode: {command_mode}")
1005 | 
1006 |     goal_source = goal_source.lower()
1007 |     if goal_source == "dynamic":
1008 |         target_vec_b, distance, heading_error = goal_heading_target_body(env, asset_cfg=asset_cfg)
1009 |         obs[:, 3] = target_vec_b[:, 0]
1010 |         obs[:, 4] = target_vec_b[:, 1]
1011 |         obs[:, 5] = distance
1012 |         obs[:, 6] = torch.sin(heading_error)
1013 |         obs[:, 7] = torch.cos(heading_error)
1014 |     elif goal_source == "short_goal":
1015 |         target_vec_b, distance, heading_error = short_goal_target_body(env, asset_cfg=asset_cfg)
1016 |         stop_phase_active = getattr(env, "_short_goal_stop_phase_active", None)
1017 |         if stop_phase_active is not None and stop_phase_active.shape == (env.num_envs,):
1018 |             # ShortGoal does not use the three command slots; reuse slot 2 for the
1019 |             # latched stop-phase flag without changing the observation dimension.
1020 |             obs[:, 2] = stop_phase_active.to(obs.dtype)
1021 | 
1022 |         encoding_mode = short_goal_encoding_mode.lower()
1023 |         if encoding_mode == "long_term":
1024 |             distance_safe = torch.clamp(distance, min=1.0e-6)
1025 |             goal_direction = target_vec_b[:, :2] / distance_safe.unsqueeze(1)
1026 |             near_range = max(float(short_goal_near_distance_range), 1.0e-6)
1027 |             global_distance_unit = max(float(short_goal_global_distance_unit), 1.0e-6)
1028 |             velocity_reference = max(float(short_goal_velocity_reference), 1.0e-6)
1029 |             near_distance = torch.clamp(distance / near_range, min=0.0, max=1.0)
1030 |             global_log_distance = torch.log2(1.0 + distance / global_distance_unit)
1031 | 
1032 |             asset: Articulation = env.scene[asset_cfg.name]
1033 |             velocity_toward_goal = torch.sum(asset.data.root_lin_vel_b[:, :2] * goal_direction, dim=1)
1034 |             velocity_toward_goal = torch.clamp(
1035 |                 velocity_toward_goal / velocity_reference,
1036 |                 min=-1.0,
1037 |                 max=1.0,
1038 |             )
1039 | 
1040 |             # Fixed long-term encoding:
1041 |             # [goal_dir_x, goal_dir_y, near_distance, log2(1 + distance / unit), velocity_toward_goal].
1042 |             obs[:, 3] = goal_direction[:, 0]
1043 |             obs[:, 4] = goal_direction[:, 1]
1044 |             obs[:, 5] = near_distance
1045 |             obs[:, 6] = global_log_distance
1046 |             obs[:, 7] = velocity_toward_goal
1047 |         elif encoding_mode == "legacy":
1048 |             target_vec_obs = target_vec_b[:, :2]
1049 |             distance_obs = distance
1050 |             if short_goal_observation_max_distance is not None:
1051 |                 max_distance = max(float(short_goal_observation_max_distance), 1.0e-6)
1052 |                 radial_scale = torch.clamp(max_distance / torch.clamp(distance, min=1.0e-6), max=1.0)
1053 |                 target_vec_obs = target_vec_obs * radial_scale.unsqueeze(1)
1054 |                 distance_obs = torch.clamp(distance, max=max_distance)
1055 | 
1056 |             obs[:, 3] = target_vec_obs[:, 0]
1057 |             obs[:, 4] = target_vec_obs[:, 1]
1058 |             obs[:, 5] = distance_obs
1059 |             obs[:, 6] = torch.sin(heading_error)
1060 |             obs[:, 7] = torch.cos(heading_error)
1061 |         else:
1062 |             raise ValueError(f"Unsupported short_goal_encoding_mode: {short_goal_encoding_mode}")
1063 |     elif goal_source == "fixed":
1064 |         goal_x_body_tensor = torch.full((env.num_envs,), float(goal_x_body), device=env.device)
1065 |         goal_y_body_tensor = torch.full((env.num_envs,), float(goal_y_body), device=env.device)
1066 |         distance = torch.sqrt(goal_x_body_tensor.square() + goal_y_body_tensor.square())
1067 |         heading_error = torch.atan2(goal_y_body_tensor, goal_x_body_tensor)
1068 |         obs[:, 3] = goal_x_body_tensor
1069 |         obs[:, 4] = goal_y_body_tensor
1070 |         obs[:, 5] = distance
1071 |         obs[:, 6] = torch.sin(heading_error)
1072 |         obs[:, 7] = torch.cos(heading_error)
1073 |     elif goal_source == "none":
1074 |         obs[:, 7] = 1.0
1075 |     else:
1076 |         raise ValueError(f"Unsupported goal_source: {goal_source}")
1077 | 
1078 |     setattr(env, COMMAND_OBS_CACHE_ATTR, obs)
1079 |     setattr(env, COMMAND_OBS_CACHE_STEP_ATTR, current_step)
1080 |     _obs_debug(f"exit policy_state/command_state shape={tuple(obs.shape)}")
1081 |     return obs
1082 | 
1083 | 
1084 | def last_action_normalized(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
1085 |     """Return clipped previous action history."""
1086 | 
1087 |     _obs_debug("enter policy_state/previous_action")
1088 |     if _policy_state_disabled():
1089 |         out = torch.zeros((env.num_envs, 8), device=env.device, dtype=torch.float32)
1090 |         _obs_debug(f"exit policy_state/previous_action disabled shape={tuple(out.shape)}")
1091 |         return out
1092 |     if action_name is None:
1093 |         out = torch.clamp(env.action_manager.action, min=-1.0, max=1.0)
1094 |     else:
1095 |         out = torch.clamp(env.action_manager.get_term(action_name).raw_actions, min=-1.0, max=1.0)
1096 |     _obs_debug(f"exit policy_state/previous_action shape={tuple(out.shape)}")
1097 |     return out
1098 | 
1099 | 
1100 | def _wheel_contact_sensor_body_ids(env: ManagerBasedEnv, sensor_name: str = "wheel_contact_forces") -> torch.Tensor:
1101 |     cached_body_ids = getattr(env, WHEEL_CONTACT_SENSOR_BODY_IDS_ATTR, None)
1102 |     if cached_body_ids is not None and cached_body_ids.numel() == 4:
1103 |         return cached_body_ids
1104 | 
1105 |     contact_sensor = env.scene.sensors[sensor_name]
1106 |     body_ids, _ = contact_sensor.find_bodies(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
1107 |     cached_body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
1108 |     setattr(env, WHEEL_CONTACT_SENSOR_BODY_IDS_ATTR, cached_body_ids)
1109 |     return cached_body_ids
1110 | 
1111 | 
1112 | def wheel_contact_force_over_weight(
1113 |     env: ManagerBasedEnv,
1114 |     sensor_name: str = "wheel_contact_forces",
1115 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
1116 | ) -> torch.Tensor:
1117 |     """Return wheel-contact force norm per wheel normalized by total robot weight."""
1118 | 
1119 |     _obs_debug("enter critic_privileged/wheel_contact_force")
1120 |     if _critic_privileged_disabled():
1121 |         out = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
1122 |         _obs_debug(f"exit critic_privileged/wheel_contact_force disabled shape={tuple(out.shape)}")
1123 |         return out
1124 |     contact_sensor = env.scene.sensors[sensor_name]
1125 |     body_ids = _wheel_contact_sensor_body_ids(env, sensor_name=sensor_name)
1126 |     net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, body_ids, :]
1127 |     contact_force = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
1128 | 
1129 |     asset: Articulation = env.scene[asset_cfg.name]
1130 |     expected_weight = torch.sum(asset.root_physx_view.get_masses(), dim=1).to(env.device) * 9.81
1131 |     out = contact_force / torch.clamp(expected_weight.unsqueeze(1), min=1.0e-6)
1132 |     _obs_debug(f"exit critic_privileged/wheel_contact_force shape={tuple(out.shape)}")
1133 |     return out
1134 | 
1135 | 
1136 | def wheel_contact_bool(
1137 |     env: ManagerBasedEnv,
1138 |     sensor_name: str = "wheel_contact_forces",
1139 |     threshold: float = 1.0,
1140 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
1141 | ) -> torch.Tensor:
1142 |     """Return binary wheel contact state in semantic ``[lr, lf, rf, rr]`` sensor order."""
1143 | 
1144 |     _obs_debug("enter critic_privileged/wheel_contact_bool")
1145 |     if _critic_privileged_disabled():
1146 |         out = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
1147 |         _obs_debug(f"exit critic_privileged/wheel_contact_bool disabled shape={tuple(out.shape)}")
1148 |         return out
1149 |     del asset_cfg
1150 |     contact_sensor = env.scene.sensors[sensor_name]
1151 |     body_ids = _wheel_contact_sensor_body_ids(env, sensor_name=sensor_name)
1152 |     net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, body_ids, :]
1153 |     force_norm = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
1154 |     out = (force_norm > float(threshold)).to(torch.float32)
1155 |     _obs_debug(f"exit critic_privileged/wheel_contact_bool shape={tuple(out.shape)}")
1156 |     return out
1157 | 
1158 | 
1159 | def root_height_state(
1160 |     env: ManagerBasedEnv,
1161 |     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
1162 | ) -> torch.Tensor:
1163 |     """Return root height used by the environment in world z coordinates."""
1164 | 
1165 |     _obs_debug("enter critic_privileged/root_height")
1166 |     if _critic_privileged_disabled():
1167 |         out = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
1168 |         _obs_debug(f"exit critic_privileged/root_height disabled shape={tuple(out.shape)}")
1169 |         return out
1170 |     asset: Articulation = env.scene[asset_cfg.name]
1171 |     out = asset.data.root_pos_w[:, 2:3]
1172 |     _obs_debug(f"exit critic_privileged/root_height shape={tuple(out.shape)}")
1173 |     return out
1174 | 
1175 | 
1176 | def local_sensor_visibility_maps(
1177 |     env: ManagerBasedEnv,
1178 |     sensor_names: tuple[str, ...],
1179 |     asset_name: str = "robot",
1180 |     x_range: tuple[float, float] = (0.0, 2.0),
1181 |     y_range: tuple[float, float] = (-0.6, 0.6),
1182 |     resolution: float = 0.1,
1183 | ) -> dict[str, torch.Tensor]:
1184 |     """Return one local valid-mask grid per sensor for visibility debugging."""
1185 | 
1186 |     visibility_maps = {}
1187 |     for sensor_name in sensor_names:
1188 |         _, valid_mask = _build_local_height_map(
1189 |             env=env,
1190 |             sensor_names=(sensor_name,),
1191 |             asset_name=asset_name,
1192 |             x_range=x_range,
1193 |             y_range=y_range,
1194 |             resolution=resolution,
1195 |         )
1196 |         visibility_maps[sensor_name] = valid_mask.to(torch.float32)
1197 |     return visibility_maps
1198 | 
1199 | 
1200 | def local_navigation_map_layers(
1201 |     env: ManagerBasedEnv,
1202 |     sensor_names: tuple[str, ...],
1203 |     asset_name: str = "robot",
1204 |     x_range: tuple[float, float] = (0.0, 2.0),
1205 |     y_range: tuple[float, float] = (-0.6, 0.6),
1206 |     resolution: float = 0.1,
1207 |     step_threshold: float = 0.08,
1208 |     height_reference_x_range: tuple[float, float] = (0.0, 0.4),
1209 |     height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
1210 |     slope_normalization: float = 0.6,
1211 |     roughness_normalization: float = 0.05,
1212 |     step_normalization: float = 0.15,
1213 |     apply_noise: bool = False,
1214 |     height_noise_std: float = 0.0,
1215 |     risk_noise_std: float = 0.0,
1216 |     valid_dropout_prob: float = 0.0,
1217 |     slope_weight: float = 0.4,
1218 |     roughness_weight: float = 0.3,
1219 |     step_weight: float = 0.3,
1220 |     unknown_penalty: float = 1.0,
1221 |     use_neutral_map: bool = False,
1222 | ) -> dict[str, torch.Tensor]:
1223 |     """Return the unflattened local navigation layers for planning-aware policies."""
1224 | 
1225 |     _obs_debug("enter policy_map/local_navigation_map_layers")
1226 |     if _policy_map_disabled():
1227 |         num_x, num_y = _navigation_map_grid_shape(x_range=x_range, y_range=y_range, resolution=resolution)
1228 |         zeros = torch.zeros((env.num_envs, num_x, num_y), device=env.device, dtype=torch.float32)
1229 |         out = {
1230 |             "height": zeros,
1231 |             "slope": zeros,
1232 |             "roughness": zeros,
1233 |             "step": zeros,
1234 |             "geometric_traversability": zeros,
1235 |             "valid_mask": zeros,
1236 |             "semantic_traversability": zeros,
1237 |             "confidence": zeros,
1238 |             "traversability": zeros,
1239 |         }
1240 |         _obs_debug(
1241 |             "exit policy_map/local_navigation_map_layers disabled "
1242 |             f"grid_shape={tuple(out['height'].shape)}"
1243 |         )
1244 |         return out
1245 | 
1246 |     if use_neutral_map:
1247 |         out = _build_neutral_navigation_map_layers(
1248 |             env=env,
1249 |             x_range=x_range,
1250 |             y_range=y_range,
1251 |             resolution=resolution,
1252 |         )
1253 |         _obs_debug(
1254 |             "exit policy_map/local_navigation_map_layers neutral "
1255 |             f"grid_shape={tuple(out['height'].shape)}"
1256 |         )
1257 |         return out
1258 | 
1259 |     height_map_raw, valid_mask = _build_local_height_map(
1260 |         env=env,
1261 |         sensor_names=sensor_names,
1262 |         asset_name=asset_name,
1263 |         x_range=x_range,
1264 |         y_range=y_range,
1265 |         resolution=resolution,
1266 |     )
1267 |     height_map = _normalize_height_map(
1268 |         height_map_raw=height_map_raw,
1269 |         valid_mask=valid_mask,
1270 |         x_range=x_range,
1271 |         y_range=y_range,
1272 |         resolution=resolution,
1273 |         reference_x_range=height_reference_x_range,
1274 |         reference_y_range=height_reference_y_range,
1275 |     )
1276 |     slope_map = _normalize_risk_map(
1277 |         _compute_slope_map(height_map, valid_mask, resolution),
1278 |         valid_mask,
1279 |         normalization=slope_normalization,
1280 |     )
1281 |     roughness_map = _normalize_risk_map(
1282 |         _compute_roughness_map(height_map, valid_mask),
1283 |         valid_mask,
1284 |         normalization=roughness_normalization,
1285 |     )
1286 |     step_map = _normalize_risk_map(
1287 |         _compute_step_map(height_map, valid_mask, step_threshold),
1288 |         valid_mask,
1289 |         normalization=step_normalization,
1290 |     )
1291 |     height_map, slope_map, roughness_map, step_map, valid_mask = _apply_local_map_noise(
1292 |         height_map=height_map,
1293 |         slope_map=slope_map,
1294 |         roughness_map=roughness_map,
1295 |         step_map=step_map,
1296 |         valid_mask=valid_mask,
1297 |         apply_noise=apply_noise,
1298 |         height_noise_std=height_noise_std,
1299 |         risk_noise_std=risk_noise_std,
1300 |         valid_dropout_prob=valid_dropout_prob,
1301 |     )
1302 |     _obs_debug("enter policy_map/geometric_traversability")
1303 |     geometric_traversability = _compute_traversability_map(
1304 |         slope_map=slope_map,
1305 |         roughness_map=roughness_map,
1306 |         step_map=step_map,
1307 |         valid_mask=valid_mask,
1308 |         slope_weight=slope_weight,
1309 |         roughness_weight=roughness_weight,
1310 |         step_weight=step_weight,
1311 |         unknown_penalty=unknown_penalty,
1312 |     )
1313 |     _obs_debug(f"exit policy_map/geometric_traversability shape={tuple(geometric_traversability.shape)}")
1314 |     semantic_traversability = torch.zeros_like(geometric_traversability)
1315 |     confidence = valid_mask.to(height_map.dtype)
1316 |     _obs_debug(f"policy_map/valid_mask shape={tuple(valid_mask.shape)}")
1317 |     _obs_debug(f"policy_map/confidence shape={tuple(confidence.shape)}")
1318 | 
1319 |     out = {
1320 |         "height": height_map,
1321 |         "slope": slope_map,
1322 |         "roughness": roughness_map,
1323 |         "step": step_map,
1324 |         "geometric_traversability": geometric_traversability,
1325 |         "valid_mask": valid_mask.to(height_map.dtype),
1326 |         "semantic_traversability": semantic_traversability,
1327 |         "confidence": confidence,
1328 |         # Backward-compatible alias for existing debug utilities that still reference the old name.
1329 |         "traversability": geometric_traversability,
1330 |     }
1331 |     _obs_debug(f"exit policy_map/local_navigation_map_layers grid_shape={tuple(height_map.shape)}")
1332 |     return out
1333 | 
1334 | 
1335 | def local_navigation_map(
1336 |     env: ManagerBasedEnv,
1337 |     sensor_names: tuple[str, ...],
1338 |     asset_name: str = "robot",
1339 |     x_range: tuple[float, float] = (0.0, 2.0),
1340 |     y_range: tuple[float, float] = (-0.6, 0.6),
1341 |     resolution: float = 0.1,
1342 |     step_threshold: float = 0.08,
1343 |     height_reference_x_range: tuple[float, float] = (0.0, 0.4),
1344 |     height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
1345 |     slope_normalization: float = 0.6,
1346 |     roughness_normalization: float = 0.05,
1347 |     step_normalization: float = 0.15,
1348 |     apply_noise: bool = False,
1349 |     height_noise_std: float = 0.0,
1350 |     risk_noise_std: float = 0.0,
1351 |     valid_dropout_prob: float = 0.0,
1352 |     slope_weight: float = 0.4,
1353 |     roughness_weight: float = 0.3,
1354 |     step_weight: float = 0.3,
1355 |     unknown_penalty: float = 1.0,
1356 |     use_neutral_map: bool = False,
1357 | ) -> torch.Tensor:
1358 |     """Build the fixed eight-channel local navigation map used by the V1 terrain-CNN policy."""
1359 | 
1360 |     _obs_debug("enter policy_map")
1361 |     if _policy_map_disabled():
1362 |         out = torch.zeros((env.num_envs, 2184), device=env.device, dtype=torch.float32)
1363 |         _obs_debug(f"exit policy_map disabled shape={tuple(out.shape)}")
1364 |         return out
1365 |     layers_dict = local_navigation_map_layers(
1366 |         env=env,
1367 |         sensor_names=sensor_names,
1368 |         asset_name=asset_name,
1369 |         x_range=x_range,
1370 |         y_range=y_range,
1371 |         resolution=resolution,
1372 |         step_threshold=step_threshold,
1373 |         height_reference_x_range=height_reference_x_range,
1374 |         height_reference_y_range=height_reference_y_range,
1375 |         slope_normalization=slope_normalization,
1376 |         roughness_normalization=roughness_normalization,
1377 |         step_normalization=step_normalization,
1378 |         apply_noise=apply_noise,
1379 |         height_noise_std=height_noise_std,
1380 |         risk_noise_std=risk_noise_std,
1381 |         valid_dropout_prob=valid_dropout_prob,
1382 |         slope_weight=slope_weight,
1383 |         roughness_weight=roughness_weight,
1384 |         step_weight=step_weight,
1385 |         unknown_penalty=unknown_penalty,
1386 |         use_neutral_map=use_neutral_map,
1387 |     )
1388 |     layers = torch.stack(
1389 |         (
1390 |             layers_dict["height"],
1391 |             layers_dict["slope"],
1392 |             layers_dict["roughness"],
1393 |             layers_dict["step"],
1394 |             layers_dict["geometric_traversability"],
1395 |             layers_dict["valid_mask"],
1396 |             layers_dict["semantic_traversability"],
1397 |             layers_dict["confidence"],
1398 |         ),
1399 |         dim=1,
1400 |     )
1401 |     out = layers.reshape(env.num_envs, -1)
1402 |     _obs_debug(f"exit policy_map shape={tuple(out.shape)}")
1403 |     return out
1404 | 
1405 | 
1406 | def local_geometric_map_layers(
1407 |     env: ManagerBasedEnv,
1408 |     sensor_names: tuple[str, ...],
1409 |     asset_name: str = "robot",
1410 |     x_range: tuple[float, float] = (0.0, 2.0),
1411 |     y_range: tuple[float, float] = (-0.6, 0.6),
1412 |     resolution: float = 0.1,
1413 |     step_threshold: float = 0.08,
1414 |     height_reference_x_range: tuple[float, float] = (0.0, 0.4),
1415 |     height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
1416 |     slope_normalization: float = 0.6,
1417 |     roughness_normalization: float = 0.05,
1418 |     step_normalization: float = 0.15,
1419 |     apply_noise: bool = False,
1420 |     height_noise_std: float = 0.0,
1421 |     risk_noise_std: float = 0.0,
1422 |     valid_dropout_prob: float = 0.0,
1423 | ) -> dict[str, torch.Tensor]:
1424 |     """Backward-compatible wrapper that returns the original five-layer map."""
1425 | 
1426 |     layers_dict = local_navigation_map_layers(
1427 |         env=env,
1428 |         sensor_names=sensor_names,
1429 |         asset_name=asset_name,
1430 |         x_range=x_range,
1431 |         y_range=y_range,
1432 |         resolution=resolution,
1433 |         step_threshold=step_threshold,
1434 |         height_reference_x_range=height_reference_x_range,
1435 |         height_reference_y_range=height_reference_y_range,
1436 |         slope_normalization=slope_normalization,
1437 |         roughness_normalization=roughness_normalization,
1438 |         step_normalization=step_normalization,
1439 |         apply_noise=apply_noise,
1440 |         height_noise_std=height_noise_std,
1441 |         risk_noise_std=risk_noise_std,
1442 |         valid_dropout_prob=valid_dropout_prob,
1443 |     )
1444 |     return {
1445 |         "height": layers_dict["height"],
1446 |         "slope": layers_dict["slope"],
1447 |         "roughness": layers_dict["roughness"],
1448 |         "step": layers_dict["step"],
1449 |         "valid_mask": layers_dict["valid_mask"],
1450 |     }
1451 | 
1452 | 
1453 | def local_geometric_map(
1454 |     env: ManagerBasedEnv,
1455 |     sensor_names: tuple[str, ...],
1456 |     asset_name: str = "robot",
1457 |     x_range: tuple[float, float] = (0.0, 2.0),
1458 |     y_range: tuple[float, float] = (-0.6, 0.6),
1459 |     resolution: float = 0.1,
1460 |     step_threshold: float = 0.08,
1461 |     height_reference_x_range: tuple[float, float] = (0.0, 0.4),
1462 |     height_reference_y_range: tuple[float, float] = (-0.3, 0.3),
1463 |     slope_normalization: float = 0.6,
1464 |     roughness_normalization: float = 0.05,
1465 |     step_normalization: float = 0.15,
1466 |     apply_noise: bool = False,
1467 |     height_noise_std: float = 0.0,
1468 |     risk_noise_std: float = 0.0,
1469 |     valid_dropout_prob: float = 0.0,
1470 | ) -> torch.Tensor:
1471 |     """Backward-compatible wrapper that returns the original five-layer map."""
1472 | 
1473 |     layers_dict = local_geometric_map_layers(
1474 |         env=env,
1475 |         sensor_names=sensor_names,
1476 |         asset_name=asset_name,
1477 |         x_range=x_range,
1478 |         y_range=y_range,
1479 |         resolution=resolution,
1480 |         step_threshold=step_threshold,
1481 |         height_reference_x_range=height_reference_x_range,
1482 |         height_reference_y_range=height_reference_y_range,
1483 |         slope_normalization=slope_normalization,
1484 |         roughness_normalization=roughness_normalization,
1485 |         step_normalization=step_normalization,
1486 |         apply_noise=apply_noise,
1487 |         height_noise_std=height_noise_std,
1488 |         risk_noise_std=risk_noise_std,
1489 |         valid_dropout_prob=valid_dropout_prob,
1490 |     )
1491 |     layers = torch.stack(
1492 |         (
1493 |             layers_dict["height"],
1494 |             layers_dict["slope"],
1495 |             layers_dict["roughness"],
1496 |             layers_dict["step"],
1497 |             layers_dict["valid_mask"],
1498 |         ),
1499 |         dim=1,
1500 |     )
1501 |     return layers.reshape(env.num_envs, -1)
1502 | 
1503 | 
1504 | def _build_local_height_map(
1505 |     env: ManagerBasedEnv,
1506 |     sensor_names: tuple[str, ...],
1507 |     asset_name: str,
1508 |     x_range: tuple[float, float],
1509 |     y_range: tuple[float, float],
1510 |     resolution: float,
1511 |     invalid_height: float = 0.0,
1512 | ) -> tuple[torch.Tensor, torch.Tensor]:
1513 |     asset: Articulation = env.scene[asset_name]
1514 |     ray_hits_w = []
1515 |     for sensor_name in sensor_names:
1516 |         sensor: RayCaster = env.scene.sensors[sensor_name]
1517 |         if hasattr(sensor.data, "ray_hits_w"):
1518 |             hits_w = sensor.data.ray_hits_w
1519 |         else:
1520 |             # RayCasterCamera stores world-space hits on the sensor object and may
1521 |             # update a subset of environments internally, so refresh all envs here.
1522 |             sensor._update_buffers_impl(slice(None))
1523 |             hits_w = sensor.ray_hits_w
1524 |         ray_hits_w.append(hits_w)
1525 |     points_w = torch.cat(ray_hits_w, dim=1)
1526 | 
1527 |     points_rel_w = points_w - asset.data.root_pos_w.unsqueeze(1)
1528 |     num_rays = points_rel_w.shape[1]
1529 |     # Use a gravity-aligned local frame: keep yaw, remove roll and pitch.
1530 |     root_yaw_quat_w = math_utils.yaw_quat(asset.data.root_quat_w)
1531 |     points_b = math_utils.quat_apply_inverse(
1532 |         root_yaw_quat_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4),
1533 |         points_rel_w.reshape(-1, 3),
1534 |     ).reshape(env.num_envs, num_rays, 3)
1535 | 
1536 |     x_min, x_max = x_range
1537 |     y_min, y_max = y_range
1538 |     num_x = int(round((x_max - x_min) / resolution)) + 1
1539 |     num_y = int(round((y_max - y_min) / resolution)) + 1
1540 |     num_cells = num_x * num_y
1541 | 
1542 |     x = points_b[..., 0]
1543 |     y = points_b[..., 1]
1544 |     z = points_b[..., 2]
1545 |     valid_points = (
1546 |         torch.isfinite(points_b).all(dim=-1)
1547 |         & (x >= x_min)
1548 |         & (x <= x_max)
1549 |         & (y >= y_min)
1550 |         & (y <= y_max)
1551 |     )
1552 | 
1553 |     ix = torch.round((x - x_min) / resolution).long().clamp(0, num_x - 1)
1554 |     iy = torch.round((y - y_min) / resolution).long().clamp(0, num_y - 1)
1555 |     local_index = ix * num_y + iy
1556 | 
1557 |     env_offsets = torch.arange(env.num_envs, device=points_b.device).unsqueeze(1) * num_cells
1558 |     flat_index = (local_index + env_offsets).reshape(-1)
1559 |     flat_valid = valid_points.reshape(-1)
1560 |     flat_z = z.reshape(-1)
1561 | 
1562 |     flat_height = torch.full((env.num_envs * num_cells,), -torch.inf, device=points_b.device)
1563 |     if flat_valid.any():
1564 |         flat_height.scatter_reduce_(
1565 |             0,
1566 |             flat_index[flat_valid],
1567 |             flat_z[flat_valid],
1568 |             reduce="amax",
1569 |             include_self=True,
1570 |         )
1571 | 
1572 |     valid_mask = torch.isfinite(flat_height).reshape(env.num_envs, num_x, num_y)
1573 |     height_map = flat_height.reshape(env.num_envs, num_x, num_y)
1574 |     height_map = torch.where(valid_mask, height_map, torch.full_like(height_map, invalid_height))
1575 |     return height_map, valid_mask
1576 | 
1577 | 
1578 | def _compute_slope_map(height_map: torch.Tensor, valid_mask: torch.Tensor, resolution: float) -> torch.Tensor:
1579 |     slope_x = torch.zeros_like(height_map)
1580 |     valid_x = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]
1581 |     diff_x = torch.abs(height_map[:, 1:, :] - height_map[:, :-1, :]) / resolution
1582 |     slope_x[:, 1:, :] = torch.where(valid_x, diff_x, torch.zeros_like(diff_x))
1583 | 
1584 |     slope_y = torch.zeros_like(height_map)
1585 |     valid_y = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]
1586 |     diff_y = torch.abs(height_map[:, :, 1:] - height_map[:, :, :-1]) / resolution
1587 |     slope_y[:, :, 1:] = torch.where(valid_y, diff_y, torch.zeros_like(diff_y))
1588 |     return torch.maximum(slope_x, slope_y)
1589 | 
1590 | 
1591 | def _normalize_height_map(
1592 |     height_map_raw: torch.Tensor,
1593 |     valid_mask: torch.Tensor,
1594 |     x_range: tuple[float, float],
1595 |     y_range: tuple[float, float],
1596 |     resolution: float,
1597 |     reference_x_range: tuple[float, float],
1598 |     reference_y_range: tuple[float, float],
1599 | ) -> torch.Tensor:
1600 |     """Shift height so locally flat support terrain stays near zero."""
1601 | 
1602 |     reference_height = _compute_height_reference(
1603 |         height_map_raw=height_map_raw,
1604 |         valid_mask=valid_mask,
1605 |         x_range=x_range,
1606 |         y_range=y_range,
1607 |         resolution=resolution,
1608 |         reference_x_range=reference_x_range,
1609 |         reference_y_range=reference_y_range,
1610 |     )
1611 |     height_map = height_map_raw - reference_height[:, None, None]
1612 |     return torch.where(valid_mask, height_map, torch.zeros_like(height_map))
1613 | 
1614 | 
1615 | def _compute_height_reference(
1616 |     height_map_raw: torch.Tensor,
1617 |     valid_mask: torch.Tensor,
1618 |     x_range: tuple[float, float],
1619 |     y_range: tuple[float, float],
1620 |     resolution: float,
1621 |     reference_x_range: tuple[float, float],
1622 |     reference_y_range: tuple[float, float],
1623 | ) -> torch.Tensor:
1624 |     """Estimate the local ground reference height from a small anchor region near the robot."""
1625 | 
1626 |     device = height_map_raw.device
1627 |     num_x = height_map_raw.shape[1]
1628 |     num_y = height_map_raw.shape[2]
1629 |     x_coords = torch.linspace(x_range[0], x_range[1], num_x, device=device)
1630 |     y_coords = torch.linspace(y_range[0], y_range[1], num_y, device=device)
1631 |     x_mask = (x_coords >= reference_x_range[0]) & (x_coords <= reference_x_range[1])
1632 |     y_mask = (y_coords >= reference_y_range[0]) & (y_coords <= reference_y_range[1])
1633 |     reference_region_mask = x_mask[:, None] & y_mask[None, :]
1634 | 
1635 |     fallback_region_mask = (x_coords >= x_range[0]) & (x_coords <= min(x_range[1], reference_x_range[1] + resolution))
1636 |     fallback_region_mask = fallback_region_mask[:, None] & torch.ones((1, num_y), dtype=torch.bool, device=device)
1637 | 
1638 |     reference_heights = []
1639 |     for env_id in range(height_map_raw.shape[0]):
1640 |         region_valid = valid_mask[env_id] & reference_region_mask
1641 |         if region_valid.any():
1642 |             reference_heights.append(height_map_raw[env_id][region_valid].median())
1643 |             continue
1644 | 
1645 |         fallback_valid = valid_mask[env_id] & fallback_region_mask
1646 |         if fallback_valid.any():
1647 |             reference_heights.append(height_map_raw[env_id][fallback_valid].median())
1648 |             continue
1649 | 
1650 |         all_valid = valid_mask[env_id]
1651 |         if all_valid.any():
1652 |             reference_heights.append(height_map_raw[env_id][all_valid].median())
1653 |         else:
1654 |             reference_heights.append(torch.tensor(0.0, device=device, dtype=height_map_raw.dtype))
1655 | 
1656 |     return torch.stack(reference_heights, dim=0)
1657 | 
1658 | 
1659 | def _normalize_risk_map(risk_map: torch.Tensor, valid_mask: torch.Tensor, normalization: float) -> torch.Tensor:
1660 |     """Clamp a raw geometric risk layer into a 0..1 intensity map."""
1661 | 
1662 |     if normalization <= 0.0:
1663 |         raise ValueError(f"Risk normalization must be positive. Received: {normalization}")
1664 |     normalized = torch.clamp(risk_map / normalization, min=0.0, max=1.0)
1665 |     return torch.where(valid_mask, normalized, torch.zeros_like(normalized))
1666 | 
1667 | 
1668 | def _build_neutral_navigation_map_layers(
1669 |     env: ManagerBasedEnv,
1670 |     x_range: tuple[float, float],
1671 |     y_range: tuple[float, float],
1672 |     resolution: float,
1673 | ) -> dict[str, torch.Tensor]:
1674 |     """Create a stage-1 neutral navigation map with fully traversable support."""
1675 | 
1676 |     num_x, num_y = _navigation_map_grid_shape(x_range=x_range, y_range=y_range, resolution=resolution)
1677 |     zeros = torch.zeros((env.num_envs, num_x, num_y), device=env.device, dtype=torch.float32)
1678 |     ones = torch.ones_like(zeros)
1679 |     return {
1680 |         "height": zeros,
1681 |         "slope": zeros,
1682 |         "roughness": zeros,
1683 |         "step": zeros,
1684 |         "geometric_traversability": ones,
1685 |         "valid_mask": ones,
1686 |         "semantic_traversability": zeros,
1687 |         "confidence": ones,
1688 |         "traversability": ones,
1689 |     }
1690 | 
1691 | 
1692 | def _compute_traversability_map(
1693 |     slope_map: torch.Tensor,
1694 |     roughness_map: torch.Tensor,
1695 |     step_map: torch.Tensor,
1696 |     valid_mask: torch.Tensor,
1697 |     slope_weight: float,
1698 |     roughness_weight: float,
1699 |     step_weight: float,
1700 |     unknown_penalty: float,
1701 | ) -> torch.Tensor:
1702 |     """Fuse geometric risk layers into a simple traversability intensity."""
1703 | 
1704 |     _obs_debug("enter policy_map/_compute_traversability_map")
1705 |     valid_mask_float = valid_mask.to(slope_map.dtype)
1706 |     cost = (
1707 |         slope_weight * slope_map
1708 |         + roughness_weight * roughness_map
1709 |         + step_weight * step_map
1710 |         + unknown_penalty * (1.0 - valid_mask_float)
1711 |     )
1712 |     cost = torch.clamp(cost, min=0.0, max=1.0)
1713 |     traversability = 1.0 - cost
1714 |     out = torch.where(valid_mask, traversability, torch.zeros_like(traversability))
1715 |     _obs_debug(f"exit policy_map/_compute_traversability_map shape={tuple(out.shape)}")
1716 |     return out
1717 | 
1718 | 
1719 | def _navigation_map_grid_shape(
1720 |     x_range: tuple[float, float],
1721 |     y_range: tuple[float, float],
1722 |     resolution: float,
1723 | ) -> tuple[int, int]:
1724 |     """Compute the discrete local map grid shape from metric bounds."""
1725 | 
1726 |     num_x = int(round((x_range[1] - x_range[0]) / resolution)) + 1
1727 |     num_y = int(round((y_range[1] - y_range[0]) / resolution)) + 1
1728 |     return num_x, num_y
1729 | 
1730 | 
1731 | def _apply_local_map_noise(
1732 |     height_map: torch.Tensor,
1733 |     slope_map: torch.Tensor,
1734 |     roughness_map: torch.Tensor,
1735 |     step_map: torch.Tensor,
1736 |     valid_mask: torch.Tensor,
1737 |     apply_noise: bool,
1738 |     height_noise_std: float,
1739 |     risk_noise_std: float,
1740 |     valid_dropout_prob: float,
1741 | ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
1742 |     """Apply lightweight noise and random visibility dropout to the local map."""
1743 | 
1744 |     if not apply_noise:
1745 |         return height_map, slope_map, roughness_map, step_map, valid_mask
1746 | 
1747 |     valid_mask_noisy = valid_mask
1748 |     if valid_dropout_prob > 0.0:
1749 |         dropout = torch.rand_like(valid_mask.to(torch.float32)) < valid_dropout_prob
1750 |         valid_mask_noisy = valid_mask & (~dropout)
1751 | 
1752 |     if height_noise_std > 0.0:
1753 |         height_map = height_map + torch.randn_like(height_map) * height_noise_std
1754 | 
1755 |     if risk_noise_std > 0.0:
1756 |         slope_map = slope_map + torch.randn_like(slope_map) * risk_noise_std
1757 |         roughness_map = roughness_map + torch.randn_like(roughness_map) * risk_noise_std
1758 |         step_map = step_map + torch.randn_like(step_map) * risk_noise_std
1759 | 
1760 |     slope_map = torch.clamp(slope_map, min=0.0, max=1.0)
1761 |     roughness_map = torch.clamp(roughness_map, min=0.0, max=1.0)
1762 |     step_map = torch.clamp(step_map, min=0.0, max=1.0)
1763 | 
1764 |     height_map = torch.where(valid_mask_noisy, height_map, torch.zeros_like(height_map))
1765 |     slope_map = torch.where(valid_mask_noisy, slope_map, torch.zeros_like(slope_map))
1766 |     roughness_map = torch.where(valid_mask_noisy, roughness_map, torch.zeros_like(roughness_map))
1767 |     step_map = torch.where(valid_mask_noisy, step_map, torch.zeros_like(step_map))
1768 | 
1769 |     return height_map, slope_map, roughness_map, step_map, valid_mask_noisy
1770 | 
1771 | 
1772 | def _compute_roughness_map(height_map: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
1773 |     height = height_map.unsqueeze(1)
1774 |     mask = valid_mask.to(height_map.dtype).unsqueeze(1)
1775 |     count = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) * 9.0
1776 |     height_sum = F.avg_pool2d(height * mask, kernel_size=3, stride=1, padding=1) * 9.0
1777 |     mean = height_sum / torch.clamp(count, min=1.0)
1778 |     var_sum = F.avg_pool2d(((height - mean) ** 2) * mask, kernel_size=3, stride=1, padding=1) * 9.0
1779 |     roughness = torch.sqrt(var_sum / torch.clamp(count, min=1.0))
1780 |     return torch.where(valid_mask, roughness.squeeze(1), torch.zeros_like(height_map))
1781 | 
1782 | 
1783 | def _compute_step_map(height_map: torch.Tensor, valid_mask: torch.Tensor, threshold: float) -> torch.Tensor:
1784 |     step_x = torch.zeros_like(height_map)
1785 |     valid_x = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]
1786 |     diff_x = torch.abs(height_map[:, 1:, :] - height_map[:, :-1, :])
1787 |     step_x[:, 1:, :] = torch.where(valid_x & (diff_x > threshold), diff_x, torch.zeros_like(diff_x))
1788 | 
1789 |     step_y = torch.zeros_like(height_map)
1790 |     valid_y = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]
1791 |     diff_y = torch.abs(height_map[:, :, 1:] - height_map[:, :, :-1])
1792 |     step_y[:, :, 1:] = torch.where(valid_y & (diff_y > threshold), diff_y, torch.zeros_like(diff_y))
1793 |     return torch.maximum(step_x, step_y)
1794 | 
```

### source/Ranger/Ranger/tasks/manager_based/ranger/ranger_env_cfg.py

Bytes: 64335
SHA-256: f7017811ddd80e74f65e28f77f2d5974c193dd6e8b286e9c51eaa9b921f76883
Lines: 1-1604 of 1604

```python
   1 | # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
   2 | # All rights reserved.
   3 | #
   4 | # SPDX-License-Identifier: BSD-3-Clause
   5 | 
   6 | import isaaclab.sim as sim_utils
   7 | from isaaclab.assets import ArticulationCfg, AssetBaseCfg
   8 | from isaaclab.envs import ManagerBasedRLEnvCfg
   9 | from isaaclab.managers import EventTermCfg as EventTerm
  10 | from isaaclab.managers import ObservationGroupCfg as ObsGroup
  11 | from isaaclab.managers import ObservationTermCfg as ObsTerm
  12 | from isaaclab.managers import RewardTermCfg as RewTerm
  13 | from isaaclab.managers import SceneEntityCfg
  14 | from isaaclab.managers import TerminationTermCfg as DoneTerm
  15 | from isaaclab.scene import InteractiveSceneCfg
  16 | from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, RayCasterCameraCfg, patterns
  17 | from isaaclab.utils import configclass
  18 | from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
  19 | 
  20 | from Ranger.assets.ranger import RANGER_CFG
  21 | 
  22 | from . import mdp
  23 | 
  24 | COMMAND_OBS_PARAMS = {
  25 |     "command_mode": "zero",
  26 |     "goal_source": "none",
  27 |     "goal_x_body": 0.0,
  28 |     "goal_y_body": 0.0,
  29 | }
  30 | 
  31 | LOCAL_NAVIGATION_MAP_PARAMS = {
  32 |     "sensor_names": ("mid360_lidar", "avia_lidar", "d435i_camera"),
  33 |     "x_range": (0.0, 2.0),
  34 |     "y_range": (-0.6, 0.6),
  35 |     "resolution": 0.1,
  36 |     "height_reference_x_range": (0.0, 0.4),
  37 |     "height_reference_y_range": (-0.3, 0.3),
  38 |     "step_threshold": 0.08,
  39 |     "slope_normalization": 0.6,
  40 |     "roughness_normalization": 0.05,
  41 |     "step_normalization": 0.15,
  42 |     "apply_noise": True,
  43 |     "height_noise_std": 0.01,
  44 |     "risk_noise_std": 0.02,
  45 |     "valid_dropout_prob": 0.02,
  46 |     "slope_weight": 0.4,
  47 |     "roughness_weight": 0.3,
  48 |     "step_weight": 0.3,
  49 |     "unknown_penalty": 1.0,
  50 |     "use_neutral_map": False,
  51 | }
  52 | 
  53 | 
  54 | def _navigation_map_grid_shape(resolution: float, x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[int, int]:
  55 |     num_x = int(round((x_range[1] - x_range[0]) / resolution)) + 1
  56 |     num_y = int(round((y_range[1] - y_range[0]) / resolution)) + 1
  57 |     return num_x, num_y
  58 | 
  59 | 
  60 | def _expected_policy_state_obs_dim() -> int:
  61 |     return 42
  62 | 
  63 | 
  64 | def _expected_policy_map_obs_dim() -> int:
  65 |     map_layers = 8
  66 |     num_x, num_y = _navigation_map_grid_shape(
  67 |         resolution=LOCAL_NAVIGATION_MAP_PARAMS["resolution"],
  68 |         x_range=LOCAL_NAVIGATION_MAP_PARAMS["x_range"],
  69 |         y_range=LOCAL_NAVIGATION_MAP_PARAMS["y_range"],
  70 |     )
  71 |     return map_layers * num_x * num_y
  72 | 
  73 | 
  74 | def _expected_policy_obs_dim() -> int:
  75 |     return _expected_policy_state_obs_dim() + _expected_policy_map_obs_dim()
  76 | 
  77 | 
  78 | def _expected_action_dim() -> int:
  79 |     return 8
  80 | 
  81 | 
  82 | ##
  83 | # 场景
  84 | ##
  85 | 
  86 | 
  87 | @configclass
  88 | class RangerSceneCfg(InteractiveSceneCfg):
  89 |     """Configuration for a scene with the Ranger robot."""
  90 | 
  91 |     # local ground plane, implemented as a thin static cuboid to avoid remote USD dependencies
  92 |     ground = AssetBaseCfg(
  93 |         prim_path="/World/ground",
  94 |         collision_group=-1,
  95 |         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
  96 |         spawn=sim_utils.MeshCuboidCfg(
  97 |             size=(300.0, 300.0, 0.02),
  98 |             rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
  99 |             collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
 100 |             physics_material=sim_utils.RigidBodyMaterialCfg(
 101 |                 friction_combine_mode="average",
 102 |                 restitution_combine_mode="average",
 103 |                 static_friction=1.0,
 104 |                 dynamic_friction=1.0,
 105 |                 restitution=0.0,
 106 |             ),
 107 |             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
 108 |         ),
 109 |     )
 110 | 
 111 |     # robot
 112 |     robot: ArticulationCfg = RANGER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
 113 | 
 114 |     # sensors
 115 |     mid360_lidar = RayCasterCfg(
 116 |         prim_path="{ENV_REGEX_NS}/Robot/mid360_link",
 117 |         ray_alignment="base",
 118 |         pattern_cfg=patterns.LidarPatternCfg(
 119 |             channels=16,
 120 |             vertical_fov_range=(-7.0, 52.0),
 121 |             horizontal_fov_range=(-180.0, 180.0),
 122 |             horizontal_res=10.0,
 123 |         ),
 124 |         max_distance=40.0,
 125 |         mesh_prim_paths=["/World/ground"],
 126 |         debug_vis=True,
 127 |     )
 128 | 
 129 |     avia_lidar = RayCasterCfg(
 130 |         prim_path="{ENV_REGEX_NS}/Robot/avia_link",
 131 |         ray_alignment="base",
 132 |         pattern_cfg=patterns.LidarPatternCfg(
 133 |             channels=16,
 134 |             vertical_fov_range=(-38.6, 38.6),
 135 |             horizontal_fov_range=(-35.2, 35.2),
 136 |             horizontal_res=2.0,
 137 |         ),
 138 |         max_distance=50.0,
 139 |         mesh_prim_paths=["/World/ground"],
 140 |         debug_vis=True,
 141 |     )
 142 | 
 143 |     d435i_camera = RayCasterCameraCfg(
 144 |         prim_path="{ENV_REGEX_NS}/Robot",
 145 |         offset=RayCasterCameraCfg.OffsetCfg(
 146 |             pos=(0.46259, 0.0, -0.21177),
 147 |             rot=(0.241845, -0.664463, 0.664463, -0.241845),
 148 |             convention="ros",
 149 |         ),
 150 |         pattern_cfg=patterns.PinholeCameraPatternCfg(
 151 |             focal_length=1.93,
 152 |             horizontal_aperture=3.80,
 153 |             width=12,
 154 |             height=10,
 155 |         ),
 156 |         max_distance=10.0,
 157 |         mesh_prim_paths=["/World/ground"],
 158 |         debug_vis=True,
 159 |     )
 160 | 
 161 |     # Training-only privileged sensor for wheel-ground contact monitoring.
 162 |     # It is intentionally kept out of the policy observations to preserve sim-to-real compatibility.
 163 |     wheel_contact_forces = ContactSensorCfg(
 164 |         prim_path="{ENV_REGEX_NS}/Robot/w_.*",
 165 |         update_period=0.0,
 166 |         history_length=1,
 167 |         track_air_time=True,
 168 |         debug_vis=False,
 169 |     )
 170 | 
 171 |     # Debug-only privileged sensor for auditing whether non-wheel collision is carrying load.
 172 |     # This is intentionally excluded from rewards and observations.
 173 |     all_body_contact_forces = ContactSensorCfg(
 174 |         prim_path="{ENV_REGEX_NS}/Robot/.*",
 175 |         update_period=0.0,
 176 |         history_length=1,
 177 |         track_air_time=False,
 178 |         debug_vis=False,
 179 |     )
 180 | 
 181 |     # lights
 182 |     dome_light = AssetBaseCfg(
 183 |         prim_path="/World/DomeLight",
 184 |         spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
 185 |     )
 186 | 
 187 | 
 188 | ##
 189 | # MDP settings
 190 | ##
 191 | 
 192 | 
 193 | @configclass
 194 | class ActionsCfg:
 195 |     """Action specifications for the MDP."""
 196 | 
 197 |     leg_hydraulic = mdp.HydraulicActuatorActionCfg(
 198 |         asset_name="robot",
 199 |         joint_names=["g_lb", "g_lf", "g_rf", "g_rb"],
 200 |         preserve_order=True,
 201 |         stroke_min=0.0,
 202 |         stroke_max=1.0,
 203 |         stroke_rate_limit=1.0,
 204 |         time_constant=0.08,
 205 |         stroke_table=(0.0, 0.5, 1.0),
 206 |         joint_pos_table=(-1.0, 0.0, 1.0),
 207 |         max_effort=300.0,
 208 |         impedance_kp=250.0,
 209 |         impedance_kd=30.0,
 210 |     )
 211 |     wheel_motor_csv = mdp.WheelMotorCSVActionCfg(
 212 |         asset_name="robot",
 213 |         joint_names=["w_lb", "w_lf", "w_rf", "w_rb"],
 214 |         preserve_order=True,
 215 |         control_mode="velocity",
 216 |         velocity_limit=20.0,
 217 |         acceleration_limit=80.0,
 218 |         command_time_constant=0.02,
 219 |         velocity_kp=10.0,
 220 |         velocity_damping=0.2,
 221 |         viscous_friction=0.05,
 222 |         effort_limit=100.0,
 223 |     )
 224 | 
 225 | 
 226 | @configclass
 227 | class ObservationsCfg:
 228 |     """Observation specifications for the MDP."""
 229 | 
 230 |     @configclass
 231 |     class PolicyStateCfg(ObsGroup):
 232 |         """Low-dimensional actor state: proprioception + command."""
 233 | 
 234 |         # observation terms (order preserved)
 235 |         base_ang_vel = ObsTerm(
 236 |             func=mdp.base_ang_vel_normalized,
 237 |             params={"scale": 3.0},
 238 |             noise=Unoise(n_min=-0.04, n_max=0.04),
 239 |         )
 240 |         projected_gravity = ObsTerm(
 241 |             func=mdp.projected_gravity_normalized,
 242 |             noise=Unoise(n_min=-0.02, n_max=0.02),
 243 |         )
 244 |         leg_joint_pos_rel = ObsTerm(
 245 |             func=mdp.joint_pos_rel_normalized,
 246 |             params={"scale": 1.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["g_.*"])},
 247 |             noise=Unoise(n_min=-0.01, n_max=0.01),
 248 |         )
 249 |         leg_joint_vel_rel = ObsTerm(
 250 |             func=mdp.joint_vel_rel_normalized,
 251 |             params={"scale": 5.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["g_.*"])},
 252 |             noise=Unoise(n_min=-0.02, n_max=0.02),
 253 |         )
 254 |         wheel_joint_vel_rel = ObsTerm(
 255 |             func=mdp.joint_vel_rel_normalized,
 256 |             params={"scale": 20.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["w_.*"])},
 257 |             noise=Unoise(n_min=-0.02, n_max=0.02),
 258 |         )
 259 |         suspension_stroke = ObsTerm(
 260 |             func=mdp.suspension_stroke_state,
 261 |             params={"action_name": "leg_hydraulic"},
 262 |             noise=Unoise(n_min=-0.01, n_max=0.01),
 263 |         )
 264 |         suspension_stroke_rate = ObsTerm(
 265 |             func=mdp.suspension_stroke_rate_state,
 266 |             params={"action_name": "leg_hydraulic", "clip": 5.0},
 267 |             noise=Unoise(n_min=-0.01, n_max=0.01),
 268 |         )
 269 |         command_state = ObsTerm(
 270 |             func=mdp.command_observation,
 271 |             params=COMMAND_OBS_PARAMS,
 272 |         )
 273 |         last_action = ObsTerm(
 274 |             func=mdp.last_action_normalized,
 275 |             noise=Unoise(n_min=-0.01, n_max=0.01),
 276 |         )
 277 | 
 278 |         def __post_init__(self) -> None:
 279 |             self.enable_corruption = True
 280 |             self.concatenate_terms = True
 281 | 
 282 |     @configclass
 283 |     class PolicyMapCfg(ObsGroup):
 284 |         """Flattened 8-channel local terrain map for the CNN branch."""
 285 | 
 286 |         local_navigation_map = ObsTerm(
 287 |             func=mdp.local_navigation_map,
 288 |             params=LOCAL_NAVIGATION_MAP_PARAMS,
 289 |         )
 290 | 
 291 |         def __post_init__(self) -> None:
 292 |             self.enable_corruption = False
 293 |             self.concatenate_terms = True
 294 | 
 295 |     @configclass
 296 |     class CriticPrivilegedCfg(ObsGroup):
 297 |         """Compact privileged critic-only features for asymmetric actor-critic."""
 298 | 
 299 |         base_lin_vel = ObsTerm(
 300 |             func=mdp.base_lin_vel_normalized,
 301 |             params={"scale": 2.0},
 302 |         )
 303 |         wheel_contact_force = ObsTerm(
 304 |             func=mdp.wheel_contact_force_over_weight,
 305 |             params={"sensor_name": "wheel_contact_forces", "asset_cfg": SceneEntityCfg("robot")},
 306 |         )
 307 |         wheel_contact_bool = ObsTerm(
 308 |             func=mdp.wheel_contact_bool,
 309 |             params={"sensor_name": "wheel_contact_forces", "threshold": 1.0},
 310 |         )
 311 |         root_height = ObsTerm(
 312 |             func=mdp.root_height_state,
 313 |             params={"asset_cfg": SceneEntityCfg("robot")},
 314 |         )
 315 | 
 316 |         def __post_init__(self) -> None:
 317 |             self.enable_corruption = False
 318 |             self.concatenate_terms = True
 319 | 
 320 |     # observation groups
 321 |     policy_state: PolicyStateCfg = PolicyStateCfg()
 322 |     policy_map: PolicyMapCfg = PolicyMapCfg()
 323 |     critic_privileged: CriticPrivilegedCfg = CriticPrivilegedCfg()
 324 | 
 325 | 
 326 | @configclass
 327 | class EventCfg:
 328 |     """Configuration for events."""
 329 | 
 330 |     # reset
 331 |     reset_base = EventTerm(
 332 |         func=mdp.reset_root_state_uniform,
 333 |         mode="reset",
 334 |         params={"pose_range": {}, "velocity_range": {}},
 335 |     )
 336 | 
 337 |     reset_robot_joints = EventTerm(
 338 |         func=mdp.reset_joints_by_offset,
 339 |         mode="reset",
 340 |         params={
 341 |             "position_range": (-0.05, 0.05),
 342 |             "velocity_range": (-0.05, 0.05),
 343 |         },
 344 |     )
 345 | 
 346 | 
 347 | @configclass
 348 | class RewardsCfg:
 349 |     """Reward terms for the MDP."""
 350 | 
 351 |     alive = RewTerm(func=mdp.is_alive, weight=0.2)
 352 |     forward_progress = RewTerm(
 353 |         func=mdp.forward_velocity_reward,
 354 |         weight=3,
 355 |         params={"speed_scale": 1.0},
 356 |     )
 357 |     upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
 358 |     base_vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
 359 |     base_roll_pitch_rate = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
 360 |     lateral_velocity = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.5)
 361 |     leg_joint_deviation = RewTerm(
 362 |         func=mdp.joint_deviation_l1,
 363 |         weight=-0.02,
 364 |         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["g_.*"])},
 365 |     )
 366 |     wheel_joint_velocity = RewTerm(
 367 |         func=mdp.joint_vel_l2,
 368 |         weight=-0.0002,
 369 |         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["w_.*"])},
 370 |     )
 371 |     leg_joint_velocity = RewTerm(
 372 |         func=mdp.joint_vel_l2,
 373 |         weight=-0.005,
 374 |         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["g_.*"])},
 375 |     )
 376 |     action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
 377 |     action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.001)
 378 |     termination = RewTerm(func=mdp.is_terminated, weight=-10.0)
 379 | 
 380 | 
 381 | @configclass
 382 | class TerminationsCfg:
 383 |     """Termination terms for the MDP."""
 384 | 
 385 |     time_out = DoneTerm(func=mdp.time_out, time_out=True)
 386 |     bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})
 387 |     root_height_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.1})
 388 | 
 389 | 
 390 | ##
 391 | # Environment configuration
 392 | ##
 393 | 
 394 | 
 395 | @configclass
 396 | class RangerEnvCfg(ManagerBasedRLEnvCfg):
 397 |     # Scene settings
 398 |     scene: RangerSceneCfg = RangerSceneCfg(num_envs=4096, env_spacing=4.0)
 399 |     # Basic settings
 400 |     observations: ObservationsCfg = ObservationsCfg()
 401 |     actions: ActionsCfg = ActionsCfg()
 402 |     events: EventCfg = EventCfg()
 403 |     # MDP settings
 404 |     rewards: RewardsCfg = RewardsCfg()
 405 |     terminations: TerminationsCfg = TerminationsCfg()
 406 |     # Optional environment-level suspension controls.
 407 |     # ``fixed_suspension_action`` overrides all four hydraulic actions when set.
 408 |     fixed_suspension_action: float | None = None
 409 |     suspension_action_scale: float = 1.0
 410 |     stop_phase_suspension_action: float | None = None
 411 |     stop_phase_suspension_mode: str = "legacy"
 412 |     stop_phase_wheel_override_enabled: bool = True
 413 | 
 414 |     # Post initialization
 415 |     def __post_init__(self) -> None:
 416 |         """Post initialization."""
 417 |         # general settings
 418 |         self.decimation = 2
 419 |         self.episode_length_s = 5
 420 |         # viewer settings
 421 |         self.viewer.eye = (4.0, -4.0, 3.0)
 422 |         self.viewer.lookat = (0.0, 0.0, 0.5)
 423 |         # simulation settings
 424 |         self.sim.dt = 1 / 120
 425 |         self.sim.render_interval = self.decimation
 426 |         # sensor settings
 427 |         self.scene.mid360_lidar.update_period = self.decimation * self.sim.dt
 428 |         self.scene.avia_lidar.update_period = self.decimation * self.sim.dt
 429 |         self.scene.d435i_camera.update_period = self.decimation * self.sim.dt
 430 |         self.scene.wheel_contact_forces.update_period = self.sim.dt
 431 |         self.scene.all_body_contact_forces.update_period = self.sim.dt
 432 |         self.stand_training_task = False
 433 |         self.enable_reset_settle = False
 434 |         self.reset_settle_steps = 0
 435 |         self.reset_settle_leg_action = -0.34
 436 |         self.enable_initial_stroke_randomization = False
 437 |         self.initial_stroke_noise_range = 0.0
 438 |         self.initial_stroke_range = (0.50, 0.50)
 439 |         self.initial_stroke_shared_across_legs = True
 440 |         self.sync_reset_root_height_to_initial_stroke = False
 441 |         self.initial_stroke_root_height_reference = 0.884
 442 |         self.initial_stroke_root_height_nominal_stroke = 0.50
 443 |         self.initial_stroke_root_height_slope = 0.38
 444 |         self.terrain_like_reset_enabled = False
 445 |         self.terrain_like_reset_base_stroke = 0.50
 446 |         self.terrain_like_reset_common_offset_range = (-0.02, 0.04)
 447 |         self.terrain_like_reset_pattern_amplitude_range = (0.03, 0.06)
 448 |         self.terrain_like_reset_stroke_clamp_range = (0.44, 0.58)
 449 |         self.terrain_like_reset_root_height_margin = 0.015
 450 |         self.terrain_like_reset_slope_deg_range = (3.0, 5.0)
 451 |         self.terrain_like_reset_twist_deg_range = (2.0, 5.0)
 452 |         self.terrain_like_reset_yaw_deg_range = (-3.0, 3.0)
 453 |         self.terrain_like_reset_linear_xy_velocity_range = (-0.10, 0.10)
 454 |         self.terrain_like_reset_angular_velocity_range = (-0.2, 0.2)
 455 |         self.debug_full_stdout_metrics = False
 456 |         print(
 457 |             "[RangerEnvCfg] Expected actor observation shape: "
 458 |             f"state={_expected_policy_state_obs_dim()} map={_expected_policy_map_obs_dim()} total={_expected_policy_obs_dim()}"
 459 |         )
 460 |         print(f"[RangerEnvCfg] Expected action shape: {_expected_action_dim()} (4 leg + 4 wheel)")
 461 | 
 462 | 
 463 | @configclass
 464 | class RangerStandEnvCfg(RangerEnvCfg):
 465 |     """Stage-1 standing configuration focused on stable four-wheel grounding."""
 466 | 
 467 |     def __post_init__(self) -> None:
 468 |         super().__post_init__()
 469 |         self.episode_length_s = 4.0
 470 |         self.stand_training_task = True
 471 |         self.scene.robot.init_state.pos = (0.0, 0.0, 0.884)
 472 |         self.actions.leg_hydraulic.joint_pos_table = (-0.5, 0.0, 0.5)
 473 |         self.actions.leg_hydraulic.joint_target_sign = (-1.0, 1.0, 1.0, -1.0)
 474 | 
 475 |         # Stage-1 uses the fixed neutral goal/map interface while we focus on posture stability.
 476 |         self.observations.policy_state.command_state.params["command_mode"] = "zero"
 477 |         self.observations.policy_state.command_state.params["goal_source"] = "none"
 478 |         self.observations.policy_map.local_navigation_map.params["use_neutral_map"] = True
 479 | 
 480 |         # Keep resets close to the nominal support pose so the policy can first learn to settle.
 481 |         self.events.reset_robot_joints.params["position_range"] = (-0.005, 0.005)
 482 |         self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
 483 |         self.events.reset_base.params["velocity_range"] = {
 484 |             "x": (0.0, 0.0),
 485 |             "y": (0.0, 0.0),
 486 |             "z": (0.0, 0.0),
 487 |             "roll": (0.0, 0.0),
 488 |             "pitch": (0.0, 0.0),
 489 |             "yaw": (0.0, 0.0),
 490 |         }
 491 |         self.events.reset_base.params["pose_range"] = {
 492 |             "x": (0.0, 0.0),
 493 |             "y": (0.0, 0.0),
 494 |             "z": (0.0, 0.0),
 495 |             "roll": (0.0, 0.0),
 496 |             "pitch": (0.0, 0.0),
 497 |             "yaw": (0.0, 0.0),
 498 |         }
 499 |         self.enable_initial_stroke_randomization = True
 500 |         self.initial_stroke_noise_range = 0.0
 501 |         self.initial_stroke_range = (0.47, 0.55)
 502 |         self.initial_stroke_shared_across_legs = True
 503 |         self.sync_reset_root_height_to_initial_stroke = True
 504 |         self.initial_stroke_root_height_reference = 0.884
 505 |         self.initial_stroke_root_height_nominal_stroke = 0.50
 506 |         self.initial_stroke_root_height_slope = 0.38
 507 |         self.terrain_like_reset_enabled = True
 508 |         self.terrain_like_reset_base_stroke = 0.50
 509 |         self.terrain_like_reset_common_offset_range = (-0.02, 0.04)
 510 |         self.terrain_like_reset_pattern_amplitude_range = (0.03, 0.06)
 511 |         self.terrain_like_reset_stroke_clamp_range = (0.44, 0.58)
 512 |         self.terrain_like_reset_root_height_margin = 0.015
 513 |         self.terrain_like_reset_slope_deg_range = (3.0, 5.0)
 514 |         self.terrain_like_reset_twist_deg_range = (2.0, 5.0)
 515 |         self.terrain_like_reset_yaw_deg_range = (-3.0, 3.0)
 516 |         self.terrain_like_reset_linear_xy_velocity_range = (-0.10, 0.10)
 517 |         self.terrain_like_reset_angular_velocity_range = (-0.2, 0.2)
 518 | 
 519 |         # Keep a realistic healthy stand envelope; low-slung postures should not count as success.
 520 |         self.terminations.bad_orientation.params["limit_angle"] = 1.4
 521 |         self.terminations.root_height_low.func = mdp.root_height_below_minimum_with_grace
 522 |         self.terminations.root_height_low.params["minimum_height"] = 0.65
 523 |         self.terminations.root_height_low.params["grace_time_s"] = 0.5
 524 |         self.terminations.root_height_low.params["asset_cfg"] = SceneEntityCfg("robot")
 525 | 
 526 |         # Turn off locomotion incentives and focus on stable support/contact quality first.
 527 |         self.rewards.forward_progress.weight = 0.0
 528 |         self.rewards.lateral_velocity.weight = -0.2
 529 |         self.rewards.base_xy_velocity = RewTerm(
 530 |             func=mdp.base_xy_speed_l2,
 531 |             weight=-0.5,
 532 |             params={"asset_cfg": SceneEntityCfg("robot")},
 533 |         )
 534 |         self.rewards.base_vertical_velocity.weight = -1.0
 535 |         self.rewards.base_roll_pitch_rate.weight = -0.5
 536 |         self.rewards.base_yaw_rate = RewTerm(
 537 |             func=mdp.base_yaw_rate_l2,
 538 |             weight=-0.3,
 539 |             params={"asset_cfg": SceneEntityCfg("robot")},
 540 |         )
 541 |         self.rewards.yaw_drift_from_reset = RewTerm(
 542 |             func=mdp.stand_yaw_drift_abs,
 543 |             weight=-0.2,
 544 |             params={"asset_cfg": SceneEntityCfg("robot")},
 545 |         )
 546 |         self.rewards.upright.weight = -3.0
 547 |         self.rewards.wheel_joint_velocity.weight = -0.002
 548 |         self.rewards.action_rate.weight = -0.04
 549 |         self.rewards.action_magnitude.weight = -0.002
 550 |         self.rewards.roll_angle = RewTerm(func=mdp.roll_angle_l2, weight=-1.0)
 551 |         self.rewards.pitch_angle = RewTerm(func=mdp.pitch_angle_l2, weight=-1.2)
 552 |         self.rewards.root_height_tracking = RewTerm(
 553 |             func=mdp.root_height_tracking_exp,
 554 |             weight=0.5,
 555 |             params={"target_height": 0.875, "std_sq": 0.01, "asset_cfg": SceneEntityCfg("robot")},
 556 |         )
 557 |         self.rewards.root_height_band = RewTerm(
 558 |             func=mdp.root_height_band_piecewise,
 559 |             weight=2.0,
 560 |             params={
 561 |                 "peak_min": 0.865,
 562 |                 "peak_max": 0.890,
 563 |                 "healthy_min": 0.84,
 564 |                 "healthy_max": 0.91,
 565 |                 "low_floor": 0.65,
 566 |                 "high_penalty_scale": 6.0,
 567 |                 "asset_cfg": SceneEntityCfg("robot"),
 568 |             },
 569 |         )
 570 |         self.rewards.base_height_low = RewTerm(
 571 |             func=mdp.base_height_below_target_l2,
 572 |             weight=-8.0,
 573 |             params={"target_height": 0.70, "asset_cfg": SceneEntityCfg("robot")},
 574 |         )
 575 |         self.rewards.base_height_high = RewTerm(
 576 |             func=mdp.base_height_above_target_l1,
 577 |             weight=-10.0,
 578 |             params={"target_height": 0.91, "asset_cfg": SceneEntityCfg("robot")},
 579 |         )
 580 |         self.rewards.actual_stroke_nominal = RewTerm(
 581 |             func=mdp.actual_stroke_nominal_l2,
 582 |             weight=-12.0,
 583 |             params={"stroke_nominal": 0.515, "action_name": "leg_hydraulic"},
 584 |         )
 585 |         self.rewards.actual_stroke_soft_limit = RewTerm(
 586 |             func=mdp.actual_stroke_soft_limit_penalty,
 587 |             weight=-4.0,
 588 |             params={"limit": 0.45, "action_name": "leg_hydraulic"},
 589 |         )
 590 |         self.rewards.stroke_range = RewTerm(
 591 |             func=mdp.stroke_range_penalty,
 592 |             weight=-2.0,
 593 |             params={"action_name": "leg_hydraulic"},
 594 |         )
 595 |         self.rewards.stroke_diagonal_balance = RewTerm(
 596 |             func=mdp.stroke_diagonal_balance_penalty,
 597 |             weight=-1.0,
 598 |             params={"action_name": "leg_hydraulic"},
 599 |         )
 600 |         self.rewards.hydraulic_action_magnitude = RewTerm(
 601 |             func=mdp.hydraulic_action_magnitude_l1,
 602 |             weight=-0.20,
 603 |             params={"action_name": "leg_hydraulic"},
 604 |         )
 605 |         self.rewards.hydraulic_action_range = RewTerm(
 606 |             func=mdp.hydraulic_action_range_penalty,
 607 |             weight=0.0,
 608 |             params={"action_name": "leg_hydraulic"},
 609 |         )
 610 |         self.rewards.low_stroke_negative_hydraulic_action = RewTerm(
 611 |             func=mdp.low_stroke_negative_hydraulic_action_penalty,
 612 |             weight=-3.0,
 613 |             params={
 614 |                 "stroke_threshold": 0.35,
 615 |                 "stroke_margin": 0.10,
 616 |                 "action_name": "leg_hydraulic",
 617 |             },
 618 |         )
 619 |         self.rewards.height_gated_low_stroke_negative_hydraulic_action = RewTerm(
 620 |             func=mdp.height_gated_low_stroke_negative_hydraulic_action_penalty,
 621 |             weight=-5.0,
 622 |             params={
 623 |                 "height_threshold": 0.72,
 624 |                 "height_margin": 0.07,
 625 |                 "stroke_threshold": 0.35,
 626 |                 "stroke_margin": 0.10,
 627 |                 "asset_cfg": SceneEntityCfg("robot"),
 628 |                 "action_name": "leg_hydraulic",
 629 |             },
 630 |         )
 631 |         self.rewards.high_height_low_stroke_negative_hydraulic_action = RewTerm(
 632 |             func=mdp.high_height_low_stroke_negative_hydraulic_action_penalty,
 633 |             weight=-4.0,
 634 |             params={
 635 |                 "height_threshold": 0.91,
 636 |                 "height_margin": 0.10,
 637 |                 "stroke_threshold": 0.45,
 638 |                 "stroke_margin": 0.10,
 639 |                 "asset_cfg": SceneEntityCfg("robot"),
 640 |                 "action_name": "leg_hydraulic",
 641 |             },
 642 |         )
 643 |         self.rewards.height_low_high_stroke_positive_hydraulic_action = RewTerm(
 644 |             func=mdp.height_low_high_stroke_positive_hydraulic_action_penalty,
 645 |             weight=0.0,
 646 |             params={
 647 |                 "height_threshold": 0.72,
 648 |                 "height_margin": 0.07,
 649 |                 "stroke_high_threshold": 0.55,
 650 |                 "stroke_margin": 0.10,
 651 |                 "asset_cfg": SceneEntityCfg("robot"),
 652 |                 "action_name": "leg_hydraulic",
 653 |             },
 654 |         )
 655 |         self.rewards.stroke_away_from_nominal_hydraulic_action = RewTerm(
 656 |             func=mdp.stroke_away_from_nominal_hydraulic_action_penalty,
 657 |             weight=-4.0,
 658 |             params={
 659 |                 "nominal_stroke": 0.515,
 660 |                 "action_name": "leg_hydraulic",
 661 |             },
 662 |         )
 663 |         self.rewards.front_rear_stroke_balance = RewTerm(
 664 |             func=mdp.front_rear_stroke_balance_penalty,
 665 |             weight=-3.0,
 666 |             params={"action_name": "leg_hydraulic"},
 667 |         )
 668 |         self.rewards.front_rear_stroke_split_action = RewTerm(
 669 |             func=mdp.front_rear_stroke_split_action_penalty,
 670 |             weight=0.0,
 671 |             params={
 672 |                 "diff_threshold": 0.10,
 673 |                 "diff_margin": 0.20,
 674 |                 "action_name": "leg_hydraulic",
 675 |             },
 676 |         )
 677 | 
 678 |         # Reward four-wheel contact coverage and balanced support forces during settling.
 679 |         self.rewards.wheel_contact_count = RewTerm(
 680 |             func=mdp.wheel_contact_count_reward,
 681 |             weight=1.5,
 682 |             params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "threshold": 1.0},
 683 |         )
 684 |         self.rewards.wheel_contact_stability = RewTerm(
 685 |             func=mdp.wheel_all_contact_reward,
 686 |             weight=1.0,
 687 |             params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "threshold": 1.0},
 688 |         )
 689 |         self.rewards.wheel_contact_balance = RewTerm(
 690 |             func=mdp.wheel_contact_force_balance_reward,
 691 |             weight=0.5,
 692 |             params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "threshold": 1.0},
 693 |         )
 694 |         self.rewards.contact_force_diag_balance = RewTerm(
 695 |             func=mdp.contact_force_diag_balance_penalty,
 696 |             weight=-1.0,
 697 |             params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"])},
 698 |         )
 699 |         self.rewards.contact_force_left_right_balance = RewTerm(
 700 |             func=mdp.contact_force_left_right_balance_penalty,
 701 |             weight=-0.4,
 702 |             params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"])},
 703 |         )
 704 |         self.rewards.contact_force_range_balance = RewTerm(
 705 |             func=mdp.contact_force_range_balance_penalty,
 706 |             weight=-0.3,
 707 |             params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"])},
 708 |         )
 709 |         self.rewards.contact_force_min_support = RewTerm(
 710 |             func=mdp.contact_force_min_support_penalty,
 711 |             weight=-0.3,
 712 |             params={
 713 |                 "min_force_threshold": 50.0,
 714 |                 "sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
 715 |             },
 716 |         )
 717 |         self.rewards.contact_force_min_ratio_support = RewTerm(
 718 |             func=mdp.contact_force_min_ratio_support_penalty,
 719 |             weight=-0.75,
 720 |             params={
 721 |                 "min_ratio_threshold": 0.10,
 722 |                 "sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
 723 |             },
 724 |         )
 725 |         self.rewards.low_contact_support = RewTerm(
 726 |             func=mdp.contact_force_min_support_penalty,
 727 |             weight=-0.2,
 728 |             params={
 729 |                 "min_force_threshold": 80.0,
 730 |                 "sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
 731 |             },
 732 |         )
 733 |         self.rewards.low_contact_ratio_support = RewTerm(
 734 |             func=mdp.contact_force_min_ratio_support_penalty,
 735 |             weight=-0.6,
 736 |             params={
 737 |                 "min_ratio_threshold": 0.15,
 738 |                 "sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
 739 |             },
 740 |         )
 741 | 
 742 | 
 743 | @configclass
 744 | class _RangerShortGoalSharedEnvCfg(RangerStandEnvCfg):
 745 |     """Shared short-goal observation, reset, safety, and reward scaffolding."""
 746 | 
 747 |     def __post_init__(self) -> None:
 748 |         super().__post_init__()
 749 |         self.stand_training_task = False
 750 |         self.enable_initial_stroke_randomization = False
 751 |         self.episode_length_s = 5.0
 752 |         self.scene.ground.spawn.size = (300.0, 300.0, 0.02)
 753 | 
 754 |         # Keep actor/critic observation shapes unchanged while swapping in a short-range dynamic goal.
 755 |         self.observations.policy_state.command_state.func = mdp.command_observation
 756 |         self.observations.policy_state.command_state.params = {
 757 |             "command_mode": "zero",
 758 |             "goal_source": "short_goal",
 759 |             "asset_cfg": SceneEntityCfg("robot"),
 760 |         }
 761 |         self.observations.policy_map.local_navigation_map.params["use_neutral_map"] = True
 762 | 
 763 |         self.events.reset_short_goal_target = EventTerm(
 764 |             func=mdp.reset_short_goal_target,
 765 |             mode="reset",
 766 |             params={
 767 |                 "distance_range": (0.5, 2.0),
 768 |                 "heading_range": (-0.7853981633974483, 0.7853981633974483),
 769 |                 "asset_cfg": SceneEntityCfg("robot"),
 770 |             },
 771 |         )
 772 | 
 773 |         self.terminations.bad_orientation.params["limit_angle"] = 1.2
 774 |         self.terminations.root_height_low.params["minimum_height"] = 0.30
 775 |         self.terminations.goal_reached = DoneTerm(
 776 |             func=mdp.short_goal_reached,
 777 |             params={"success_distance": 0.25, "asset_cfg": SceneEntityCfg("robot")},
 778 |         )
 779 | 
 780 |         # Preserve only the stand safety constraints; do not leave positive stand rewards
 781 |         # that let the policy score by standing still near the reset pose.
 782 |         self.rewards.alive.weight = 0.0
 783 |         self.rewards.forward_progress.weight = 0.0
 784 |         self.rewards.lateral_velocity.weight = 0.0
 785 |         self.rewards.wheel_joint_velocity.weight = 0.0
 786 |         self.rewards.base_xy_velocity.weight = 0.0
 787 |         self.rewards.base_yaw_rate.weight = 0.0
 788 |         self.rewards.yaw_drift_from_reset.weight = 0.0
 789 |         self.rewards.root_height_tracking.weight = 0.0
 790 |         self.rewards.root_height_band.weight = 0.0
 791 |         self.rewards.wheel_contact_count.weight = 0.0
 792 |         self.rewards.wheel_contact_stability.weight = 0.0
 793 |         self.rewards.wheel_contact_balance.weight = 0.0
 794 |         self.rewards.action_rate.weight = -0.02
 795 |         self.rewards.action_magnitude.weight = -0.001
 796 |         self.rewards.termination.weight = -20.0
 797 | 
 798 |         self.rewards.progress_to_goal = RewTerm(
 799 |             func=mdp.short_goal_progress_reward,
 800 |             weight=20.0,
 801 |             params={"asset_cfg": SceneEntityCfg("robot")},
 802 |         )
 803 |         self.rewards.goal_velocity = RewTerm(
 804 |             func=mdp.short_goal_velocity_towards_target,
 805 |             weight=4.0,
 806 |             params={"max_velocity": 0.8, "min_reward": -1.0, "max_reward": 1.5, "asset_cfg": SceneEntityCfg("robot")},
 807 |         )
 808 |         self.rewards.goal_success = RewTerm(
 809 |             func=mdp.short_goal_success_reward,
 810 |             weight=12.0,
 811 |             params={"success_distance": 0.25, "asset_cfg": SceneEntityCfg("robot")},
 812 |         )
 813 |         self.rewards.near_goal_stop = RewTerm(
 814 |             func=mdp.short_goal_near_stop_penalty,
 815 |             weight=-0.8,
 816 |             params={"stop_distance": 0.5, "yaw_weight": 0.5, "asset_cfg": SceneEntityCfg("robot")},
 817 |         )
 818 |         self.rewards.heading_alignment = RewTerm(
 819 |             func=mdp.short_goal_heading_alignment,
 820 |             weight=0.0,
 821 |             params={"asset_cfg": SceneEntityCfg("robot")},
 822 |         )
 823 | 
 824 | 
 825 | @configclass
 826 | class _RangerShortGoalSideTargetEnvCfg(_RangerShortGoalSharedEnvCfg):
 827 |     """Shared side-target distribution and heading-gated short-goal shaping."""
 828 | 
 829 |     def __post_init__(self) -> None:
 830 |         super().__post_init__()
 831 |         self.actions.wheel_motor_csv.velocity_limit = 80.0
 832 |         self.events.reset_short_goal_target = EventTerm(
 833 |             func=mdp.reset_short_goal_turn_target,
 834 |             mode="reset",
 835 |             params={
 836 |                 "distance_range": (1.0, 1.5),
 837 |                 "left_heading_range_deg": (25.0, 45.0),
 838 |                 "right_heading_range_deg": (-45.0, -25.0),
 839 |                 "asset_cfg": SceneEntityCfg("robot"),
 840 |             },
 841 |         )
 842 | 
 843 |         self.rewards.progress_to_goal = RewTerm(
 844 |             func=mdp.short_goal_progress_reward_heading_gated,
 845 |             weight=20.0,
 846 |             params={"heading_error_threshold": 0.35, "asset_cfg": SceneEntityCfg("robot")},
 847 |         )
 848 |         self.rewards.goal_velocity = RewTerm(
 849 |             func=mdp.short_goal_velocity_towards_target_heading_gated,
 850 |             weight=4.0,
 851 |             params={
 852 |                 "max_velocity": 0.8,
 853 |                 "min_reward": -1.0,
 854 |                 "max_reward": 1.5,
 855 |                 "heading_error_threshold": 0.35,
 856 |                 "asset_cfg": SceneEntityCfg("robot"),
 857 |             },
 858 |         )
 859 |         self.rewards.goal_success = RewTerm(
 860 |             func=mdp.short_goal_success_reward,
 861 |             weight=12.0,
 862 |             params={"success_distance": 0.25, "asset_cfg": SceneEntityCfg("robot")},
 863 |         )
 864 |         self.rewards.near_goal_stop = RewTerm(
 865 |             func=mdp.short_goal_near_stop_penalty,
 866 |             weight=-0.8,
 867 |             params={"stop_distance": 0.5, "yaw_weight": 0.5, "asset_cfg": SceneEntityCfg("robot")},
 868 |         )
 869 |         self.rewards.heading_error_reduction = RewTerm(
 870 |             func=mdp.short_goal_heading_error_reduction,
 871 |             weight=6.0,
 872 |             params={"min_progress": -0.5, "max_progress": 0.5, "asset_cfg": SceneEntityCfg("robot")},
 873 |         )
 874 |         self.rewards.turn_toward_goal = RewTerm(
 875 |             func=mdp.short_goal_turn_toward_goal,
 876 |             weight=2.0,
 877 |             params={"min_reward": -1.0, "max_reward": 1.0, "asset_cfg": SceneEntityCfg("robot")},
 878 |         )
 879 |         self.rewards.hydraulic_action_magnitude = RewTerm(
 880 |             func=mdp.hydraulic_action_magnitude_l1,
 881 |             weight=-0.20,
 882 |             params={"action_name": "leg_hydraulic"},
 883 |         )
 884 |         self.rewards.stroke_range = RewTerm(
 885 |             func=mdp.stroke_range_penalty,
 886 |             weight=-2.0,
 887 |             params={"action_name": "leg_hydraulic"},
 888 |         )
 889 |         self.rewards.stroke_diagonal_balance = RewTerm(
 890 |             func=mdp.stroke_diagonal_balance_penalty,
 891 |             weight=-1.0,
 892 |             params={"action_name": "leg_hydraulic"},
 893 |         )
 894 |         self.rewards.contact_force_diag_balance = RewTerm(
 895 |             func=mdp.contact_force_diag_balance_penalty,
 896 |             weight=-1.0,
 897 |             params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"])},
 898 |         )
 899 |         self.rewards.contact_force_left_right_balance = RewTerm(
 900 |             func=mdp.contact_force_left_right_balance_penalty,
 901 |             weight=-0.4,
 902 |             params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"])},
 903 |         )
 904 |         self.rewards.low_contact_ratio_support = RewTerm(
 905 |             func=mdp.contact_force_min_ratio_support_penalty,
 906 |             weight=-0.6,
 907 |             params={
 908 |                 "min_ratio_threshold": 0.15,
 909 |                 "sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
 910 |             },
 911 |         )
 912 | 
 913 | 
 914 | @configclass
 915 | class RangerShortGoalFlatEnvCfg(_RangerShortGoalSideTargetEnvCfg):
 916 |     """Side-goal heading task with policy-controlled suspension and wheel-differential priors."""
 917 | 
 918 |     def __post_init__(self) -> None:
 919 |         super().__post_init__()
 920 |         self.actions.wheel_motor_csv.velocity_limit = 80.0
 921 |         self.observations.policy_state.wheel_joint_vel_rel.params["scale"] = 80.0
 922 | 
 923 |         self.events.reset_short_goal_target = EventTerm(
 924 |             func=mdp.reset_short_goal_turn_target,
 925 |             mode="reset",
 926 |             params={
 927 |                 "distance_range": (1.0, 1.5),
 928 |                 "left_heading_range_deg": (25.0, 45.0),
 929 |                 "right_heading_range_deg": (-45.0, -25.0),
 930 |                 "asset_cfg": SceneEntityCfg("robot"),
 931 |             },
 932 |         )
 933 | 
 934 |         self.rewards.progress_to_goal.weight = 0.0
 935 |         self.rewards.goal_velocity.weight = 0.0
 936 |         self.rewards.goal_success.weight = 0.0
 937 |         self.rewards.near_goal_stop.weight = 0.0
 938 |         self.rewards.heading_error_reduction.weight = 9.0
 939 |         self.rewards.turn_toward_goal.weight = 4.0
 940 | 
 941 |         self.rewards.short_goal_wheel_diff_prior = RewTerm(
 942 |             func=mdp.short_goal_wheel_diff_prior_l1,
 943 |             weight=-1.0,
 944 |             params={"turn_gain": 40.0, "action_name": "wheel_motor_csv", "asset_cfg": SceneEntityCfg("robot")},
 945 |         )
 946 |         self.rewards.short_goal_forward_common_mode = RewTerm(
 947 |             func=mdp.short_goal_forward_common_mode_penalty,
 948 |             weight=-0.20,
 949 |             params={
 950 |                 "heading_error_threshold": 0.35,
 951 |                 "action_name": "wheel_motor_csv",
 952 |                 "asset_cfg": SceneEntityCfg("robot"),
 953 |             },
 954 |         )
 955 | 
 956 |         self.rewards.hydraulic_action_magnitude.weight = -0.05
 957 |         self.rewards.hydraulic_action_rate = RewTerm(
 958 |             func=mdp.hydraulic_action_rate_l1,
 959 |             weight=-0.03,
 960 |         )
 961 |         self.rewards.stroke_range.weight = -0.5
 962 |         self.rewards.stroke_diagonal_balance.weight = -0.25
 963 |         self.rewards.actual_stroke_nominal.weight = 0.0
 964 |         self.rewards.actual_stroke_soft_limit.weight = -1.0
 965 |         self.rewards.low_stroke_negative_hydraulic_action.weight = 0.0
 966 |         self.rewards.height_gated_low_stroke_negative_hydraulic_action.weight = 0.0
 967 |         self.rewards.high_height_low_stroke_negative_hydraulic_action.weight = 0.0
 968 |         self.rewards.stroke_away_from_nominal_hydraulic_action.weight = 0.0
 969 |         self.rewards.front_rear_stroke_balance.weight = 0.0
 970 | 
 971 |         self.rewards.wheel_contact_count.weight = 0.0
 972 |         self.rewards.wheel_contact_stability.weight = 0.0
 973 |         self.rewards.wheel_contact_balance.weight = 0.0
 974 |         if hasattr(self.rewards, "wheel_all_contact"):
 975 |             self.rewards.wheel_all_contact.weight = 0.0
 976 |         self.rewards.contact_force_diag_balance.weight = -0.15
 977 |         self.rewards.contact_force_left_right_balance.weight = 0.0
 978 |         self.rewards.contact_force_range_balance.weight = 0.0
 979 |         self.rewards.contact_force_min_support.weight = 0.0
 980 |         self.rewards.contact_force_min_ratio_support.weight = 0.0
 981 |         self.rewards.low_contact_support.weight = 0.0
 982 |         self.rewards.low_contact_ratio_support.weight = -0.15
 983 |         self.rewards.unloaded_wheel_spin = RewTerm(
 984 |             func=mdp.unloaded_wheel_spin_penalty,
 985 |             weight=-0.003,
 986 |             params={
 987 |                 "force_threshold": 20.0,
 988 |                 "sensor_cfg": SceneEntityCfg(
 989 |                     "wheel_contact_forces",
 990 |                     body_names=["w_lf", "w_lb", "w_rf", "w_rb"],
 991 |                 ),
 992 |                 "asset_cfg": SceneEntityCfg(
 993 |                     "robot",
 994 |                     joint_names=["w_lf", "w_lb", "w_rf", "w_rb"],
 995 |                 ),
 996 |             },
 997 |         )
 998 | 
 999 |         for reward_name in (
1000 |             "yaw_rate_tracking",
1001 |             "yaw_rate_command_tracking",
1002 |             "yaw_rate_error_penalty",
1003 |             "turn_direction",
1004 |             "wheel_turn_difference_error",
1005 |             "wheel_command_tracking",
1006 |             "wheel_command_error",
1007 |         ):
1008 |             if hasattr(self.rewards, reward_name):
1009 |                 getattr(self.rewards, reward_name).weight = 0.0
1010 |         self.rewards.forward_velocity_during_turn = RewTerm(
1011 |             func=mdp.short_goal_forward_velocity_during_turn_penalty,
1012 |             weight=0.0,
1013 |             params={"free_speed": 0.12, "asset_cfg": SceneEntityCfg("robot")},
1014 |         )
1015 |         self.rewards.signed_yaw_rate_tracking = RewTerm(
1016 |             func=mdp.short_goal_signed_yaw_rate_tracking,
1017 |             weight=0.0,
1018 |             params={"target_yaw_rate": 0.30, "heading_deadband": 0.10, "sigma": 0.25, "asset_cfg": SceneEntityCfg("robot")},
1019 |         )
1020 |         self.rewards.too_small_yaw_rate_when_error_large = RewTerm(
1021 |             func=mdp.short_goal_too_small_yaw_rate_when_error_large_penalty,
1022 |             weight=0.0,
1023 |             params={"min_yaw_rate": 0.10, "heading_threshold": 0.25, "asset_cfg": SceneEntityCfg("robot")},
1024 |         )
1025 |         self.rewards.wrong_direction_yaw = RewTerm(
1026 |             func=mdp.short_goal_wrong_direction_yaw_penalty,
1027 |             weight=0.0,
1028 |             params={"heading_deadband": 0.10, "asset_cfg": SceneEntityCfg("robot")},
1029 |         )
1030 |         self.rewards.wrong_direction_yaw_command = RewTerm(
1031 |             func=mdp.wrong_direction_yaw_command_penalty,
1032 |             weight=0.0,
1033 |             params={"active_threshold": 0.02, "asset_cfg": SceneEntityCfg("robot")},
1034 |         )
1035 |         self.rewards.too_small_yaw_rate_when_error_large_command = RewTerm(
1036 |             func=mdp.too_small_yaw_rate_command_penalty,
1037 |             weight=0.0,
1038 |             params={"active_threshold": 0.05, "asset_cfg": SceneEntityCfg("robot")},
1039 |         )
1040 |         self.rewards.wheel_turn_mode_soft_limit = RewTerm(
1041 |             func=mdp.wheel_turn_mode_target_soft_limit_penalty,
1042 |             weight=0.0,
1043 |             params={"action_name": "wheel_motor_csv", "soft_limit": 8.0},
1044 |         )
1045 |         self.rewards.wheel_target_abs_soft_limit = RewTerm(
1046 |             func=mdp.wheel_target_abs_soft_limit_penalty,
1047 |             weight=0.0,
1048 |             params={"action_name": "wheel_motor_csv", "soft_limit": 8.0},
1049 |         )
1050 |         self.rewards.wheel_joint_vel_abs_soft_limit = RewTerm(
1051 |             func=mdp.wheel_joint_vel_abs_soft_limit_penalty,
1052 |             weight=0.0,
1053 |             params={"action_name": "wheel_motor_csv", "soft_limit": 10.0, "asset_cfg": SceneEntityCfg("robot")},
1054 |         )
1055 |         self.rewards.wasted_turn_when_yaw_small = RewTerm(
1056 |             func=mdp.wasted_turn_when_yaw_small_penalty,
1057 |             weight=0.0,
1058 |             params={"action_name": "wheel_motor_csv", "active_threshold": 0.05, "asset_cfg": SceneEntityCfg("robot")},
1059 |         )
1060 | 
1061 | 
1062 | @configclass
1063 | class RangerShortGoalFlatV1EnvCfg(RangerShortGoalFlatEnvCfg):
1064 |     """Turn, approach, decelerate, and stop with four independent wheel outputs."""
1065 | 
1066 |     def __post_init__(self) -> None:
1067 |         super().__post_init__()
1068 |         self.episode_length_s = 22.0
1069 |         self.short_goal_stop_phase_enter_distance = 0.30
1070 |         self.short_goal_stop_success_distance = 0.50
1071 |         self.short_goal_stop_max_xy_speed = 0.15
1072 |         self.short_goal_stop_max_yaw_rate = 0.20
1073 |         self.short_goal_stop_required_hold_steps = 24
1074 |         self.short_goal_wheel_radius = 0.2024
1075 |         self.observations.policy_state.command_state.params.update(
1076 |             {
1077 |                 "short_goal_encoding_mode": "long_term",
1078 |                 "short_goal_observation_max_distance": None,
1079 |                 "short_goal_near_distance_range": 3.0,
1080 |                 "short_goal_global_distance_unit": 1.0,
1081 |                 "short_goal_velocity_reference": 1.5,
1082 |             }
1083 |         )
1084 | 
1085 |         # Keep wheel targets in a physically useful range and normalize wheel-state observations consistently.
1086 |         self.actions.wheel_motor_csv.velocity_limit = 20.0
1087 |         self.observations.policy_state.wheel_joint_vel_rel.params["scale"] = 20.0
1088 | 
1089 |         self.events.reset_short_goal_target = EventTerm(
1090 |             func=mdp.reset_short_goal_turn_target,
1091 |             mode="reset",
1092 |             params={
1093 |                 "distance_range": (5.0, 8.0),
1094 |                 "left_heading_range_deg": (25.0, 45.0),
1095 |                 "right_heading_range_deg": (-45.0, -25.0),
1096 |                 "paired_sides": True,
1097 |                 "asset_cfg": SceneEntityCfg("robot"),
1098 |             },
1099 |         )
1100 | 
1101 |         # A low chassis is unsafe for the under-body sensors.
1102 |         self.terminations.root_height_low.params["minimum_height"] = 0.65
1103 |         self.terminations.goal_reached = None
1104 |         self.terminations.stopped_goal_reached = DoneTerm(
1105 |             func=mdp.short_goal_stopped,
1106 |             params={
1107 |                 "success_distance": 0.50,
1108 |                 "max_xy_speed": 0.15,
1109 |                 "max_yaw_rate": 0.20,
1110 |                 "required_hold_steps": 24,
1111 |                 "asset_cfg": SceneEntityCfg("robot"),
1112 |             },
1113 |         )
1114 | 
1115 |         # Remove inherited historical/zero-weight terms so the active reward set is explicit.
1116 |         for reward_name in (
1117 |             "alive",
1118 |             "forward_progress",
1119 |             "upright",
1120 |             "lateral_velocity",
1121 |             "leg_joint_deviation",
1122 |             "wheel_joint_velocity",
1123 |             "leg_joint_velocity",
1124 |             "action_rate",
1125 |             "action_magnitude",
1126 |             "base_xy_velocity",
1127 |             "base_yaw_rate",
1128 |             "yaw_drift_from_reset",
1129 |             "root_height_tracking",
1130 |             "root_height_band",
1131 |             "actual_stroke_nominal",
1132 |             "actual_stroke_soft_limit",
1133 |             "stroke_diagonal_balance",
1134 |             "hydraulic_action_range",
1135 |             "low_stroke_negative_hydraulic_action",
1136 |             "height_gated_low_stroke_negative_hydraulic_action",
1137 |             "high_height_low_stroke_negative_hydraulic_action",
1138 |             "height_low_high_stroke_positive_hydraulic_action",
1139 |             "stroke_away_from_nominal_hydraulic_action",
1140 |             "front_rear_stroke_balance",
1141 |             "front_rear_stroke_split_action",
1142 |             "wheel_contact_count",
1143 |             "wheel_contact_stability",
1144 |             "wheel_contact_balance",
1145 |             "contact_force_diag_balance",
1146 |             "contact_force_left_right_balance",
1147 |             "contact_force_range_balance",
1148 |             "contact_force_min_support",
1149 |             "contact_force_min_ratio_support",
1150 |             "low_contact_support",
1151 |             "goal_velocity",
1152 |             "goal_success",
1153 |             "near_goal_stop",
1154 |             "heading_alignment",
1155 |             "turn_toward_goal",
1156 |             "signed_yaw_rate_tracking",
1157 |             "too_small_yaw_rate_when_error_large",
1158 |             "wrong_direction_yaw",
1159 |             "wrong_direction_yaw_command",
1160 |             "too_small_yaw_rate_when_error_large_command",
1161 |             "forward_velocity_during_turn",
1162 |             "wheel_turn_mode_soft_limit",
1163 |             "wheel_target_abs_soft_limit",
1164 |             "wheel_joint_vel_abs_soft_limit",
1165 |             "wasted_turn_when_yaw_small",
1166 |         ):
1167 |             if hasattr(self.rewards, reward_name):
1168 |                 setattr(self.rewards, reward_name, None)
1169 | 
1170 |         # Task progress and turn-to-approach transition.
1171 |         self.rewards.progress_to_goal = RewTerm(
1172 |             func=mdp.short_goal_progress_reward_alignment_gated,
1173 |             weight=4.0,
1174 |             params={
1175 |                 "heading_deadband": 0.20,
1176 |                 "heading_full": 0.80,
1177 |                 "alignment_floor": 0.15,
1178 |                 "asset_cfg": SceneEntityCfg("robot"),
1179 |             },
1180 |         )
1181 |         self.rewards.heading_error_reduction = RewTerm(
1182 |             func=mdp.short_goal_heading_error_reduction,
1183 |             weight=9.0,
1184 |             params={
1185 |                 "min_progress": -0.5,
1186 |                 "max_progress": 0.5,
1187 |                 "use_turn_distance_gate": True,
1188 |                 "turn_gate_start_distance": 0.50,
1189 |                 "turn_gate_full_distance": 0.80,
1190 |                 "asset_cfg": SceneEntityCfg("robot"),
1191 |             },
1192 |         )
1193 |         self.rewards.heading_error_persistent = RewTerm(
1194 |             func=mdp.short_goal_heading_error_cost,
1195 |             weight=-0.25,
1196 |             params={
1197 |                 "fade_start_distance": 0.50,
1198 |                 "full_distance": 0.80,
1199 |                 "asset_cfg": SceneEntityCfg("robot"),
1200 |             },
1201 |         )
1202 |         self.rewards.turn_toward_goal = None
1203 |         self.rewards.yaw_rate_tracking = RewTerm(
1204 |             func=mdp.short_goal_continuous_yaw_rate_tracking_penalty,
1205 |             weight=-0.12,
1206 |             params={
1207 |                 "yaw_rate_max": 0.35,
1208 |                 "heading_deadband": 0.04,
1209 |                 "heading_scale": 0.40,
1210 |                 "yaw_rate_reference": 0.35,
1211 |                 "max_normalized_error": 2.0,
1212 |                 "use_turn_distance_gate": True,
1213 |                 "turn_gate_start_distance": 0.50,
1214 |                 "turn_gate_full_distance": 0.80,
1215 |                 "asset_cfg": SceneEntityCfg("robot"),
1216 |             },
1217 |         )
1218 |         self.rewards.short_goal_speed_profile = RewTerm(
1219 |             func=mdp.short_goal_speed_profile_penalty,
1220 |             weight=-2.0,
1221 |             params={
1222 |                 "stop_distance": 0.30,
1223 |                 "braking_acceleration": 0.60,
1224 |                 "reaction_time": 0.20,
1225 |                 "braking_margin": 0.08,
1226 |                 "near_distance": 1.0,
1227 |                 "yaw_rate_ref": 0.80,
1228 |                 "yaw_component_weight": 0.5,
1229 |                 "asset_cfg": SceneEntityCfg("robot"),
1230 |             },
1231 |         )
1232 |         self.rewards.cruise_underspeed = RewTerm(
1233 |             func=mdp.short_goal_cruise_underspeed_penalty,
1234 |             weight=-0.30,
1235 |             params={
1236 |                 "capture_distance": 0.30,
1237 |                 "approach_full_distance": 0.60,
1238 |                 "cruise_full_distance": 1.20,
1239 |                 "approach_speed": 0.35,
1240 |                 "cruise_speed": 0.80,
1241 |                 "heading_deadband": 0.20,
1242 |                 "heading_full": 0.80,
1243 |                 "alignment_floor": 0.15,
1244 |                 "asset_cfg": SceneEntityCfg("robot"),
1245 |             },
1246 |         )
1247 |         self.rewards.near_goal_away_speed = RewTerm(
1248 |             func=mdp.short_goal_near_goal_away_speed_penalty,
1249 |             weight=-0.30,
1250 |             params={
1251 |                 "stop_distance": 0.30,
1252 |                 "active_distance": 1.0,
1253 |                 "speed_reference": 0.30,
1254 |                 "asset_cfg": SceneEntityCfg("robot"),
1255 |             },
1256 |         )
1257 |         # Do not force wheel speed toward zero at the 0.30 m boundary. The
1258 |         # vehicle now crosses the boundary with finite speed, then the stop
1259 |         # phase overrides only the wheel action to zero.
1260 |         self.rewards.approach_wheel_target_excess = None
1261 |         self.rewards.stopped_goal_success = RewTerm(
1262 |             func=mdp.short_goal_stopped_success_reward,
1263 |             weight=5.0,
1264 |             params={
1265 |                 "success_distance": 0.50,
1266 |                 "max_xy_speed": 0.15,
1267 |                 "max_yaw_rate": 0.20,
1268 |                 "required_hold_steps": 24,
1269 |                 "asset_cfg": SceneEntityCfg("robot"),
1270 |             },
1271 |         )
1272 |         self.rewards.stop_phase_base_xy_speed = RewTerm(
1273 |             func=mdp.short_goal_stop_phase_xy_speed_penalty,
1274 |             weight=-2.0,
1275 |             params={"asset_cfg": SceneEntityCfg("robot")},
1276 |         )
1277 |         self.rewards.stop_phase_yaw_rate = RewTerm(
1278 |             func=mdp.short_goal_stop_phase_yaw_rate_penalty,
1279 |             weight=-1.0,
1280 |             params={"asset_cfg": SceneEntityCfg("robot")},
1281 |         )
1282 |         self.rewards.stop_phase_wheel_target = RewTerm(
1283 |             func=mdp.short_goal_stop_phase_wheel_target_penalty,
1284 |             weight=-0.2,
1285 |             params={"action_name": "wheel_motor_csv"},
1286 |         )
1287 |         self.rewards.stop_phase_distance_drift = RewTerm(
1288 |             func=mdp.short_goal_stop_phase_distance_drift_penalty,
1289 |             weight=-2.0,
1290 |             params={"enter_distance": 0.50, "asset_cfg": SceneEntityCfg("robot")},
1291 |         )
1292 | 
1293 |         # Keep four independent wheel outputs while adding explicit flat-ground coordination priors.
1294 |         self.rewards.short_goal_wheel_diff_prior = RewTerm(
1295 |             func=mdp.short_goal_wheel_turn_mode_prior_l1,
1296 |             weight=-0.05,
1297 |             params={
1298 |                 "turn_gain": 0.5,
1299 |                 "action_name": "wheel_motor_csv",
1300 |                 "use_turn_distance_gate": True,
1301 |                 "turn_gate_start_distance": 0.50,
1302 |                 "turn_gate_full_distance": 0.80,
1303 |                 "asset_cfg": SceneEntityCfg("robot"),
1304 |             },
1305 |         )
1306 |         # Disable the wheel-mean common-mode penalty: same-side front/rear opposition can cancel in the mean.
1307 |         self.rewards.short_goal_forward_common_mode = None
1308 |         self.rewards.wheel_same_side_target_consistency = RewTerm(
1309 |             func=mdp.wheel_same_side_target_consistency_l1,
1310 |             weight=-0.20,
1311 |             params={"deadband": 0.05, "action_name": "wheel_motor_csv"},
1312 |         )
1313 |         self.rewards.wheel_same_side_opposite_sign = RewTerm(
1314 |             func=mdp.wheel_same_side_opposite_sign_penalty,
1315 |             weight=-1.0,
1316 |             params={"margin": 0.03, "action_name": "wheel_motor_csv"},
1317 |         )
1318 |         self.rewards.loaded_wheel_longitudinal_slip = RewTerm(
1319 |             func=mdp.loaded_wheel_longitudinal_slip_penalty,
1320 |             weight=-0.05,
1321 |             params={
1322 |                 "wheel_radius": 0.2024,
1323 |                 "min_contact_force": 20.0,
1324 |                 "min_motion_speed": 0.20,
1325 |                 "absolute_margin": 0.10,
1326 |                 "relative_margin": 0.15,
1327 |                 "excess_speed_reference": 0.50,
1328 |                 "max_normalized_excess": 2.0,
1329 |                 "sensor_cfg": SceneEntityCfg(
1330 |                     "wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]
1331 |                 ),
1332 |                 "asset_cfg": SceneEntityCfg("robot"),
1333 |             },
1334 |         )
1335 |         self.rewards.wheel_target_rate = RewTerm(
1336 |             func=mdp.wheel_target_rate_l1,
1337 |             weight=-0.02,
1338 |             params={"action_name": "wheel_motor_csv"},
1339 |         )
1340 | 
1341 |         # Posture, sensor-clearance, contact, and actuator safety.
1342 |         self.rewards.base_vertical_velocity.weight = -0.8
1343 |         self.rewards.base_roll_pitch_rate.weight = -0.6
1344 |         self.rewards.roll_angle.weight = -1.3
1345 |         self.rewards.pitch_angle.weight = -1.5
1346 |         self.rewards.base_height_low.weight = -8.0
1347 |         self.rewards.base_height_low.params["target_height"] = 0.70
1348 |         self.rewards.base_height_high.weight = -10.0
1349 |         self.rewards.base_height_high.params["target_height"] = 0.91
1350 |         self.rewards.termination = RewTerm(
1351 |             func=mdp.failure_termination_penalty,
1352 |             weight=-20.0,
1353 |             params={"excluded_terms": ("time_out", "stopped_goal_reached")},
1354 |         )
1355 |         self.rewards.hydraulic_action_magnitude.weight = -0.05
1356 |         self.rewards.hydraulic_action_rate.weight = -0.06
1357 |         self.rewards.stroke_range.weight = -0.3
1358 |         self.rewards.stroke_high_soft_limit = RewTerm(
1359 |             func=mdp.stroke_high_soft_limit_penalty,
1360 |             weight=-1.0,
1361 |             params={"soft_start": 0.55, "hard_reference": 0.60, "action_name": "leg_hydraulic"},
1362 |         )
1363 |         self.rewards.low_contact_ratio_support.weight = -0.20
1364 |         self.rewards.unloaded_wheel_spin.weight = -0.006
1365 | 
1366 | 
1367 | @configclass
1368 | class RangerShortGoalFlatV2EnvCfg(RangerShortGoalFlatV1EnvCfg):
1369 |     """Wheel-only flat short-goal stage with suspension fixed at nominal mid-stroke."""
1370 | 
1371 |     def __post_init__(self) -> None:
1372 |         super().__post_init__()
1373 |         # Keep the 8-D action interface for checkpoint compatibility, but remove active
1374 |         # suspension control at the environment boundary. A normalized action of zero
1375 |         # corresponds to the nominal 0.50 m stroke used by the current hydraulic action term.
1376 |         self.fixed_suspension_action = 0.0
1377 | 
1378 | 
1379 | @configclass
1380 | class RangerShortGoalFlatV3EnvCfg(RangerShortGoalFlatV1EnvCfg):
1381 |     """Stage A: restore limited suspension authority on the original goal distribution."""
1382 | 
1383 |     def __post_init__(self) -> None:
1384 |         super().__post_init__()
1385 |         self.fixed_suspension_action = None
1386 |         self.suspension_action_scale = 0.05
1387 |         # Flat-ground stopping does not need an asymmetric latched suspension target.
1388 |         self.stop_phase_suspension_action = 0.0
1389 | 
1390 | 
1391 | @configclass
1392 | class RangerShortGoalFlatV4EnvCfg(RangerShortGoalFlatV3EnvCfg):
1393 |     """Stage B: widen heading coverage while keeping the 5--8 m distance range."""
1394 | 
1395 |     def __post_init__(self) -> None:
1396 |         super().__post_init__()
1397 |         self.events.reset_short_goal_target = EventTerm(
1398 |             func=mdp.reset_short_goal_stratified_target,
1399 |             mode="reset",
1400 |             params={
1401 |                 "distance_bands": ((5.0, 8.0),),
1402 |                 "distance_weights": (1.0,),
1403 |                 "heading_bands_deg": (
1404 |                     (-60.0, -45.0),
1405 |                     (-45.0, -15.0),
1406 |                     (-15.0, 15.0),
1407 |                     (15.0, 45.0),
1408 |                     (45.0, 60.0),
1409 |                 ),
1410 |                 "heading_weights": (0.20, 0.20, 0.20, 0.20, 0.20),
1411 |                 "asset_cfg": SceneEntityCfg("robot"),
1412 |             },
1413 |         )
1414 | 
1415 | 
1416 | @configclass
1417 | class RangerShortGoalFlatV5EnvCfg(RangerShortGoalFlatV4EnvCfg):
1418 |     """Stage C: widen distance coverage to 3--12 m with stratified sampling."""
1419 | 
1420 |     def __post_init__(self) -> None:
1421 |         super().__post_init__()
1422 |         self.events.reset_short_goal_target.params.update(
1423 |             {
1424 |                 "distance_bands": ((3.0, 5.0), (5.0, 8.0), (8.0, 12.0)),
1425 |                 "distance_weights": (0.25, 0.50, 0.25),
1426 |             }
1427 |         )
1428 | 
1429 | 
1430 | @configclass
1431 | class RangerShortGoalFlatV6EnvCfg(RangerShortGoalFlatV5EnvCfg):
1432 |     """Stage D: final flat curriculum over 3--12 m and headings up to 75 degrees."""
1433 | 
1434 |     def __post_init__(self) -> None:
1435 |         super().__post_init__()
1436 |         self.events.reset_short_goal_target.params.update(
1437 |             {
1438 |                 "heading_bands_deg": (
1439 |                     (-75.0, -45.0),
1440 |                     (-45.0, -15.0),
1441 |                     (-15.0, 15.0),
1442 |                     (15.0, 45.0),
1443 |                     (45.0, 75.0),
1444 |                 ),
1445 |                 "heading_weights": (0.20, 0.20, 0.20, 0.20, 0.20),
1446 |             }
1447 |         )
1448 | 
1449 | 
1450 | @configclass
1451 | class RangerShortGoalFlatV7EnvCfg(RangerShortGoalFlatV6EnvCfg):
1452 |     """Stage V7: full-authority suspension and wheel fine-tuning on the V6 curriculum."""
1453 | 
1454 |     def __post_init__(self) -> None:
1455 |         super().__post_init__()
1456 |         self.fixed_suspension_action = None
1457 |         self.suspension_action_scale = 1.0
1458 |         self.stop_phase_suspension_action = 0.0
1459 | 
1460 | 
1461 | @configclass
1462 | class RangerShortGoalFlatV8EnvCfg(RangerShortGoalFlatV7EnvCfg):
1463 |     """Stage V8: require a balanced, settled suspension posture before declaring success."""
1464 | 
1465 |     def __post_init__(self) -> None:
1466 |         super().__post_init__()
1467 |         self.short_goal_stop_max_roll = 0.035
1468 |         self.short_goal_stop_max_pitch = 0.035
1469 |         self.short_goal_stop_max_stroke_tracking_error = 0.02
1470 |         self.short_goal_stop_count_only_after_phase = True
1471 |         self.short_goal_stop_required_hold_steps = 60
1472 |         posture_success_params = {
1473 |             "required_hold_steps": self.short_goal_stop_required_hold_steps,
1474 |             "max_roll": self.short_goal_stop_max_roll,
1475 |             "max_pitch": self.short_goal_stop_max_pitch,
1476 |             "max_stroke_tracking_error": self.short_goal_stop_max_stroke_tracking_error,
1477 |             "action_name": "leg_hydraulic",
1478 |         }
1479 |         self.terminations.stopped_goal_reached.params.update(posture_success_params)
1480 |         self.rewards.stopped_goal_success.params.update(posture_success_params)
1481 | 
1482 | 
1483 | @configclass
1484 | class RangerShortGoalFlatV9EnvCfg(RangerShortGoalFlatV8EnvCfg):
1485 |     """Stage V9: preserve policy suspension control while forcing zero wheel action in stop phase."""
1486 | 
1487 |     def __post_init__(self) -> None:
1488 |         super().__post_init__()
1489 |         self.stop_phase_suspension_mode = "policy"
1490 |         self.stop_phase_suspension_action = None
1491 | 
1492 |         # V9 removes global suspension regularizers that can oppose necessary
1493 |         # posture correction and future terrain-following motion. Safety,
1494 |         # posture, wheel-support, and stop-phase terms remain active.
1495 |         self.rewards.hydraulic_action_magnitude = None
1496 |         self.rewards.hydraulic_action_rate = None
1497 |         self.rewards.stroke_range = None
1498 | 
1499 |         self.rewards.stop_phase_roll_error = RewTerm(
1500 |             func=mdp.short_goal_stop_phase_roll_error_penalty,
1501 |             weight=-0.10,
1502 |             params={
1503 |                 "reference_angle": self.short_goal_stop_max_roll,
1504 |                 "max_normalized_error": 4.0,
1505 |                 "asset_cfg": SceneEntityCfg("robot"),
1506 |             },
1507 |         )
1508 |         self.rewards.stop_phase_pitch_error = RewTerm(
1509 |             func=mdp.short_goal_stop_phase_pitch_error_penalty,
1510 |             weight=-0.10,
1511 |             params={
1512 |                 "reference_angle": self.short_goal_stop_max_pitch,
1513 |                 "max_normalized_error": 4.0,
1514 |                 "asset_cfg": SceneEntityCfg("robot"),
1515 |             },
1516 |         )
1517 | 
1518 | 
1519 | @configclass
1520 | class RangerShortGoalFlatV10EnvCfg(RangerShortGoalFlatV9EnvCfg):
1521 |     """Stage V10: capture and stop inside 0.50 m, with an optional 0.30 m precision bonus."""
1522 | 
1523 |     def __post_init__(self) -> None:
1524 |         super().__post_init__()
1525 | 
1526 |         # Align stop-phase capture with the actual success radius. Stable success
1527 |         # still requires 60 consecutive steps satisfying all V8/V9 stop gates.
1528 |         self.short_goal_stop_phase_enter_distance = self.short_goal_stop_success_distance
1529 |         self.rewards.stop_phase_distance_drift.params["enter_distance"] = (
1530 |             self.short_goal_stop_success_distance
1531 |         )
1532 | 
1533 |         # Reaching 0.30 m is no longer mandatory for success. It remains a
1534 |         # one-time precision objective worth less than the final stopped success.
1535 |         self.rewards.precision_reach_bonus = RewTerm(
1536 |             func=mdp.short_goal_precision_reach_reward,
1537 |             weight=2.0,
1538 |             params={
1539 |                 "precision_distance": 0.30,
1540 |                 "asset_cfg": SceneEntityCfg("robot"),
1541 |             },
1542 |         )
1543 | 
1544 | 
1545 | @configclass
1546 | class RangerShortGoalFlatV11EnvCfg(RangerShortGoalFlatV10EnvCfg):
1547 |     """Stage V11: policy-controlled braking inside the latched 0.50 m stop phase."""
1548 | 
1549 |     def __post_init__(self) -> None:
1550 |         super().__post_init__()
1551 | 
1552 |         # Keep the stop-phase latch and observation flag, but execute the policy's
1553 |         # wheel outputs instead of replacing them with four zeros.
1554 |         self.stop_phase_wheel_override_enabled = False
1555 | 
1556 |         # Replace immediate-stop penalties with distance-dependent envelopes.
1557 |         # At 0.50 m the policy may still move slowly; the admissible motion shrinks
1558 |         # linearly to zero at the 0.30 m precision radius.
1559 |         self.rewards.stop_phase_base_xy_speed = None
1560 |         self.rewards.stop_phase_yaw_rate = None
1561 |         self.rewards.stop_phase_wheel_target = None
1562 |         self.rewards.stop_phase_xy_speed_envelope = RewTerm(
1563 |             func=mdp.short_goal_stop_phase_xy_speed_envelope_penalty,
1564 |             weight=-2.0,
1565 |             params={
1566 |                 "enter_distance": 0.50,
1567 |                 "zero_distance": 0.30,
1568 |                 "allowed_speed_at_enter": 0.25,
1569 |                 "asset_cfg": SceneEntityCfg("robot"),
1570 |             },
1571 |         )
1572 |         self.rewards.stop_phase_yaw_rate_envelope = RewTerm(
1573 |             func=mdp.short_goal_stop_phase_yaw_rate_envelope_penalty,
1574 |             weight=-1.0,
1575 |             params={
1576 |                 "enter_distance": 0.50,
1577 |                 "zero_distance": 0.30,
1578 |                 "allowed_yaw_rate_at_enter": 0.30,
1579 |                 "asset_cfg": SceneEntityCfg("robot"),
1580 |             },
1581 |         )
1582 |         self.rewards.stop_phase_wheel_target_envelope = RewTerm(
1583 |             func=mdp.short_goal_stop_phase_wheel_target_envelope_penalty,
1584 |             weight=-1.0,
1585 |             params={
1586 |                 "enter_distance": 0.50,
1587 |                 "zero_distance": 0.30,
1588 |                 "allowed_action_at_enter": 0.20,
1589 |                 "action_name": "wheel_motor_csv",
1590 |                 "asset_cfg": SceneEntityCfg("robot"),
1591 |             },
1592 |         )
1593 | 
1594 |         # Dense shaping in 0.30--0.50 m complements the one-time 0.30 m bonus.
1595 |         self.rewards.stop_phase_precision_closeness = RewTerm(
1596 |             func=mdp.short_goal_stop_phase_precision_closeness_reward,
1597 |             weight=1.0,
1598 |             params={
1599 |                 "enter_distance": 0.50,
1600 |                 "precision_distance": 0.30,
1601 |                 "asset_cfg": SceneEntityCfg("robot"),
1602 |             },
1603 |         )
1604 | 
```

### README.md

Bytes: 5213
SHA-256: 0157211dd8b10478cac5f88bc75bdc21e77e12fa928aaec8b4e1ddfba2f823d1
Lines: 1-188 of 188

```markdown
  1 | 
  2 | # Isaac Lab 项目模板
  3 | 
  4 | ## 概述
  5 | 
  6 | 本项目/仓库是一个基于 Isaac Lab 构建项目或扩展的模板。
  7 | 它允许你在 Isaac Lab 核心仓库之外的独立环境中进行开发。
  8 | 
  9 | **主要特点：**
 10 | 
 11 | * `隔离性`：在 Isaac Lab 核心仓库之外工作，确保你的开发内容保持自包含，不会污染官方源码。
 12 | * `灵活性`：该模板可以让你的代码作为 Omniverse 中的扩展运行。
 13 | 
 14 | **关键词：** extension、template、isaaclab
 15 | 
 16 | ---
 17 | 
 18 | ## 安装
 19 | 
 20 | * 按照 [安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。
 21 |   推荐使用 conda 安装，因为这样可以更方便地从终端调用 Python 脚本。
 22 | 
 23 | * 将本项目/仓库克隆或复制到 Isaac Lab 安装目录之外的位置，也就是不要放在 `IsaacLab` 目录里面。
 24 | 
 25 | * 使用已经安装 Isaac Lab 的 Python 解释器，以 editable mode 安装本库：
 26 | 
 27 |   ```bash
 28 |   # 如果 Isaac Lab 没有安装在 Python venv 或 conda 中，
 29 |   # 请使用 'PATH_TO_isaaclab.sh|bat -p' 代替 'python'
 30 |   python -m pip install -e source/Ranger
 31 |   ```
 32 | 
 33 | ---
 34 | 
 35 | ## 验证扩展是否正确安装
 36 | 
 37 | ### 1. 列出可用任务
 38 | 
 39 | 注意：如果任务名称发生变化，可能需要更新 `scripts/list_envs.py` 文件中的搜索模式 `"Template-"`，这样任务才能被列出来。
 40 | 
 41 | ```bash
 42 | # 如果 Isaac Lab 没有安装在 Python venv 或 conda 中，
 43 | # 请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
 44 | python scripts/list_envs.py
 45 | ```
 46 | 
 47 | ---
 48 | 
 49 | ### 2. 运行一个任务
 50 | 
 51 | ```bash
 52 | # 如果 Isaac Lab 没有安装在 Python venv 或 conda 中，
 53 | # 请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
 54 | python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
 55 | ```
 56 | 
 57 | 其中：
 58 | 
 59 | ```text
 60 | <RL_LIBRARY> 表示你选择的强化学习库，例如 rsl_rl、rl_games、skrl、sb3
 61 | <TASK_NAME> 表示具体任务名称
 62 | ```
 63 | 
 64 | ---
 65 | 
 66 | ### 3. 使用 dummy agents 运行任务
 67 | 
 68 | 这些 dummy agents 会输出零动作或随机动作。
 69 | 它们用于检查环境配置是否正确。
 70 | 
 71 | #### 零动作 agent
 72 | 
 73 | ```bash
 74 | # 如果 Isaac Lab 没有安装在 Python venv 或 conda 中，
 75 | # 请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
 76 | python scripts/zero_agent.py --task=<TASK_NAME>
 77 | ```
 78 | 
 79 | #### 随机动作 agent
 80 | 
 81 | ```bash
 82 | # 如果 Isaac Lab 没有安装在 Python venv 或 conda 中，
 83 | # 请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
 84 | python scripts/random_agent.py --task=<TASK_NAME>
 85 | ```
 86 | 
 87 | ---
 88 | 
 89 | ## 设置 IDE（可选）
 90 | 
 91 | 如果要设置 IDE，请按以下步骤操作：
 92 | 
 93 | * 运行 VS Code Tasks：按下 `Ctrl+Shift+P`，选择 `Tasks: Run Task`，然后在下拉菜单中运行 `setup_python_env`。
 94 | 
 95 | 运行该任务时，系统会提示你添加 Isaac Sim 安装路径的绝对路径。
 96 | 
 97 | 如果一切执行正确，它会在 `.vscode` 目录下创建一个 `.python.env` 文件。
 98 | 该文件包含 Isaac Sim 和 Omniverse 提供的所有扩展的 Python 路径。
 99 | 这有助于 VS Code 在你写代码时索引 Python 模块，从而提供智能提示。
100 | 
101 | ---
102 | 
103 | ## 设置为 Omniverse 扩展（可选）
104 | 
105 | 本模板提供了一个示例 UI 扩展。启用你的扩展后，该示例会加载：
106 | 
107 | ```text
108 | source/Ranger/Ranger/ui_extension_example.py
109 | ```
110 | 
111 | 要启用你的扩展，请按以下步骤操作：
112 | 
113 | ### 1. 将本项目/仓库的搜索路径添加到扩展管理器
114 | 
115 | * 通过 `Window` -> `Extensions` 打开扩展管理器。
116 | * 点击 **Hamburger Icon**，也就是三横线菜单，然后进入 `Settings`。
117 | * 在 `Extension Search Paths` 中输入本项目/仓库的 `source` 目录的绝对路径。
118 | * 如果还没有添加 Isaac Lab 的扩展目录，也需要在 `Extension Search Paths` 中添加指向 Isaac Lab `source` 目录的路径，例如：
119 | 
120 | ```text
121 | IsaacLab/source
122 | ```
123 | 
124 | * 点击 **Hamburger Icon**，然后点击 `Refresh`。
125 | 
126 | ---
127 | 
128 | ### 2. 搜索并启用你的扩展
129 | 
130 | * 在 `Third Party` 分类下找到你的扩展。
131 | * 打开开关以启用该扩展。
132 | 
133 | ---
134 | 
135 | ## 代码格式化
136 | 
137 | 本模板提供了一个 pre-commit 模板，可以自动格式化你的代码。
138 | 
139 | 安装 pre-commit：
140 | 
141 | ```bash
142 | pip install pre-commit
143 | ```
144 | 
145 | 然后运行：
146 | 
147 | ```bash
148 | pre-commit run --all-files
149 | ```
150 | 
151 | ---
152 | 
153 | ## 故障排查
154 | 
155 | ### Pylance 没有索引扩展
156 | 
157 | 在某些 VS Code 版本中，部分扩展可能无法被正确索引。
158 | 
159 | 这种情况下，可以在 `.vscode/settings.json` 中的 `"python.analysis.extraPaths"` 字段下添加你的扩展路径：
160 | 
161 | ```json
162 | {
163 |     "python.analysis.extraPaths": [
164 |         "<path-to-ext-repo>/source/Ranger"
165 |     ]
166 | }
167 | ```
168 | 
169 | ---
170 | 
171 | ### Pylance 崩溃
172 | 
173 | 如果遇到 `pylance` 崩溃，可能是因为索引的文件过多，导致内存不足。
174 | 
175 | 一种解决方法是排除一些项目中用不到的 Omniverse 包。
176 | 
177 | 具体做法是修改 `.vscode/settings.json`，在 `"python.analysis.extraPaths"` 字段下，注释掉一些不使用的包。
178 | 
179 | 一些可能可以排除的包包括：
180 | 
181 | ```json
182 | "<path-to-isaac-sim>/extscache/omni.anim.*"         // 动画相关包
183 | "<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI 工具相关包
184 | "<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI 工具相关包
185 | "<path-to-isaac-sim>/extscache/omni.services.*"     // Services 工具相关包
186 | ...
187 | ```
188 | 
```

## Skipped Files

- source/Ranger/Ranger/tasks/manager_based/ranger/forward_debug_env.py [File is too large (312792 bytes). Limit: 120000 bytes.]
- .ai-bridge/ [not a file]
