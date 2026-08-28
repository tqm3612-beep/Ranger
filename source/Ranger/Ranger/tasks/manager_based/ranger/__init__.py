# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Ranger-Debug-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-Stand-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StandPPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-ShortGoalFlat-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-ShortGoalFlat-v1",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-ShortGoalFlat-v2",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV2PPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-ShortGoalFlat-v3",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatLimitedSuspensionPPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-ShortGoalFlatJoint-v3",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatLimitedJointPPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-ShortGoalFlat-v4",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV4EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatFrozenSuspensionWheelPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Template-Ranger-ShortGoalFlat-v5",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV5EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatFrozenSuspensionWheelPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Template-Ranger-ShortGoalFlat-v6",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV6EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatFrozenSuspensionWheelPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-ShortGoalFlat-v7",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV7PPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-ShortGoalFlat-v8",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV8PPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-ShortGoalFlat-v9",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV9PPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-ShortGoalFlat-v10",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatCRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-ShortGoalFlat-C-Recurrent",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatCRecurrentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatCRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-ShortGoalMemory-v1",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalMemoryV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatCRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-ShortGoalMemory-v2",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalMemoryV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatCRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Perception-P0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerPerceptionP0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PerceptionP0RecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Perception-P0-Neutral",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerPerceptionP0NeutralEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PerceptionP0RecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-ControlTransfer",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ControlTransferEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ControlTransferPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP0PPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05PPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2PPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2BootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap-B5",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2WheelControlBootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap-B5-Closeout",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2CloseoutBootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap-B6-RateResidual",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2HeadingRateResidualBootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap-B7-BalancedRateResidual",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2BalancedRateResidualBootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap-B8-FastRateResidual",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2FastBalancedRateResidualBootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap-B9-UnifiedWheelResidual",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2UnifiedWheelResidualBootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap-B11-TaperedCommonResidual",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2TaperedCommonResidualBootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap-B12-HeadingRateFeedback",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2HeadingRateEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2HeadingRateFeedbackBootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap-B13-FixedRolloutIntensive",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2HeadingRateEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2FixedRolloutIntensiveBootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap-B14-FixedRolloutFastLR",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2HeadingRateEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2FixedRolloutFastLRBootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Exit-V2-Bootstrap-B10-UnifiedWheelControl",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05ExitV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05ExitV2UnifiedWheelControlBootstrapPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Stop",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05StopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05StopPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-Obstacle-P05-Full",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2ObstacleP05FullEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2ObstacleP05FullPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-Memory-M0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2MemoryM0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2MemoryM0PPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-Memory-M1",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2MemoryM1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2MemoryM1PPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-Terrain-P0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2TerrainP0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2TerrainP0PPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-Terrain-P1",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2TerrainP1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2TerrainP1PPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Stage2-Terrain-P15-BrakeAssist",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2TerrainP15BrakeAssistEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2TerrainP15BrakeAssistPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-Stage2-Terrain-P15",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStage2TerrainP15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage2TerrainP15PPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Perception-Terrain-P0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerPerceptionTerrainP0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PerceptionP0RecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-Wave-Adaptation-PPO",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerPerceptionTerrainP0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaveAdaptationPPORunnerCfg",
    },
)


gym.register(
    id="Template-Ranger-ShortGoalFlat-C-Recurrent-TeacherPPO",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatCRecurrentEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatCRecurrentTeacherPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Template-Ranger-ShortGoalFlat-v11",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalFlatV11EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV11PPORunnerCfg",
    },
)
