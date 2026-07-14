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
    id="Template-Ranger-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-Visual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerVisualEnvCfg",
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
    id="Template-Ranger-Stand-Visual-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerStandVisualEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-Forward-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerForwardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-SpeedCommand-Flat-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerSpeedCommandFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-GoalHeading-Flat-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerGoalHeadingFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-Forward-Visual-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerForwardVisualEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-TurnToTarget-Flat-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerTurnToTargetFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
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
    id="Template-Ranger-ShortGoalTurnFlat-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalTurnFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-ShortGoalTurnFreeHydraulicFlat-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalTurnFreeHydraulicFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-ShortGoalTurnFreeHydraulicFlat-v1",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerShortGoalTurnFreeHydraulicFlatV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-YawTurnSupportFlat-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerYawTurnSupportFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Ranger-YawRateCommandFlat-v0",
    entry_point=f"{__name__}.forward_debug_env:RangerForwardDebugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ranger_env_cfg:RangerYawRateCommandFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatPPORunnerCfg",
    },
)
