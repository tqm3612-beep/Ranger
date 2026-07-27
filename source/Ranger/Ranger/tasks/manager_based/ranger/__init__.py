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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShortGoalFlatV10PPORunnerCfg",
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
