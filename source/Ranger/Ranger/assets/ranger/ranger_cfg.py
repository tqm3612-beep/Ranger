# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the project-local Ranger robot asset."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

RANGER_ASSET_DIR = Path(__file__).resolve().parent
RANGER_URDF_PATH = RANGER_ASSET_DIR / "urdf" / "ranger.urdf"
RANGER_USD_PATH = RANGER_ASSET_DIR / "usd" / "ranger.usd"

RANGER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(RANGER_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),

    # Actuators are configured to stay compatible with effort-based motor and hydraulic action terms.
    actuators={
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=["g_.*"],
            effort_limit_sim=300.0, # 最大等效力矩
            velocity_limit_sim=5.0, # 最大关节角速度
            stiffness=0.0,
            damping=2.0,
        ),
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=["w_.*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=2.0,
        ),
    },
)
"""Configuration for the Ranger robot.

The USD is expected at ``assets/ranger/usd/ranger.usd``. Generate it from
``assets/ranger/urdf/ranger.urdf`` before creating the environment.
"""
