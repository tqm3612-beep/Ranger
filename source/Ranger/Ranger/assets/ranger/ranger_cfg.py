# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the project-local Ranger robot asset."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

RANGER_PROJECT_ROOT = Path(__file__).resolve().parents[5]
RANGER_MODEL_DIR = RANGER_PROJECT_ROOT / "RANGER"
RANGER_URDF_PATH = RANGER_MODEL_DIR / "urdf" / "RANGER.urdf"
RANGER_USD_PATH = RANGER_MODEL_DIR / "usd" / "RANGER.usd"

if not RANGER_USD_PATH.is_file():
    raise FileNotFoundError(f"Ranger USD asset not found: {RANGER_USD_PATH}")

RANGER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(RANGER_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10000.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    soft_joint_pos_limit_factor=0.7,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            # Recomputed from the current URDF and wheel meshes so that, with
            # base_link spawned at z=0.7, the four wheel lowest points are
            # nearly coplanar at ground height.
            "g_lb": 0,
            "g_lf": 0,
            "g_rf": 0,
            "g_rb": 0,
            "w_lb": 0.0,
            "w_lf": 0.0,
            "w_rf": 0.0,
            "w_rb": 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    # Align with robot1-style implicit actuators:
    # legs use high-stiffness position-servo-like settings; wheels use zero-stiffness, high-damping velocity-like settings.
    actuators={
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=["g_lb", "g_lf", "g_rf", "g_rb"],
            effort_limit_sim=800.0,
            velocity_limit_sim=3.0,
            stiffness=1e6,
            damping=100000.0,
        ),
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=["w_lb", "w_lf", "w_rf", "w_rb"],
            effort_limit=130.0,
            velocity_limit=40.0,
            stiffness=0.0,
            damping=1000000.0,
        ),
    },
)
"""Configuration for the Ranger robot.

The robot model assets are stored under the repository-level ``RANGER``
directory. The runtime asset is ``RANGER/usd/RANGER.usd`` and is generated
from ``RANGER/urdf/RANGER.urdf``.
"""
