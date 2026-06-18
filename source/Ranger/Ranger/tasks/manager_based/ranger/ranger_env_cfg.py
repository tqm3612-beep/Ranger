# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, RayCasterCameraCfg, patterns
from isaaclab.utils import configclass

from Ranger.assets.ranger import RANGER_CFG

from . import mdp


##
# 场景
##


@configclass
class RangerSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with the Ranger robot."""

    # local ground plane, implemented as a thin static cuboid to avoid remote USD dependencies
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        collision_group=-1,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(100.0, 100.0, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
        ),
    )

    # robot
    robot: ArticulationCfg = RANGER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    mid360_lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mid360_link",
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-7.0, 52.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=10.0,
        ),
        max_distance=40.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=True,
    )

    avia_lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/avia_link",
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-38.6, 38.6),
            horizontal_fov_range=(-35.2, 35.2),
            horizontal_res=2.0,
        ),
        max_distance=50.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=True,
    )

    d435i_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435i_link",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=1.93,
            horizontal_aperture=3.80,
            width=12,
            height=10,
        ),
        max_distance=10.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=True,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    leg_hydraulic = mdp.HydraulicCylinderActionCfg(
        asset_name="robot",
        joint_names=["g_lb", "g_lf", "g_rf", "g_rb"],
        preserve_order=True,
        stroke_min=0.0, # 等效最小行程
        stroke_max=1.0, # 等效最大行程
        stroke_rate_limit=1.0, # 等效行程速率限制
        time_constant=0.08, # 一阶响应时间常数
        stroke_table=(0.0, 0.5, 1.0), # 查找表输入：虚拟液压缸行程
        joint_pos_table=(-1.0, 0.0, 1.0), # 查找表输出：轮-腿关节目标位置（弧度）
    )
    wheel_joint_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot", 
        joint_names=["w_.*"], 
        scale=20.0 # 轮关节速度缩放因子，RL输出乘以该因子后作为轮关节的速度目标
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        local_height_scan = ObsTerm(
            func=mdp.local_height_scan,
            params={
                "sensor_names": ("mid360_lidar", "avia_lidar"),
                "x_range": (0.0, 2.0),
                "y_range": (-0.6, 0.6),
                "resolution": 0.1,
                "invalid_height": 0.0,
            },
        )
        local_height_scan_valid_mask = ObsTerm(
            func=mdp.local_height_scan_valid_mask,
            params={
                "sensor_names": ("mid360_lidar", "avia_lidar"),
                "x_range": (0.0, 2.0),
                "y_range": (-0.6, 0.6),
                "resolution": 0.1,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.05, 0.05),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class RangerEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RangerSceneCfg = RangerSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (4.0, -4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        # sensor settings
        self.scene.mid360_lidar.update_period = self.decimation * self.sim.dt
        self.scene.avia_lidar.update_period = self.decimation * self.sim.dt
        self.scene.d435i_camera.update_period = self.decimation * self.sim.dt
