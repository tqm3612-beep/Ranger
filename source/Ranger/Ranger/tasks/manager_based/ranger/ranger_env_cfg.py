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
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, RayCasterCameraCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from Ranger.assets.ranger import RANGER_CFG

from . import mdp
from .terrain_cfg import SIMPLE_TERRAIN_CFG

GOAL_STATE_PARAMS = {
    # Stage-1 neutral input: goal_valid=0 and [sin, cos] = [0, 1].
    # Later stages can enable a real goal without changing the policy interface.
    "goal_x_body": 0.0,
    "goal_y_body": 0.0,
    "goal_range": 5.0,
    "goal_enabled": False,
}

LOCAL_NAVIGATION_MAP_PARAMS = {
    "sensor_names": ("mid360_lidar", "avia_lidar", "d435i_camera"),
    "x_range": (0.0, 2.0),
    "y_range": (-0.6, 0.6),
    "resolution": 0.1,
    "height_reference_x_range": (0.0, 0.4),
    "height_reference_y_range": (-0.3, 0.3),
    "step_threshold": 0.08,
    "slope_normalization": 0.6,
    "roughness_normalization": 0.05,
    "step_normalization": 0.15,
    "apply_noise": True,
    "height_noise_std": 0.01,
    "risk_noise_std": 0.02,
    "valid_dropout_prob": 0.02,
    "slope_weight": 0.4,
    "roughness_weight": 0.3,
    "step_weight": 0.3,
    "unknown_penalty": 1.0,
    # Stage-1 neutral map placeholder can be enabled later without changing observation shape.
    "use_neutral_map": False,
}

FLAT_TERRAIN_REWARD_MAP_PARAMS = {
    "sensor_names": LOCAL_NAVIGATION_MAP_PARAMS["sensor_names"],
    "x_range": LOCAL_NAVIGATION_MAP_PARAMS["x_range"],
    "y_range": LOCAL_NAVIGATION_MAP_PARAMS["y_range"],
    "resolution": LOCAL_NAVIGATION_MAP_PARAMS["resolution"],
    "step_threshold": LOCAL_NAVIGATION_MAP_PARAMS["step_threshold"],
    "height_reference_x_range": LOCAL_NAVIGATION_MAP_PARAMS["height_reference_x_range"],
    "height_reference_y_range": LOCAL_NAVIGATION_MAP_PARAMS["height_reference_y_range"],
    "slope_normalization": LOCAL_NAVIGATION_MAP_PARAMS["slope_normalization"],
    "roughness_normalization": LOCAL_NAVIGATION_MAP_PARAMS["roughness_normalization"],
    "step_normalization": LOCAL_NAVIGATION_MAP_PARAMS["step_normalization"],
    "slope_weight": LOCAL_NAVIGATION_MAP_PARAMS["slope_weight"],
    "roughness_weight": LOCAL_NAVIGATION_MAP_PARAMS["roughness_weight"],
    "step_weight": LOCAL_NAVIGATION_MAP_PARAMS["step_weight"],
    "height_range_weight": 0.2,
    "flatness_gain": 4.0,
}


def _navigation_map_grid_shape(resolution: float, x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[int, int]:
    num_x = int(round((x_range[1] - x_range[0]) / resolution)) + 1
    num_y = int(round((y_range[1] - y_range[0]) / resolution)) + 1
    return num_x, num_y


def _expected_policy_obs_dim() -> int:
    low_dim_terms = 45
    goal_dim = 6
    map_layers = 6
    num_x, num_y = _navigation_map_grid_shape(
        resolution=LOCAL_NAVIGATION_MAP_PARAMS["resolution"],
        x_range=LOCAL_NAVIGATION_MAP_PARAMS["x_range"],
        y_range=LOCAL_NAVIGATION_MAP_PARAMS["y_range"],
    )
    return low_dim_terms + goal_dim + map_layers * num_x * num_y


def _expected_action_dim() -> int:
    return 8


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
        prim_path="{ENV_REGEX_NS}/Robot",
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.46259, 0.0, -0.21177),
            rot=(0.241845, -0.664463, 0.664463, -0.241845),
            convention="ros",
        ),
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

    # Training-only privileged sensor for wheel-ground contact monitoring.
    # It is intentionally kept out of the policy observations to preserve sim-to-real compatibility.
    wheel_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/w_.*",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class RangerSimpleTerrainSceneCfg(RangerSceneCfg):
    """Scene configuration that swaps the flat cuboid for a low-difficulty generated terrain."""

    ground = None
    terrain = SIMPLE_TERRAIN_CFG


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # The first four policy outputs are normalized desired suspension-stroke
    # commands. On hardware these correspond to the four suspension
    # actuators' desired cylinder strokes. In simulation we map stroke_des
    # through stroke_table -> joint_pos_table and execute it with a fixed
    # impedance/PD effort controller on the equivalent g_* joints.
    leg_hydraulic = mdp.HydraulicActuatorActionCfg(
        asset_name="robot",
        joint_names=["g_lb", "g_lf", "g_rf", "g_rb"],
        preserve_order=True,
        stroke_min=0.0,
        stroke_max=1.0,
        stroke_rate_limit=1.0,
        time_constant=0.08,
        stroke_table=(0.0, 0.5, 1.0),
        joint_pos_table=(-1.0, 0.0, 1.0),
        max_effort=300.0,
        impedance_kp=250.0,
        impedance_kd=30.0,
    )
    wheel_motor_csv = mdp.WheelMotorCSVActionCfg(
        asset_name="robot",
        joint_names=["w_lb", "w_lf", "w_rf", "w_rb"],
        preserve_order=True,
        velocity_limit=20.0,
        acceleration_limit=80.0,
        command_time_constant=0.02,
        velocity_kp=10.0,
        velocity_damping=0.2,
        viscous_friction=0.05,
        effort_limit=100.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel_normalized,
            params={"scale": 2.0},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_normalized,
            params={"scale": 3.0},
            noise=Unoise(n_min=-0.04, n_max=0.04),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_normalized,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        leg_joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_normalized,
            params={"scale": 1.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["g_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        leg_joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel_normalized,
            params={"scale": 5.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["g_.*"])},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        wheel_joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel_normalized,
            params={"scale": 20.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["w_.*"])},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        hydraulic_stroke_state = ObsTerm(
            func=mdp.hydraulic_stroke_state_normalized,
            params={"action_name": "leg_hydraulic", "stroke_min": 0.0, "stroke_max": 1.0},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        hydraulic_effort_state = ObsTerm(
            func=mdp.hydraulic_effort_state_normalized,
            params={"action_name": "leg_hydraulic", "effort_limit": 300.0},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        wheel_velocity_target = ObsTerm(
            func=mdp.wheel_velocity_target_normalized,
            params={"action_name": "wheel_motor_csv", "velocity_limit": 20.0},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        wheel_torque_state = ObsTerm(
            func=mdp.wheel_torque_state_normalized,
            params={"action_name": "wheel_motor_csv", "effort_limit": 100.0},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        goal_state = ObsTerm(
            func=mdp.goal_state,
            params=GOAL_STATE_PARAMS,
        )
        last_action = ObsTerm(
            func=mdp.last_action_normalized,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # Feed the policy a fixed six-layer local navigation map. Stage-1
        # training can later swap in a neutral map without changing this shape.
        local_navigation_map = ObsTerm(
            func=mdp.local_navigation_map,
            params=LOCAL_NAVIGATION_MAP_PARAMS,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
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

    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    forward_progress = RewTerm(
        func=mdp.forward_velocity_reward,
        weight=3,
        params={"speed_scale": 1.0},
    )
    velocity_tracking = RewTerm(
        func=mdp.forward_velocity_tracking_exp,
        weight=0.0,
        params={"target_speed": 0.45, "std": 0.2},
    )
    overspeed = RewTerm(
        func=mdp.overspeed_l2,
        weight=0.0,
        params={"target_speed": 0.45},
    )
    underspeed = RewTerm(
        func=mdp.underspeed_l2,
        weight=0.0,
        params={"target_speed": 0.45},
    )
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    roll_angle = RewTerm(func=mdp.roll_angle_l2, weight=0.0)
    roll_angle_limit = RewTerm(
        func=mdp.roll_angle_limit_l2,
        weight=0.0,
        params={"allowed_roll_deg": 3.0},
    )
    pitch_angle = RewTerm(func=mdp.pitch_angle_l2, weight=0.0)
    pitch_angle_limit = RewTerm(
        func=mdp.pitch_angle_limit_l2,
        weight=0.0,
        params={"allowed_pitch_deg": 3.0},
    )
    base_height_low = RewTerm(
        func=mdp.base_height_below_target_l2,
        weight=0.0,
        params={"target_height": 0.42},
    )
    base_vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    base_roll_pitch_rate = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    lateral_velocity = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.5)
    leg_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["g_.*"])},
    )
    wheel_joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["w_.*"])},
    )
    leg_joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["g_.*"])},
    )
    joint_limit_margin = RewTerm(
        func=mdp.joint_limit_margin_penalty,
        weight=0.0,
        params={"margin_ratio": 0.15, "asset_cfg": SceneEntityCfg("robot", joint_names=["g_.*"])},
    )
    wheel_semantic_velocity_symmetry = RewTerm(
        func=mdp.wheel_semantic_velocity_symmetry_l2,
        weight=0.0,
    )
    front_rear_wheel_height_balance = RewTerm(
        func=mdp.front_rear_wheel_height_balance_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    left_right_wheel_height_balance = RewTerm(
        func=mdp.left_right_wheel_height_balance_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    front_rear_stroke_balance = RewTerm(
        func=mdp.front_rear_stroke_balance_l2,
        weight=0.0,
        params={**FLAT_TERRAIN_REWARD_MAP_PARAMS},
    )
    base_pitch_stroke_compensation = RewTerm(
        func=mdp.base_pitch_stroke_compensation_l2,
        weight=0.0,
        params={"k_pitch": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    left_right_stroke_balance = RewTerm(
        func=mdp.left_right_stroke_balance_l2,
        weight=0.0,
    )
    flat_stroke_nominal = RewTerm(
        func=mdp.flat_stroke_nominal_l2,
        weight=0.0,
        params={**FLAT_TERRAIN_REWARD_MAP_PARAMS, "stroke_nominal": 0.4},
    )
    flat_stroke_high = RewTerm(
        func=mdp.flat_stroke_high_l2,
        weight=0.0,
        params={**FLAT_TERRAIN_REWARD_MAP_PARAMS, "stroke_mean_limit": 0.55},
    )
    flat_base_clearance = RewTerm(
        func=mdp.flat_base_clearance_l2,
        weight=0.0,
        params={**FLAT_TERRAIN_REWARD_MAP_PARAMS, "asset_cfg": SceneEntityCfg("robot"), "clearance_nominal": 0.85},
    )
    flat_root_height = RewTerm(
        func=mdp.flat_root_height_l2,
        weight=0.0,
        params={**FLAT_TERRAIN_REWARD_MAP_PARAMS, "asset_cfg": SceneEntityCfg("robot"), "root_height_nominal": 0.85},
    )
    flat_attitude = RewTerm(
        func=mdp.flat_attitude_l2,
        weight=0.0,
        params={**FLAT_TERRAIN_REWARD_MAP_PARAMS, "asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.001)
    termination = RewTerm(func=mdp.is_terminated, weight=-10.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})
    root_height_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.1})


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
        self.scene.wheel_contact_forces.update_period = self.sim.dt
        print(f"[RangerEnvCfg] Expected policy observation shape: {_expected_policy_obs_dim()} (legacy 1410 -> current 1689)")
        print(f"[RangerEnvCfg] Expected action shape: {_expected_action_dim()} (4 leg + 4 wheel)")


@configclass
class RangerStandEnvCfg(RangerEnvCfg):
    """Stage-1 standing configuration focused on stable four-wheel grounding."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 4.0

        # Stage-1 uses the fixed neutral goal/map interface while we focus on posture stability.
        self.observations.policy.goal_state.params["goal_enabled"] = False
        self.observations.policy.local_navigation_map.params["use_neutral_map"] = True

        # Keep resets close to the nominal support pose so the policy can first learn to settle.
        self.events.reset_robot_joints.params["position_range"] = (-0.005, 0.005)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.01, 0.01)

        # Relax early failure slightly so the robot has time to stabilize after touchdown.
        self.terminations.bad_orientation.params["limit_angle"] = 1.4
        self.terminations.root_height_low.params["minimum_height"] = 0.05

        # Turn off locomotion incentives and focus on stable support/contact quality first.
        self.rewards.forward_progress.weight = 0.0
        self.rewards.lateral_velocity.weight = -0.2
        self.rewards.base_vertical_velocity.weight = -1.0
        self.rewards.base_roll_pitch_rate.weight = -0.5
        self.rewards.upright.weight = -3.0
        self.rewards.wheel_joint_velocity.weight = -0.002
        self.rewards.action_rate.weight = -0.02
        self.rewards.action_magnitude.weight = -0.002

        # Reward four-wheel contact coverage and balanced support forces during settling.
        self.rewards.wheel_contact_count = RewTerm(
            func=mdp.wheel_contact_count_reward,
            weight=2.0,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "threshold": 1.0},
        )
        self.rewards.wheel_contact_balance = RewTerm(
            func=mdp.wheel_contact_force_balance_reward,
            weight=1.0,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "threshold": 1.0},
        )


@configclass
class RangerVisualEnvCfg(RangerEnvCfg):
    """Smaller training scene tuned for interactive visualization."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Keep only a handful of environments so the GUI stays readable during training.
        self.scene.num_envs = 4
        self.scene.env_spacing = 8.0
        # Pull the camera back slightly so all visible robots fit in the initial view.
        self.viewer.eye = (5.0, 30.0, 5.0)
        self.viewer.lookat = (5.0, 5.0, 0.8)


@configclass
class RangerStandVisualEnvCfg(RangerStandEnvCfg):
    """Standing-stage scene tuned for interactive visualization."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 8.0
        self.viewer.eye = (8.0, -8.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)


@configclass
class RangerForwardEnvCfg(RangerStandEnvCfg):
    """Stage-2 forward configuration with a fixed goal and neutral navigation map."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep the policy interface fixed while enabling a simple forward objective.
        self.observations.policy.goal_state.params["goal_enabled"] = True
        self.observations.policy.goal_state.params["goal_x_body"] = 3.0
        self.observations.policy.goal_state.params["goal_y_body"] = 0.0
        self.observations.policy.local_navigation_map.params["use_neutral_map"] = True

        # Re-enable only the simple forward-progress incentive for the next stage.
        self.rewards.forward_progress.weight = 2.0
        self.rewards.wheel_joint_velocity.weight = 0.0
        self.rewards.wheel_contact_count.weight = 1.0
        self.rewards.wheel_contact_balance.weight = 0.5


@configclass
class RangerForwardVisualEnvCfg(RangerForwardEnvCfg):
    """Forward-stage scene tuned for interactive visualization."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 8.0
        self.viewer.eye = (8.0, -8.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)


@configclass
class RangerSimpleTerrainEnvCfg(RangerForwardEnvCfg):
    """Stage-3 forward locomotion on mild terrain with neutral local map input."""

    scene: RangerSimpleTerrainSceneCfg = RangerSimpleTerrainSceneCfg(num_envs=4096, env_spacing=4.0)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 10.0
        self.observations.policy.local_navigation_map.params["use_neutral_map"] = True
        self.actions.leg_hydraulic.stroke_min = 0.25
        self.actions.leg_hydraulic.stroke_max = 0.70

        self.terminations.bad_orientation.params["limit_angle"] = 0.85
        self.terminations.root_height_low.params["minimum_height"] = 0.4

        self.rewards.forward_progress.weight = 0.0
        self.rewards.velocity_tracking.weight = 2.5
        self.rewards.velocity_tracking.params["target_speed"] = 1.0
        self.rewards.overspeed.weight = -2.0
        self.rewards.overspeed.params["target_speed"] = 1.0
        self.rewards.underspeed.weight = -2.0
        self.rewards.underspeed.params["target_speed"] = 1.0
        self.rewards.roll_angle.weight = -3.0
        self.rewards.roll_angle_limit.weight = -20.0
        self.rewards.pitch_angle.weight = -5.0
        self.rewards.pitch_angle_limit.weight = -25.0
        self.rewards.base_height_low.weight = -5.0
        self.rewards.base_height_low.params["target_height"] = 0.75
        self.rewards.joint_limit_margin.weight = -1.0
        self.rewards.wheel_semantic_velocity_symmetry.weight = -0.03
        self.rewards.front_rear_wheel_height_balance.weight = 0.0
        self.rewards.left_right_wheel_height_balance.weight = -3.0
        self.rewards.front_rear_stroke_balance.weight = -3.0
        self.rewards.base_pitch_stroke_compensation.weight = -3.0
        self.rewards.base_pitch_stroke_compensation.params["k_pitch"] = -1.0
        self.rewards.left_right_stroke_balance.weight = -1.0
        self.rewards.flat_stroke_nominal.weight = -6.0
        self.rewards.flat_stroke_high.weight = -10.0
        self.rewards.flat_base_clearance.weight = 0.0
        self.rewards.flat_root_height.weight = -6.0
        self.rewards.flat_attitude.weight = -2.0
        self.rewards.action_rate.weight = -0.03


@configclass
class RangerSimpleTerrainVisualEnvCfg(RangerSimpleTerrainEnvCfg):
    """Simple-terrain forward scene tuned for interactive visualization."""

    scene: RangerSimpleTerrainSceneCfg = RangerSimpleTerrainSceneCfg(num_envs=4, env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 8.0
        self.viewer.eye = (-10.0, 20.0, 5.0)
        self.viewer.lookat = (-10.0, 5.0, 0.0)


@configclass
class RangerMapPostureEnvCfg(RangerSimpleTerrainEnvCfg):
    """Stage-4 posture task that enables the real local map while keeping the stage-3 interface."""

    def __post_init__(self) -> None:
        super().__post_init__()

        local_map_params = self.observations.policy.local_navigation_map.params
        local_map_params["use_neutral_map"] = False
        local_map_params["apply_noise"] = False
        local_map_params["valid_dropout_prob"] = 0.0
        local_map_params["height_noise_std"] = 0.0
        local_map_params["risk_noise_std"] = 0.0

        self.rewards.velocity_tracking.weight = 2.5
        self.rewards.velocity_tracking.params["target_speed"] = 1.0
        self.rewards.overspeed.weight = -2.0
        self.rewards.overspeed.params["target_speed"] = 1.0
        self.rewards.underspeed.weight = -2.0
        self.rewards.underspeed.params["target_speed"] = 1.0

        self.rewards.roll_angle.weight = -4.0
        self.rewards.pitch_angle.weight = -6.0
        self.rewards.roll_angle_limit.weight = -25.0
        self.rewards.pitch_angle_limit.weight = -30.0
        self.rewards.base_vertical_velocity.weight = -1.0
        self.rewards.base_roll_pitch_rate.weight = -0.3
        self.rewards.joint_limit_margin.weight = -1.5
        self.rewards.action_rate.weight = -0.04


@configclass
class RangerMapPostureVisualEnvCfg(RangerMapPostureEnvCfg):
    """Stage-4 posture task tuned for interactive visualization."""

    scene: RangerSimpleTerrainSceneCfg = RangerSimpleTerrainSceneCfg(num_envs=4, env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 8.0
        self.viewer.eye = (-10.0, 20.0, 5.0)
        self.viewer.lookat = (-10.0, 0.0, 0.8)
