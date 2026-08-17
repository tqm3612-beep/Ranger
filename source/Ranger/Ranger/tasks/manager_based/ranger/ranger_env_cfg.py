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

COMMAND_OBS_PARAMS = {
    "command_mode": "zero",
    "goal_source": "none",
    "goal_x_body": 0.0,
    "goal_y_body": 0.0,
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
    "use_neutral_map": False,
}


def _navigation_map_grid_shape(resolution: float, x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[int, int]:
    num_x = int(round((x_range[1] - x_range[0]) / resolution)) + 1
    num_y = int(round((y_range[1] - y_range[0]) / resolution)) + 1
    return num_x, num_y


def _expected_policy_state_obs_dim() -> int:
    return 42


def _expected_policy_map_obs_dim() -> int:
    map_layers = 8
    num_x, num_y = _navigation_map_grid_shape(
        resolution=LOCAL_NAVIGATION_MAP_PARAMS["resolution"],
        x_range=LOCAL_NAVIGATION_MAP_PARAMS["x_range"],
        y_range=LOCAL_NAVIGATION_MAP_PARAMS["y_range"],
    )
    return map_layers * num_x * num_y


def _expected_policy_obs_dim() -> int:
    return _expected_policy_state_obs_dim() + _expected_policy_map_obs_dim()


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
            size=(300.0, 300.0, 0.02),
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

    # Debug-only privileged sensor for auditing whether non-wheel collision is carrying load.
    # This is intentionally excluded from rewards and observations.
    all_body_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        debug_vis=False,
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
        control_mode="velocity",
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
    class PolicyStateCfg(ObsGroup):
        """Low-dimensional actor state: proprioception + command."""

        # observation terms (order preserved)
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
        suspension_stroke = ObsTerm(
            func=mdp.suspension_stroke_state,
            params={"action_name": "leg_hydraulic"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        suspension_stroke_rate = ObsTerm(
            func=mdp.suspension_stroke_rate_state,
            params={"action_name": "leg_hydraulic", "clip": 5.0},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        command_state = ObsTerm(
            func=mdp.command_observation,
            params=COMMAND_OBS_PARAMS,
        )
        last_action = ObsTerm(
            func=mdp.last_action_normalized,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyMapCfg(ObsGroup):
        """Flattened 8-channel local terrain map for the CNN branch."""

        local_navigation_map = ObsTerm(
            func=mdp.local_navigation_map,
            params=LOCAL_NAVIGATION_MAP_PARAMS,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticPrivilegedCfg(ObsGroup):
        """Compact privileged critic-only features for asymmetric actor-critic."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel_normalized,
            params={"scale": 2.0},
        )
        wheel_contact_force = ObsTerm(
            func=mdp.wheel_contact_force_over_weight,
            params={"sensor_name": "wheel_contact_forces", "asset_cfg": SceneEntityCfg("robot")},
        )
        wheel_contact_bool = ObsTerm(
            func=mdp.wheel_contact_bool,
            params={"sensor_name": "wheel_contact_forces", "threshold": 1.0},
        )
        root_height = ObsTerm(
            func=mdp.root_height_state,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy_state: PolicyStateCfg = PolicyStateCfg()
    policy_map: PolicyMapCfg = PolicyMapCfg()
    critic_privileged: CriticPrivilegedCfg = CriticPrivilegedCfg()


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
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
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
    # Optional environment-level suspension controls.
    # ``fixed_suspension_action`` overrides all four hydraulic actions when set.
    fixed_suspension_action: float | None = None
    suspension_action_scale: float = 1.0
    stop_phase_suspension_action: float | None = None
    stop_phase_suspension_mode: str = "legacy"
    stop_phase_wheel_override_enabled: bool = True

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
        self.scene.all_body_contact_forces.update_period = self.sim.dt
        self.stand_training_task = False
        self.enable_reset_settle = False
        self.reset_settle_steps = 0
        self.reset_settle_leg_action = -0.34
        self.enable_initial_stroke_randomization = False
        self.initial_stroke_noise_range = 0.0
        self.initial_stroke_range = (0.50, 0.50)
        self.initial_stroke_shared_across_legs = True
        self.sync_reset_root_height_to_initial_stroke = False
        self.initial_stroke_root_height_reference = 0.884
        self.initial_stroke_root_height_nominal_stroke = 0.50
        self.initial_stroke_root_height_slope = 0.38
        self.terrain_like_reset_enabled = False
        self.terrain_like_reset_base_stroke = 0.50
        self.terrain_like_reset_common_offset_range = (-0.02, 0.04)
        self.terrain_like_reset_pattern_amplitude_range = (0.03, 0.06)
        self.terrain_like_reset_stroke_clamp_range = (0.44, 0.58)
        self.terrain_like_reset_root_height_margin = 0.015
        self.terrain_like_reset_slope_deg_range = (3.0, 5.0)
        self.terrain_like_reset_twist_deg_range = (2.0, 5.0)
        self.terrain_like_reset_yaw_deg_range = (-3.0, 3.0)
        self.terrain_like_reset_linear_xy_velocity_range = (-0.10, 0.10)
        self.terrain_like_reset_angular_velocity_range = (-0.2, 0.2)
        self.debug_full_stdout_metrics = False
        print(
            "[RangerEnvCfg] Expected actor observation shape: "
            f"state={_expected_policy_state_obs_dim()} map={_expected_policy_map_obs_dim()} total={_expected_policy_obs_dim()}"
        )
        print(f"[RangerEnvCfg] Expected action shape: {_expected_action_dim()} (4 leg + 4 wheel)")


@configclass
class RangerStandEnvCfg(RangerEnvCfg):
    """Stage-1 standing configuration focused on stable four-wheel grounding."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 4.0
        self.stand_training_task = True
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.884)
        self.actions.leg_hydraulic.joint_pos_table = (-0.5, 0.0, 0.5)
        self.actions.leg_hydraulic.joint_target_sign = (-1.0, 1.0, 1.0, -1.0)

        # Stage-1 uses the fixed neutral goal/map interface while we focus on posture stability.
        self.observations.policy_state.command_state.params["command_mode"] = "zero"
        self.observations.policy_state.command_state.params["goal_source"] = "none"
        self.observations.policy_map.local_navigation_map.params["use_neutral_map"] = True

        # Keep resets close to the nominal support pose so the policy can first learn to settle.
        self.events.reset_robot_joints.params["position_range"] = (-0.005, 0.005)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.enable_initial_stroke_randomization = True
        self.initial_stroke_noise_range = 0.0
        self.initial_stroke_range = (0.47, 0.55)
        self.initial_stroke_shared_across_legs = True
        self.sync_reset_root_height_to_initial_stroke = True
        self.initial_stroke_root_height_reference = 0.884
        self.initial_stroke_root_height_nominal_stroke = 0.50
        self.initial_stroke_root_height_slope = 0.38
        self.terrain_like_reset_enabled = True
        self.terrain_like_reset_base_stroke = 0.50
        self.terrain_like_reset_common_offset_range = (-0.02, 0.04)
        self.terrain_like_reset_pattern_amplitude_range = (0.03, 0.06)
        self.terrain_like_reset_stroke_clamp_range = (0.44, 0.58)
        self.terrain_like_reset_root_height_margin = 0.015
        self.terrain_like_reset_slope_deg_range = (3.0, 5.0)
        self.terrain_like_reset_twist_deg_range = (2.0, 5.0)
        self.terrain_like_reset_yaw_deg_range = (-3.0, 3.0)
        self.terrain_like_reset_linear_xy_velocity_range = (-0.10, 0.10)
        self.terrain_like_reset_angular_velocity_range = (-0.2, 0.2)

        # Keep a realistic healthy stand envelope; low-slung postures should not count as success.
        self.terminations.bad_orientation.params["limit_angle"] = 1.4
        self.terminations.root_height_low.func = mdp.root_height_below_minimum_with_grace
        self.terminations.root_height_low.params["minimum_height"] = 0.65
        self.terminations.root_height_low.params["grace_time_s"] = 0.5
        self.terminations.root_height_low.params["asset_cfg"] = SceneEntityCfg("robot")

        # Turn off locomotion incentives and focus on stable support/contact quality first.
        self.rewards.forward_progress.weight = 0.0
        self.rewards.lateral_velocity.weight = -0.2
        self.rewards.base_xy_velocity = RewTerm(
            func=mdp.base_xy_speed_l2,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.base_vertical_velocity.weight = -1.0
        self.rewards.base_roll_pitch_rate.weight = -0.5
        self.rewards.base_yaw_rate = RewTerm(
            func=mdp.base_yaw_rate_l2,
            weight=-0.3,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.yaw_drift_from_reset = RewTerm(
            func=mdp.stand_yaw_drift_abs,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.upright.weight = -3.0
        self.rewards.wheel_joint_velocity.weight = -0.002
        self.rewards.action_rate.weight = -0.04
        self.rewards.action_magnitude.weight = -0.002
        self.rewards.roll_angle = RewTerm(func=mdp.roll_angle_l2, weight=-1.0)
        self.rewards.pitch_angle = RewTerm(func=mdp.pitch_angle_l2, weight=-1.2)
        self.rewards.root_height_tracking = RewTerm(
            func=mdp.root_height_tracking_exp,
            weight=0.5,
            params={"target_height": 0.875, "std_sq": 0.01, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.root_height_band = RewTerm(
            func=mdp.root_height_band_piecewise,
            weight=2.0,
            params={
                "peak_min": 0.865,
                "peak_max": 0.890,
                "healthy_min": 0.84,
                "healthy_max": 0.91,
                "low_floor": 0.65,
                "high_penalty_scale": 6.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.base_height_low = RewTerm(
            func=mdp.base_height_below_target_l2,
            weight=-8.0,
            params={"target_height": 0.70, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.base_height_high = RewTerm(
            func=mdp.base_height_above_target_l1,
            weight=-10.0,
            params={"target_height": 0.91, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.actual_stroke_nominal = RewTerm(
            func=mdp.actual_stroke_nominal_l2,
            weight=-12.0,
            params={"stroke_nominal": 0.515, "action_name": "leg_hydraulic"},
        )
        self.rewards.actual_stroke_soft_limit = RewTerm(
            func=mdp.actual_stroke_soft_limit_penalty,
            weight=-4.0,
            params={"limit": 0.45, "action_name": "leg_hydraulic"},
        )
        self.rewards.stroke_range = RewTerm(
            func=mdp.stroke_range_penalty,
            weight=-2.0,
            params={"action_name": "leg_hydraulic"},
        )
        self.rewards.stroke_diagonal_balance = RewTerm(
            func=mdp.stroke_diagonal_balance_penalty,
            weight=-1.0,
            params={"action_name": "leg_hydraulic"},
        )
        self.rewards.hydraulic_action_magnitude = RewTerm(
            func=mdp.hydraulic_action_magnitude_l1,
            weight=-0.20,
            params={"action_name": "leg_hydraulic"},
        )
        self.rewards.hydraulic_action_range = RewTerm(
            func=mdp.hydraulic_action_range_penalty,
            weight=0.0,
            params={"action_name": "leg_hydraulic"},
        )
        self.rewards.low_stroke_negative_hydraulic_action = RewTerm(
            func=mdp.low_stroke_negative_hydraulic_action_penalty,
            weight=-3.0,
            params={
                "stroke_threshold": 0.35,
                "stroke_margin": 0.10,
                "action_name": "leg_hydraulic",
            },
        )
        self.rewards.height_gated_low_stroke_negative_hydraulic_action = RewTerm(
            func=mdp.height_gated_low_stroke_negative_hydraulic_action_penalty,
            weight=-5.0,
            params={
                "height_threshold": 0.72,
                "height_margin": 0.07,
                "stroke_threshold": 0.35,
                "stroke_margin": 0.10,
                "asset_cfg": SceneEntityCfg("robot"),
                "action_name": "leg_hydraulic",
            },
        )
        self.rewards.high_height_low_stroke_negative_hydraulic_action = RewTerm(
            func=mdp.high_height_low_stroke_negative_hydraulic_action_penalty,
            weight=-4.0,
            params={
                "height_threshold": 0.91,
                "height_margin": 0.10,
                "stroke_threshold": 0.45,
                "stroke_margin": 0.10,
                "asset_cfg": SceneEntityCfg("robot"),
                "action_name": "leg_hydraulic",
            },
        )
        self.rewards.height_low_high_stroke_positive_hydraulic_action = RewTerm(
            func=mdp.height_low_high_stroke_positive_hydraulic_action_penalty,
            weight=0.0,
            params={
                "height_threshold": 0.72,
                "height_margin": 0.07,
                "stroke_high_threshold": 0.55,
                "stroke_margin": 0.10,
                "asset_cfg": SceneEntityCfg("robot"),
                "action_name": "leg_hydraulic",
            },
        )
        self.rewards.stroke_away_from_nominal_hydraulic_action = RewTerm(
            func=mdp.stroke_away_from_nominal_hydraulic_action_penalty,
            weight=-4.0,
            params={
                "nominal_stroke": 0.515,
                "action_name": "leg_hydraulic",
            },
        )
        self.rewards.front_rear_stroke_balance = RewTerm(
            func=mdp.front_rear_stroke_balance_penalty,
            weight=-3.0,
            params={"action_name": "leg_hydraulic"},
        )
        self.rewards.front_rear_stroke_split_action = RewTerm(
            func=mdp.front_rear_stroke_split_action_penalty,
            weight=0.0,
            params={
                "diff_threshold": 0.10,
                "diff_margin": 0.20,
                "action_name": "leg_hydraulic",
            },
        )

        # Reward four-wheel contact coverage and balanced support forces during settling.
        self.rewards.wheel_contact_count = RewTerm(
            func=mdp.wheel_contact_count_reward,
            weight=1.5,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "threshold": 1.0},
        )
        self.rewards.wheel_contact_stability = RewTerm(
            func=mdp.wheel_all_contact_reward,
            weight=1.0,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "threshold": 1.0},
        )
        self.rewards.wheel_contact_balance = RewTerm(
            func=mdp.wheel_contact_force_balance_reward,
            weight=0.5,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "threshold": 1.0},
        )
        self.rewards.contact_force_diag_balance = RewTerm(
            func=mdp.contact_force_diag_balance_penalty,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"])},
        )
        self.rewards.contact_force_left_right_balance = RewTerm(
            func=mdp.contact_force_left_right_balance_penalty,
            weight=-0.4,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"])},
        )
        self.rewards.contact_force_range_balance = RewTerm(
            func=mdp.contact_force_range_balance_penalty,
            weight=-0.3,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"])},
        )
        self.rewards.contact_force_min_support = RewTerm(
            func=mdp.contact_force_min_support_penalty,
            weight=-0.3,
            params={
                "min_force_threshold": 50.0,
                "sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
            },
        )
        self.rewards.contact_force_min_ratio_support = RewTerm(
            func=mdp.contact_force_min_ratio_support_penalty,
            weight=-0.75,
            params={
                "min_ratio_threshold": 0.10,
                "sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
            },
        )
        self.rewards.low_contact_support = RewTerm(
            func=mdp.contact_force_min_support_penalty,
            weight=-0.2,
            params={
                "min_force_threshold": 80.0,
                "sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
            },
        )
        self.rewards.low_contact_ratio_support = RewTerm(
            func=mdp.contact_force_min_ratio_support_penalty,
            weight=-0.6,
            params={
                "min_ratio_threshold": 0.15,
                "sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
            },
        )


@configclass
class _RangerShortGoalSharedEnvCfg(RangerStandEnvCfg):
    """Shared short-goal observation, reset, safety, and reward scaffolding."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.stand_training_task = False
        self.enable_initial_stroke_randomization = False
        self.episode_length_s = 5.0
        self.scene.ground.spawn.size = (300.0, 300.0, 0.02)

        # Keep actor/critic observation shapes unchanged while swapping in a short-range dynamic goal.
        self.observations.policy_state.command_state.func = mdp.command_observation
        self.observations.policy_state.command_state.params = {
            "command_mode": "zero",
            "goal_source": "short_goal",
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.observations.policy_map.local_navigation_map.params["use_neutral_map"] = True

        self.events.reset_short_goal_target = EventTerm(
            func=mdp.reset_short_goal_target,
            mode="reset",
            params={
                "distance_range": (0.5, 2.0),
                "heading_range": (-0.7853981633974483, 0.7853981633974483),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.terminations.bad_orientation.params["limit_angle"] = 1.2
        self.terminations.root_height_low.params["minimum_height"] = 0.30
        self.terminations.goal_reached = DoneTerm(
            func=mdp.short_goal_reached,
            params={"success_distance": 0.25, "asset_cfg": SceneEntityCfg("robot")},
        )

        # Preserve only the stand safety constraints; do not leave positive stand rewards
        # that let the policy score by standing still near the reset pose.
        self.rewards.alive.weight = 0.0
        self.rewards.forward_progress.weight = 0.0
        self.rewards.lateral_velocity.weight = 0.0
        self.rewards.wheel_joint_velocity.weight = 0.0
        self.rewards.base_xy_velocity.weight = 0.0
        self.rewards.base_yaw_rate.weight = 0.0
        self.rewards.yaw_drift_from_reset.weight = 0.0
        self.rewards.root_height_tracking.weight = 0.0
        self.rewards.root_height_band.weight = 0.0
        self.rewards.wheel_contact_count.weight = 0.0
        self.rewards.wheel_contact_stability.weight = 0.0
        self.rewards.wheel_contact_balance.weight = 0.0
        self.rewards.action_rate.weight = -0.02
        self.rewards.action_magnitude.weight = -0.001
        self.rewards.termination.weight = -20.0

        self.rewards.progress_to_goal = RewTerm(
            func=mdp.short_goal_progress_reward,
            weight=20.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.goal_velocity = RewTerm(
            func=mdp.short_goal_velocity_towards_target,
            weight=4.0,
            params={"max_velocity": 0.8, "min_reward": -1.0, "max_reward": 1.5, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.goal_success = RewTerm(
            func=mdp.short_goal_success_reward,
            weight=12.0,
            params={"success_distance": 0.25, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.near_goal_stop = RewTerm(
            func=mdp.short_goal_near_stop_penalty,
            weight=-0.8,
            params={"stop_distance": 0.5, "yaw_weight": 0.5, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.heading_alignment = RewTerm(
            func=mdp.short_goal_heading_alignment,
            weight=0.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class _RangerShortGoalSideTargetEnvCfg(_RangerShortGoalSharedEnvCfg):
    """Shared side-target distribution and heading-gated short-goal shaping."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.wheel_motor_csv.velocity_limit = 80.0
        self.events.reset_short_goal_target = EventTerm(
            func=mdp.reset_short_goal_turn_target,
            mode="reset",
            params={
                "distance_range": (1.0, 1.5),
                "left_heading_range_deg": (25.0, 45.0),
                "right_heading_range_deg": (-45.0, -25.0),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.progress_to_goal = RewTerm(
            func=mdp.short_goal_progress_reward_heading_gated,
            weight=20.0,
            params={"heading_error_threshold": 0.35, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.goal_velocity = RewTerm(
            func=mdp.short_goal_velocity_towards_target_heading_gated,
            weight=4.0,
            params={
                "max_velocity": 0.8,
                "min_reward": -1.0,
                "max_reward": 1.5,
                "heading_error_threshold": 0.35,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.goal_success = RewTerm(
            func=mdp.short_goal_success_reward,
            weight=12.0,
            params={"success_distance": 0.25, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.near_goal_stop = RewTerm(
            func=mdp.short_goal_near_stop_penalty,
            weight=-0.8,
            params={"stop_distance": 0.5, "yaw_weight": 0.5, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.heading_error_reduction = RewTerm(
            func=mdp.short_goal_heading_error_reduction,
            weight=6.0,
            params={"min_progress": -0.5, "max_progress": 0.5, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.turn_toward_goal = RewTerm(
            func=mdp.short_goal_turn_toward_goal,
            weight=2.0,
            params={"min_reward": -1.0, "max_reward": 1.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.hydraulic_action_magnitude = RewTerm(
            func=mdp.hydraulic_action_magnitude_l1,
            weight=-0.20,
            params={"action_name": "leg_hydraulic"},
        )
        self.rewards.stroke_range = RewTerm(
            func=mdp.stroke_range_penalty,
            weight=-2.0,
            params={"action_name": "leg_hydraulic"},
        )
        self.rewards.stroke_diagonal_balance = RewTerm(
            func=mdp.stroke_diagonal_balance_penalty,
            weight=-1.0,
            params={"action_name": "leg_hydraulic"},
        )
        self.rewards.contact_force_diag_balance = RewTerm(
            func=mdp.contact_force_diag_balance_penalty,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"])},
        )
        self.rewards.contact_force_left_right_balance = RewTerm(
            func=mdp.contact_force_left_right_balance_penalty,
            weight=-0.4,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"])},
        )
        self.rewards.low_contact_ratio_support = RewTerm(
            func=mdp.contact_force_min_ratio_support_penalty,
            weight=-0.6,
            params={
                "min_ratio_threshold": 0.15,
                "sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]),
            },
        )


@configclass
class RangerShortGoalFlatEnvCfg(_RangerShortGoalSideTargetEnvCfg):
    """Side-goal heading task with policy-controlled suspension and wheel-differential priors."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.wheel_motor_csv.velocity_limit = 80.0
        self.observations.policy_state.wheel_joint_vel_rel.params["scale"] = 80.0

        self.events.reset_short_goal_target = EventTerm(
            func=mdp.reset_short_goal_turn_target,
            mode="reset",
            params={
                "distance_range": (1.0, 1.5),
                "left_heading_range_deg": (25.0, 45.0),
                "right_heading_range_deg": (-45.0, -25.0),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.progress_to_goal.weight = 0.0
        self.rewards.goal_velocity.weight = 0.0
        self.rewards.goal_success.weight = 0.0
        self.rewards.near_goal_stop.weight = 0.0
        self.rewards.heading_error_reduction.weight = 9.0
        self.rewards.turn_toward_goal.weight = 4.0

        self.rewards.short_goal_wheel_diff_prior = RewTerm(
            func=mdp.short_goal_wheel_diff_prior_l1,
            weight=-1.0,
            params={"turn_gain": 40.0, "action_name": "wheel_motor_csv", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.short_goal_forward_common_mode = RewTerm(
            func=mdp.short_goal_forward_common_mode_penalty,
            weight=-0.20,
            params={
                "heading_error_threshold": 0.35,
                "action_name": "wheel_motor_csv",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.hydraulic_action_magnitude.weight = -0.05
        self.rewards.hydraulic_action_rate = RewTerm(
            func=mdp.hydraulic_action_rate_l1,
            weight=-0.03,
        )
        self.rewards.stroke_range.weight = -0.5
        self.rewards.stroke_diagonal_balance.weight = -0.25
        self.rewards.actual_stroke_nominal.weight = 0.0
        self.rewards.actual_stroke_soft_limit.weight = -1.0
        self.rewards.low_stroke_negative_hydraulic_action.weight = 0.0
        self.rewards.height_gated_low_stroke_negative_hydraulic_action.weight = 0.0
        self.rewards.high_height_low_stroke_negative_hydraulic_action.weight = 0.0
        self.rewards.stroke_away_from_nominal_hydraulic_action.weight = 0.0
        self.rewards.front_rear_stroke_balance.weight = 0.0

        self.rewards.wheel_contact_count.weight = 0.0
        self.rewards.wheel_contact_stability.weight = 0.0
        self.rewards.wheel_contact_balance.weight = 0.0
        if hasattr(self.rewards, "wheel_all_contact"):
            self.rewards.wheel_all_contact.weight = 0.0
        self.rewards.contact_force_diag_balance.weight = -0.15
        self.rewards.contact_force_left_right_balance.weight = 0.0
        self.rewards.contact_force_range_balance.weight = 0.0
        self.rewards.contact_force_min_support.weight = 0.0
        self.rewards.contact_force_min_ratio_support.weight = 0.0
        self.rewards.low_contact_support.weight = 0.0
        self.rewards.low_contact_ratio_support.weight = -0.15
        self.rewards.unloaded_wheel_spin = RewTerm(
            func=mdp.unloaded_wheel_spin_penalty,
            weight=-0.003,
            params={
                "force_threshold": 20.0,
                "sensor_cfg": SceneEntityCfg(
                    "wheel_contact_forces",
                    body_names=["w_lf", "w_lb", "w_rf", "w_rb"],
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["w_lf", "w_lb", "w_rf", "w_rb"],
                ),
            },
        )

        for reward_name in (
            "yaw_rate_tracking",
            "yaw_rate_command_tracking",
            "yaw_rate_error_penalty",
            "turn_direction",
            "wheel_turn_difference_error",
            "wheel_command_tracking",
            "wheel_command_error",
        ):
            if hasattr(self.rewards, reward_name):
                getattr(self.rewards, reward_name).weight = 0.0
        self.rewards.forward_velocity_during_turn = RewTerm(
            func=mdp.short_goal_forward_velocity_during_turn_penalty,
            weight=0.0,
            params={"free_speed": 0.12, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.signed_yaw_rate_tracking = RewTerm(
            func=mdp.short_goal_signed_yaw_rate_tracking,
            weight=0.0,
            params={"target_yaw_rate": 0.30, "heading_deadband": 0.10, "sigma": 0.25, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.too_small_yaw_rate_when_error_large = RewTerm(
            func=mdp.short_goal_too_small_yaw_rate_when_error_large_penalty,
            weight=0.0,
            params={"min_yaw_rate": 0.10, "heading_threshold": 0.25, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.wrong_direction_yaw = RewTerm(
            func=mdp.short_goal_wrong_direction_yaw_penalty,
            weight=0.0,
            params={"heading_deadband": 0.10, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.wrong_direction_yaw_command = RewTerm(
            func=mdp.wrong_direction_yaw_command_penalty,
            weight=0.0,
            params={"active_threshold": 0.02, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.too_small_yaw_rate_when_error_large_command = RewTerm(
            func=mdp.too_small_yaw_rate_command_penalty,
            weight=0.0,
            params={"active_threshold": 0.05, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.wheel_turn_mode_soft_limit = RewTerm(
            func=mdp.wheel_turn_mode_target_soft_limit_penalty,
            weight=0.0,
            params={"action_name": "wheel_motor_csv", "soft_limit": 8.0},
        )
        self.rewards.wheel_target_abs_soft_limit = RewTerm(
            func=mdp.wheel_target_abs_soft_limit_penalty,
            weight=0.0,
            params={"action_name": "wheel_motor_csv", "soft_limit": 8.0},
        )
        self.rewards.wheel_joint_vel_abs_soft_limit = RewTerm(
            func=mdp.wheel_joint_vel_abs_soft_limit_penalty,
            weight=0.0,
            params={"action_name": "wheel_motor_csv", "soft_limit": 10.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.wasted_turn_when_yaw_small = RewTerm(
            func=mdp.wasted_turn_when_yaw_small_penalty,
            weight=0.0,
            params={"action_name": "wheel_motor_csv", "active_threshold": 0.05, "asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class RangerShortGoalFlatV1EnvCfg(RangerShortGoalFlatEnvCfg):
    """Turn, approach, decelerate, and stop with four independent wheel outputs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 22.0
        self.short_goal_stop_phase_enter_distance = 0.30
        self.short_goal_stop_success_distance = 0.50
        self.short_goal_stop_max_xy_speed = 0.15
        self.short_goal_stop_max_yaw_rate = 0.20
        self.short_goal_stop_required_hold_steps = 24
        self.short_goal_wheel_radius = 0.2024
        self.observations.policy_state.command_state.params.update(
            {
                "short_goal_encoding_mode": "long_term",
                "short_goal_observation_max_distance": None,
                "short_goal_near_distance_range": 3.0,
                "short_goal_global_distance_unit": 1.0,
                "short_goal_velocity_reference": 1.5,
            }
        )

        # Keep wheel targets in a physically useful range and normalize wheel-state observations consistently.
        self.actions.wheel_motor_csv.velocity_limit = 20.0
        self.observations.policy_state.wheel_joint_vel_rel.params["scale"] = 20.0

        self.events.reset_short_goal_target = EventTerm(
            func=mdp.reset_short_goal_turn_target,
            mode="reset",
            params={
                "distance_range": (5.0, 8.0),
                "left_heading_range_deg": (25.0, 45.0),
                "right_heading_range_deg": (-45.0, -25.0),
                "paired_sides": True,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # A low chassis is unsafe for the under-body sensors.
        self.terminations.root_height_low.params["minimum_height"] = 0.65
        self.terminations.goal_reached = None
        self.terminations.stopped_goal_reached = DoneTerm(
            func=mdp.short_goal_stopped,
            params={
                "success_distance": 0.50,
                "max_xy_speed": 0.15,
                "max_yaw_rate": 0.20,
                "required_hold_steps": 24,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Remove inherited historical/zero-weight terms so the active reward set is explicit.
        for reward_name in (
            "alive",
            "forward_progress",
            "upright",
            "lateral_velocity",
            "leg_joint_deviation",
            "wheel_joint_velocity",
            "leg_joint_velocity",
            "action_rate",
            "action_magnitude",
            "base_xy_velocity",
            "base_yaw_rate",
            "yaw_drift_from_reset",
            "root_height_tracking",
            "root_height_band",
            "actual_stroke_nominal",
            "actual_stroke_soft_limit",
            "stroke_diagonal_balance",
            "hydraulic_action_range",
            "low_stroke_negative_hydraulic_action",
            "height_gated_low_stroke_negative_hydraulic_action",
            "high_height_low_stroke_negative_hydraulic_action",
            "height_low_high_stroke_positive_hydraulic_action",
            "stroke_away_from_nominal_hydraulic_action",
            "front_rear_stroke_balance",
            "front_rear_stroke_split_action",
            "wheel_contact_count",
            "wheel_contact_stability",
            "wheel_contact_balance",
            "contact_force_diag_balance",
            "contact_force_left_right_balance",
            "contact_force_range_balance",
            "contact_force_min_support",
            "contact_force_min_ratio_support",
            "low_contact_support",
            "goal_velocity",
            "goal_success",
            "near_goal_stop",
            "heading_alignment",
            "turn_toward_goal",
            "signed_yaw_rate_tracking",
            "too_small_yaw_rate_when_error_large",
            "wrong_direction_yaw",
            "wrong_direction_yaw_command",
            "too_small_yaw_rate_when_error_large_command",
            "forward_velocity_during_turn",
            "wheel_turn_mode_soft_limit",
            "wheel_target_abs_soft_limit",
            "wheel_joint_vel_abs_soft_limit",
            "wasted_turn_when_yaw_small",
        ):
            if hasattr(self.rewards, reward_name):
                setattr(self.rewards, reward_name, None)

        # Task progress and turn-to-approach transition.
        self.rewards.progress_to_goal = RewTerm(
            func=mdp.short_goal_progress_reward_alignment_gated,
            weight=4.0,
            params={
                "heading_deadband": 0.20,
                "heading_full": 0.80,
                "alignment_floor": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.heading_error_reduction = RewTerm(
            func=mdp.short_goal_heading_error_reduction,
            weight=9.0,
            params={
                "min_progress": -0.5,
                "max_progress": 0.5,
                "use_turn_distance_gate": True,
                "turn_gate_start_distance": 0.50,
                "turn_gate_full_distance": 0.80,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.heading_error_persistent = RewTerm(
            func=mdp.short_goal_heading_error_cost,
            weight=-0.25,
            params={
                "fade_start_distance": 0.50,
                "full_distance": 0.80,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.turn_toward_goal = None
        self.rewards.yaw_rate_tracking = RewTerm(
            func=mdp.short_goal_continuous_yaw_rate_tracking_penalty,
            weight=-0.12,
            params={
                "yaw_rate_max": 0.35,
                "heading_deadband": 0.04,
                "heading_scale": 0.40,
                "yaw_rate_reference": 0.35,
                "max_normalized_error": 2.0,
                "use_turn_distance_gate": True,
                "turn_gate_start_distance": 0.50,
                "turn_gate_full_distance": 0.80,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.short_goal_speed_profile = RewTerm(
            func=mdp.short_goal_speed_profile_penalty,
            weight=-2.0,
            params={
                "stop_distance": 0.30,
                "braking_acceleration": 0.60,
                "reaction_time": 0.20,
                "braking_margin": 0.08,
                "near_distance": 1.0,
                "yaw_rate_ref": 0.80,
                "yaw_component_weight": 0.5,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.cruise_underspeed = RewTerm(
            func=mdp.short_goal_cruise_underspeed_penalty,
            weight=-0.30,
            params={
                "capture_distance": 0.30,
                "approach_full_distance": 0.60,
                "cruise_full_distance": 1.20,
                "approach_speed": 0.35,
                "cruise_speed": 0.80,
                "heading_deadband": 0.20,
                "heading_full": 0.80,
                "alignment_floor": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.near_goal_away_speed = RewTerm(
            func=mdp.short_goal_near_goal_away_speed_penalty,
            weight=-0.30,
            params={
                "stop_distance": 0.30,
                "active_distance": 1.0,
                "speed_reference": 0.30,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # Do not force wheel speed toward zero at the 0.30 m boundary. The
        # vehicle now crosses the boundary with finite speed, then the stop
        # phase overrides only the wheel action to zero.
        self.rewards.approach_wheel_target_excess = None
        self.rewards.stopped_goal_success = RewTerm(
            func=mdp.short_goal_stopped_success_reward,
            weight=5.0,
            params={
                "success_distance": 0.50,
                "max_xy_speed": 0.15,
                "max_yaw_rate": 0.20,
                "required_hold_steps": 24,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stop_phase_base_xy_speed = RewTerm(
            func=mdp.short_goal_stop_phase_xy_speed_penalty,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.stop_phase_yaw_rate = RewTerm(
            func=mdp.short_goal_stop_phase_yaw_rate_penalty,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.stop_phase_wheel_target = RewTerm(
            func=mdp.short_goal_stop_phase_wheel_target_penalty,
            weight=-0.2,
            params={"action_name": "wheel_motor_csv"},
        )
        self.rewards.stop_phase_distance_drift = RewTerm(
            func=mdp.short_goal_stop_phase_distance_drift_penalty,
            weight=-2.0,
            params={"enter_distance": 0.50, "asset_cfg": SceneEntityCfg("robot")},
        )

        # Keep four independent wheel outputs while adding explicit flat-ground coordination priors.
        self.rewards.short_goal_wheel_diff_prior = RewTerm(
            func=mdp.short_goal_wheel_turn_mode_prior_l1,
            weight=-0.05,
            params={
                "turn_gain": 0.5,
                "action_name": "wheel_motor_csv",
                "use_turn_distance_gate": True,
                "turn_gate_start_distance": 0.50,
                "turn_gate_full_distance": 0.80,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # Disable the wheel-mean common-mode penalty: same-side front/rear opposition can cancel in the mean.
        self.rewards.short_goal_forward_common_mode = None
        self.rewards.wheel_same_side_target_consistency = RewTerm(
            func=mdp.wheel_same_side_target_consistency_l1,
            weight=-0.20,
            params={"deadband": 0.05, "action_name": "wheel_motor_csv"},
        )
        self.rewards.wheel_same_side_opposite_sign = RewTerm(
            func=mdp.wheel_same_side_opposite_sign_penalty,
            weight=-1.0,
            params={"margin": 0.03, "action_name": "wheel_motor_csv"},
        )
        self.rewards.loaded_wheel_longitudinal_slip = RewTerm(
            func=mdp.loaded_wheel_longitudinal_slip_penalty,
            weight=-0.05,
            params={
                "wheel_radius": 0.2024,
                "min_contact_force": 20.0,
                "min_motion_speed": 0.20,
                "absolute_margin": 0.10,
                "relative_margin": 0.15,
                "excess_speed_reference": 0.50,
                "max_normalized_excess": 2.0,
                "sensor_cfg": SceneEntityCfg(
                    "wheel_contact_forces", body_names=["w_lf", "w_lb", "w_rf", "w_rb"]
                ),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.wheel_target_rate = RewTerm(
            func=mdp.wheel_target_rate_l1,
            weight=-0.02,
            params={"action_name": "wheel_motor_csv"},
        )

        # Posture, sensor-clearance, contact, and actuator safety.
        self.rewards.base_vertical_velocity.weight = -0.8
        self.rewards.base_roll_pitch_rate.weight = -0.6
        self.rewards.roll_angle.weight = -1.3
        self.rewards.pitch_angle.weight = -1.5
        self.rewards.base_height_low.weight = -8.0
        self.rewards.base_height_low.params["target_height"] = 0.70
        self.rewards.base_height_high.weight = -10.0
        self.rewards.base_height_high.params["target_height"] = 0.91
        self.rewards.termination = RewTerm(
            func=mdp.failure_termination_penalty,
            weight=-20.0,
            params={"excluded_terms": ("time_out", "stopped_goal_reached")},
        )
        self.rewards.hydraulic_action_magnitude.weight = -0.05
        self.rewards.hydraulic_action_rate.weight = -0.06
        self.rewards.stroke_range.weight = -0.3
        self.rewards.stroke_high_soft_limit = RewTerm(
            func=mdp.stroke_high_soft_limit_penalty,
            weight=-1.0,
            params={"soft_start": 0.55, "hard_reference": 0.60, "action_name": "leg_hydraulic"},
        )
        self.rewards.low_contact_ratio_support.weight = -0.20
        self.rewards.unloaded_wheel_spin.weight = -0.006


@configclass
class RangerShortGoalFlatV2EnvCfg(RangerShortGoalFlatV1EnvCfg):
    """Wheel-only flat short-goal stage with suspension fixed at nominal mid-stroke."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Keep the 8-D action interface for checkpoint compatibility, but remove active
        # suspension control at the environment boundary. A normalized action of zero
        # corresponds to the nominal 0.50 m stroke used by the current hydraulic action term.
        self.fixed_suspension_action = 0.0


@configclass
class RangerShortGoalFlatV3EnvCfg(RangerShortGoalFlatV1EnvCfg):
    """Stage A: restore limited suspension authority on the original goal distribution."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.fixed_suspension_action = None
        self.suspension_action_scale = 0.05
        # Flat-ground stopping does not need an asymmetric latched suspension target.
        self.stop_phase_suspension_action = 0.0


@configclass
class RangerShortGoalFlatV4EnvCfg(RangerShortGoalFlatV3EnvCfg):
    """Stage B: widen heading coverage while keeping the 5--8 m distance range."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_short_goal_target = EventTerm(
            func=mdp.reset_short_goal_stratified_target,
            mode="reset",
            params={
                "distance_bands": ((5.0, 8.0),),
                "distance_weights": (1.0,),
                "heading_bands_deg": (
                    (-60.0, -45.0),
                    (-45.0, -15.0),
                    (-15.0, 15.0),
                    (15.0, 45.0),
                    (45.0, 60.0),
                ),
                "heading_weights": (0.20, 0.20, 0.20, 0.20, 0.20),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class RangerShortGoalFlatV5EnvCfg(RangerShortGoalFlatV4EnvCfg):
    """Stage C: widen distance coverage to 3--12 m with stratified sampling."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_short_goal_target.params.update(
            {
                "distance_bands": ((3.0, 5.0), (5.0, 8.0), (8.0, 12.0)),
                "distance_weights": (0.25, 0.50, 0.25),
            }
        )


@configclass
class RangerShortGoalFlatV6EnvCfg(RangerShortGoalFlatV5EnvCfg):
    """Stage D: final flat curriculum over 3--12 m and headings up to 75 degrees."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_short_goal_target.params.update(
            {
                "heading_bands_deg": (
                    (-75.0, -45.0),
                    (-45.0, -15.0),
                    (-15.0, 15.0),
                    (15.0, 45.0),
                    (45.0, 75.0),
                ),
                "heading_weights": (0.20, 0.20, 0.20, 0.20, 0.20),
            }
        )


@configclass
class RangerShortGoalFlatV7EnvCfg(RangerShortGoalFlatV6EnvCfg):
    """Stage V7: full-authority suspension and wheel fine-tuning on the V6 curriculum."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.fixed_suspension_action = None
        self.suspension_action_scale = 1.0
        self.stop_phase_suspension_action = 0.0


@configclass
class RangerShortGoalFlatV8EnvCfg(RangerShortGoalFlatV7EnvCfg):
    """Stage V8: require a balanced, settled suspension posture before declaring success."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.short_goal_stop_max_roll = 0.035
        self.short_goal_stop_max_pitch = 0.035
        self.short_goal_stop_max_stroke_tracking_error = 0.02
        self.short_goal_stop_count_only_after_phase = True
        self.short_goal_stop_required_hold_steps = 60
        posture_success_params = {
            "required_hold_steps": self.short_goal_stop_required_hold_steps,
            "max_roll": self.short_goal_stop_max_roll,
            "max_pitch": self.short_goal_stop_max_pitch,
            "max_stroke_tracking_error": self.short_goal_stop_max_stroke_tracking_error,
            "action_name": "leg_hydraulic",
        }
        self.terminations.stopped_goal_reached.params.update(posture_success_params)
        self.rewards.stopped_goal_success.params.update(posture_success_params)


@configclass
class RangerShortGoalFlatV9EnvCfg(RangerShortGoalFlatV8EnvCfg):
    """Stage V9: preserve policy suspension control while forcing zero wheel action in stop phase."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.stop_phase_suspension_mode = "policy"
        self.stop_phase_suspension_action = None

        # V9 removes global suspension regularizers that can oppose necessary
        # posture correction and future terrain-following motion. Safety,
        # posture, wheel-support, and stop-phase terms remain active.
        self.rewards.hydraulic_action_magnitude = None
        self.rewards.hydraulic_action_rate = None
        self.rewards.stroke_range = None

        self.rewards.stop_phase_roll_error = RewTerm(
            func=mdp.short_goal_stop_phase_roll_error_penalty,
            weight=-0.10,
            params={
                "reference_angle": self.short_goal_stop_max_roll,
                "max_normalized_error": 4.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stop_phase_pitch_error = RewTerm(
            func=mdp.short_goal_stop_phase_pitch_error_penalty,
            weight=-0.10,
            params={
                "reference_angle": self.short_goal_stop_max_pitch,
                "max_normalized_error": 4.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class RangerShortGoalFlatV10EnvCfg(RangerShortGoalFlatV9EnvCfg):
    """Stage V10: capture and stop inside 0.50 m, with an optional 0.30 m precision bonus."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Align stop-phase capture with the actual success radius. Stable success
        # still requires 60 consecutive steps satisfying all V8/V9 stop gates.
        self.short_goal_stop_phase_enter_distance = self.short_goal_stop_success_distance
        self.rewards.stop_phase_distance_drift.params["enter_distance"] = (
            self.short_goal_stop_success_distance
        )

        # Reaching 0.30 m is no longer mandatory for success. It remains a
        # one-time precision objective worth less than the final stopped success.
        self.rewards.precision_reach_bonus = RewTerm(
            func=mdp.short_goal_precision_reach_reward,
            weight=2.0,
            params={
                "precision_distance": 0.30,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class RangerShortGoalFlatCRecurrentEnvCfg(RangerShortGoalFlatV10EnvCfg):
    """C-stage experiment alias: identical environment/reward behavior to V10."""

    pass


@configclass
class RangerShortGoalFlatV11EnvCfg(RangerShortGoalFlatV10EnvCfg):
    """Stage V11: policy-controlled braking inside the latched 0.50 m stop phase."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep the stop-phase latch and observation flag, but execute the policy's
        # wheel outputs instead of replacing them with four zeros.
        self.stop_phase_wheel_override_enabled = False

        # Replace immediate-stop penalties with distance-dependent envelopes.
        # At 0.50 m the policy may still move slowly; the admissible motion shrinks
        # linearly to zero at the 0.30 m precision radius.
        self.rewards.stop_phase_base_xy_speed = None
        self.rewards.stop_phase_yaw_rate = None
        self.rewards.stop_phase_wheel_target = None
        self.rewards.stop_phase_xy_speed_envelope = RewTerm(
            func=mdp.short_goal_stop_phase_xy_speed_envelope_penalty,
            weight=-2.0,
            params={
                "enter_distance": 0.50,
                "zero_distance": 0.30,
                "allowed_speed_at_enter": 0.25,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stop_phase_yaw_rate_envelope = RewTerm(
            func=mdp.short_goal_stop_phase_yaw_rate_envelope_penalty,
            weight=-1.0,
            params={
                "enter_distance": 0.50,
                "zero_distance": 0.30,
                "allowed_yaw_rate_at_enter": 0.30,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stop_phase_wheel_target_envelope = RewTerm(
            func=mdp.short_goal_stop_phase_wheel_target_envelope_penalty,
            weight=-1.0,
            params={
                "enter_distance": 0.50,
                "zero_distance": 0.30,
                "allowed_action_at_enter": 0.20,
                "action_name": "wheel_motor_csv",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Dense shaping in 0.30--0.50 m complements the one-time 0.30 m bonus.
        self.rewards.stop_phase_precision_closeness = RewTerm(
            func=mdp.short_goal_stop_phase_precision_closeness_reward,
            weight=1.0,
            params={
                "enter_distance": 0.50,
                "precision_distance": 0.30,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
