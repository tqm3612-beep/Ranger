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
        self.viewer.eye = (8.0, -8.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)


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
class RangerSpeedCommandFlatEnvCfg(RangerForwardEnvCfg):
    """Low-level flat-ground speed-command tracking task with dynamic commands."""

    command_stage: str = "A"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 10.0
        self.scene.ground.spawn.size = (300.0, 300.0, 0.02)

        command_stage = self.command_stage.upper()
        if command_stage not in ("A", "B"):
            raise ValueError(f"Unsupported speed-command stage: {self.command_stage}")

        # Replace the fixed goal descriptor with a dynamic velocity-command descriptor.
        command_params = {
            "stage": command_stage,
            "v_x_range": (0.03, 0.45) if command_stage == "A" else (0.05, 0.35),
            "v_x_bins": ((0.03, 0.12), (0.12, 0.28), (0.28, 0.45)) if command_stage == "A" else None,
            "yaw_rate_range": (0.0, 0.0) if command_stage == "A" else (-0.4, 0.4),
            "command_duration_range": (2.0, 5.0),
            "smoothing_alpha": 0.1,
            "max_command_duration": 5.0,
            "small_yaw_prob": 0.10 if command_stage == "A" else 0.0,
            "small_yaw_range": (0.03, 0.08),
        }
        self.observations.policy.goal_state.func = mdp.speed_command_state
        self.observations.policy.goal_state.params = command_params
        self.observations.policy.local_navigation_map.params["use_neutral_map"] = True

        self.events.reset_speed_command = EventTerm(
            func=mdp.reset_speed_command,
            mode="reset",
            params={
                "stage": command_stage,
                "v_x_range": command_params["v_x_range"],
                "v_x_bins": command_params["v_x_bins"],
                "yaw_rate_range": command_params["yaw_rate_range"],
                "command_duration_range": command_params["command_duration_range"],
                "small_yaw_prob": command_params["small_yaw_prob"],
                "small_yaw_range": command_params["small_yaw_range"],
            },
        )

        # Keep the posture task alive long enough for command-conditioned wheel control to emerge.
        self.terminations.root_height_low.params["minimum_height"] = 0.30
        self.actions.wheel_motor_csv.velocity_limit = 6.0
        self.observations.policy.wheel_velocity_target.params["velocity_limit"] = 6.0

        # Disable fixed-speed forward shaping from the old straight-line stage.
        self.rewards.forward_progress.weight = 0.0
        self.rewards.wheel_joint_velocity.weight = 0.0

        # Keep stability constraints, but tune them around the default flat-ground support posture.
        self.rewards.base_vertical_velocity.weight = -0.8
        self.rewards.base_roll_pitch_rate.weight = -0.4
        self.rewards.lateral_velocity.weight = -0.3
        self.rewards.action_rate.weight = -0.02
        self.rewards.action_magnitude.weight = -0.0003
        self.rewards.termination.weight = -30.0
        self.rewards.roll_angle = RewTerm(func=mdp.roll_angle_l2, weight=-2.0)
        self.rewards.pitch_angle = RewTerm(func=mdp.pitch_angle_l2, weight=-3.0)
        self.rewards.root_height_tracking = RewTerm(
            func=mdp.root_height_tracking_exp,
            weight=4.0,
            params={"target_height": 0.74, "std_sq": 0.01, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.base_height_low = RewTerm(
            func=mdp.base_height_below_target_l2,
            weight=-12.0,
            params={"target_height": 0.72, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.actual_stroke_nominal = RewTerm(
            func=mdp.actual_stroke_nominal_l2,
            weight=-10.0,
            params={"stroke_nominal": 0.33, "action_name": "leg_hydraulic"},
        )
        self.rewards.actual_stroke_soft_limit = RewTerm(
            func=mdp.actual_stroke_soft_limit_penalty,
            weight=-12.0,
            params={"limit": 0.45, "action_name": "leg_hydraulic"},
        )
        self.rewards.wheel_semantic_velocity_symmetry = RewTerm(
            func=mdp.wheel_semantic_velocity_symmetry_l2,
            weight=0.0,
        )

        # Command-conditioned speed rewards.
        self.rewards.lin_vel_x_tracking = RewTerm(
            func=mdp.lin_vel_x_command_tracking_exp,
            weight=3.0,
            params={"std_sq": 0.03, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.lin_vel_x_error_penalty = RewTerm(
            func=mdp.lin_vel_x_command_error_l1,
            weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.low_cmd_speed_error_penalty = RewTerm(
            func=mdp.low_cmd_speed_error_l1,
            weight=-8.0,
            params={"command_threshold": 0.12, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.mid_cmd_speed_error_penalty = RewTerm(
            func=mdp.mid_cmd_speed_error_l1,
            weight=-4.0,
            params={"command_min": 0.12, "command_max": 0.28, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.normalized_cmd_speed_error_penalty = RewTerm(
            func=mdp.normalized_cmd_speed_error_l1,
            weight=-1.0,
            params={"min_command": 0.08, "max_error": 2.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.overspeed_cmd_penalty = RewTerm(
            func=mdp.overspeed_command_penalty,
            weight=-8.0,
            params={"margin": 0.03, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.backward_penalty = RewTerm(
            func=mdp.backward_velocity_penalty,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Flat stage-1 scaffold: enforce strong support quality while validating low-level speed control.
        # Future mixed-terrain stages should replace this with softer support-quality rewards that allow
        # short single-wheel lift-off over uneven terrain.
        self.rewards.wheel_contact_count.weight = 5.0
        self.rewards.wheel_contact_balance.weight = 1.0
        self.rewards.wheel_all_contact = RewTerm(
            func=mdp.wheel_all_contact_reward,
            weight=3.0,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "threshold": 1.0},
        )
        self.rewards.wheel_contact_force_low = RewTerm(
            func=mdp.wheel_contact_force_low_penalty,
            weight=-0.01,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "min_force": 80.0},
        )

        yaw_weight = 0.2 if command_stage == "A" else 3.0
        yaw_error_weight = -0.1 if command_stage == "A" else -1.0
        turn_direction_weight = 0.0 if command_stage == "A" else 0.5
        turn_difference_weight = 0.0 if command_stage == "A" else -0.05
        yaw_gain = 0.0 if command_stage == "A" else 2.5
        forward_sign = 1.0
        self.rewards.yaw_rate_tracking = RewTerm(
            func=mdp.yaw_rate_command_tracking_exp,
            weight=yaw_weight,
            params={"std_sq": 0.06, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.yaw_rate_error_penalty = RewTerm(
            func=mdp.yaw_rate_command_error_l1,
            weight=yaw_error_weight,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.turn_direction = RewTerm(
            func=mdp.turn_direction_reward,
            weight=turn_direction_weight,
            params={"yaw_cmd_deadband": 0.1, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.wheel_command_tracking = RewTerm(
            func=mdp.wheel_command_tracking_exp,
            weight=0.3,
            params={
                "forward_gain": 9.0,
                "yaw_gain": yaw_gain,
                "max_abs_speed": 6.0,
                "forward_sign": forward_sign,
                "std_sq": 4.0,
                "action_name": "wheel_motor_csv",
            },
        )
        self.rewards.wheel_command_error = RewTerm(
            func=mdp.wheel_command_error_l1,
            weight=-0.15,
            params={
                "forward_gain": 9.0,
                "yaw_gain": yaw_gain,
                "max_abs_speed": 6.0,
                "forward_sign": forward_sign,
                "action_name": "wheel_motor_csv",
            },
        )
        self.rewards.wheel_target_magnitude = RewTerm(
            func=mdp.wheel_target_magnitude_l1,
            weight=-0.005,
            params={"action_name": "wheel_motor_csv"},
        )
        self.rewards.wheel_turn_difference_error = RewTerm(
            func=mdp.wheel_turn_difference_command_l1,
            weight=turn_difference_weight,
            params={"yaw_gain": yaw_gain, "action_name": "wheel_motor_csv"},
        )


@configclass
class RangerGoalHeadingFlatEnvCfg(RangerStandEnvCfg):
    """Flat target-heading pretraining task without speed-command or map inputs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 10.0
        self.scene.ground.spawn.size = (300.0, 300.0, 0.02)

        # Keep the network interface checkpoint-compatible while replacing the neutral goal slot
        # with a sampled target-heading descriptor. The local map remains a neutral placeholder.
        self.observations.policy.goal_state.func = mdp.goal_heading_state
        self.observations.policy.goal_state.params = {"goal_range": 5.0, "asset_cfg": SceneEntityCfg("robot")}
        self.observations.policy.local_navigation_map.params["use_neutral_map"] = True

        self.events.reset_goal_heading_target = EventTerm(
            func=mdp.reset_goal_heading_target,
            mode="reset",
            params={
                "distance_range": (2.0, 5.0),
                "heading_range": (-0.7853981633974483, 0.7853981633974483),
                "heading_bins": (
                    (0.2617993877991494, 0.7853981633974483),
                    (-0.7853981633974483, -0.2617993877991494),
                    (-0.13962634015954636, 0.13962634015954636),
                ),
                "heading_bin_probs": (0.40, 0.40, 0.20),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Keep wheel targets bounded for stand-checkpoint fine-tuning.
        self.actions.wheel_motor_csv.velocity_limit = 6.0
        self.observations.policy.wheel_velocity_target.params["velocity_limit"] = 6.0

        # This task learns target-heading response directly, not fixed +x speed or speed-command tracking.
        self.rewards.forward_progress.weight = 0.0
        self.rewards.wheel_joint_velocity.weight = 0.0

        # Health constraints copied conservatively from the flat speed-control scaffold.
        self.terminations.bad_orientation.params["limit_angle"] = 1.4
        self.terminations.root_height_low.params["minimum_height"] = 0.30
        self.rewards.base_vertical_velocity.weight = -0.8
        self.rewards.base_roll_pitch_rate.weight = -0.4
        self.rewards.lateral_velocity.weight = -0.05
        self.rewards.action_rate.weight = -0.02
        self.rewards.action_magnitude.weight = -0.0003
        self.rewards.termination.weight = -30.0
        self.rewards.roll_angle = RewTerm(func=mdp.roll_angle_l2, weight=-2.0)
        self.rewards.pitch_angle = RewTerm(func=mdp.pitch_angle_l2, weight=-3.0)
        self.rewards.root_height_tracking = RewTerm(
            func=mdp.root_height_tracking_exp,
            weight=4.0,
            params={"target_height": 0.74, "std_sq": 0.01, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.base_height_low = RewTerm(
            func=mdp.base_height_below_target_l2,
            weight=-12.0,
            params={"target_height": 0.72, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.actual_stroke_nominal = RewTerm(
            func=mdp.actual_stroke_nominal_l2,
            weight=-10.0,
            params={"stroke_nominal": 0.33, "action_name": "leg_hydraulic"},
        )
        self.rewards.actual_stroke_soft_limit = RewTerm(
            func=mdp.actual_stroke_soft_limit_penalty,
            weight=-12.0,
            params={"limit": 0.45, "action_name": "leg_hydraulic"},
        )
        self.rewards.wheel_contact_count.weight = 5.0
        self.rewards.wheel_contact_balance.weight = 1.0
        self.rewards.wheel_all_contact = RewTerm(
            func=mdp.wheel_all_contact_reward,
            weight=1.5,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "threshold": 1.0},
        )
        self.rewards.wheel_contact_force_low = RewTerm(
            func=mdp.wheel_contact_force_low_penalty,
            weight=-0.01,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=["w_.*"]), "min_force": 80.0},
        )

        # Target-heading rewards: direct target direction -> wheel targets + stable motion.
        self.rewards.goal_heading_alignment = RewTerm(
            func=mdp.goal_heading_alignment,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.goal_velocity_towards_target = RewTerm(
            func=mdp.goal_heading_velocity_towards_target,
            weight=6.0,
            params={"max_velocity": 1.0, "min_reward": -0.2, "max_reward": 0.7, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.goal_turn_direction = RewTerm(
            func=mdp.goal_heading_turn_direction,
            weight=0.5,
            params={"heading_deadband": 0.08, "yaw_rate_deadband": 0.02, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.goal_heading_error_progress = RewTerm(
            func=mdp.goal_heading_error_progress,
            weight=2.0,
            params={"min_reward": 0.0, "max_reward": 0.5, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.goal_wheel_prior = RewTerm(
            func=mdp.goal_heading_wheel_prior_exp,
            weight=0.5,
            params={
                "forward_gain": 4.0,
                "turn_gain": 3.0,
                "max_abs_speed": 6.0,
                "std_sq": 4.0,
                "action_name": "wheel_motor_csv",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.goal_wheel_diff_error = RewTerm(
            func=mdp.goal_heading_wheel_diff_l1,
            weight=-0.1,
            params={
                "turn_gain": 3.0,
                "max_abs_diff": 6.0,
                "action_name": "wheel_motor_csv",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.goal_front_wheel_diff_error = RewTerm(
            func=mdp.goal_heading_front_wheel_diff_l1,
            weight=-0.06,
            params={
                "heading_deadband": 0.13962634015954636,
                "action_name": "wheel_motor_csv",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.wheel_target_magnitude = RewTerm(
            func=mdp.wheel_target_magnitude_l1,
            weight=-0.005,
            params={"action_name": "wheel_motor_csv"},
        )


@configclass
class RangerForwardVisualEnvCfg(RangerForwardEnvCfg):
    """Forward-stage scene tuned for interactive visualization."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 8.0
        self.viewer.eye = (8.0, -8.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)
