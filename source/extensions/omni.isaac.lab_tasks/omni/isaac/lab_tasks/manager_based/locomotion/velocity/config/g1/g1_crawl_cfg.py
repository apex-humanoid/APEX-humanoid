# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Crawl training configuration.

Ordinary Isaac Lab configclasses; edit these values to train a new experiment.
"""
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators.actuator_cfg import IdealPDActuatorCfg
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.envs.common import ViewerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from omni.isaac.lab.envs.mdp.commands.commands_cfg import ClimbCommandCfg, UniformVelocityCommandCfg
from omni.isaac.lab.envs.ui.manager_based_rl_env_window import ManagerBasedRLEnvWindow
from omni.isaac.lab.managers import ObservationGroupCfg
from omni.isaac.lab.managers.manager_term_cfg import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from omni.isaac.lab.managers.recorder_manager import RecorderManagerBaseCfg
from omni.isaac.lab.managers.scene_entity_cfg import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.contact_sensor.contact_ground_sensor_z_cfg import ContactGroundSensorZCfg
from omni.isaac.lab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from omni.isaac.lab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg
from omni.isaac.lab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg
from omni.isaac.lab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.spawners.lights.lights_cfg import DomeLightCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.sim.spawners.materials.visual_materials_cfg import MdlFileCfg
from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from omni.isaac.lab.terrains.terrain_importer_cfg import TerrainImporterCfg
from omni.isaac.lab.terrains.trimesh.mesh_terrains_cfg import MeshChangingBoxTerrainCfg
from omni.isaac.lab.utils.noise.noise_cfg import UniformNoiseCfg
from omni.isaac.lab_assets.unitree import G1_USD_PATH
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.events import (
    push_by_adding_velocity,
    randomize_joint_default_pos_crawl,
    randomize_rigid_body_com,
    randomize_rigid_body_mass_crawl,
    randomize_rigid_body_material,
    reset_joints_to_lying_by_offset_specified_joint,
    reset_root_state_uniform,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.locomotion_rewards import (
    ang_vel_xy,
    base_height_range,
    lin_vel_z,
    track_ang_vel_z_world_exp,
    track_lin_vel_xy_ground_tangent_exp,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import (
    base_ang_vel,
    base_lin_vel,
    joint_pos_rel_crawl,
    joint_vel_rel,
    projected_gravity,
    real_last_processed_action,
    tri_phase,
    velocity_commands,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.rewards import (
    contact_forces,
    is_alive,
    is_terminated_climbup,
    joint_acc_l2,
    joint_pos_limits,
    joint_torques,
    joint_vel_l2_climbdown,
    lying_joint_deviation_l2,
    processed_action_rate_l2,
    undesired_contacts,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.terminations import (
    illegal_contact,
    leg_stretched,
    time_out,
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    num_envs = 4096
    env_spacing = 2.5
    lazy_sensor_update = True
    replicate_physics = True
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                retain_accelerations=False,
            ),
            activate_contact_sensors=True,
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
            usd_path=G1_USD_PATH,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0.0, 0.4),
            rot=(0.7071, 0.0, 0.7071, 0.0),
            joint_pos={".*_hip_pitch_joint": -0.1, ".*_knee_joint": 0.3, ".*_ankle_pitch_joint": -0.2},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    "waist_roll_joint",
                    "waist_yaw_joint",
                    "waist_pitch_joint",
                ],
                effort_limit={
                    ".*_hip_yaw_joint": 88,
                    ".*_hip_roll_joint": 88,
                    ".*_hip_pitch_joint": 88,
                    ".*_knee_joint": 139,
                    "waist_roll_joint": 50,
                    "waist_yaw_joint": 88,
                    "waist_pitch_joint": 50,
                },
                velocity_limit={
                    ".*_hip_yaw_joint": 32.0,
                    ".*_hip_roll_joint": 32.0,
                    ".*_hip_pitch_joint": 32.0,
                    ".*_knee_joint": 20.0,
                    "waist_roll_joint": 37.0,
                    "waist_yaw_joint": 32.0,
                    "waist_pitch_joint": 37.0,
                },
                stiffness={
                    ".*_hip_yaw_joint": 100.0,
                    ".*_hip_roll_joint": 100.0,
                    ".*_hip_pitch_joint": 100.0,
                    ".*_knee_joint": 150.0,
                    "waist_roll_joint": 100.0,
                    "waist_yaw_joint": 100.0,
                    "waist_pitch_joint": 100.0,
                },
                damping={
                    ".*_hip_yaw_joint": 4.0,
                    ".*_hip_roll_joint": 4.0,
                    ".*_hip_pitch_joint": 4.0,
                    ".*_knee_joint": 5.0,
                    "waist_roll_joint": 4.0,
                    "waist_yaw_joint": 4.0,
                    "waist_pitch_joint": 4.0,
                },
                armature=0.01,
                friction=0.0,
            ),
            "feet": IdealPDActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit=50,
                velocity_limit=37,
                stiffness=40.0,
                damping={".*_ankle_pitch_joint": 2, ".*_ankle_roll_joint": 2},
                armature=0.01,
                friction=0.0,
            ),
            "arms": IdealPDActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
                effort_limit={
                    ".*_shoulder_pitch_joint": 25,
                    ".*_shoulder_roll_joint": 25,
                    ".*_shoulder_yaw_joint": 25,
                    ".*_elbow_joint": 25,
                    ".*_wrist_roll_joint": 25,
                    ".*_wrist_pitch_joint": 5,
                    ".*_wrist_yaw_joint": 5,
                },
                velocity_limit={
                    ".*_shoulder_pitch_joint": 37,
                    ".*_shoulder_roll_joint": 37,
                    ".*_shoulder_yaw_joint": 37,
                    ".*_elbow_joint": 37,
                    ".*_wrist_roll_joint": 37,
                    ".*_wrist_pitch_joint": 22,
                    ".*_wrist_yaw_joint": 22,
                },
                stiffness={
                    ".*_shoulder_pitch_joint": 100,
                    ".*_shoulder_roll_joint": 100,
                    ".*_shoulder_yaw_joint": 50,
                    ".*_elbow_joint": 50,
                    ".*_wrist_roll_joint": 100,
                    ".*_wrist_pitch_joint": 100,
                    ".*_wrist_yaw_joint": 100,
                },
                damping={
                    ".*_shoulder_pitch_joint": 4,
                    ".*_shoulder_roll_joint": 4,
                    ".*_shoulder_yaw_joint": 2.5,
                    ".*_elbow_joint": 2.5,
                    ".*_wrist_roll_joint": 4.0,
                    ".*_wrist_pitch_joint": 4.0,
                    ".*_wrist_yaw_joint": 4.0,
                },
                armature=0.01,
                friction=0.0,
            ),
        },
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        num_envs=4096,
        terrain_generator=TerrainGeneratorCfg(
            size=(10.0, 10.0),
            border_width=2,
            num_rows=10,
            num_cols=20,
            sub_terrains={
                "box": MeshChangingBoxTerrainCfg(box_height_range=(0.0, 0.0), platform_width_range=(3.0, 3.0))
            },
        ),
        env_spacing=2.5,
        visual_material=MdlFileCfg(
            mdl_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        max_init_terrain_level=5,
    )
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", update_period=0.005, history_length=3, track_air_time=True
    )
    bodies_ground_contact = ContactGroundSensorZCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True
    )
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        update_period=0.02,
        mesh_prim_paths=["/World/ground"],
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=GridPatternCfg(resolution=0.04, size=[1.0, 1.0]),
        drift_mode="split_xyz",
        apply_drift=True,
        compute_no_drift=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            texture_file="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    camera = None


@configclass
class ObservationsCfgPolicyCfg(ObservationGroupCfg):
    concatenate_terms = True
    enable_corruption = True
    history_length = 6
    flatten_history_dim = True
    root_ang_vel = ObservationTermCfg(func=base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2), history_length=6)
    projected_gravity = ObservationTermCfg(func=projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
    velocity_commands = ObservationTermCfg(
        func=velocity_commands, params={"command_name": "base_velocity", "lin_scale": 2, "ang_scale": 0.25}
    )
    joint_pos = ObservationTermCfg(func=joint_pos_rel_crawl, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01))
    joint_vel = ObservationTermCfg(func=joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5))
    actions = ObservationTermCfg(func=real_last_processed_action, params={"action_name": "joint_pos"})
    phase = ObservationTermCfg(func=tri_phase, params={"command_name": "base_velocity", "period": 0.65})


@configclass
class ObservationsCfgCriticCfg(ObservationGroupCfg):
    concatenate_terms = True
    enable_corruption = False
    history_length = 6
    flatten_history_dim = True
    base_lin_vel = ObservationTermCfg(func=base_lin_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))
    base_ang_vel = ObservationTermCfg(func=base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))
    projected_gravity = ObservationTermCfg(func=projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
    velocity_commands = ObservationTermCfg(
        func=velocity_commands, params={"command_name": "base_velocity", "lin_scale": 2, "ang_scale": 0.25}
    )
    joint_pos = ObservationTermCfg(func=joint_pos_rel_crawl, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01))
    joint_vel = ObservationTermCfg(func=joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5))
    actions = ObservationTermCfg(func=real_last_processed_action, params={"action_name": "joint_pos"})
    phase = ObservationTermCfg(func=tri_phase, params={"command_name": "base_velocity", "period": 0.65})


@configclass
class ObservationsCfg:
    policy = ObservationsCfgPolicyCfg()
    critic = ObservationsCfgCriticCfg()


@configclass
class ActionsCfg:
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        clip={".*_joint": (-100.0, 100.0)},
        joint_names=[".*"],
        scale=0.25,
        action_start_step=0,
        init_joint_mode="zero",
        live_default_pose=False,
    )


@configclass
class EventCfg:
    base_com = EventTermCfg(
        func=randomize_rigid_body_com,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
        mode="startup",
    )
    physics_material = EventTermCfg(
        func=randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "make_consistent": True,
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
        mode="startup",
    )
    add_base_mass = EventTermCfg(
        func=randomize_rigid_body_mass_crawl,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
        mode="startup",
    )
    add_joint_default_pos = EventTermCfg(
        func=randomize_joint_default_pos_crawl,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
        mode="startup",
    )
    reset_base_on_ground = EventTermCfg(
        func=reset_root_state_uniform,
        params={
            "pose_range": {
                "x": (-0.65, -0.65),
                "y": (-0.6, 0.6),
                "z": (0.0, 0.0),
                "roll": (-0.5235987755982988, 0.5235987755982988),
                "pitch": (0.349, 0.349),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
        mode="reset",
    )
    reset_joint_lying = EventTermCfg(
        func=reset_joints_to_lying_by_offset_specified_joint,
        params={
            "position_range": {
                "waist_yaw_joint": (-0.15, 0.15),
                "waist_roll_joint": (-0.17, 0.15),
                "waist_pitch_joint": (-0.15, 0.29),
                ".*_hip_pitch_joint": (-0.15, 0.15),
                ".*_hip_yaw_joint": (-0.15, 0.15),
                "left_hip_roll_joint": (-0.15, 0.26),
                "right_hip_roll_joint": (-0.21, 0.15),
                ".*_knee_joint": (-0.15, 0.24),
                ".*_ankle_pitch_joint": (-0.15, 0.19),
                ".*_ankle_roll_joint": (-0.15, 0.15),
                ".*_shoulder_pitch_joint": (-0.15, 0.67),
                ".*_shoulder_roll_joint": (-0.15, 0.19),
                "left_shoulder_yaw_joint": (-0.15, 0.49),
                "right_shoulder_yaw_joint": (-0.66, 0.15),
                ".*_elbow_joint": (-0.71, 0.15),
                ".*wrist_.*": (-0.1, 0.1),
            },
            "velocity_range": (0.0, 0.0),
        },
        mode="reset",
    )
    push_robot_walk = EventTermCfg(
        func=push_by_adding_velocity,
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "pitch": (-0.52, 0.52),
                "roll": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            }
        },
        mode="interval",
        interval_range_s=(1.0, 3.0),
    )


@configclass
class G1Rewards:
    track_lin_vel_xy_exp = RewardTermCfg(
        func=track_lin_vel_xy_ground_tangent_exp,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg(name="robot", body_names=["torso_link"]),
        },
        weight=1.3,
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=track_ang_vel_z_world_exp, params={"command_name": "base_velocity", "std": 0.5}, weight=1.3
    )
    lin_vel_z = RewardTermCfg(func=lin_vel_z, params={"asset_cfg": SceneEntityCfg(name="robot")}, weight=-2)
    ang_vel_xy = RewardTermCfg(func=ang_vel_xy, params={"asset_cfg": SceneEntityCfg(name="robot")}, weight=-0.05)
    base_height = RewardTermCfg(
        func=base_height_range,
        params={"min_height": 0.31, "max_height": 0.35, "asset_cfg": SceneEntityCfg(name="robot")},
        weight=-10,
    )
    joint_acc_penalty = RewardTermCfg(
        func=joint_acc_l2, params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])}, weight=-2.5e-07
    )
    joint_vel_penalty = RewardTermCfg(
        func=joint_vel_l2_climbdown,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])},
        weight=-0.001,
    )
    action_rate_penalty = RewardTermCfg(
        func=processed_action_rate_l2, params={"action_name": "joint_pos"}, weight=-0.05
    )
    joint_pos_limits = RewardTermCfg(
        func=joint_pos_limits, params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])}, weight=-5
    )
    alive = RewardTermCfg(func=is_alive, weight=10)
    termination = RewardTermCfg(func=is_terminated_climbup, weight=-100)
    joint_deviation_penalty = RewardTermCfg(
        func=lying_joint_deviation_l2,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])},
        weight=-1.0,
    )
    contact_forces_penalty = RewardTermCfg(
        func=contact_forces,
        params={"sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*"]), "threshold": 500.0},
        weight=-0.01,
    )
    torque_penalty = RewardTermCfg(
        func=joint_torques, params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])}, weight=-1e-05
    )
    undesired_contact = RewardTermCfg(
        func=undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces",
                body_names=[
                    "torso_link",
                    "head_link",
                    ".*shoulder.*",
                    "pelvis",
                    ".*hip_pitch_link",
                    ".*hip_roll_link",
                    ".*ankle.*",
                    ".*wrist_yaw.*",
                    ".*wrist_pitch.*",
                ],
            ),
            "threshold": 0.1,
        },
        weight=-1,
    )


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=time_out, time_out=True)
    torso_head_contact = TerminationTermCfg(
        func=illegal_contact,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["torso_link", "head_link"]),
        },
    )
    leg_stretched = TerminationTermCfg(
        func=leg_stretched,
        params={"threshold": 0.65, "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*_ankle_roll_link"])},
    )


@configclass
class CommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        asset_name="robot",
        heading_control_stiffness=0.5,
        rel_standing_envs=0.1,
        rel_heading_envs=0.5,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.3, 0.3),
            heading=(-3.141592653589793, 3.141592653589793),
        ),
        planar_deadband=0.1,
    )
    climb_command = ClimbCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        standing_joint={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_elbow_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": -0.0,
            "right_shoulder_roll_joint": -0.0,
            "right_shoulder_pitch_joint": -0.0,
        },
        reset_min_height_from_body=False,
    )


@configclass
class G1CrawlEnvCfg(ManagerBasedRLEnvCfg):
    viewer = ViewerCfg(eye=(0, 2.5, 1), lookat=(0, 0.0, 0.5))
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        disable_contact_processing=True,
        physx=PhysxCfg(gpu_max_rigid_patch_count=327680, gpu_temp_buffer_capacity=167772160),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
    )
    ui_window_class_type = ManagerBasedRLEnvWindow
    seed = 42
    decimation = 4
    scene = MySceneCfg()
    recorders = RecorderManagerBaseCfg()
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rerender_on_reset = False
    is_finite_horizon = False
    episode_length_s = 20
    rewards = G1Rewards()
    terminations = TerminationsCfg()
    curriculum = None
    commands = CommandsCfg()
    height_scanner = None
    events_before_observations = True
