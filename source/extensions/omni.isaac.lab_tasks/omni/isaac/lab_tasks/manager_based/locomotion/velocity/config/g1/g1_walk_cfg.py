# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Walk training configuration.

Ordinary Isaac Lab configclasses; edit these values to train a new experiment.
"""
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators.actuator_cfg import IdealPDActuatorCfg
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.envs.common import ViewerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import WalkJointActionCfg
from omni.isaac.lab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from omni.isaac.lab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from omni.isaac.lab.envs.ui.manager_based_rl_env_window import ManagerBasedRLEnvWindow
from omni.isaac.lab.managers import ObservationGroupCfg
from omni.isaac.lab.managers.manager_term_cfg import CurriculumTermCfg
from omni.isaac.lab.managers.manager_term_cfg import EventTermCfg
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.managers.manager_term_cfg import TerminationTermCfg
from omni.isaac.lab.managers.recorder_manager import RecorderManagerBaseCfg
from omni.isaac.lab.managers.scene_entity_cfg import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from omni.isaac.lab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg
from omni.isaac.lab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.simulation_cfg import PhysxCfg
from omni.isaac.lab.sim.simulation_cfg import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.spawners.lights.lights_cfg import DomeLightCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.sim.spawners.materials.visual_materials_cfg import MdlFileCfg
from omni.isaac.lab.terrains.height_field.hf_terrains import pyramid_sloped_terrain
from omni.isaac.lab.terrains.height_field.hf_terrains import random_uniform_terrain
from omni.isaac.lab.terrains.height_field.hf_terrains_cfg import HfPyramidSlopedTerrainCfg
from omni.isaac.lab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from omni.isaac.lab.terrains.terrain_importer_cfg import TerrainImporterCfg
from omni.isaac.lab.utils.noise.noise_cfg import UniformNoiseCfg
from omni.isaac.lab_assets.unitree import G1_USD_PATH
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.events import push_by_adding_velocity
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.events import randomize_joint_default_pos_crawl
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.events import randomize_rigid_body_com
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.events import randomize_rigid_body_mass_crawl
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.events import randomize_rigid_body_material
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.events import reset_root_state_uniform
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.locomotion_rewards import ang_vel_xy
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.locomotion_rewards import feet_air_time
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.locomotion_rewards import lin_vel_z
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.locomotion_rewards import track_ang_vel_z_world_exp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import base_ang_vel
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import base_lin_vel
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import joint_pos_rel_crawl
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import joint_vel_rel
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import projected_gravity
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import real_last_processed_action
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import tri_phase
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import velocity_commands
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.rewards import is_alive
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.rewards import joint_pos_limits
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.rewards import joint_torques
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.rewards import processed_action_rate_l2
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.rewards import undesired_contacts
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.terminations import root_height_below_minimum
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.terminations import time_out
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_actions import WalkJointAction
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_events import (
    reset_joints_by_offset_specified_joint,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_locomotion_curriculums import terrain_levels_vel
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_locomotion_rewards import base_height
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_locomotion_rewards import orientation_l2
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_locomotion_rewards import (
    track_lin_vel_xy_yaw_frame_exp,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_rewards import body_slipping_l2
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_rewards import feet_swing_height
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_rewards import joint_acc_with_action
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_rewards import joint_deviation_l2
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_rewards import joint_vel_with_action
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.walk_rewards import phase_contact


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
            pos=(0.0, 0.0, 0.79),
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
            curriculum=True,
            size=(8.0, 8.0),
            border_width=19.0,
            num_rows=10,
            num_cols=20,
            sub_terrains={
                "random_rough": HfRandomUniformTerrainCfg(
                    function=random_uniform_terrain,
                    proportion=0.5,
                    size=(8.0, 8.0),
                    flat_patch_sampling=None,
                    border_width=0.25,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    slope_threshold=0.75,
                    noise_range=(0.02, 0.1),
                    noise_step=0.02,
                    downsampled_scale=None,
                ),
                "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
                    function=pyramid_sloped_terrain,
                    proportion=0.5,
                    size=(8.0, 8.0),
                    flat_patch_sampling=None,
                    border_width=0.25,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    slope_threshold=0.75,
                    slope_range=(0.0, 0.0),
                    platform_width=2.0,
                    inverted=False,
                ),
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
    base_ang_vel = ObservationTermCfg(func=base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2), history_length=6)
    projected_gravity = ObservationTermCfg(
        func=projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05), history_length=6
    )
    velocity_commands = ObservationTermCfg(
        func=velocity_commands,
        params={"command_name": "base_velocity", "lin_scale": 2, "ang_scale": 0.25},
        history_length=6,
    )
    joint_pos = ObservationTermCfg(
        func=joint_pos_rel_crawl,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        history_length=6,
    )
    joint_vel = ObservationTermCfg(
        func=joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])},
        noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
        history_length=6,
    )
    actions = ObservationTermCfg(func=real_last_processed_action, params={"action_name": "joint_pos"}, history_length=6)
    phase = ObservationTermCfg(
        func=tri_phase, params={"period": 0.65, "command_name": "base_velocity"}, history_length=6
    )


@configclass
class ObservationsCfgCriticCfg(ObservationGroupCfg):
    concatenate_terms = True
    enable_corruption = False
    history_length = 6
    flatten_history_dim = True
    base_lin_vel = ObservationTermCfg(func=base_lin_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2), history_length=6)
    base_ang_vel = ObservationTermCfg(func=base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2), history_length=6)
    projected_gravity = ObservationTermCfg(
        func=projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05), history_length=6
    )
    velocity_commands = ObservationTermCfg(
        func=velocity_commands,
        params={"command_name": "base_velocity", "lin_scale": 2, "ang_scale": 0.25},
        history_length=6,
    )
    joint_pos = ObservationTermCfg(
        func=joint_pos_rel_crawl,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        history_length=6,
    )
    joint_vel = ObservationTermCfg(
        func=joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])},
        noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
        history_length=6,
    )
    actions = ObservationTermCfg(func=real_last_processed_action, params={"action_name": "joint_pos"}, history_length=6)
    phase = ObservationTermCfg(
        func=tri_phase, params={"period": 0.65, "command_name": "base_velocity"}, history_length=6
    )


@configclass
class ObservationsCfg:
    policy = ObservationsCfgPolicyCfg()
    critic = ObservationsCfgCriticCfg()


@configclass
class ActionsCfg:
    joint_pos = WalkJointActionCfg(
        class_type=WalkJointAction,
        asset_name="robot",
        debug_vis=False,
        clip={".*_joint": (-100.0, 100.0)},
        joint_names=[".*hip.*", ".*ankle.*", ".*knee.*"],
        scale=0.25,
        offset=0.0,
        preserve_order=False,
        use_default_offset=True,
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
                "x": (-1, 1.0),
                "y": (-1, 1.0),
                "z": (0.0, 0.0),
                "yaw": (-3.141592653589793, 3.141592653589793),
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
    push_robot = EventTermCfg(
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
    reset_joint = EventTermCfg(
        func=reset_joints_by_offset_specified_joint,
        params={
            "position_range": {
                ".*_hip_pitch_joint": (-0.41, 0.15),
                ".*_knee_joint": (-0.15, 0.2),
                ".*_ankle_pitch_joint": (-0.15, 0.15),
                ".*_elbow_joint": (-0.27, 0.15),
                ".*_shoulder_pitch_joint": (-0.48, 0.15),
                "waist_yaw_joint": (-0.1, 0.1),
                "left_hip_roll_joint": (-0.15, 0.3587),
                "right_hip_roll_joint": (-0.2049, 0.15),
                "waist_roll_joint": (-0.1, 0.1),
                ".*hip_yaw_joint": (-0.54, 0.37),
                "waist_pitch_joint": (-0.1, 0.3),
                ".*_shoulder_roll_joint": (-0.5006, 0.47),
                ".*ankle_roll_joint": (-0.25, 0.15),
                "left_shoulder_yaw_joint": (-0.37, 0.15),
                "right_shoulder_yaw_joint": (-0.15, 0.37),
                ".*wrist_.*": (-0.01, 0.01),
            },
            "velocity_range": (-0.5, 0.5),
        },
        mode="reset",
    )


@configclass
class G1Rewards:
    track_lin_vel_xy_exp = RewardTermCfg(
        func=track_lin_vel_xy_yaw_frame_exp, params={"command_name": "base_velocity", "std": 0.5}, weight=1.3
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=track_ang_vel_z_world_exp, params={"command_name": "base_velocity", "std": 1.0}, weight=1.2
    )
    lin_vel_z = RewardTermCfg(func=lin_vel_z, params={"asset_cfg": SceneEntityCfg(name="robot")}, weight=-2)
    ang_vel_xy = RewardTermCfg(func=ang_vel_xy, params={"asset_cfg": SceneEntityCfg(name="robot")}, weight=-0.15)
    orientation = RewardTermCfg(func=orientation_l2, params={"asset_cfg": SceneEntityCfg(name="robot")}, weight=-1.0)
    base_height = RewardTermCfg(
        func=base_height, params={"target_height": 0.78, "asset_cfg": SceneEntityCfg(name="robot")}, weight=-10
    )
    joint_acc_penalty = RewardTermCfg(
        func=joint_acc_with_action,
        params={"asset_cfg": SceneEntityCfg(name="robot"), "action_term_name": "joint_pos"},
        weight=-2.5e-07,
    )
    joint_vel_penalty = RewardTermCfg(
        func=joint_vel_with_action,
        params={"asset_cfg": SceneEntityCfg(name="robot"), "action_term_name": "joint_pos"},
        weight=-0.0015,
    )
    action_rate_penalty = RewardTermCfg(func=processed_action_rate_l2, params={"action_name": "joint_pos"}, weight=-0.1)
    joint_pos_limits = RewardTermCfg(
        func=joint_pos_limits,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*hip.*", ".*ankle.*", ".*knee.*"])},
        weight=-5,
    )
    alive = RewardTermCfg(func=is_alive, weight=0.2)
    hip_roll_yaw_penalty = RewardTermCfg(
        func=joint_deviation_l2,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*hip_roll.*", ".*hip_yaw.*"])},
        weight=-1,
    )
    contact_slip_penalty = RewardTermCfg(
        func=body_slipping_l2,
        params={"sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*ankle_roll.*"])},
        weight=-0.2,
    )
    feet_swing_height = RewardTermCfg(
        func=feet_swing_height,
        params={"sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*ankle_roll.*"])},
        weight=-20,
    )
    phase_contact = RewardTermCfg(
        func=phase_contact,
        params={"sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*ankle_roll.*"]), "period": 0.65},
        weight=0.18,
    )
    feet_air_time = RewardTermCfg(
        func=feet_air_time,
        params={
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*ankle_roll.*"]),
            "threshold": 0.5,
            "command_name": "base_velocity",
        },
        weight=0.1,
    )
    torque_penalty = RewardTermCfg(
        func=joint_torques,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*hip.*", ".*ankle.*", ".*knee.*"])},
        weight=-1e-05,
    )
    undesired_contact = RewardTermCfg(
        func=undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*hip.*", ".*knee.*"]),
            "threshold": 0.1,
        },
        weight=-1,
    )


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=time_out, time_out=True)
    height_low = TerminationTermCfg(func=root_height_below_minimum, params={"minimum_height": 0.35})


@configclass
class CurriculumCfg:
    terrain_levels = CurriculumTermCfg(func=terrain_levels_vel)


@configclass
class CommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        class_type=UniformVelocityCommand,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.05,
        rel_heading_envs=0.5,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.141592653589793, 3.141592653589793),
        ),
        planar_deadband=0.1,
    )


@configclass
class G1WalkEnvCfg(ManagerBasedRLEnvCfg):
    viewer = ViewerCfg(eye=(-5.0, 0, 2), lookat=(0.0, 0.0, 0.5))
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
    curriculum = CurriculumCfg()
    commands = CommandsCfg()
    height_scanner = None
    events_before_observations = True
