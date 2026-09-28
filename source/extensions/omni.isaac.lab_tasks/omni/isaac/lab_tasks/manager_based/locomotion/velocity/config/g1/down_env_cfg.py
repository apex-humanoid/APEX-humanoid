# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Climbdown training configuration.

Ordinary Isaac Lab configclasses; edit these values to train a new experiment.
"""
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators.actuator_cfg import IdealPDActuatorCfg
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.envs.common import ViewerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from omni.isaac.lab.envs.mdp.commands.commands_cfg import ClimbCommandCfg
from omni.isaac.lab.envs.ui.manager_based_rl_env_window import ManagerBasedRLEnvWindow
from omni.isaac.lab.managers import ObservationGroupCfg
from omni.isaac.lab.managers.manager_term_cfg import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from omni.isaac.lab.managers.recorder_manager import RecorderManagerBaseCfg
from omni.isaac.lab.managers.scene_entity_cfg import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.contact_sensor.contact_ground_sensor_z_cfg import ContactGroundSensorZCfg
from omni.isaac.lab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from omni.isaac.lab.sensors.contact_sensor.contact_sensor_z_cfg import ContactSensorZCfg
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
from omni.isaac.lab.utils.noise.noise_cfg import UniformEulerNoiseOnQuatCfg, UniformNoiseCfg
from omni.isaac.lab_assets.unitree import G1_USD_PATH
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.events import (
    push_by_adding_velocity,
    randomize_joint_default_pos_climbup,
    randomize_rigid_body_com,
    randomize_rigid_body_mass_climbdown,
    randomize_rigid_body_material,
    reset_joints_to_lying_by_offset,
    reset_root_state_uniform,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.locomotion_curriculums import terrain_levels_height_down
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.locomotion_rewards import (
    base_lin_ang_acc_climbdown,
    body_pressure_down,
    com_forward_penalty,
    down_wait_penalty,
    hip_yaw_roll_joint_deviation,
    joint_velocity_limits,
    standing_ang_vel_down,
    standing_height_down,
    standing_lin_vel_down,
    upward_penalty,
    waist_joint_deviation,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import (
    base_ang_vel,
    body_ang_vel_w,
    body_lin_vel_w,
    body_pos_w,
    body_quat_w,
    box_height,
    height_scan,
    joint_pos_rel_climbup,
    joint_vel_rel,
    progress_memory_down,
    projected_gravity,
    real_last_processed_action,
    time,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.rewards import (
    ang_vel_xy_l2,
    body_lin_acc_l2,
    body_slipping_contact,
    contact_forces_exp,
    is_alive,
    is_terminated_climbdown,
    joint_acc_l2,
    joint_pos_limits,
    joint_torques_l2,
    joint_vel_exp,
    joint_vel_l2_climbdown,
    power_consumption,
    processed_action_rate_l2,
    standing_flat_orientation_down,
    standing_joint_deviation_down,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.terminations import (
    knee_straight_down,
    root_height_below_minimum,
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
            curriculum=True,
            size=(4.0, 4.0),
            border_width=2,
            num_rows=10,
            num_cols=20,
            sub_terrains={
                "box": MeshChangingBoxTerrainCfg(
                    size=(4.0, 4.0), box_height_range=(0.55, 0.84), platform_width_range=(3.0, 3.0)
                )
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
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        update_period=0.02,
        mesh_prim_paths=["/World/ground"],
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=GridPatternCfg(resolution=0.04, size=[1.0, 1.0]),
        drift_mode="split_xy_z",
        apply_drift=False,
        compute_no_drift=False,
    )
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", update_period=0.005, history_length=3, track_air_time=True
    )
    bodies_ground_contact = ContactGroundSensorZCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True
    )
    contact_forces_z = ContactSensorZCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True)
    contact_on_torso = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        history_length=1,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/pelvis",
            "/World/envs/env_.*/Robot/left_hip_pitch_link",
            "/World/envs/env_.*/Robot/left_hip_roll_link",
            "/World/envs/env_.*/Robot/left_hip_yaw_link",
            "/World/envs/env_.*/Robot/left_knee_link",
            "/World/envs/env_.*/Robot/left_ankle_pitch_link",
            "/World/envs/env_.*/Robot/left_ankle_roll_link",
            "/World/envs/env_.*/Robot/pelvis_contour_link",
            "/World/envs/env_.*/Robot/right_hip_pitch_link",
            "/World/envs/env_.*/Robot/right_hip_roll_link",
            "/World/envs/env_.*/Robot/right_hip_yaw_link",
            "/World/envs/env_.*/Robot/right_knee_link",
            "/World/envs/env_.*/Robot/right_ankle_pitch_link",
            "/World/envs/env_.*/Robot/right_ankle_roll_link",
            "/World/envs/env_.*/Robot/waist_yaw_link",
            "/World/envs/env_.*/Robot/waist_roll_link",
            "/World/envs/env_.*/Robot/torso_link",
            "/World/envs/env_.*/Robot/head_link",
            "/World/envs/env_.*/Robot/left_shoulder_pitch_link",
            "/World/envs/env_.*/Robot/left_shoulder_roll_link",
            "/World/envs/env_.*/Robot/left_shoulder_yaw_link",
            "/World/envs/env_.*/Robot/left_elbow_link",
            "/World/envs/env_.*/Robot/left_wrist_roll_link",
            "/World/envs/env_.*/Robot/left_wrist_pitch_link",
            "/World/envs/env_.*/Robot/left_wrist_yaw_link",
            "/World/envs/env_.*/Robot/right_shoulder_pitch_link",
            "/World/envs/env_.*/Robot/right_shoulder_roll_link",
            "/World/envs/env_.*/Robot/right_shoulder_yaw_link",
            "/World/envs/env_.*/Robot/right_elbow_link",
            "/World/envs/env_.*/Robot/right_wrist_roll_link",
            "/World/envs/env_.*/Robot/right_wrist_pitch_link",
            "/World/envs/env_.*/Robot/right_wrist_yaw_link",
            "/World/envs/env_.*/Robot/waist_support_link",
        ],
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
    history_length = None
    flatten_history_dim = True
    root_ang_vel = ObservationTermCfg(func=base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2), history_length=6)
    projected_gravity = ObservationTermCfg(
        func=projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05), history_length=6
    )
    joint_pos = ObservationTermCfg(
        func=joint_pos_rel_climbup, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01), history_length=6
    )
    joint_vel = ObservationTermCfg(func=joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5), history_length=6)
    actions = ObservationTermCfg(func=real_last_processed_action, params={"action_name": "joint_pos"}, history_length=6)
    height_scan = ObservationTermCfg(
        func=height_scan,
        params={"sensor_cfg": SceneEntityCfg(name="height_scanner"), "offset": 0.0},
        noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
    )


@configclass
class ObservationsCfgCriticCfg(ObservationGroupCfg):
    concatenate_terms = True
    enable_corruption = False
    history_length = None
    flatten_history_dim = True
    torso_lin_vel = ObservationTermCfg(
        func=body_lin_vel_w,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=["torso_link"])},
        noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        history_length=6,
    )
    torso_ang_vel = ObservationTermCfg(
        func=body_ang_vel_w,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=["torso_link"])},
        noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        history_length=6,
    )
    torso_quat = ObservationTermCfg(
        func=body_quat_w,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=["torso_link"]), "make_quat_unique": True},
        noise=UniformEulerNoiseOnQuatCfg(n_min=[-0.05, -0.05, -0.05], n_max=[0.05, 0.05, 0.05]),
        history_length=6,
    )
    projected_gravity = ObservationTermCfg(
        func=projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05), history_length=6
    )
    torso_pos = ObservationTermCfg(
        func=body_pos_w,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=["torso_link"])},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        history_length=6,
    )
    joint_pos = ObservationTermCfg(
        func=joint_pos_rel_climbup, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01), history_length=6
    )
    joint_vel = ObservationTermCfg(func=joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5), history_length=6)
    actions = ObservationTermCfg(func=real_last_processed_action, params={"action_name": "joint_pos"}, history_length=6)
    box_height = ObservationTermCfg(func=box_height, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01))
    progress_memory = ObservationTermCfg(func=progress_memory_down, history_length=1)
    time = ObservationTermCfg(func=time, history_length=1)
    height_scan = ObservationTermCfg(
        func=height_scan,
        params={"sensor_cfg": SceneEntityCfg(name="height_scanner"), "offset": 0.0},
        noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
    )


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
        func=randomize_rigid_body_mass_climbdown,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names="torso_link"),
            "mass_distribution_params": (-1, 1),
            "operation": "add",
        },
        mode="startup",
    )
    add_joint_default_pos = EventTermCfg(
        func=randomize_joint_default_pos_climbup,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
        mode="startup",
    )
    reset_base = EventTermCfg(
        func=reset_root_state_uniform,
        params={
            "pose_range": {
                "x": (-0.7, -0.55),
                "y": (-0.6, 0.6),
                "z": (0.05, 0.05),
                "yaw": (-0.2617993877991494, 0.2617993877991494),
                "roll": (-1.2566370614359172, 1.2566370614359172),
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
    reset_robot_joints = EventTermCfg(
        func=reset_joints_to_lying_by_offset,
        params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0)},
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


@configclass
class G1Rewards:
    standing_joint = RewardTermCfg(
        func=standing_joint_deviation_down,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"], body_names=[".*_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg(name="bodies_ground_contact", body_names=[".*_ankle_roll_link"]),
        },
        weight=1,
    )
    standing_orientation = RewardTermCfg(
        func=standing_flat_orientation_down,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=["torso_link"]),
            "feet_cfg": SceneEntityCfg(name="robot", body_names=[".*_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg(name="bodies_ground_contact", body_names=[".*_ankle_roll_link"]),
        },
        weight=1,
    )
    standing_lin_vel = RewardTermCfg(
        func=standing_lin_vel_down,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=["torso_link"]),
            "feet_cfg": SceneEntityCfg(name="robot", body_names=[".*_ankle_roll_link"]),
        },
        weight=1,
    )
    standing_ang_vel = RewardTermCfg(
        func=standing_ang_vel_down,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=["torso_link"]),
            "feet_cfg": SceneEntityCfg(name="robot", body_names=[".*_ankle_roll_link"]),
        },
        weight=1,
    )
    standing_height = RewardTermCfg(
        func=standing_height_down,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*shoulder_pitch_link"]),
            "desired_height": 1.07,
            "feet_cfg": SceneEntityCfg(name="robot", body_names=[".*_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg(name="bodies_ground_contact", body_names=[".*_ankle_roll_link"]),
        },
        weight=1,
    )
    upward_penalty = RewardTermCfg(
        func=upward_penalty,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg(name="bodies_ground_contact", body_names=[".*_ankle_roll_link"]),
        },
        weight=-4,
    )
    forward_penalty = RewardTermCfg(
        func=com_forward_penalty,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*_ankle_roll_link"]),
            "upper_cfg": SceneEntityCfg(name="robot", body_names=[".*wrist.*", ".*shoulder.*", ".*elbow_link"]),
            "upper_sensor_cfg": SceneEntityCfg(
                name="bodies_ground_contact", body_names=[".*wrist.*", ".*shoulder.*", ".*elbow_link"]
            ),
            "sensor_cfg": SceneEntityCfg(name="bodies_ground_contact", body_names=[".*_ankle_roll_link"]),
        },
        weight=-4,
    )
    alive_reward = RewardTermCfg(func=is_alive, weight=15)
    early_termination = RewardTermCfg(func=is_terminated_climbdown, weight=-800.0)
    wait_penalty = RewardTermCfg(func=down_wait_penalty, params={"command_name": "target_pos_e"}, weight=-1)
    pressure_penalty = RewardTermCfg(
        func=body_pressure_down,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg(name="bodies_ground_contact", body_names=[".*_ankle_roll_link"]),
        },
        weight=-1,
    )
    contact_exp_penalty = RewardTermCfg(
        func=contact_forces_exp,
        params={
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*"]),
            "threshold": 500.0,
            "grad_scale": 0.005,
        },
        weight=-1,
    )
    head_contact_penalty = RewardTermCfg(
        func=contact_forces_exp,
        params={
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["head_link"]),
            "threshold": 0,
            "grad_scale": 0.1,
        },
        weight=-1,
    )
    body_slipping_penalty = RewardTermCfg(
        func=body_slipping_contact,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*"]),
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*"]),
        },
        weight=-0.5,
    )
    action_rate_penalty = RewardTermCfg(func=processed_action_rate_l2, params={"action_name": "joint_pos"}, weight=-0.2)
    hip_joint_penalty = RewardTermCfg(
        func=hip_yaw_roll_joint_deviation,
        params={
            "hip_yaw_cfg": SceneEntityCfg(name="robot", joint_names=[".*hip_yaw_joint"]),
            "hip_roll_cfg": SceneEntityCfg(name="robot", joint_names=[".*hip_roll_joint"]),
            "hip_pitch_cfg": SceneEntityCfg(name="robot", joint_names=[".*_hip_pitch_joint"]),
        },
        weight=-6,
    )
    waist_joint_penalty = RewardTermCfg(
        func=waist_joint_deviation,
        params={"waist_cfg": SceneEntityCfg(name="robot", joint_names=[".*waist_yaw_joint"])},
        weight=-6,
    )
    joint_pos_limits = RewardTermCfg(func=joint_pos_limits, weight=-10)
    joint_vel_penalty = RewardTermCfg(
        func=joint_vel_l2_climbdown,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])},
        weight=-0.001,
    )
    joint_vel_exp_penalty = RewardTermCfg(
        func=joint_vel_exp,
        params={"grad_scale": 0.6, "threshold": 10, "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])},
        weight=-1,
    )
    joint_vel_lim_penalty = RewardTermCfg(
        func=joint_velocity_limits,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])},
        weight=-10,
    )
    joint_acc_penalty = RewardTermCfg(
        func=joint_acc_l2, params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])}, weight=-2e-08
    )
    torque_penalty = RewardTermCfg(
        func=joint_torques_l2, params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])}, weight=-1.5e-05
    )
    power_penalty = RewardTermCfg(func=power_consumption, weight=-1e-05)
    ang_vel_xy_l2 = RewardTermCfg(func=ang_vel_xy_l2, weight=-0.005)
    base_acc_penalty = RewardTermCfg(
        func=base_lin_ang_acc_climbdown,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=["torso_link"])},
        weight=-0.0001,
    )
    rigid_body_acc_penalty = RewardTermCfg(
        func=body_lin_acc_l2, params={"asset_cfg": SceneEntityCfg(name="robot", body_names=[".*"])}, weight=-0.0002
    )


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=time_out, time_out=True)
    knee_straight = TerminationTermCfg(
        func=knee_straight_down, params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*_knee_joint"])}
    )
    height_low = TerminationTermCfg(func=root_height_below_minimum, params={"minimum_height": 0.45})


@configclass
class DownHeightCurriculumCfg:
    terrain_levels = CurriculumTermCfg(
        func=terrain_levels_height_down,
        params={
            "update_prob": 0.8,
            "desired_height": 0.78,
            "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg(name="bodies_ground_contact", body_names=[".*_ankle_roll_link"]),
        },
    )


@configclass
class ClimbCommandsCfg:
    climb_command = ClimbCommandCfg(
        resampling_time_range=(6.0, 6.0), activated=False, climb_down=True, reset_min_height_from_body=False
    )


@configclass
class G1DownEnvCfg(ManagerBasedRLEnvCfg):
    viewer = ViewerCfg(eye=(-1.0, 2.5, 1), lookat=(-1, 0.0, 0.5))
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
    episode_length_s = 7
    rewards = G1Rewards()
    terminations = TerminationsCfg()
    curriculum = DownHeightCurriculumCfg()
    commands = ClimbCommandsCfg()
    events_before_observations = False
