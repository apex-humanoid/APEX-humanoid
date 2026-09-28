# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils import configclass

from .null_command import NullCommand
from .pose_2d_command import TerrainBasedPose2dCommand, UniformPose2dCommand
from .pose_command import UniformPoseCommand
from .velocity_command import NormalVelocityCommand, UniformVelocityCommand
from .target_command import TargetCommand
from .climb_command import ClimbCommand


@configclass
class NullCommandCfg(CommandTermCfg):
    """Configuration for the null command generator."""

    class_type: type = NullCommand

    def __post_init__(self):
        """Post initialization."""
        # set the resampling time range to infinity to avoid resampling
        self.resampling_time_range = (math.inf, math.inf)

@configclass
class ClimbCommandCfg(CommandTermCfg):
    reset_min_height_from_body: bool = False
    """Initialize standup progress from body heights on reset."""

    class_type: type= ClimbCommand
    asset_name: str = 'robot'
    activated: bool = True
    climb_down: bool = False
    target_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/target_pos"
    )
    target_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    standing_joint:dict[str, float] = {
            # ".*_hip_pitch_joint": -0.2,
            # ".*_knee_joint": 0.42,
            # ".*_ankle_pitch_joint": -0.23,
            # ".*_elbow_joint": 0.0,
            # "left_shoulder_roll_joint": 0.16,
            # "left_shoulder_pitch_joint": -0.35,
            # "right_shoulder_roll_joint": -0.16,
            # "right_shoulder_pitch_joint": -0.35,
            
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,

        }
    lying_joint:dict[str, float] = {
            ".*_hip_pitch_joint": -1.5,
            "left_hip_roll_joint": 0.3,
            "right_hip_roll_joint": -0.3, 
            ".*_knee_joint": 1.5,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": 0.0,
            "left_shoulder_roll_joint": 0.5,
            "left_shoulder_pitch_joint": -1.8,
            "right_shoulder_roll_joint": -0.5,
            "right_shoulder_pitch_joint": -1.8,
        }
    
    standup_joint:dict[str, float] = {
    "left_hip_pitch_joint": -0.9861,
    "right_hip_pitch_joint": -1.5081,
    "waist_yaw_joint": 0.2382,
    "left_hip_roll_joint": 0.1,
    "right_hip_roll_joint": 0.0987,
    "waist_roll_joint": -0.1870,
    "left_hip_yaw_joint": 0.2,
    "right_hip_yaw_joint": 0.2447,
    "waist_pitch_joint": 0.5199,
    "left_knee_joint": 0.1316,
    "right_knee_joint": 0.4283,
    "left_shoulder_pitch_joint": -1.9388,
    "right_shoulder_pitch_joint": -2.0780,
    "left_ankle_pitch_joint": -0.7568,
    "right_ankle_pitch_joint": -0.6016,
    "left_shoulder_roll_joint": 0.1490,
    "right_shoulder_roll_joint": -0.2497,
    "left_ankle_roll_joint": 0.2618,
    "right_ankle_roll_joint": 0.2599,
    "left_shoulder_yaw_joint": 0.3949,
    "right_shoulder_yaw_joint": -0.7154,
    "left_elbow_joint": 1.1275,
    "right_elbow_joint": 1.1890,
    "left_wrist_roll_joint": 0.0014,
    "right_wrist_roll_joint": 0.0172,
    "left_wrist_pitch_joint": 0.0353,
    "right_wrist_pitch_joint": -0.0013,
    "left_wrist_yaw_joint": 0.0063,
    "right_wrist_yaw_joint": -0.0062,
}



@configclass
class TargetCommandCfg(CommandTermCfg):
    class_type: type= TargetCommand
    asset_name: str = 'robot'
    radius_range: tuple[float, float] = (2.0, 5.0)

    target_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/target_pos"
    )
    target_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    success_threshold: float = 0.06

@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    planar_deadband: float = 0.0
    """Zero sampled planar velocity components below this magnitude."""

    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class NormalVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = NormalVelocityCommand
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.

    @configclass
    class Ranges:
        """Normal distribution ranges for the velocity commands."""

        mean_vel: tuple[float, float, float] = MISSING
        """Mean velocity for the normal distribution (in m/s).

        The tuple contains the mean linear-x, linear-y, and angular-z velocity.
        """

        std_vel: tuple[float, float, float] = MISSING
        """Standard deviation for the normal distribution (in m/s).

        The tuple contains the standard deviation linear-x, linear-y, and angular-z velocity.
        """

        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class UniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class UniformPose2dCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = UniformPose2dCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    """The configuration for the goal pose visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.2, 0.2, 0.8)
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)


@configclass
class TerrainBasedPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for the terrain-based position command generator."""

    class_type = TerrainBasedPose2dCommand

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the sampled commands."""
