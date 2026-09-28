# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab.utils.string as string_utils




if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import NormalVelocityCommandCfg,  ClimbCommandCfg


class ClimbCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: ClimbCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: ClimbCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)
       

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        num_bodies = self.robot.num_bodies

        #self.box_height_range = env.cfg.scene.terrain.terrain_generator.sub_terrains["box"].box_height_range
        #self.box_height =self.box_height_range[0]*torch.ones(self.num_envs, device=self.device)
        #print(f"Box height range: {self.box_height_range}")
        self.climb_command = torch.zeros(self.num_envs, device=self.device)

        self.min_dist_to_box = None
        self.max_avg_height = torch.zeros(self.num_envs, device=self.device)
        self.max_avg_whole_height = torch.zeros(self.num_envs, device=self.device)
        self.min_avg_height = torch.ones(self.num_envs, device=self.device)
        #self.max_height_to_torso = -torch.ones(self.num_envs, device=self.device)
        self.max_com_x = torch.zeros(self.num_envs, device=self.device)
        self.min_com_x = torch.ones(self.num_envs, device=self.device)
        self.min_feet_com = torch.ones(self.num_envs,2, device=self.device)
        self.min_feet_avg_com = torch.ones(self.num_envs, device=self.device)
        self.max_feet_cosine = -torch.ones( self.num_envs,2, device=self.device)
        self.min_feet_angle = 3.14*torch.ones( self.num_envs,2, device=self.device)
        self.min_feet_fwd_angle = 3.14*torch.ones( self.num_envs,2, device=self.device)
        self.max_knee_joint = torch.zeros(self.num_envs,2, device=self.device)
        self.min_feet_box = 2*torch.ones(self.num_envs,2, device=self.device)
        self.mass = self.robot.root_physx_view.get_masses().clone().to(self.device)
        self.contact_count = torch.zeros(self.num_envs, device=self.device)
        self.entered_up_stage = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.consecutive_backward = torch.zeros(self.num_envs, device=self.device)
        self.min_torso_angle = 3.14*torch.ones(self.num_envs, device=self.device)
        self.min_upper_force = 500*torch.ones(self.num_envs, device=self.device)
        self.max_proj_x = torch.zeros(self.num_envs, device=self.device)
        self.min_head_height = 2*torch.ones(self.num_envs, device=self.device)

        self.filtered_upper_force = 500*torch.ones(self.num_envs, device=self.device)

        self.condition_reward_success_counter = 0

        self.standing_joint = torch.zeros(self._env.num_envs, self.robot.num_joints, device=self.device)
        # resolve the dictionary config
        index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.standing_joint, self.robot.joint_names)
        self.standing_joint[:, index_list] = torch.tensor(value_list, device=self.device)

        self.lying_joint = torch.zeros(self._env.num_envs, self.robot.num_joints, device=self.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.lying_joint, self.robot.joint_names)
        self.lying_joint[:, index_list] = torch.tensor(value_list, device=self.device)

        self.standup_joint = torch.zeros(self._env.num_envs, self.robot.num_joints, device=self.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.standup_joint, self.robot.joint_names)
        self.standup_joint[:, index_list] = torch.tensor(value_list, device=self.device)



    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "ClimbCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        # msg += f"\tHeading command: {self.cfg.heading_command}\n"
        # if self.cfg.heading_command:
        #     msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        # msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return self.climb_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass
        

   
    def _resample_command(self, env_ids: Sequence[int]):
        if hasattr(self._env, "reset_buf"):

            # If the reset buffer is present, use it to determine which environments to reset
            reset_buf = self._env.reset_buf
            reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        else:
            # If the reset buffer is not present, reset all environments
            reset_env_ids = torch.arange(self.num_envs, device=self.device)
        if self.cfg.climb_down:
            self.climb_command[env_ids] = 0
        else:
            self.climb_command[env_ids] = 1
        if self.cfg.activated:
            if self.cfg.climb_down:
                # If the climb down command is activated, set the command to 1
                self.climb_command[reset_env_ids] = 2
            else:
                self.climb_command[reset_env_ids] =  0
        
        

        self.mass.copy_(self.robot.root_physx_view.get_masses())

        # reset avg height of all rigid bodies
        box_height = self._env.scene.env_origins[reset_env_ids,2]
        self.max_avg_height[reset_env_ids] = torch.mean(self.robot.data.body_pos_w[reset_env_ids, :, 2].clip(max=box_height.unsqueeze(1)+0.02), dim=1)
        #self.max_avg_whole_height[reset_env_ids] = torch.mean(self.robot.data.body_pos_w[reset_env_ids, :, 2], dim=1)
        self.max_avg_whole_height[reset_env_ids] = torch.zeros_like(self.max_avg_whole_height[reset_env_ids])
        # self.min_avg_height[reset_env_ids] = torch.mean(self.robot.data.body_pos_w[reset_env_ids, :, 2].clip(max=box_height.unsqueeze(1)+0.02), dim=1)
        self.min_avg_height[reset_env_ids] = (torch.mean(self.robot.data.body_pos_w[reset_env_ids, :, 2].clip(max=box_height.unsqueeze(1)+0.02), dim=1)
                                            if self.cfg.reset_min_height_from_body else 1)        
        #self.max_height_to_torso[reset_env_ids] = -1
        if self.min_dist_to_box is None:
            self.min_dist_to_box = torch.abs(self.robot.data.body_pos_w[:, :, 2]).clone()
        else:
            self.min_dist_to_box[reset_env_ids] = torch.abs(self.robot.data.body_pos_w[reset_env_ids, :, 2])
        self.max_com_x[reset_env_ids] = -2# torch.sum(body_coms * self.mass[reset_env_ids], dim=1) / torch.sum(self.mass[reset_env_ids], dim=1)
        # body_coms = self.robot.data.body_pos_w[reset_env_ids, :, 0] - self._env.scene.env_origins[reset_env_ids, 0].unsqueeze(1)
        # self.min_com_x[reset_env_ids] = torch.sum(body_coms * self.mass[reset_env_ids], dim=1) / torch.sum(self.mass[reset_env_ids], dim=1)
        self.min_feet_box[reset_env_ids] = 2
        self.min_feet_com[reset_env_ids] = 1
        self.min_feet_avg_com[reset_env_ids] = 1
        self.max_feet_cosine[reset_env_ids] = -1
        self.min_feet_angle[reset_env_ids] = 3.14
        self.min_feet_fwd_angle[reset_env_ids] = 3.14
        self.max_knee_joint[reset_env_ids] = 0
        self.consecutive_backward[reset_env_ids] = 0
        self.contact_count[reset_env_ids] = 0
        self.min_torso_angle[reset_env_ids] = 3.14
        self.min_upper_force[reset_env_ids] = 500
        self.filtered_upper_force[reset_env_ids] = 500
        self.max_proj_x[reset_env_ids] = 0
        self.min_head_height[reset_env_ids] = 2


        

        #self.box_height[reset_env_ids] = self._env.scene.terrain.terrain_levels[reset_env_ids] * (self.box_height_range[1] - self.box_height_range[0]) + self.box_height_range[0]

    
    def update_min_head_height(self, head_height):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.min_head_height = torch.min(self.min_head_height, head_height,)

    def update_max_proj_x(self, proj_x):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.max_proj_x = torch.max(self.max_proj_x, proj_x,)
        
    def update_contact_count(self, contact):
        #contact count +1 if contact is True
        self.contact_count += contact.float()

    def update_max_com_x(self, current_com: torch.Tensor):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.max_com_x = torch.max(self.max_com_x, current_com,)

    def update_min_com_x(self, current_com: torch.Tensor):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.min_com_x = torch.min(self.min_com_x, current_com,)

    def update_min_feet_box(self, feet_dist,):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.min_feet_box = torch.min(self.min_feet_box, feet_dist,)

    def update_min_feet_com(self, feet_com_dist, on_box):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        min_feet_com = torch.min(self.min_feet_com, feet_com_dist,)
        self.min_feet_com = torch.where(on_box.unsqueeze(-1), min_feet_com, self.min_feet_com)

    def update_min_feet_avg_com(self, feet_com_dist, on_box):

        self.current_feet_avg_com = feet_com_dist.clone()
        min_feet_avg_com = torch.min(self.min_feet_avg_com, feet_com_dist,)
        self.min_feet_avg_com = torch.where(on_box, min_feet_avg_com, self.min_feet_avg_com)

    def update_max_feet_cosine(self, cosine, on_box):

        # update the max height
        max_feet_cosine = torch.max(self.max_feet_cosine, cosine,)
        self.max_feet_cosine = torch.where(on_box, max_feet_cosine, self.max_feet_cosine)

    def update_min_feet_angle(self, angle, on_box):

        min_feet_angle = torch.min(self.min_feet_angle, angle,)
        self.min_feet_angle = torch.where(on_box, min_feet_angle, self.min_feet_angle)
    
    def update_min_torso_angle(self, angle, on_box):

        min_torso_angle = torch.min(self.min_torso_angle, angle,)
        self.min_torso_angle = torch.where(on_box, min_torso_angle, self.min_torso_angle)

    def update_min_upper_force(self, force, on_box):

        min_upper_force = torch.min(self.min_upper_force, force,)
        self.min_upper_force = torch.where(on_box, min_upper_force, self.min_upper_force)
        self.filtered_upper_force = torch.where(on_box, force, self.filtered_upper_force)

    def update_min_feet_fwd_angle(self, angle, on_box):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        min_feet_fwd_angle = torch.min(self.min_feet_fwd_angle, angle,)
        self.min_feet_fwd_angle = torch.where(on_box, min_feet_fwd_angle, self.min_feet_fwd_angle)

    def update_max_knee_joint(self, knee_joint_pos, on_box):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        max_knee_joint = torch.max(self.max_knee_joint, knee_joint_pos,)
        self.max_knee_joint = torch.where(on_box, max_knee_joint, self.max_knee_joint)

    def update_min_dist_to_box(self, current_dist: torch.Tensor):
        """Update the minimum distance to the box in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the min distance
        if self.min_dist_to_box is None:
            self.min_dist_to_box = current_dist.clone()
        else:      
            self.min_dist_to_box = torch.min(self.min_dist_to_box, current_dist,)

    def update_max_avg_height(self, current_avg: torch.Tensor):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.max_avg_height = torch.max(self.max_avg_height, current_avg,)

    def update_max_avg_whole_height(self, current_avg: torch.Tensor):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.max_avg_whole_height = torch.max(self.max_avg_whole_height, current_avg,)

    def update_min_avg_height(self, current_avg: torch.Tensor):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.min_avg_height = torch.min(self.min_avg_height, current_avg,)

    def update_consecutive_fail(self, penalty):
        """Update the consecutive backward count of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        #print('penalty:',penalty)
        fail_envs = penalty.nonzero(as_tuple=False)
        #print('fail_envs:',fail_envs)
        self.consecutive_backward[fail_envs] += 1
        success_envs =torch.where(penalty==0)[0]
        #print('success_envs:',success_envs)
        self.consecutive_backward[success_envs] = 0
    # def update_max_height_to_torso(self, current_avg: torch.Tensor):
    #     """Update the average height of the robot in the environment.

    #     Args:
    #         env_ids: The indices of the environments to update.
    #     """
    #     # update the max height
    #     self.max_height_to_torso = torch.max(self.max_height_to_torso, current_avg,)

    def update_entered_up_stage(self, entered):
        """Update the entered up stage flag of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        self.entered_up_stage = entered

    def _update_command(self):
        if self.cfg.activated:
            if self.cfg.climb_down:
                start_climb_ids = ((self._env.episode_length_buf >= 50) & (self.climb_command==2)).nonzero().flatten()
                self.climb_command[start_climb_ids] = 1
       
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        target_pos_w=self.robot.data.root_link_pos_w.clone()
        target_pos_w[:,2] -= 0.2
        # -- resolve the scales and quaternions
        #
        robot_to_target_quat=math_utils.quat_from_euler_xyz(torch.zeros_like(target_pos_w[:,0])+0.5*torch.pi,torch.zeros_like(target_pos_w[:,0]),torch.zeros_like(target_pos_w[:,0]))
        default_scale = self.target_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(target_pos_w.shape[0], 1)
        arrow_scale = torch.where(self.entered_up_stage.unsqueeze(1), arrow_scale, torch.zeros_like(arrow_scale))
        # display markers
        self.target_visualizer.visualize(target_pos_w, robot_to_target_quat,arrow_scale)
   