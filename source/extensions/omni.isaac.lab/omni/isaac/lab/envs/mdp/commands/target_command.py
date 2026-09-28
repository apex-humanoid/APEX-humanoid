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



if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import NormalVelocityCommandCfg,  TargetCommandCfg


class TargetCommand(CommandTerm):
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

    cfg: TargetCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: TargetCommandCfg, env: ManagerBasedEnv):
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

        # crete buffers to store the command
        # -- command: x, y, z(=0) in env coordinates
        self.target_command_e = torch.zeros(self.num_envs, 3, device=self.device)
        self.radius_range = cfg.radius_range
        self.max_avg_height = torch.zeros(self.num_envs, device=self.device)
        self.max_com_x = torch.zeros(self.num_envs, device=self.device)
        self.mass = self.robot.root_physx_view.get_masses().clone().to(self.device)
       
        # # -- metrics
        # self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)


    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "TargetCommand:\n"
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
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.target_command_e

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        target_pos_w=self.target_command_e+self._env.scene.env_origins
        target_pos_w[:,2]=0
        
        remaining_distance = torch.norm(target_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        near = remaining_distance < self.cfg.success_threshold

        sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link", body_ids=[6,12])
        contact_sensor: ContactSensor = self._env.scene.sensors[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        contact=torch.all(torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] 
                      > 1, dim=1)
        
        successes=near

        self.metrics["consecutive_success"] += successes.float()

        

    # def _resample_command(self, env_ids: Sequence[int]):
    #     num_envs = len(env_ids)
    #     R=self.radius_range[1]
    #     r=self.radius_range[0]
    #     # 随机生成半径，范围在内径和外径之间
    #     radii = torch.sqrt(torch.rand(num_envs,device=self.device) * (R**2 - r**2) + r**2)
    #     # 随机生成角度
    #     angles = torch.rand(num_envs,device=self.device) * 2 * torch.pi
    #     # 计算点的相对坐标
    #     x_offsets = radii * torch.cos(angles)
    #     y_offsets = radii * torch.sin(angles)
    #     # 生成最终的点
    #     self.target_command_e[env_ids,:2]= torch.stack((x_offsets, y_offsets), dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        num_envs = len(env_ids)
        
        R=self.radius_range[1]
        r=self.radius_range[0]
        # 随机生成半径，范围在内径和外径之间
        #radii = torch.sqrt(torch.rand(num_envs,device=self.device) * (R**2 - r**2) + r**2)
        # 随机生成角度
        radii = torch.rand(num_envs,device=self.device) * (R - r) + r

        # 计算点的相对坐标
        x_offsets = -radii 
        y_offsets = self.robot.data.root_link_pos_w[env_ids,1] - self._env.scene.env_origins[env_ids,1]
        # 生成最终的点
        self.target_command_e[env_ids,:2]= torch.stack((x_offsets, y_offsets), dim=1)


        self.mass.copy_(self.robot.root_physx_view.get_masses())

        # reset avg height of all rigid bodies
        self.max_avg_height[env_ids] = torch.mean(self.robot.data.body_pos_w[env_ids, :, 2].clip(max=0.02), dim=1)

        body_coms = self.robot.data.body_pos_w[env_ids, :, 0] - self._env.scene.env_origins[env_ids, 0].unsqueeze(1)
        self.max_com_x[env_ids] =  torch.sum(body_coms * self.mass[env_ids], dim=1) / torch.sum(self.mass[env_ids], dim=1)


    def update_max_com_x(self, current_com: torch.Tensor):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.max_com_x = torch.max(self.max_com_x, current_com,)

    def update_max_avg_height(self, current_avg: torch.Tensor):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.max_avg_height = torch.max(self.max_avg_height, current_avg,)

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        pass
       
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
        target_pos_w=self._env.scene.env_origins+self.target_command_e
        target_pos_w[:,2]=1
        # -- resolve the scales and quaternions
        #
        robot_to_target_quat=math_utils.quat_from_euler_xyz(torch.zeros_like(target_pos_w[:,0])+0.5*torch.pi,torch.zeros_like(target_pos_w[:,0]),torch.zeros_like(target_pos_w[:,0]))
        default_scale = self.target_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(target_pos_w.shape[0], 1)
        # display markers
        self.target_visualizer.visualize(target_pos_w, robot_to_target_quat,arrow_scale)
