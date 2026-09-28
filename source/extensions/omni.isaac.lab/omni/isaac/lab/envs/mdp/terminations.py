# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
import omni.isaac.lab.utils.math as math_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length

def resample_on_collision(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, active=True) -> torch.Tensor:
    """Terminate the episode when the specified body parts collide with the environment."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    # check if any contact force exceeds the threshold
    collision = torch.any(
        torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1) > 1.0, dim=1)
    # print('collision:', collision)
    
    return collision & (env.episode_length_buf <= 1)


def command_resample(env: ManagerBasedRLEnv, command_name: str, num_resamples: int = 1) -> torch.Tensor:
    """Terminate the episode based on the total number of times commands have been re-sampled.

    This makes the maximum episode length fluid in nature as it depends on how the commands are
    sampled. It is useful in situations where delayed rewards are used :cite:`rudin2022advanced`.
    """
    command: CommandTerm = env.command_manager.get_term(command_name)
    return torch.logical_and((command.time_left <= env.step_dt), (command.command_counter == num_resamples))


"""
Root terminations.
"""


def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle


def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, active=True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    #return (asset.data.root_link_pos_w[:, 2]-env.scene.env_origins[:,2]) < minimum_height
    return ((asset.data.root_link_pos_w[:, 2]) < minimum_height)*active

def foot_on_ground(
    env: ManagerBasedRLEnv, active=True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    #return (asset.data.root_link_pos_w[:, 2]-env.scene.env_origins[:,2]) < minimum_height
    feet_near_grd = asset.data.body_pos_w[:,asset_cfg.body_ids, 2].max(dim=-1)[0] < 0.3
    #too_back = ((asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).min(dim=1)[0]) < -1.4
    ending = env.episode_length_buf>150
    # input('Press Enter to continue...')  # Debugging line to pause execution
    # print('episode step:',env.episode_length_buf)
    # print('min x:',(asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).min(dim=1)[0])
    return torch.logical_and(feet_near_grd, ending)*active

def foot_off_ground(
    env: ManagerBasedRLEnv, active=True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    #return (asset.data.root_link_pos_w[:, 2]-env.scene.env_origins[:,2]) < minimum_height
    foot_off_grd = (asset.data.body_pos_w[:,asset_cfg.body_ids, 2].max(dim=-1)[0] - env.scene.env_origins[:, 2]) > 0.25
    return foot_off_grd

def x_too_back(
    env: ManagerBasedRLEnv, active =True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    #return (asset.data.root_link_pos_w[:, 2]-env.scene.env_origins[:,2]) < minimum_height
    #print('min x:',(asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).min(dim=1)[0])
    too_back = ((asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).min(dim=1)[0]) < -1.4
    ending = env.episode_length_buf>100
    return torch.logical_and(too_back, ending)*active

def x_too_back_down(
    env: ManagerBasedRLEnv, active =True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    #return (asset.data.root_link_pos_w[:, 2]-env.scene.env_origins[:,2]) < minimum_height
    #print('min x:',(asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).min(dim=1)[0])
    too_back = ((asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).min(dim=1)[0]) < -1.4
    # print('x pos:', (asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).min(dim=1)[0])
    return too_back*active

def knee_straight_down(
    env: ManagerBasedRLEnv,  active = True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's leg is stretched.

    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    knee_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # print('knee pos:', knee_pos)
    return ((knee_pos < 0.3).all(dim=1)*(env.episode_length_buf<50))|((knee_pos < 0.1).all(dim=1))

def bend_over(
    env: ManagerBasedRLEnv,  active = True, torso_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's torso bend over.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[torso_cfg.name]

    torso_quat = asset.data.body_quat_w[:, torso_cfg.body_ids, :].squeeze(1)
    torso_proj_grav = math_utils.quat_rotate_inverse(torso_quat, asset.data.GRAVITY_VEC_W)
    torso_cosine = (-torso_proj_grav[:,2])
    #return (torso_angle>2.3)*active
    return (torso_cosine < -0.666276)*active  

def leg_stretched(
    env: ManagerBasedRLEnv,  active = True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's leg is stretched.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    CoM = (asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2] )
    feet_pos = asset.data.body_pos_w[:,asset_cfg.body_ids, :2] - env.scene.env_origins[:,:2].unsqueeze(1)
    feet_CoM_dist = torch.linalg.norm(feet_pos - CoM.unsqueeze(1), dim=-1)
    print('feet_CoM_dist values:', feet_CoM_dist)
    print('feet_CoM_dist:', (feet_CoM_dist>0.6).any(dim=1))
    return (feet_CoM_dist>0.6).any(dim=1)

def leg_stretched_down(
    env: ManagerBasedRLEnv,  active = True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's leg is stretched.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    CoM = (asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2] )
    feet_pos = asset.data.body_pos_w[:,asset_cfg.body_ids, :2] - env.scene.env_origins[:,:2].unsqueeze(1)
    feet_CoM_dist = torch.linalg.norm(feet_pos - CoM.unsqueeze(1), dim=-1)
    
    return (feet_CoM_dist>0.65).any(dim=1)

def lean_back(
    env: ManagerBasedRLEnv,  torso_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's torso lean back.


    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[torso_cfg.name]
    proj_grav = asset.data.projected_gravity_b[:,0]
    return proj_grav < -0.5

def lean_back_lydown(
    env: ManagerBasedRLEnv,  torso_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's torso lean back.


    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[torso_cfg.name]
    proj_grav = asset.data.projected_gravity_b[:,0]
    return proj_grav < -0.15

def lose_balance(
    env: ManagerBasedRLEnv,  shoulder_cfg,upper_sensor_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[upper_sensor_cfg.name]
    net_contact_forces = contact_sensor._data.force_matrix_w[:, upper_sensor_cfg.body_ids,0, 2]
    upper_contact = torch.any(net_contact_forces > 1.0, dim=1)
    body_mean_feet_xy_dist = torch.linalg.norm(asset.data.body_pos_w[:, shoulder_cfg.body_ids, :2].mean(dim=1) 
                                               - asset.data.body_pos_w[:, asset_cfg.body_ids, :2].mean(dim=1), dim=1)
    mean_height = torch.mean(asset.data.body_pos_w[:, :, 2], dim=1)
    lose_balance = torch.logical_and(~upper_contact, body_mean_feet_xy_dist>0.4)
    lose_balance = torch.logical_and(lose_balance, mean_height>0.4)
    # print('body_mean_feet_xy_dist:', body_mean_feet_xy_dist)
    # print('mean_height:', mean_height)
    # print('upper_contact:', upper_contact)
    return lose_balance


def joint_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_low_limits = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.joint_limits[:, asset_cfg.joint_ids, 0]) < 0.01
  
    out_of_high_limits = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.joint_limits[:, asset_cfg.joint_ids, 1]) < 0.01
    out_of_limits = torch.logical_or(out_of_low_limits, out_of_high_limits)
    return out_of_limits.any(dim=1)


def foot_too_back(
    env: ManagerBasedRLEnv,  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    #return (asset.data.root_link_pos_w[:, 2]-env.scene.env_origins[:,2]) < minimum_height
    #print('min x:',(asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).min(dim=1)[0])
    too_back = ((asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).min(dim=1)[0]) < -1.
    ending = env.episode_length_buf>225
    return torch.logical_and(too_back, ending)

def consecutive_backward(
    env: ManagerBasedRLEnv, threshold: int = 10
) -> torch.Tensor:
    """Terminate when the asset has been moving backward for a certain number of consecutive steps.
    """
    command_term = env.command_manager.get_term('climb_command')
    backward_penalty_cfg = env.reward_manager.get_term_cfg('backward_penalty')
    value = backward_penalty_cfg.func(env, **backward_penalty_cfg.params)
    #print('backward penalty metrics:', value)
    
    command_term.update_consecutive_fail(value)
    # input('Press Enter to continue...')  # Debugging line to pause execution
    # print('consecutive backward:', command_term.consecutive_backward)
    return command_term.consecutive_backward >= threshold

def root_height_above_maximum(
    env: ManagerBasedRLEnv, maximum_height=0.7, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is above the maximum height.

    Note:
        This is currently only supported for flat terrains, i.e. the maximum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] > maximum_height

def stepped_on(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, platform_width: float, reached_distance:float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root link is in contact with the ground.

    Note:
        This is currently only supported for flat terrains.
    """
    # extract the used quantities (to enable type-hinting) 
    asset: RigidObject = env.scene[asset_cfg.name]
    # high=torch.logical_and(asset.data.body_pos_w[:, feet_ids[0], 2]>0.02, asset.data.body_pos_w[:, feet_ids[1], 2]>0.02)
    # high = torch.logical_and(high, asset.data.root_link_pos_w[:, 2] > 0.2)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact=torch.all(torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1)
    target_pos_e = env.command_manager.get_command("target_pos_e")
    target_pos_w=target_pos_e+env.scene.env_origins
    target_pos_w[:,2]=0
    
    remaining_distance = torch.norm(target_pos_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    near = remaining_distance < reached_distance
    
    #far = torch.abs(asset.data.root_link_pos_w[:,0]-env.scene.env_origins[:,0])>(platform_width/2+0.05)
    stepped=torch.logical_and(near, contact)
    # print('near', near)
    # print('contact', contact)
    return stepped

def on_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    on_air=~torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    on_air = torch.logical_and(on_air, env.episode_length_buf>20)
    # print('history length',contact_sensor.cfg.history_length)
    # print('body names', contact_sensor.body_names)
    # print('contact', torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1))    
    # print('on_air', on_air)
    return on_air

def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int) -> torch.Tensor:
    """Check if the task has been completed consecutively for a certain number of times.

    Args:
        env: The environment object.
        num_success: Threshold for the number of consecutive successes required.
        command_name: The command term to be used for extracting the goal.
    """
    
    command_term = env.command_manager.get_term('target_pos_e')
    consecutive_near=command_term.metrics["consecutive_success"] >= num_success

    return consecutive_near

def max_contact_force(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    #threshold = 0
    max_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    #print('max contact', max_contact)
    max_contact = torch.max(max_contact,dim=1)[0]
    return max_contact > threshold

def max_contact_force_value(env: ManagerBasedRLEnv,  sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    #threshold = 0
    max_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    #print('max contact', max_contact)
    max_contact = torch.max(max_contact,dim=1)[0]
    #print('max contact force:', max_contact)
    return max_contact 




"""
Joint terminations.
"""


def joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos > asset.data.soft_joint_pos_limits[..., 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos < asset.data.soft_joint_pos_limits[..., 0], dim=1)
    return torch.logical_or(out_of_upper_limits[:, asset_cfg.joint_ids], out_of_lower_limits[:, asset_cfg.joint_ids])


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_vel_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    limits = asset.data.soft_joint_vel_limits
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > limits[:, asset_cfg.joint_ids], dim=1)


def joint_vel_out_of_manual_limit(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity, dim=1)


def joint_effort_out_of_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = torch.isclose(
        asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
    )
    return torch.any(out_of_limits, dim=1)


"""
Contact sensor.
"""


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )

def standing_illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    return torch.logical_and(env.command_manager.get_command('climb_command') ==0, contact)