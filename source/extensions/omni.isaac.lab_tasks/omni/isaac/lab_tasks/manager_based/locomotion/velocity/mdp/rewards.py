# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion task functions."""
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from omni.isaac.lab.sensors import ContactSensor, ContactSensorZ, ContactGroundSensorZ
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.sensors import ContactSensor, RayCaster
import omni.isaac.lab.utils.math as math_utils
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def is_alive_up(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    climb_command = env.command_manager.get_command('climb_command')
    return (torch.logical_and(~env.termination_manager.terminated,climb_command>0)).float()


def is_terminated_climbup(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    
    # if env.termination_manager.time_outs:
    #     print('Episodic timeout, no penalty applied.')
    # if env.termination_manager.terminated:
    #     print('Episode terminated**************************************, penalty applied.')
    reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    # print('reset env ids:', reset_env_ids)
    return env.termination_manager.terminated.float()


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_com_ang_vel_b[:, :2]), dim=1)


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    # print('body accs:', torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1))
    # print('body vels:',torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1))
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # for i in range(len(asset.data.applied_torque[0])):
    #input("Input Enter")
    # print("joint names:", asset.data.joint_names, asset_cfg.joint_names," joint ids:", asset_cfg.joint_ids)
    # print("applied torque:", asset.data.applied_torque[:, asset_cfg.joint_ids[:23]])
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids[:23]]), dim=1)


def joint_vel_l2_climbup(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # input("Input Enter")
    # print('joint velocities:', torch.max(asset.data.joint_vel[:, asset_cfg.joint_ids],dim=-1)[0])
    max_id=torch.max(asset.data.joint_vel[:, asset_cfg.joint_ids],dim=-1)[1]
    # print('max id:', max_id)
    # print('max joint id:', asset.data.joint_names[max_id])
    if (asset.data.joint_vel.shape[-1]) == 29:
        return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids[0:23]]), dim=1)
    else:
        return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_exp(env: ManagerBasedRLEnv, grad_scale: float, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    max_vel = torch.max(asset.data.joint_vel[:, asset_cfg.joint_ids],dim=-1)[0]
    max_vel = (max_vel-threshold).clip(min=0.)
    rew=torch.exp(grad_scale*max_vel)-1
    return rew.clip(max=200)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids[:23]]), dim=1)


def climb_lying_joint_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    climb_command = env.command_manager.get_command('climb_command')
    asset: Articulation = env.scene[asset_cfg.name]
    box_height = env.scene.env_origins[:,2]
    default_joint = env.command_manager.get_term('climb_command').lying_joint
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(angle), dim=1)
    reward = torch.exp(-0.1 * reward)  # Exponential kernel to penalize deviations from the default joint position
    # success = torch.logical_and(asset.data.body_pos_w[:, :, 2].min(dim=1)[0] > (box_height+0.02), #both_leg_contact)
    #                             #(asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:,0].unsqueeze(1)).min(dim=1)[0] > -1)
    #                              asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0] > -0.9)
    
    # success = torch.logical_and(asset.data.body_pos_w[:, :, 2].min(dim=1)[0] > (box_height-0.25), 
    #                             asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0] > -0.9)
    success = torch.logical_and(asset.data.body_pos_w[:, :, 2].min(dim=1)[0] > (box_height), 
                                asset.data.body_pos_w[:, :, 0].min(dim=1)[0] > -1, )
    success = torch.logical_and(success,  env.episode_length_buf>200)
    return torch.where(success, reward, torch.zeros_like(reward),)


def lying_contact(env: ManagerBasedRLEnv, width, feet_cfg, sensor_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    net_contact_forces = contact_sensor._data.force_matrix_w[:, sensor_cfg.body_ids,0, 2]
    is_contact = net_contact_forces > 1.0
    # input("Input Enter")
    # print('is contact:', is_contact)
    # print('sensor body ids:', sensor_cfg.body_ids)
    # print('asset body ids:', asset_cfg.body_ids)

    # print('sensor body names:', [contact_sensor.body_names[i] for i in sensor_cfg.body_ids])

    # print('asset body names:', [asset.body_names[i] for i in asset_cfg.body_ids])
    # feet_on_box = ((asset.data.body_pos_w[:, feet_cfg.body_ids, 1] - env.scene.env_origins[:,1].unsqueeze(1)).abs() < width/2).all(dim=-1)
    on_box = (asset.data.body_pos_w[:, asset_cfg.body_ids, 2]> env.scene.env_origins[:,2].unsqueeze(1)) & (asset.data.body_pos_w[:, asset_cfg.body_ids, 0]-env.scene.env_origins[:,0].unsqueeze(1) > -0.95 )
    # print('feet on box:', feet_on_box)
    on_box_success = torch.logical_and(asset.data.body_pos_w[:, :, 2].min(dim=1)[0] > env.scene.env_origins[:,2], 
                                asset.data.body_pos_w[:, :, 0].min(dim=1)[0] > -1, )
    success = (env.episode_length_buf>200).unsqueeze(-1) & is_contact & on_box & on_box_success.unsqueeze(-1)
    success = success.sum(dim=1) 
    return success.float()


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    # joint_names = asset.joint_names
    # #print('joint names:', asset.data.joint_names)
    # violation_mask = out_of_limits > 0  # shape: [num_envs, num_joints]

    # for env_id in range(violation_mask.shape[0]):
    #     violated_names = [joint_names[j] for j in range(len(joint_names)) if violation_mask[env_id, j]]
    #     if violated_names:
    #         print(f"[Env {env_id}] Joint limits violated: {violated_names}")
    # penalty = torch.sum(out_of_limits, dim=1)
    # climb_command = env.command_manager.get_command('climb_command')
    # return torch.where(climb_command > 0, penalty, 10*penalty)  # penalize more when climbing
    return torch.sum(out_of_limits, dim=1)


def processed_action_rate_l2(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    #print('processed actions: ', env.action_manager.get_term(action_name).processed_actions)
    return torch.sum(torch.square( env.action_manager.get_term(action_name).processed_actions[:, :23]
                                   - env.action_manager.get_term(action_name).last_processed_actions[:, :23]), dim=1)


def power_consumption(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the power consumption using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    #print('power consumption: ', asset.data.applied_torque * asset.data.joint_vel)
    return torch.sum(torch.abs(asset.data.applied_torque[:,:23] * asset.data.joint_vel[:, :23]), dim=1)


def contact_forces_exp(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg,grad_scale:float) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation

            
    max_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    max_contact_force = torch.max(max_contact,dim=1)[0]
    # input("Input Enter")
    # print("max contact force: ", max_contact_force)

    # max_contact_id = torch.max(max_contact,dim=1)[1]
    # body_names = contact_sensor.body_names
    # if max_contact_force >500:
    #     print("MAX CONTACT force: ", max_contact_force)
    #     print('max contact body:', body_names[max_contact_id])
    #print('body names:', body_names)
    rew=torch.exp(grad_scale*(max_contact_force-threshold).clip(min=0.0))-1
    # print("VIOLATION: ", rew)
    return rew.clip(max=200)


class body_slipping_down_climbup(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        contact_sensor: ContactSensor = env.scene.sensors[cfg.params['sensor_cfg'].name]
        asset: RigidObject = env.scene[cfg.params['asset_cfg'].name]
        # compute the contact forces
        # check if contact force is above threshold
        self.id_asset = asset.find_bodies(contact_sensor.body_names, preserve_order=True)[0]

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,asset_cfg) -> torch.Tensor:

        # Return the unweighted reward for the termination terms
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: RigidObject = env.scene[asset_cfg.name]
        # compute the contact forces
        net_contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
        # check if contact force is above threshold
        is_contact = net_contact_forces > 1.
        # compute the dragging condition
        
        vel_norm = torch.norm(asset.data.body_lin_vel_w[:,self.id_asset,:2], dim=-1) 
        penalty = torch.where(is_contact, vel_norm, torch.zeros_like(vel_norm))

        return torch.sum(penalty, dim=1)


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def is_terminated_climbdown(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    
    # if env.termination_manager.time_outs:
    #     print('Episodic timeout, no penalty applied.')
    # if env.termination_manager.terminated:
    #     print('Episode terminated**************************************, penalty applied.')
    # reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    # print('reset env ids:', reset_env_ids)
    return env.termination_manager.terminated.float()


def standing_flat_orientation_down(env: ManagerBasedRLEnv,sensor_cfg,feet_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    climb_command = env.command_manager.get_command('climb_command')
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]

    asset: RigidObject = env.scene[asset_cfg.name]
    feet: RigidObject = env.scene[feet_cfg.name]

    # quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)  # (num_envs, num_bodies, 4) -> (num_envs, 4)
    # proj_grav = math_utils.quat_rotate_inverse(quat, asset.data.GRAVITY_VEC_W)

    # print('root link quat w in projected gravity:', asset.data.root_link_quat_w)
    # print('gravity vec w in projected gravity:', asset.data.GRAVITY_VEC_W)
    # reward = torch.square(r) + torch.square(p)
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward = torch.exp(-5 * reward)  # Exponential kernel to penalize deviations from the desired height``

    # reward = torch.sum(torch.square(proj_grav[:, :2]), dim=1)
    # reward = torch.exp(-3 * reward)  # Exponential kernel to penalize deviations from the desired height``
    # print('proj_grav:', proj_grav)
    # print('reward:', reward)
    root_x= asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]

    feet_near_grd = feet.data.body_pos_w[:,feet_cfg.body_ids, 2].max(dim=-1)[0] < 0.1
    # net_contact_forces = contact_sensor.data.net_forces_w
    # feet_contact = (torch.norm(net_contact_forces[:,  sensor_cfg.body_ids], dim=-1)>0.1).any(dim=-1)
    # root_height = asset.data.root_pos_w[:, 2]<=0.8
    root_off_box = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0] < -1
    success = torch.logical_and(feet_near_grd, root_off_box)
    # success = torch.logical_and(success, root_height)
    # success = torch.logical_and(success, )
    # input("Input Enter")
    # print("success:", success)
    # print('projected gravity:', asset.data.projected_gravity_b[:, :])
    success = torch.logical_and(success, env.episode_length_buf>200)
    stage_reward = torch.where(success, reward, torch.zeros_like(reward))
    # print("stage_reward:", stage_reward)

    # return torch.where(torch.logical_and(climb_command == 0,root_x<-0.9), reward,torch.zeros_like(reward), )
    return stage_reward 


def joint_vel_l2_climbdown(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # input("Input Enter")
    # if torch.max(asset.data.joint_vel[:, asset_cfg.joint_ids],dim=-1)[0]>10.:
    #     print('joint velocities:', torch.max(asset.data.joint_vel[:, asset_cfg.joint_ids],dim=-1)[0])
    #     max_id=torch.max(asset.data.joint_vel[:, asset_cfg.joint_ids],dim=-1)[1]
    # # print('max id:', max_id)
    #     print('max joint id:', asset.data.joint_names[max_id])
    if (asset.data.joint_vel.shape[-1]) == 29:
        return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids[0:23]]), dim=1)
    else:
        return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def standing_joint_deviation_down(env: ManagerBasedRLEnv, sensor_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint = env.command_manager.get_term('climb_command').standing_joint
    #print('joint names:', asset.data.joint_names)
    # compute out of limits constraints
    #input("Input Enter")
    # angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # input("Input Enter")
    # print('default joint:', default_joint[:, asset_cfg.joint_ids])
    # print('current joint:', asset.data.joint_pos[:, asset_cfg.joint_ids])
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(angle), dim=1)
    reward = torch.exp(-0.1 * reward)  # Exponential kernel to penalize deviations from the default joint position
    root_x= asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]

    feet_near_grd = asset.data.body_pos_w[:,asset_cfg.body_ids, 2].max(dim=-1)[0] < 0.1
    # net_contact_forces = contact_sensor.data.net_forces_w
    # feet_contact = (torch.norm(net_contact_forces[:,  sensor_cfg.body_ids], dim=-1)>0.1).any(dim=-1)
    # root_height = asset.data.root_pos_w[:, 2]<=0.8
    root_off_box = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0] < -1
    success = torch.logical_and(feet_near_grd, root_off_box)
    # success = torch.logical_and(success, root_height)
    # success = torch.logical_and(success, )
    success = torch.logical_and(success, env.episode_length_buf>200)
    stage_reward = torch.where(success, reward, torch.zeros_like(reward))

    # return torch.where(torch.logical_and(climb_command == 0,root_x<-0.9),reward, torch.zeros_like(reward), )
    return  stage_reward


class body_slipping_contact(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        contact_sensor: ContactSensor = env.scene.sensors[cfg.params['sensor_cfg'].name]
        asset: RigidObject = env.scene[cfg.params['asset_cfg'].name]
       
        self.id_asset = asset.find_bodies(contact_sensor.body_names, preserve_order=True)[0]

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,asset_cfg) -> torch.Tensor:

        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: RigidObject = env.scene[asset_cfg.name]
        # compute the contact forces
        net_contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
        
        # check if contact force is above threshold
        is_contact = net_contact_forces > 1.

        # compute the dragging condition
        vel_norm = torch.norm(asset.data.body_lin_vel_w[:,self.id_asset,:3], dim=-1) 

        penalty = torch.where(is_contact, vel_norm*net_contact_forces/100, torch.zeros_like(vel_norm))

        return torch.sum(penalty, dim=1)


def lydown_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    box_height = env.scene.env_origins[:,2]
    cosine = asset.data.projected_gravity_b[:, 0] 
    reward = cosine.clip(min=0, max = 0.9)
    reward = torch.where( asset.data.projected_gravity_b[:, 2]<0,reward, 0.9*torch.ones_like(reward))
    current_height = torch.mean(asset.data.body_pos_w[:, :, 2], dim=1)
    success = current_height < 0.3
    success = torch.logical_and(success,  env.episode_length_buf>200)
    return torch.where(success, reward,torch.zeros_like(reward))


def lydown_joint_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint = env.command_manager.get_term('climb_command').lying_joint
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(angle), dim=1)
    reward = torch.exp(-0.1 * reward)  # Exponential kernel to penalize deviations from the default joint position
    # success = torch.logical_and(asset.data.body_pos_w[:, :, 2].min(dim=1)[0] > (box_height+0.02), #both_leg_contact)
    #                             #(asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:,0].unsqueeze(1)).min(dim=1)[0] > -1)
    #                              asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0] > -0.9)
    
    # current_height = torch.mean(asset.data.body_pos_w[:, :, 2], dim=1)
    shoulder_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].mean(dim=-1)  
    success = shoulder_height < 0.45
    success = torch.logical_and(success,  env.episode_length_buf>200)
    # success = env.episode_length_buf>200
    return torch.where(success, reward, torch.zeros_like(reward),)


class lying_contact_down(ManagerTermBase):
    """Penalize joint positions that deviate from the default one."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # extract the used quantities
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.left_upper_cfg: SceneEntityCfg = cfg.params["left_upper_cfg"]
        self.right_upper_cfg: SceneEntityCfg = cfg.params["right_upper_cfg"]
        self.left_lower_cfg: SceneEntityCfg = cfg.params["left_lower_cfg"]
        self.right_lower_cfg: SceneEntityCfg = cfg.params["right_lower_cfg"]

        # resolve the body indices on the asset
        asset: Articulation = env.scene[self.asset_cfg.name]
        self.left_upper_body_ids = asset.find_bodies(self.left_upper_cfg.body_names, preserve_order=True)[0]
        self.right_upper_body_ids = asset.find_bodies(self.right_upper_cfg.body_names, preserve_order=True)[0]
        self.left_lower_body_ids = asset.find_bodies(self.left_lower_cfg.body_names, preserve_order=True)[0]
        self.right_lower_body_ids = asset.find_bodies(self.right_lower_cfg.body_names, preserve_order=True)[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        left_upper_cfg: SceneEntityCfg,
        right_upper_cfg: SceneEntityCfg,
        left_lower_cfg: SceneEntityCfg,
        right_lower_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactGroundSensorZ = env.scene.sensors[left_upper_cfg.name]

        # Extract z-force once: shape (num_envs, num_bodies)
        force_z = contact_sensor._data.force_matrix_w[:, :, 0, 2]

        def contact_reward(cfg, body_ids_asset):
            # Forces on the specified bodies
            forces = force_z[:, cfg.body_ids]  # (num_envs, num_cfg_bodies)
            contact = (forces > 1.0).any(dim=-1).float()  # (num_envs,)

            # Distance of the corresponding body to the ground (z)
            dist = asset.data.body_pos_w[:, body_ids_asset, 2].min(dim=1)[0]  # (num_envs,)

            # Same formula as before: max(contact, exp(-15 * dist^2))
            return torch.maximum(contact, torch.exp(-15.0 * dist ** 2))

        # Compute rewards for all four body groups
        L_upper_reward = contact_reward(left_upper_cfg, self.left_upper_body_ids)
        R_upper_reward = contact_reward(right_upper_cfg, self.right_upper_body_ids)
        L_lower_reward = contact_reward(left_lower_cfg, self.left_lower_body_ids)
        R_lower_reward = contact_reward(right_lower_cfg, self.right_lower_body_ids)

        # Project gravity to body frame
        quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
        proj_grav = math_utils.quat_rotate_inverse(quat, asset.data.GRAVITY_VEC_W)

        # Same contact count expression
        contact_cnt = (L_upper_reward + R_upper_reward) * (proj_grav[:, 0] > 0.75) + L_lower_reward + R_lower_reward

        success = (env.episode_length_buf > 200).float()
        return contact_cnt * success


def self_collision_torso(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor._data.force_matrix_w[:, 0,:, :].norm(dim=-1)
    is_contact = net_contact_forces > 1.0
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def standing_flat_orientation_up(env: ManagerBasedRLEnv,shoulder_cfg, sensor_cfg,feet_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    climb_command = env.command_manager.get_command('climb_command')

    asset: RigidObject = env.scene[asset_cfg.name]

    # reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward = torch.exp(-5 * reward)  # Exponential kernel to penalize deviations from the desired height``
    cosine = -asset.data.projected_gravity_b[:, 2] 
    reward = cosine.clip(min=0)
    ending = env.episode_length_buf>250

    # contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]
    # net_contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    # is_contact = net_contact_forces > 1.
    # on_box_success = is_contact.all(dim=-1)
    # on_box_success =on_box_success.logical_and((asset.data.body_pos_w[:,feet_cfg.body_ids, 0] - env.scene.env_origins[:,0].unsqueeze(1)).min(dim=1)[0] > -0.95)
    # quat = asset.data.body_quat_w[:, feet_cfg.body_ids, :].squeeze(1)  # (num_envs, num_bodies, 4) -> (num_envs, 4)
    # proj_grav = math_utils.quat_rotate_inverse(quat, asset.data.GRAVITY_VEC_W.unsqueeze(1))
    # cosine = (-proj_grav[:, :,2]).clip(max=1, min=-1)#cos(30 degree)
    # angle = torch.acos(cosine)
    # on_box_success = on_box_success.logical_and(angle.max(dim=1)[0]<0.05)

    # masses= env.command_manager.get_term('climb_command').mass
    # body_coms = asset.data.body_com_pos_w[:, :, :2] - env.scene.env_origins[:, :2].unsqueeze(1) #shape: (num_envs, num_bodies)
    # CoM = torch.sum(body_coms * masses.unsqueeze(-1), dim=1) / torch.sum(masses.unsqueeze(-1), dim=1)

    # feet_pos = (asset.data.body_pos_w[:,feet_cfg.body_ids, :2] - env.scene.env_origins[:,:2].unsqueeze(1))
    # feet_CoM_dist = torch.linalg.norm((CoM.unsqueeze(1) - feet_pos),dim=2)
    # feet_com_close = feet_CoM_dist.max(dim=1)[0] < 0.15
    # on_box_success = on_box_success.logical_and(feet_com_close)
    # ending = torch.logical_and(ending,on_box_success)
    shoulder_height = asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].mean(dim=-1) - env.scene.env_origins[:, 2]
    # print('shoulder height:', shoulder_height)
    ending = ending & (shoulder_height > 0.9)
    return reward.masked_fill_(~ending, 0.0)


def standing_joint_deviation_up(env: ManagerBasedRLEnv, shoulder_cfg,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint = env.command_manager.get_term('climb_command').standing_joint

    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint[:, asset_cfg.joint_ids]
    # print('joint names:', asset.data.joint_names)
    reward = torch.sum(torch.square(angle), dim=1)
    reward = torch.exp(-0.1 * reward)  # Exponential kernel to penalize deviations from the default joint position

    ending = env.episode_length_buf>250
    shoulder_height = asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].mean(dim=-1) - env.scene.env_origins[:, 2]
    ending = ending & (shoulder_height > 0.9)    
    return reward.masked_fill_(~ending, 0.0)


class body_slipping_down_standup(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        contact_sensor: ContactSensor = env.scene.sensors[cfg.params['sensor_cfg'].name]
        asset: RigidObject = env.scene[cfg.params['asset_cfg'].name]
        # compute the contact forces
        # check if contact force is above threshold
        self.id_asset = asset.find_bodies(contact_sensor.body_names, preserve_order=True)[0]

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,asset_cfg) -> torch.Tensor:

        # Return the unweighted reward for the termination terms
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: RigidObject = env.scene[asset_cfg.name]
        # compute the contact forces
        net_contact_forces = torch.sum(torch.square(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]), dim=-1)
        # check if contact force is above threshold
        is_contact = net_contact_forces > 1.
        # compute the dragging condition
        
        vel_norm = torch.norm(asset.data.body_lin_vel_w[:,self.id_asset,:2], dim=-1) 
        return (vel_norm * is_contact.float()).sum(dim=1)


def joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # for i in range(len(asset.data.applied_torque[0])):
    #input("Input Enter")
    # print("joint names:", asset.data.joint_names[23:])
    # print("applied torque:", asset.data.applied_torque[:, asset_cfg.joint_ids[23:]])
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def lying_joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint = env.command_manager.get_term('climb_command').lying_joint
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(angle), dim=1)
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    #print("VIOLATION: ", violation.clip(min=0.0))
    # if (violation>0).any():
    #     max_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    #     max_contact_force = torch.max(max_contact,dim=1)[0]
    #     max_contact_id = torch.max(max_contact,dim=1)[1]
    #     body_names = [contact_sensor.body_names[i] for i in sensor_cfg.body_ids]
    #     print("max contact force: ", max_contact_force,'max contact body:', [body_names[i] for i in max_contact_id.tolist()])
    return torch.sum(violation.clip(min=0.0), dim=1)

def crawl_joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint = env.command_manager.get_term('climb_command').lying_joint
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(angle), dim=1)

    term = env.command_manager.get_term('base_velocity')
    vel_cmd = term.vel_command_b[:, :3]
    is_low_vel = torch.all(torch.abs(vel_cmd) < 0.1, dim=1)
    is_standing_env = term.is_standing_env

    is_zero_phase = is_low_vel | is_standing_env
    return torch.where(is_zero_phase, 10*reward, reward)
