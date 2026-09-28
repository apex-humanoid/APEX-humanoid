# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

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

"""
General.
"""


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()

def is_alive_up(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    climb_command = env.command_manager.get_command('climb_command')
    return (torch.logical_and(~env.termination_manager.terminated,climb_command>0)).float()


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    
    # if env.termination_manager.time_outs:
    #     print('Episodic timeout, no penalty applied.')
    # if env.termination_manager.terminated:
    #     print('Episode terminated**************************************, penalty applied.')
    # reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    # print('reset env ids:', reset_env_ids)
    return env.termination_manager.terminated.float()




class is_terminated_term(ManagerTermBase):
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
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


"""
Root penalties.
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_com_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_com_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)

def standing_flat_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    climb_command = env.command_manager.get_command('climb_command')

    asset: RigidObject = env.scene[asset_cfg.name]
    # quat = asset.data.body_quat_w[:, asset_cfg.body_ids].squeeze(1)  
    # r,p,y=math_utils.euler_xyz_from_quat(quat)
    # r = torch.where(r>torch.pi, r-torch.pi*2, r)
    # p = torch.where(p>torch.pi, p-torch.pi*2, p)
    # y = torch.where(y>torch.pi, y-torch.pi*2, y)
    #input("Input Enter")
    # print('r:', r, 'p:', p, 'y:', y)
    #print('foot height:', asset.data.body_link_pos_w[:, asset_cfg.body_ids]-env.scene.env_origins)
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward = torch.square(r) + torch.square(p)
    reward = torch.exp(-5 * reward)  # Exponential kernel to penalize deviations from the desired height``
    return torch.where(climb_command == 0, reward,torch.zeros_like(reward), )

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
    ending = torch.logical_and(ending, shoulder_height > 0.9)

    return torch.where(ending, reward,torch.zeros_like(reward))

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

def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + sensor.data.pos_w[:, 2]
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height)


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    # print('body accs:', torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1))
    # print('body vels:',torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1))
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


"""
Joint penalties.
"""


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

def joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # for i in range(len(asset.data.applied_torque[0])):
    #input("Input Enter")
    # print("joint names:", asset.data.joint_names, asset_cfg.joint_names," joint ids:", asset_cfg.joint_ids)
    # print("applied torque:", asset.data.applied_torque[:, asset_cfg.joint_ids[:23]])
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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
    
def joint_vel_with_action(env: ManagerBasedRLEnv, action_term_name,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_with_action = env.action_manager.get_term(action_term_name)._joint_ids
    return torch.sum(torch.square(asset.data.joint_vel[:, joint_with_action]), dim=1)

def joint_vel_exp(env: ManagerBasedRLEnv, grad_scale: float, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    max_vel = torch.max(asset.data.joint_vel[:, asset_cfg.joint_ids],dim=-1)[0]
    max_vel = (max_vel-threshold).clip(min=0.)
    rew=torch.exp(grad_scale*max_vel)-1
    return rew.clip(max=200)

def joint_vel_clip(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    max_vel = torch.max(asset.data.joint_vel[:, asset_cfg.joint_ids],dim=-1)[0]
    return (max_vel-threshold).clip(min=0.)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids[:23]]), dim=1)

def joint_acc_with_action(env: ManagerBasedRLEnv, action_term_name,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_with_action = env.action_manager.get_term(action_term_name)._joint_ids
    return torch.sum(torch.square(asset.data.joint_acc[:, joint_with_action]), dim=1)

def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)

def standing_joint_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    climb_command = env.command_manager.get_command('climb_command')
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint = env.command_manager.get_term('climb_command').standing_joint
    #input("Input Enter")
    #print('joint names:', asset.data.joint_names)
    # compute out of limits constraints
    #input("Input Enter")
    # angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(angle), dim=1)
    reward = torch.exp(-0.1 * reward)  # Exponential kernel to penalize deviations from the default joint position
    return torch.where(climb_command > 0, torch.zeros_like(reward), reward)

def standing_joint_deviation_up(env: ManagerBasedRLEnv, shoulder_cfg,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint = env.command_manager.get_term('climb_command').standing_joint

    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(angle), dim=1)
    reward = torch.exp(-0.1 * reward)  # Exponential kernel to penalize deviations from the default joint position

    ending = env.episode_length_buf>250
    shoulder_height = asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].mean(dim=-1) - env.scene.env_origins[:, 2]
    ending = torch.logical_and(ending, shoulder_height > 0.9)
    return torch.where(ending, reward,torch.zeros_like(reward))

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

def standing_joint_deviation_down_exp(env: ManagerBasedRLEnv, sensor_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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


    # return torch.where(torch.logical_and(climb_command == 0,root_x<-0.9),reward, torch.zeros_like(reward), )
    return  stage_reward


def lying_joint_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    climb_command = env.command_manager.get_command('climb_command')
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint = env.command_manager.get_term('climb_command').lying_joint
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(angle), dim=1)
    reward = torch.exp(-0.1 * reward)  # Exponential kernel to penalize deviations from the default joint position
    return torch.where(climb_command==2, reward, torch.zeros_like(reward),)

def lying_joint_move(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    climb_command = env.command_manager.get_command('climb_command')
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint = env.command_manager.get_term('climb_command').lying_joint
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(angle), dim=1)
    reward = torch.exp(-0.1 * reward)  # Exponential kernel to penalize deviations from the default joint position
    return reward

def lying_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    box_height = env.scene.env_origins[:,2]
    cosine = asset.data.projected_gravity_b[:, 0] 
    # print('lying orientation cosine:', cosine)
    reward = cosine.clip(min=0, max = 0.7)
    success = torch.logical_and(asset.data.body_pos_w[:, :, 2].min(dim=1)[0] > (box_height-0.1), 
                                asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0] > -0.9)
    success = torch.logical_and(success,  env.episode_length_buf>200)
    return torch.where(success, reward,torch.zeros_like(reward))

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

def lying_contact_down(env: ManagerBasedRLEnv, asset_cfg, 
                       left_upper_cfg, right_upper_cfg, 
                       left_lower_cfg, right_lower_cfg) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[left_upper_cfg.name]

    # Extract z-force once: shape (num_envs, num_bodies)
    force_z = contact_sensor._data.force_matrix_w[:, :, 0, 2]

    def contact_reward(cfg):
        # Forces on the specified bodies
        forces = force_z[:, cfg.body_ids]                        # (num_envs, num_cfg_bodies)
        contact = (forces > 1.0).any(dim=-1).float()             # (num_envs,)

        # Distance of the corresponding body to the ground (z)
        body_id = asset.find_bodies(cfg.body_names, preserve_order=True)[0]
        dist = asset.data.body_pos_w[:, body_id, 2].squeeze(-1)  # (num_envs,)

        # Same formula as before: max(contact, exp(-15 * dist^2))
        return torch.maximum(contact, torch.exp(-15.0 * dist ** 2))

    # Compute rewards for all four body groups
    L_upper_reward = contact_reward(left_upper_cfg)
    R_upper_reward = contact_reward(right_upper_cfg)
    L_lower_reward = contact_reward(left_lower_cfg)
    R_lower_reward = contact_reward(right_lower_cfg)
    # print('L_upper_reward:', L_upper_reward)
    # print('R_upper_reward:', R_upper_reward)
    # print('L_lower_reward:', L_lower_reward)    
    # print('R_lower_reward:', R_lower_reward)

    # Project gravity to body frame
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    proj_grav = math_utils.quat_rotate_inverse(quat, asset.data.GRAVITY_VEC_W)

    # Same contact count expression

    contact_cnt = (L_upper_reward + R_upper_reward) * (proj_grav[:, 0] > 0.75) + L_lower_reward + R_lower_reward
    
    success = (env.episode_length_buf > 200).float()
    return contact_cnt * success


def standing_contact(env: ManagerBasedRLEnv,  upper_sensor_cfg, sensor_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    feet_contact_forces = contact_sensor._data.force_matrix_w[:, sensor_cfg.body_ids,0, 2].sum(dim=-1)
    is_contact = (feet_contact_forces > 300) & (feet_contact_forces < 380)

    foot_ids = torch.as_tensor(sensor_cfg.body_ids,  dtype=torch.long)

    num_bodies = contact_sensor._data.force_matrix_w.shape[1]
    # Mask: True = keep, False = exclude
    mask = torch.ones(num_bodies, dtype=torch.bool)
    mask[foot_ids] = False  # turn off the feet

    # Index with mask directly (no need for all_ids / isin)
    other_contact_forces = (contact_sensor.data.net_forces_w[:, mask, :].norm(dim=-1)).sum(dim=-1)
    # input("Input Enter")
    # print('is contact:', is_contact)
    # print('sensor body ids:', sensor_cfg.body_ids)
    # print('asset body ids:', asset_cfg.body_ids)

    # print('sensor body names:', [contact_sensor.body_names[i] for i in sensor_cfg.body_ids])

    # print('asset body names:', [asset.body_names[i] for i in asset_cfg.body_ids])
    masses= env.command_manager.get_term('climb_command').mass
    on_ground = (asset.data.body_pos_w[:, asset_cfg.body_ids, 2]< 0.15)  
    # upper_no_contact = (contact_sensor.data.net_forces_w[:, upper_sensor_cfg.body_ids].norm(dim=-1) < 1.0).all(dim=-1)
    upper_no_contact = other_contact_forces < 40.0
    upper_no_contact = (upper_no_contact & on_ground.all(dim=-1)).float() *2

    feet_on_ground = (is_contact & on_ground.all(dim=-1)).float() * 2

    return (feet_on_ground + upper_no_contact) * (env.episode_length_buf>200)
    # success = (env.episode_length_buf>200).unsqueeze(-1) & is_contact & on_ground #& feet_on_box.unsqueeze(-1)
    # success = success.sum(dim=1) 
    # return success.float()


def joint_deviation_exp(env: ManagerBasedRLEnv, scale: float, threshold: float,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = (torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids])-threshold).clip(min=0.0)
    #print("joint deviation: ", torch.exp(angle)-1)
    rew = torch.exp(scale*angle)-1
    return torch.sum(rew, dim=1)

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

def wrist_joint_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - 0.7*asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - 0.7*asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)

    return torch.sum(out_of_limits, dim=1)

def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


"""
Action penalties.
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    out_of_limits = out_of_limits.clip_(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def processed_action_rate_l2(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    #print('processed actions: ', env.action_manager.get_term(action_name).processed_actions)
    return torch.sum(torch.square( env.action_manager.get_term(action_name).processed_actions[:, :23]
                                   - env.action_manager.get_term(action_name).last_processed_actions[:, :23]), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)

def power_consumption(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the power consumption using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    #print('power consumption: ', asset.data.applied_torque * asset.data.joint_vel)
    return torch.sum(torch.abs(asset.data.applied_torque[:,:23] * asset.data.joint_vel[:, :23]), dim=1)


"""
Contact sensor.
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def contact_forces_exp(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg,grad_scale:float) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation

            
    max_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    max_contact_force = torch.max(max_contact,dim=1)[0]
    # input("Input Enter")
    # if max_contact_force > 400:
    #     max_contact_id = torch.max(max_contact,dim=1)[1]
    #     body_names = [contact_sensor.body_names[i] for i in sensor_cfg.body_ids]

    #     print("max contact force: ", max_contact_force, "max contact body:", body_names[max_contact_id])

    # if max_contact_force >500:
    # print("MAX CONTACT force: ", max_contact_force)
    # print('max contact body:', body_names[max_contact_id])
    #print('body names:', body_names)
    rew=torch.exp(grad_scale*(max_contact_force-threshold).clip(min=0.0))-1
    # print("VIOLATION: ", rew)
    return rew.clip(max=200)

def contact_forces_exp_1s(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg,grad_scale:float) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation

            
    max_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    max_contact_force = torch.max(max_contact,dim=1)[0]
  
    rew=torch.exp(grad_scale*(max_contact_force-threshold).clip(min=0.0))-1
    return (rew.clip(max=200))* (env.episode_length_buf < 50)

def self_collision_torso(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor._data.force_matrix_w[:, 0,:, :].norm(dim=-1)
    is_contact = net_contact_forces > 1.0
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)

def upper_no_contact(env: ManagerBasedRLEnv, weight_scale,feet_cfg, sensor_cfg, upper_sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w

    # Condition check of entering the up stage
    net_contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    is_contact = net_contact_forces > 1.
    on_box_success = is_contact.all(dim=-1)
    on_box_success =on_box_success.logical_and((asset.data.body_pos_w[:,feet_cfg.body_ids, 0] - env.scene.env_origins[:,0].unsqueeze(1)).min(dim=1)[0] > -0.95)
    quat = asset.data.body_quat_w[:, feet_cfg.body_ids, :].squeeze(1)  # (num_envs, num_bodies, 4) -> (num_envs, 4)
    proj_grav = math_utils.quat_rotate_inverse(quat, asset.data.GRAVITY_VEC_W.unsqueeze(1))
    cosine = (-proj_grav[:, :,2]).clip(max=1, min=-1)#cos(30 degree)
    # angle = torch.acos(cosine)
    on_box_success = on_box_success.logical_and(cosine.min(dim=1)[0]>0.99875026)

    # masses= env.command_manager.get_term('climb_command').mass
    # body_coms = asset.data.body_com_pos_w[:, :, :2] - env.scene.env_origins[:, :2].unsqueeze(1) #shape: (num_envs, num_bodies)
    # CoM = torch.sum(body_coms * masses.unsqueeze(-1), dim=1) / torch.sum(masses.unsqueeze(-1), dim=1)

    # feet_pos = (asset.data.body_pos_w[:,feet_cfg.body_ids, :2].mean(dim=1) - env.scene.env_origins[:,:2])
    # feet_CoM_dist = torch.linalg.norm((CoM - feet_pos),dim=1)
    # feet_com_close = feet_CoM_dist < 0.1
    
    # on_box_success = on_box_success.logical_and(feet_com_close)

    # compute the violation
    max_contact = (torch.norm(contact_sensor.data.net_forces_w[:, upper_sensor_cfg.body_ids], dim=-1)).max(dim=-1)[0]
    no_contact = (max_contact < 1)

    return torch.where(on_box_success.logical_and(no_contact), weight_scale* torch.ones_like(max_contact), torch.zeros_like(max_contact))
    
class contact_on_wall(ManagerTermBase):
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
        asset: RigidObject = env.scene["robot"]
        sensor_body_names = [contact_sensor.body_names[i] for i in cfg.params['sensor_cfg'].body_ids]
        self.id_asset = asset.find_bodies(sensor_body_names, preserve_order=True)[0]

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,) -> torch.Tensor:

        # Return the unweighted reward for the termination terms
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: RigidObject = env.scene["robot"]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        contact = ( torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.).squeeze()

        # off_ground = asset.data.body_com_state_w[:, id_asset, 2] - env.scene.env_origins[:, 2].unsqueeze(1) > 0.12
        # below_box = asset.data.body_com_state_w[:, id_asset, 2] < 0.
        # near_wall = asset.data.body_com_state_w[:, id_asset, 0] - env.scene.env_origins[:, 0].unsqueeze(1) < wall_x
        off_ground = asset.data.body_com_state_w[:, self.id_asset, 2]  > 0.12
        below_box = asset.data.body_com_state_w[:, self.id_asset, 2] < env.scene.env_origins[:, 2].unsqueeze(1)
        near_wall = asset.data.body_com_state_w[:, self.id_asset, 0] - env.scene.env_origins[:, 0].unsqueeze(1) < -1


        contact_wall = contact & off_ground & below_box & near_wall
        # print('contact_wall:', contact_wall[0].nonzero().squeeze(-1).tolist())
        # id_list=torch.nonzero(contact_wall[0]).squeeze(-1).tolist()
        # print('contact_wall names:', [contact_sensor.body_names[i] for i in id_list]) 
    
        rew=torch.sum(contact_wall, dim=1).float()
        return rew

# def contact_on_wall(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, wall_x: float) -> torch.Tensor:
#     """Penalize contact forces as the amount of violations of the net contact force."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     net_contact_forces = contact_sensor.data.net_forces_w_history
#     # print('time:', env.episode_length_buf* env.step_dt)
#     # compute the violation
#     asset: RigidObject = env.scene["robot"]
#     id_asset = asset.find_bodies(sensor_cfg.body_names, preserve_order=True)[0]
#     # print('id_asset:', id_asset)
#     print('sesor body names:', [contact_sensor.body_names[i] for i in sensor_cfg.body_ids])
#     print('found bodies:', [asset.body_names[i] for i in id_asset])

#     contact = ( torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.).squeeze()

#     # off_ground = asset.data.body_com_state_w[:, id_asset, 2] - env.scene.env_origins[:, 2].unsqueeze(1) > 0.12
#     # below_box = asset.data.body_com_state_w[:, id_asset, 2] < 0.
#     # near_wall = asset.data.body_com_state_w[:, id_asset, 0] - env.scene.env_origins[:, 0].unsqueeze(1) < wall_x
#     off_ground = asset.data.body_com_state_w[:, id_asset, 2]  > 0.12
#     below_box = asset.data.body_com_state_w[:, id_asset, 2] < env.scene.env_origins[:, 2].unsqueeze(1)
#     near_wall = asset.data.body_com_state_w[:, id_asset, 0] - env.scene.env_origins[:, 0].unsqueeze(1) < -1


#     contact_wall = contact & off_ground & below_box & near_wall
#     # print('contact_wall:', contact_wall[0].nonzero().squeeze(-1).tolist())
#     # id_list=torch.nonzero(contact_wall[0]).squeeze(-1).tolist()
#     # print('contact_wall names:', [contact_sensor.body_names[i] for i in id_list]) 
   
#     rew=torch.sum(contact_wall, dim=1).float()
#     return rew
    
def contact_down(env: ManagerBasedRLEnv, upper_sensor_cfg: SceneEntityCfg,) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[upper_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # print('time:', env.episode_length_buf* env.step_dt)
    # compute the violation

    contact = ( torch.max(torch.norm(net_contact_forces[:, :, upper_sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.)
    #print('contact:', contact)

   
    rew=torch.sum(contact, dim=1).float()
    return  rew

def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    # compute the violation
    z_force = net_contact_forces[:, sensor_cfg.body_ids,2].abs()
    max_contact = torch.max(z_force,dim=-1)[0]-threshold
    # compute the penalty
    #print("VIOLATION: ", violation.clip(min=0.0))
    return max_contact.clip(min=0.0)

def self_collision(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, ground_sensor_cfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    ground_contact_sensor: ContactGroundSensorZ = env.scene.sensors[ground_sensor_cfg.name]

    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids,2]
    ground_contact_forces = ground_contact_sensor._data.force_matrix_w[:, ground_sensor_cfg.body_ids,0, 2]

    is_ground_like = (contact_forces - ground_contact_forces).abs() < 1


    self_collision = (contact_forces>1) * (~is_ground_like)
    return torch.sum(self_collision, dim=1).float()





def body_dragging(env: ManagerBasedRLEnv, vel_threshold: float , sensor_cfg: SceneEntityCfg, asset_cfg:SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize rigid bodies that are in contact whose velocity is over a threshold."""

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the contact forces
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if contact force is above threshold
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.
    # compute the dragging condition
    dragging_condition = torch.norm(asset.data.body_lin_vel_w[:,:,:2], dim=-1) > vel_threshold
    # input("Input Enter")
    # print("dragging condition: ", dragging_condition& is_contact)
    # print('dragging penalty: ', torch.sum(is_contact & dragging_condition, dim=1))
    return torch.sum(is_contact & dragging_condition, dim=1)

def body_slipping(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg:SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize rigid bodies that are in contact whose velocity is over a threshold."""

    # extract the used quantities (to enable type-hinting)
    #input("Input Enter")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the contact forces
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if contact force is above threshold
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.
    # compute the dragging condition
    vel_norm = torch.norm(asset.data.body_lin_vel_w[:,:,:3], dim=-1) 
    penalty = torch.where(is_contact, vel_norm, torch.zeros_like(vel_norm))
    # input("Input Enter")
    # print("slippig penalty: ", penalty)
    # print('slippig penalty sum: ', torch.sum(penalty, dim=1))
    return torch.sum(penalty, dim=1)


class body_slipping_down(ManagerTermBase):
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

# def body_slipping_down(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg:SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize rigid bodies that are in contact whose velocity is over a threshold."""

#     # extract the used quantities (to enable type-hinting)
#     #input("Input Enter")

#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     asset: RigidObject = env.scene[asset_cfg.name]
#     # compute the contact forces
#     net_contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
#     # check if contact force is above threshold
#     is_contact = net_contact_forces > 1.
#     id_asset = asset.find_bodies(contact_sensor.body_names, preserve_order=True)[0]
#     # compute the dragging condition
    
#     vel_norm = torch.norm(asset.data.body_lin_vel_w[:,id_asset,:2], dim=-1) 
#     penalty = torch.where(is_contact, vel_norm, torch.zeros_like(vel_norm))

#     return torch.sum(penalty, dim=1)

def body_slipping_l2(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg:SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize rigid bodies that are in contact whose velocity is over a threshold."""

    # extract the used quantities (to enable type-hinting)
    #input("Input Enter")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the contact forces
    net_contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    # check if contact force is above threshold
    is_contact = net_contact_forces > 1.
    id_asset = asset.find_bodies(sensor_cfg.body_names, preserve_order=True)[0]
    # compute the dragging condition
    vel_norm = torch.square(asset.data.body_lin_vel_w[:,id_asset,:3]).sum(dim=-1) 
    penalty = torch.where(is_contact, vel_norm, torch.zeros_like(vel_norm))

    return torch.sum(penalty, dim=1)

# def body_slipping_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg:SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize rigid bodies that are in contact whose velocity is over a threshold."""

#     # extract the used quantities (to enable type-hinting)
#     #input("Input Enter")

#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     asset: RigidObject = env.scene[asset_cfg.name]
#     # compute the contact forces
#     net_contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    
#     # check if contact force is above threshold
#     is_contact = net_contact_forces > 1.
#     id_asset = asset.find_bodies(contact_sensor.body_names, preserve_order=True)[0]

#     # compute the dragging condition
#     vel_norm = torch.norm(asset.data.body_lin_vel_w[:,id_asset,:3], dim=-1) 

#     penalty = torch.where(is_contact, vel_norm*net_contact_forces/100, torch.zeros_like(vel_norm))

#     return torch.sum(penalty, dim=1)

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


def feet_swing_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg:SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize rigid bodies that are in contact whose velocity is over a threshold."""

    # extract the used quantities (to enable type-hinting)
    #input("Input Enter")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the contact forces
    net_contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    
    # check if contact force is above threshold
    is_contact = net_contact_forces > 1.
    id_asset = asset.find_bodies(sensor_cfg.body_names, preserve_order=True)[0]

    # compute the dragging condition
    feet_height = asset.data.body_pos_w[:,id_asset,2] - env.scene.env_origins[:,2].unsqueeze(1)

    penalty = torch.where(~is_contact, torch.square(0.08 - feet_height), torch.zeros_like(feet_height))

    return torch.sum(penalty, dim=1)

def phase_contact(env:ManagerBasedRLEnv, period: float, sensor_cfg: SceneEntityCfg,) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        time = env.episode_length_buf * env.step_dt
    else:
        time = torch.zeros(env.num_envs, device=env.device)
    # print('time:', time)
    phase = (time % period) / period
    phase_2 = (phase + 0.5) % 1
    leg_phase = torch.cat([phase.unsqueeze(-1), phase_2.unsqueeze(-1)], dim=-1)
    res = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the contact forces
    net_contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    # check if contact force is above threshold
    is_contact = net_contact_forces > 1.
    for i in range(2):
        is_stance = leg_phase[:, i] < 0.55
        res += ~(is_contact[:,i] ^ is_stance)
    return res

"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_com_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)

def track_lin_vel_norm_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_com_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_b[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)

def foot_on_obj(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    #print("CONTACT SENSOR: ", contact_sensor)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_ids=[13,14,15,16]
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    is_contact_mask=is_contact.int()
    plane_height=0.02    
    feet_height=asset.data.body_pos_w[:, feet_ids,2] - plane_height
    contact_feet_height = feet_height * is_contact_mask
    # sum over contacts for each environment
    return torch.sum(torch.square(contact_feet_height), dim=1)