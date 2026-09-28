# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion task functions."""
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
from omni.isaac.lab.sensors import ContactSensor, ContactGroundSensorZ

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


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
    # print('root height:', asset.data.root_link_pos_w[:, 2])
    # print('body_mean_feet_xy_dist:', body_mean_feet_xy_dist)
    # print('mean_height:', mean_height)
    # print('upper_contact:', upper_contact)
    return lose_balance


def root_height_below_minimum_standup(
    env: ManagerBasedRLEnv, shoulder_cfg, minimum_height: float, active=True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    max_shoulder_height = env.command_manager.get_term('climb_command').max_avg_whole_height
    shoulder_height = asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].mean(dim=-1) - env.scene.env_origins[:, 2]
    too_low = (shoulder_height < 0.7) | ((asset.data.root_link_pos_w[:, 2]) < minimum_height)
    #return (asset.data.root_link_pos_w[:, 2]-env.scene.env_origins[:,2]) < minimum_height
    too_far = torch.linalg.norm(asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2],dim=-1)> 0.85
    # print('root height:', asset.data.root_link_pos_w[:, 2])
    # print('xy dist:', torch.linalg.norm(asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2],dim=-1))
    # print('max_shoulder_height:', max_shoulder_height)
    # print('return value:', ((too_low & (max_shoulder_height>0.9)) | too_far ))
    return  ((too_low & (max_shoulder_height>0.9)) | too_far )*active


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


def kneeling(
    env: ManagerBasedRLEnv,  arm_cfg, active =True,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    knee_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    #arms_height = asset.data.body_pos_w[:, arm_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    # print('arm heights:', arms_height.min(dim=-1)[0])
    # print('knee heights:', knee_height.min(dim=-1)[0])
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)  # (num_envs, num_bodies, 4) -> (num_envs, 4)
    proj_grav = math_utils.quat_rotate_inverse(quat, asset.data.GRAVITY_VEC_W.unsqueeze(1))
    cosine = -proj_grav[:, :,2] 
    unstable = (cosine < 0.94).all(dim=1)
    # print('cosine:', cosine)
    kneeling = (knee_height < 0.25).all(dim=-1) #& (arms_height > 0.1).all(dim=-1)
    return (kneeling|unstable) & (env.episode_length_buf>200)


def leg_twisted(
    env: ManagerBasedRLEnv,  hip_pitch_cfg, foot_cfg,active = True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's leg is twisted.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    hip_yaw_joint_pos = env.scene['robot'].data.joint_pos[:, asset_cfg.joint_ids]
    knee_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    twist = torch.logical_and(hip_yaw_joint_pos.abs() > 0.8, knee_height < 0.07)

    # hip_pos = asset.data.body_pos_w[:, hip_pitch_cfg.body_ids, :]
    # hip_vec = hip_pos[:,1,:2] - hip_pos[:,0,:2]
    # hip_len_sq = torch.sum(hip_vec * hip_vec, dim=-1) 

    # foot_pos = asset.data.body_pos_w[:, foot_cfg.body_ids, :]
    # feet_vec = foot_pos[:,1,:2] - foot_pos[:,0,:2]
    # # dist_sq = torch.sum(feet_vec * feet_vec, dim=-1,)

    # hip_pos = asset.data.body_pos_w[:, hip_pitch_cfg.body_ids, :]
    # hip_vec = hip_pos[:,1,:2] - hip_pos[:,0,:2]
    # hip_len_sq = torch.sum(hip_vec * hip_vec, dim=-1) 
    # dot_feet_hip = torch.sum(feet_vec * hip_vec, dim=-1)
    # projected_width_sq = (dot_feet_hip ** 2) / hip_len_sq

    return (twist.any(dim=1))# | (hip_len_sq < 0.012) #| (projected_width_sq.squeeze(-1) > 0.49)


def torso_angle(
    env: ManagerBasedRLEnv,  active = True, torso_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's torso bend over.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[torso_cfg.name]

    torso_quat = asset.data.body_quat_w[:, torso_cfg.body_ids, :].squeeze(1)
    torso_proj_grav = math_utils.quat_rotate_inverse(torso_quat, asset.data.GRAVITY_VEC_W)
    torso_cosine = (-torso_proj_grav[:,2])
    torso_sin = (torso_proj_grav[:,0])
    return ((torso_cosine < -0.6)|(torso_sin<-0.5))*active  


def leg_stretched_up(
    env: ManagerBasedRLEnv,  active = True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's leg is stretched.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    CoM = (asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2] )
    feet_pos = asset.data.body_pos_w[:,asset_cfg.body_ids, :2] - env.scene.env_origins[:,:2].unsqueeze(1)
    feet_CoM_dist_l2 = torch.sum(torch.square(feet_pos - CoM.unsqueeze(1)), dim=-1)

    ending = env.episode_length_buf > 150
    # print('feet_CoM_dist:', feet_CoM_dist_l2.sqrt())
    # 0.4225 = 0.65^2, 0.25=0.5^2
    # return (feet_CoM_dist_l2>0.4225).any(dim=1) | (ending & (feet_CoM_dist_l2>0.25).any(dim=1))
    return (feet_CoM_dist_l2>0.4225).any(dim=1) | (ending & (feet_CoM_dist_l2>0.25).any(dim=1))


def leg_stretched(
    env: ManagerBasedRLEnv, active = True, threshold = 0.65, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
) -> torch.Tensor:
    """Terminate when the asset's leg is stretched.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    CoM = (asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2] )
    feet_pos = asset.data.body_pos_w[:,asset_cfg.body_ids, :2] - env.scene.env_origins[:,:2].unsqueeze(1)
    feet_CoM_dist = torch.linalg.norm(feet_pos - CoM.unsqueeze(1), dim=-1)
    # print('feet_CoM_dist:', feet_CoM_dist)
    return (feet_CoM_dist>threshold).all(dim=1) # any / all


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
