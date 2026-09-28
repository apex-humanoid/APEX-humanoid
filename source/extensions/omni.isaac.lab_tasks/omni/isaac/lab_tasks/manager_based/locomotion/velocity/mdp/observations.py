# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion task functions."""
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from omni.isaac.lab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera, ContactGroundSensorZ
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
import numpy as np

def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # input("Input Enter")
    return asset.data.root_com_ang_vel_b


def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def body_pos_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset torso position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :].squeeze(1)  # (num_envs, num_bodies, 3) -> (num_envs, 3)
    # body_pos[:,0] = body_pos[:,0] - env.scene.env_origins[:,0] + 2.5
    # return body_pos
    #print('torso pos w:', asset.data.body_pos_w[:, asset_cfg.body_ids, :].squeeze(1) - env.scene.env_origins)
    # pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :].squeeze(1) - env.scene.env_origins
    # pos[:,0]-=0.1
    # return pos
    return asset.data.body_pos_w[:, asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)


def body_quat_w(
    env: ManagerBasedEnv, make_quat_unique = True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset torso orientation (w, x, y, z) in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)  # (num_envs, num_bodies, 4) -> (num_envs, 4)
    # input("Input Enter")
    # print('body quat w:', quat)
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def body_lin_vel_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset torso linear velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]


def body_ang_vel_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset torso angular velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]


def joint_pos_rel_climbup(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # input("Input Enter")
    # print('joint names:', asset.joint_names)
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


def height_scan(env: ManagerBasedEnv,  sensor_cfg: SceneEntityCfg, asset_cfg=None, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.

    Robustness: If a ray does not hit within the sensor max distance, the underlying ray-caster fills the
    hit position with +inf. That would result in -inf heights. Here we replace non-finite hits with a value that
    saturates the height to the sensor's max range (max_distance - offset).
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    pos_z = sensor.data.pos_w[:, 2].unsqueeze(1)
    hit_z = sensor.data.ray_hits_w[..., 2]
    # print('hit z shape:', hit_z.shape)
    # if asset_cfg is not None:
    #     asset: RigidObject = env.scene[asset_cfg.name]
    #     torso_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :].squeeze(1)
    #     torso_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    #     pos_ = sensor.data.pos_w[:, :]
    #     quat_ = sensor.data.quat_w[:, :]
    #     print('pos:', pos_)
    #     print('torso pos:', torso_pos)
    #     print('pos error:', torch.norm(torso_pos - pos_))
    #     print('quat error:', torch.norm(torso_quat - quat_))
    ############ --- IGNORE ---
    # Replace non-finite hits (no intersection) with a virtual hit at (sensor_z - max_distance)
    # so that: height = pos_z - (pos_z - max_distance) - offset = max_distance - offset
    if not torch.isfinite(hit_z).all():
        fallback_hit_z = pos_z - sensor.cfg.max_distance
        hit_z = torch.where(torch.isfinite(hit_z), hit_z, fallback_hit_z)
    height = pos_z - hit_z - offset
    # Final guard: ensure finite tensor (should be already, but guard anyway)
    if not torch.isfinite(height).all():
        height = torch.nan_to_num(height, posinf=sensor.cfg.max_distance - offset, neginf=-(sensor.cfg.max_distance))
    height = torch.clamp(height, min=-1, max=2)
    return height


def box_height(env: ManagerBasedEnv) -> torch.Tensor:
    # climb_command = env.command_manager.get_command('climb_command')
    # return torch.where(climb_command>0, 0-env.scene.env_origins[:, 2], torch.zeros(env.num_envs,1, device=env.device))
    return (env.scene.env_origins[:, 2]).unsqueeze(-1)


def time(env: ManagerBasedEnv) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        #climb_command = env.command_manager.get_command('climb_command')
        #return torch.where((climb_command>0).unsqueeze(-1),torch.zeros(env.num_envs, 1,device=env.device),(env.episode_length_buf * env.step_dt).unsqueeze(-1))
        # return torch.zeros(env.num_envs,1, device=env.device)
        # input("Input Enter")
        # print('time:', env.episode_length_buf)
        return (env.episode_length_buf * env.step_dt).unsqueeze(-1)
    else:
        return torch.zeros(env.num_envs,1, device=env.device)


def progress_memory(env: ManagerBasedEnv) -> torch.Tensor:
    max_avg_height = env.command_manager.get_term('climb_command').max_avg_height.unsqueeze(-1)
    max_com_x = env.command_manager.get_term('climb_command').max_com_x.unsqueeze(-1)
    min_feet_angle=env.command_manager.get_term('climb_command').min_feet_angle
    min_feet_box=env.command_manager.get_term('climb_command').min_feet_box
    min_torso_angle = env.command_manager.get_term('climb_command').min_torso_angle.unsqueeze(-1)
    min_upper_force = env.command_manager.get_term('climb_command').min_upper_force.unsqueeze(-1)
    max_shoulder_height = env.command_manager.get_term('climb_command').max_avg_whole_height.unsqueeze(-1)
    min_feet_avg_com=env.command_manager.get_term('climb_command').min_feet_avg_com.unsqueeze(-1)


    # progress = torch.cat([max_avg_height, max_com_x, min_feet_angle, min_feet_box, min_torso_angle, min_upper_force, max_shoulder_height, min_feet_avg_com], dim=1)
    progress = torch.cat([max_avg_height, max_com_x, min_feet_angle, min_feet_box, min_torso_angle, min_upper_force, ], dim=1)
    return progress


def real_last_processed_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    
    return env.action_manager.get_term(action_name).processed_actions


def progress_memory_down(env: ManagerBasedEnv) -> torch.Tensor:
   
    min_avg_height = env.command_manager.get_term('climb_command').min_avg_height.unsqueeze(-1)
    min_com_x = env.command_manager.get_term('climb_command').min_com_x.unsqueeze(-1)

    progress = torch.cat([min_avg_height, min_com_x], dim=-1)
    return progress


def body_orientation(env,asset_cfg)-> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    

    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)  # (num_envs, num_bodies, 4) -> (num_envs, 4)
    proj_grav = math_utils.quat_rotate_inverse(quat, asset.data.GRAVITY_VEC_W.unsqueeze(1))
    #print('proj grav:', proj_grav)
    cosine = -proj_grav[:, :,2] 
    return cosine


def progress_memory_standup(env: ManagerBasedEnv) -> torch.Tensor:
    min_feet_angle=env.command_manager.get_term('climb_command').min_feet_angle
    min_torso_angle = env.command_manager.get_term('climb_command').min_torso_angle.unsqueeze(-1)
    min_upper_force = env.command_manager.get_term('climb_command').min_upper_force.unsqueeze(-1)
    max_shoulder_height = env.command_manager.get_term('climb_command').max_avg_whole_height.unsqueeze(-1)
    min_feet_avg_com=env.command_manager.get_term('climb_command').min_feet_avg_com.unsqueeze(-1)


    # progress = torch.cat([max_avg_height, max_com_x, min_feet_angle, min_feet_box, min_torso_angle, min_upper_force, max_shoulder_height, min_feet_avg_com], dim=1)
    progress = torch.cat([min_feet_angle, min_torso_angle, min_upper_force, max_shoulder_height,min_feet_avg_com], dim=1)
    return progress


def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_com_lin_vel_b


def joint_pos_rel_crawl(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # print('joint names:', asset.joint_names)
    # if hasattr(env, "episode_length_buf"):
    #     print('time:', env.episode_length_buf)
    target_joint_idx = 8
    target_joint_pos_rel = torch.mean(asset.data.joint_pos[:, target_joint_idx] - asset.data.default_joint_pos[:, target_joint_idx], dim=0)
    target_joint_name = asset.joint_names[target_joint_idx]
    # print(f"[INFO] Target joint: {target_joint_name} is {target_joint_pos_rel}")
    
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def velocity_commands(env: ManagerBasedRLEnv, command_name: str, lin_scale, ang_scale,) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    cmd = env.command_manager.get_command(command_name)
    # cmd[:,:2]*=lin_scale
    # cmd[:,2]*=ang_scale
    if hasattr(env, "is_waiting"):  # zero vel cmd for waiting env.
        cmd[env.is_waiting] = 0.0
        # print(f"[DEBUG] cmd set to zero!")
    return torch.cat((cmd[:,:2]*lin_scale,cmd[:,2].unsqueeze(-1)*ang_scale),dim=1)


def tri_phase(env: ManagerBasedRLEnv, period, command_name) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        time = env.episode_length_buf * env.step_dt
    else:
        time = torch.zeros(env.num_envs, device=env.device)
    
    phase = (time % period) / period
    sin_phase = torch.sin(2 * torch.pi * phase ).unsqueeze(1)
    cos_phase = torch.cos(2 * torch.pi * phase ).unsqueeze(1)
    ret_phase = torch.cat((sin_phase, cos_phase), dim=1)

    term = env.command_manager.get_term(command_name)
    vel_cmd = term.vel_command_b[:, :2]
    is_low_vel = torch.all(torch.abs(vel_cmd) < 0.1, dim=1)
    is_standing_env = term.is_standing_env
    
    is_zero_phase = is_low_vel | is_standing_env
    ret_phase[is_zero_phase] = 0.0

    # print(f"[DEBUG] ret phase is: {ret_phase}")
    # print(f"[DEBUG] NO Clipping Phase")
    return ret_phase
