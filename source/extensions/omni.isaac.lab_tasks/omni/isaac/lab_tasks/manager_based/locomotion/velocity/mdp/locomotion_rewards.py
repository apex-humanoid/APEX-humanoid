# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion task functions."""
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorZ, ContactGroundSensorZ
from omni.isaac.lab.utils.math import (
    quat_rotate_inverse,
    yaw_quat,
    _R_from_quat_batch,
    euler_xyz_from_quat,
    quat_rotate,
)
from omni.isaac.lab.managers.manager_base import ManagerTermBase
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.assets import Articulation, RigidObject
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def downward_penalty_up(env, feet_cfg, shoulder_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    max_avg_height = env.command_manager.get_term('climb_command').max_avg_height
    climb_command = env.command_manager.get_command('climb_command')
    box_height = env.scene.env_origins[:,2]

    
    # input("Input Enter")
    # print('on box success:',on_box_success)
    # knee_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids,]
    # on_box_success = on_box_success.logical_and(knee_joint_pos.min(dim=1)[0] >= 2)


    current_height = torch.mean(asset.data.body_pos_w[:, :, 2].clip(max=box_height.unsqueeze(1)+0.02), dim=1)
    downward=max_avg_height-current_height
    penalty=torch.logical_or(downward>0 , downward.abs()<0.00001)
    on_foot_on_box = asset.data.body_pos_w[:, :, 2].min(dim=1)[0]>box_height
    # on_foot_on_box = (asset.data.body_pos_w[:, :, 0].min(dim=1)[0]-env.scene.env_origins[:,0] > -0.95) & (asset.data.body_pos_w[:, feet_cfg.body_ids, 2].min(dim=1)[0]>box_height)
    #torch.logical_and(asset.data.body_pos_w[:, feet_cfg.body_ids, 2].min(dim=1)[0]>box_height,asset.data.body_pos_w[:, feet_cfg.body_ids, 0].max(dim=1)[0]-env.scene.env_origins[:,0] > -0.95)

    penalty = penalty.logical_and(~on_foot_on_box)
    # input("Input Enter")
    # print('penalty after foot on box:',penalty)
    # bonus = (max_avg_height < box_height+0.019 ) & (box_height+0.019 <= current_height)

    env.command_manager.get_term('climb_command').update_max_avg_height(current_height)
    reward = torch.where(penalty, torch.ones_like(downward), torch.zeros_like(downward))
    reward = torch.where(climb_command > 0, reward, torch.zeros_like(reward))
    return reward# - bonus * 100


def com_backward_penalty_up(env, clip_x,feet_cfg,left_sensor_cfg: SceneEntityCfg, right_sensor_cfg: SceneEntityCfg,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[left_sensor_cfg.name]

    # left_legs_air_time=contact_sensor.data.current_air_time[:, left_sensor_cfg.body_ids]
    # right_legs_air_time=contact_sensor.data.current_air_time[:, right_sensor_cfg.body_ids]

    climb_command = env.command_manager.get_command('climb_command')
    max_com_x = env.command_manager.get_term('climb_command').max_com_x
    #print('max com x:',max_com_x)
    #compute body com X position relative to the env origin
    box_height = env.scene.env_origins[:,2]
    body_coms = asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1) #shape: (num_envs, num_bodies)
    
    masses= env.command_manager.get_term('climb_command').mass

    #calculate center of mass of all bodies, which means the mean of body coms multiplied by the mass of each body
    # current_com_x = torch.sum(body_coms * masses, dim=1) / torch.sum(masses, dim=1)
    # mean_x = torch.mean((asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).clip(max=-0.8), dim=1)
    if clip_x:
        current_com_x = torch.mean((asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).clip(max=-0.8), dim=1)
    else:
        current_com_x = torch.sum(body_coms * masses, dim=1) / torch.sum(masses, dim=1)

    backward = max_com_x - current_com_x
    # print('current com x:',current_com_x, 'max com x:', max_com_x)
    # bonus = (max_com_x < -0.75) & (-0.75 <= current_com_x)
    #success = (asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:,0].unsqueeze(1)).min(dim=1)[0] > -1.
    # success = torch.logical_and(asset.data.body_pos_w[:, feet_cfg.body_ids, 2].min(dim=1)[0]>0.2, asset.data.body_pos_w[:, feet_cfg.body_ids, 2].max(dim=1)[0]>box_height-0.2)
    # success = (asset.data.body_pos_w[:, :, 0].min(dim=1)[0]-env.scene.env_origins[:,0] > -0.95) & (asset.data.body_pos_w[:, feet_cfg.body_ids, 2].min(dim=1)[0]>box_height)
    # torch.logical_and(asset.data.body_pos_w[:, feet_cfg.body_ids, 2].min(dim=1)[0]>0.2, asset.data.body_pos_w[:, feet_cfg.body_ids, 0].max(dim=1)[0]-env.scene.env_origins[:,0] > -0.95)
    success = (asset.data.body_pos_w[:, :, 0].min(dim=1)[0]-env.scene.env_origins[:,0] > -1) 
    # input('Press Enter to continue...')  # Debugging line to pause execution
    # print('feet height:',asset.data.body_pos_w[:, feet_cfg.body_ids, 2].min(dim=1)[0])
    # on_box = asset.data.body_pos_w[:, :, 2].min(dim=1)[0] > (box_height-0.1)
    # success = torch.logical_and(success, on_box)
    # input('Press Enter to continue...')  # Debugging line to pause execution
    # print('episode step:',env.episode_length_buf)
    # print('min com x:',(asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:,0].unsqueeze(1)).min(dim=1)[0])
    
    penalty = torch.logical_or(backward > 0, backward.abs() < 0.001)
    penalty = torch.logical_and(penalty > 0,~success)
    #penalty = torch.logical_and(backward > 0,~success)
    # input("Input Enter")
    # print('success:',success)
    env.command_manager.get_term('climb_command').update_max_com_x(current_com_x)
    reward = torch.where(penalty, torch.ones_like(backward), torch.zeros_like(backward))
    #print('penalty:',torch.where(climb_command > 0, reward, torch.zeros_like(reward)))
    # if climb_command.any() > 0 :
    #     print('climb command:',climb_command)
    reward= torch.where(climb_command > 0, reward, torch.zeros_like(reward))
    return reward# - bonus * 100


def wait_penalty_up(env, command_name: str,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for waiting."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    climb_command = env.command_manager.get_command('climb_command')
    norm_vel=asset.data.root_lin_vel_w.norm(dim=1)
    # input("Input Enter")
    box_height = env.scene.env_origins[:,2]
    current_height = asset.data.root_pos_w[:, 2]
    min_height = asset.data.body_pos_w[:,:, 2].min(dim=1)[0]
    #print('current_height:',current_height)
    # pos_error = torch.norm(env.command_manager.get_command(command_name)[:, :2]
    #                        +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2], dim=1)
    #print('wait penalty:',torch.where(torch.logical_and(norm_vel<0.15, pos_error>0.2),torch.ones_like(norm_vel),torch.zeros_like(norm_vel)))
    # return torch.where(torch.logical_and(norm_vel<0.15, pos_error>0.2),
    #                    torch.ones_like(norm_vel),torch.zeros_like(norm_vel))
    #print('wait penalty:',torch.where(torch.logical_and(norm_vel<0.15, current_height<-0.01),torch.ones_like(norm_vel),torch.zeros_like(norm_vel)))
    reward = torch.where(torch.logical_and(norm_vel<0.15, min_height<box_height),
                          torch.ones_like(norm_vel),torch.zeros_like(norm_vel))
    return torch.where(climb_command > 0, reward, torch.zeros_like(reward))


def joint_velocity_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset=env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0)
    return torch.sum(out_of_limits[:, :23], dim=1)


def hip_yaw_roll_joint_deviation(
    env: ManagerBasedRLEnv, hip_yaw_cfg, hip_roll_cfg, hip_pitch_cfg
) -> torch.Tensor:
    """Penalize hip joint deviation from zero."""
    # extract the used quantities (to enable type-hinting)
    asset=env.scene[hip_yaw_cfg.name]
    # compute out of limits constraints
    hip_yaw_joint = torch.abs(asset.data.joint_pos[:, hip_yaw_cfg.joint_ids])
    yaw_penalty = (torch.max(hip_yaw_joint,dim=-1)[0]>1.5) 
    hip_roll_joint = torch.abs(asset.data.joint_pos[:, hip_roll_cfg.joint_ids])
    roll_penalty = (torch.max(hip_roll_joint,dim=-1)[0]>1.4)
    
    return yaw_penalty | roll_penalty


def waist_joint_deviation(
    env: ManagerBasedRLEnv, waist_cfg) -> torch.Tensor:
    """Penalize waist joint deviation from zero."""
    # extract the used quantities (to enable type-hinting)
    asset=env.scene[waist_cfg.name]
    # compute out of limits constraints
    waist_joint = torch.abs(asset.data.joint_pos[:, waist_cfg.joint_ids])

    return (waist_joint > 1.4).squeeze(-1)


def base_lin_ang_acc_climbup(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset=env.scene[asset_cfg.name]
    root_id=asset.find_bodies("torso_link")[0]
    
    return (torch.sum(torch.square(asset.data.body_lin_acc_w[:, root_id,:]), dim=-1).squeeze(-1)
            +0.02*torch.sum(torch.square(asset.data.body_ang_acc_w[:,root_id, :]), dim=-1).squeeze(-1))


def group_air_time_orig(env: ManagerBasedRLEnv, upper_sensor_cfg: SceneEntityCfg, lower_sensor_cfg: SceneEntityCfg,feet_sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # penelize the time spent when there are at least one group of bodies in the air
    asset = env.scene[asset_cfg.name]

    contact_sensor: ContactGroundSensorZ = env.scene.sensors[upper_sensor_cfg.name]
    
    upper_bodies_air_time=contact_sensor.data.current_air_time[:, upper_sensor_cfg.body_ids]
    lower_bodies_air_time=contact_sensor.data.current_air_time[:, lower_sensor_cfg.body_ids]
    feet_air_time = contact_sensor.data.current_air_time[:, feet_sensor_cfg.body_ids]
    

    upper_air_time=upper_bodies_air_time.min(dim=1)[0]
    lower_air_time=lower_bodies_air_time.min(dim=1)[0]
    
    #TODO: check if max or min
    feet_air_time=feet_air_time.max(dim=1)[0]

    upper_lower_air_time=torch.maximum(upper_air_time, lower_air_time)

    air_time=torch.minimum(upper_lower_air_time, feet_air_time)



    climb_command = env.command_manager.get_command('climb_command')
    air_time=torch.where(climb_command > 0, air_time, torch.zeros_like(air_time))

    return (torch.exp(20*air_time)-1).clip(max=200.0)


def body_pressure(env: ManagerBasedRLEnv, asset_cfg, sensor_cfg,Lx: float = 0.20311,
                Ly: float = 0.06547,
                Lz: float = 0.01851,
                rO_local = (-0.03592, 0.0, 0.02517),
                dtype    = torch.float32, h_plane= 0.01) -> torch.Tensor:

    hx = Lx / 2.0
    hy = Ly / 2.0
    hz = Lz / 2.0
    rO = torch.as_tensor(rO_local, device=env.device, dtype=dtype)

    # 8 local vertices  (1,8,3)  -> kept as buffer
    v_local = torch.tensor([[sx*hx, sy*hy, sz*hz]
                                    for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)],
                                device=env.device, dtype=dtype)          # (8,3)

    # 12 edge index pairs (1,12,2)
    edges   = torch.tensor([[0,1],[0,2],[0,4], [1,3],[1,5], [2,3],[2,6],
                                    [3,7], [4,5],[4,6],[5,7],[6,7]],
                                device=env.device, dtype=torch.long)     # (12,2)

   
    asset  = env.scene[asset_cfg.name]
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]


    quat = asset.data.body_quat_w[:,asset_cfg.body_ids]
    O_world = asset.data.body_pos_w[:,asset_cfg.body_ids]#-env.scene.env_origins.unsqueeze(1)  # (n,k,3)  
    n, k, _ = O_world.shape
    B       = n * k
    dev     = env.device
    dt      = dtype

    # flatten
    Of = O_world.reshape(B,3)
    Quat =quat.reshape(B,4)

    # rotation & center
    R  = _R_from_quat_batch(Quat)
    Cf = Of - (R @ rO)                          # (B,3)

    # world vertices   (B,8,3)
    verts_w = torch.einsum('bij,vj->bvi', R, v_local) + Cf[:,None,:]

    # edges
    verts_pair = verts_w[:, edges]              # (B,12,2,3)
    z_pair     = verts_pair[...,2]                   # (B,12,2)

    # broadcast h_plane to (B,)
    h_plane = torch.as_tensor(h_plane, device=dev, dtype=dt)
    if h_plane.ndim == 0:
        hb = h_plane.expand(B)
    else:
        hb = h_plane.reshape(-1) if h_plane.numel()==B else h_plane.repeat_interleave(k)
        hb = hb.to(device=dev, dtype=dt)

    side     = z_pair - hb[:,None,None]              # (B,12,2)
    straddle = (side[...,0]*side[...,1]) < 0         # (B,12)

    # t & intersection points
    t        = (hb[:,None] - z_pair[...,0]) / (z_pair[...,1]-z_pair[...,0] + 1e-12)
    t        = t.clamp(0,1)
    inter_xy = verts_pair[...,0,:2] + t.unsqueeze(-1) * (verts_pair[...,1,:2]-verts_pair[...,0,:2])
    inter_xy[~straddle] = float('nan')

    # --- GPU vectorised centroid, polar sort, shoelace ---

    mask      = torch.isfinite(inter_xy[...,0])           # (B,12)
    valid_cnt = mask.sum(dim=1)                           # (B,)

    # 质心
    centroid  = (torch.where(mask.unsqueeze(-1), inter_xy, 0.)
                    .sum(dim=1) / valid_cnt.clamp(min=1).unsqueeze(-1))   # (B,2)

    dx = inter_xy[...,0] - centroid[:,0,None]
    dy = inter_xy[...,1] - centroid[:,1,None]
    angles = torch.atan2(dy, dx)
    angles[~mask] = float('inf')                          # 无效点排到末尾

    idx         = torch.argsort(angles, dim=1)            # (B,12)
    pts_sorted  = torch.gather(inter_xy, 1, idx.unsqueeze(-1).expand(-1,-1,2))
    mask_sorted = torch.gather(mask,      1, idx)         # 有效 True 全在前面

    # 真实每批顶点数 m 以及全局最大 m_max (≤6)
    m      = mask_sorted.sum(dim=1)                       # (B,)
    m_max  = int(m.max().item())
    if m_max == 0:
        return torch.zeros(n, device=dev, dtype=dt)

    # 取前 m_max 个位置的坐标 (不足 m_max 的行后面是无效点，但我们会屏蔽)
    pts_m  = pts_sorted[:, :m_max, :]                     # (B, m_max, 2)

    x = pts_m[...,0]                                      # (B, m_max)
    y = pts_m[...,1]

    # 构造 next 索引：对每一行 i, next_j = (j+1) % m_i
    arange = torch.arange(m_max, device=x.device).unsqueeze(0)          # (1, m_max)
    m_b    = m.unsqueeze(1).clamp(min=1)                                # (B,1)
    next_idx = (arange + 1) % m_b                                       # (B, m_max)

    # 按行 gather 下一个顶点
    x_next = torch.gather(x, 1, next_idx)
    y_next = torch.gather(y, 1, next_idx)

    # 只保留 j < m_i 的位置
    valid_pos = arange < m_b                                            # (B, m_max)

    cross = (x * y_next - y * x_next)
    cross = torch.where(valid_pos, cross, torch.zeros_like(cross))

    areas = 0.5 * cross.sum(dim=1).abs()
    areas[m < 3] = 0.   # 少于3点无面积

    #input("Input Enter")
    #print('area:', areas.view(n,k))
    area = areas.view(n,k)
    net_contact_forces = contact_sensor.data.net_forces_w
    max_contact = torch.norm(net_contact_forces[:,  sensor_cfg.body_ids], dim=-1)
    #print('max_contact:',max_contact)
    #area = torch.zeros_like(area)
    pressure=torch.where(area>1e-8,max_contact/area,0)
    #print('pressure:',pressure)
    # print('max pressure:',torch.max(pressure,dim=1).values)
    max_pressure = torch.max(pressure, dim=1).values
    penalty = (max_pressure-100000).clip(min=0) * 0.0001  # penalize high pressure
    return penalty.clip(max=100.0)  # clip the penalty to avoid extreme values


def upward_penalty(env, sensor_cfg,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]

    min_avg_height = env.command_manager.get_term('climb_command').min_avg_height
    #input("Input Enter")
    #print('asset cfg:',asset_cfg)
    
    box_height = env.scene.env_origins[:,2]
    current_height = torch.mean(asset.data.body_pos_w[:, :, 2].clip(max=box_height.unsqueeze(1)+0.02), dim=1)
    #print('root height:',asset.data.root_pos_w[:, 2], "root x:", asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0])
    upward = current_height - min_avg_height

    # feet_near_grd = asset.data.body_pos_w[:,asset_cfg.body_ids, 2].max(dim=-1)[0] < 0.036
    # net_contact_forces = contact_sensor.data.net_forces_w
    # feet_contact = (torch.norm(net_contact_forces[:,  sensor_cfg.body_ids], dim=-1)>0.1).all(dim=-1)

    # feet_near_grd = asset.data.body_pos_w[:,asset_cfg.body_ids, 2].min(dim=-1)[0] < 0.1
    # net_contact_forces = contact_sensor.data.net_forces_w
    # feet_contact = (torch.norm(net_contact_forces[:,  sensor_cfg.body_ids], dim=-1)>0.1).any(dim=-1)
    # root_height = asset.data.root_pos_w[:, 2]<=0.8
    # root_off_box = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0] < -1
    # success = torch.logical_and(feet_near_grd, feet_contact)
    # success = torch.logical_and(success, root_height)
    # success = torch.logical_and(success, root_off_box)
    #print('feet near grd:',asset.data.body_pos_w[:,asset_cfg.body_ids, 2])
    #print('upward:',upward)
    #print('success:',success)

    success = asset.data.body_pos_w[:,asset_cfg.body_ids, 2].max(dim=-1)[0] < 0.1
    
    penalty=torch.logical_or(upward>0 , upward.abs()<0.00001)
    #penalty=downward>0
    penalty=penalty.logical_and(~success)
    # print('penalty after:',penalty)
    #print('penalty:',penalty)
    env.command_manager.get_term('climb_command').update_min_avg_height(current_height)
    reward = torch.where(penalty, torch.ones_like(upward), torch.zeros_like(upward))
    return reward


def com_forward_penalty(env,upper_cfg,upper_sensor_cfg,sensor_cfg: SceneEntityCfg,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]


    min_com_x = env.command_manager.get_term('climb_command').min_com_x
    #compute body com X position relative to the env origin
    body_coms = (asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1)).clip(min=-1) #shape: (num_envs, num_bodies)
    body_coms_z =asset.data.body_com_pos_w[:, upper_cfg.body_ids, 2] - env.scene.env_origins[:, 2].clip(max=0.78).unsqueeze(1)
    # print('innner_body_com:', body_coms.max(dim=1)[0])  
    masses= env.command_manager.get_term('climb_command').mass
    #calculate center of mass of all bodies, which means the mean of body coms multiplied by the mass of each body
    current_com_x = torch.sum(body_coms * masses, dim=1) / torch.sum(masses, dim=1)
    #input("Input Enter")
    # print('com:',current_com_x)
    # print('min com x:',min_com_x)
    forward = current_com_x - min_com_x
    #print('forward:',forward)
    # feet_near_grd = asset.data.body_pos_w[:,asset_cfg.body_ids, 2].max(dim=-1)[0] < 0.036
    # net_contact_forces = contact_sensor.data.net_forces_w
    # feet_contact = (torch.norm(net_contact_forces[:,  sensor_cfg.body_ids], dim=-1)>0.1).all(dim=-1)
    # success = torch.logical_and(feet_near_grd, feet_contact)

    # feet_near_grd = asset.data.body_pos_w[:,asset_cfg.body_ids, 2].min(dim=-1)[0] < 0.1
    # net_contact_forces = contact_sensor.data.net_forces_w
    # feet_contact = (torch.norm(net_contact_forces[:,  sensor_cfg.body_ids], dim=-1)>0.1).any(dim=-1)
    # root_height = asset.data.root_pos_w[:, 2]<=0.8
    # root_off_box = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0] < -1
    # success = torch.logical_and(feet_near_grd, feet_contact)
    # success = torch.logical_and(success, root_height)
    # success = torch.logical_and(success, root_off_box)
    # upper_body_no_contact = contact_sensor.data.net_forces_w[:, upper_sensor_cfg.body_ids, 2].abs().max(dim=-1)[0] < 0.1
    # success = upper_body_no_contact & (body_coms.max(dim=1)[0] < -0.8)
    success = (body_coms.max(dim=1)[0] < -0.8) & (body_coms_z.min(dim=1)[0] > 0.05)
    # print('body coms z:', body_coms_z.min(dim=1)[0])
    # print('forward success:',success)
    
    #print('current root x:',asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0])
    penalty = torch.logical_or(forward > 0, forward.abs() < 0.0001)
    penalty = torch.logical_and(penalty > 0,~success)
    #print('penalty:',penalty)

    env.command_manager.get_term('climb_command').update_min_com_x(current_com_x)
    reward = torch.where(penalty, torch.ones_like(forward), torch.zeros_like(forward))

    return reward


def down_wait_penalty(env, command_name: str,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for waiting."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    norm_vel=asset.data.root_lin_vel_w.norm(dim=1)
    # input("Input Enter")
    # print('norm vel:',norm_vel) 
    box_height = env.scene.env_origins[:,2]
    #current_height = torch.mean(asset.data.body_pos_w[:, :, 2].clip(max=box_height.unsqueeze(1)), dim=1)
    root_x= asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    #print('current_height:',current_height)
    # pos_error = torch.norm(env.command_manager.get_command(command_name)[:, :2]
    #                        +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2], dim=1)
    #print('wait penalty:',torch.where(torch.logical_and(norm_vel<0.15, pos_error>0.2),torch.ones_like(norm_vel),torch.zeros_like(norm_vel)))
    # return torch.where(torch.logical_and(norm_vel<0.15, pos_error>0.2),
    #                    torch.ones_like(norm_vel),torch.zeros_like(norm_vel))
    # input('input enter')
    # print('norm vel:',norm_vel)
    # print('wait penalty:',torch.where(torch.logical_and(norm_vel<0.15, current_height>box_height-0.05),
    #                       torch.ones_like(norm_vel),torch.zeros_like(norm_vel)))

    reward = torch.where(torch.logical_and(norm_vel<0.1, root_x>-0.9),
                          torch.ones_like(norm_vel),torch.zeros_like(norm_vel))
    return reward


def standing_lin_vel_down(
    env: ManagerBasedRLEnv, feet_cfg,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the linear velocity of the base in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    lin_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    reward =  torch.sum(torch.square(lin_vel), dim=-1).squeeze(-1)  # L2 penalty on the linear velocity
    reward =  torch.exp(-5 * reward)  # Exponential kernel to penalize deviations from the desired height
    # root_x= asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    feet_near_grd = asset.data.body_pos_w[:,feet_cfg.body_ids, 2].max(dim=-1)[0] < 0.1
    return (feet_near_grd & (env.episode_length_buf>200)).float()*reward


def standing_ang_vel_down(
    env: ManagerBasedRLEnv, feet_cfg,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the linear velocity of the base in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    reward =  torch.sum(torch.square(ang_vel), dim=-1).squeeze(-1)  # L2 penalty on the linear velocity
    reward =  torch.exp(-2 * reward)  # Exponential kernel to penalize deviations from the desired height
    # root_x= asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    feet_near_grd = asset.data.body_pos_w[:,feet_cfg.body_ids, 2].max(dim=-1)[0] < 0.1
    return (feet_near_grd & (env.episode_length_buf>200)).float()*reward


def standing_height_down(
    env: ManagerBasedRLEnv,  desired_height: float ,sensor_cfg,feet_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the height of the base in the world frame."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]

    asset = env.scene[asset_cfg.name]
    feet = env.scene[feet_cfg.name]
    # compute out of limits constraints
    #height = (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).clip(max=desired_height)  # relative to the environment origin
    height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].mean(dim=-1).clip(max=desired_height)
    #height = (asset.data.root_pos_w[:, 2])
    #print('height:',asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2])
    # reward = torch.square(height - desired_height)
    # reward=torch.exp(-20*reward)  # Exponential kernel to penalize deviations from the desired height
    reward = ((height - 0.8)/ (desired_height - 0.8)).clip(min=0, max=1)
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
    # return torch.where(torch.logical_and(climb_command == 0,root_x<-0.9), reward,torch.zeros_like(reward), )
    return stage_reward


def base_lin_ang_acc_climbdown(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset=env.scene[asset_cfg.name]
    
    return (torch.sum(torch.square(asset.data.body_lin_acc_w[:, asset_cfg.body_ids,:]), dim=-1).squeeze(-1)
            +0.02*torch.sum(torch.square(asset.data.body_ang_acc_w[:,asset_cfg.body_ids, :]), dim=-1).squeeze(-1))


def body_pressure_down(env: ManagerBasedRLEnv, asset_cfg, sensor_cfg,Lx: float = 0.20311,
                Ly: float = 0.06547,
                Lz: float = 0.01851,
                rO_local = (-0.03592, 0.0, 0.02517),
                dtype    = torch.float32, h_plane= 0.01) -> torch.Tensor:

    hx = Lx / 2.0
    hy = Ly / 2.0
    hz = Lz / 2.0
    rO = torch.as_tensor(rO_local, device=env.device, dtype=dtype)

    # 8 local vertices  (1,8,3)  -> kept as buffer
    v_local = torch.tensor([[sx*hx, sy*hy, sz*hz]
                                    for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)],
                                device=env.device, dtype=dtype)          # (8,3)

    # 12 edge index pairs (1,12,2)
    edges   = torch.tensor([[0,1],[0,2],[0,4], [1,3],[1,5], [2,3],[2,6],
                                    [3,7], [4,5],[4,6],[5,7],[6,7]],
                                device=env.device, dtype=torch.long)     # (12,2)

   
    asset  = env.scene[asset_cfg.name]
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]


    quat = asset.data.body_quat_w[:,asset_cfg.body_ids]
    O_world = asset.data.body_pos_w[:,asset_cfg.body_ids]#-env.scene.env_origins.unsqueeze(1)  # (n,k,3)  
    n, k, _ = O_world.shape
    B       = n * k
    dev     = env.device
    dt      = dtype

    # flatten
    Of = O_world.reshape(B,3)
    Quat =quat.reshape(B,4)

    # rotation & center
    R  = _R_from_quat_batch(Quat)
    Cf = Of - (R @ rO)                          # (B,3)

    # world vertices   (B,8,3)
    verts_w = torch.einsum('bij,vj->bvi', R, v_local) + Cf[:,None,:]

    # edges
    verts_pair = verts_w[:, edges]              # (B,12,2,3)
    z_pair     = verts_pair[...,2]                   # (B,12,2)

    # broadcast h_plane to (B,)
    h_plane = torch.as_tensor(h_plane, device=dev, dtype=dt)
    if h_plane.ndim == 0:
        hb = h_plane.expand(B)
    else:
        hb = h_plane.reshape(-1) if h_plane.numel()==B else h_plane.repeat_interleave(k)
        hb = hb.to(device=dev, dtype=dt)

    side     = z_pair - hb[:,None,None]              # (B,12,2)
    straddle = (side[...,0]*side[...,1]) < 0         # (B,12)

    # t & intersection points
    t        = (hb[:,None] - z_pair[...,0]) / (z_pair[...,1]-z_pair[...,0] + 1e-12)
    t        = t.clamp(0,1)
    inter_xy = verts_pair[...,0,:2] + t.unsqueeze(-1) * (verts_pair[...,1,:2]-verts_pair[...,0,:2])
    inter_xy[~straddle] = float('nan')

    # --- GPU vectorised centroid, polar sort, shoelace ---

    mask      = torch.isfinite(inter_xy[...,0])           # (B,12)
    valid_cnt = mask.sum(dim=1)                           # (B,)

    # 质心
    centroid  = (torch.where(mask.unsqueeze(-1), inter_xy, 0.)
                    .sum(dim=1) / valid_cnt.clamp(min=1).unsqueeze(-1))   # (B,2)

    dx = inter_xy[...,0] - centroid[:,0,None]
    dy = inter_xy[...,1] - centroid[:,1,None]
    angles = torch.atan2(dy, dx)
    angles[~mask] = float('inf')                          # 无效点排到末尾

    idx         = torch.argsort(angles, dim=1)            # (B,12)
    pts_sorted  = torch.gather(inter_xy, 1, idx.unsqueeze(-1).expand(-1,-1,2))
    mask_sorted = torch.gather(mask,      1, idx)         # 有效 True 全在前面

    # 真实每批顶点数 m 以及全局最大 m_max (≤6)
    m      = mask_sorted.sum(dim=1)                       # (B,)
    m_max  = int(m.max().item())
    if m_max == 0:
        return torch.zeros(n, device=dev, dtype=dt)

    # 取前 m_max 个位置的坐标 (不足 m_max 的行后面是无效点，但我们会屏蔽)
    pts_m  = pts_sorted[:, :m_max, :]                     # (B, m_max, 2)

    x = pts_m[...,0]                                      # (B, m_max)
    y = pts_m[...,1]

    # 构造 next 索引：对每一行 i, next_j = (j+1) % m_i
    arange = torch.arange(m_max, device=x.device).unsqueeze(0)          # (1, m_max)
    m_b    = m.unsqueeze(1).clamp(min=1)                                # (B,1)
    next_idx = (arange + 1) % m_b                                       # (B, m_max)

    # 按行 gather 下一个顶点
    x_next = torch.gather(x, 1, next_idx)
    y_next = torch.gather(y, 1, next_idx)

    # 只保留 j < m_i 的位置
    valid_pos = arange < m_b                                            # (B, m_max)

    cross = (x * y_next - y * x_next)
    cross = torch.where(valid_pos, cross, torch.zeros_like(cross))

    areas = 0.5 * cross.sum(dim=1).abs()
    areas[m < 3] = 0.   # 少于3点无面积

    #input("Input Enter")
    #print('area:', areas.view(n,k))
    area = areas.view(n,k)
    net_contact_forces = contact_sensor.data.net_forces_w
    max_contact = torch.norm(net_contact_forces[:,  sensor_cfg.body_ids], dim=-1)
    #print('max_contact:',max_contact)
    #area = torch.zeros_like(area)
    pressure=torch.where(area>1e-8,max_contact/area,0)

    # print('pressure:',pressure)
    #print('pressure:',pressure)
    # print('max pressure:',torch.max(pressure,dim=1).values)
    max_pressure = torch.max(pressure, dim=1).values
    penalty = (max_pressure-10000).clip(min=0) * 0.00001  # penalize high pressure
    return penalty.clip(max=100.0)  # clip the penalty to avoid extreme values

def upward_penalty_ly(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    min_avg_height = env.command_manager.get_term('climb_command').min_avg_height
    # print('asset cfg:',asset_cfg)
    
    current_height = torch.mean(asset.data.body_pos_w[:, :, 2], dim=1)
    # print('root height:',asset.data.root_pos_w[:, 2], "root x:", asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0])
    upward = current_height - min_avg_height
    success = current_height < 0.35
    # print('current height:',current_height)
    penalty = torch.logical_or(upward > 0 , upward.abs()<0.0005)
    #penalty=downward>0
    penalty=penalty.logical_and(~success)
    # print('penalty after:',penalty)
    #print('penalty:',penalty)
    env.command_manager.get_term('climb_command').update_min_avg_height(current_height)
    reward = torch.where(penalty, torch.ones_like(upward), torch.zeros_like(upward))
    return reward


def head_upward_penalty(env, shoulder_cfg,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

   

    min_head_height = env.command_manager.get_term('climb_command').min_head_height
    # print('asset cfg:',asset_cfg)
    
    shoulder_height = asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].mean(dim=-1)  
    upward = shoulder_height - min_head_height
    success = shoulder_height < 0.27
    # print('head upward:',upward)

    # TODO: increase threshold value, terminate lean back
    penalty = torch.logical_or(upward > 0 , upward.abs()<0.0005) 
    penalty=penalty.logical_and(~success)

    env.command_manager.get_term('climb_command').update_min_head_height(shoulder_height)
    reward = torch.where(penalty, torch.ones_like(upward), torch.zeros_like(upward))
    return reward


def hip_joint_deviation_liedown(
    env: ManagerBasedRLEnv, hip_yaw_cfg, hip_roll_cfg, hip_pitch_cfg
) -> torch.Tensor:
    """Penalize hip joint deviation from zero."""
    # extract the used quantities (to enable type-hinting)
    asset=env.scene[hip_yaw_cfg.name]
    # compute out of limits constraints
    hip_yaw_joint = torch.abs(asset.data.joint_pos[:, hip_yaw_cfg.joint_ids])
    yaw_penalty = (torch.max(hip_yaw_joint,dim=-1)[0]>1.4) | (torch.min(hip_yaw_joint,dim=-1)[0]>0.9)

    hip_roll_joint = torch.abs(asset.data.joint_pos[:, hip_roll_cfg.joint_ids])
    roll_penalty = (torch.max(hip_roll_joint,dim=-1)[0]>1.4) | (torch.min(hip_roll_joint,dim=-1)[0]>0.9)

    hip_pitch_joint = asset.data.joint_pos[:, hip_pitch_cfg.joint_ids]
    pitch_penalty = (hip_pitch_joint > 0.6).any(dim=-1)

    # input('input enter')
    # print('hip pitch joint:', hip_pitch_joint)
    # print('hip roll joint:', hip_roll_joint)
    # print('hip yaw joint:', asset.data.joint_pos[:, hip_yaw_cfg.joint_ids])
    # print('hip penalty:', yaw_penalty | roll_penalty | pitch_penalty)

    return yaw_penalty | roll_penalty | pitch_penalty


def foot_dist_l2_down(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the distance between the feet."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    dist = torch.norm(foot_pos[:, 0, :2] - foot_pos[:, 1, :2], dim=1)
    on_box_for_angle = torch.logical_and(foot_pos[:,:, 2].min(dim=1)[0]>env.scene.env_origins[:,2], foot_pos[:,:, 0].min(dim=1)[0]-env.scene.env_origins[:,0] > -0.95)

    reward_large = torch.where((dist >= 0.6) & on_box_for_angle, torch.ones_like(dist), torch.zeros_like(dist))
    reward_small = torch.where((dist < 0.12) & on_box_for_angle, torch.ones_like(dist), torch.zeros_like(dist))
    return reward_large + reward_small


def stand_downward_penalty_up(env, upper_sensor_cfg, sensor_cfg, feet_cfg, torso_cfg, shoulder_cfg,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset = env.scene[asset_cfg.name]
    climb_command = env.command_manager.get_command('climb_command')
    box_height = env.scene.env_origins[:,2]

   
    
    # print('on box success:',on_box_success)

    # Shoulder height progress
    max_avg_shoulder_height = env.command_manager.get_term('climb_command').max_avg_whole_height
    shoulder_height = asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].mean(dim=-1) - box_height
    current_shoulder_height = (env.episode_length_buf>15).float() * shoulder_height
    whole_downward=max_avg_shoulder_height-current_shoulder_height
    height_penalty=torch.logical_or(whole_downward>0 , whole_downward.abs()<0.001) #0.0001
    env.command_manager.get_term('climb_command').update_max_avg_whole_height(current_shoulder_height)


    # Final penalty computation
    # penalty = torch.logical_and(height_penalty, angle_penalty)
    penalty = height_penalty

    penalty = penalty.logical_and(env.episode_length_buf>15)

    success = shoulder_height > 1
    penalty=penalty.logical_and(~success)
    # print('whole downward:',whole_downward)
    # print('penalty:',penalty)
    
    # reward = torch.where(penalty, torch.ones_like(whole_downward), torch.zeros_like(whole_downward))
    #print('weight scale:',weight_scale)
    return penalty


def feet_com_farther_penalty(env,shoulder_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]


    # Compute feet CoM distance
    min_feet_com=env.command_manager.get_term('climb_command').min_feet_com
    min_feet_avg_com=env.command_manager.get_term('climb_command').min_feet_avg_com

    masses= env.command_manager.get_term('climb_command').mass
    body_coms = asset.data.body_com_pos_w[:, :, :2] - env.scene.env_origins[:, :2].unsqueeze(1) #shape: (num_envs, num_bodies)
    CoM = torch.sum(body_coms * masses.unsqueeze(-1), dim=1) / torch.sum(masses.unsqueeze(-1), dim=1)
    # CoM = (asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2] ).clip(min=-0.95)


    # feet_pos = asset.data.body_pos_w[:,asset_cfg.body_ids, :2] - env.scene.env_origins[:,:2].unsqueeze(1)
    # feet_CoM_dist = torch.linalg.norm(feet_pos - CoM.unsqueeze(1), dim=-1).clip(min=0.249)  # (N,F)
    # feet_com_farther = ((feet_CoM_dist - min_feet_com) > -0.001).all(dim=-1)
    # env.command_manager.get_term('climb_command').update_min_feet_com(feet_com_dist=feet_CoM_dist, on_box=on_box_for_angle)
    

    feet_pos = (asset.data.body_pos_w[:,asset_cfg.body_ids, :2].mean(dim=1) - env.scene.env_origins[:,:2])
    # feet_pos = (asset.data.body_pos_w[:,asset_cfg.body_ids, :3].mean(dim=1) - env.scene.env_origins[:,:3])
    # CoM = torch.cat([CoM, 0.036*torch.ones_like(feet_pos[:, 2:3])], dim=1)

    feet_CoM_dist = torch.linalg.norm((CoM - feet_pos),dim=1).clip(min = 0.099)  

    feet_com_farther = feet_CoM_dist - min_feet_avg_com
    # print('feet com farther:', feet_com_farther)
    # print('feet com dist:', feet_CoM_dist, 'min feet avg com:', min_feet_avg_com)
    feet_com_farther = torch.logical_or(feet_com_farther>0, torch.abs(feet_com_farther)<0.001)
    # env.command_manager.get_term('climb_command').update_min_feet_avg_com(feet_com_dist=feet_CoM_dist, on_box=(torch.ones_like(feet_CoM_dist))>0)
    env.command_manager.get_term('climb_command').update_min_feet_avg_com(feet_com_dist=feet_CoM_dist, on_box=env.episode_length_buf>15)
    feet_CoM_dist = feet_CoM_dist.unsqueeze(-1)
    #input('Press Enter to continue...')  # Debugging line to pause execution
    # print('min feet com:', min_feet_avg_com)
    # print('feet com dist:', feet_CoM_dist)
    
    shoulder_height = asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].mean(dim=-1) 
    # success = (shoulder_height > 0.5) | (feet_CoM_dist.max(dim=1)[0]< 0.15)
    success =  (feet_CoM_dist.max(dim=1)[0]< 0.1) | (shoulder_height > 0.9)
    feet_com_farther = feet_com_farther.logical_and(~success)

    penalty = feet_com_farther
    
    return penalty.logical_and(env.episode_length_buf>15)


def wait_penalty_standup(env,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for waiting."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_l2=torch.square(asset.data.root_lin_vel_w).sum(dim=1)
    shoulder_height = asset.data.body_pos_w[:,  asset_cfg.body_ids, 2].mean(dim=-1)
    return  (vel_l2 < 0.01) & (shoulder_height < 0.9) & (env.episode_length_buf > 15)


def hip_joint_deviation_standup(
    env: ManagerBasedRLEnv, hip_yaw_cfg, hip_roll_cfg, hip_pitch_cfg
) -> torch.Tensor:
    """Penalize hip joint deviation from zero."""
    # extract the used quantities (to enable type-hinting)
    asset=env.scene[hip_yaw_cfg.name]
    # compute out of limits constraints
    hip_yaw_joint = (asset.data.joint_pos[:, hip_yaw_cfg.joint_ids])
    hip_yaw_joint_abs = hip_yaw_joint.abs()
    yaw_penalty = (torch.max(hip_yaw_joint_abs,dim=-1)[0]>1.25) | (torch.min(hip_yaw_joint_abs,dim=-1)[0]>0.9)
    # yaw_penalty = yaw_penalty | (hip_yaw_joint[:,0] < -1.15) | (hip_yaw_joint[:,1] > 1.15)
    # print('yaw penalty:', yaw_penalty)

    hip_roll_joint = torch.abs(asset.data.joint_pos[:, hip_roll_cfg.joint_ids])
    roll_penalty = (torch.max(hip_roll_joint,dim=-1)[0]>1.4) | (torch.min(hip_roll_joint,dim=-1)[0]>0.9)

    hip_pitch_joint = asset.data.joint_pos[:, hip_pitch_cfg.joint_ids]
    pitch_penalty = (hip_pitch_joint > 0.6).any(dim=-1)

    # input('input enter')
    # print('hip pitch joint:', hip_pitch_joint)
    # print('hip roll joint:', asset.data.joint_pos[:, hip_roll_cfg.joint_ids])
    # print('hip yaw joint:', asset.data.joint_pos[:, hip_yaw_cfg.joint_ids])

    return yaw_penalty | roll_penalty | pitch_penalty


def waist_joint_deviation_up(
    env: ManagerBasedRLEnv, waist_yaw_cfg, waist_pitch_cfg, waist_roll_cfg) -> torch.Tensor:
    """Penalize waist joint deviation from zero."""
    # extract the used quantities (to enable type-hinting)
    asset=env.scene[waist_yaw_cfg.name]
    # compute out of limits constraints
    waist_yaw_joint = torch.abs(asset.data.joint_pos[:, waist_yaw_cfg.joint_ids])
    waist_pitch_joint = torch.abs(asset.data.joint_pos[:, waist_pitch_cfg.joint_ids])
    waist_roll_joint = torch.abs(asset.data.joint_pos[:, waist_roll_cfg.joint_ids])

    return (waist_yaw_joint > 1.4).squeeze(-1) | (waist_pitch_joint > 0.46).squeeze(-1) | (waist_roll_joint > 0.46).squeeze(-1)


def shoulder_joint_deviation(
    env: ManagerBasedRLEnv, shoulder_pitch_cfg) -> torch.Tensor:
    """Penalize shoulder joint deviation from zero."""
    # extract the used quantities (to enable type-hinting)
    asset=env.scene[shoulder_pitch_cfg.name]
    # compute out of limits constraints
    shoulder_joint = asset.data.joint_pos[:, shoulder_pitch_cfg.joint_ids]
    # print('shoulder joint:', shoulder_joint)
    return (shoulder_joint > 0.5).any(dim=-1)


def foot_dist_l2_standup(
    env: ManagerBasedRLEnv, hip_pitch_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the distance between the feet."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    # compute out of limits constraints
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_vec = foot_pos[:,1,:2] - foot_pos[:,0,:2]
    dist_sq = torch.sum(feet_vec * feet_vec, dim=-1,)

    # hip_pos = asset.data.body_pos_w[:, hip_pitch_cfg.body_ids, :]
    # hip_vec = hip_pos[:,1,:2] - hip_pos[:,0,:2]
    # hip_len_sq = torch.sum(hip_vec * hip_vec, dim=-1) 
    # dot_feet_hip = torch.sum(feet_vec * hip_vec, dim=-1)
    # projected_width_sq = (dot_feet_hip ** 2) / hip_len_sq
    # print('hip len:', hip_len_sq.sqrt())
    # print('projected width:', projected_width_sq.sqrt(),'foot dist:', dist_sq.sqrt())
    # print('foot dist:', dist_sq.sqrt())

    reward_large = (dist_sq >= 0.64)#|(projected_width_sq.squeeze(-1) > 0.49)).float() 
    reward_small = (dist_sq < 0.0144).float()
    # print('foot dist:', torch.sqrt(dist))
    return reward_large + reward_small


def standing_lin_vel_up(
    env: ManagerBasedRLEnv, shoulder_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the linear velocity of the base in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    lin_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    reward =  torch.sum(torch.square(lin_vel), dim=-1).squeeze(-1)  # L2 penalty on the linear velocity
    reward =  torch.exp(-5 * reward)  # Exponential kernel to penalize deviations from the desired height
    ending = env.episode_length_buf>250
    shoulder_height = asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].mean(dim=-1) - env.scene.env_origins[:, 2]
    ending = ending & (shoulder_height > 0.9)
    return reward.masked_fill_(~ending, 0.0)


def standing_ang_vel_up(
    env: ManagerBasedRLEnv, shoulder_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the linear velocity of the base in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    reward =  torch.sum(torch.square(ang_vel), dim=-1).squeeze(-1)  # L2 penalty on the linear velocity
    reward =  torch.exp(-2 * reward)  
    ending = env.episode_length_buf>250
    shoulder_height = asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].mean(dim=-1) - env.scene.env_origins[:, 2]
    ending = ending & (shoulder_height > 0.9)
    return reward.masked_fill_(~ending, 0.0)


def standing_height_up(
    env: ManagerBasedRLEnv,  desired_height: float ,asset_cfg
) -> torch.Tensor:
    """Penalize the height of the base in the world frame."""
    # extract the used quantities (to enable type-hinting)

    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    #height = (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).clip(max=desired_height)  # relative to the environment origin
    box_height = env.scene.env_origins[:,2]
    height = (asset.data.body_pos_w[:, asset_cfg.body_ids, 2].mean(dim=-1)-box_height).clip(max=desired_height)
    #height = (asset.data.root_pos_w[:, 2])
    # print('height:',height)
    reward = torch.square(height - desired_height)
    reward=torch.exp(-20*reward)  # Exponential kernel to penalize deviations from the desired height
    ending = env.episode_length_buf>250
    ending = ending & (height > 0.9)
    return reward.masked_fill_(~ending, 0.0)


def standing_feet_orientation_up(
    env: ManagerBasedRLEnv ,asset_cfg, shoulder_cfg,
) -> torch.Tensor:
    """Penalize the height of the base in the world frame."""
    # extract the used quantities (to enable type-hinting)

    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    #height = (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).clip(max=desired_height)  # relative to the environment origin
    box_height = env.scene.env_origins[:,2]
    height = (asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].mean(dim=-1)-box_height)


    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)  # (num_envs, num_bodies, 4) -> (num_envs, 4)
    proj_grav = math_utils.quat_rotate_inverse(quat, asset.data.GRAVITY_VEC_W.unsqueeze(1))
    # print('proj grav:', proj_grav[:, :,2])
    cosine = -proj_grav[:, :,2] 
    reward = cosine
    reward = (reward).sum(dim=1)

    ending = env.episode_length_buf>250
    ending = ending & (height > 0.9)
    return reward.masked_fill_(~ending, 0.0)


def both_feet_in_air(env: ManagerBasedRLEnv,feet_sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    contact_sensor: ContactGroundSensorZ = env.scene.sensors[feet_sensor_cfg.name]
    
    
    feet_air = (contact_sensor.data.net_forces_w_history[:, :,feet_sensor_cfg.body_ids, 2].min(dim=1)[0] < 100).all(dim=1)
    all_forces = contact_sensor._data.force_matrix_w[:, :,0, 2]

    num_bodies = all_forces.shape[1]
    non_feet_mask = torch.ones(num_bodies, dtype=torch.bool, device=all_forces.device)
    non_feet_mask[feet_sensor_cfg.body_ids] = False
    others_no_contact = (all_forces[:, non_feet_mask].max(dim=1)[0] < 0.1)


    return feet_air & others_no_contact & (env.episode_length_buf>15)


def lin_vel_z(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for upward linear velocity of the base."""
    asset = env.scene[asset_cfg.name]
    lin_vel_z = asset.data.root_com_lin_vel_w[:, 2]
    return torch.square(lin_vel_z)


def track_lin_vel_xy_ground_tangent_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1) 
    def project_tangent(v, n):
        return v - (v*n).sum(dim=1, keepdim=True) * n
    # print('quat_w:',quat_w.shape)
    # print('upward vec:',torch.tensor([0.0, 0.0, 1.0], device=quat_w.device).expand(quat_w.shape[0], 3))
    upward_vec = torch.tensor([0.0, 0.0, 1.0], device=quat_w.device).expand(quat_w.shape[0], 3)
    forward_vec = quat_rotate(quat_w, upward_vec)
    ground_n_vec = upward_vec
    vel_tangent = project_tangent(asset.data.body_vel_w[:, asset_cfg.body_ids, :3].squeeze(1), ground_n_vec)
    forward_tangent_vec = torch.nn.functional.normalize( project_tangent(forward_vec, ground_n_vec), dim=1)
    side_tangent_vec = torch.cross(ground_n_vec, forward_tangent_vec, dim=1)  
    # print('vel tangent:',vel_tangent)
    # print('forward vec tangent:',forward_tangent_vec)
    # print('side vec:',side_tangent_vec)
    vel_forward = (vel_tangent * forward_tangent_vec).sum(dim=1)      # forward
    vel_side = (vel_tangent * side_tangent_vec).sum(dim=1)
    # print('vel forward:',vel_forward)
    # print('vel cmd forward:',env.command_manager.get_command(command_name)[:, 0])
    # print('vel side:',vel_side)

    # forward_err = torch.where(env.command_manager.get_command(command_name)[:, 0].abs() > 0.1,
    #                             torch.square((env.command_manager.get_command(command_name)[:, 0] - vel_forward)/env.command_manager.get_command(command_name)[:, 0]),
    #                             torch.square((0 - vel_forward)))
    
    # side_err = torch.where(env.command_manager.get_command(command_name)[:, 1].abs() > 0.1,
    #                             torch.square((env.command_manager.get_command(command_name)[:, 1] - vel_side)/env.command_manager.get_command(command_name)[:, 1]),
    #                             torch.square((0 - vel_side)))

    forward_err = torch.where(env.command_manager.get_command(command_name)[:, 0].abs() > 0.1,
                                torch.square(env.command_manager.get_command(command_name)[:, 0] - vel_forward),torch.square((0 - vel_forward)))
    
    side_err = torch.where(env.command_manager.get_command(command_name)[:, 1].abs() > 0.1,
                                torch.square(env.command_manager.get_command(command_name)[:, 1] - vel_side),torch.square((0 - vel_side)))
    
    lin_vel_error = forward_err + side_err

    #print('lin vel error:',lin_vel_error)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def ang_vel_xy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # Penalize xy axes base angular velocity
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_com_ang_vel_w[:, :2]), dim=1)


def base_height_range(env, min_height: float, max_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for base height within a target height range."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:,2]
    # use squared distance to the closest bound
    below_min = torch.square(torch.clamp(min_height - height, min=0.0))
    above_max = torch.square(torch.clamp(height - max_height, min=0.0))
    # print('height:',height)
    return below_min + above_max


def project_tangent(v, n):
    # v, n: (N, 3); n should be unit-length
    return v - (v * n).sum(dim=1, keepdim=True) * n
