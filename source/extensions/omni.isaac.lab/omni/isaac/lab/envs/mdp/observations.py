# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from omni.isaac.lab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera,ContactGroundSensorZ

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv




"""
Root state.
"""


def base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)


def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_com_lin_vel_b


def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # input("Input Enter")
    return asset.data.root_com_ang_vel_b

def base_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    root_id=asset.find_bodies("torso_link")[0]
    return torch.sum(torch.square(asset.data.body_lin_acc_w[:, root_id,:]), dim=-1)


def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def root_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    return asset.data.root_link_pos_w - env.scene.env_origins




def root_pos_target(env: ManagerBasedEnv, command_name:str,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return (env.command_manager.get_command(command_name)
                            +env.scene.env_origins-asset.data.root_pos_w)

def root_quat_w(
    env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_link_quat_w
    # make the quaternion real-part positive if configured

    return math_utils.quat_unique(quat) if make_quat_unique else quat


def base_yaw_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw and roll of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_link_quat_w)
    # normalize angle to [-pi, pi]
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return yaw.unsqueeze(-1)

def root_euler_w(env: ManagerBasedEnv, make_quat_unique: bool = True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_link_quat_w
    # make the quaternion real-part positive if configured
    quat = math_utils.quat_unique(quat) if make_quat_unique else quat
    r,p,y=math_utils.euler_xyz_from_quat(quat)
    
    return torch.cat((r.unsqueeze(-1), p.unsqueeze(-1), y.unsqueeze(-1)), dim=-1)


def root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_com_lin_vel_w


def root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    return asset.data.root_com_ang_vel_w

def height_fail(
        env: ManagerBasedRLEnv, minimum_height: float, maximum_height: float,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.logical_or(asset.data.root_link_pos_w[:, 2] < minimum_height, asset.data.root_link_pos_w[:, 2] > maximum_height)

"""
Torso
"""

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

"""
Joint state.
"""


def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # input("Input Enter")
    # print('joint names:', asset.joint_names)
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

def joint_pos_rel_realg1(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    isaac_to_g1_leg = [0,2,4,6,8,10,1,3,5,7,9,11]
    pos_rel=asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # print('joint names:', asset.joint_names)
    # print('joint ids:', asset_cfg.joint_ids)
    # print('joint pos rel realg1:', pos_rel)
    # print('joint pos rel realg1 reordered:', pos_rel[..., isaac_to_g1_leg].contiguous())
    # return torch.zeros_like(pos_rel)
    return pos_rel[..., isaac_to_g1_leg].contiguous()  



def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )


def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


def joint_vel_rel_realg1(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    vel_rel = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    isaac_to_g1_leg = [0,2,4,6,8,10,1,3,5,7,9,11]
    # return torch.zeros_like(vel_rel)
    return vel_rel[..., isaac_to_g1_leg].contiguous()


"""
Sensors.
"""


# def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
#     """Height scan from the given sensor w.r.t. the sensor's frame.

#     The provided offset (Defaults to 0.5) is subtracted from the returned values.
#     """
#     # extract the used quantities (to enable type-hinting)
#     sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
#     # height scan: height = sensor_height - hit_point_z - offset
#     return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

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

def height_scan_nodrift(env: ManagerBasedEnv,  sensor_cfg: SceneEntityCfg, asset_cfg=None, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.

    Robustness: If a ray does not hit within the sensor max distance, the underlying ray-caster fills the
    hit position with +inf. That would result in -inf heights. Here we replace non-finite hits with a value that
    saturates the height to the sensor's max range (max_distance - offset).
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    pos_z = sensor.data.pos_w_nodrift[:, 2].unsqueeze(1)
    hit_z = sensor.data.ray_hits_w_nodrift[..., 2]
   
    if not torch.isfinite(hit_z).all():
        fallback_hit_z = pos_z - sensor.cfg.max_distance
        hit_z = torch.where(torch.isfinite(hit_z), hit_z, fallback_hit_z)
    height = pos_z - hit_z - offset
    # Final guard: ensure finite tensor (should be already, but guard anyway)
    if not torch.isfinite(height).all():
        height = torch.nan_to_num(height, posinf=sensor.cfg.max_distance - offset, neginf=-(sensor.cfg.max_distance))
    height = torch.clamp(height, min=-1, max=2)
    return height




def body_incoming_wrench(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    link_incoming_forces = asset.root_physx_view.get_link_incoming_joint_force()[:, asset_cfg.body_ids]
    return link_incoming_forces.view(env.num_envs, -1)


def imu_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation in the simulation world frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        Orientation in the world frame in (w, x, y, z) quaternion form. Shape is (num_envs, 4).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the orientation quaternion
    return asset.data.quat_w


def imu_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor angular velocity w.r.t. environment origin expressed in the sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The angular velocity (rad/s) in the sensor frame. Shape is (num_envs, 3).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the angular velocity
    return asset.data.ang_vel_b


def imu_lin_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor linear acceleration w.r.t. the environment origin expressed in sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The linear acceleration (m/s^2) in the sensor frame. Shape is (num_envs, 3).
    """
    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.lin_acc_b

def box_height(env: ManagerBasedEnv) -> torch.Tensor:
    # climb_command = env.command_manager.get_command('climb_command')
    # return torch.where(climb_command>0, 0-env.scene.env_origins[:, 2], torch.zeros(env.num_envs,1, device=env.device))
    return (env.scene.env_origins[:, 2]).unsqueeze(-1)
    #return env.command_manager.get_term('climb_command').box_height.unsqueeze(-1)# - env.scene.env_origins[:, 2].unsqueeze(-1)


def climb_command(env: ManagerBasedEnv) -> torch.Tensor:
    # print('climb command:', env.command_manager.get_command('climb_command'))
    cmd = env.command_manager.get_command('climb_command').unsqueeze(-1)
    # return torch.ones_like(cmd) 
    if hasattr(env, "episode_length_buf"):
        # print('time:', env.episode_length_buf* env.step_dt)
        #climb_command = env.command_manager.get_command('climb_command')
        #return torch.where((climb_command>0).unsqueeze(-1),torch.zeros(env.num_envs, 1,device=env.device),(env.episode_length_buf * env.step_dt).unsqueeze(-1))
        time = (env.episode_length_buf * env.step_dt).unsqueeze(-1)
    else:
        time = torch.zeros(env.num_envs,1, device=env.device)
    return torch.logical_or(cmd>0, time>1.0)

def body_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mass of the body links of an articulation in the simulation world frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an Articulation. Defaults to SceneEntityCfg("robot").

    Returns:
        Mass of the body links in kg. Shape is (num_envs, num_bodies).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()

    return masses[:, asset_cfg.body_ids]

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
    
def body_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    # compute the violation    
    #print('body contact:',torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]>0.1)
    contact = torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1)#>0.1
    #print('contact:', torch.where(contact, torch.ones_like(contact), torch.zeros_like(contact)))
    return contact

def max_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation    
    forces=torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    max_contact = torch.max(forces,dim=1)[0]
    return max_contact

def bodies_contact_state(env: ManagerBasedRLEnv, L_Leg_cfg,R_Leg_cfg, L_Arm_cfg,R_Arm_cfg, torso_cfg,head_cfg
                         ) -> torch.Tensor:
    """Get the contact states of the six bady groups. At least one body in a group in contact, the group is considered in contact."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[L_Leg_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    forces = torch.norm(net_contact_forces[:, :,:], dim=-1)
    L_Leg_contact = torch.max(forces[:, L_Leg_cfg.body_ids], dim=1)[0]>1
    R_Leg_contact = torch.max(forces[:, R_Leg_cfg.body_ids], dim=1)[0]>1
    L_Arm_contact = torch.max(forces[:, L_Arm_cfg.body_ids], dim=1)[0]>1
    R_Arm_contact = torch.max(forces[:, R_Arm_cfg.body_ids], dim=1)[0]>1
    torso_contact = torch.max(forces[:, torso_cfg.body_ids], dim=1)[0]>1
    head_contact = torch.max(forces[:, head_cfg.body_ids], dim=1)[0]>1
    # print('L_Leg_contact:', torch.max(forces[:, L_Leg_cfg.body_ids], dim=1)[0])
    print('R_Leg_contact:', torch.max(forces[:, R_Leg_cfg.body_ids], dim=1)[0])
    # print('L_Arm_contact:', torch.max(forces[:, L_Arm_cfg.body_ids], dim=1)[0])
    # print('R_Arm_contact:', torch.max(forces[:, R_Arm_cfg.body_ids], dim=1)[0])
    contact_states = torch.stack((L_Leg_contact,R_Leg_contact,L_Arm_contact,R_Arm_contact,torso_contact,head_contact),dim=1) 
    return contact_states.float()       

def body_orientation(env,asset_cfg)-> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    

    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)  # (num_envs, num_bodies, 4) -> (num_envs, 4)
    proj_grav = math_utils.quat_rotate_inverse(quat, asset.data.GRAVITY_VEC_W.unsqueeze(1))
    #print('proj grav:', proj_grav)
    cosine = -proj_grav[:, :,2] 
    return cosine

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

def progress_memory_down(env: ManagerBasedEnv) -> torch.Tensor:
   
    min_avg_height = env.command_manager.get_term('climb_command').min_avg_height.unsqueeze(-1)
    min_com_x = env.command_manager.get_term('climb_command').min_com_x.unsqueeze(-1)

    progress = torch.cat([min_avg_height, min_com_x], dim=-1)
    return progress

def progress_memory_standup(env: ManagerBasedEnv) -> torch.Tensor:
    min_feet_angle=env.command_manager.get_term('climb_command').min_feet_angle
    min_torso_angle = env.command_manager.get_term('climb_command').min_torso_angle.unsqueeze(-1)
    min_upper_force = env.command_manager.get_term('climb_command').min_upper_force.unsqueeze(-1)
    max_shoulder_height = env.command_manager.get_term('climb_command').max_avg_whole_height.unsqueeze(-1)
    min_feet_avg_com=env.command_manager.get_term('climb_command').min_feet_avg_com.unsqueeze(-1)


    # progress = torch.cat([max_avg_height, max_com_x, min_feet_angle, min_feet_box, min_torso_angle, min_upper_force, max_shoulder_height, min_feet_avg_com], dim=1)
    progress = torch.cat([min_feet_angle, min_torso_angle, min_upper_force, max_shoulder_height,min_feet_avg_com], dim=1)
    return progress


def min_height_over_minimum(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    box_height = env.scene.env_origins[:, 2]
    #return (asset.data.root_link_pos_w[:, 2]-env.scene.env_origins[:,2]) < minimum_height
    
    return asset.data.body_pos_w[:, :, 2].min(dim=1)[0] >box_height - 0.1

def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    return images.clone()


class image_features(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This term uses models from the model zoo in PyTorch and extracts features from the images.

    It calls the :func:`image` function to get the images and then processes them using the model zoo.

    A user can provide their own model zoo configuration to use different models for feature extraction.
    The model zoo configuration should be a dictionary that maps different model names to a dictionary
    that defines the model, preprocess and inference functions. The dictionary should have the following
    entries:

    - "model": A callable that returns the model when invoked without arguments.
    - "reset": A callable that resets the model. This is useful when the model has a state that needs to be reset.
    - "inference": A callable that, when given the model and the images, returns the extracted features.

    If the model zoo configuration is not provided, the default model zoo configurations are used. The default
    model zoo configurations include the models from Theia :cite:`shang2024theia` and ResNet :cite:`he2016deep`.
    These models are loaded from `Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`_ and
    `PyTorch torchvision <https://pytorch.org/vision/stable/models.html>`_ respectively.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The sensor data type. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        model_zoo_cfg: A user-defined dictionary that maps different model names to their respective configurations.
            Defaults to None. If None, the default model zoo configurations are used.
        model_name: The name of the model to use for inference. Defaults to "resnet18".
        model_device: The device to store and infer the model on. This is useful when offloading the computation
            from the environment simulation device. Defaults to the environment device.
        inference_kwargs: Additional keyword arguments to pass to the inference function. Defaults to None,
            which means no additional arguments are passed.

    Returns:
        The extracted features tensor. Shape is (num_envs, feature_dim).

    Raises:
        ValueError: When the model name is not found in the provided model zoo configuration.
        ValueError: When the model name is not found in the default model zoo configuration.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # extract parameters from the configuration
        self.model_zoo_cfg: dict = cfg.params.get("model_zoo_cfg")  # type: ignore
        self.model_name: str = cfg.params.get("model_name", "resnet18")  # type: ignore
        self.model_device: str = cfg.params.get("model_device", env.device)  # type: ignore

        # List of Theia models - These are configured through `_prepare_theia_transformer_model` function
        default_theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]
        # List of ResNet models - These are configured through `_prepare_resnet_model` function
        default_resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]

        # Check if model name is specified in the model zoo configuration
        if self.model_zoo_cfg is not None and self.model_name not in self.model_zoo_cfg:
            raise ValueError(
                f"Model name '{self.model_name}' not found in the provided model zoo configuration."
                " Please add the model to the model zoo configuration or use a different model name."
                f" Available models in the provided list: {list(self.model_zoo_cfg.keys())}."
                "\nHint: If you want to use a default model, consider using one of the following models:"
                f" {default_theia_models + default_resnet_models}. In this case, you can remove the"
                " 'model_zoo_cfg' parameter from the observation term configuration."
            )
        if self.model_zoo_cfg is None:
            if self.model_name in default_theia_models:
                model_config = self._prepare_theia_transformer_model(self.model_name, self.model_device)
            elif self.model_name in default_resnet_models:
                model_config = self._prepare_resnet_model(self.model_name, self.model_device)
            else:
                raise ValueError(
                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
                    f" Available models: {default_theia_models + default_resnet_models}."
                )
        else:
            model_config = self.model_zoo_cfg[self.model_name]

        # Retrieve the model, preprocess and inference functions
        self._model = model_config["model"]()
        self._reset_fn = model_config.get("reset")
        self._inference_fn = model_config["inference"]

    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the model if a reset function is provided
        # this might be useful when the model has a state that needs to be reset
        # for example: video transformers
        if self._reset_fn is not None:
            self._reset_fn(self._model, env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
    ) -> torch.Tensor:
        # obtain the images from the sensor
        image_data = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=False,  # we pre-process based on model
        )
        # store the device of the image
        image_device = image_data.device
        # forward the images through the model
        features = self._inference_fn(self._model, image_data, **(inference_kwargs or {}))

        # move the features back to the image device
        return features.detach().to(image_device)

    """
    Helper functions.
    """

    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the Theia transformer model for inference.

        Args:
            model_name: The name of the Theia transformer model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from transformers import AutoModel

        def _load_model() -> torch.nn.Module:
            """Load the Theia transformer model."""
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the Theia transformer model.

            Args:
                model: The Theia transformer model.
                images: The preprocessed image tensor. Shape is (num_envs, height, width, channel).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # Move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # Normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # Taken from Transformers; inference converted to be GPU only
            features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
            return features.last_hidden_state[:, 1:]

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}

    def _prepare_resnet_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the ResNet model for inference.

        Args:
            model_name: The name of the ResNet model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            """Load the ResNet model."""
            # map the model name to the weights
            resnet_weights = {
                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
                "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
            }

            # load the model
            model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the ResNet model.

            Args:
                model: The ResNet model.
                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # forward the image through the model
            return model(image_proc)

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}


"""
Actions.
"""


def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions
    
def last_action_realg1(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    isaac_to_g1_leg = [0,2,4,6,8,10,1,3,5,7,9,11]
    if action_name is None:
        # return torch.zeros_like(env.action_manager.action)
        return env.action_manager.action[..., isaac_to_g1_leg].contiguous()
    else:
        # return torch.zeros_like(env.action_manager.action)
        return env.action_manager.get_term(action_name).raw_actions[..., isaac_to_g1_leg].contiguous()


def last_processed_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last processed action to the environment.

    The name of the action term for which the action is required. If None, the
    entire processed action tensor is returned.
    """
   
    return 0.25*env.action_manager.get_term(action_name).processed_actions

# Remove the 0.25
def real_last_processed_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    return env.action_manager.get_term(action_name).processed_actions

def walk_last_processed_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    walk_joint_ids = [0,1,3,4,6,7,9,10,13,14, 17,18]
    return env.action_manager.get_term(action_name).processed_actions[:,walk_joint_ids]
"""
Commands.
"""


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)

def velocity_commands(env: ManagerBasedRLEnv, command_name: str, lin_scale, ang_scale,) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    cmd = env.command_manager.get_command(command_name)
    # cmd[:,:2]*=lin_scale
    # cmd[:,2]*=ang_scale
    return torch.cat((cmd[:,:2]*lin_scale,cmd[:,2].unsqueeze(-1)*ang_scale),dim=1)

def target_pos_root_frame(env: ManagerBasedRLEnv, command_name: str,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The target position in the root frame of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    target_pos_w=env.command_manager.get_command(command_name)+env.scene.env_origins
    target_pos_w[:,2]=0
    return target_pos_w-asset.data.root_pos_w

def target_pos_w(env: ManagerBasedRLEnv, command_name: str,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    target_pos_w=env.command_manager.get_command(command_name)
    target_pos_w[:,2]=0
    return target_pos_w

def tri_phase(env: ManagerBasedRLEnv, period) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        time = env.episode_length_buf * env.step_dt
    else:
        time = torch.zeros(env.num_envs, device=env.device)
    # print('time:', time)
    phase = (time % period) / period
    sin_phase = torch.sin(2 * torch.pi * phase ).unsqueeze(1)
    cos_phase = torch.cos(2 * torch.pi * phase ).unsqueeze(1)
    return torch.cat((sin_phase, cos_phase), dim=1)

def teacher_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    asset: RigidObject = env.scene["robot"]
    id = torch.where((asset.data.root_link_pos_w[:,0] - env.scene.env_origins[:,0]>-1.35).unsqueeze(-1), torch.ones(env.num_envs,1, device=env.device), torch.zeros(env.num_envs,1, device=env.device))
    # print('teacher id:', id.shape)
    return id
    # return torch.ones(env.num_envs,1, device=env.device)