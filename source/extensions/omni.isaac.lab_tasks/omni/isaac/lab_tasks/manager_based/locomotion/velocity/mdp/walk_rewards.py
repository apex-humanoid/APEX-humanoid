# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Walking task functions."""
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

def joint_vel_with_action(env: ManagerBasedRLEnv, action_term_name,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_with_action = env.action_manager.get_term(action_term_name)._joint_ids
    return torch.sum(torch.square(asset.data.joint_vel[:, joint_with_action]), dim=1)


def joint_acc_with_action(env: ManagerBasedRLEnv, action_term_name,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_with_action = env.action_manager.get_term(action_term_name)._joint_ids
    return torch.sum(torch.square(asset.data.joint_acc[:, joint_with_action]), dim=1)


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)


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

