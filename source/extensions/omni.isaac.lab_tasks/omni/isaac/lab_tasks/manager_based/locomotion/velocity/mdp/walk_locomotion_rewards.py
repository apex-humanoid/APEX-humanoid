# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Walking task functions."""
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorZ, ContactGroundSensorZ
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat, _R_from_quat_batch, euler_xyz_from_quat, quat_rotate
from omni.isaac.lab.managers.manager_base import ManagerTermBase
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.sensors import RayCaster
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_link_quat_w), asset.data.root_com_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    #print('lin vel error:',lin_vel_error)
    return torch.exp(-lin_vel_error / std**2)


def orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_height(env, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for base height close to a target height."""
    asset = env.scene[asset_cfg.name]
    height_error = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:,2] - target_height
    return torch.square(height_error)

