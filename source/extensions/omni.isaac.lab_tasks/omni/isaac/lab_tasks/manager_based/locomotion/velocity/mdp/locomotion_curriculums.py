# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion task functions."""
from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter
import carb
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import ContactGroundSensorZ
from omni.isaac.lab.managers import CurriculumTermCfg, ManagerTermBase
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def terrain_levels_height_up(
    env: ManagerBasedRLEnv, env_ids: Sequence[int],shoulder_cfg,update_prob:float=0.8,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the of the robot to the target pos.

    This term is used to increase the difficulty of the terrain when the robot walks close enough to the target and decrease the
    difficulty ...

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    box_height = env.scene.env_origins[:,2]
    
    # robots that walked far enough progress to harder terrains
    lowest_height =  asset.data.body_pos_w[env_ids, :, 2].min(dim=1)[0] - box_height[env_ids]
    back_x = asset.data.body_pos_w[env_ids,:,0].min(dim=1)[0]-env.scene.env_origins[env_ids,0]
    move_up = (lowest_height > 0.02) & (back_x > -0.95)
    # shoulder_height = asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].mean(dim=-1)
    # move_up = (shoulder_height[env_ids]-box_height[env_ids])> 0.8
    # move_up = (asset.data.root_link_pos_w[env_ids, 2]-env.scene.env_origins[env_ids, 2])>0.6
    # robots that walked less than half of their required distance go to simpler terrains
    #move_down = (asset.data.root_link_pos_w[env_ids, 2]) < 0.55
    move_down = env.termination_manager.terminated[env_ids]
    move_down *= ~move_up
    # update terrain levels
    #terrain.update_env_origins(env_ids, move_up, move_down)
    terrain.update_env_origins_prob(env_ids, move_up, move_down, update_prob)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_height_down(
    env: ManagerBasedRLEnv, env_ids: Sequence[int],sensor_cfg,desired_height=0.78,update_prob:float=0.8,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the of the robot to the target pos.

    This term is used to increase the difficulty of the terrain when the robot walks close enough to the target and decrease the
    difficulty ...

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    contact_sensor: ContactGroundSensorZ = env.scene.sensors[sensor_cfg.name]
    
    feet_near_grd = asset.data.body_pos_w[:,asset_cfg.body_ids, 2].max(dim=-1)[0] < 0.05
    # net_contact_forces = contact_sensor.data.net_forces_w[env_ids]
    # feet_contact = (torch.norm(net_contact_forces[:,  sensor_cfg.body_ids], dim=-1)>0.1).all(dim=-1)
    # feet_on_grd = torch.logical_and(feet_near_grd[env_ids], feet_contact)
    stand = asset.data.root_pos_w[env_ids, 2]> desired_height-0.1
    #print('body height:',asset.data.root_pos_w[:, 2])
    # robots that walked far enough progress to harder terrains
    #move_up = lowest_height > 0.02
    move_up = torch.logical_and(feet_near_grd[env_ids], stand)
    # robots that walked less than half of their required distance go to simpler terrains
    # move_down = asset.data.root_link_pos_w[env_ids, 2] < 0.45
    move_down = torch.logical_or(asset.data.root_link_pos_w[env_ids, 0]-env.scene.env_origins[env_ids,0] > -0.8, asset.data.root_link_pos_w[env_ids, 2] < 0.45) 
    move_down *= ~move_up
    # update terrain levels
    #terrain.update_env_origins(env_ids, move_up, move_down)
    terrain.update_env_origins_prob(env_ids, move_up, move_down, update_prob)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
