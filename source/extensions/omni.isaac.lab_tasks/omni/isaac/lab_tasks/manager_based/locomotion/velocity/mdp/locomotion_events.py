# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion task functions."""
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter
import omni.isaac.lab.utils.math as math_utils
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

def reset_root_state_on_ground_uniform_with_terrain_width(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    platform_width_range: tuple[float, float] = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity with dynamic y-range based on terrain width.
    
    This function randomizes the root position and velocity of the asset, with the y-position range
    dynamically adjusted based on the current box width determined by terrain difficulty level.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation. The y-range is dynamically calculated based on platform width.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. The y-range in pose_range will be overridden by the calculated box width.
    
    Args:
        env: The environment instance.
        env_ids: The environment IDs to reset.
        pose_range: The range of the pose. It should have the keys: "x", "y", "z", "roll", "pitch", "yaw".
                   Note: The "y" range will be overridden by the calculated box width.
        velocity_range: The range of the root velocity. It should have the keys: "x", "y", "z",
                       "roll", "pitch", "yaw".
        platform_width_range: Tuple of (min_width, max_width) that defines how box width changes
                             with difficulty. Default is (3.0, 1.0) for 3m at easy to 1m at hard.
        asset_cfg: The asset configuration.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    
    # Calculate dynamic y-range based on terrain level
    if platform_width_range is not None:
        terrain_levels = terrain.terrain_levels[env_ids]
        max_level = terrain.max_terrain_level - 1  # num_rows - 1
        
        # Calculate difficulty for each environment (0.0 to 1.0)
        difficulty = terrain_levels.float() / max(max_level, 1)
        
        # Calculate current box width for each environment
        # width = min_width + difficulty * (max_width - min_width)
        min_width, max_width = platform_width_range
        box_widths = min_width + difficulty * (max_width - min_width)
        
        # Use half the box width as the y-range (so robot spawns within the box)
        # Add a small safety margin (0.1m) to ensure robot is well within bounds
        safety_margin = 0.
        y_half_range = (box_widths / 2.0) - safety_margin
        
        # Create dynamic y_range for each environment: (-y_half_range, +y_half_range)
        dynamic_y_range = torch.stack([-y_half_range, y_half_range], dim=1)  # (num_envs, 2)
    else:
        # Fallback: if no terrain curriculum, use the original y-range
        y_range = pose_range.get("y", (0.0, 0.0))
        dynamic_y_range = torch.tensor([y_range] * len(env_ids), device=env.device)
    
    # poses - sample x, z, roll, pitch, yaw from ranges
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples_xz = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 5), device=asset.device)
    
    # Sample y separately using dynamic range per environment

    rand_y = torch.rand(len(env_ids), 1, device=asset.device) * (
    dynamic_y_range[:, 1:2] - dynamic_y_range[:, 0:1]) + dynamic_y_range[:, 0:1]
    # Combine samples: [x, y, z, roll, pitch, yaw]
    rand_samples = torch.cat([
        rand_samples_xz[:, 0:1],  # x
        rand_y,                    # y (dynamic)
        rand_samples_xz[:, 1:4],  # z, roll, pitch
        rand_samples_xz[:, 4:5]   # yaw
    ], dim=1)
    
    # if the box is high, limit the x spawn range to avoid penetration
    rand_samples[:, 0] = torch.where(env.scene.env_origins[env_ids][:,2]> 0.9, rand_samples[:, 0].clip(max = -1.35), rand_samples[:,0])
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]

    positions[:, 2] -= env.scene.env_origins[env_ids, 2]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    
    velocities = root_states[:, 7:13] + rand_samples
    
    # set into the physics simulation
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)
