# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Walking task functions."""
from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Literal
import carb
import omni.physics.tensors.impl.api as physx
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators import ImplicitActuator
from omni.isaac.lab.assets import Articulation, DeformableObject, RigidObject
from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

class reset_joints_by_offset_specified_joint(ManagerTermBase):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        # call super
        super().__init__(cfg, env)
        
        # extract params
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        position_range: dict[str, tuple[float, float]] = cfg.params["position_range"]
        velocity_range: tuple[float, float] = cfg.params["velocity_range"]
        num_buckets: int = cfg.params.get("num_buckets", 1000)

        # get asset
        asset: Articulation = env.scene[asset_cfg.name]
        
        # pre-computation for position
        self._pos_noise_buckets = torch.zeros(num_buckets, asset.num_joints, device=asset.device)
        
        for joint_name, val_range in position_range.items():
            ids, _ = asset.find_joints(joint_name)
            if len(ids) > 0:
                self._pos_noise_buckets[:, ids] += math_utils.sample_uniform(
                    *val_range, (num_buckets, len(ids)), asset.device
                )

        # pre-computation for velocity
        self._vel_noise_buckets = math_utils.sample_uniform(
            *velocity_range, (num_buckets, asset.num_joints), asset.device
        )
        
        # 10% prob, we do not add any noise
        num_zero_buckets = int(0.1 * num_buckets)
        self._pos_noise_buckets[:num_zero_buckets] = 0.0
        self._vel_noise_buckets[:num_zero_buckets] = 0.0
        
        self._num_buckets = num_buckets

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        position_range: dict[str, tuple[float, float]],
        velocity_range: tuple[float, float],
        num_buckets: int = 1000,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]


        # get default joint state
        joint_pos = asset.data.default_joint_pos[env_ids].clone()
        joint_vel = asset.data.default_joint_vel[env_ids].clone()

        # sample bucket indices
        bucket_ids = torch.randint(0, self._num_buckets, (len(env_ids),), device=env.device)

        # bias positions
        joint_pos += self._pos_noise_buckets[bucket_ids]

        # bias velocities
        joint_vel += self._vel_noise_buckets[bucket_ids]

        # clamp joint pos to limits
        joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        # clamp joint vel to limits
        joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        # set into the physics simulation
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

