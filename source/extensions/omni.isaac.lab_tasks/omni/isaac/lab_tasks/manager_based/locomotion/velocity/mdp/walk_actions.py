# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Walking action with the branch's independent previous-target buffer."""
from __future__ import annotations
import torch
from omni.isaac.lab.envs.mdp.actions.joint_actions import JointAction
from omni.isaac.lab.envs.mdp.actions import actions_cfg
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

class WalkJointAction(JointAction):         # NEW WALK JOINT ACTION
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.WalkJointActionCfg
    """The configuration of the action term."""
    
    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # different from other settting
        self._processed_actions = torch.zeros(self.num_envs, 29, device=self.device)
        self.last_processed_actions =torch.zeros(self.num_envs, 29, device=self.device)

        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos.clone()   # 29 dim
            print(f"[DEBUG] walk offset = {self._offset.shape}")
        print(f"[DEBUG] processed_actions = {self._processed_actions.shape}")

    def process_actions(self, actions: torch.Tensor):

        # print(f"[DEBUG] self.processed_actions = {self.processed_actions.shape}")
        self._raw_actions[:] = actions
        self.last_processed_actions[:] = self._processed_actions[:]
        
        # clip actions
        if self.cfg.clip is not None:
            actions = torch.clamp(  # this is 12 dim
                self._raw_actions[:], min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        
        # apply the affine transformations
        self._processed_actions[:] = self._offset[:]
        self._processed_actions[:, self._joint_ids] += self._scale * actions

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, )
