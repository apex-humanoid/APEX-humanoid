# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Walking task functions."""
import torch
import math
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.symmetry import _flip_sign_joint_indices_29
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.symmetry import _get_mapping_indices_crawl as _get_mapping_indices
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.symmetry import _joint_map_29
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.symmetry import _remap_and_flip_joints_crawl as _remap_and_flip_joints

def data_augmentation_func_walk(obs, actions, env, is_critic):
    if obs is not None:
        # Use the new specialized mirroring depending on group
        offset = env.env.scene["robot"].data.default_joint_pos[0, :]
        if is_critic:
            obs_mirror = mirror_critic_obs_xz_walk(obs, joint_offset=offset)
        else:
            obs_mirror = mirror_policy_obs_xz_walk(obs, joint_offset=offset)
        obs_batch = torch.cat((obs, obs_mirror), dim=0)
    else:
        obs_batch = None
    if actions is not None:
        # Get action offset and scale from environment
        walk_joint_ids = [0,1,3,4,6,7,9,10,13,14, 17,18]
        action_offset = env.env.scene["robot"].data.default_joint_pos[0, walk_joint_ids]
        # Action scale is typically 0.25 (can be configured in env if needed)
        action_scale = 0.25
        mean_actions_batch = mirror_action_xz_plane_walk(actions, action_offset=action_offset, action_scale=action_scale)
        action_batch = torch.cat((actions, mean_actions_batch), dim=0)
    else:
        action_batch = None
    return obs_batch, action_batch


def mirror_policy_obs_xz_walk(observation: torch.Tensor, joint_offset: torch.Tensor = None) -> torch.Tensor:
    """Mirror the critic observation about x-z plane.

    Structure (history length 6 for first 5 terms):
      torso_lin_vel(18), torso_ang_vel(18), torso_quat(24), projected_gravity(18), torso_pos(18),
      joint_pos(174), joint_vel(174), actions(174), box_height(1),
      contact_forces(222), rigid_body_pos(333), progress_memory(8), time(1)
      
    Args:
        observation: Input observation tensor
        joint_offset: Offset tensor for joint positions/velocities where obs = real - offset (default None)
    """
    mirrored = observation.clone()
    pos = 0

 

    # torso_ang_vel: pseudovector -> flip x and z
    ang_vel = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    ang_vel[..., 0] = -ang_vel[..., 0]
    ang_vel[..., 2] = -ang_vel[..., 2]
    mirrored[..., pos:pos+18] = ang_vel.view(*mirrored.shape[:-1], 18)
    pos += 18


    # projected_gravity: vector -> flip y
    gravity = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    gravity[..., 1] = -gravity[..., 1]
    mirrored[..., pos:pos+18] = gravity.view(*mirrored.shape[:-1], 18)
    pos += 18

    # vel commands: vector -> flip y
    vel_cmd = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    vel_cmd[..., 1] = -vel_cmd[..., 1]
    mirrored[..., pos:pos+18] = vel_cmd.view(*mirrored.shape[:-1], 18)
    pos += 18

    # joint_pos: 6 frames of 29 (with offset)
    n_j = 29
    jmap = _joint_map_29()
    fset = _flip_sign_joint_indices_29()
    joint_pos = mirrored[..., pos:pos+6*n_j].view(*mirrored.shape[:-1], 6, n_j).clone()
    for f in range(6):
        joint_pos[..., f, :] = _remap_and_flip_joints(joint_pos[..., f, :], n_j, jmap, fset, offset=joint_offset)
    mirrored[..., pos:pos+6*n_j] = joint_pos.view(*mirrored.shape[:-1], 6*n_j)
    pos += 6 * n_j

    # joint_vel: same mapping/signs (no offset for velocity)
    joint_vel = mirrored[..., pos:pos+6*n_j].view(*mirrored.shape[:-1], 6, n_j).clone()
    for f in range(6):
        joint_vel[..., f, :] = _remap_and_flip_joints(joint_vel[..., f, :], n_j, jmap, fset, offset=None)
    mirrored[..., pos:pos+6*n_j] = joint_vel.view(*mirrored.shape[:-1], 6*n_j)
    pos += 6 * n_j

    # actions: same mapping/signs (NO offset - actions in obs have no offset)
    actions = mirrored[..., pos:pos+6*n_j].view(*mirrored.shape[:-1], 6, n_j).clone()
    for f in range(6):
        actions[..., f, :] = _remap_and_flip_joints(actions[..., f, :], n_j, jmap, fset, offset=None)
    mirrored[..., pos:pos+6*n_j] = actions.view(*mirrored.shape[:-1], 6*n_j)
    pos += 6 * n_j


    # phase: scalar (unchanged)
    pos += 6 * 2


    # Validate that we've processed the entire observation
    expected_size = observation.shape[-1]
    if pos != expected_size:
        raise ValueError(
            f"Critic observation size mismatch: processed {pos} elements but input has {expected_size}. "
            f"Expected structure: torso_lin_vel(18) + torso_ang_vel(18) + torso_quat(24) + "
            f"projected_gravity(18) + torso_pos(18) + joint_pos(174) + joint_vel(174) + "
            f"actions(174) + box_height(1) + contact_forces(222) + rigid_body_pos(333) + "
            f"progress_memory(8) + time(1) = 1163 total."
        )

    return mirrored


def mirror_critic_obs_xz_walk(observation: torch.Tensor, joint_offset: torch.Tensor = None) -> torch.Tensor:
    """Mirror the critic observation about x-z plane.

    Structure (history length 6 for first 5 terms):
      torso_lin_vel(18), torso_ang_vel(18), torso_quat(24), projected_gravity(18), torso_pos(18),
      joint_pos(174), joint_vel(174), actions(174), box_height(1),
      contact_forces(222), rigid_body_pos(333), progress_memory(8), time(1)
      
    Args:
        observation: Input observation tensor
        joint_offset: Offset tensor for joint positions/velocities where obs = real - offset (default None)
    """
    mirrored = observation.clone()
    pos = 0

    # torso_lin_vel: vector -> flip y
    lin_vel = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    lin_vel[..., 1] = -lin_vel[..., 1]
    mirrored[..., pos:pos+18] = lin_vel.view(*mirrored.shape[:-1], 18)
    pos += 18

    # torso_ang_vel: pseudovector -> flip x and z
    ang_vel = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    ang_vel[..., 0] = -ang_vel[..., 0]
    ang_vel[..., 2] = -ang_vel[..., 2]
    mirrored[..., pos:pos+18] = ang_vel.view(*mirrored.shape[:-1], 18)
    pos += 18


    # projected_gravity: vector -> flip y
    gravity = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    gravity[..., 1] = -gravity[..., 1]
    mirrored[..., pos:pos+18] = gravity.view(*mirrored.shape[:-1], 18)
    pos += 18

    # vel commands: vector -> flip y
    vel_cmd = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    vel_cmd[..., 1] = -vel_cmd[..., 1]
    mirrored[..., pos:pos+18] = vel_cmd.view(*mirrored.shape[:-1], 18)
    pos += 18

    # joint_pos: 6 frames of 29 (with offset)
    n_j = 29
    jmap = _joint_map_29()
    fset = _flip_sign_joint_indices_29()
    joint_pos = mirrored[..., pos:pos+6*n_j].view(*mirrored.shape[:-1], 6, n_j).clone()
    for f in range(6):
        joint_pos[..., f, :] = _remap_and_flip_joints(joint_pos[..., f, :], n_j, jmap, fset, offset=joint_offset)
    mirrored[..., pos:pos+6*n_j] = joint_pos.view(*mirrored.shape[:-1], 6*n_j)
    pos += 6 * n_j

    # joint_vel: same mapping/signs (no offset for velocity)
    joint_vel = mirrored[..., pos:pos+6*n_j].view(*mirrored.shape[:-1], 6, n_j).clone()
    for f in range(6):
        joint_vel[..., f, :] = _remap_and_flip_joints(joint_vel[..., f, :], n_j, jmap, fset, offset=None)
    mirrored[..., pos:pos+6*n_j] = joint_vel.view(*mirrored.shape[:-1], 6*n_j)
    pos += 6 * n_j

    # actions: same mapping/signs (NO offset - actions in obs have no offset)
    actions = mirrored[..., pos:pos+6*n_j].view(*mirrored.shape[:-1], 6, n_j).clone()
    for f in range(6):
        actions[..., f, :] = _remap_and_flip_joints(actions[..., f, :], n_j, jmap, fset, offset=None)
    mirrored[..., pos:pos+6*n_j] = actions.view(*mirrored.shape[:-1], 6*n_j)
    pos += 6 * n_j


    # phase: scalar (unchanged)
    pos += 6 * 2


    # Validate that we've processed the entire observation
    expected_size = observation.shape[-1]
    if pos != expected_size:
        raise ValueError(
            f"Critic observation size mismatch: processed {pos} elements but input has {expected_size}. "
            f"Expected structure: torso_lin_vel(18) + torso_ang_vel(18) + torso_quat(24) + "
            f"projected_gravity(18) + torso_pos(18) + joint_pos(174) + joint_vel(174) + "
            f"actions(174) + box_height(1) + contact_forces(222) + rigid_body_pos(333) + "
            f"progress_memory(8) + time(1) = 1163 total."
        )

    return mirrored


def mirror_action_xz_plane_walk(action, action_offset: torch.Tensor = None, action_scale: float = 0.25):
    """
    Perform x-z plane symmetry transformation on 23-DoF action tensor (excludes wrist joints).
    
    The action transformation in Isaac Lab is: real_target = scale * action + offset
    
    To mirror:
    1. Convert to real target positions: real = scale * action + offset
    2. Mirror the real target positions (swap left/right, flip signs)
    3. Convert back to action space: action_mirrored = (real_mirrored - offset) / scale
    
    Args:
        action: [..., 23] action tensor (policy output, typically in [-1, 1] range)
        action_offset: Offset tensor (default joint positions) (default None)
        action_scale: Scale factor applied to actions (default 0.25)
    """
    # Step 1: Convert action to real target positions
    # real_target = scale * action + offset
    if action_offset is not None:
        real_target = action_scale * action + action_offset
    else:
        real_target = action_scale * action
    
    # Step 2: Mirror the real target positions (remap and flip signs) - vectorized with caching
    n_joints = 12
    mapping_indices, flip_indices = _get_mapping_indices(n_joints, action.device)
    mirrored_action = real_target[..., mapping_indices].clone()  # Clone to avoid in-place modification
    
    # Flip signs for specified joints (vectorized)
    mirrored_action[..., flip_indices] = -mirrored_action[..., flip_indices]
    
    # Step 3: Convert back to action space
    # action_mirrored = (real_mirrored - offset) / scale
    if action_offset is not None:
        mirrored_action = (mirrored_action - action_offset) / action_scale
    else:
        mirrored_action = mirrored_action / action_scale

    return mirrored_action
