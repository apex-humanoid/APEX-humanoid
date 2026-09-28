import torch
import math


# Cache for index tensors to avoid recreating them
_CACHED_INDICES = {}


def _get_mapping_indices(n_joints, device):
    """Get cached mapping and flip indices for the given number of joints.
    
    Args:
        n_joints: Number of joints (23 or 29)
        device: Torch device
        
    Returns:
        Tuple of (mapping_indices, flip_indices)
    """
    cache_key = (n_joints, str(device))
    
    if cache_key not in _CACHED_INDICES:
        if n_joints == 23:
            joint_map = _joint_map_23()
            flip_set = _flip_sign_joint_indices_23()
        elif n_joints == 29:
            joint_map = _joint_map_29()
            flip_set = _flip_sign_joint_indices_29()
        else:
            raise ValueError(f"Unsupported n_joints: {n_joints}")
        
        mapping_indices = torch.tensor([joint_map[i] for i in range(n_joints)], device=device, dtype=torch.long)
        flip_indices = torch.tensor(list(flip_set), device=device, dtype=torch.long)
        
        _CACHED_INDICES[cache_key] = (mapping_indices, flip_indices)
    
    return _CACHED_INDICES[cache_key]


def data_augmentation_func_g1(obs, actions, env, is_critic):
    if obs is not None:
        # Use the new specialized mirroring depending on group
        if is_critic:
            obs_mirror = mirror_critic_obs_xz(obs)
        else:
            obs_mirror = mirror_policy_obs_xz(obs)
        obs_batch = torch.cat((obs, obs_mirror), dim=0)
    else:
        obs_batch = None
    if actions is not None:
        # Get action offset and scale from environment
        action_offset = env.env.scene["robot"].data.default_joint_pos[0, :23]
        # Action scale is typically 0.25 (can be configured in env if needed)
        action_scale = 0.25
        mean_actions_batch = mirror_action_xz_plane(actions, action_offset=action_offset, action_scale=action_scale)
        action_batch = torch.cat((actions, mean_actions_batch), dim=0)
    else:
        action_batch = None
    return obs_batch, action_batch

def data_augmentation_func_g1_down(obs, actions, env, is_critic):
    if obs is not None:
        # Use the new specialized mirroring depending on group
        if is_critic:
            obs_mirror = mirror_critic_obs_xz_down(obs)
        else:
            obs_mirror = mirror_policy_obs_xz(obs)
        obs_batch = torch.cat((obs, obs_mirror), dim=0)
    else:
        obs_batch = None
    if actions is not None:
        # Get action offset and scale from environment
        action_offset = env.env.scene["robot"].data.default_joint_pos[0, :23]
        # Action scale is typically 0.25 (can be configured in env if needed)
        action_scale = 0.25
        mean_actions_batch = mirror_action_xz_plane(actions, action_offset=action_offset, action_scale=action_scale)
        action_batch = torch.cat((actions, mean_actions_batch), dim=0)
    else:
        action_batch = None
    return obs_batch, action_batch

def data_augmentation_func_g1_standup(obs, actions, env, is_critic):
    if obs is not None:
        # Use the new specialized mirroring depending on group
        if is_critic:
            obs_mirror = mirror_critic_obs_xz_standup(obs)
        else:
            obs_mirror = mirror_policy_obs_xz_standup(obs)
        obs_batch = torch.cat((obs, obs_mirror), dim=0)
    else:
        obs_batch = None
    if actions is not None:
        # Get action offset and scale from environment
        action_offset = env.env.scene["robot"].data.default_joint_pos[0, :23]
        # Action scale is typically 0.25 (can be configured in env if needed)
        action_scale = 0.25
        mean_actions_batch = mirror_action_xz_plane(actions, action_offset=action_offset, action_scale=action_scale)
        action_batch = torch.cat((actions, mean_actions_batch), dim=0)
    else:
        action_batch = None
    return obs_batch, action_batch


def data_augmentation_func_g1_mocap(obs, actions, env, is_critic):
    if obs is not None:
        # Use the new specialized mirroring depending on group
        if is_critic:
            obs_mirror = mirror_critic_obs_xz(obs)
        else:
            obs_mirror = mirror_policy_obs_xz_mocap(obs)
        obs_batch = torch.cat((obs, obs_mirror), dim=0)
    else:
        obs_batch = None
    if actions is not None:
        # Get action offset and scale from environment
        action_offset = env.env.scene["robot"].data.default_joint_pos[0, :23]
        # Action scale is typically 0.25 (can be configured in env if needed)
        action_scale = 0.25
        mean_actions_batch = mirror_action_xz_plane(actions, action_offset=action_offset, action_scale=action_scale)
        action_batch = torch.cat((actions, mean_actions_batch), dim=0)
    else:
        action_batch = None
    return obs_batch, action_batch

def _joint_map_29():
    """Return joint index mapping for left-right swap with 29-DoF (includes wrists).

    Order provided by user:
    ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
     'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
     'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
     'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
     'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
     'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
     'left_elbow_joint', 'right_elbow_joint',
     'left_wrist_roll_joint', 'right_wrist_roll_joint',
     'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
     'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
    """
    return {
        0: 1, 1: 0,  # hip pitch L<->R
        3: 4, 4: 3,  # hip roll L<->R
        6: 7, 7: 6,  # hip yaw L<->R
        9: 10, 10: 9,  # knee L<->R
        11: 12, 12: 11,  # shoulder pitch L<->R
        13: 14, 14: 13,  # ankle pitch L<->R
        15: 16, 16: 15,  # shoulder roll L<->R
        17: 18, 18: 17,  # ankle roll L<->R
        19: 20, 20: 19,  # shoulder yaw L<->R
        21: 22, 22: 21,  # elbow L<->R
        23: 24, 24: 23,  # wrist roll L<->R
        25: 26, 26: 25,  # wrist pitch L<->R
        27: 28, 28: 27,  # wrist yaw L<->R
        # central joints remain
        2: 2, 5: 5, 8: 8,
    }


def _flip_sign_joint_indices_29():
    """Indices whose sign flips under x-z plane mirroring.

    - Roll joints: hip roll, waist roll, shoulder roll, ankle roll, wrist roll
    - Yaw joints: waist yaw, hip yaw, shoulder yaw, wrist yaw
    (Pitch joints keep their sign.)
    """
    return {
        # Roll
        3, 4, 5, 15, 16, 17, 18, 23, 24,
        # Yaw
        2, 6, 7, 19, 20, 27, 28,
    }


def _joint_map_23():
    """Return joint index mapping for left-right swap with 23-DoF (excludes wrists).

    First 23 joints (no wrists):
    ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
     'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
     'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
     'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
     'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
     'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
     'left_elbow_joint', 'right_elbow_joint']
    """
    return {
        0: 1, 1: 0,  # hip pitch L<->R
        3: 4, 4: 3,  # hip roll L<->R
        6: 7, 7: 6,  # hip yaw L<->R
        9: 10, 10: 9,  # knee L<->R
        11: 12, 12: 11,  # shoulder pitch L<->R
        13: 14, 14: 13,  # ankle pitch L<->R
        15: 16, 16: 15,  # shoulder roll L<->R
        17: 18, 18: 17,  # ankle roll L<->R
        19: 20, 20: 19,  # shoulder yaw L<->R
        21: 22, 22: 21,  # elbow L<->R
        # central joints remain
        2: 2, 5: 5, 8: 8,
    }


def _flip_sign_joint_indices_23():
    """Indices whose sign flips under x-z plane mirroring for 23-DoF (excludes wrists).

    - Roll joints: hip roll, waist roll, shoulder roll, ankle roll
    - Yaw joints: waist yaw, hip yaw, shoulder yaw
    (Pitch joints keep their sign.)
    """
    return {
        # Roll
        3, 4, 5, 15, 16, 17, 18,
        # Yaw
        2, 6, 7, 19, 20,
    }


def _remap_and_flip_joints(vec, n_joints: int, joint_map: dict[int, int], flip_set: set[int], offset: torch.Tensor = None):
    """Helper to remap a length-n_joints slice with left-right swap and sign flips.
    
    The offset represents: observed_value = real_value - offset
    So we: (1) add offset to get real values, (2) mirror, (3) subtract offset
    
    Args:
        vec: Input vector containing joints
        n_joints: Number of joints to process
        joint_map: Mapping dictionary for left-right swap
        flip_set: Set of indices to flip signs
        offset: Offset tensor to add before mirroring and subtract after (default None)
    """
    tmp = vec.clone()
    
    # Add offset to get real values
    if offset is not None:
        tmp = tmp + offset
    
    # Mirror: remap and flip signs (vectorized with caching)
    mapping_indices, flip_indices = _get_mapping_indices(n_joints, vec.device)
    vec = tmp[..., mapping_indices].clone()  # Clone to avoid in-place modification
    
    # Flip signs for specified joints (vectorized)
    vec[..., flip_indices] = -vec[..., flip_indices]
    
    # Subtract offset to get observed values
    if offset is not None:
        vec = vec - offset
    
    return vec


# --- Helper: mirror flattened height scan with provided grid (nx, ny) ---

def mirror_height_scan(flat: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    """Mirror a flattened height scan by flipping the inner y-axis (x-z plane mirror).

    Assumes flattened order is 'xy' (x major, y minor), i.e., index = x*ny + y.
    Preserves the input shape.

    Args:
        flat: Tensor with trailing dimension nx*ny (e.g., [..., nx*ny]).
        nx: grid resolution along x (outer dimension).
        ny: grid resolution along y (inner dimension).

    Returns:
        Tensor of the same shape as `flat`, mirrored along the y-axis.
    
    Raises:
        ValueError: If the trailing dimension does not match nx*ny.
    """
    if flat is None:
        return flat
    N = flat.shape[-1]
    if N != nx * ny:
        raise ValueError(f"Height scan shape mismatch: expected trailing dim {nx*ny} (nx={nx}, ny={ny}), got {N}")
    grid = flat.view(*flat.shape[:-1], nx, ny)
    grid = torch.flip(grid, dims=(-2,))  # flip inner y
    return grid.reshape(flat.shape)


def mirror_policy_obs_xz(observation: torch.Tensor, joint_offset: torch.Tensor = None) -> torch.Tensor:
    """Mirror the policy (actor) observation about x-z plane.

    Structure (history length 6):
      root_ang_vel(18=6*3), projected_gravity(18=6*3),
      joint_pos(174=6*29), joint_vel(174=6*29), actions(174=6*29),
      height_scan(676) [flattened grid, ordering='xy']
      
    Args:
        observation: Input observation tensor
        joint_offset: Offset tensor for joint positions/velocities where obs = real - offset (default None)
    """
    mirrored = observation.clone()
    pos = 0

    # root_ang_vel: 6 frames of (wx, wy, wz) — pseudovector -> flip x and z
    ang_vel = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    ang_vel[..., 0] = -ang_vel[..., 0]  # flip x
    ang_vel[..., 2] = -ang_vel[..., 2]  # flip z
    mirrored[..., pos:pos+18] = ang_vel.view(*mirrored.shape[:-1], 18)
    pos += 18

    # projected_gravity: 6 frames of (gx, gy, gz) — vector -> flip y
    gravity = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    gravity[..., 1] = -gravity[..., 1]  # flip y
    mirrored[..., pos:pos+18] = gravity.view(*mirrored.shape[:-1], 18)
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

    # height_scan: flattened (xy ordering) -> reverse inner y for each x
    h_start, h_end = pos, pos + 676
    h = mirrored[..., h_start:h_end]
    mirrored[..., h_start:h_end] = mirror_height_scan(h, nx=26, ny=26)
    pos += 676

    # Validate that we've processed the entire observation
    expected_size = observation.shape[-1]
    if pos != expected_size:
        raise ValueError(
            f"Policy observation size mismatch: processed {pos} elements but input has {expected_size}. "
            f"Expected structure: root_ang_vel(18) + projected_gravity(18) + joint_pos(174) + "
            f"joint_vel(174) + actions(174) + height_scan(676) = 1234 total."
        )

    return mirrored

def mirror_policy_obs_xz_mocap(observation: torch.Tensor, joint_offset: torch.Tensor = None) -> torch.Tensor:
    """Mirror the policy (actor) observation about x-z plane.
    Structure (history length 6):
      root_ang_vel(18=6*3), projected_gravity(18=6*3),
      joint_pos(174=6*29), joint_vel(174=6*29), actions(174=6*29),
      height_scan(676) [flattened grid, ordering='xy']
    Args:
        observation: Input observation tensor
        joint_offset: Offset tensor for joint positions/velocities where obs = real - offset (default None)
    """
    mirrored = observation.clone()
    pos = 0
    # root_ang_vel: 6 frames of (wx, wy, wz) — pseudovector -> flip x and z
    ang_vel = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    ang_vel[..., 0] = -ang_vel[..., 0]  # flip x
    ang_vel[..., 2] = -ang_vel[..., 2]  # flip z
    mirrored[..., pos:pos+18] = ang_vel.view(*mirrored.shape[:-1], 18)
    pos += 18
    # projected_gravity: 6 frames of (gx, gy, gz) — vector -> flip y
    gravity = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    gravity[..., 1] = -gravity[..., 1]  # flip y
    mirrored[..., pos:pos+18] = gravity.view(*mirrored.shape[:-1], 18)
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
    # box_height: scalar (unchanged)
    pos += 1
    # torso_pos: vector -> flip y
    torso_pos = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    torso_pos[..., 1] = -torso_pos[..., 1]
    mirrored[..., pos:pos+18] = torso_pos.view(*mirrored.shape[:-1], 18)
    pos += 18
    # torso_quat: (w,x,y,z) -> negate y component
    quat = mirrored[..., pos:pos+24].view(*mirrored.shape[:-1], 6, 4).clone()
    quat[..., 1] = -quat[..., 1]
    quat[..., 3] = -quat[..., 3]
    mirrored[..., pos:pos+24] = quat.view(*mirrored.shape[:-1], 24)
    pos += 24
    # Validate that we've processed the entire observation
    expected_size = observation.shape[-1]
    if pos != expected_size:
        raise ValueError(
            f"Policy observation size mismatch: processed {pos} elements but input has {expected_size}. "
            f"Expected structure: root_ang_vel(18) + projected_gravity(18) + joint_pos(174) + "
            f"joint_vel(174) + actions(174) + height_scan(676) = 1234 total."
        )
    return mirrored

def mirror_policy_obs_xz_standup(observation: torch.Tensor, joint_offset: torch.Tensor = None) -> torch.Tensor:
    """Mirror the policy (actor) observation about x-z plane.

    Structure (history length 6):
      root_ang_vel(18=6*3), projected_gravity(18=6*3),
      joint_pos(174=6*29), joint_vel(174=6*29), actions(174=6*29),
      
    Args:
        observation: Input observation tensor
        joint_offset: Offset tensor for joint positions/velocities where obs = real - offset (default None)
    """
    mirrored = observation.clone()
    pos = 0

    # root_ang_vel: 6 frames of (wx, wy, wz) — pseudovector -> flip x and z
    ang_vel = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    ang_vel[..., 0] = -ang_vel[..., 0]  # flip x
    ang_vel[..., 2] = -ang_vel[..., 2]  # flip z
    mirrored[..., pos:pos+18] = ang_vel.view(*mirrored.shape[:-1], 18)
    pos += 18

    # projected_gravity: 6 frames of (gx, gy, gz) — vector -> flip y
    gravity = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    gravity[..., 1] = -gravity[..., 1]  # flip y
    mirrored[..., pos:pos+18] = gravity.view(*mirrored.shape[:-1], 18)
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

    # Validate that we've processed the entire observation
    expected_size = observation.shape[-1]
    if pos != expected_size:
        raise ValueError(
            f"Policy observation size mismatch: processed {pos} elements but input has {expected_size}. "
            f"Expected structure: root_ang_vel(18) + projected_gravity(18) + joint_pos(174) + "
            f"joint_vel(174) + actions(174) + height_scan(676) = 1234 total."
        )

    return mirrored

def mirror_critic_obs_xz_down(observation: torch.Tensor, joint_offset: torch.Tensor = None) -> torch.Tensor:
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

    # torso_quat: (w,x,y,z) -> negate y component
    quat = mirrored[..., pos:pos+24].view(*mirrored.shape[:-1], 6, 4).clone()
    quat[..., 1] = -quat[..., 1]
    quat[..., 3] = -quat[..., 3]
    mirrored[..., pos:pos+24] = quat.view(*mirrored.shape[:-1], 24)
    pos += 24

    # projected_gravity: vector -> flip y
    gravity = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    gravity[..., 1] = -gravity[..., 1]
    mirrored[..., pos:pos+18] = gravity.view(*mirrored.shape[:-1], 18)
    pos += 18

    # torso_pos: vector -> flip y
    torso_pos = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    torso_pos[..., 1] = -torso_pos[..., 1]
    mirrored[..., pos:pos+18] = torso_pos.view(*mirrored.shape[:-1], 18)
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

    # box_height: scalar (unchanged)
    pos += 1

    # contact_forces: assume per-body chunks of 6 = (fx, fy, fz, tx, ty, tz)
    # cf_start, cf_end = pos, pos + 222
    # cf = mirrored[..., cf_start:cf_end]
    # if cf.shape[-1] % 6 == 0:
    #     cf_reshaped = cf.view(*cf.shape[:-1], -1, 6).clone()
    #     # flip y force, and x/z torque (pseudovector)
    #     cf_reshaped[..., 1] = -cf_reshaped[..., 1]
    #     cf_reshaped[..., 3] = -cf_reshaped[..., 3]
    #     cf_reshaped[..., 5] = -cf_reshaped[..., 5]
    #     mirrored[..., cf_start:cf_end] = cf_reshaped.view(*cf.shape[:-1], -1)
    # pos += 222

    # rigid_body_pos: assume chunks of 3 (x, y, z) -> flip y
    # rb_start, rb_end = pos, pos + 333
    # rb = mirrored[..., rb_start:rb_end]
    # if rb.shape[-1] % 3 == 0:
    #     rb_reshaped = rb.view(*rb.shape[:-1], -1, 3).clone()
    #     rb_reshaped[..., 1] = -rb_reshaped[..., 1]
    #     mirrored[..., rb_start:rb_end] = rb_reshaped.view(*rb.shape[:-1], -1)
    # pos += 333

    # progress_memory: leave unchanged
    pos += 2

    # time: unchanged
    pos += 1

    # height_scan: flattened (xy ordering) -> reverse inner y for each x
    h_start, h_end = pos, pos + 676
    h = mirrored[..., h_start:h_end]

    mirrored[..., h_start:h_end] = mirror_height_scan(h, nx=26, ny=26)
    pos += 676

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

def mirror_critic_obs_xz(observation: torch.Tensor, joint_offset: torch.Tensor = None) -> torch.Tensor:
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

    # torso_quat: (w,x,y,z) -> negate y component
    quat = mirrored[..., pos:pos+24].view(*mirrored.shape[:-1], 6, 4).clone()
    quat[..., 1] = -quat[..., 1]
    quat[..., 3] = -quat[..., 3]
    mirrored[..., pos:pos+24] = quat.view(*mirrored.shape[:-1], 24)
    pos += 24

    # projected_gravity: vector -> flip y
    gravity = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    gravity[..., 1] = -gravity[..., 1]
    mirrored[..., pos:pos+18] = gravity.view(*mirrored.shape[:-1], 18)
    pos += 18

    # torso_pos: vector -> flip y
    torso_pos = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    torso_pos[..., 1] = -torso_pos[..., 1]
    mirrored[..., pos:pos+18] = torso_pos.view(*mirrored.shape[:-1], 18)
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

    # box_height: scalar (unchanged)
    pos += 1

    # contact_forces: assume per-body chunks of 6 = (fx, fy, fz, tx, ty, tz)
    # cf_start, cf_end = pos, pos + 222
    # cf = mirrored[..., cf_start:cf_end]
    # if cf.shape[-1] % 6 == 0:
    #     cf_reshaped = cf.view(*cf.shape[:-1], -1, 6).clone()
    #     # flip y force, and x/z torque (pseudovector)
    #     cf_reshaped[..., 1] = -cf_reshaped[..., 1]
    #     cf_reshaped[..., 3] = -cf_reshaped[..., 3]
    #     cf_reshaped[..., 5] = -cf_reshaped[..., 5]
    #     mirrored[..., cf_start:cf_end] = cf_reshaped.view(*cf.shape[:-1], -1)
    # pos += 222

    # rigid_body_pos: assume chunks of 3 (x, y, z) -> flip y
    # rb_start, rb_end = pos, pos + 333
    # rb = mirrored[..., rb_start:rb_end]
    # if rb.shape[-1] % 3 == 0:
    #     rb_reshaped = rb.view(*rb.shape[:-1], -1, 3).clone()
    #     rb_reshaped[..., 1] = -rb_reshaped[..., 1]
    #     mirrored[..., rb_start:rb_end] = rb_reshaped.view(*rb.shape[:-1], -1)
    # pos += 333

    # progress_memory: leave unchanged
    pos += 8

    # time: unchanged
    pos += 1

    # height_scan: flattened (xy ordering) -> reverse inner y for each x
    h_start, h_end = pos, pos + 676
    h = mirrored[..., h_start:h_end]

    mirrored[..., h_start:h_end] = mirror_height_scan(h, nx=26, ny=26)
    pos += 676

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

def mirror_critic_obs_xz_standup(observation: torch.Tensor, joint_offset: torch.Tensor = None) -> torch.Tensor:
    """Mirror the critic observation about x-z plane.

    Structure (history length 6 for first 5 terms):
      torso_lin_vel(18), torso_ang_vel(18), torso_quat(24), projected_gravity(18), torso_pos(18),
      joint_pos(174), joint_vel(174), actions(174), feet_orientation(6=3*2),
       progress_memory(6), time(1)
      
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

    # torso_quat: (w,x,y,z) -> negate y component
    quat = mirrored[..., pos:pos+24].view(*mirrored.shape[:-1], 6, 4).clone()
    quat[..., 1] = -quat[..., 1]
    quat[..., 3] = -quat[..., 3]
    mirrored[..., pos:pos+24] = quat.view(*mirrored.shape[:-1], 24)
    pos += 24

    # projected_gravity: vector -> flip y
    gravity = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    gravity[..., 1] = -gravity[..., 1]
    mirrored[..., pos:pos+18] = gravity.view(*mirrored.shape[:-1], 18)
    pos += 18

    # torso_pos: vector -> flip y
    torso_pos = mirrored[..., pos:pos+18].view(*mirrored.shape[:-1], 6, 3).clone()
    torso_pos[..., 1] = -torso_pos[..., 1]
    mirrored[..., pos:pos+18] = torso_pos.view(*mirrored.shape[:-1], 18)
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

    feet_orientations = mirrored[..., pos:pos+6].view(*mirrored.shape[:-1], 3, 2).clone()
    for f in range(3):
        feet_orientations[..., f, :] = torch.flip(feet_orientations[..., f, :], dims=[-1])
    mirrored[..., pos:pos+6] = feet_orientations.view(*mirrored.shape[:-1], 6)
    pos += 6



    # progress_memory: swap first two items in last dim
    progress_memory = mirrored[..., pos:pos+6].clone()
    # Swap the first two elements
    progress_memory[..., 0], progress_memory[..., 1] = progress_memory[..., 1].clone(), progress_memory[..., 0].clone()
    mirrored[..., pos:pos+6] = progress_memory
    pos += 6

    # time: unchanged
    pos += 1



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


def mirror_xz_plane(observation):
    """
    Backward compatibility: old helper. For new observations, use mirror_policy_obs_xz.
    """
    return mirror_policy_obs_xz(observation)


def mirror_action_xz_plane(action, action_offset: torch.Tensor = None, action_scale: float = 0.25):
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
    n_joints = 23
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