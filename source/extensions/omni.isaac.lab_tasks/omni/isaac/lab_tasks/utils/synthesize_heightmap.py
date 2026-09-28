import torch


class SynHeightmap:

    def __init__(self, size, resolution, obstacle_height, obstacle_x_threshold):

        indexing = "xy"
        # define grid pattern
        x = torch.arange(start=-size[0] / 2, end=size[0] / 2 + 1.0e-9, step=resolution)
        y = torch.arange(start=-size[1] / 2, end=size[1] / 2 + 1.0e-9, step=resolution)
        grid_x, grid_y = torch.meshgrid(x, y, indexing=indexing)

        # store into ray starts
        num_rays = grid_x.numel()
        ray_starts = torch.zeros(num_rays, 3,)
        ray_starts[:, 0] = grid_x.flatten()
        ray_starts[:, 1] = grid_y.flatten()

        self.ray_starts = ray_starts
        self.num_rays = num_rays

        offset_pos = torch.tensor(list((0, 0, 20)))
        self.ray_starts += offset_pos
        self.obstacle_height = obstacle_height * torch.ones(num_rays)
        self.obstacle_x_threshold = obstacle_x_threshold
    
    def update(self, torso_pos, torso_quat):
        """Update the heightmap with new height data.
        
        Args:
            height_data (torch.Tensor): A tensor containing the new height data.
        """
        ray_starts_w = self.quat_apply_yaw(torso_quat.repeat(1, self.num_rays), self.ray_starts)
        ray_starts_w += torso_pos.unsqueeze(1)
        raw_hits = torch.where(ray_starts_w[..., 0] <= self.obstacle_x_threshold, 0, self.obstacle_height)
        return raw_hits
    
    def normalize(self, x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        
        return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

    def yaw_quat(self, quat: torch.Tensor) -> torch.Tensor:

        shape = quat.shape
        quat_yaw = quat.clone().view(-1, 4)
        qw = quat_yaw[:, 0]
        qx = quat_yaw[:, 1]
        qy = quat_yaw[:, 2]
        qz = quat_yaw[:, 3]
        yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        quat_yaw[:] = 0.0
        quat_yaw[:, 3] = torch.sin(yaw / 2)
        quat_yaw[:, 0] = torch.cos(yaw / 2)
        quat_yaw = self.normalize(quat_yaw)
        return quat_yaw.view(shape)
    
    def quat_apply(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:

        # store shape
        shape = vec.shape
        # reshape to (N, 3) for multiplication
        quat = quat.reshape(-1, 4)
        vec = vec.reshape(-1, 3)
        # extract components from quaternions
        xyz = quat[:, 1:]
        t = xyz.cross(vec, dim=-1) * 2
        return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

    def quat_apply_yaw(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate a vector only around the yaw-direction.

        Args:
            quat: The orientation in (w, x, y, z). Shape is (N, 4).
            vec: The vector in (x, y, z). Shape is (N, 3).

        Returns:
            The rotated vector in (x, y, z). Shape is (N, 3).
        """
        
        quat_yaw = self.yaw_quat(quat)
        return self.quat_apply(quat_yaw, vec)

    