import math
import os
import time
from pathlib import Path
from typing import Literal, Optional, Tuple, Union
from jaxtyping import Float

import numpy as np
import torch
import tyro
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from PIL import Image
from torch import Tensor, optim
# from knn_cuda import KNN

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )

def knn_cuml(x: torch.Tensor, k: int):
    # Convert tensor to numpy array
    x_np = x.cpu().numpy()
    x_np = x_np[np.isfinite(x_np).all(1)]

    # RAPIDS cuML kNN model
    from cuml.neighbors import NearestNeighbors
    model = NearestNeighbors(n_neighbors=k + 1)
    model.fit(x_np)
    distances, indices = model.predictkneighbors(x_np)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

def k_nearest_cuda(x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        knn = KNN(k=k + 1, transpose_mode=True)
        
        # Find the k-nearest neighbors
        distances, indices = knn(x.unsqueeze(0), x.unsqueeze(0))  # total 307200 points, tried to allocate 351.56 GiB...

        # Exclude the point itself from the result and return
        return distances[0, :, 1:], indices[0, :, 1:]

def k_nearest_sklearn(x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()
        x_np = x_np[np.isfinite(x_np).all(1)]

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

def projection_matrix(znear, zfar, fovx, fovy, device: Union[str, torch.device] = "cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )


class GSplat:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        points_xyz: torch.Tensor,
        points_color: torch.Tensor,
        gs_rot, gs_scale, gs_opacity
    ):
        self.device = torch.device("cuda:0")
        self.num_points = points_xyz.shape[0]

        # self.points_xyz, self.points_color = points_xyz, points_color
        # self._init_gaussians()

        self.means = points_xyz
        self.scales = gs_scale
        self.quats = gs_rot
        self.opacities = gs_opacity
        self.rgbs = points_color  # the 0th degree SH coefficient
        self.background = torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32)

    def _init_gaussians(self):
        """Use point cloud to init gaussians"""

        self.means = self.points_xyz.to(self.device)

        # find the average of the three nearest neighbors for each point and use that as the scale
        # gs_start_time = time.time()
        # # distances, _ = k_nearest_sklearn(self.means.data, 3)
        # distances, _ = knn_cuml(self.means.data, 3)
        # # distances, _ = k_nearest_cuda(self.means, 3)
        # # print(np.max(distances), np.min(distances), np.median(distances), np.mean(distances))
        # gs_end_time = time.time()
        # print(f'knn in {gs_end_time - gs_start_time:.2f}s')
        # distances = torch.from_numpy(distances)

        distances = torch.ones_like(self.means) * 0.016

        avg_dist = distances.mean(dim=-1, keepdim=True)
        self.scales = torch.log(avg_dist.repeat(1, 3)).to(self.device)

        self.quats = random_quat_tensor(self.num_points).to(self.device)
        self.opacities = torch.logit(0.1 * torch.ones(self.num_points, 1)).to(self.device)
        self.rgbs = torch.logit(self.points_color, eps=1e-10).to(self.device)  # the 0th degree SH coefficient

        self.background = torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32)

    def save_ply(self, save_name):
        import open3d as o3d

        map_to_tensors = {}
        positions = self.means.cpu().numpy()
        n = positions.shape[0]
        map_to_tensors["positions"] = positions
        map_to_tensors["normals"] = np.zeros_like(positions, dtype=np.float32)
        colors = torch.clamp(self.rgbs.clone(), 0.0, 1.0).data.cpu().numpy()
        map_to_tensors["colors"] = (colors * 255).astype(np.uint8)
        map_to_tensors["opacity"] = self.opacities.data.cpu().numpy()

        scales = self.scales.data.cpu().numpy()
        for i in range(3):
            # map_to_tensors[f"scale_{i}"] = scales[:, i, None]
            map_to_tensors[f"scale_{i}"] = np.ones_like(scales[:, i, None], dtype=np.float32) * 0.1

        quats = self.quats.data.cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=1))
            n_after = np.sum(select)
            if n_after < n_before:
                print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select, :]

        pcd = o3d.t.geometry.PointCloud(map_to_tensors)

        o3d.t.io.write_point_cloud(str(save_name), pcd)

        # import struct
        # with open( save_name, 'wb') as f:
        #     # Header
        #     f.write(b"ply\n")
        #     # f.write("format ascii 1.0\n")
        #     f.write(b"format binary_little_endian 1.0\n")
        #     f.write(f"element vertex {len(self.means)}\n".encode())
        #     f.write(b"property float x\n")
        #     f.write(b"property float y\n")
        #     f.write(b"property float z\n")
        #     f.write(b"property float scale_0\n")
        #     f.write(b"property float scale_1\n")
        #     f.write(b"property float scale_2\n")
        #     f.write(b"property float rot_0\n")
        #     f.write(b"property float rot_1\n")
        #     f.write(b"property float rot_2\n")
        #     f.write(b"property float rot_3\n")
        #     f.write(b"property float opacity\n")
        #     f.write(b"property uint8 f_dc0\n")
        #     f.write(b"property uint8 f_dc1\n")
        #     f.write(b"property uint8 f_dc2\n")
        #     f.write(b"end_header\n")
        #     for i in range(len(self.means)):
        #         x, y, z = tuple(self.means[i].cpu().numpy())
        #         scale_0, scale_1, scale_2 = tuple(self.scales[i].cpu().numpy())
        #         rot_0, rot_1, rot_2, rot_3 = tuple(self.quats[i].cpu().numpy())
        #         opacity = self.opacities[i].item()
        #         r, g, b = tuple(self.rgbs[i].cpu().numpy())
        #         # f.write(f"{x:8f} {y:8f} {z:8f} {scale_0:8f} {scale_1:8f} {scale_2:8f} {rot_0:8f} {rot_1:8f} {rot_2:8f} {rot_3:8f} {opacity:8f} {r:8f} {g:8f} {b:8f}\n")
        #         binary_data = struct.pack('ffffffffffffff', x, y, z, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3, opacity, r, g, b)
        #         f.write(binary_data)
    
    def forward(self, camera: dict, pose_r, pose_t):
        self.H, self.W = camera['height'], camera['width']
        self.fx = camera['fx']
        self.fy = camera['fy']
        self.cx = camera['cx']
        self.cy = camera['cy']

        world_to_camera = torch.eye(4, device='cpu', dtype=torch.float32)
        # world_to_camera[:3, :] = camera['world_to_camera']
        world_to_camera[:3, :3] = pose_r
        world_to_camera[:3, 3] = pose_t[:, 0]

        viewmat = world_to_camera.to(self.device)

        # calculate the FOV of the camera given fx and fy, width and height
        fovx = 2 * math.atan(self.W / (2 * self.fx))
        fovy = 2 * math.atan(self.H / (2 * self.fy))
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)  # camera to screen space

        B_SIZE = 16

        (
            xys,
            depths,
            radii,
            conics,
            compensation,
            num_tiles_hit,
            cov3d,
        ) = project_gaussians(
            self.means,
            self.scales,  # torch.exp(self.scales),
            1,
            self.quats,  # self.quats / self.quats.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            # projmat.squeeze() @ viewmat.squeeze(),  # world to screen space
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.H,
            self.W,
            B_SIZE,
        )

        out_img, alpha = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            self.rgbs,  # torch.sigmoid(self.rgbs),
            self.opacities,  # torch.sigmoid(self.opacities),
            self.H,
            self.W,
            B_SIZE,
            self.background,
            return_alpha=True
        )  # [..., :3]
        out_img = torch.clamp(out_img, max=1.0)
        
        return out_img, alpha
