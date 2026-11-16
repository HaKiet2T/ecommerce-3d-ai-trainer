"""
Convolutional Reconstruction Model (CRM) Architecture
====================================================
End-to-end 3D reconstruction model:
1. Multi-view & CCM feature representation (36 input channels)
2. Convolutional U-Net generating Triplane features (XY, XZ, YZ planes)
3. Implicit MLP Decoder querying Signed Distance Function (SDF) and RGB Albedo
4. Marching Cubes extraction for textured 3D meshes
"""

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

try:
    import mcubes
except ImportError:
    mcubes = None


class DoubleConv(nn.Module):
    """Double convolution block with GroupNorm and GELU."""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block via MaxPool2d and DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsampling block with skip connection."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CRMUNet(nn.Module):
    """
    CRM Convolutional U-Net for Triplane Generation.
    Input: 6 RGB views (concatenated) + 6 CCMs = [B, 36, H, W]
    Output: Triplane features [B, triplane_channels * 3, H, W] (XY, XZ, YZ planes)
    """
    def __init__(self, in_channels: int = 36, triplane_channels: int = 32, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.triplane_channels = triplane_channels
        out_channels = triplane_channels * 3

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 8)

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 4)
        self.up2 = Up(base_channels * 8, base_channels * 2)
        self.up3 = Up(base_channels * 4, base_channels)
        self.up4 = Up(base_channels * 2, base_channels)

        # Output head
        self.outc = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


class TriplaneMLP(nn.Module):
    """
    MLP decoder: triplane features -> SDF + Color per 3D point.
    """
    def __init__(self, triplane_channels: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.sdf_net = nn.Sequential(
            nn.Linear(triplane_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.color_net = nn.Sequential(
            nn.Linear(triplane_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sdf = self.sdf_net(features)
        color = self.color_net(features)
        return sdf, color


class CRMModel(nn.Module):
    """
    Full CRM Model: U-Net encoder + Triplane + MLP decoder.
    """
    def __init__(
        self,
        in_channels: int = 36,
        triplane_channels: int = 32,
        base_channels: int = 64,
        hidden_dim: int = 64,
        triplane_resolution: int = 256
    ):
        super().__init__()
        self.triplane_channels = triplane_channels
        self.triplane_resolution = triplane_resolution

        self.unet = CRMUNet(
            in_channels=in_channels,
            triplane_channels=triplane_channels,
            base_channels=base_channels
        )

        self.mlp = TriplaneMLP(
            triplane_channels=triplane_channels,
            hidden_dim=hidden_dim
        )

    def query_triplane(self, triplane: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Truy van triplane tai cac diem 3D.
        Args:
            triplane: [B, C*3, H, W]
            points: [B, N, 3] trong khoang [-1, 1] hoac [-0.5, 0.5]
        Returns:
            features: [B, N, C]
        """
        B, _, H, W = triplane.shape
        C = self.triplane_channels
        N = points.shape[1]

        xy_plane = triplane[:, :C]
        xz_plane = triplane[:, C:2*C]
        yz_plane = triplane[:, 2*C:]

        # Sample coordinates directly matching train_stage2.py
        xy_coords = points[:, :, :2].unsqueeze(2)
        xz_coords = points[:, :, [0, 2]].unsqueeze(2)
        yz_coords = points[:, :, [1, 2]].unsqueeze(2)

        f_xy = F.grid_sample(xy_plane, xy_coords, align_corners=True, mode="bilinear")
        f_xz = F.grid_sample(xz_plane, xz_coords, align_corners=True, mode="bilinear")
        f_yz = F.grid_sample(yz_plane, yz_coords, align_corners=True, mode="bilinear")

        f_xy = f_xy.squeeze(-1).permute(0, 2, 1)
        f_xz = f_xz.squeeze(-1).permute(0, 2, 1)
        f_yz = f_yz.squeeze(-1).permute(0, 2, 1)

        features = f_xy + f_xz + f_yz
        return features

    def forward(
        self,
        views: torch.Tensor,
        ccms: torch.Tensor,
        points: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            views: [B, 6, 3, H, W]
            ccms:  [B, 6, 3, H, W]
            points: [B, N, 3] optional
        """
        B = views.shape[0]
        # Concat views and ccms along channel dim: [B, 36, H, W]
        x = torch.cat([views.view(B, 18, *views.shape[3:]),
                       ccms.view(B, 18, *ccms.shape[3:])], dim=1)

        triplane = self.unet(x)
        output = {"triplane": triplane}

        if points is not None:
            features = self.query_triplane(triplane, points)
            sdf, color = self.mlp(features)
            output["pred_sdf"] = sdf
            output["pred_rgb"] = color

        return output

    @torch.no_grad()
    def extract_mesh(
        self,
        triplane: torch.Tensor,
        resolution: int = 64,
        threshold: float = 0.0,
        chunk_size: int = 65536
    ) -> trimesh.Trimesh:
        """
        Extract 3D triangle mesh from SDF volume via Marching Cubes.
        """
        device = triplane.device
        grid_vals = np.linspace(-0.5, 0.5, resolution, dtype=np.float32)
        X, Y, Z = np.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
        grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        num_pts = grid_points.shape[0]
        sdf_vals = np.zeros(num_pts, dtype=np.float32)

        for start_idx in range(0, num_pts, chunk_size):
            end_idx = min(start_idx + chunk_size, num_pts)
            chunk_pts = torch.from_numpy(grid_points[start_idx:end_idx]).unsqueeze(0).to(device)
            feat = self.query_triplane(triplane, chunk_pts)
            pred_sdf, _ = self.mlp(feat)
            sdf_vals[start_idx:end_idx] = pred_sdf.squeeze().cpu().numpy()

        sdf_volume = sdf_vals.reshape(resolution, resolution, resolution)

        # Dynamic threshold adjustment if level is outside volume range
        vol_min = float(sdf_volume.min())
        vol_max = float(sdf_volume.max())
        target_level = threshold
        if target_level <= vol_min or target_level >= vol_max:
            # Fallback to median/mean level to ensure surface extraction
            target_level = float(np.percentile(sdf_volume, 30))

        if mcubes is not None:
            vertices, triangles = mcubes.marching_cubes(sdf_volume, target_level)
            vertices = vertices / (resolution - 1) - 0.5
        else:
            from skimage import measure
            vertices, triangles, _, _ = measure.marching_cubes(
                sdf_volume, level=target_level, spacing=(1.0/resolution, 1.0/resolution, 1.0/resolution)
            )
            vertices = vertices - 0.5

        if len(vertices) == 0:
            return trimesh.creation.icosphere(radius=0.3)

        # Query vertex RGB
        vert_tensor = torch.from_numpy(vertices.astype(np.float32)).unsqueeze(0).to(device)
        vert_feat = self.query_triplane(triplane, vert_tensor)
        _, vert_rgb = self.mlp(vert_feat)
        vert_rgb_np = (vert_rgb.squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=triangles,
            vertex_colors=vert_rgb_np,
            process=True
        )
        return mesh


# Alias for backward compatibility
ConvolutionalReconstructionModel = CRMModel
MultiViewUNet = CRMUNet
TriplaneSDFDecoder = TriplaneMLP
