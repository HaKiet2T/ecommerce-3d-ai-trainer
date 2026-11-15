import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

try:
    import mcubes
except ImportError:
    mcubes = None

class ConvBlock(nn.Module):
    """Khoi Convolution co ban gom 2 layer Conv2d + GroupNorm + SiLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class DownSample(nn.Module):
    """Khoi giam chieu khong gian bang Conv stride 2."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)

class UpSample(nn.Module):
    """Khoi tang chieu khong gian bang Bilinear Upsample + Conv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

class MultiViewUNet(nn.Module):
    """
    Mang U-Net tich chap bien doi 6 goc nhin truc giao (RGB + CCM)
    thanh 3 mat phang truc giao Triplane (XY, XZ, YZ).
    Dung kien truc Convolution thay vi Transformer giup tiet kiem VRAM,
    chay tot tren 1 GPU don le (T4/A100) va tang toc do suy luan gap 5-10 lan.
    """
    def __init__(
        self,
        in_channels_per_view: int = 6, # 3 channels RGB + 3 channels CCM
        num_views: int = 6,
        triplane_channels: int = 32,
        base_features: int = 64
    ):
        super().__init__()
        self.num_views = num_views
        self.triplane_channels = triplane_channels

        # Dau vao la 6 views ghep lai: total_in = 6 views * 6 channels = 36 channels
        total_in = in_channels_per_view * num_views

        # Encoder U-Net
        self.inc = ConvBlock(total_in, base_features)
        self.down1 = DownSample(base_features, base_features * 2)
        self.conv1 = ConvBlock(base_features * 2, base_features * 2)

        self.down2 = DownSample(base_features * 2, base_features * 4)
        self.conv2 = ConvBlock(base_features * 4, base_features * 4)

        self.down3 = DownSample(base_features * 4, base_features * 8)
        self.conv3 = ConvBlock(base_features * 8, base_features * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(base_features * 8, base_features * 8)

        # Decoder U-Net
        self.up1 = UpSample(base_features * 8, base_features * 4)
        self.up_conv1 = ConvBlock(base_features * 8, base_features * 4)

        self.up2 = UpSample(base_features * 4, base_features * 2)
        self.up_conv2 = ConvBlock(base_features * 4, base_features * 2)

        self.up3 = UpSample(base_features * 2, base_features)
        self.up_conv3 = ConvBlock(base_features * 2, base_features)

        # Head sinh 3 mat phang Triplane (XY, XZ, YZ): 3 * triplane_channels
        self.out_head = nn.Conv2d(
            base_features,
            3 * triplane_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, images: torch.Tensor, ccms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 6, 3, H, W]
            ccms:   [B, 6, 3, H, W]
        Returns:
            triplanes: [B, 3, triplane_channels, H, W]
        """
        B, V, C_img, H, W = images.shape
        # Ghep kenh RGB va CCM cua moi view -> [B, 6, 6, H, W]
        view_features = torch.cat([images, ccms], dim=2)
        # Reshape ghep 6 view thanh chieu kenh duy nhat -> [B, 36, H, W]
        x = view_features.view(B, V * (C_img + ccms.shape[2]), H, W)

        # Encoder
        x0 = self.inc(x)
        x1 = self.conv1(self.down1(x0))
        x2 = self.conv2(self.down2(x1))
        x3 = self.conv3(self.down3(x2))

        # Bottleneck
        bn = self.bottleneck(x3)

        # Decoder voi Skip Connections
        d1 = self.up1(bn)
        d1 = self.up_conv1(torch.cat([d1, x2], dim=1))

        d2 = self.up2(d1)
        d2 = self.up_conv2(torch.cat([d2, x1], dim=1))

        d3 = self.up3(d2)
        d3 = self.up_conv3(torch.cat([d3, x0], dim=1))

        # Triplanes projection
        out = self.out_head(d3) # [B, 3 * C_plane, H, W]
        triplanes = out.view(B, 3, self.triplane_channels, H, W)
        return triplanes

class TriplaneSDFDecoder(nn.Module):
    """
    Giai ma dac trung Triplane tai toa do diem 3D bat ky (x, y, z):
    1. Chieu p len 3 mat phang XY, XZ, YZ
    2. Trich xuat feature bang phep noi suy song tuyen (Bilinear Interpolation)
    3. Ghep dac trung va du doan SDF kem mau RGB (Albedo) qua MLP
    """
    def __init__(self, triplane_channels: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.triplane_channels = triplane_channels

        # Input vao MLP: 3 * triplane_channels + 3 toa do (x, y, z)
        in_dim = 3 * triplane_channels + 3

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True)
        )

        # Dau ra SDF (1 channel)
        self.sdf_head = nn.Linear(hidden_dim, 1)
        # Dau ra RGB Color (3 channels trong khoang [0, 1])
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def sample_triplane_features(
        self,
        triplanes: torch.Tensor,
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Trich xuat dac trung tai toa do diem 3D.
        Args:
            triplanes: [B, 3, C, H, W]
            points: [B, N, 3] trong khoang [-0.5, 0.5]
        Returns:
            features: [B, N, 3 * C]
        """
        B, N, _ = points.shape
        C = self.triplane_channels

        # Chuan hoa toa do ve dai [-1, 1] cho F.grid_sample
        # Do diem 3D nam trong [-0.5, 0.5]
        norm_pts = points * 2.0 # [-1, 1]

        # Tinh toa do tren tung mat phang chieu
        # XY plane: x, y
        grid_xy = norm_pts[:, :, [0, 1]].view(B, N, 1, 2)
        # XZ plane: x, z
        grid_xz = norm_pts[:, :, [0, 2]].view(B, N, 1, 2)
        # YZ plane: y, z
        grid_yz = norm_pts[:, :, [1, 2]].view(B, N, 1, 2)

        feat_xy = F.grid_sample(triplanes[:, 0], grid_xy, mode="bilinear", padding_mode="border", align_corners=False)
        feat_xz = F.grid_sample(triplanes[:, 1], grid_xz, mode="bilinear", padding_mode="border", align_corners=False)
        feat_yz = F.grid_sample(triplanes[:, 2], grid_yz, mode="bilinear", padding_mode="border", align_corners=False)

        # Reshape ve [B, N, C]
        feat_xy = feat_xy.view(B, C, N).permute(0, 2, 1)
        feat_xz = feat_xz.view(B, C, N).permute(0, 2, 1)
        feat_yz = feat_yz.view(B, C, N).permute(0, 2, 1)

        # Ghep 3 mat phang thanh dac trung tong hop [B, N, 3 * C]
        combined_feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=-1)
        return combined_feat

    def forward(
        self,
        triplanes: torch.Tensor,
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Du doan SDF va mau RGB tai tap diem 3D.
        Returns:
            pred_sdf: [B, N, 1]
            pred_rgb: [B, N, 3]
        """
        sampled_features = self.sample_triplane_features(triplanes, points)
        # Ghep toa do goc vao vector dac trung
        mlp_in = torch.cat([sampled_features, points], dim=-1)

        hidden = self.mlp(mlp_in)
        pred_sdf = self.sdf_head(hidden)
        pred_rgb = self.rgb_head(hidden)

        return pred_sdf, pred_rgb

class ConvolutionalReconstructionModel(nn.Module):
    """
    Toan bo kien truc CRM (Convolutional Reconstruction Model):
    - Input: 6 goc nhin RGB + 6 CCMs
    - Triplane Generator: MultiViewUNet sinh 3 mat phang XY, XZ, YZ
    - Implicit Decoder: TriplaneSDFDecoder giai ma SDF va Color
    - Khong gian mesh: Marching Cubes hoac FlexiCubes de xuat 3D mesh textured
    """
    def __init__(
        self,
        triplane_channels: int = 32,
        base_features: int = 64,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.unet = MultiViewUNet(
            in_channels_per_view=6,
            num_views=6,
            triplane_channels=triplane_channels,
            base_features=base_features
        )
        self.decoder = TriplaneSDFDecoder(
            triplane_channels=triplane_channels,
            hidden_dim=hidden_dim
        )

    def forward(
        self,
        images: torch.Tensor,
        ccms: torch.Tensor,
        points: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Lan truyen thuan:
        - Sinh Triplane features
        - Neu co points: truy van SDF va RGB cho huan luyen
        """
        triplanes = self.unet(images, ccms)
        output = {"triplanes": triplanes}

        if points is not None:
            pred_sdf, pred_rgb = self.decoder(triplanes, points)
            output["pred_sdf"] = pred_sdf
            output["pred_rgb"] = pred_rgb

        return output

    @torch.no_grad()
    def extract_mesh(
        self,
        triplanes: torch.Tensor,
        resolution: int = 64,
        threshold: float = 0.0,
        chunk_size: int = 65536
    ) -> trimesh.Trimesh:
        """
        Tich xuat tam giac 3D Mesh tu truong SDF bang Marching Cubes.
        Chia nho chunk_size de dam bao khong bi tran VRAM tren GPU T4.
        """
        device = triplanes.device
        grid_vals = np.linspace(-0.5, 0.5, resolution, dtype=np.float32)
        X, Y, Z = np.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
        grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        num_pts = grid_points.shape[0]
        sdf_vals = np.zeros(num_pts, dtype=np.float32)

        # Truy van SDF theo tung batch de tiet kiem bo nho
        for start_idx in range(0, num_pts, chunk_size):
            end_idx = min(start_idx + chunk_size, num_pts)
            chunk_pts = torch.from_numpy(grid_points[start_idx:end_idx]).unsqueeze(0).to(device)
            pred_sdf, _ = self.decoder(triplanes, chunk_pts)
            sdf_vals[start_idx:end_idx] = pred_sdf.squeeze().cpu().numpy()

        sdf_volume = sdf_vals.reshape(resolution, resolution, resolution)

        # Chay Marching Cubes
        if mcubes is not None:
            vertices, triangles = mcubes.marching_cubes(sdf_volume, threshold)
            # Chuyen toa do index ve khong gian thuc [-0.5, 0.5]
            vertices = vertices / (resolution - 1) - 0.5
        else:
            # Fallback thuat toan Marching Cubes don gian neu chua compile PyMCubes
            from skimage import measure
            vertices, triangles, _, _ = measure.marching_cubes(
                sdf_volume, level=threshold, spacing=(1.0/resolution, 1.0/resolution, 1.0/resolution)
            )
            vertices = vertices - 0.5

        if len(vertices) == 0:
            # Fallback hinh cau neu vat the bi rong
            return trimesh.creation.icosphere(radius=0.3)

        # Truy van mau sac RGB tai cac dinh mesh
        vert_tensor = torch.from_numpy(vertices.astype(np.float32)).unsqueeze(0).to(device)
        _, vert_rgb = self.decoder(triplanes, vert_tensor)
        vert_rgb_np = (vert_rgb.squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Tao trimesh kem mau dinh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=triangles,
            vertex_colors=vert_rgb_np,
            process=True
        )
        return mesh
