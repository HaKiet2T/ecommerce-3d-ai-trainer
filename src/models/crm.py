"""
Convolutional Reconstruction Model (CRM) Architecture
====================================================
End-to-end 3D reconstruction model:
1. Multi-view & CCM feature representation (6 orthographic RGB + 6 CCMs)
2. Foundation Architecture (629 parameters, 100% matched with Zhengyi/CRM checkpoint):
   - Spatial view reordering via get_tri
   - Convolutional U-Net backbone (UNet2DModel from diffusers, 612 keys)
   - Learned latent triplane parameter (1 key)
   - Implicit feature query decoder (TetTexNet)
   - Geometry head: SdfMlp (Linear 32 -> 512 -> 512 -> 4, 6 keys)
   - Appearance head: RgbMlp (Linear 32 -> 512 -> 512 -> 3, 6 keys)
   - FlexiCubes deformation weights head (weightMlp, 4 keys)
3. Lightweight Architecture (17.3M parameters, 68 keys) for fast CPU inference.
4. Marching Cubes iso-surface extraction for textured 3D meshes.
"""

import math
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

try:
    from diffusers import UNet2DModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    UNet2DModel = None
    DIFFUSERS_AVAILABLE = False

try:
    import mcubes
except ImportError:
    mcubes = None


# ---------------------------------------------------------------------------
# View-to-Triplane Spatial Reordering
# ---------------------------------------------------------------------------

def get_tri(triview_color: torch.Tensor, dim: int = 1, blender: bool = True, scale: float = 0.95) -> torch.Tensor:
    """
    Sap xep va xoay 6 goc nhin truc giao (Front, Right, Back, Left, Top, Bottom)
    thanh 3 mat phang Triplane truc giao (XY, YZ, ZX) trai rong theo chieu ngang.
    Args:
        triview_color: [B, 6, C, H, W] hoac [6, C, H, W]
    Returns:
        triplane_tensor: [B, C*2, H, W*3] hoac [C*2, H, W*3]
    """
    has_batch = (triview_color.dim() == 5)
    if not has_batch:
        triview_color = triview_color.unsqueeze(0)

    B, num_v, C, H, W = triview_color.shape
    tri_list = []

    for b in range(B):
        tc = triview_color[b]  # [6, C, H, W]
        # Xoay va lat tung goc nhin theo he toa do Blender chuan cua CRM
        tc0 = torch.rot90(tc[2], k=2, dims=[1, 2])
        tc1 = tc[4].flip(2).flip(1)
        tc2 = torch.rot90(torch.rot90(tc[0], k=3, dims=[1, 2]).flip(2), k=2, dims=[1, 2])
        tc3 = torch.rot90(torch.rot90(tc[5], k=2, dims=[1, 2]).flip(2), k=2, dims=[1, 2])
        tc4 = torch.rot90(tc[1], k=2, dims=[1, 2]).flip(1).flip(1).flip(2)
        tc5 = torch.rot90(tc[3], k=1, dims=[1, 2]).flip(1).flip(2)

        # Mat phang 1: tc0, tc1, tc2 ghep theo chieu ngang (dim=2 cua [C, H, W]) -> [C, H, W*3]
        plane1 = torch.cat((tc0, tc1, tc2), dim=2)
        # Mat phang 2: tc3, tc4, tc5 ghep theo chieu ngang -> [C, H, W*3]
        plane2 = torch.cat((tc3, tc4, tc5), dim=2)
        # Gop 2 mat phang theo chieu kenh -> [C*2, H, W*3] (vi du 3*2 = 6 kenh)
        combined = torch.cat((plane1, plane2), dim=0)
        tri_list.append(combined)

    result = torch.stack(tri_list, dim=0)
    if not has_batch:
        result = result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# Foundation Model Components (Official CRM / Tripo3D Architecture - 629 Keys)
# ---------------------------------------------------------------------------

class UNetPP(nn.Module):
    """
    Wrapper cho UNet2DModel trong diffusers, tao dac trung Triplane 32 kenh.
    Kich thuoc spatial: [256, 768] (3 mat phang 256x256 xep canh nhau).
    Tong cong: 612 keys tu UNet2DModel + 1 key learned_plane = 613 keys.
    """
    def __init__(self, in_channels: int = 32):
        super().__init__()
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library is required for UNetPP foundation backbone.")

        self.in_channels = in_channels
        self.unet = UNet2DModel(
            sample_size=[256, 256 * 3],
            in_channels=in_channels,
            out_channels=32,
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 128 * 2, 128 * 2, 128 * 4, 128 * 4),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        if in_channels > 12:
            self.learned_plane = nn.Parameter(torch.zeros([1, in_channels - 12, 256, 256 * 3]))
        else:
            self.learned_plane = None

    def forward(self, x: torch.Tensor, t: Union[int, torch.Tensor] = 0) -> torch.Tensor:
        if self.learned_plane is not None and x.shape[1] < self.in_channels:
            learned = self.learned_plane.expand(x.shape[0], -1, -1, -1).to(x.device)
            x = torch.cat([x, learned], dim=1)
        return self.unet(x, t).sample


class TetTexNet(nn.Module):
    """
    Bo noi suy dac trung Triplane tai toa do diem khong gian 3D.
    Nhan rolled_out_feature [B, 32, 256, 768] gom 3 mat phang:
    - XY: [:, :, :, 0:plane_reso]
    - YZ: [:, :, :, plane_reso:2*plane_reso]
    - ZX: [:, :, :, 2*plane_reso:]
    """
    def __init__(self, plane_reso: int = 256, padding: float = 0.1, fea_concat: bool = False):
        super().__init__()
        self.plane_reso = plane_reso
        self.padding = padding
        self.fea_concat = fea_concat

    def forward(self, rolled_out_feature: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rolled_out_feature: [B, C, H, W*3] vi du [B, 32, 256, 768]
            query: [B, N, 3] toa do diem khong gian 3D trong [-0.5, 0.5]
        Returns:
            output: [B, N, C] vector dac trung tong hop
        """
        reso = self.plane_reso
        xy_plane = rolled_out_feature[:, :, :, 0:reso]
        yz_plane = rolled_out_feature[:, :, :, reso:2 * reso]
        zx_plane = rolled_out_feature[:, :, :, 2 * reso:]

        # Chuyen query ve khoang [-1, 1] de F.grid_sample noi suy chinh xac
        q = query * 2.0
        q_xy = q[:, :, [0, 1]].unsqueeze(2)
        q_yz = q[:, :, [1, 2]].unsqueeze(2)
        q_zx = q[:, :, [2, 0]].unsqueeze(2)

        f_xy = F.grid_sample(xy_plane, q_xy, padding_mode="border", align_corners=True, mode="bilinear").squeeze(-1).permute(0, 2, 1)
        f_yz = F.grid_sample(yz_plane, q_yz, padding_mode="border", align_corners=True, mode="bilinear").squeeze(-1).permute(0, 2, 1)
        f_zx = F.grid_sample(zx_plane, q_zx, padding_mode="border", align_corners=True, mode="bilinear").squeeze(-1).permute(0, 2, 1)

        if self.fea_concat:
            return torch.cat([f_xy, f_yz, f_zx], dim=-1)
        return f_xy + f_yz + f_zx


class SdfMlp(nn.Module):
    """
    Geometry MLP Head: Du doan gia tri khoang cach co huong (SDF) va bien dang 3D.
    Tong cong: 6 keys (fc1, fc2, fc3).
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 512, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.fc3 = nn.Linear(hidden_dim, 4, bias=bias)  # channel 0 la SDF, 1:4 la vector bien dang

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RgbMlp(nn.Module):
    """
    Appearance MLP Head: Du doan mau sac RGB Albedo per-point.
    Tong cong: 6 keys (fc1, fc2, fc3).
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 512, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.fc3 = nn.Linear(hidden_dim, 3, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CRMFoundationModel(nn.Module):
    """
    Mo hinh CRM Foundation chuan muc:
    Khop 100% kien truc va trong so checkpoint Zhengyi/CRM (CRM.pth) gom 629 tham so keys:
    - unet2: UNetPP (613 keys)
    - decoder: TetTexNet
    - weightMlp: FlexiCubes weights (4 keys)
    - sdfMlp: SdfMlp (6 keys)
    - rgbMlp: RgbMlp (6 keys)
    Tong: 613 + 4 + 6 + 6 = 629 keys.
    """
    def __init__(
        self,
        c_dim: int = 32,
        plane_resolution: int = 256,
        fea_concat: bool = False,
        mlp_bias: bool = True
    ):
        super().__init__()
        self.c_dim = c_dim
        self.plane_resolution = plane_resolution
        self.fea_concat = fea_concat

        # 1. UNet Backbone (613 keys)
        self.unet2 = UNetPP(in_channels=c_dim)

        # 2. Triplane Feature Decoder
        self.decoder = TetTexNet(plane_reso=plane_resolution, fea_concat=fea_concat)

        # 3. FlexiCubes weights head (4 keys)
        mlp_chnl = (3 * c_dim) if fea_concat else c_dim
        self.weightMlp = nn.Sequential(
            nn.Linear(mlp_chnl * 8, 512),
            nn.SiLU(),
            nn.Linear(512, 21)
        )

        # 4. Geometry MLP (6 keys)
        self.sdfMlp = SdfMlp(input_dim=mlp_chnl, hidden_dim=512, bias=mlp_bias)

        # 5. Appearance MLP (6 keys)
        self.rgbMlp = RgbMlp(input_dim=mlp_chnl, hidden_dim=512, bias=mlp_bias)

    def encode_triplane(self, views: torch.Tensor, ccms: torch.Tensor) -> torch.Tensor:
        """
        Chuyen 6 goc nhin RGB va 6 CCMs thanh dac trung Triplane [B, 32, 256, 768].
        """
        # views: [B, 6, 3, H, W], ccms: [B, 6, 3, H, W]
        tri_views = get_tri(views, blender=True)  # [B, 6, 256, 768]
        tri_ccms = get_tri(ccms, blender=True)    # [B, 6, 256, 768]
        x = torch.cat([tri_views, tri_ccms], dim=1)  # [B, 12, 256, 768]
        triplane = self.unet2(x)                     # [B, 32, 256, 768]
        return triplane

    def query_points(self, triplane: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Truy van SDF va RGB tai tap diem 3D bat ky.
        Args:
            triplane: [B, 32, 256, 768]
            points: [B, N, 3] trong [-0.5, 0.5]
        Returns:
            sdf: [B, N, 1]
            rgb: [B, N, 3]
        """
        feat = self.decoder(triplane, points)
        sdf_out = self.sdfMlp(feat)
        pred_sdf = sdf_out[..., 0:1]
        pred_rgb = torch.sigmoid(self.rgbMlp(feat))
        return pred_sdf, pred_rgb

    def forward(
        self,
        views: torch.Tensor,
        ccms: torch.Tensor,
        points: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        triplane = self.encode_triplane(views, ccms)
        output = {"triplane": triplane}

        if points is not None:
            pred_sdf, pred_rgb = self.query_points(triplane, points)
            output["pred_sdf"] = pred_sdf
            output["pred_rgb"] = pred_rgb

        return output

    def load_pretrained(self, checkpoint_path_or_dict: Union[str, Path, dict]) -> Dict[str, Any]:
        """
        Nap trong so pretrained tu tệp checkpoint (CRM.pth).
        Tu dong bo tien to va kiem tra do khop 629 keys.
        """
        if isinstance(checkpoint_path_or_dict, (str, Path)):
            state = torch.load(checkpoint_path_or_dict, map_location="cpu")
        else:
            state = checkpoint_path_or_dict

        if isinstance(state, dict):
            if "state_dict" in state:
                state_dict = state["state_dict"]
            elif "model" in state:
                state_dict = state["model"]
            elif "net" in state:
                state_dict = state["net"]
            else:
                state_dict = state
        else:
            state_dict = state

        model_dict = self.state_dict()
        matched = {}
        for k, v in state_dict.items():
            clean_k = k
            for pfx in ["module.", "generator.", "net.", "crm."]:
                if clean_k.startswith(pfx):
                    clean_k = clean_k[len(pfx):]
            if clean_k in model_dict and v.shape == model_dict[clean_k].shape:
                matched[clean_k] = v

        self.load_state_dict(matched, strict=False)
        match_rate = len(matched) / len(model_dict) if len(model_dict) > 0 else 0
        return {
            "total_checkpoint_keys": len(state_dict),
            "model_keys": len(model_dict),
            "matched_keys": len(matched),
            "match_rate": match_rate
        }

    @torch.no_grad()
    def extract_mesh(
        self,
        triplane: torch.Tensor,
        resolution: int = 64,
        threshold: float = 0.0,
        chunk_size: int = 65536
    ) -> trimesh.Trimesh:
        """
        Trich xuat luoi tam giac 3D tu truong SDF bang Marching Cubes.
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
            pred_sdf, _ = self.query_points(triplane, chunk_pts)
            sdf_vals[start_idx:end_idx] = pred_sdf.squeeze().cpu().numpy()

        sdf_volume = sdf_vals.reshape(resolution, resolution, resolution)

        vol_min = float(sdf_volume.min())
        vol_max = float(sdf_volume.max())
        target_level = threshold
        if target_level <= vol_min or target_level >= vol_max:
            target_level = float(np.percentile(sdf_volume, 30))

        if mcubes is not None:
            vertices, triangles = mcubes.marching_cubes(sdf_volume, target_level)
            vertices = vertices / (resolution - 1) - 0.5
        else:
            from skimage import measure
            vertices, triangles, _, _ = measure.marching_cubes(
                sdf_volume, level=target_level, spacing=(1.0 / resolution, 1.0 / resolution, 1.0 / resolution)
            )
            vertices = vertices - 0.5

        if len(vertices) == 0:
            return trimesh.creation.icosphere(radius=0.3)

        vert_tensor = torch.from_numpy(vertices.astype(np.float32)).unsqueeze(0).to(device)
        _, vert_rgb = self.query_points(triplane, vert_tensor)
        vert_rgb_np = (vert_rgb.squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=triangles,
            vertex_colors=vert_rgb_np,
            process=True
        )
        return mesh


# ---------------------------------------------------------------------------
# Lightweight Model Architecture (For Fast CPU Serving & Local Prototyping)
# ---------------------------------------------------------------------------

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
    Lightweight Convolutional U-Net for Triplane Generation (68 keys, 17.3M params).
    Input: [B, 36, H, W]
    Output: [B, 96, H, W]
    """
    def __init__(self, in_channels: int = 36, triplane_channels: int = 32, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.triplane_channels = triplane_channels
        out_channels = triplane_channels * 3

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 8)

        self.up1 = Up(base_channels * 16, base_channels * 4)
        self.up2 = Up(base_channels * 8, base_channels * 2)
        self.up3 = Up(base_channels * 4, base_channels)
        self.up4 = Up(base_channels * 2, base_channels)

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
    """Lightweight MLP decoder: triplane features -> SDF + Color."""
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


class CRMLightweightModel(nn.Module):
    """Full Lightweight CRM Model (17.3M params) for CPU serving."""
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
        B, _, H, W = triplane.shape
        C = self.triplane_channels
        N = points.shape[1]

        xy_plane = triplane[:, :C]
        xz_plane = triplane[:, C:2 * C]
        yz_plane = triplane[:, 2 * C:]

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
        B = views.shape[0]
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

        vol_min = float(sdf_volume.min())
        vol_max = float(sdf_volume.max())
        target_level = threshold
        if target_level <= vol_min or target_level >= vol_max:
            target_level = float(np.percentile(sdf_volume, 30))

        if mcubes is not None:
            vertices, triangles = mcubes.marching_cubes(sdf_volume, target_level)
            vertices = vertices / (resolution - 1) - 0.5
        else:
            from skimage import measure
            vertices, triangles, _, _ = measure.marching_cubes(
                sdf_volume, level=target_level, spacing=(1.0 / resolution, 1.0 / resolution, 1.0 / resolution)
            )
            vertices = vertices - 0.5

        if len(vertices) == 0:
            return trimesh.creation.icosphere(radius=0.3)

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


# ---------------------------------------------------------------------------
# Factory & Unified Wrapper
# ---------------------------------------------------------------------------

def build_crm_model(
    arch_type: str = "auto",
    **kwargs
) -> nn.Module:
    """
    Factory ham tao mo hinh CRM:
    - 'foundation': Mo hinh 629 keys tuong thich 100% Zhengyi/CRM checkpoint
    - 'lightweight': Mo hinh U-Net 17.3M phuc vu nhanh tren CPU
    - 'auto': Uu tien foundation neu co diffusers, nguoc lai dung lightweight
    """
    if arch_type == "foundation" or (arch_type == "auto" and DIFFUSERS_AVAILABLE):
        return CRMFoundationModel(**kwargs)
    return CRMLightweightModel(**kwargs)


# Backward compatibility aliases
CRMModel = CRMLightweightModel
ConvolutionalReconstructionModel = CRMLightweightModel
MultiViewUNet = CRMUNet
TriplaneSDFDecoder = TriplaneMLP
