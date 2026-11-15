import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import trimesh
from scipy.spatial import cKDTree

# 6 goc nhin truc giao chuan (Orthographic Views) cua CRM
VIEW_NAMES = ["front", "right", "back", "left", "top", "bottom"]

def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Chuan hoa mesh ve tam (0, 0, 0) va nam tron trong hop don vi [-0.5, 0.5]^3.
    Giup dam bao toa do dong nhat cho FlexiCubes va Triplane.
    """
    vertices = mesh.vertices - mesh.bounding_box.centroid
    max_extent = np.max(mesh.bounding_box.extents)
    if max_extent > 0:
        vertices = vertices / (max_extent * 1.05) # Giu le 5% de tranh cham bien
    normalized_mesh = mesh.copy()
    normalized_mesh.vertices = vertices
    return normalized_mesh

def sample_sdf_points_from_mesh(
    mesh: trimesh.Trimesh,
    num_points: int = 16384,
    surface_ratio: float = 0.7,
    noise_std: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lay mau diem 3D va tinh SDF (Signed Distance Function) tu mesh:
    - 70% mau quanh be mat (surface points + Gaussian noise) de hoc chi tiet hinh hoc sac net
    - 30% mau dong deu trong khong gian hop [-0.5, 0.5]^3 de hoc vung rong ngoai vat the
    """
    n_surface = int(num_points * surface_ratio)
    n_uniform = num_points - n_surface

    # 1. Lay mau tren be mat mesh
    surf_pts, face_indices = trimesh.sample.sample_surface(mesh, n_surface)
    noise = np.random.normal(scale=noise_std, size=surf_pts.shape)
    near_surf_pts = surf_pts + noise

    # 2. Lay mau dong deu trong hop don vi [-0.5, 0.5]^3
    uniform_pts = np.random.uniform(low=-0.5, high=0.5, size=(n_uniform, 3))

    all_pts = np.concatenate([near_surf_pts, uniform_pts], axis=0).astype(np.float32)

    # 3. Tinh khoang cach co dau (Signed Distance Function)
    # Dung Ray-casting va Nearest Point Query cua trimesh
    proximity_query = trimesh.proximity.ProximityQuery(mesh)
    distances = proximity_query.signed_distance(all_pts)
    
    # SDF trimesh quy uoc: duong la ben trong, am la ben ngoai
    # Ta dao dau de theo quy uoc chuan Computer Graphics: am la ben trong, duong la ben ngoai
    sdf_values = -distances.astype(np.float32)[:, np.newaxis]

    return all_pts, sdf_values

def generate_ccm_from_rgb(rgb_img: Image.Image, mask: np.ndarray, view_name: str) -> np.ndarray:
    """
    Tao Canonical Coordinate Map (CCM) tong hop tu toa do truc giao 
    khi khong co render toa do 3D truc tiep.
    Toa do nam trong dai [0, 1]^3 tuong ung (R, G, B).
    """
    w, h = rgb_img.size
    u = np.linspace(0.0, 1.0, w)
    v = np.linspace(0.0, 1.0, h)
    uu, vv = np.meshgrid(u, v)

    # Khoi tao toa do khong gian 3D [0, 1] cho moi goc nhin truc giao
    if view_name == "front":
        x = uu
        y = 1.0 - vv
        z = np.ones_like(uu) * 0.5
    elif view_name == "back":
        x = 1.0 - uu
        y = 1.0 - vv
        z = np.ones_like(uu) * 0.5
    elif view_name == "left":
        x = np.ones_like(uu) * 0.5
        y = 1.0 - vv
        z = 1.0 - uu
    elif view_name == "right":
        x = np.ones_like(uu) * 0.5
        y = 1.0 - vv
        z = uu
    elif view_name == "top":
        x = uu
        y = np.ones_like(uu) * 0.5
        z = vv
    elif view_name == "bottom":
        x = uu
        y = np.ones_like(uu) * 0.5
        z = 1.0 - vv
    else:
        x, y, z = uu, 1.0 - vv, np.ones_like(uu) * 0.5

    ccm = np.stack([x, y, z], axis=-1).astype(np.float32)
    # Vung background ngoai mask duoc gan bang 0
    ccm = ccm * mask[:, :, np.newaxis]
    return ccm

class CRMDataset(Dataset):
    """
    PyTorch Dataset nap du lieu huan luyen CRM:
    - 6 anh goc nhin truc giao (RGB) kich thuoc [6, 3, H, W]
    - 6 Canonical Coordinate Maps (CCM) kich thuoc [6, 3, H, W]
    - 6 Mask vat the kich thuoc [6, 1, H, W]
    - Tap diem lay mau 3D kem gia tri SDF chuan [N, 3] va [N, 1]
    """
    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        num_points: int = 16384,
        is_training: bool = True,
        preload_mesh: bool = False
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.num_points = num_points
        self.is_training = is_training
        self.preload_mesh = preload_mesh

        # Quet danh sach mau trong data_root
        # Moi thu muc con la mot san pham (desk, chair, monitor, etc.)
        self.sample_dirs = [
            d for d in self.data_root.iterdir()
            if d.is_dir() and (d / "views").exists() or (d / "front.png").exists()
        ]

        if len(self.sample_dirs) == 0:
            # Thu quet de qui 1 tang
            self.sample_dirs = [
                d for d in self.data_root.glob("*/*")
                if d.is_dir() and ((d / "views").exists() or (d / "front.png").exists())
            ]

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def _load_view_image(self, item_dir: Path, view_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Doc mot goc nhin: anh RGB, anh CCM, va Mask nhi phan.
        """
        # Thu muc co the chua truc tiep hoac trong subfolder 'views'
        img_path = item_dir / f"{view_name}.png"
        if not img_path.exists():
            img_path = item_dir / "views" / f"{view_name}.png"

        ccm_path = item_dir / f"{view_name}_ccm.png"
        if not ccm_path.exists():
            ccm_path = item_dir / "views" / f"{view_name}_ccm.png"

        mask_path = item_dir / f"{view_name}_mask.png"
        if not mask_path.exists():
            mask_path = item_dir / "views" / f"{view_name}_mask.png"

        if img_path.exists():
            raw_img = Image.open(img_path).convert("RGBA")
            # Tach mask tu alpha channel hoac file mask rieng
            alpha_np = np.array(raw_img)[:, :, 3] / 255.0
            mask_np = (alpha_np > 0.1).astype(np.float32)
            rgb_pil = Image.fromarray(np.array(raw_img)[:, :, :3])
        else:
            # Fallback mau xam neu thieu goc render
            rgb_pil = Image.new("RGB", (self.image_size, self.image_size), (127, 127, 127))
            mask_np = np.ones((self.image_size, self.image_size), dtype=np.float32)

        if mask_path.exists():
            mask_pil = Image.open(mask_path).convert("L")
            mask_np = (np.array(mask_pil) > 127).astype(np.float32)
        else:
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

        # Xu ly CCM (neu khong co file rieng thi tong hop truc tiep tu toa do chuan)
        if ccm_path.exists():
            ccm_pil = Image.open(ccm_path).convert("RGB")
            ccm_tensor = self.transform(ccm_pil)
        else:
            ccm_np = generate_ccm_from_rgb(rgb_pil, mask_np, view_name)
            ccm_tensor = torch.from_numpy(ccm_np).permute(2, 0, 1).float()
            # Chuyen ve khoang [-1, 1] giong transform
            ccm_tensor = ccm_tensor * 2.0 - 1.0

        rgb_tensor = self.transform(rgb_pil)
        mask_tensor = self.mask_transform(mask_pil)

        return rgb_tensor, ccm_tensor, mask_tensor

    def _load_sdf_points(self, item_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Nap hoac tao diem lay mau 3D kem gia tri SDF chuan.
        """
        npz_path = item_dir / "sdf_samples.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            pts = data["points"].astype(np.float32)
            sdfs = data["sdf"].astype(np.float32)
            if len(pts) >= self.num_points:
                indices = np.random.choice(len(pts), self.num_points, replace=False)
                return torch.from_numpy(pts[indices]), torch.from_numpy(sdfs[indices])

        # Neu chua co npz, tim file mesh 3D (.obj / .glb) de tinh online
        mesh_candidates = list(item_dir.glob("*.obj")) + list(item_dir.glob("*.glb"))
        if mesh_candidates:
            mesh = trimesh.load(mesh_candidates[0], force="mesh")
            mesh = normalize_mesh(mesh)
            pts, sdfs = sample_sdf_points_from_mesh(mesh, num_points=self.num_points)
            return torch.from_numpy(pts), torch.from_numpy(sdfs)

        # Fallback mac dinh: lay mau hinh cau don vi r=0.35 lam placeholder
        pts = np.random.uniform(-0.5, 0.5, size=(self.num_points, 3)).astype(np.float32)
        radius = np.linalg.norm(pts, axis=-1, keepdims=True)
        sdfs = (radius - 0.35).astype(np.float32)
        return torch.from_numpy(pts), torch.from_numpy(sdfs)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        item_dir = self.sample_dirs[idx]

        rgb_list, ccm_list, mask_list = [], [], []
        for view in VIEW_NAMES:
            rgb, ccm, mask = self._load_view_image(item_dir, view)
            rgb_list.append(rgb)
            ccm_list.append(ccm)
            mask_list.append(mask)

        # Ghep 6 goc nhin theo chieu [6, C, H, W]
        images_tensor = torch.stack(rgb_list, dim=0)   # [6, 3, H, W]
        ccms_tensor = torch.stack(ccm_list, dim=0)       # [6, 3, H, W]
        masks_tensor = torch.stack(mask_list, dim=0)     # [6, 1, H, W]

        pts_tensor, sdf_tensor = self._load_sdf_points(item_dir)

        return {
            "images": images_tensor,
            "ccms": ccms_tensor,
            "masks": masks_tensor,
            "points": pts_tensor,      # [N, 3]
            "sdf": sdf_tensor,          # [N, 1]
            "sample_id": item_dir.name
        }

def build_dataloader(
    data_root: str,
    batch_size: int = 2,
    num_workers: int = 2,
    is_training: bool = True
) -> DataLoader:
    """
    Khoi tao DataLoader cho huan luyen CRM voi che do pin_memory toi uu tren GPU.
    """
    dataset = CRMDataset(data_root=data_root, is_training=is_training)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_training
    )
    return loader
