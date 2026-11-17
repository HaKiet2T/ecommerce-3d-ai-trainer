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

def compute_winding_numbers(points: np.ndarray, triangles: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    """
    Tinh Generalized Winding Number (Jacobson et al., 2013) bang PyTorch/NumPy.
    Xac dinh chinh xac diem nam ben trong (wn >= 0.5) hay ben ngoai (wn < 0.5)
    ke ca khi luoi 3D non-watertight hoac co lo thung, khong phu thuoc vao thu vien C rtree.
    """
    N = len(points)
    wn = np.zeros(N, dtype=np.float32)
    tri = torch.from_numpy(triangles.copy()).float()

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        p = torch.from_numpy(points[i:end].copy()).float()

        # Vector tu diem truy van den 3 dinh tam giac
        a = tri[:, 0].unsqueeze(0) - p.unsqueeze(1)
        b = tri[:, 1].unsqueeze(0) - p.unsqueeze(1)
        c = tri[:, 2].unsqueeze(0) - p.unsqueeze(1)

        ra = torch.norm(a, dim=-1)
        rb = torch.norm(b, dim=-1)
        rc = torch.norm(c, dim=-1)

        cross_bc = torch.cross(b, c, dim=-1)
        num = torch.sum(a * cross_bc, dim=-1)

        dot_ab = torch.sum(a * b, dim=-1)
        dot_bc = torch.sum(b * c, dim=-1)
        dot_ca = torch.sum(c * a, dim=-1)
        den = ra * rb * rc + dot_ab * rc + dot_bc * ra + dot_ca * rb

        omega = 2.0 * torch.atan2(num, den)
        chunk_wn = torch.sum(omega, dim=-1) / (4.0 * np.pi)
        wn[i:end] = chunk_wn.numpy()

    return wn


def sample_mesh_surface_and_colors(mesh: trimesh.Trimesh, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lay mau diem tren be mat mesh va trich xuat mau sac UV chan thuc:
    1. Noi suy UV map (trimesh.visual.color.uv_to_color) tu texture CAD goc
    2. Fallback sang vertex_colors noi suy qua toa do trong tam barycentric
    3. Fallback sang face_colors
    Triet tieu hoan toan loi mau xam mac dinh [0.5, 0.5, 0.5].
    """
    surf_pts, face_indices, bary = trimesh.sample.sample_surface(
        mesh, num_points, return_barycentric=True
    )

    colors = None

    # 1. Trich xuat mau tu Texture Visuals (UV map va image)
    if (
        hasattr(mesh, "visual")
        and hasattr(mesh.visual, "uv")
        and mesh.visual.uv is not None
        and hasattr(mesh.visual, "material")
        and getattr(mesh.visual.material, "image", None) is not None
    ):
        try:
            tri_uvs = mesh.visual.uv[mesh.faces[face_indices]]
            interp_uv = np.sum(tri_uvs * bary[:, :, None], axis=1)
            rgba = trimesh.visual.color.uv_to_color(interp_uv, mesh.visual.material.image)
            colors = rgba[:, :3].astype(np.float32) / 255.0
        except Exception:
            colors = None

    # 2. Trich xuat mau tu vertex_colors
    if colors is None and hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) > 0:
        try:
            tri_colors = mesh.visual.vertex_colors[mesh.faces[face_indices]][:, :, :3].astype(np.float32) / 255.0
            colors = np.sum(tri_colors * bary[:, :, None], axis=1)
        except Exception:
            colors = None

    # 3. Trich xuat mau tu face_colors
    if colors is None and hasattr(mesh, "visual") and hasattr(mesh.visual, "face_colors") and len(mesh.visual.face_colors) > 0:
        try:
            colors = mesh.visual.face_colors[face_indices][:, :3].astype(np.float32) / 255.0
        except Exception:
            colors = None

    # 4. Fallback mau sang nhe
    if colors is None:
        colors = np.ones((len(surf_pts), 3), dtype=np.float32) * 0.8

    return surf_pts, face_indices, colors


def sample_sdf_points_from_mesh(
    mesh: trimesh.Trimesh,
    num_points: int = 16384,
    surface_ratio: float = 0.7,
    noise_std: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lay mau diem 3D va tinh Signed Distance Function (SDF) + mau sac:
    - SDF < 0 cho cac diem ben trong vat the
    - SDF = 0 cho cac diem tren be mat
    - SDF > 0 cho cac diem ben ngoai khong gian
    Su dung cKDTree ket hop Generalized Winding Number (0 dependency rtree).
    """
    n_surface = int(num_points * surface_ratio)
    n_uniform = num_points - n_surface

    # 1. Lay mau tren be mat mesh kem mau sac thuc te
    surf_pts, face_indices, surf_colors = sample_mesh_surface_and_colors(mesh, n_surface)
    noise = np.random.normal(scale=noise_std, size=surf_pts.shape).astype(np.float32)
    near_surf_pts = surf_pts + noise
    near_dist = np.linalg.norm(noise, axis=-1, keepdims=True).astype(np.float32)

    # Dau gan be mat theo phap tuyen
    if hasattr(mesh, "face_normals") and len(mesh.face_normals) > 0:
        normals = mesh.face_normals[face_indices]
        near_sign = np.sign(np.sum(noise * normals, axis=-1, keepdims=True))
        near_sign[near_sign == 0] = 1.0
        near_sdf = (near_dist * near_sign).astype(np.float32)
    else:
        near_sdf = near_dist
    near_colors = surf_colors.copy()

    # 2. Lay mau dong deu trong hop don vi [-0.5, 0.5]^3
    uniform_pts = np.random.uniform(low=-0.5, high=0.5, size=(n_uniform, 3)).astype(np.float32)

    # Khoang cach Euclid bang cKDTree
    tree = cKDTree(surf_pts)
    vol_dist, vol_idx = tree.query(uniform_pts)
    vol_colors = surf_colors[vol_idx]

    # Dau am/duong cho volume points qua Generalized Winding Number
    if len(mesh.faces) > 1500:
        try:
            proxy_mesh = mesh.simplify_quadratic_decimation(1500)
            if len(proxy_mesh.faces) == 0:
                proxy_mesh = mesh
        except Exception:
            proxy_mesh = mesh
    else:
        proxy_mesh = mesh

    vol_wn = compute_winding_numbers(uniform_pts, proxy_mesh.triangles)
    is_inside = vol_wn >= 0.5
    vol_sign = np.where(is_inside, -1.0, 1.0).astype(np.float32)
    vol_sdf = (vol_dist * vol_sign)[:, np.newaxis].astype(np.float32)

    # Gop toan bo diem, SDF va mau sac
    all_pts = np.concatenate([near_surf_pts, uniform_pts], axis=0).astype(np.float32)
    all_sdf = np.concatenate([near_sdf, vol_sdf], axis=0).astype(np.float32)
    all_colors = np.concatenate([near_colors, vol_colors], axis=0).astype(np.float32)

    return all_pts, all_sdf, all_colors

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
            if d.is_dir() and (
                (d / "views").exists()
                or (d / "front.png").exists()
                or (d / "view_0.png").exists()
                or (d / "input.png").exists()
            )
        ]

        if len(self.sample_dirs) == 0:
            # Thu quet de qui 1 tang
            self.sample_dirs = [
                d for d in self.data_root.glob("*/*")
                if d.is_dir() and (
                    (d / "views").exists()
                    or (d / "front.png").exists()
                    or (d / "view_0.png").exists()
                    or (d / "input.png").exists()
                )
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
        Ho tro ca ten goc nhin truc giao (front, right...) va chi so so hoa (view_0, view_1...).
        """
        view_idx = VIEW_NAMES.index(view_name) if view_name in VIEW_NAMES else None

        # Danh sach ung vien anh RGB
        candidates_img = [item_dir / f"{view_name}.png", item_dir / "views" / f"{view_name}.png"]
        if view_idx is not None:
            candidates_img.extend([item_dir / f"view_{view_idx}.png", item_dir / "views" / f"view_{view_idx}.png"])

        img_path = None
        for p in candidates_img:
            if p.exists():
                img_path = p
                break

        # Danh sach ung vien anh CCM
        candidates_ccm = [item_dir / f"{view_name}_ccm.png", item_dir / "views" / f"{view_name}_ccm.png"]
        if view_idx is not None:
            candidates_ccm.extend([
                item_dir / f"ccm_{view_idx}.png",
                item_dir / "views" / f"ccm_{view_idx}.png",
                item_dir / f"ccm_{view_name}.png"
            ])

        ccm_path = None
        for p in candidates_ccm:
            if p.exists():
                ccm_path = p
                break

        # Danh sach ung vien anh Mask
        candidates_mask = [item_dir / f"{view_name}_mask.png", item_dir / "views" / f"{view_name}_mask.png"]
        if view_idx is not None:
            candidates_mask.extend([
                item_dir / f"mask_{view_idx}.png",
                item_dir / "views" / f"mask_{view_idx}.png",
                item_dir / f"mask_{view_name}.png"
            ])

        mask_path = None
        for p in candidates_mask:
            if p.exists():
                mask_path = p
                break

        if img_path and img_path.exists():
            raw_img = Image.open(img_path).convert("RGBA")
            # Tach mask tu alpha channel hoac file mask rieng
            alpha_np = np.array(raw_img)[:, :, 3] / 255.0
            mask_np = (alpha_np > 0.1).astype(np.float32)
            rgb_pil = Image.fromarray(np.array(raw_img)[:, :, :3])
        else:
            # Fallback mau xam neu thieu goc render
            rgb_pil = Image.new("RGB", (self.image_size, self.image_size), (127, 127, 127))
            mask_np = np.ones((self.image_size, self.image_size), dtype=np.float32)

        if mask_path and mask_path.exists():
            mask_pil = Image.open(mask_path).convert("L")
            mask_np = (np.array(mask_pil) > 127).astype(np.float32)
        else:
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

        # Xu ly CCM (neu khong co file rieng thi tong hop truc tiep tu toa do chuan)
        if ccm_path and ccm_path.exists():
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

    def _load_sdf_points(self, item_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Nap hoac tao diem lay mau 3D kem gia tri SDF chuan va mau sac thuc te.
        """
        # 1. Kiem tra dinh dang npy tach roi tu Kaggle data prep
        points_npy = item_dir / "points.npy"
        sdf_npy = item_dir / "sdf.npy"
        colors_npy = item_dir / "colors.npy"
        if points_npy.exists() and sdf_npy.exists():
            pts = np.load(points_npy).astype(np.float32)
            sdfs = np.load(sdf_npy).astype(np.float32)
            if colors_npy.exists():
                cols = np.load(colors_npy).astype(np.float32)
            else:
                cols = np.ones((len(pts), 3), dtype=np.float32) * 0.8

            if len(pts) >= self.num_points:
                indices = np.random.choice(len(pts), self.num_points, replace=False)
                return torch.from_numpy(pts[indices]), torch.from_numpy(sdfs[indices]), torch.from_numpy(cols[indices])
            return torch.from_numpy(pts), torch.from_numpy(sdfs), torch.from_numpy(cols)

        # 2. Kiem tra file npz tong hop
        npz_path = item_dir / "sdf_samples.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            pts = data["points"].astype(np.float32)
            sdfs = data["sdf"].astype(np.float32)
            cols = data["colors"].astype(np.float32) if "colors" in data else np.ones((len(pts), 3), dtype=np.float32) * 0.8
            if len(pts) >= self.num_points:
                indices = np.random.choice(len(pts), self.num_points, replace=False)
                return torch.from_numpy(pts[indices]), torch.from_numpy(sdfs[indices]), torch.from_numpy(cols[indices])
            return torch.from_numpy(pts), torch.from_numpy(sdfs), torch.from_numpy(cols)

        # 3. Neu chua co du lieu diem tien xu ly, tim file mesh 3D de tinh toan online
        mesh_candidates = list(item_dir.glob("*.obj")) + list(item_dir.glob("*.glb"))
        if mesh_candidates:
            mesh = trimesh.load(mesh_candidates[0], force="mesh")
            mesh = normalize_mesh(mesh)
            pts, sdfs, cols = sample_sdf_points_from_mesh(mesh, num_points=self.num_points)
            return torch.from_numpy(pts), torch.from_numpy(sdfs), torch.from_numpy(cols)

        # 4. Fallback mac dinh: lay mau hinh cau don vi r=0.35 lam placeholder
        pts = np.random.uniform(-0.5, 0.5, size=(self.num_points, 3)).astype(np.float32)
        radius = np.linalg.norm(pts, axis=-1, keepdims=True)
        sdfs = (radius - 0.35).astype(np.float32)
        cols = np.ones((self.num_points, 3), dtype=np.float32) * 0.8
        return torch.from_numpy(pts), torch.from_numpy(sdfs), torch.from_numpy(cols)

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

        pts_tensor, sdf_tensor, colors_tensor = self._load_sdf_points(item_dir)

        return {
            "images": images_tensor,
            "ccms": ccms_tensor,
            "masks": masks_tensor,
            "points": pts_tensor,      # [N, 3]
            "sdf": sdf_tensor,          # [N, 1]
            "colors": colors_tensor,    # [N, 3]
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
