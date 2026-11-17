"""
CRM 3D Inference Engine & Production Serving Pipeline
=====================================================
Modular inference engine delivering high-fidelity 3D meshes (GLB/OBJ) from single
e-commerce product photos.

Key architectural features:
1. Two-stage pipeline integration:
   - Stage 1: Multi-View Diffusion (Zero123++ with LoRA adapter) when GPU/diffusers available.
   - Fast Geometric Multi-View Synthesizer: viewpoint-aware profile & reflection projection (fallback).
2. Stage 2: Convolutional Reconstruction Model (CRM):
   - 36-channel input (6 RGB views + 6 Canonical Coordinate Maps CCMs).
   - U-Net Triplane generator (XY, XZ, YZ planes, 32 channels each).
   - Marching Cubes iso-surface extraction with adaptive zero-crossing calibration.
3. Multi-View Normal-Weighted Texture Baking:
   - Eliminates monochrome artifacts by projecting authentic colors from 6 viewpoints
     directly onto decimated 3D mesh vertices weighted by surface normals.
4. Quadric decimation & automated validation for instant Three.js / Next.js storefront rendering.
"""

import io
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageOps, ImageEnhance

try:
    import rembg
except ImportError:
    rembg = None

from src.data.dataset import generate_ccm_from_rgb, VIEW_NAMES
from src.models.crm import CRMModel, ConvolutionalReconstructionModel

# Camera directional vectors corresponding to VIEW_NAMES:
# ["front", "right", "back", "left", "top", "bottom"]
VIEW_CAMERA_DIRECTIONS = {
    "front": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "back": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    "left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "top": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "bottom": np.array([0.0, -1.0, 0.0], dtype=np.float32),
}


# ---------------------------------------------------------------------------
# 1. Product Image Preprocessor
# ---------------------------------------------------------------------------

def preprocess_product_image(
    image_bytes: bytes,
    target_size: int = 256,
    remove_background: bool = True,
    padding_ratio: float = 0.10
) -> Tuple[Image.Image, np.ndarray]:
    """
    Tien xu ly anh san pham dau vao:
    1. Tach phong nen bang rembg hoac thuat toan phan doan mau bien.
    2. Trich xuat mat na vat the (Foreground Mask).
    3. Can giua hop bao vat the trong khung vuong voi bien dem 10% (tranh cham bien).
    4. Tra ve anh RGB va mask float32 chuan hoa [0, 1].
    """
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if remove_background and rembg is not None:
        try:
            rgba = rembg.remove(input_image)
            alpha = np.array(rgba)[:, :, 3].astype(np.float32) / 255.0
            mask_raw = (alpha > 0.15).astype(np.float32)
            rgb_raw = rgba.convert("RGB")
        except Exception:
            rgb_raw, mask_raw = _fallback_background_segmentation(input_image)
    else:
        rgb_raw, mask_raw = _fallback_background_segmentation(input_image)

    # Tim bounding box cua vat the
    rows = np.any(mask_raw > 0.5, axis=1)
    cols = np.any(mask_raw > 0.5, axis=0)

    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bbox_w = xmax - xmin + 1
        bbox_h = ymax - ymin + 1
        max_dim = max(bbox_w, bbox_h)

        # Tao canvas vuong co padding
        pad = int(max_dim * padding_ratio)
        canvas_dim = max_dim + 2 * pad
        square_rgb = Image.new("RGB", (canvas_dim, canvas_dim), color=(255, 255, 255))
        square_mask = Image.new("L", (canvas_dim, canvas_dim), color=0)

        crop_rgb = rgb_raw.crop((xmin, ymin, xmax + 1, ymax + 1))
        crop_mask = Image.fromarray((mask_raw[ymin:ymax+1, xmin:xmax+1] * 255).astype(np.uint8))

        offset_x = (canvas_dim - bbox_w) // 2
        offset_y = (canvas_dim - bbox_h) // 2
        square_rgb.paste(crop_rgb, (offset_x, offset_y))
        square_mask.paste(crop_mask, (offset_x, offset_y))

        final_rgb = square_rgb.resize((target_size, target_size), Image.Resampling.LANCZOS)
        final_mask_img = square_mask.resize((target_size, target_size), Image.Resampling.NEAREST)
        final_mask = (np.array(final_mask_img).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    else:
        final_rgb = rgb_raw.resize((target_size, target_size), Image.Resampling.LANCZOS)
        final_mask = np.ones((target_size, target_size), dtype=np.float32)

    return final_rgb, final_mask


def _fallback_background_segmentation(img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    """Phan doan phong nen bang mau goc va nguong tuong phan khi chua co rembg."""
    img_np = np.array(img).astype(np.float32) / 255.0
    # Kiem tra mau 4 goc de uoc luong mau phong nen
    corners = np.concatenate([
        img_np[:10, :10].reshape(-1, 3),
        img_np[:10, -10:].reshape(-1, 3),
        img_np[-10:, :10].reshape(-1, 3),
        img_np[-10:, -10:].reshape(-1, 3)
    ], axis=0)
    bg_color = np.median(corners, axis=0)

    # Khoang cach mau Euclid toi phong nen
    color_dist = np.linalg.norm(img_np - bg_color, axis=-1)
    mask = (color_dist > 0.12).astype(np.float32)
    return img, mask


# ---------------------------------------------------------------------------
# 2. Stage 1 Multi-View Synthesizer (Diffusion Bridge & Geometric Fallback)
# ---------------------------------------------------------------------------

class MultiViewSynthesizer:
    """
    Dieu phoi sinh 6 goc nhin truc giao chuan (Front, Right, Back, Left, Top, Bottom):
    - Che do Diffusion: Ket noi Stage 1 (Zero123++ LoRA) khi co GPU va thu vien diffusers.
    - Che do Geometric Synthesis: Bien doi hinh hoc da chieu (Profile, Reflection, Plan-view)
      thay vi nhan ban nguyen xi 1 anh, giup tranh xung dot triplane va chong bien dang.
    """

    def __init__(self, lora_weights_path: Optional[str] = None, enable_diffusion: bool = False):
        self.enable_diffusion = enable_diffusion
        self.lora_weights_path = lora_weights_path or "checkpoints/stage1_lora"
        self._stage1_pipeline = None

    def _get_diffusion_pipeline(self):
        if self._stage1_pipeline is None and self.enable_diffusion:
            try:
                from src.pipeline.inference_stage1 import Zero123PlusLoRAInference
                if torch.cuda.is_available():
                    print("[Stage 1] Khoi tao Zero123++ LoRA Multi-View Diffusion...")
                    self._stage1_pipeline = Zero123PlusLoRAInference(
                        lora_weights_path=self.lora_weights_path if os.path.exists(self.lora_weights_path) else None,
                        device="cuda"
                    )
                else:
                    print("[Stage 1] GPU khong kha dung. Chuyen ve bo sinh hinh hoc Geometric Synthesizer.")
            except Exception as e:
                print(f"[Stage 1] Khong the nap Zero123++ pipeline ({e}). Su dung Geometric Synthesizer.")
        return self._stage1_pipeline

    def synthesize_views(
        self,
        rgb_img: Image.Image,
        mask: np.ndarray,
        use_diffusion: bool = False
    ) -> List[Image.Image]:
        """
        Sinh 6 goc nhin PIL: [Front, Right, Back, Left, Top, Bottom].
        """
        pipe = self._get_diffusion_pipeline() if use_diffusion else None

        if pipe is not None:
            try:
                res = pipe.generate_views(rgb_img, num_inference_steps=30, guidance_scale=4.0)
                print(f"[Stage 1] Sinh 6 goc nhin Zero123++ thanh cong trong {res['inference_time_sec']}s")
                return res["views"]
            except Exception as e:
                print(f"[Stage 1] Loi suy luan Zero123++ ({e}), fallback sang Geometric Synthesizer.")

        # Geometric Synthesis Mode (Dam bao moi goc nhin co ban sac rieng, khong trung lap)
        views = []
        target_size = rgb_img.size[0]

        # 0. Front View
        views.append(rgb_img)

        # 1. Right View: Mo phong goc nhin canh (Profile) bang bien doi ti le ngang va do toi
        right_w = max(int(target_size * 0.75), 32)
        right_scaled = rgb_img.resize((right_w, target_size), Image.Resampling.LANCZOS)
        right_canvas = Image.new("RGB", (target_size, target_size), color=(255, 255, 255))
        right_canvas.paste(right_scaled, ((target_size - right_w) // 2, 0))
        # Tang nhe tuong phan mo phong do bong goc nghieng
        enhancer = ImageEnhance.Brightness(right_canvas)
        views.append(enhancer.enhance(0.92))

        # 2. Back View: Lat ngang Mat truoc va giam do sang (Ambient Occlusion o mat sau)
        back_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
        back_enhancer = ImageEnhance.Brightness(back_img)
        views.append(back_enhancer.enhance(0.85))

        # 3. Left View: Lat ngang goc Right voi do sang doi lap
        left_canvas = right_canvas.transpose(Image.FLIP_LEFT_RIGHT)
        left_enhancer = ImageEnhance.Brightness(left_canvas)
        views.append(left_enhancer.enhance(0.90))

        # 4. Top View: Mo phong goc nhin tu tren xuong (Plan-view) bang nen chieu doc
        top_h = max(int(target_size * 0.70), 32)
        top_scaled = rgb_img.resize((target_size, top_h), Image.Resampling.LANCZOS)
        top_canvas = Image.new("RGB", (target_size, target_size), color=(255, 255, 255))
        top_canvas.paste(top_scaled, (0, (target_size - top_h) // 2))
        top_enhancer = ImageEnhance.Brightness(top_canvas)
        views.append(top_enhancer.enhance(0.95))

        # 5. Bottom View: Mat day (Base) voi do bong tiep xuc day
        bottom_h = max(int(target_size * 0.65), 32)
        bottom_scaled = rgb_img.resize((target_size, bottom_h), Image.Resampling.LANCZOS)
        bottom_canvas = Image.new("RGB", (target_size, target_size), color=(255, 255, 255))
        bottom_canvas.paste(bottom_scaled, (0, (target_size - bottom_h) // 2))
        bottom_enhancer = ImageEnhance.Brightness(bottom_canvas)
        views.append(bottom_enhancer.enhance(0.78))

        return views


def prepare_tensors_from_views(
    views_pil: List[Image.Image],
    mask: np.ndarray,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chuyen doi 6 anh PIL va mask sang tensor dinh dang Float32:
    - images: [1, 6, 3, 256, 256] chuan hoa [-1, 1]
    - ccms:   [1, 6, 3, 256, 256] chuan hoa [-1, 1]
    """
    images_list, ccms_list = [], []

    for i, view_name in enumerate(VIEW_NAMES):
        img_np = np.array(views_pil[i]).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1) * 2.0 - 1.0
        images_list.append(img_t)

        ccm_np = generate_ccm_from_rgb(views_pil[i], mask, view_name).astype(np.float32)
        ccm_t = torch.from_numpy(ccm_np).permute(2, 0, 1) * 2.0 - 1.0
        ccms_list.append(ccm_t)

    images = torch.stack(images_list, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)
    ccms = torch.stack(ccms_list, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)
    return images, ccms


def synthesize_multiview_tensors(
    rgb: Image.Image,
    mask: np.ndarray,
    device: torch.device,
    target_size: int = 256,
    synthesizer: Optional[MultiViewSynthesizer] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chuan bi tensor 6 goc nhin truc giao tu anh san pham duy nhat (Backward Compatibility).
    Su dung MultiViewSynthesizer da goc nhin de tao 6 view dac trung tranh bien dang.
    """
    if synthesizer is None:
        synthesizer = MultiViewSynthesizer()
    views_pil = synthesizer.synthesize_views(rgb, mask)
    return prepare_tensors_from_views(views_pil, mask, device)


# ---------------------------------------------------------------------------
# 3. Multi-View Normal-Weighted Texture Baking
# ---------------------------------------------------------------------------

def bake_multiview_vertex_colors(
    mesh: trimesh.Trimesh,
    views_pil: List[Image.Image],
    exponent: float = 2.0
) -> trimesh.Trimesh:
    """
    Bake mau sac truc tiep tu 6 goc nhin san pham len cac dinh cua luoi 3D:
    - Chieu toa do dinh (x, y, z) len tung mat phang camera truc giao.
    - Tinh trong so dong gop cua tung goc dua tren tich vo huong phap tuyen (Normal Dot Product).
    - Triet tieu hoan toan van de mesh bi toan mau xam do du lieu huan luyen thieu color label.
    """
    verts = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals.astype(np.float32)
    num_verts = len(verts)

    if num_verts == 0:
        return mesh

    def project_uv(v: np.ndarray, view: str) -> Tuple[np.ndarray, np.ndarray]:
        if view == "front":
            return v[:, 0] + 0.5, 0.5 - v[:, 1]
        elif view == "back":
            return 0.5 - v[:, 0], 0.5 - v[:, 1]
        elif view == "right":
            return v[:, 2] + 0.5, 0.5 - v[:, 1]
        elif view == "left":
            return 0.5 - v[:, 2], 0.5 - v[:, 1]
        elif view == "top":
            return v[:, 0] + 0.5, v[:, 2] + 0.5
        elif view == "bottom":
            return v[:, 0] + 0.5, 0.5 - v[:, 2]
        return v[:, 0] + 0.5, 0.5 - v[:, 1]

    blended_rgb = np.zeros((num_verts, 3), dtype=np.float32)
    sum_weights = np.zeros(num_verts, dtype=np.float32)

    for i, name in enumerate(VIEW_NAMES):
        cam_dir = VIEW_CAMERA_DIRECTIONS[name]
        # Cosine goc giua phap tuyen mat va huong camera
        dot = np.maximum(0.0, np.sum(normals * cam_dir, axis=-1))
        weight = dot ** exponent

        u, v = project_uv(verts, name)
        u_px = np.clip((u * 255.0).astype(np.int32), 0, 255)
        v_px = np.clip((v * 255.0).astype(np.int32), 0, 255)

        img_arr = np.array(views_pil[i].convert("RGB")).astype(np.float32)
        sample_rgb = img_arr[v_px, u_px] # [N, 3]

        blended_rgb += sample_rgb * weight[:, None]
        sum_weights += weight

    valid_mask = sum_weights > 1e-5
    blended_rgb[valid_mask] /= sum_weights[valid_mask, None]
    blended_rgb[~valid_mask] = 127.0 # Mau fallback cho dinh khong nhin thay ro

    vertex_colors = np.clip(blended_rgb, 0, 255).astype(np.uint8)
    vertex_colors_rgba = np.concatenate([
        vertex_colors,
        np.ones((num_verts, 1), dtype=np.uint8) * 255
    ], axis=-1)

    mesh.visual.vertex_colors = vertex_colors_rgba
    return mesh


# ---------------------------------------------------------------------------
# 4. CRM Serving Engine (Singleton Microservice Manager)
# ---------------------------------------------------------------------------

class CRMInferenceEngine:
    """Singleton inference engine managing model lifecycle and high-precision 3D reconstruction."""

    _instance: Optional["CRMInferenceEngine"] = None

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device_name: Optional[str] = None,
        enable_diffusion: bool = False
    ):
        self.device = torch.device(
            device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = checkpoint_path or os.environ.get("CRM_CHECKPOINT", "checkpoints/crm_best.pt")
        self.model = self._load_model()
        self.synthesizer = MultiViewSynthesizer(
            lora_weights_path="checkpoints/stage1_lora",
            enable_diffusion=enable_diffusion
        )

    @classmethod
    def get_instance(
        cls,
        checkpoint_path: Optional[str] = None,
        enable_diffusion: bool = False
    ) -> "CRMInferenceEngine":
        if cls._instance is None:
            cls._instance = cls(checkpoint_path=checkpoint_path, enable_diffusion=enable_diffusion)
        return cls._instance

    def _load_model(self) -> CRMModel:
        print(f"[Serving] Dang khoi tao mo hinh CRM tren thiet bi: {self.device}...")
        model = CRMModel(
            triplane_resolution=256,
            triplane_channels=32,
            base_channels=64,
            hidden_dim=64
        ).to(self.device)

        if os.path.exists(self.checkpoint_path):
            print(f"[Serving] Nap trong so CRM tu: {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print("[Serving] Mo hinh CRM da san sang phuc vu!")
        else:
            print(f"[Serving] Canh bao: Khong tim thay '{self.checkpoint_path}'. Chay o che do weights khoi tao.")

        model.eval()
        return model

    def generate_mesh(
        self,
        image_bytes: bytes,
        resolution: int = 64,
        target_faces: int = 25000,
        remove_background: bool = True,
        use_diffusion: bool = False,
        bake_texture: bool = True
    ) -> Tuple[trimesh.Trimesh, float, Dict[str, Union[int, float, bool]]]:
        """
        Quy trinh suy luan 3D dau-cuoi hoan chinh:
        1. Preprocessing: Xu ly phong nen va can giua vat the.
        2. Multi-view Synthesis: Sinh 6 goc nhin truc giao.
        3. Triplane Inference: Mạng U-Net sinh 3 mat phang dac trung [XY, XZ, YZ].
        4. Marching Cubes: Trich xuat luoi da giac 3D iso-surface SDF = 0.
        5. Quadric Decimation: Rut gon so mat de tai nhanh tren storefront Next.js.
        6. Texture Baking: Ban mau sac that tu anh san pham len be mat luoi 3D.
        """
        start_time = time.time()

        # 1. Preprocessing
        rgb_img, mask = preprocess_product_image(
            image_bytes,
            target_size=256,
            remove_background=remove_background
        )

        # 2. Multi-view Synthesis
        views_pil = self.synthesizer.synthesize_views(
            rgb_img,
            mask,
            use_diffusion=use_diffusion
        )

        # 3. Chuan bi tensors
        images_tensor, ccms_tensor = prepare_tensors_from_views(
            views_pil,
            mask,
            device=self.device
        )

        # 4. Model inference & Marching Cubes
        with torch.no_grad():
            out = self.model(images_tensor, ccms_tensor)
            triplane = out["triplane"]
            raw_mesh = self.model.extract_mesh(triplane, resolution=resolution)

        # 5. Decimation / Simplification for Web
        final_mesh = raw_mesh
        if target_faces > 0 and len(raw_mesh.faces) > target_faces:
            try:
                final_mesh = raw_mesh.simplify_quadric_decimation(face_count=target_faces)
            except Exception as e:
                print(f"[Decimation] Bo qua decimation do: {e}")
                final_mesh = raw_mesh

        # 6. Multi-View Normal-Weighted Texture Baking (Bake SAU khi decimate de giu 100% mau)
        if bake_texture:
            final_mesh = bake_multiview_vertex_colors(final_mesh, views_pil)

        elapsed = time.time() - start_time

        metadata = {
            "elapsed_sec": round(elapsed, 2),
            "raw_faces": len(raw_mesh.faces),
            "final_faces": len(final_mesh.faces),
            "final_vertices": len(final_mesh.vertices),
            "is_watertight": bool(final_mesh.is_watertight),
            "resolution": resolution,
            "device": str(self.device),
            "texture_baked": bake_texture
        }

        return final_mesh, elapsed, metadata
