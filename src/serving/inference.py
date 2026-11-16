"""
CRM 3D Inference Engine & Mesh Generation Pipeline
==================================================
Handles image preprocessing, multiview synthesis, Marching Cubes extraction,
mesh decimation, and GLB/OBJ formatting for e-commerce storefronts.
"""

import io
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image

try:
    import rembg
except ImportError:
    rembg = None

from src.data.dataset import generate_ccm_from_rgb, VIEW_NAMES
from src.models.crm import ConvolutionalReconstructionModel

def preprocess_product_image(image_bytes: bytes, target_size: int = 256) -> Tuple[Image.Image, np.ndarray]:
    """
    Tien xu ly anh san pham dau vao:
    1. Loai bo phong nen (Background Removal) bang rembg de lay mask chuan
    2. Can giua vat the trong khung hinh vuong
    3. Tra ve anh RGB va mask nhi phan
    """
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if rembg is not None:
        rgba = rembg.remove(input_image)
        alpha = np.array(rgba)[:, :, 3] / 255.0
        mask = (alpha > 0.1).astype(np.float32)
        rgb = rgba.convert("RGB")
    else:
        img_np = np.array(input_image)
        is_bg = (img_np[:, :, 0] > 240) & (img_np[:, :, 1] > 240) & (img_np[:, :, 2] > 240)
        mask = (~is_bg).astype(np.float32)
        rgb = input_image

    rgb = rgb.resize((target_size, target_size), Image.Resampling.LANCZOS)
    mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (target_size, target_size), Image.Resampling.NEAREST
    )) / 255.0

    return rgb, mask

def synthesize_multiview_tensors(
    rgb: Image.Image,
    mask: np.ndarray,
    device: torch.device,
    target_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chuan bi tensor 6 goc nhin truc giao tu anh san pham duy nhat:
    Tao 6 goc nhin RGB va 6 mat phang toa do CCM chuan hoa.
    """
    rgb_np = np.array(rgb).astype(np.float32) / 255.0
    rgb_tensor_single = torch.from_numpy(rgb_np).permute(2, 0, 1) * 2.0 - 1.0

    images_list, ccms_list = [], []
    for view in VIEW_NAMES:
        images_list.append(rgb_tensor_single)

        ccm_np = generate_ccm_from_rgb(rgb, mask, view)
        ccm_tensor = torch.from_numpy(ccm_np).permute(2, 0, 1) * 2.0 - 1.0
        ccms_list.append(ccm_tensor)

    images = torch.stack(images_list, dim=0).unsqueeze(0).to(device) # [1, 6, 3, H, W]
    ccms = torch.stack(ccms_list, dim=0).unsqueeze(0).to(device)     # [1, 6, 3, H, W]
    return images, ccms

class CRMInferenceEngine:
    """Singleton inference engine managing model lifecycle and 3D mesh reconstruction."""

    _instance: Optional["CRMInferenceEngine"] = None

    def __init__(self, checkpoint_path: Optional[str] = None, device_name: Optional[str] = None):
        self.device = torch.device(
            device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = checkpoint_path or os.environ.get("CRM_CHECKPOINT", "checkpoints/crm_best.pt")
        self.model = self._load_model()

    @classmethod
    def get_instance(cls, checkpoint_path: Optional[str] = None) -> "CRMInferenceEngine":
        if cls._instance is None:
            cls._instance = cls(checkpoint_path=checkpoint_path)
        return cls._instance

    def _load_model(self) -> ConvolutionalReconstructionModel:
        print(f"[Serving] Dang khoi tao mo hinh CRM tren: {self.device}...")
        model = ConvolutionalReconstructionModel(
            triplane_resolution=256,
            triplane_channels=32,
            base_features=64,
            hidden_dim=64
        ).to(self.device)

        if os.path.exists(self.checkpoint_path):
            print(f"[Serving] Nap weights tu: {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print("[Serving] Mo hinh CRM da san sang phuc vu!")
        else:
            print(f"[Serving] Canh bao: Khong tim thay '{self.checkpoint_path}'. Chay o che do weights mac dinh.")

        model.eval()
        return model

    def generate_mesh(
        self,
        image_bytes: bytes,
        resolution: int = 64,
        target_faces: int = 25000,
        remove_background: bool = True
    ) -> Tuple[trimesh.Trimesh, float]:
        """Runs end-to-end 3D reconstruction and returns the simplified mesh."""
        start_time = time.time()

        # 1. Preprocessing
        rgb_img, mask = preprocess_product_image(image_bytes, target_size=256)

        # 2. Multiview tensor synthesis
        images, ccms = synthesize_multiview_tensors(rgb_img, mask, device=self.device, target_size=256)

        # 3. Model inference & Marching Cubes
        with torch.no_grad():
            triplanes = self.model.unet(images, ccms)
            mesh = self.model.extract_mesh(triplanes, resolution=resolution)

        # 4. Decimation / Simplification for Web
        if target_faces > 0 and len(mesh.faces) > target_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)

        elapsed = time.time() - start_time
        return mesh, elapsed
