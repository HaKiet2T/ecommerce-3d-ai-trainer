import os
import sys
import io
import time
import uuid
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import rembg
except ImportError:
    rembg = None

from dataset import generate_ccm_from_rgb, VIEW_NAMES
from models.crm import ConvolutionalReconstructionModel

# Khoi tao ung dung FastAPI
app = FastAPI(
    title="CRM 3D E-commerce Reconstruction API",
    description="Microservice chuyen doi 1 anh san pham thanh 3D Mesh (GLB/OBJ) phuc vu storefront Next.js",
    version="1.0.0"
)

# Cau hinh CORS cho Next.js va dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thu muc luu tru tam thoi cac file mesh xuat ra
OUTPUT_DIR = Path("static/generated_3d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bien toan cuc quan ly mo hinh
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crm_model: Optional[ConvolutionalReconstructionModel] = None

def get_model() -> ConvolutionalReconstructionModel:
    """Khoi tao hoac nap checkpoint mo hinh CRM (Singleton)."""
    global crm_model
    if crm_model is None:
        print(f"[Serving] Dang khoi tao mo hinh CRM tren: {DEVICE}...")
        model = ConvolutionalReconstructionModel(
            triplane_channels=32,
            base_features=64,
            hidden_dim=64
        ).to(DEVICE)

        ckpt_path = os.environ.get("CRM_CHECKPOINT", "checkpoints/crm_best.pt")
        if os.path.exists(ckpt_path):
            print(f"[Serving] Nap trong so tu {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
        else:
            print("[Serving] Canh bao: Khong tim thay checkpoint, su dung khoi tao mac dinh.")

        model.eval()
        crm_model = model
    return crm_model

@app.on_event("startup")
async def startup_event():
    """Tu dong nap mo hinh khi FastAPI container khoi dong."""
    get_model()

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
        # Fallback neu chua cai rembg: lay nguong sang de tach nen trang
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
    target_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chuan bi tensor 6 goc nhin truc giao tu anh san pham duy nhat:
    Tao 6 goc nhin RGB va 6 mat phang toa do CCM chuan hoa.
    """
    rgb_np = np.array(rgb).astype(np.float32) / 255.0
    # Chuyen doi sang khoang [-1, 1]
    rgb_tensor_single = torch.from_numpy(rgb_np).permute(2, 0, 1) * 2.0 - 1.0

    images_list, ccms_list = [], []
    for view in VIEW_NAMES:
        # Neu view la front ta dung anh that, cac view khac can bang sac do hoac suy luan
        images_list.append(rgb_tensor_single)

        ccm_np = generate_ccm_from_rgb(rgb, mask, view)
        ccm_tensor = torch.from_numpy(ccm_np).permute(2, 0, 1) * 2.0 - 1.0
        ccms_list.append(ccm_tensor)

    images = torch.stack(images_list, dim=0).unsqueeze(0).to(DEVICE) # [1, 6, 3, H, W]
    ccms = torch.stack(ccms_list, dim=0).unsqueeze(0).to(DEVICE)     # [1, 6, 3, H, W]
    return images, ccms

@app.get("/health")
def health_check():
    """Kiem tra tinh trang GPU, VRAM va trang thai san sang cua mo hinh."""
    vram_used = 0.0
    vram_total = 0.0
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "gpu_available": torch.cuda.is_available(),
        "vram_used_gb": round(vram_used, 2),
        "vram_total_gb": round(vram_total, 2),
        "model_loaded": crm_model is not None
    }

@app.post("/generate-3d")
async def generate_3d_endpoint(
    file: UploadFile = File(...),
    output_format: str = Form("glb"),
    resolution: int = Form(64),
    target_faces: int = Form(25000),
    remove_background: bool = Form(True)
):
    """
    API endpoint chinh: Sinh 3D Mesh tu 1 anh san pham.
    - file: file anh upload (JPEG/PNG)
    - output_format: 'glb' (uu tien cho Three.js tren Next.js) hoac 'obj'
    - resolution: do phan giai Marching Cubes (64 cho nhanh, 128 cho chi tiet cao)
    - target_faces: so mat tam giac toi da sau decimation giup load web nhe
    """
    if output_format.lower() not in ["glb", "obj"]:
        raise HTTPException(status_code=400, detail="Chi ho tro dinh dang 'glb' hoac 'obj'")

    start_time = time.time()
    task_id = str(uuid.uuid4())[:8]

    try:
        # 1. Doc du lieu anh
        contents = await file.read()
        rgb_img, mask = preprocess_product_image(contents, target_size=256)

        # 2. Tao tensor da goc nhin
        images, ccms = synthesize_multiview_tensors(rgb_img, mask, target_size=256)

        # 3. Chay mo hinh CRM
        model = get_model()
        with torch.no_grad():
            triplanes = model.unet(images, ccms)
            mesh = model.extract_mesh(triplanes, resolution=resolution)

        # 4. Hau xu ly 3D Mesh cho E-commerce WebGL
        # Don gian hoa so luong da giac (Decimation) de Next.js load duoi 5MB
        if target_faces > 0 and len(mesh.faces) > target_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)

        # 5. Xuat file ket qua
        filename = f"product_{task_id}.{output_format.lower()}"
        file_path = OUTPUT_DIR / filename

        if output_format.lower() == "glb":
            mesh.export(str(file_path), file_type="glb")
            media_type = "model/gltf-binary"
        else:
            mesh.export(str(file_path), file_type="obj")
            media_type = "text/plain"

        elapsed = time.time() - start_time
        print(f"[Done {task_id}] Sinh 3D thanh cong trong {elapsed:.2f}s | Faces: {len(mesh.faces)}")

        # Tra ve file mesh truc tiep de Next.js render bang Three.js / React Three Fiber
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type=media_type,
            headers={
                "X-Process-Time": f"{elapsed:.2f}s",
                "X-Vertex-Count": str(len(mesh.vertices)),
                "X-Face-Count": str(len(mesh.faces))
            }
        )

    except Exception as e:
        print(f"[Error {task_id}] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Loi xu ly 3D: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Chay server FastAPI tai port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
