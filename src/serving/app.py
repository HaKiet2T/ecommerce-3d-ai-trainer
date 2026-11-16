"""
FastAPI Microservice for CRM 3D Product Reconstruction
======================================================
Serves REST endpoints for generating GLB/OBJ 3D models from 2D e-commerce photos.
"""

import io
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.serving.inference import CRMInferenceEngine

OUTPUT_DIR = Path("static/generated_3d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="CRM 3D E-commerce Reconstruction API",
    description="Microservice chuyen doi 1 anh san pham thanh 3D Mesh (GLB/OBJ) phuc vu storefront Next.js",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Tu dong khoi tao inference engine khi FastAPI khoi dong."""
    CRMInferenceEngine.get_instance()

@app.get("/health")
def health_check():
    """Kiem tra tinh trang GPU, VRAM va trang thai san sang cua mo hinh."""
    engine = CRMInferenceEngine.get_instance()
    vram_used = 0.0
    vram_total = 0.0
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    return {
        "status": "healthy",
        "device": str(engine.device),
        "gpu_available": torch.cuda.is_available(),
        "vram_used_gb": round(vram_used, 2),
        "vram_total_gb": round(vram_total, 2),
        "model_loaded": engine.model is not None
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

    task_id = str(uuid.uuid4())[:8]

    try:
        contents = await file.read()
        engine = CRMInferenceEngine.get_instance()
        mesh, elapsed = engine.generate_mesh(
            image_bytes=contents,
            resolution=resolution,
            target_faces=target_faces,
            remove_background=remove_background
        )

        filename = f"product_{task_id}.{output_format.lower()}"
        file_path = OUTPUT_DIR / filename

        if output_format.lower() == "glb":
            mesh.export(str(file_path), file_type="glb")
            media_type = "model/gltf-binary"
        else:
            mesh.export(str(file_path), file_type="obj")
            media_type = "text/plain"

        print(f"[Done {task_id}] Sinh 3D thanh cong trong {elapsed:.2f}s | Faces: {len(mesh.faces)}")

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

def main():
    import uvicorn
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
