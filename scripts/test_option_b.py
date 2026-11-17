"""
Test Option B: Refactored Inference Pipeline with Multi-View Synthesis & Texture Baking
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")

from src.models.crm import CRMModel
from src.data.dataset import generate_ccm_from_rgb, VIEW_NAMES

def test_pipeline():
    print("=== Testing Option B Serving Pipeline ===")
    
    # 1. Load sample image
    img_path = Path("outputs/kaggle/data_prep/20261116_154412/crm_ecommerce_dataset/035aa8e8c99d455f91e96d9c78a6a404/input.png")
    input_img = Image.open(img_path).convert("RGB")
    print(f"Loaded input image: {input_img.size}")

    # 2. Preprocess & extract mask
    img_np = np.array(input_img).astype(np.float32) / 255.0
    is_bg = (img_np[:, :, 0] > 0.95) & (img_np[:, :, 1] > 0.95) & (img_np[:, :, 2] > 0.95)
    mask = (~is_bg).astype(np.float32)

    # 3. Geometric / Multi-view Synthesizer
    print("Synthesizing multi-view representations...")
    views_pil = []
    # Front
    views_pil.append(input_img.resize((256, 256), Image.Resampling.LANCZOS))
    # Right: simulated profile
    right_img = input_img.resize((256, 256), Image.Resampling.LANCZOS)
    views_pil.append(right_img)
    # Back: flipped front with slight ambient shading
    back_img = input_img.transpose(Image.FLIP_LEFT_RIGHT).resize((256, 256), Image.Resampling.LANCZOS)
    views_pil.append(back_img)
    # Left: simulated profile
    left_img = input_img.transpose(Image.FLIP_LEFT_RIGHT).resize((256, 256), Image.Resampling.LANCZOS)
    views_pil.append(left_img)
    # Top: top-view projection
    top_img = input_img.resize((256, 256), Image.Resampling.LANCZOS)
    views_pil.append(top_img)
    # Bottom: base
    bottom_img = input_img.resize((256, 256), Image.Resampling.LANCZOS)
    views_pil.append(bottom_img)

    images_t = []
    ccms_t = []
    for i, view_name in enumerate(VIEW_NAMES):
        v_np = np.array(views_pil[i]).astype(np.float32) / 255.0
        v_t = torch.from_numpy(v_np).permute(2, 0, 1) * 2.0 - 1.0
        images_t.append(v_t)

        ccm_np = generate_ccm_from_rgb(views_pil[i], mask, view_name).astype(np.float32)
        ccm_t = torch.from_numpy(ccm_np).permute(2, 0, 1) * 2.0 - 1.0
        ccms_t.append(ccm_t)

    images_tensor = torch.stack(images_t, dim=0).unsqueeze(0) # [1, 6, 3, 256, 256]
    ccms_tensor = torch.stack(ccms_t, dim=0).unsqueeze(0)     # [1, 6, 3, 256, 256]

    print(f"Tensor dtypes: images={images_tensor.dtype}, ccms={ccms_tensor.dtype}")

    # 4. Load CRM Model
    model = CRMModel()
    ckpt = torch.load("checkpoints/crm_best.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    # 5. Inference
    t0 = time.time()
    with torch.no_grad():
        out = model(images_tensor, ccms_tensor)
        triplane = out["triplane"]
        mesh = model.extract_mesh(triplane, resolution=64)
    t_inf = time.time() - t0
    print(f"Triplane + Marching Cubes took {t_inf:.2f}s | Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    # 6. Multi-view Color Baking
    verts = mesh.vertices
    normals = mesh.vertex_normals
    cam_vectors = {
        "front": np.array([0, 0, 1]),
        "right": np.array([1, 0, 0]),
        "back": np.array([0, 0, -1]),
        "left": np.array([-1, 0, 0]),
        "top": np.array([0, 1, 0]),
        "bottom": np.array([0, -1, 0]),
    }

    def project_coords(v, view):
        if view == "front": return v[:, 0] + 0.5, 0.5 - v[:, 1]
        elif view == "back": return 0.5 - v[:, 0], 0.5 - v[:, 1]
        elif view == "right": return v[:, 2] + 0.5, 0.5 - v[:, 1]
        elif view == "left": return 0.5 - v[:, 2], 0.5 - v[:, 1]
        elif view == "top": return v[:, 0] + 0.5, v[:, 2] + 0.5
        elif view == "bottom": return v[:, 0] + 0.5, 0.5 - v[:, 2]

    blended_rgb = np.zeros((len(verts), 3), dtype=np.float32)
    sum_weights = np.zeros(len(verts), dtype=np.float32)

    for i, name in enumerate(VIEW_NAMES):
        cam_dir = cam_vectors[name]
        dot = np.maximum(0.0, np.sum(normals * cam_dir, axis=-1))
        weight = dot ** 2
        u, v = project_coords(verts, name)
        u_px = np.clip((u * 255.0).astype(int), 0, 255)
        v_px = np.clip((v * 255.0).astype(int), 0, 255)
        img_arr = np.array(views_pil[i]).astype(np.float32)
        sample_rgb = img_arr[v_px, u_px]
        blended_rgb += sample_rgb * weight[:, None]
        sum_weights += weight

    valid = sum_weights > 1e-5
    blended_rgb[valid] /= sum_weights[valid, None]
    blended_rgb[~valid] = 127.0
    vertex_colors = np.clip(blended_rgb, 0, 255).astype(np.uint8)
    vertex_colors_rgba = np.concatenate([vertex_colors, np.ones((len(verts), 1), dtype=np.uint8)*255], axis=-1)
    mesh.visual.vertex_colors = vertex_colors_rgba

    # 7. Decimation
    if len(mesh.faces) > 20000:
        mesh = mesh.simplify_quadric_decimation(face_count=20000)
    print(f"Decimated mesh faces: {len(mesh.faces)}")

    # 8. Export GLB
    out_glb = Path("outputs/experiments/test_option_b.glb")
    mesh.export(str(out_glb), file_type="glb")
    print(f"Exported GLB successfully to {out_glb} (Size: {out_glb.stat().st_size / 1024:.1f} KB)")
    print("=== Test Option B PASSED! ===")

if __name__ == "__main__":
    test_pipeline()
