"""
End-to-End Serving Verification for Option B
===========================================
Executes full serving flow on real e-commerce objects:
1. Preprocesses 2D single photo (background extraction & bounding box centering)
2. Synthesizes multi-view representations & CCMs
3. Computes Triplane features with CRM U-Net
4. Extracts 3D mesh via Marching Cubes
5. Applies Quadric Decimation for web performance
6. Bakes multi-view authentic vertex colors
7. Exports production-ready GLB & OBJ models
8. Logs detailed MLOps verification metrics
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, ".")

from src.serving.inference import CRMInferenceEngine

def run_verification():
    print("=" * 75)
    print("KIEM CHUNG THUC NGHIEM SERVING OPTION B (MULTI-VIEW SYNTHESIS + COLOR BAKING)")
    print("=" * 75)

    engine = CRMInferenceEngine.get_instance()
    out_dir = Path("outputs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path("outputs/kaggle/data_prep/20261116_154412/crm_ecommerce_dataset")
    test_ids = [
        "035aa8e8c99d455f91e96d9c78a6a404",
        "0583a563b6ba45c486d4a926e65e61ca"
    ]

    results = []

    for obj_id in test_ids:
        print(f"\n--- Dang xu ly vat the e-commerce: {obj_id} ---")
        img_path = dataset_dir / obj_id / "input.png"
        if not img_path.exists():
            print(f"Khong tim thay {img_path}")
            continue

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        t0 = time.time()
        mesh, elapsed, meta = engine.generate_mesh(
            image_bytes=img_bytes,
            resolution=64,
            target_faces=20000,
            remove_background=True,
            use_diffusion=False,
            bake_texture=True
        )
        total_time = time.time() - t0

        glb_path = out_dir / f"serving_v2_{obj_id[:8]}.glb"
        obj_path = out_dir / f"serving_v2_{obj_id[:8]}.obj"

        mesh.export(str(glb_path), file_type="glb")
        mesh.export(str(obj_path), file_type="obj")

        colors = mesh.visual.vertex_colors
        color_std = float(np.std(colors[:, :3]))
        color_mean = [float(x) for x in np.mean(colors[:, :3], axis=0)]

        print(f"-> Thoi gian xu ly: {total_time:.2f}s (Inference core: {elapsed:.2f}s)")
        print(f"-> So dinh: {len(mesh.vertices):,} | So mat: {len(mesh.faces):,}")
        print(f"-> Watertight: {mesh.is_watertight} | Volume: {mesh.volume if mesh.is_watertight else 'N/A'}")
        print(f"-> Dung luong file GLB: {glb_path.stat().st_size / 1024:.1f} KB")
        print(f"-> Do bien thien mau sac (Color Std): {color_std:.2f} (Mean RGB: {color_mean})")

        res_entry = {
            "object_id": obj_id,
            "total_time_sec": round(total_time, 2),
            "inference_time_sec": round(elapsed, 2),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "is_watertight": bool(mesh.is_watertight),
            "glb_size_kb": round(glb_path.stat().st_size / 1024, 1),
            "color_std": round(color_std, 2),
            "color_mean_rgb": color_mean,
            "glb_file": str(glb_path).replace("\\", "/"),
            "obj_file": str(obj_path).replace("\\", "/")
        }
        results.append(res_entry)

    summary_file = out_dir / "serving_v2_verification_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline": "Option B (Refactored Serving + Texture Baking)",
            "results": results
        }, f, indent=2)

    print(f"\n[XONG] Da ghi tong hop kiem chung tai: {summary_file}")
    return results

if __name__ == "__main__":
    run_verification()
