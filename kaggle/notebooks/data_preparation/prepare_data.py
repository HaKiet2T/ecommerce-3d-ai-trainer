"""
Data Preparation: Download + Filter Objaverse + Render 6 Views
===============================================================
Kaggle Script - Chạy trên CPU (không cần GPU)

Pipeline:
1. Download Objaverse subset (furniture/tech categories)
2. Filter objects phù hợp domain E-commerce
3. Render 6 orthographic views qua trimesh/pyrender
4. Generate CCMs (Color Coordinate Maps)
5. Sample 3D points + compute SDF
6. Upload dataset kết quả lên Kaggle Datasets

Output: /kaggle/working/crm_ecommerce_dataset/
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def install_dependencies():
    packages = [
        "objaverse==0.1.7",
        "trimesh[easy]",
        "pyrender",
        "Pillow",
        "tqdm",
        "scipy",
    ]
    for pkg in packages:
        os.system(f"pip install -q {pkg}")
    # Headless rendering support on Linux (Kaggle Ubuntu 24.04 noble)
    if os.name != "nt":
        os.system("apt-get update -qq && apt-get install -y -qq xvfb libgl1 libglx-mesa0 libosmesa6-dev libglu1-mesa mesa-utils")
        os.system("Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &")
        os.environ["DISPLAY"] = ":99"
    os.system("pip install -q pyglet==1.5.28")
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

print("[1/6] Installing dependencies...")
install_dependencies()

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class DataConfig:
    # Categories to download (E-commerce relevant)
    target_categories = [
        "desk", "table", "chair", "office chair", "gaming chair",
        "monitor", "display", "screen", "laptop",
        "keyboard", "mouse", "headset", "headphone",
        "speaker", "webcam", "microphone",
        "desk lamp", "lamp", "light",
        "shelf", "bookshelf", "cabinet",
    ]

    # Render settings
    image_size = 256
    num_views = 6  # front, right, back, left, top, bottom
    camera_distance = 2.0

    # 6 canonical camera poses (azimuth, elevation in degrees)
    camera_poses = [
        (0, 0),      # front
        (90, 0),     # right
        (180, 0),    # back
        (270, 0),    # left
        (0, 90),     # top
        (0, -90),    # bottom
    ]

    # SDF sampling
    num_surface_points = 8192
    num_volume_points = 4096

    # Limits
    max_objects = 120  # Giới hạn số objects cho Phase 1 tinh gọn, chất lượng cao
    min_faces = 500    # Bỏ objects quá đơn giản
    max_faces = 150000 # Bỏ objects quá phức tạp tránh nặng bộ nhớ

    # Output
    output_dir = "/kaggle/working/crm_ecommerce_dataset"

config = DataConfig()

# ---------------------------------------------------------------------------
# Objaverse Download
# ---------------------------------------------------------------------------

def download_objaverse_subset(config):
    """Download filtered Objaverse objects."""
    import objaverse

    print("[2/6] Loading Objaverse annotations...")
    annotations = objaverse.load_annotations()

    # Filter by relevant categories
    filtered_uids = []
    for uid, ann in annotations.items():
        name = ann.get("name", "").lower()
        tags = [t.get("name", "").lower() for t in ann.get("tags", [])]
        categories = ann.get("categories", [])
        cat_names = [c.get("name", "").lower() for c in categories]

        all_text = " ".join([name] + tags + cat_names)

        for target in config.target_categories:
            if target in all_text:
                filtered_uids.append(uid)
                break

    print(f"  Found {len(filtered_uids)} matching objects")

    # Limit
    if len(filtered_uids) > config.max_objects:
        np.random.seed(42)
        filtered_uids = list(np.random.choice(
            filtered_uids, config.max_objects, replace=False
        ))

    print(f"  Downloading {len(filtered_uids)} objects...")
    objects = objaverse.load_objects(uids=filtered_uids)

    return objects, filtered_uids


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def get_rotation_matrix(az_deg: float, el_deg: float) -> np.ndarray:
    """Tạo ma trận xoay 4x4 chuẩn xác theo góc Azimuth và Elevation."""
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    # Xoay quanh trục Y (Azimuth)
    Ry = np.array([
        [np.cos(-az), 0, np.sin(-az), 0],
        [0, 1, 0, 0],
        [-np.sin(-az), 0, np.cos(-az), 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    # Xoay quanh trục X (Elevation)
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(el), -np.sin(el), 0],
        [0, np.sin(el), np.cos(el), 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    return Rx @ Ry


def render_views(mesh, config):
    """
    Render 6 orthographic views từ trimesh scene bằng cách xoay trực tiếp mesh.
    Returns: list of PIL images
    """
    views = []

    # Normalize mesh to unit cube centered at origin [-0.5, 0.5]
    mesh_centered = mesh.copy()
    centroid = mesh_centered.centroid
    mesh_centered.apply_translation(-centroid)
    scale = mesh_centered.extents.max()
    if scale > 0:
        mesh_centered.apply_scale(1.0 / scale)

    for az, el in config.camera_poses:
        try:
            rotated_mesh = mesh_centered.copy()
            R = get_rotation_matrix(az, el)
            rotated_mesh.apply_transform(R)

            scene = trimesh.Scene(rotated_mesh)
            scene.set_camera(
                distance=config.camera_distance,
                center=[0, 0, 0],
            )

            # Render
            data = scene.save_image(
                resolution=(config.image_size, config.image_size),
                visible=False
            )

            img = Image.open(trimesh.util.wrap_as_stream(data))
            views.append(img.convert("RGB"))

        except Exception as e:
            # Fallback: log error and blank image
            print(f"  [WARN] render_views view_{az}_{el} error: {e}")
            views.append(
                Image.new("RGB", (config.image_size, config.image_size), (128, 128, 128))
            )

    return views


def generate_ccm(mesh, config):
    """
    Generate Canonical Coordinate Maps (CCM) cho 6 views:
    Mã hóa tọa độ XYZ gốc trong [-0.5, 0.5] thành RGB cố định,
    sau đó xoay mesh và render từ 6 góc nhìn trực giao.
    """
    ccms = []

    mesh_centered = mesh.copy()
    centroid = mesh_centered.centroid
    mesh_centered.apply_translation(-centroid)
    scale = mesh_centered.extents.max()
    if scale > 0:
        mesh_centered.apply_scale(1.0 / scale)

    # 1. Gán màu đỉnh cố định theo tọa độ không gian chuẩn [-0.5, 0.5]
    verts = mesh_centered.vertices
    # Chuẩn hóa về [0, 255]
    colors = np.clip((verts + 0.5) * 255.0, 0, 255).astype(np.uint8)
    vertex_colors = np.column_stack([colors, np.full(len(colors), 255, dtype=np.uint8)])

    for az, el in config.camera_poses:
        try:
            rotated_mesh = mesh_centered.copy()
            # Gán ColorVisuals rõ ràng để ghi đè mọi texture trước đó
            rotated_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=rotated_mesh, vertex_colors=vertex_colors
            )
            R = get_rotation_matrix(az, el)
            rotated_mesh.apply_transform(R)

            scene = trimesh.Scene(rotated_mesh)
            scene.set_camera(
                distance=config.camera_distance,
                center=[0, 0, 0],
            )

            data = scene.save_image(
                resolution=(config.image_size, config.image_size),
                visible=False
            )
            ccm = Image.open(trimesh.util.wrap_as_stream(data)).convert("RGB")
            ccms.append(ccm)

        except Exception as e:
            print(f"  [WARN] generate_ccm view_{az}_{el} error: {e}")
            ccms.append(
                Image.new("RGB", (config.image_size, config.image_size), (0, 0, 0))
            )

    return ccms


def sample_points_and_sdf(mesh, config):
    """
    Sample 3D points and compute SDF values using fast, safe cKDTree.
    Safe against non-watertight meshes, uses <50MB RAM, zero OOM risk.
    """
    from scipy.spatial import cKDTree

    mesh_centered = mesh.copy()
    centroid = mesh_centered.centroid
    mesh_centered.apply_translation(-centroid)
    scale = mesh_centered.extents.max()
    if scale > 0:
        mesh_centered.apply_scale(1.0 / scale)

    # 1. Surface points (ground truth SDF = 0)
    surf_pts, face_indices = trimesh.sample.sample_surface(
        mesh_centered, config.num_surface_points
    )
    surf_sdf = np.zeros((len(surf_pts), 1), dtype=np.float32)

    # Surface colors
    if hasattr(mesh_centered.visual, "face_colors") and len(mesh_centered.visual.face_colors) > 0:
        surf_colors = mesh_centered.visual.face_colors[face_indices][:, :3] / 255.0
    else:
        surf_colors = np.ones((len(surf_pts), 3), dtype=np.float32) * 0.5

    # 2. Near-surface points (surface + Gaussian noise sigma=0.02)
    noise = np.random.normal(scale=0.02, size=surf_pts.shape).astype(np.float32)
    near_pts = surf_pts + noise
    near_dist = np.linalg.norm(noise, axis=-1, keepdims=True).astype(np.float32)
    near_colors = surf_colors.copy()

    # Dấu gần bề mặt dựa vào pháp tuyến bề mặt (normal)
    if hasattr(mesh_centered, "face_normals") and len(mesh_centered.face_normals) > 0:
        normals = mesh_centered.face_normals[face_indices]
        near_sign = np.sign(np.sum(noise * normals, axis=-1, keepdims=True))
        near_sign[near_sign == 0] = 1.0
        near_sdf = (near_dist * near_sign).astype(np.float32)
    else:
        near_sdf = near_dist

    # 3. Volume points trong bounding box [-0.6, 0.6]^3
    vol_pts = np.random.uniform(-0.6, 0.6, (config.num_volume_points, 3)).astype(np.float32)

    # Fast cKDTree query against surface points (tránh mesh.contains raycast OOM)
    tree = cKDTree(surf_pts)
    vol_dist, vol_idx = tree.query(vol_pts)
    vol_dist = vol_dist.astype(np.float32)[:, np.newaxis]
    vol_colors = surf_colors[vol_idx]

    # Combine: surface + near surface + volume
    points = np.vstack([surf_pts, near_pts, vol_pts]).astype(np.float32)
    sdf = np.vstack([surf_sdf, near_sdf, vol_dist]).astype(np.float32)
    colors = np.vstack([surf_colors, near_colors, vol_colors]).astype(np.float32)

    return points, sdf, colors


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def process_object(mesh_path, obj_id, output_dir, config):
    """Process một object: render + CCM + sample points."""
    obj_dir = Path(output_dir) / obj_id
    obj_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load mesh
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # Filter by complexity
        if len(mesh.faces) < config.min_faces:
            return False, "too simple"
        if len(mesh.faces) > config.max_faces:
            return False, "too complex"

        # Render 6 views
        views = render_views(mesh, config)
        for i, view in enumerate(views):
            view.save(obj_dir / f"view_{i}.png")

        # Save input (front view)
        views[0].save(obj_dir / "input.png")

        # Generate CCMs
        ccms = generate_ccm(mesh, config)
        for i, ccm in enumerate(ccms):
            ccm.save(obj_dir / f"ccm_{i}.png")

        # Sample points + SDF
        points, sdf, colors = sample_points_and_sdf(mesh, config)
        np.save(obj_dir / "points.npy", points)
        np.save(obj_dir / "sdf.npy", sdf)
        np.save(obj_dir / "colors.npy", colors)

        # Metadata
        meta = {
            "uid": obj_id,
            "num_faces": int(len(mesh.faces)),
            "num_vertices": int(len(mesh.vertices)),
            "bounds": mesh.bounds.tolist(),
        }
        with open(obj_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return True, "ok"

    except Exception as e:
        # Cleanup failed object
        if obj_dir.exists():
            shutil.rmtree(obj_dir)
        return False, str(e)


def find_custom_models():
    """Tìm các model 3D custom được tải lên (Hướng C: Hybrid)."""
    search_dirs = [
        Path("/kaggle/input"),
        Path("./custom_models"),
        Path("./data/custom"),
    ]
    custom_models = {}
    for s_dir in search_dirs:
        if s_dir.exists():
            for ext in ["*.glb", "*.gltf", "*.obj"]:
                for f in s_dir.rglob(ext):
                    if "crm_ecommerce_dataset" not in str(f):
                        uid = f"custom_{f.stem}_{len(custom_models):03d}"
                        custom_models[uid] = str(f)
    return custom_models


def main():
    print("=" * 60)
    print("  CRM 3D Data Preparation (Hướng C: Hybrid)")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 60)

    os.makedirs(config.output_dir, exist_ok=True)

    # Download Objaverse objects
    objects, uids = download_objaverse_subset(config)

    # Hybrid (Hướng C): Tìm và nạp thêm custom models nếu có
    custom_objects = find_custom_models()
    if custom_objects:
        print(f"\n[Hybrid Mode] Phát hiện {len(custom_objects)} custom 3D models từ input:")
        for cuid, cpath in custom_objects.items():
            print(f"  + {cuid}: {cpath}")
            objects[cuid] = cpath
            uids.append(cuid)

    # Process each object
    print(f"\n[3/6] Processing {len(objects)} objects...")
    success_count = 0
    fail_count = 0
    stats = {"success": [], "failed": []}

    for uid in tqdm(uids, desc="Processing"):
        if uid not in objects:
            continue

        mesh_path = objects[uid]
        ok, msg = process_object(mesh_path, uid, config.output_dir, config)

        if ok:
            success_count += 1
            stats["success"].append(uid)
        else:
            fail_count += 1
            stats["failed"].append({"uid": uid, "reason": msg})

    # Summary
    print(f"\n[4/6] Results:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")

    # Save stats
    with open(Path(config.output_dir) / "dataset_stats.json", "w") as f:
        json.dump({
            "total_processed": success_count + fail_count,
            "success": success_count,
            "failed": fail_count,
            "categories": config.target_categories,
            "image_size": config.image_size,
            "num_views": config.num_views,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    # Zip for download
    print("[5/6] Creating archive...")
    archive_path = "/kaggle/working/crm_ecommerce_dataset"
    if success_count > 0:
        shutil.make_archive(
            archive_path, "zip", config.output_dir
        )
        archive_size = os.path.getsize(f"{archive_path}.zip") / (1024**2)
        print(f"  Archive: {archive_path}.zip ({archive_size:.1f} MB)")
    else:
        print("[ERROR] 0 objects processed successfully. Check dependencies or mesh loader.")
        sys.exit(1)

    print("[6/6] Done!")
    print(f"  Dataset: {config.output_dir}")
    print(f"  Objects: {success_count}")


if __name__ == "__main__":
    main()
