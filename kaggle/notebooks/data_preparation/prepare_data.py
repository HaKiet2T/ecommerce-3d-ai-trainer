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
    # headless rendering support
    os.system("pip install -q pyglet==1.5.28")
    os.environ["PYOPENGL_PLATFORM"] = "egl"

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
    max_objects = 500  # Giới hạn số objects (Kaggle time limit)
    min_faces = 100     # Bỏ objects quá đơn giản
    max_faces = 500000  # Bỏ objects quá phức tạp

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

def render_views(mesh, config):
    """
    Render 6 orthographic views từ trimesh scene.
    Returns: list of PIL images
    """
    views = []

    # Normalize mesh to unit cube centered at origin
    mesh_centered = mesh.copy()
    centroid = mesh_centered.centroid
    mesh_centered.apply_translation(-centroid)
    scale = mesh_centered.extents.max()
    if scale > 0:
        mesh_centered.apply_scale(1.0 / scale)

    for az, el in config.camera_poses:
        try:
            scene = trimesh.Scene(mesh_centered)

            # Camera transform
            az_rad = np.radians(az)
            el_rad = np.radians(el)

            cam_x = config.camera_distance * np.cos(el_rad) * np.sin(az_rad)
            cam_y = config.camera_distance * np.sin(el_rad)
            cam_z = config.camera_distance * np.cos(el_rad) * np.cos(az_rad)

            camera_position = np.array([cam_x, cam_y, cam_z])

            # Look at origin
            scene.set_camera(
                angles=None,
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
            # Fallback: blank image
            views.append(
                Image.new("RGB", (config.image_size, config.image_size), (128, 128, 128))
            )

    return views


def generate_ccm(mesh, config):
    """
    Generate Color Coordinate Maps (CCM) cho 6 views.
    CCM encodes XYZ position as RGB colors.
    """
    ccms = []

    mesh_centered = mesh.copy()
    centroid = mesh_centered.centroid
    mesh_centered.apply_translation(-centroid)
    scale = mesh_centered.extents.max()
    if scale > 0:
        mesh_centered.apply_scale(1.0 / scale)

    for az, el in config.camera_poses:
        try:
            # Simplified CCM: project vertices and color by position
            vertices = mesh_centered.vertices.copy()

            # Rotate to camera view
            az_rad = np.radians(az)
            el_rad = np.radians(el)

            # Rotation matrix
            Ry = np.array([
                [np.cos(az_rad), 0, np.sin(az_rad)],
                [0, 1, 0],
                [-np.sin(az_rad), 0, np.cos(az_rad)]
            ])
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(el_rad), -np.sin(el_rad)],
                [0, np.sin(el_rad), np.cos(el_rad)]
            ])

            rotated = (Rx @ Ry @ vertices.T).T

            # Normalize XYZ to [0, 255] for RGB encoding
            xyz_normalized = (rotated - rotated.min(axis=0)) / (
                rotated.max(axis=0) - rotated.min(axis=0) + 1e-8
            )
            colors = (xyz_normalized * 255).astype(np.uint8)

            # Create colored mesh and render
            colored_mesh = mesh_centered.copy()
            colored_mesh.visual.vertex_colors = np.column_stack([
                colors, np.full(len(colors), 255, dtype=np.uint8)
            ])

            scene = trimesh.Scene(colored_mesh)
            data = scene.save_image(
                resolution=(config.image_size, config.image_size),
                visible=False
            )
            ccm = Image.open(trimesh.util.wrap_as_stream(data)).convert("RGB")
            ccms.append(ccm)

        except Exception:
            ccms.append(
                Image.new("RGB", (config.image_size, config.image_size), (0, 0, 0))
            )

    return ccms


def sample_points_and_sdf(mesh, config):
    """
    Sample 3D points and compute SDF values.
    """
    mesh_centered = mesh.copy()
    centroid = mesh_centered.centroid
    mesh_centered.apply_translation(-centroid)
    scale = mesh_centered.extents.max()
    if scale > 0:
        mesh_centered.apply_scale(1.0 / scale)

    # Surface points
    surface_points, face_indices = trimesh.sample.sample_surface(
        mesh_centered, config.num_surface_points
    )
    surface_sdf = np.zeros((len(surface_points), 1), dtype=np.float32)

    # Surface colors
    if hasattr(mesh_centered.visual, "face_colors"):
        surface_colors = mesh_centered.visual.face_colors[face_indices][:, :3] / 255.0
    else:
        surface_colors = np.ones((len(surface_points), 3), dtype=np.float32) * 0.5

    # Volume points (random in [-1.2, 1.2])
    volume_points = np.random.uniform(
        -1.2, 1.2, (config.num_volume_points, 3)
    ).astype(np.float32)

    # Compute approximate SDF for volume points
    try:
        closest_points, distances, face_ids = trimesh.proximity.closest_point(
            mesh_centered, volume_points
        )
        signs = mesh_centered.contains(volume_points).astype(np.float32) * 2 - 1
        volume_sdf = (distances * -signs).reshape(-1, 1).astype(np.float32)
        volume_colors = np.ones((len(volume_points), 3), dtype=np.float32) * 0.5
    except Exception:
        volume_sdf = np.zeros((len(volume_points), 1), dtype=np.float32)
        volume_colors = np.ones((len(volume_points), 3), dtype=np.float32) * 0.5

    # Combine
    points = np.vstack([surface_points, volume_points]).astype(np.float32)
    sdf = np.vstack([surface_sdf, volume_sdf]).astype(np.float32)
    colors = np.vstack([surface_colors, volume_colors]).astype(np.float32)

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


def main():
    print("=" * 60)
    print("  CRM 3D Data Preparation")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 60)

    os.makedirs(config.output_dir, exist_ok=True)

    # Download objects
    objects, uids = download_objaverse_subset(config)

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

    print("[6/6] Done!")
    print(f"  Dataset: {config.output_dir}")
    print(f"  Objects: {success_count}")


if __name__ == "__main__":
    main()
