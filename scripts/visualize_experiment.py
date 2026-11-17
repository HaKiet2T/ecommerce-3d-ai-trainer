"""
Visual Inspection Generator for 3D CRM Reconstruction Experiment
=================================================================
Renders 2D/3D multi-angle visual comparisons:
[Input 2D Photo] vs [3D Reconstructed Front] vs [3D Side] vs [3D Isometric Perspective]
Supports authentic vertex color rendering and saves high-res comparison plots to outputs/experiments/
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def render_comparison(
    input_img_path: str,
    mesh_path: str,
    output_png_path: str,
    title: str = "3D Reconstruction Verification"
):
    fig = plt.figure(figsize=(16, 5))
    
    # 1. Original Input Image
    ax0 = fig.add_subplot(1, 4, 1)
    if Path(input_img_path).exists():
        img = Image.open(input_img_path).convert("RGB")
        ax0.imshow(img)
    ax0.set_title("1. Anh San Pham Dau Vao (2D)", fontsize=11, fontweight="bold")
    ax0.axis("off")

    # Load 3D Mesh
    mesh = trimesh.load(mesh_path, force="mesh")
    verts = mesh.vertices
    faces = mesh.faces

    # Subsample faces if mesh is heavy for faster plotting
    max_plot_faces = 4000
    if len(faces) > max_plot_faces:
        idx = np.random.choice(len(faces), max_plot_faces, replace=False)
        plot_faces = faces[idx]
    else:
        plot_faces = faces

    has_colors = hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) > 0
    if has_colors:
        v_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        face_colors = v_colors[plot_faces].mean(axis=1)
    else:
        face_colors = None

    tri_polys = verts[plot_faces] # [M, 3, 3]

    def add_mesh_view(ax, elev, azim, view_title):
        if face_colors is not None:
            collection = Poly3DCollection(tri_polys, facecolors=face_colors, shade=False)
            ax.add_collection3d(collection)
        else:
            ax.plot_trisurf(verts[:, 0], verts[:, 2], plot_faces, verts[:, 1],
                            cmap="Spectral", edgecolor="none", alpha=0.9, shade=True)

        ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
        ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
        ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_title, fontsize=11, fontweight="bold")
        ax.axis("off")

    # 2. 3D Isometric View
    ax1 = fig.add_subplot(1, 4, 2, projection="3d")
    add_mesh_view(ax1, elev=25, azim=45, view_title="2. Luoi 3D Isometric (Xoay 45 deg)")

    # 3. 3D Front View
    ax2 = fig.add_subplot(1, 4, 3, projection="3d")
    add_mesh_view(ax2, elev=0, azim=0, view_title="3. Goc Truc Dien (Front View)")

    # 4. 3D Side Profile View
    ax3 = fig.add_subplot(1, 4, 4, projection="3d")
    add_mesh_view(ax3, elev=0, azim=90, view_title="4. Goc Canh Ben (Profile View)")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"-> Da luu visual comparison: {output_png_path}")

def main():
    exp_dir = Path("outputs/experiments")
    ds_dir = Path("outputs/kaggle/data_prep/20261116_154412/crm_ecommerce_dataset")

    # Item 1 - V2 Serving Model
    id1 = "035aa8e8c99d455f91e96d9c78a6a404"
    mesh1_v2 = exp_dir / f"serving_v2_{id1[:8]}.obj"
    img1 = ds_dir / id1 / "input.png"
    out1_v2 = exp_dir / f"visual_comparison_v2_{id1[:8]}.png"
    if mesh1_v2.exists() and img1.exists():
        render_comparison(str(img1), str(mesh1_v2), str(out1_v2),
                          title=f"Kiem Chung Serving Option B: Vat The {id1[:8]} (Texture Baked)")

    # Item 2 - V2 Serving Model
    id2 = "0583a563b6ba45c486d4a926e65e61ca"
    mesh2_v2 = exp_dir / f"serving_v2_{id2[:8]}.obj"
    img2 = ds_dir / id2 / "input.png"
    out2_v2 = exp_dir / f"visual_comparison_v2_{id2[:8]}.png"
    if mesh2_v2.exists() and img2.exists():
        render_comparison(str(img2), str(mesh2_v2), str(out2_v2),
                          title=f"Kiem Chung Serving Option B: Vat The {id2[:8]} (Texture Baked)")

if __name__ == "__main__":
    main()
