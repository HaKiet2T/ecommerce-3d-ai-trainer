"""
Visual Inspection Generator for 3D CRM Reconstruction Experiment
=================================================================
Renders 2D/3D multi-angle visual comparisons:
[Input 2D Photo] vs [3D Reconstructed Front] vs [3D Side] vs [3D Isometric Perspective]
Saves high-res comparison plots to outputs/experiments/
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
    if len(faces) > 4000:
        idx = np.random.choice(len(faces), 4000, replace=False)
        plot_faces = faces[idx]
    else:
        plot_faces = faces

    # 2. 3D Isometric View
    ax1 = fig.add_subplot(1, 4, 2, projection="3d")
    ax1.plot_trisurf(verts[:, 0], verts[:, 2], plot_faces, verts[:, 1],
                     cmap="Spectral", edgecolor="none", alpha=0.9, shade=True)
    ax1.view_init(elev=25, azim=45)
    ax1.set_title("2. Luoi 3D Isometric (Xoay 45 deg)", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # 3. 3D Front View
    ax2 = fig.add_subplot(1, 4, 3, projection="3d")
    ax2.plot_trisurf(verts[:, 0], verts[:, 2], plot_faces, verts[:, 1],
                     cmap="viridis", edgecolor="none", alpha=0.9, shade=True)
    ax2.view_init(elev=0, azim=0)
    ax2.set_title("3. Goc Truc Dien (Front View)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # 4. 3D Side Profile View
    ax3 = fig.add_subplot(1, 4, 4, projection="3d")
    ax3.plot_trisurf(verts[:, 0], verts[:, 2], plot_faces, verts[:, 1],
                     cmap="plasma", edgecolor="none", alpha=0.9, shade=True)
    ax3.view_init(elev=0, azim=90)
    ax3.set_title("4. Goc Canh Ben (Profile View)", fontsize=11, fontweight="bold")
    ax3.axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"-> Da luu visual comparison: {output_png_path}")

def main():
    exp_dir = Path("outputs/experiments")
    ds_dir = Path("outputs/kaggle/data_prep/20261116_154412/crm_ecommerce_dataset")

    # Item 1
    id1 = "035aa8e8c99d455f91e96d9c78a6a404"
    mesh1 = exp_dir / "reconstructed_035aa8e8.obj"
    img1 = ds_dir / id1 / "input.png"
    out1 = exp_dir / "visual_comparison_035aa8e8.png"
    if mesh1.exists() and img1.exists():
        render_comparison(str(img1), str(mesh1), str(out1),
                          title=f"Thuc Nghiem Tai Tao 3D CRM: Vat The {id1[:8]}")

    # Item 2
    id2 = "0583a563b6ba45c486d4a926e65e61ca"
    mesh2 = exp_dir / "reconstructed_0583a563.obj"
    img2 = ds_dir / id2 / "input.png"
    out2 = exp_dir / "visual_comparison_0583a563.png"
    if mesh2.exists() and img2.exists():
        render_comparison(str(img2), str(mesh2), str(out2),
                          title=f"Thuc Nghiem Tai Tao 3D CRM: Vat The {id2[:8]}")

if __name__ == "__main__":
    main()
