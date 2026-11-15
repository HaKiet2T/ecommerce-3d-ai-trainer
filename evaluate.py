import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import trimesh
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

try:
    import lpips
except ImportError:
    lpips = None

from dataset import CRMDataset, VIEW_NAMES
from models.crm import ConvolutionalReconstructionModel

def compute_mesh_chamfer(mesh_pred: trimesh.Trimesh, mesh_gt: trimesh.Trimesh, num_samples: int = 10000) -> float:
    """
    Tinh Chamfer Distance giua 2 mesh 3D:
    Lay mau 10,000 diem tren be mat moi mesh va do khoang cach euclidean hai chieu.
    """
    pts_pred, _ = trimesh.sample.sample_surface(mesh_pred, num_samples)
    pts_gt, _ = trimesh.sample.sample_surface(mesh_gt, num_samples)

    tree_gt = trimesh.proximity.ProximityQuery(mesh_gt)
    dist_pred_to_gt, _, _ = tree_gt.on_surface(pts_pred)

    tree_pred = trimesh.proximity.ProximityQuery(mesh_pred)
    dist_gt_to_pred, _, _ = tree_pred.on_surface(pts_gt)

    chamfer = np.mean(dist_pred_to_gt ** 2) + np.mean(dist_gt_to_pred ** 2)
    return float(chamfer)

def evaluate_model(
    checkpoint_path: str,
    val_data_dir: str,
    output_report_path: str = "eval_report.json",
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """
    Danh gia dinh luong mo hinh CRM tren tap Validation:
    - PSNR: Do sac net va do nhieu cua anh chieu
    - SSIM: Do tuong dong cau truc hinh hoc
    - LPIPS: Sai so tri giac thi giac (cang nho cang tot)
    - Chamfer Distance: Sai so hinh hoc 3D (cang nho cang chuan)
    """
    device = torch.device(device_name)
    print(f"[Eval] Khoi tao mo hinh tu {checkpoint_path} tren {device}...")

    model = ConvolutionalReconstructionModel(triplane_channels=32).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.eval()

    dataset = CRMDataset(data_root=val_data_dir, is_training=False)
    print(f"[Eval] So mau danh gia: {len(dataset)}")

    lpips_model = None
    if lpips is not None and torch.cuda.is_available():
        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    all_psnr, all_ssim, all_lpips, all_chamfer = [], [], [], []

    for idx in range(len(dataset)):
        item = dataset[idx]
        images = item["images"].unsqueeze(0).to(device) # [1, 6, 3, H, W]
        ccms = item["ccms"].unsqueeze(0).to(device)     # [1, 6, 3, H, W]
        sample_id = item["sample_id"]

        with torch.no_grad():
            triplanes = model.unet(images, ccms)
            pred_mesh = model.extract_mesh(triplanes, resolution=64)

        # 1. Danh gia hinh hoc 3D neu co ground truth mesh
        item_dir = dataset.sample_dirs[idx]
        gt_mesh_files = list(item_dir.glob("*.obj")) + list(item_dir.glob("*.glb"))
        if gt_mesh_files:
            gt_mesh = trimesh.load(gt_mesh_files[0], force="mesh")
            from dataset import normalize_mesh
            gt_mesh = normalize_mesh(gt_mesh)
            cd = compute_mesh_chamfer(pred_mesh, gt_mesh)
            all_chamfer.append(cd)

        # 2. Danh gia anh chieu mat phang
        # So sanh kenh RGB chieu voi anh view that
        pred_proj = torch.tanh(triplanes[:, :, :3, :, :]).squeeze(0) # [3, 3, H, W]
        gt_views = images[:, [0, 4, 1]].squeeze(0) # front, top, right [3, 3, H, W]

        pred_np = ((pred_proj.permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        gt_np = ((gt_views.permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

        for v in range(3):
            cur_psnr = psnr_fn(gt_np[v], pred_np[v], data_range=255)
            cur_ssim = ssim_fn(gt_np[v], pred_np[v], channel_axis=-1, data_range=255)
            all_psnr.append(cur_psnr)
            all_ssim.append(cur_ssim)

        if lpips_model is not None:
            with torch.no_grad():
                l_val = lpips_model(pred_proj, gt_views).squeeze().cpu().numpy()
                all_lpips.extend(l_val.tolist() if isinstance(l_val, np.ndarray) else [float(l_val)])

    report = {
        "mean_psnr": float(np.mean(all_psnr)) if all_psnr else 0.0,
        "mean_ssim": float(np.mean(all_ssim)) if all_ssim else 0.0,
        "mean_lpips": float(np.mean(all_lpips)) if all_lpips else 0.0,
        "mean_chamfer_distance": float(np.mean(all_chamfer)) if all_chamfer else 0.0,
        "num_samples_evaluated": len(dataset)
    }

    print("\n" + "=" * 60)
    print("KET QUA DANH GIA DINH LUONG CRM 3D")
    print(f"PSNR:             {report['mean_psnr']:.2f} dB (Chuan: > 24 dB)")
    print(f"SSIM:             {report['mean_ssim']:.4f} (Chuan: > 0.82)")
    print(f"LPIPS:            {report['mean_lpips']:.4f} (Chuan: < 0.15)")
    print(f"Chamfer Distance: {report['mean_chamfer_distance']:.6f} (Cang nho cang tot)")
    print("=" * 60)

    import json
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danh gia dinh luong CRM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crm_best.pt")
    parser.add_argument("--val_data", type=str, default="data/val")
    parser.add_argument("--report", type=str, default="eval_report.json")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.val_data, args.report)
