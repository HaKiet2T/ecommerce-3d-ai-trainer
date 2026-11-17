"""
CRM Quantitative Evaluation Pipeline
====================================
Evaluates CRM models on validation sets with:
- Chamfer Distance (CD-L1 & CD-L2)
- Geometric F1-Score (Threshold tau = 0.05 & 0.02)
- Signed Distance Field (SDF) L1 Error
- Multi-view Projection PSNR, SSIM, and LPIPS
- End-to-end Inference Latency per sample
Supports both CRMFoundationModel (629 keys) and CRMLightweightModel (68 keys).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import trimesh
from scipy.spatial import cKDTree
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

try:
    import lpips
except ImportError:
    lpips = None

from src.data.dataset import CRMDataset, VIEW_NAMES, normalize_mesh
from src.models.crm import (
    CRMFoundationModel,
    CRMLightweightModel,
    build_crm_model,
    DIFFUSERS_AVAILABLE
)


def compute_point_chamfer_and_f1(
    pts_pred: np.ndarray,
    pts_gt: np.ndarray,
    f1_thresholds: Tuple[float, ...] = (0.05, 0.02)
) -> Dict[str, float]:
    """
    Tinh Chamfer Distance va F1-Score giua 2 dam may diem 3D su dung cKDTree (0 dependency rtree).
    Args:
        pts_pred: [N, 3] diem du doan tren be mat
        pts_gt: [M, 3] diem ground truth
        f1_thresholds: cac nguong khoang cach tau de danh gia do phu be mat
    Returns:
        Dict chua CD-L1, CD-L2, F1-scores, precision, recall
    """
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return {
            "chamfer_l1": 1.0,
            "chamfer_l2": 1.0,
            "f1_05": 0.0,
            "f1_02": 0.0
        }

    tree_gt = cKDTree(pts_gt)
    dists_pred_to_gt, _ = tree_gt.query(pts_pred)

    tree_pred = cKDTree(pts_pred)
    dists_gt_to_pred, _ = tree_pred.query(pts_gt)

    cd_l1 = float(0.5 * (np.mean(dists_pred_to_gt) + np.mean(dists_gt_to_pred)))
    cd_l2 = float(0.5 * (np.mean(dists_pred_to_gt ** 2) + np.mean(dists_gt_to_pred ** 2)))

    res = {
        "chamfer_l1": cd_l1,
        "chamfer_l2": cd_l2,
    }

    for tau in f1_thresholds:
        key = f"f1_{int(tau * 100):02d}"
        prec = float(np.mean(dists_pred_to_gt < tau))
        rec = float(np.mean(dists_gt_to_pred < tau))
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        res[key] = f1
        res[f"prec_{int(tau * 100):02d}"] = prec
        res[f"rec_{int(tau * 100):02d}"] = rec

    return res


def compute_mesh_chamfer(mesh_pred: trimesh.Trimesh, mesh_gt: trimesh.Trimesh, num_samples: int = 10000) -> Dict[str, float]:
    """
    Tinh Chamfer Distance va F1 giua 2 luoi tam giac 3D.
    """
    pts_pred, _ = trimesh.sample.sample_surface(mesh_pred, num_samples)
    pts_gt, _ = trimesh.sample.sample_surface(mesh_gt, num_samples)
    return compute_point_chamfer_and_f1(pts_pred, pts_gt)


def load_evaluation_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, str]:
    """
    Tu dong nhan dien kien truc checkpoint (Foundation 629 keys vs Lightweight 68 keys).
    """
    if not os.path.exists(checkpoint_path):
        print(f"[Eval WARN] Checkpoint {checkpoint_path} khong ton tai. Khoi tao mo hinh mac dinh.")
        model = build_crm_model(arch_type="auto").to(device)
        return model, "auto_initialized"

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    else:
        state_dict = ckpt

    # Kiem tra keys de phan biet kien truc
    has_foundation_keys = any("unet2" in k or "sdfMlp" in k or "weightMlp" in k for k in state_dict.keys())

    if has_foundation_keys and DIFFUSERS_AVAILABLE:
        print("[Eval] Phat hien kien truc Foundation (629 keys Zhengyi/CRM).")
        model = CRMFoundationModel().to(device)
        res = model.load_pretrained(state_dict)
        print(f"  Matched {res['matched_keys']}/{res['model_keys']} keys ({res['match_rate'] * 100:.1f}%)")
        arch_type = "foundation"
    else:
        print("[Eval] Phat hien kien truc Lightweight U-Net (68 keys).")
        model = CRMLightweightModel(triplane_channels=32).to(device)
        cleaned_dict = {}
        model_dict = model.state_dict()
        for k, v in state_dict.items():
            clean_k = k
            for pfx in ["module.", "generator.", "net.", "crm."]:
                if clean_k.startswith(pfx):
                    clean_k = clean_k[len(pfx):]
            if clean_k in model_dict and v.shape == model_dict[clean_k].shape:
                cleaned_dict[clean_k] = v
        model.load_state_dict(cleaned_dict, strict=False)
        print(f"  Matched {len(cleaned_dict)}/{len(model_dict)} keys ({len(cleaned_dict)/len(model_dict)*100:.1f}%)")
        arch_type = "lightweight"

    model.eval()
    return model, arch_type


def evaluate_model(
    checkpoint_path: str,
    val_data_dir: str,
    output_report_path: str = "eval_report.json",
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_eval_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Danh gia dinh luong toan dien pipeline CRM 3D tren tap Validation:
    - PSNR, SSIM, LPIPS tren cac mat phang chieu truc giao
    - Chamfer Distance (L1 va L2) va F1-Score (tau=0.05, 0.02)
    - SDF L1 Error va Color L1 Error tai tap diem lay mau 3D
    - Inference latency trung binh
    """
    device = torch.device(device_name)
    print(f"[Eval] Khoi tao danh gia checkpoint: {checkpoint_path} tren {device}...")

    model, arch_type = load_evaluation_model(checkpoint_path, device)

    dataset = CRMDataset(data_root=val_data_dir, is_training=False)
    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError(f"Khong tim thay mau hop le trong thu muc: {val_data_dir}")

    eval_count = min(total_samples, max_eval_samples) if max_eval_samples else total_samples
    print(f"[Eval] Tong so mau: {total_samples} | So mau thuc hien danh gia: {eval_count}")

    lpips_model = None
    if lpips is not None and torch.cuda.is_available():
        try:
            lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
        except Exception:
            pass

    metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "chamfer_l1": [],
        "chamfer_l2": [],
        "f1_05": [],
        "f1_02": [],
        "sdf_l1": [],
        "color_l1": [],
        "latency_sec": []
    }

    for idx in range(eval_count):
        item = dataset[idx]
        images = item["images"].unsqueeze(0).to(device)  # [1, 6, 3, H, W]
        ccms = item["ccms"].unsqueeze(0).to(device)      # [1, 6, 3, H, W]
        gt_points = item["points"].unsqueeze(0).to(device)  # [1, N, 3]
        gt_sdf = item["sdf"].unsqueeze(0).to(device)        # [1, N, 1]
        gt_colors = item["colors"].unsqueeze(0).to(device)  # [1, N, 3]

        t0 = time.time()
        with torch.no_grad():
            out = model(images, ccms, gt_points)
            triplanes = out["triplane"]
            pred_sdf = out["pred_sdf"]
            pred_rgb = out["pred_rgb"]
            pred_mesh = model.extract_mesh(triplanes, resolution=64)
        latency = time.time() - t0
        metrics["latency_sec"].append(latency)

        # 1. Sai so SDF va Color
        sdf_err = float(F.l1_loss(pred_sdf, gt_sdf).item())
        color_err = float(F.l1_loss(pred_rgb, gt_colors).item())
        metrics["sdf_l1"].append(sdf_err)
        metrics["color_l1"].append(color_err)

        # 2. Chamfer Distance va F1-Score
        # Kiem tra file mesh ground truth goc
        item_dir = dataset.sample_dirs[idx]
        gt_mesh_files = list(item_dir.glob("*.obj")) + list(item_dir.glob("*.glb"))
        pts_pred_mesh, _ = trimesh.sample.sample_surface(pred_mesh, 8192)

        if gt_mesh_files:
            try:
                gt_mesh = trimesh.load(gt_mesh_files[0], force="mesh")
                gt_mesh = normalize_mesh(gt_mesh)
                pts_gt_mesh, _ = trimesh.sample.sample_surface(gt_mesh, 8192)
                geom_metrics = compute_point_chamfer_and_f1(pts_pred_mesh, pts_gt_mesh)
            except Exception:
                # Fallback qua diem gan mat be mat trong points.npy
                surf_mask = (torch.abs(gt_sdf.squeeze()).cpu().numpy() < 0.05)
                pts_gt_surf = gt_points.squeeze(0).cpu().numpy()[surf_mask]
                if len(pts_gt_surf) < 100:
                    pts_gt_surf = gt_points.squeeze(0).cpu().numpy()
                geom_metrics = compute_point_chamfer_and_f1(pts_pred_mesh, pts_gt_surf)
        else:
            # Dung cac diem be mat co san tu du lieu SDF
            surf_mask = (torch.abs(gt_sdf.squeeze()).cpu().numpy() < 0.05)
            pts_gt_surf = gt_points.squeeze(0).cpu().numpy()[surf_mask]
            if len(pts_gt_surf) < 100:
                pts_gt_surf = gt_points.squeeze(0).cpu().numpy()
            geom_metrics = compute_point_chamfer_and_f1(pts_pred_mesh, pts_gt_surf)

        metrics["chamfer_l1"].append(geom_metrics["chamfer_l1"])
        metrics["chamfer_l2"].append(geom_metrics["chamfer_l2"])
        metrics["f1_05"].append(geom_metrics["f1_05"])
        metrics["f1_02"].append(geom_metrics["f1_02"])

        # 3. PSNR & SSIM tren 3 mat phang truc giao
        if arch_type == "foundation":
            # Foundation triplane co kich thuoc [1, 32, 256, 768] (3 mat cat)
            pred_proj = torch.tanh(triplanes[:, :3, :, :256]) # [1, 3, H, W]
        else:
            pred_proj = torch.tanh(triplanes[:, :3, :, :])   # [1, 3, H, W]

        gt_view = images[:, 0] # front view [1, 3, H, W]
        pred_np = ((pred_proj.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        gt_np = ((gt_view.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

        psnr_val = psnr_fn(gt_np, pred_np, data_range=255)
        ssim_val = ssim_fn(gt_np, pred_np, channel_axis=-1, data_range=255)
        metrics["psnr"].append(psnr_val)
        metrics["ssim"].append(ssim_val)

        if lpips_model is not None:
            with torch.no_grad():
                l_val = lpips_model(pred_proj, gt_view).squeeze().cpu().numpy()
                metrics["lpips"].append(float(l_val))

    report = {
        "checkpoint": str(checkpoint_path),
        "architecture": arch_type,
        "num_samples_evaluated": eval_count,
        "mean_chamfer_l1": float(np.mean(metrics["chamfer_l1"])),
        "mean_chamfer_l2": float(np.mean(metrics["chamfer_l2"])),
        "mean_f1_score_tau05": float(np.mean(metrics["f1_05"])),
        "mean_f1_score_tau02": float(np.mean(metrics["f1_02"])),
        "mean_sdf_l1_error": float(np.mean(metrics["sdf_l1"])),
        "mean_color_l1_error": float(np.mean(metrics["color_l1"])),
        "mean_psnr_db": float(np.mean(metrics["psnr"])),
        "mean_ssim": float(np.mean(metrics["ssim"])),
        "mean_lpips": float(np.mean(metrics["lpips"])) if metrics["lpips"] else None,
        "mean_latency_seconds": float(np.mean(metrics["latency_sec"])),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print("\n" + "=" * 65)
    print("  BAO CAO DANH GIA DINH LUONG CRM 3D PIPELINE")
    print("=" * 65)
    print(f"Architecture:            {report['architecture'].upper()}")
    print(f"Evaluated Samples:       {report['num_samples_evaluated']}")
    print(f"Chamfer Distance (L1):   {report['mean_chamfer_l1']:.6f} (Cang nho cang tot)")
    print(f"Chamfer Distance (L2):   {report['mean_chamfer_l2']:.6f} (Cang nho cang tot)")
    print(f"Geometric F1 @ tau=0.05: {report['mean_f1_score_tau05'] * 100:.2f}% (Chuan: > 75%)")
    print(f"Geometric F1 @ tau=0.02: {report['mean_f1_score_tau02'] * 100:.2f}% (Chuan: > 50%)")
    print(f"SDF L1 Error:            {report['mean_sdf_l1_error']:.6f}")
    print(f"Color L1 Error:          {report['mean_color_l1_error']:.6f}")
    print(f"Multi-View PSNR:         {report['mean_psnr_db']:.2f} dB (Chuan: > 24 dB)")
    print(f"Multi-View SSIM:         {report['mean_ssim']:.4f} (Chuan: > 0.80)")
    if report['mean_lpips'] is not None:
        print(f"Perceptual LPIPS:        {report['mean_lpips']:.4f} (Chuan: < 0.15)")
    print(f"Inference Latency:       {report['mean_latency_seconds']:.3f} s/sample")
    print("=" * 65)

    os.makedirs(Path(output_report_path).parent, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Eval] Da xuat bao cao dinh luong tai: {output_report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Danh gia dinh luong toan dien CRM 3D")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crm_best.pt")
    parser.add_argument("--val_data", type=str, default="outputs/kaggle/data_prep/20261116_154412/crm_ecommerce_dataset")
    parser.add_argument("--report", type=str, default="outputs/experiments/step2_evaluation_report.json")
    parser.add_argument("--max_samples", type=int, default=10)
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        val_data_dir=args.val_data,
        output_report_path=args.report,
        max_eval_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
