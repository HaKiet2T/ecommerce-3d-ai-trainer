"""
CRM Training Pipeline
=====================
Pipeline huan luyen Convolutional Reconstruction Model (CRM) tren tap du lieu E-commerce.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import trimesh

try:
    import yaml
except ImportError:
    yaml = None

try:
    import lpips
except ImportError:
    lpips = None

from src.data.dataset import CRMDataset, build_dataloader
from src.models.crm import ConvolutionalReconstructionModel

def compute_chamfer_distance(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor
) -> torch.Tensor:
    """
    Tinh Chamfer Distance giua 2 dam may diem 3D:
    CD = 1/|P| * sum(min ||p - q||_2^2) + 1/|Q| * sum(min ||q - p||_2^2)
    Args:
        points_pred: [B, N, 3]
        points_gt:   [B, M, 3]
    Returns:
        loss: scalar tensor
    """
    diff = points_pred.unsqueeze(2) - points_gt.unsqueeze(1) # [B, N, M, 3]
    dist_sq = torch.sum(diff ** 2, dim=-1) # [B, N, M]

    min_dist_pred_to_gt = torch.min(dist_sq, dim=2)[0] # [B, N]
    min_dist_gt_to_pred = torch.min(dist_sq, dim=1)[0] # [B, M]

    loss = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)
    return loss

class CRMLoss(nn.Module):
    """
    Ham mat mat ket hop da thanh phan cho CRM:
    1. L_sdf: Mat mat L1 tren truong khoang cach co dau SDF
    2. L_cd:  Chamfer Distance giua be mat du doan va be mat thuc te
    3. L_rgb: Mat mat L1 giua mau sac du doan va mau sac thuc te
    4. L_lpips: Mat mat tri giac LPIPS (VGG) giup texture sac net
    """
    def __init__(
        self,
        weight_sdf: float = 10.0,
        weight_cd: float = 2.0,
        weight_rgb: float = 5.0,
        weight_lpips: float = 1.0,
        device: str = "cuda"
    ):
        super().__init__()
        self.weight_sdf = weight_sdf
        self.weight_cd = weight_cd
        self.weight_rgb = weight_rgb
        self.weight_lpips = weight_lpips

        if lpips is not None and torch.cuda.is_available():
            self.lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
        else:
            self.lpips_fn = None

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_dict: ket qua du doan gom 'triplanes', 'pred_sdf', 'pred_rgb'
            batch: batch du lieu nap tu DataLoader
        """
        pred_sdf = pred_dict["pred_sdf"] # [B, N, 1]
        target_sdf = batch["sdf"].to(pred_sdf.device) # [B, N, 1]

        # 1. SDF L1 Loss
        sdf_loss = F.l1_loss(pred_sdf, target_sdf)

        # 2. Chamfer Distance Loss tren cac diem co SDF gan 0 (be mat)
        pts = batch["points"].to(pred_sdf.device)
        is_surface = (torch.abs(target_sdf.squeeze(-1)) < 0.05)

        cd_loss = torch.tensor(0.0, device=pred_sdf.device)
        if torch.sum(is_surface) > 100:
            surf_pts_gt = pts[is_surface].unsqueeze(0) # [1, K, 3]
            pred_surf_mask = (torch.abs(pred_sdf.squeeze(-1)) < 0.05)
            if torch.sum(pred_surf_mask) > 100:
                surf_pts_pred = pts[pred_surf_mask].unsqueeze(0)
                cd_loss = compute_chamfer_distance(surf_pts_pred, surf_pts_gt)

        # 3. Mau sac RGB Loss (L1)
        pred_rgb = pred_dict.get("pred_rgb", None)
        rgb_loss = torch.tensor(0.0, device=pred_sdf.device)
        if pred_rgb is not None and "points_rgb" in batch:
            target_rgb = batch["points_rgb"].to(pred_rgb.device)
            rgb_loss = F.l1_loss(pred_rgb, target_rgb)

        # 4. LPIPS Loss tren anh mat phang Triplane chieu
        lpips_val = torch.tensor(0.0, device=pred_sdf.device)
        if self.lpips_fn is not None and "triplanes" in pred_dict:
            triplanes = pred_dict["triplanes"] # [B, 3, C, H, W]
            proj_rgb = torch.tanh(triplanes[:, :, :3, :, :]) # [B, 3, 3, H, W]
            B = proj_rgb.shape[0]
            ref_views = batch["images"][:, [0, 4, 1]].to(pred_sdf.device) # [B, 3, 3, H, W]
            p_flat = proj_rgb.view(B * 3, 3, proj_rgb.shape[-2], proj_rgb.shape[-1])
            r_flat = ref_views.view(B * 3, 3, ref_views.shape[-2], ref_views.shape[-1])
            lpips_val = torch.mean(self.lpips_fn(p_flat, r_flat))

        total_loss = (
            self.weight_sdf * sdf_loss +
            self.weight_cd * cd_loss +
            self.weight_rgb * rgb_loss +
            self.weight_lpips * lpips_val
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_sdf": sdf_loss.item(),
            "loss_cd": cd_loss.item() if isinstance(cd_loss, torch.Tensor) else 0.0,
            "loss_rgb": rgb_loss.item() if isinstance(rgb_loss, torch.Tensor) else 0.0,
            "loss_lpips": lpips_val.item() if isinstance(lpips_val, torch.Tensor) else 0.0
        }
        return total_loss, metrics

def train_crm(
    data_dir: str,
    output_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    triplane_resolution: int = 256,
    triplane_channels: int = 32,
    resume_checkpoint: Optional[str] = None
):
    """
    Pipeline huan luyen CRM toan dien tren GPU don le (T4 16GB hoac A100):
    1. Khoi tao mo hinh CRM voi kien truc MultiViewUNet va TriplaneSDFDecoder
    2. Chay Training Loop voi Automatic Mixed Precision (AMP)
    3. Luu checkpoint va xuat mau Mesh kiem tra dinh ky
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Khoi tao Huan luyen CRM tren Thiet bi: {device}")
    print(f"Thu muc du lieu: {data_dir} | Output: {output_dir}")
    print("=" * 60)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mesh_out_path = out_path / "eval_meshes"
    mesh_out_path.mkdir(exist_ok=True)

    # 1. Khoi tao DataLoader
    print("[1/4] Dang tai bo du lieu E-commerce...")
    train_loader = build_dataloader(
        data_root=data_dir,
        batch_size=batch_size,
        num_workers=4 if torch.cuda.is_available() else 0,
        is_training=True
    )
    print(f"[Dataset] So luong batch moi epoch: {len(train_loader)}")

    # 2. Khoi tao mo hinh CRM
    print("[2/4] Khoi tao mang Convolutional Reconstruction Model...")
    model = ConvolutionalReconstructionModel(
        triplane_resolution=triplane_resolution,
        triplane_channels=triplane_channels,
        base_features=64,
        hidden_dim=64
    ).to(device)

    start_epoch = 1
    best_loss = float("inf")

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"-> Dang nạp checkpoint tiep tuc: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("loss", float("inf"))

    # 3. Optimizer, Scheduler va Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    criterion = CRMLoss(device=str(device))

    # 4. Training Loop
    print("[3/4] Bat dau qua trinh toi uu hoa (Training Loop)...")
    model.train()

    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device) # [B, 6, 3, H, W]
            ccms = batch["ccms"].to(device)     # [B, 6, 3, H, W]
            points = batch["points"].to(device) # [B, N, 3]

            optimizer.zero_grad()

            with autocast(enabled=(device.type == "cuda")):
                # Forward: sinh triplanes tu 6 goc nhin
                triplanes = model.unet(images, ccms)
                # Truy van gia tri SDF va mau RGB tai tap diem 3D
                pred_sdf, pred_rgb = model.query_sdf_and_rgb(triplanes, points)

                pred_dict = {
                    "triplanes": triplanes,
                    "pred_sdf": pred_sdf,
                    "pred_rgb": pred_rgb
                }
                loss, metrics = criterion(pred_dict, batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % 10 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] "
                    f"Total Loss: {metrics['loss_total']:.4f} | "
                    f"SDF: {metrics['loss_sdf']:.4f} | "
                    f"CD: {metrics['loss_cd']:.4f} | "
                    f"LR: {cur_lr:.6f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch} Done] Trung binh Loss: {avg_loss:.4f} - Thoi gian: {elapsed:.1f}s")

        # Luu checkpoint moi 5 epoch hoac khi loss tot nhat
        if avg_loss < best_loss or epoch % 5 == 0:
            best_loss = min(avg_loss, best_loss)
            ckpt_file = out_path / f"crm_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_file)
            print(f"[Checkpoint] Da luu model tai: {ckpt_file}")

            # Kiem dinh truc quan: sinh mesh mau de danh gia
            model.eval()
            with torch.no_grad():
                val_triplanes = model.unet(images[:1], ccms[:1])
                mesh = model.extract_mesh(val_triplanes, resolution=64)
                mesh_path = mesh_out_path / f"val_epoch_{epoch:03d}.obj"
                mesh.export(str(mesh_path))
                print(f"[Visual Mesh] Da xuat mesh kiem tra tai: {mesh_path}")
            model.train()

    print("=" * 60)
    print("Huan luyen CRM hoan tat thanh cong!")
    print(f"Model tot nhat luu tai thu muc: {output_dir}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Huan luyen CRM 3D")
    parser.add_argument("--config", type=str, default=None, help="Duong dan file cau hinh YAML")
    parser.add_argument("--data_dir", type=str, default="data/ecommerce_processed", help="Thu muc du lieu")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Thu muc luu weights")
    parser.add_argument("--epochs", type=int, default=30, help="So epoch")
    parser.add_argument("--batch_size", type=int, default=2, help="Kich thuoc batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Duong dan checkpoint tiep tuc")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    resume = args.resume

    # Nap cau hinh tu file YAML neu duoc chi dinh
    if args.config and os.path.exists(args.config) and yaml is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if "data" in cfg and "dataset_path" in cfg["data"]:
            data_dir = cfg["data"]["dataset_path"]
        if "output" in cfg and "checkpoint_dir" in cfg["output"]:
            output_dir = cfg["output"]["checkpoint_dir"]
        if "training" in cfg:
            tcfg = cfg["training"]
            epochs = tcfg.get("epochs", epochs)
            batch_size = tcfg.get("batch_size", batch_size)
            lr = float(tcfg.get("learning_rate", lr))

    train_crm(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        resume_checkpoint=resume
    )

if __name__ == "__main__":
    main()
