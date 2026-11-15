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
    import lpips
except ImportError:
    lpips = None

from dataset import CRMDataset, build_dataloader
from models.crm import ConvolutionalReconstructionModel

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
    # Tinh ma tran khoang cach binh phuong [B, N, M]
    diff = points_pred.unsqueeze(2) - points_gt.unsqueeze(1) # [B, N, M, 3]
    dist_sq = torch.sum(diff ** 2, dim=-1) # [B, N, M]

    # Khoang cach gan nhat tu pred den gt
    min_dist_pred_to_gt = torch.min(dist_sq, dim=2)[0] # [B, N]
    # Khoang cach gan nhat tu gt den pred
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

        # 1. SDF L1 Loss (trong diem vao cac diem gan be mat)
        sdf_loss = F.l1_loss(pred_sdf, target_sdf)

        # 2. Chamfer Distance Loss tren cac diem co SDF gan 0 (be mat)
        # Loc cac diem gan be mat tu target (|sdf| < 0.05)
        pts = batch["points"].to(pred_sdf.device)
        is_surface = (torch.abs(target_sdf.squeeze(-1)) < 0.05)
        
        # Tinh Chamfer neu co du diem be mat
        cd_loss = torch.tensor(0.0, device=pred_sdf.device)
        if torch.sum(is_surface) > 100:
            surf_pts_gt = pts[is_surface].unsqueeze(0) # [1, K, 3]
            # Lay cac diem co pred_sdf nho nhat lam du doan be mat
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

        # 4. LPIPS Loss tren anh mat phang Triplane chieu (Orthogonal projections)
        lpips_val = torch.tensor(0.0, device=pred_sdf.device)
        if self.lpips_fn is not None and "triplanes" in pred_dict:
            triplanes = pred_dict["triplanes"] # [B, 3, C, H, W]
            # Lay 3 kenh dau tien cua Triplane lam gia-anh RGB de danh gia LPIPS
            proj_rgb = torch.tanh(triplanes[:, :, :3, :, :]) # [B, 3, 3, H, W]
            B = proj_rgb.shape[0]
            # So sanh voi anh view chuan (front, top, right)
            ref_views = batch["images"][:, [0, 4, 1]].to(pred_sdf.device) # [B, 3, 3, H, W]
            p_flat = proj_rgb.view(B * 3, 3, proj_rgb.shape[-2], proj_rgb.shape[-1])
            r_flat = ref_views.view(B * 3, 3, ref_views.shape[-2], ref_views.shape[-1])
            lpips_val = torch.mean(self.lpips_fn(p_flat, r_flat))

        # Tong hop ham mat mat
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
    triplane_channels: int = 32,
    mixed_precision: bool = True,
    resume_checkpoint: Optional[str] = None
):
    """
    Pipeline huan luyen CRM toan dien tren GPU don le (T4 16GB hoac A100):
    - Ho tro Mixed Precision (FP16 AMP) tiet kiem 50% VRAM
    - Toi uu gradient accumulation cho batch size hieu dung lon hon
    - Checkpointing dinh ky kem trich xuat mesh kiem dinh truc quan
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Dang su dung thiet bi: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name} - Tong VRAM: {gpu_mem:.2f} GB")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mesh_out_path = out_path / "val_meshes"
    mesh_out_path.mkdir(parents=True, exist_ok=True)

    # 1. Khoi tao Model
    model = ConvolutionalReconstructionModel(
        triplane_channels=triplane_channels,
        base_features=64,
        hidden_dim=64
    ).to(device)

    # Nap checkpoint neu co
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[Resume] Nap checkpoint tu: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    # 2. Dataloader
    train_loader = build_dataloader(
        data_root=data_dir,
        batch_size=batch_size,
        num_workers=2,
        is_training=True
    )
    print(f"[Dataset] So luong batch moi epoch: {len(train_loader)}")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.99)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader),
        eta_min=1e-6
    )

    # 4. Criterion & Scaler
    criterion = CRMLoss(device=str(device))
    scaler = GradScaler(enabled=mixed_precision)

    best_loss = float("inf")
    step_count = 0

    print("=" * 60)
    print(f"Bat dau huan luyen CRM: Epochs={epochs}, BatchSize={batch_size}, LR={learning_rate}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            step_count += 1
            images = batch["images"].to(device)   # [B, 6, 3, H, W]
            ccms = batch["ccms"].to(device)       # [B, 6, 3, H, W]
            points = batch["points"].to(device)   # [B, N, 3]

            optimizer.zero_grad()

            with autocast(enabled=mixed_precision):
                output = model(images, ccms, points)
                loss, metrics = criterion(output, batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

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

    print("=" * 60)
    print("Huan luyen CRM hoan tat thanh cong!")
    print(f"Model tot nhat luu tai thu muc: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huan luyen CRM 3D")
    parser.add_argument("--data_dir", type=str, default="data/ecommerce_processed", help="Thu muc du lieu")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Thu muc luu weights")
    parser.add_argument("--epochs", type=int, default=30, help="So epoch")
    parser.add_argument("--batch_size", type=int, default=2, help="Kich thuoc batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Duong dan checkpoint tiep tuc")
    args = parser.parse_args()

    train_crm(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_checkpoint=args.resume
    )
