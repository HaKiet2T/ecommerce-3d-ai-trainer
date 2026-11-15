"""
Stage 1: LoRA Fine-tune Zero123++ Multi-View Diffusion
======================================================
Kaggle Notebook Script - Chạy trên GPU T4 x2 (32GB VRAM)

Pipeline:
1. Tải checkpoint Zero123++ v1.2 từ HuggingFace
2. Tải CRM checkpoint (cho CCM generation)
3. Load dataset đã render sẵn (6 views + CCM)
4. LoRA fine-tune Zero123++ trên domain sản phẩm E-commerce
5. Lưu LoRA weights vào /kaggle/working/

Tài nguyên:
- GPU: T4 x2 (16GB x 2)
- RAM: ~29GB
- Thời gian: ~8-10h cho 5000 steps
- VRAM usage: ~14GB (LoRA rank 16)
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. Cài thư viện cần thiết (Kaggle cho phép pip install khi enable_internet=true)
# ---------------------------------------------------------------------------

def install_dependencies():
    """Cài đặt dependencies chưa có sẵn trên Kaggle."""
    packages = [
        "diffusers==0.27.2",
        "transformers==4.40.2",
        "accelerate==0.30.1",
        "peft==0.10.0",
        "safetensors",
        "einops",
        "omegaconf",
        "lpips",
        "trimesh",
        "huggingface_hub",
    ]
    for pkg in packages:
        os.system(f"pip install -q {pkg}")

print("[1/7] Installing dependencies...")
install_dependencies()

# Hugging Face Authentication
HF_TOKEN = os.environ.get("HF_TOKEN") or "None"
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    except Exception:
        pass

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("[HF] Authenticated successfully with Hugging Face Hub.")
    except Exception as e:
        print(f"[HF] Note: {e}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# PEFT for LoRA
from peft import LoraConfig, get_peft_model

# Diffusers
from diffusers import DiffusionPipeline, DDIMScheduler

# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------

class TrainConfig:
    # Model (High-Precision Configuration)
    model_name = "sudo-ai/zero123plus-v1.2"
    lora_rank = 32       # Tăng rank lên 32 để học chi tiết sắc nét của gear và ghế công thái học
    lora_alpha = 64      # Tỉ lệ scaling alpha = 2 * rank chuẩn hóa
    lora_target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",  # Attention modules
        "conv1", "conv2",                    # ResNet convolutional blocks
    ]

    # Training (Được tối ưu cho độ chính xác cao nhất)
    batch_size = 1  # T4 16GB
    gradient_accumulation_steps = 8  # Effective batch = 8 giúp ước lượng gradient ổn định
    learning_rate = 8e-5             # LR tinh chỉnh phù hợp cho rank 32
    weight_decay = 0.01
    num_train_steps = 8000           # Mở rộng lên 8000 steps đảm bảo hội tụ sâu
    warmup_steps = 800               # Khởi động làm ấm 10% tổng số steps
    save_every = 1000
    log_every = 50

    # Data
    image_size = 256
    num_views = 6  # CRM canonical views

    # Paths (Kaggle specific)
    dataset_path = "/kaggle/input/crm-ecommerce-views"  # Kaggle dataset
    output_dir = "/kaggle/working/stage1_output"
    checkpoint_dir = "/kaggle/working/stage1_checkpoints"

    # Mixed precision
    mixed_precision = "fp16"
    gradient_checkpointing = True

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_file = "/kaggle/working/stage1_train.log"


config = TrainConfig()

# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------

class MultiViewDataset(Dataset):
    """
    Dataset format:
    dataset_path/
        object_001/
            input.png          # single input image
            view_0.png         # front view
            view_1.png         # right view
            view_2.png         # back view
            view_3.png         # left view
            view_4.png         # top view
            view_5.png         # bottom view
            ccm_0.png          # Color Coordinate Map view 0
            ...
        object_002/
            ...
    """

    def __init__(self, root_dir, image_size=256, split="train"):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        # Lấy danh sách objects
        all_objects = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and (d / "input.png").exists()
        ])

        # Split 90/10
        split_idx = int(len(all_objects) * 0.9)
        if split == "train":
            self.objects = all_objects[:split_idx]
        else:
            self.objects = all_objects[split_idx:]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        print(f"[Dataset] {split}: {len(self.objects)} objects from {root_dir}")

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        obj_dir = self.objects[idx]

        # Load input image
        input_img = Image.open(obj_dir / "input.png").convert("RGB")
        input_tensor = self.transform(input_img)

        # Load 6 target views
        views = []
        for i in range(6):
            view_path = obj_dir / f"view_{i}.png"
            if view_path.exists():
                view = Image.open(view_path).convert("RGB")
            else:
                # Fallback: dùng input nếu thiếu view
                view = input_img
            views.append(self.transform(view))

        target_views = torch.stack(views, dim=0)  # [6, 3, H, W]

        return {
            "input": input_tensor,           # [3, H, W]
            "target_views": target_views,     # [6, 3, H, W]
            "object_id": obj_dir.name
        }


# ---------------------------------------------------------------------------
# 4. Model setup: Zero123++ + LoRA
# ---------------------------------------------------------------------------

def setup_model(config):
    """Load Zero123++ và thêm LoRA adapters."""
    print("[2/7] Loading Zero123++ model...")

    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32,
        trust_remote_code=True,
        token=HF_TOKEN
    )

    # Lấy UNet
    unet = pipe.unet

    # Bật gradient checkpointing để tiết kiệm VRAM
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Thêm LoRA
    print("[3/7] Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=0.05,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)

    # Đếm params
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params (LoRA): {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    unet = unet.to(config.device)

    return pipe, unet


# ---------------------------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------------------------

def train(config):
    """Main training loop."""
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Log
    log_f = open(config.log_file, "w")
    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    log("=" * 60)
    log("CRM Stage 1: LoRA Fine-tune Zero123++")
    log(f"Device: {config.device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            log(f"  GPU {i}: {name} ({mem:.1f} GB)")
    log("=" * 60)

    # Dataset
    log("[4/7] Loading dataset...")
    dataset_path = config.dataset_path

    # Kiểm tra dataset có tồn tại không
    if not Path(dataset_path).exists() or not any(Path(dataset_path).glob("*/input.png")):
        log(f"[WARN] Dataset not found directly at {dataset_path}")
        candidate_paths = [
            Path("/kaggle/input/crm-3d-data-preparation-objaverse/crm_ecommerce_dataset"),
            Path("/kaggle/input/crm-3d-data-preparation-objaverse"),
            Path("/kaggle/input/crm-3d-data-prep/crm_ecommerce_dataset"),
            Path("/kaggle/input/crm-3d-data-prep"),
            Path("/kaggle/input/crm-ecommerce-views"),
            Path("/kaggle/input/crm-ecommerce-dataset"),
            Path("/kaggle/working/crm_ecommerce_dataset"),
        ]
        found = None
        for p in candidate_paths:
            if p.exists() and any(p.glob("*/input.png")):
                found = str(p)
                log(f"[INFO] Found dataset directory at {found}")
                break
            # Kiểm tra file zip từ output của kernel data_prep
            zip_candidates = [
                p.with_suffix(".zip"),
                p / "crm_ecommerce_dataset.zip",
                Path("/kaggle/input/crm-3d-data-preparation-objaverse/crm_ecommerce_dataset.zip"),
                Path("/kaggle/input/crm-3d-data-prep/crm_ecommerce_dataset.zip")
            ]
            for zp in zip_candidates:
                if zp.exists():
                    import zipfile
                    log(f"[INFO] Unzipping dataset from {zp}...")
                    extract_target = Path("/kaggle/working/extracted_dataset")
                    extract_target.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(zp, "r") as zip_ref:
                        zip_ref.extractall(extract_target)
                    found = str(extract_target)
                    log(f"[INFO] Extracted dataset to {found}")
                    break
            if found:
                break

        if found:
            dataset_path = found
        else:
            log("  Tạo dummy dataset để test pipeline...")
            dataset_path = create_dummy_dataset(config)

    train_dataset = MultiViewDataset(dataset_path, config.image_size, split="train")
    val_dataset = MultiViewDataset(dataset_path, config.image_size, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # Model
    pipe, unet = setup_model(config)
    noise_scheduler = pipe.scheduler

    # Optimizer
    log("[5/7] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )

    # LR scheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.num_train_steps, T_mult=1
    )

    # Scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda") if config.mixed_precision == "fp16" else None

    # Training
    log("[6/7] Training...")
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []
    start_time = time.time()

    unet.train()
    epoch = 0

    while global_step < config.num_train_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= config.num_train_steps:
                break

            input_images = batch["input"].to(config.device)       # [B, 3, H, W]
            target_views = batch["target_views"].to(config.device) # [B, 6, 3, H, W]

            B = input_images.shape[0]

            # Flatten views for batch processing
            # [B*6, 3, H, W]
            target_flat = target_views.reshape(B * 6, 3,
                                               config.image_size,
                                               config.image_size)

            # Encode targets to latent
            with torch.no_grad():
                latents = pipe.vae.encode(
                    target_flat.half() if config.mixed_precision == "fp16" else target_flat
                ).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=config.device
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Encode input image as conditioning
            with torch.no_grad():
                cond_latents = pipe.vae.encode(
                    input_images.half() if config.mixed_precision == "fp16" else input_images
                ).latent_dist.sample()
                cond_latents = cond_latents * pipe.vae.config.scaling_factor
                # Repeat for each view
                cond_latents = cond_latents.repeat(6, 1, 1, 1)

            # UNet forward
            with torch.amp.autocast("cuda", enabled=(config.mixed_precision == "fp16")):
                # Concatenate conditioning
                unet_input = torch.cat([noisy_latents, cond_latents], dim=1)
                noise_pred = unet(
                    unet_input,
                    timesteps,
                    return_dict=False
                )[0]

                # MSE loss
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / config.gradient_accumulation_steps

            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

            train_losses.append(loss.item() * config.gradient_accumulation_steps)
            global_step += 1

            # Log
            if global_step % config.log_every == 0:
                avg_loss = sum(train_losses[-config.log_every:]) / config.log_every
                elapsed = (time.time() - start_time) / 60
                lr = optimizer.param_groups[0]["lr"]
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9

                log(
                    f"Step {global_step}/{config.num_train_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                    f"GPU Mem: {gpu_mem:.1f}GB | "
                    f"Time: {elapsed:.1f}min"
                )

            # Save checkpoint
            if global_step % config.save_every == 0:
                ckpt_path = Path(config.checkpoint_dir) / f"lora_step_{global_step}"
                ckpt_path.mkdir(parents=True, exist_ok=True)

                # Lưu LoRA weights
                unet.save_pretrained(str(ckpt_path))

                # Lưu training state
                state = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                    "train_losses": train_losses[-1000:],
                    "config": {
                        "lora_rank": config.lora_rank,
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size
                    }
                }
                torch.save(state, ckpt_path / "training_state.pt")
                log(f"  Saved checkpoint: {ckpt_path}")

                # Dọn VRAM
                gc.collect()
                torch.cuda.empty_cache()

    # Final save
    log("[7/7] Saving final model...")
    final_path = Path(config.output_dir) / "lora_final"
    final_path.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(final_path))

    # Training summary
    total_time = (time.time() - start_time) / 60
    log("=" * 60)
    log("Training Complete!")
    log(f"  Total steps: {global_step}")
    log(f"  Total time: {total_time:.1f} min")
    log(f"  Final loss: {train_losses[-1]:.4f}")
    log(f"  Best val loss: {best_val_loss:.4f}")
    log(f"  Output: {config.output_dir}")
    log("=" * 60)

    # Lưu summary JSON cho automation script
    summary = {
        "stage": "stage1_lora_zero123",
        "status": "complete",
        "total_steps": global_step,
        "total_time_min": round(total_time, 1),
        "final_loss": round(train_losses[-1], 6),
        "best_val_loss": round(best_val_loss, 6),
        "lora_rank": config.lora_rank,
        "output_dir": config.output_dir
    }
    with open(Path(config.output_dir) / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log_f.close()


def create_dummy_dataset(config, num_objects=10):
    """Tạo dummy dataset để test pipeline khi chưa có data thật."""
    dummy_dir = "/kaggle/working/dummy_dataset"
    os.makedirs(dummy_dir, exist_ok=True)

    for i in range(num_objects):
        obj_dir = Path(dummy_dir) / f"object_{i:04d}"
        obj_dir.mkdir(exist_ok=True)

        # Tạo ảnh ngẫu nhiên
        for name in ["input"] + [f"view_{j}" for j in range(6)]:
            img = Image.fromarray(
                np.random.randint(0, 255, (config.image_size, config.image_size, 3),
                                  dtype=np.uint8)
            )
            img.save(obj_dir / f"{name}.png")

    return dummy_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CRM 3D Stage 1: LoRA Fine-tune Zero123++")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 60)

    train(config)
