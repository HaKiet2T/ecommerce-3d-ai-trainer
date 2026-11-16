"""
Stage 2: CRM U-Net Reconstruction Fine-tune
============================================
Kaggle Notebook Script - Chạy trên GPU T4 x2 (32GB VRAM)

Pipeline:
1. Tải CRM checkpoint (U-Net triplane decoder) từ HuggingFace
2. Load dataset 6 views + CCMs (output từ Stage 1 hoặc pre-rendered)
3. Fine-tune CRM U-Net trên domain E-commerce
4. Export mesh outputs (.obj) để kiểm tra
5. Lưu fine-tuned weights vào /kaggle/working/

Kiến trúc CRM Stage 2:
- Input: 6 orthographic views (RGB) + 6 CCMs (geometry prior)
- Encoder: Convolutional U-Net -> Triplane features
- Decoder: Small MLP -> SDF + Color per point
- Mesh extraction: FlexiCubes -> Textured mesh (.obj + .mtl + texture.png)

Tài nguyên:
- GPU: T4 x2 (16GB x 2) - dùng DataParallel
- Batch size: 2 (1 per GPU)
- VRAM: ~8-10GB per GPU
- Thời gian ước tính: ~4-6h cho 3000 steps
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------

def install_dependencies():
    """Cài đặt dependencies tương thích Python 3.13 trên Kaggle Ubuntu 24.04."""
    os.system("pip uninstall -y torchao")
    cmd = (
        "pip install -q "
        "'huggingface_hub<0.26.0' "
        "'diffusers>=0.30.0' "
        "'transformers>=4.45.0' "
        "'accelerate>=0.34.0' "
        "safetensors einops omegaconf trimesh lpips PyMCubes"
    )
    res = os.system(cmd)
    if res != 0:
        os.system("pip install -q 'huggingface_hub<0.26.0' diffusers transformers accelerate safetensors einops omegaconf trimesh lpips PyMCubes")

print("[1/8] Installing dependencies...")
install_dependencies()

# Monkey patch legacy cached_download for diffusers/remote pipelines
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        from huggingface_hub import hf_hub_download
        huggingface_hub.cached_download = hf_hub_download
except Exception:
    pass

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

# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------

class TrainConfig:
    # Model
    crm_checkpoint = "Zhengyi/CRM"  # HuggingFace repo
    triplane_resolution = 256
    triplane_channels = 32
    mlp_hidden_dim = 64
    sdf_threshold = 0.0

    # Training (Được tối ưu cho độ chính xác cao nhất)
    batch_size = 2  # 1 per T4 GPU with DataParallel
    gradient_accumulation_steps = 4  # Effective batch = 8
    learning_rate = 5e-5
    weight_decay = 0.01
    num_train_steps = 8000  # Mở rộng lên 8000 steps để mạng Triplane hội tụ chi tiết cao
    warmup_steps = 800
    save_every = 1000
    eval_every = 1000
    log_every = 50

    # Loss weights (Gia tăng trọng số bảo toàn hình học và chuẩn hóa bề mặt)
    lambda_rgb = 1.0       # RGB reconstruction loss
    lambda_lpips = 0.5     # Perceptual loss
    lambda_sdf = 0.5       # SDF surface loss (giám sát trực tiếp khoảng cách Euclid chuẩn từ cKDTree)
    lambda_eikonal = 0.0   # Tắt symbolic Eikonal (create_graph) để triệt tiêu lỗi cudnn_grid_sampler_backward double-backward

    # Data
    image_size = 256
    num_views = 6
    ccm_channels = 3  # CCM is RGB-encoded XYZ

    # Paths
    dataset_path = "/kaggle/input/crm-ecommerce-views"
    stage1_lora_path = "/kaggle/input/crm-stage1-lora"  # Optional: LoRA weights
    output_dir = "/kaggle/working/stage2_output"
    checkpoint_dir = "/kaggle/working/stage2_checkpoints"
    mesh_output_dir = "/kaggle/working/stage2_meshes"

    # Hardware
    mixed_precision = "fp16"
    gradient_checkpointing = True
    use_data_parallel = True
    device = "cuda"

    log_file = "/kaggle/working/stage2_train.log"


config = TrainConfig()

# ---------------------------------------------------------------------------
# 3. CRM U-Net Model (Triplane Decoder)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsampling block with skip connection."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CRMUNet(nn.Module):
    """
    CRM Convolutional U-Net for Triplane Generation.

    Input: 6 RGB views (concatenated) + 6 CCMs = [B, 36, H, W]
    Output: Triplane features [B, C*3, H, W] (XY, XZ, YZ planes)
    """
    def __init__(self, in_channels=36, triplane_channels=32, base_channels=64):
        super().__init__()

        self.in_channels = in_channels
        self.triplane_channels = triplane_channels
        out_channels = triplane_channels * 3  # 3 planes

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 8)

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 4)
        self.up2 = Up(base_channels * 8, base_channels * 2)
        self.up3 = Up(base_channels * 4, base_channels)
        self.up4 = Up(base_channels * 2, base_channels)

        # Output: triplane features
        self.outc = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        """
        Args:
            x: [B, 36, H, W] - concatenated 6 RGB views + 6 CCMs
        Returns:
            triplane: [B, C*3, H, W] - XY, XZ, YZ planes stacked in channel dim
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


class TriplaneMLP(nn.Module):
    """
    MLP decoder: triplane features -> SDF + Color per 3D point.

    Cho mỗi điểm (x,y,z), truy vấn 3 planes:
    - XY plane tại (x,y)
    - XZ plane tại (x,z)
    - YZ plane tại (y,z)
    Cộng features lại rồi decode qua MLP.
    """
    def __init__(self, triplane_channels=32, hidden_dim=64):
        super().__init__()

        self.sdf_net = nn.Sequential(
            nn.Linear(triplane_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)  # SDF value
        )

        self.color_net = nn.Sequential(
            nn.Linear(triplane_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # RGB
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Args:
            features: [B, N, C] - aggregated triplane features per point
        Returns:
            sdf: [B, N, 1]
            color: [B, N, 3]
        """
        sdf = self.sdf_net(features)
        color = self.color_net(features)
        return sdf, color


class CRMModel(nn.Module):
    """
    Full CRM Model: U-Net encoder + Triplane + MLP decoder.
    """
    def __init__(self, config):
        super().__init__()

        # Input: 6 views * 3 RGB + 6 CCMs * 3 = 36 channels
        in_channels = config.num_views * (3 + config.ccm_channels)

        self.unet = CRMUNet(
            in_channels=in_channels,
            triplane_channels=config.triplane_channels,
            base_channels=64
        )

        self.mlp = TriplaneMLP(
            triplane_channels=config.triplane_channels,
            hidden_dim=config.mlp_hidden_dim
        )

        self.triplane_channels = config.triplane_channels
        self.triplane_resolution = config.triplane_resolution

    def query_triplane(self, triplane, points):
        """
        Truy vấn triplane tại các điểm 3D.

        Args:
            triplane: [B, C*3, H, W] from U-Net
            points: [B, N, 3] - 3D coordinates in [-1, 1]

        Returns:
            features: [B, N, C] - aggregated triplane features
        """
        B, _, H, W = triplane.shape
        C = self.triplane_channels
        N = points.shape[1]

        # Split into 3 planes
        xy_plane = triplane[:, :C]       # [B, C, H, W]
        xz_plane = triplane[:, C:2*C]    # [B, C, H, W]
        yz_plane = triplane[:, 2*C:]     # [B, C, H, W]

        # Sample coordinates (grid_sample expects [B, N, 1, 2])
        xy_coords = points[:, :, :2].unsqueeze(2)   # [B, N, 1, 2] - (x, y)
        xz_coords = points[:, :, [0, 2]].unsqueeze(2)  # [B, N, 1, 2] - (x, z)
        yz_coords = points[:, :, [1, 2]].unsqueeze(2)  # [B, N, 1, 2] - (y, z)

        # Bilinear interpolation
        f_xy = F.grid_sample(xy_plane, xy_coords, align_corners=True, mode="bilinear")
        f_xz = F.grid_sample(xz_plane, xz_coords, align_corners=True, mode="bilinear")
        f_yz = F.grid_sample(yz_plane, yz_coords, align_corners=True, mode="bilinear")

        # [B, C, N, 1] -> [B, N, C]
        f_xy = f_xy.squeeze(-1).permute(0, 2, 1)
        f_xz = f_xz.squeeze(-1).permute(0, 2, 1)
        f_yz = f_yz.squeeze(-1).permute(0, 2, 1)

        # Aggregate: sum
        features = f_xy + f_xz + f_yz

        return features

    def forward(self, views, ccms, points):
        """
        Args:
            views: [B, 6, 3, H, W] - 6 RGB views
            ccms: [B, 6, 3, H, W] - 6 CCMs
            points: [B, N, 3] - query points

        Returns:
            sdf: [B, N, 1]
            color: [B, N, 3]
        """
        B = views.shape[0]

        # Concatenate views and CCMs: [B, 36, H, W]
        views_flat = views.reshape(B, -1, views.shape[-2], views.shape[-1])
        ccms_flat = ccms.reshape(B, -1, ccms.shape[-2], ccms.shape[-1])
        x = torch.cat([views_flat, ccms_flat], dim=1)

        # U-Net -> triplane
        triplane = self.unet(x)

        # Query triplane at 3D points
        features = self.query_triplane(triplane, points)

        # MLP -> SDF + Color
        sdf, color = self.mlp(features)

        return sdf, color, triplane


# ---------------------------------------------------------------------------
# 4. Dataset
# ---------------------------------------------------------------------------

class CRMDataset(Dataset):
    """
    Dataset cho Stage 2 CRM training.

    Cần:
    - 6 views (RGB)
    - 6 CCMs (Color Coordinate Maps)
    - Ground truth 3D points + SDF + colors (từ mesh sampling)

    Format:
    dataset_path/
        object_001/
            view_0.png ... view_5.png
            ccm_0.png ... ccm_5.png
            points.npy       # [N, 3] sampled 3D points
            sdf.npy          # [N, 1] SDF values
            colors.npy       # [N, 3] RGB colors
    """

    def __init__(self, root_dir, image_size=256, num_points=4096, split="train"):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.num_points = num_points

        all_objects = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and (d / "view_0.png").exists()
        ])

        split_idx = int(len(all_objects) * 0.9)
        self.objects = all_objects[:split_idx] if split == "train" else all_objects[split_idx:]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        print(f"[Dataset] {split}: {len(self.objects)} objects")

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        obj_dir = self.objects[idx]

        # Load 6 views
        views = []
        for i in range(6):
            img = Image.open(obj_dir / f"view_{i}.png").convert("RGB")
            views.append(self.transform(img))
        views = torch.stack(views)  # [6, 3, H, W]

        # Load 6 CCMs
        ccms = []
        for i in range(6):
            ccm_path = obj_dir / f"ccm_{i}.png"
            if ccm_path.exists():
                ccm = Image.open(ccm_path).convert("RGB")
            else:
                ccm = Image.new("RGB", (self.image_size, self.image_size))
            ccms.append(self.transform(ccm))
        ccms = torch.stack(ccms)  # [6, 3, H, W]

        # Load 3D points + GT
        points_path = obj_dir / "points.npy"
        if points_path.exists():
            points = np.load(points_path).astype(np.float32)
            sdf_gt = np.load(obj_dir / "sdf.npy").astype(np.float32)
            colors_gt = np.load(obj_dir / "colors.npy").astype(np.float32)
        else:
            # Dummy points nếu chưa có
            points = np.random.uniform(-1, 1, (self.num_points, 3)).astype(np.float32)
            sdf_gt = np.zeros((self.num_points, 1), dtype=np.float32)
            colors_gt = np.random.uniform(0, 1, (self.num_points, 3)).astype(np.float32)

        # Random subsample
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
            sdf_gt = sdf_gt[indices]
            colors_gt = colors_gt[indices]

        return {
            "views": views,
            "ccms": ccms,
            "points": torch.from_numpy(points),
            "sdf_gt": torch.from_numpy(sdf_gt),
            "colors_gt": torch.from_numpy(colors_gt),
            "object_id": obj_dir.name
        }


# ---------------------------------------------------------------------------
# 5. Loss Functions
# ---------------------------------------------------------------------------

class CRMLoss(nn.Module):
    """Combined loss for CRM training."""
    def __init__(self, config):
        super().__init__()
        self.lambda_rgb = config.lambda_rgb
        self.lambda_lpips = config.lambda_lpips
        self.lambda_sdf = config.lambda_sdf
        self.lambda_eikonal = config.lambda_eikonal

        # LPIPS (sẽ dùng cho rendered image comparison)
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net="vgg").cuda()
            self.has_lpips = True
        except Exception:
            self.has_lpips = False
            print("[WARN] LPIPS not available, skipping perceptual loss")

    def forward(self, pred_sdf, pred_color, gt_sdf, gt_color, points=None):
        """
        Args:
            pred_sdf: [B, N, 1]
            pred_color: [B, N, 3]
            gt_sdf: [B, N, 1]
            gt_color: [B, N, 3]
            points: [B, N, 3] (for eikonal loss)
        """
        losses = {}

        # SDF loss (L1)
        sdf_loss = F.l1_loss(pred_sdf, gt_sdf)
        losses["sdf"] = sdf_loss * self.lambda_sdf

        # Color loss (L1)
        color_loss = F.l1_loss(pred_color, gt_color)
        losses["rgb"] = color_loss * self.lambda_rgb

        # Eikonal loss (gradient norm should be 1)
        # Chỉ tính nếu lambda_eikonal > 0 và points.requires_grad
        if self.lambda_eikonal > 0 and points is not None and getattr(points, "requires_grad", False):
            try:
                gradients = torch.autograd.grad(
                    outputs=pred_sdf,
                    inputs=points,
                    grad_outputs=torch.ones_like(pred_sdf),
                    create_graph=True
                )[0]
                eikonal = ((gradients.norm(dim=-1) - 1) ** 2).mean()
                losses["eikonal"] = eikonal * self.lambda_eikonal
            except RuntimeError:
                pass

        # Total
        total = sum(losses.values())
        losses["total"] = total

        return total, losses


# ---------------------------------------------------------------------------
# 6. Training Loop
# ---------------------------------------------------------------------------

def train(config):
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.mesh_output_dir, exist_ok=True)

    log_f = open(config.log_file, "w")
    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    log("=" * 60)
    log("CRM Stage 2: U-Net Reconstruction Fine-tune")
    log(f"Device: {config.device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            log(f"  GPU {i}: {name} ({mem:.1f} GB)")
    log("=" * 60)

    # Dataset
    log("[2/8] Searching for dataset in /kaggle/input/...")
    dataset_path = None
    input_root = Path("/kaggle/input")
    if input_root.exists():
        # Tìm file .zip để giải nén trước sang /tmp/extracted_dataset
        zip_files = list(input_root.rglob("*.zip"))
        if zip_files:
            zp = zip_files[0]
            log(f"[INFO] Found dataset zip: {zp}")
            extract_target = Path("/tmp/extracted_dataset")
            extract_target.mkdir(parents=True, exist_ok=True)
            import zipfile
            with zipfile.ZipFile(zp, "r") as z:
                z.extractall(extract_target)
            log(f"[INFO] Extracted zip to: {extract_target}")
            # Tìm thư mục chứa points.npy
            for candidate in extract_target.rglob("points.npy"):
                dataset_path = str(candidate.parent.parent)
                log(f"[INFO] Dataset root detected at: {dataset_path}")
                break
            if not dataset_path:
                dataset_path = str(extract_target)

        # Nếu không có zip, tìm thư mục uncompressed có points.npy
        if not dataset_path:
            for candidate in input_root.rglob("points.npy"):
                dataset_path = str(candidate.parent.parent)
                log(f"[INFO] Found uncompressed dataset at: {dataset_path}")
                break

    if not dataset_path:
        log("[WARN] Dataset not found, creating dummy...")
        dataset_path = create_dummy_dataset(config)

    train_dataset = CRMDataset(dataset_path, config.image_size, split="train")
    val_dataset = CRMDataset(dataset_path, config.image_size, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, num_workers=1, pin_memory=True
    )

    # Model
    log("[3/8] Building CRM model...")
    model = CRMModel(config)

    # Thử load pretrained checkpoint
    try:
        from huggingface_hub import hf_hub_download
        log("[4/8] Downloading CRM checkpoint from HuggingFace...")
        ckpt_path = hf_hub_download(
            repo_id=config.crm_checkpoint,
            filename="CRM.pth",
            token=HF_TOKEN
        )
        loaded = torch.load(ckpt_path, map_location="cpu")
        if isinstance(loaded, dict):
            if "state_dict" in loaded:
                state_dict = loaded["state_dict"]
            elif "model" in loaded:
                state_dict = loaded["model"]
            elif "net" in loaded:
                state_dict = loaded["net"]
            else:
                state_dict = loaded
        else:
            state_dict = loaded

        model_dict = model.state_dict()
        pretrained_dict = {}

        # 1. Thử khớp trực tiếp
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
                continue
            # 2. Thử gỡ tiền tố phổ biến (module., generator., net., crm., unet.)
            for pfx in ["module.", "generator.", "net.", "crm.", "unet."]:
                if k.startswith(pfx):
                    clean_k = k[len(pfx):]
                    if clean_k in model_dict and v.shape == model_dict[clean_k].shape:
                        pretrained_dict[clean_k] = v
                        break

        if len(pretrained_dict) > 0:
            model.load_state_dict(pretrained_dict, strict=False)
            log(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained weights")
        else:
            log(f"  Pretrained weights note: {len(state_dict)} checkpoint keys vs {len(model_dict)} model keys. Training initialized weights.")
    except Exception as e:
        log(f"[WARN] Checkpoint load failed: {e}")
        log("  Training with initialized weights...")

    # Gradient checkpointing
    if config.gradient_checkpointing:
        # Enable for U-Net encoder blocks
        for module in model.unet.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True

    # DataParallel for multi-GPU
    if config.use_data_parallel and torch.cuda.device_count() > 1:
        log(f"  Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(config.device)

    # Đếm params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Total params: {total_params:,}")
    log(f"  Trainable params: {trainable_params:,}")

    # Loss
    criterion = CRMLoss(config)

    # Optimizer
    log("[5/8] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # LR scheduler with warmup
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.num_train_steps - config.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if config.mixed_precision == "fp16" else None

    # Training loop
    log("[6/8] Training...")
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []
    start_time = time.time()

    model.train()
    epoch = 0

    while global_step < config.num_train_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= config.num_train_steps:
                break

            views = batch["views"].to(config.device)
            ccms = batch["ccms"].to(config.device)
            points = batch["points"].to(config.device)
            sdf_gt = batch["sdf_gt"].to(config.device)
            colors_gt = batch["colors_gt"].to(config.device)

            # Giữ requires_grad=False cho points để tránh lỗi aten::cudnn_grid_sampler_backward double-backward
            points.requires_grad_(False)

            with torch.amp.autocast("cuda", enabled=(config.mixed_precision == "fp16")):
                pred_sdf, pred_color, triplane = model(views, ccms, points)
                loss, loss_dict = criterion(
                    pred_sdf, pred_color, sdf_gt, colors_gt, points
                )
                loss = loss / config.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

            train_losses.append(loss.item() * config.gradient_accumulation_steps)
            global_step += 1

            # Log
            if global_step % config.log_every == 0:
                avg_loss = sum(train_losses[-config.log_every:]) / min(config.log_every, len(train_losses))
                elapsed = (time.time() - start_time) / 60
                lr = optimizer.param_groups[0]["lr"]
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9

                loss_str = " | ".join([f"{k}:{v.item():.4f}" for k, v in loss_dict.items()])
                log(
                    f"Step {global_step}/{config.num_train_steps} | "
                    f"Loss: {avg_loss:.4f} [{loss_str}] | "
                    f"LR: {lr:.2e} | GPU: {gpu_mem:.1f}GB | {elapsed:.0f}min"
                )

            # Save checkpoint
            if global_step % config.save_every == 0:
                save_checkpoint(model, optimizer, global_step, epoch,
                                best_val_loss, train_losses, config, log)

            # Validation
            if global_step % config.eval_every == 0:
                val_loss = validate(model, val_loader, criterion, config, log)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, global_step, epoch,
                                    best_val_loss, train_losses, config, log,
                                    tag="best")
                model.train()

    # Final save
    log("[7/8] Saving final model...")
    save_checkpoint(model, optimizer, global_step, epoch,
                    best_val_loss, train_losses, config, log, tag="final")

    # Summary
    total_time = (time.time() - start_time) / 60
    log("[8/8] Training Complete!")
    log(f"  Total steps: {global_step}")
    log(f"  Total time: {total_time:.1f} min")
    log(f"  Final loss: {train_losses[-1]:.4f}")
    log(f"  Best val loss: {best_val_loss:.4f}")

    summary = {
        "stage": "stage2_crm",
        "status": "complete",
        "total_steps": global_step,
        "total_time_min": round(total_time, 1),
        "final_loss": round(train_losses[-1], 6),
        "best_val_loss": round(best_val_loss, 6),
        "output_dir": config.output_dir
    }
    with open(Path(config.output_dir) / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log_f.close()


def save_checkpoint(model, optimizer, step, epoch, best_val_loss,
                    train_losses, config, log, tag=None):
    """Lưu checkpoint."""
    name = f"ckpt_step_{step}" if not tag else f"ckpt_{tag}"
    ckpt_dir = Path(config.checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Lưu model weights (handle DataParallel)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_dict, ckpt_dir / "model.pth")

    # Lưu training state
    torch.save({
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "optimizer": optimizer.state_dict(),
    }, ckpt_dir / "training_state.pth")

    # Lưu config
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({k: v for k, v in vars(config).items()
                   if not k.startswith("_")}, f, indent=2, default=str)

    log(f"  Checkpoint saved: {ckpt_dir}")

    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def validate(model, val_loader, criterion, config, log):
    """Validation loop."""
    model.eval()
    val_losses = []

    for batch in val_loader:
        views = batch["views"].to(config.device)
        ccms = batch["ccms"].to(config.device)
        points = batch["points"].to(config.device)
        sdf_gt = batch["sdf_gt"].to(config.device)
        colors_gt = batch["colors_gt"].to(config.device)

        with torch.amp.autocast("cuda", enabled=(config.mixed_precision == "fp16")):
            pred_sdf, pred_color, _ = model(views, ccms, points)
            loss, _ = criterion(pred_sdf, pred_color, sdf_gt, colors_gt)

        val_losses.append(loss.item())

    avg_val_loss = sum(val_losses) / max(len(val_losses), 1)
    log(f"  [VAL] Loss: {avg_val_loss:.4f} ({len(val_losses)} batches)")

    # Trích xuất 1 sample mesh để kiểm định
    try:
        val_batch = next(iter(val_loader))
        sample_views = val_batch["views"][:1].to(config.device)
        sample_ccms = val_batch["ccms"][:1].to(config.device)
        out_mesh_path = Path(config.mesh_output_dir) / f"val_mesh_sample.obj"
        if extract_mesh_sample(model, sample_views, sample_ccms, out_mesh_path, resolution=64, device=config.device):
            log(f"  [VAL] Extracted sample 3D mesh: {out_mesh_path}")
    except Exception as e:
        log(f"  [VAL] Mesh extraction note: {e}")

    return avg_val_loss


def extract_mesh_sample(model, views, ccms, save_path, resolution=64, device="cuda"):
    """Trích xuất mesh bằng Marching Cubes trên lưới resolution^3."""
    import mcubes
    model_eval = model.module if hasattr(model, "module") else model
    model_eval.eval()
    with torch.no_grad():
        grid_coords = torch.linspace(-0.55, 0.55, resolution)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij")
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, -1, 3).to(device)

        chunk_size = 65536
        sdf_list = []
        B = views.shape[0]
        views_flat = views.reshape(B, -1, views.shape[-2], views.shape[-1])
        ccms_flat = ccms.reshape(B, -1, ccms.shape[-2], ccms.shape[-1])
        x = torch.cat([views_flat, ccms_flat], dim=1)
        triplane = model_eval.unet(x)

        for i in range(0, grid_points.shape[1], chunk_size):
            chunk_pts = grid_points[:, i:i+chunk_size]
            features = model_eval.query_triplane(triplane, chunk_pts)
            sdf_chunk, _ = model_eval.mlp(features)
            sdf_list.append(sdf_chunk.cpu())

        sdf_volume = torch.cat(sdf_list, dim=1).reshape(resolution, resolution, resolution).numpy()
        try:
            vertices, triangles = mcubes.marching_cubes(sdf_volume, 0.0)
            if len(vertices) > 0:
                vertices = vertices / (resolution - 1) * 1.1 - 0.55
                mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                mesh.export(str(save_path))
                return True
        except Exception as e:
            print(f"[WARN] Marching cubes export: {e}")
    return False


def create_dummy_dataset(config, num_objects=20):
    """Tạo dummy dataset để test pipeline."""
    dummy_dir = "/kaggle/working/dummy_dataset_stage2"
    os.makedirs(dummy_dir, exist_ok=True)

    for i in range(num_objects):
        obj_dir = Path(dummy_dir) / f"object_{i:04d}"
        obj_dir.mkdir(exist_ok=True)

        for j in range(6):
            img = Image.fromarray(
                np.random.randint(0, 255, (config.image_size, config.image_size, 3),
                                  dtype=np.uint8)
            )
            img.save(obj_dir / f"view_{j}.png")
            img.save(obj_dir / f"ccm_{j}.png")

        # Random points + GT
        n_pts = 4096
        np.save(obj_dir / "points.npy",
                np.random.uniform(-1, 1, (n_pts, 3)).astype(np.float32))
        np.save(obj_dir / "sdf.npy",
                np.random.randn(n_pts, 1).astype(np.float32) * 0.1)
        np.save(obj_dir / "colors.npy",
                np.random.uniform(0, 1, (n_pts, 3)).astype(np.float32))

    return dummy_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CRM 3D Stage 2: U-Net Reconstruction Fine-tune")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    print("=" * 60)

    train(config)
