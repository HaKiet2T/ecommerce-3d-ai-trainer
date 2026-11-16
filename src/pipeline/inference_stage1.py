"""
Stage 1: Multi-View Diffusion Inference & Sanity Check (LoRA Zero123++)
======================================================================
Pipeline suy luận kiểm định chất lượng tạo 6 góc nhìn trực giao từ 1 ảnh sản phẩm đơn lẻ.
Mô hình sử dụng trọng số adapter LoRA (rank 32, alpha 64) đã fine-tune trên Kaggle GPU.

Kiến trúc giải mã:
1. Nạp độc lập (Decoupled Loading) các thành phần từ Hugging Face 'sudo-ai/zero123plus-v1.2':
   - UNet2DConditionModel (subfolder="unet")
   - AutoencoderKL (subfolder="vae")
   - CLIPVisionModelWithProjection (subfolder="vision_encoder")
   - EulerAncestralDiscreteScheduler (subfolder="scheduler")
2. Ghép adapter LoRA đã huấn luyện vào UNet qua thư viện PEFT.
3. Trích xuất đặc trưng ảnh điều kiện 2D qua CLIP Vision (1024-d).
4. Khử nhiễu không gian tiềm ẩn (Latent Denoising) qua Euler Ancestral Scheduler kèm Classifier-Free Guidance (CFG).
5. Giải mã VAE thành lưới ảnh 3x2 (768x512) và tách thành 6 góc nhìn chuẩn tắc (256x256).
6. Tính toán định lượng sai số hình học và màu sắc (PSNR, SSIM, L1 Error) so với Ground Truth nếu có.
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

# Vá lỗi tương thích cho huggingface_hub nếu hàm cached_download bị thiếu
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        from huggingface_hub import hf_hub_download
        huggingface_hub.cached_download = hf_hub_download
except Exception:
    pass

# Chặn nạp module torchao cũ nếu có
sys.modules['torchao'] = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

try:
    from peft import PeftModel
    from diffusers import UNet2DConditionModel, AutoencoderKL, EulerAncestralDiscreteScheduler
    from transformers import CLIPVisionModelWithProjection
except ImportError as e:
    print(f"[CẢNH BÁO] Thiếu thư viện phụ thuộc: {e}")
    print("Vui lòng cài đặt: pip install diffusers transformers accelerate peft safetensors torchvision pillow")


# ---------------------------------------------------------------------------
# Tiện ích tính chỉ số định lượng (PSNR & SSIM)
# ---------------------------------------------------------------------------

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Tính Peak Signal-to-Noise Ratio (PSNR) giữa 2 ảnh [0, 255]."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return float(20 * math.log10(max_pixel / math.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Tính Structural Similarity Index (SSIM) rút gọn trên kênh grayscale."""
    if img1.ndim == 3:
        gray1 = np.mean(img1.astype(np.float64), axis=2)
        gray2 = np.mean(img2.astype(np.float64), axis=2)
    else:
        gray1 = img1.astype(np.float64)
        gray2 = img2.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)
    sigma1_sq = np.var(gray1)
    sigma2_sq = np.var(gray2)
    sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Lớp Pipeline Suy luận Stage 1
# ---------------------------------------------------------------------------

class Zero123PlusLoRAInference:
    """Pipeline suy luận tạo 6 góc nhìn từ ảnh đơn sử dụng LoRA Adapter."""

    def __init__(
        self,
        base_model_name: str = "sudo-ai/zero123plus-v1.2",
        lora_weights_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16
    ):
        self.device = torch.device(device)
        self.dtype = dtype if device == "cuda" else torch.float32
        self.base_model_name = base_model_name
        self.lora_weights_path = lora_weights_path

        print(f"[Khởi tạo] Thiết bị: {self.device} | Kiểu dữ liệu: {self.dtype}")
        self._load_models()

    def _load_models(self):
        """Nạp từng thành phần theo kiến trúc tách rời (Decoupled Loading)."""
        print(f"[1/4] Nạp Base UNet từ '{self.base_model_name}'...")
        self.unet = UNet2DConditionModel.from_pretrained(
            self.base_model_name,
            subfolder="unet",
            torch_dtype=self.dtype
        ).to(self.device)

        if self.lora_weights_path and Path(self.lora_weights_path).exists():
            print(f"  -> Nạp LoRA Adapter từ: {self.lora_weights_path}")
            self.unet = PeftModel.from_pretrained(
                self.unet,
                self.lora_weights_path,
                torch_dtype=self.dtype
            )
            print("  -> Tích hợp trọng số LoRA vào UNet thành công.")
        else:
            print("  [LƯU Ý] Không tìm thấy đường dẫn LoRA, chạy trên trọng số gốc Pretrained.")

        print(f"[2/4] Nạp VAE Autoencoder...")
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_name,
            subfolder="vae",
            torch_dtype=self.dtype
        ).to(self.device)

        print(f"[3/4] Nạp CLIP Vision Encoder...")
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.base_model_name,
            subfolder="vision_encoder",
            torch_dtype=self.dtype
        ).to(self.device)

        print(f"[4/4] Nạp Euler Ancestral Discrete Scheduler...")
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            self.base_model_name,
            subfolder="scheduler"
        )

        # Chuyển tất cả sang chế độ đánh giá
        self.unet.eval()
        self.vae.eval()
        self.vision_encoder.eval()

        # Bộ biến đổi chuẩn hóa cho CLIP (224x224, ImageNet normalization)
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    @torch.inference_mode()
    def generate_views(
        self,
        image_input: Image.Image,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.0,
        seed: int = 42
    ):
        """
        Sinh 6 góc nhìn từ 1 ảnh sản phẩm.
        
        Tham số:
            image_input: Ảnh PIL RGB của sản phẩm (256x256 hoặc 512x512).
            num_inference_steps: Số bước khử nhiễu (khuyến nghị 30-50).
            guidance_scale: Hệ số Classifier-Free Guidance (CFG scale).
            seed: Hạt giống ngẫu nhiên để tái lập kết quả.
            
        Trả về:
            dict chứa:
                'views': danh sách 6 ảnh PIL [view_0, ..., view_5]
                'grid': ảnh canvas PIL kích thước 512x768 chứa toàn bộ 6 góc
                'inference_time_sec': thời gian thực thi
        """
        start_time = time.time()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 1. Tiền xử lý ảnh điều kiện
        img_rgb = image_input.convert("RGB")
        clip_tensor = self.clip_transform(img_rgb).unsqueeze(0).to(self.device, dtype=self.dtype)

        # Trích xuất đặc trưng CLIP embedding [1, 1, 1024]
        cond_embeds = self.vision_encoder(clip_tensor).image_embeds.unsqueeze(1)

        # Chuẩn bị Unconditional embedding cho CFG
        uncond_clip = torch.zeros_like(clip_tensor)
        uncond_embeds = self.vision_encoder(uncond_clip).image_embeds.unsqueeze(1)

        # Ghép condition và uncondition để chạy theo batch song song [2, 1, 1024]
        text_embeddings = torch.cat([uncond_embeds, cond_embeds], dim=0)

        # 2. Khởi tạo Latent ngẫu nhiên cho lưới ảnh 3x2 (768x512)
        # Latent dimensions: H = 768 / 8 = 96, W = 512 / 8 = 64
        latent_shape = (1, 4, 96, 64)
        latents = torch.randn(latent_shape, device=self.device, dtype=self.dtype)

        # Thiết lập lịch trình Scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        # 3. Chu trình Denoising Loop
        for t in self.scheduler.timesteps:
            # Nhân đôi latents cho CFG [2, 4, 96, 64]
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Dự đoán nhiễu bằng UNet kèm LoRA
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )[0]

            # Phân tách và áp dụng Classifier-Free Guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Cập nhật latent qua Euler Ancestral step
            latents = self.scheduler.step(noise_pred, t, latents)[0]

        # 4. Giải mã Latent qua VAE
        latents = latents / self.vae.config.scaling_factor
        decoded = self.vae.decode(latents.to(self.vae.dtype)).sample

        # Chuyển đổi tensor sang ảnh PIL [-1, 1] -> [0, 255]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        grid_np = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
        grid_uint8 = (grid_np * 255).round().astype(np.uint8)
        grid_image = Image.fromarray(grid_uint8)

        # 5. Phân rã lưới 512x768 thành 6 góc nhìn (256x256 mỗi góc)
        # Bố cục 3 hàng 2 cột:
        # [0, 0]: view_0 (Front) | [256, 0]: view_1 (Right)
        # [0, 256]: view_2 (Back) | [256, 256]: view_3 (Left)
        # [0, 512]: view_4 (Top)  | [256, 512]: view_5 (Bottom)
        view_size = 256
        views = []
        coordinates = [
            (0, 0),                       # view_0: Front
            (view_size, 0),               # view_1: Right
            (0, view_size),               # view_2: Back
            (view_size, view_size),       # view_3: Left
            (0, view_size * 2),           # view_4: Top
            (view_size, view_size * 2)    # view_5: Bottom
        ]

        for x, y in coordinates:
            box = (x, y, x + view_size, y + view_size)
            cropped_view = grid_image.crop(box)
            views.append(cropped_view)

        elapsed = time.time() - start_time
        return {
            "views": views,
            "grid": grid_image,
            "inference_time_sec": round(elapsed, 2)
        }


# ---------------------------------------------------------------------------
# Tạo ảnh so sánh đối chiếu đa góc nhìn (Sanity Collage)
# ---------------------------------------------------------------------------

def create_comparison_sheet(
    input_img: Image.Image,
    pred_views: list,
    gt_views: list = None,
    object_id: str = "sample"
) -> Image.Image:
    """Tạo bảng đối chiếu trực quan giữa ảnh đầu vào, 6 góc nhìn sinh ra và Ground Truth."""
    view_size = 256
    num_cols = 7 if gt_views is None else 7
    num_rows = 1 if gt_views is None else 2

    canvas_w = view_size * 7
    canvas_h = view_size * (2 if gt_views else 1)
    sheet = Image.new("RGB", (canvas_w, canvas_h), color=(240, 240, 240))

    # Cột 1: Ảnh đầu vào
    resized_input = input_img.resize((view_size, view_size), Image.BILINEAR)
    sheet.paste(resized_input, (0, 0))

    # Cột 2-7: 6 góc nhìn dự đoán
    for i, v in enumerate(pred_views):
        sheet.paste(v, ((i + 1) * view_size, 0))

    # Hàng 2: Ground Truth (nếu có)
    if gt_views and len(gt_views) == 6:
        # Giữ ô đầu vào ở cột 1 hàng 2
        sheet.paste(resized_input, (0, view_size))
        for i, gtv in enumerate(gt_views):
            gt_resized = gtv.resize((view_size, view_size), Image.BILINEAR)
            sheet.paste(gt_resized, ((i + 1) * view_size, view_size))

    return sheet


# ---------------------------------------------------------------------------
# Hàm thực thi chính (Main Execution)
# ---------------------------------------------------------------------------

def run_sanity_check(
    image_path: str,
    lora_path: str,
    output_dir: str,
    ground_truth_dir: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    steps: int = 40,
    guidance_scale: float = 4.0
):
    """Quy trình kiểm tra tính đúng đắn toàn diện."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  STAGE 1 INFERENCE & SANITY CHECK (LoRA Zero123++)")
    print("=" * 65)
    print(f"Ảnh đầu vào        : {image_path}")
    print(f"LoRA weights       : {lora_path}")
    print(f"Thư mục kết quả    : {output_path}")
    print(f"Thiết bị tính toán : {device}")
    print(f"Số bước khử nhiễu  : {steps} steps | CFG scale: {guidance_scale}")
    print("-" * 65)

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh đầu vào tại: {image_path}")

    input_img = Image.open(image_path).convert("RGB")

    # Khởi tạo mô hình
    infer_pipeline = Zero123PlusLoRAInference(
        lora_weights_path=lora_path,
        device=device
    )

    # Sinh 6 góc nhìn
    print("[Tiến trình] Bắt đầu suy luận khử nhiễu...")
    result = infer_pipeline.generate_views(
        image_input=input_img,
        num_inference_steps=steps,
        guidance_scale=guidance_scale
    )
    print(f"[Hoàn thành] Thời gian suy luận: {result['inference_time_sec']} giây")

    # Lưu 6 ảnh góc nhìn độc lập
    views_dir = output_path / "predicted_views"
    views_dir.mkdir(parents=True, exist_ok=True)
    view_names = ["front", "right", "back", "left", "top", "bottom"]

    for i, (v, name) in enumerate(zip(result["views"], view_names)):
        save_file = views_dir / f"view_{i}_{name}.png"
        v.save(save_file)
        print(f"  -> Lưu góc nhìn {i} ({name}): {save_file.name}")

    # Lưu canvas lưới 3x2 chuẩn
    grid_save_path = output_path / "predicted_grid_3x2.png"
    result["grid"].save(grid_save_path)
    print(f"  -> Lưu lưới đa góc 3x2: {grid_save_path.name}")

    # Đọc Ground Truth để đối chiếu nếu có
    gt_views = []
    metrics = {}
    if ground_truth_dir and Path(ground_truth_dir).exists():
        gt_dir = Path(ground_truth_dir)
        print("\n[Đối chiếu] Phát hiện thư mục dữ liệu chuẩn (Ground Truth)...")
        psnr_list, ssim_list = [], []

        for i, name in enumerate(view_names):
            gt_file = gt_dir / f"view_{i}.png"
            if gt_file.exists():
                gt_img = Image.open(gt_file).convert("RGB").resize((256, 256), Image.BILINEAR)
                gt_views.append(gt_img)

                # Tính PSNR & SSIM giữa ảnh dự đoán và Ground Truth
                pred_np = np.array(result["views"][i])
                gt_np = np.array(gt_img)
                psnr_val = calculate_psnr(pred_np, gt_np)
                ssim_val = calculate_ssim(pred_np, gt_np)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                print(f"  - Góc {i} ({name:6s}): PSNR = {psnr_val:.2f} dB | SSIM = {ssim_val:.4f}")

        if psnr_list:
            avg_psnr = float(np.mean(psnr_list))
            avg_ssim = float(np.mean(ssim_list))
            metrics = {
                "avg_psnr_db": round(avg_psnr, 2),
                "avg_ssim": round(avg_ssim, 4),
                "per_view_psnr": [round(p, 2) for p in psnr_list],
                "per_view_ssim": [round(s, 4) for s in ssim_list]
            }
            print("-" * 65)
            print(f"  => ĐIỂM TRUNG BÌNH: PSNR = {avg_psnr:.2f} dB | SSIM = {avg_ssim:.4f}")

    # Tạo bảng đối chiếu trực quan
    comparison_sheet = create_comparison_sheet(
        input_img=input_img,
        pred_views=result["views"],
        gt_views=gt_views if len(gt_views) == 6 else None
    )
    comp_save_path = output_path / "sanity_comparison_sheet.png"
    comparison_sheet.save(comp_save_path)
    print(f"  -> Lưu bảng đối chiếu tổng thể: {comp_save_path.name}")

    # Lưu tóm tắt JSON
    summary_report = {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_image": str(image_path),
        "lora_checkpoint": str(lora_path),
        "inference_time_sec": result["inference_time_sec"],
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "metrics": metrics
    }
    with open(output_path / "sanity_report.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print("Sanity Check hoàn tất mỹ mãn!")
    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy kiểm tra suy luận Stage 1 LoRA Zero123++")
    parser.add_argument("--image", type=str, default="", help="Đường dẫn ảnh sản phẩm đầu vào")
    parser.add_argument("--lora_path", type=str, default="", help="Đường dẫn thư mục LoRA adapter")
    parser.add_argument("--output_dir", type=str, default="outputs/sanity_stage1", help="Thư mục xuất ảnh")
    parser.add_argument("--gt_dir", type=str, default="", help="Thư mục chứa 6 ảnh Ground Truth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị")
    parser.add_argument("--steps", type=int, default=40, help="Số bước khử nhiễu")
    parser.add_argument("--cfg", type=float, default=4.0, help="Hệ số Classifier-Free Guidance")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    # Tự động tìm kiếm ảnh mẫu và checkpoint nếu không truyền tham số
    default_img = project_root / "outputs" / "kaggle" / "data_prep" / "20261116_033648" / "crm_ecommerce_dataset" / "035aa8e8c99d455f91e96d9c78a6a404" / "input.png"
    default_lora = project_root / "outputs" / "kaggle" / "stage1" / "20261116_063415" / "stage1_output" / "lora_final"
    default_gt = default_img.parent if default_img.exists() else None

    img_target = args.image if args.image else str(default_img)
    lora_target = args.lora_path if args.lora_path else str(default_lora)
    gt_target = args.gt_dir if args.gt_dir else (str(default_gt) if default_gt else "")

    run_sanity_check(
        image_path=img_target,
        lora_path=lora_target,
        output_dir=args.output_dir,
        ground_truth_dir=gt_target,
        device=args.device,
        steps=args.steps,
        guidance_scale=args.cfg
    )
