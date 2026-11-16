"""
Stage 1 Sanity Check & Visual Evaluation on Kaggle GPU
======================================================
Kaggle Notebook Script: Đánh giá định lượng & trực quan LoRA Zero123++
Nguồn dữ liệu đầu vào:
1. crm-3d-data-preparation-objaverse (Tập dữ liệu 80 vật thể E-commerce)
2. crm-3d-stage1 (Checkpoint LoRA rank 32 đã huấn luyện 8.000 steps)
"""

import os
import sys
import json
import time
import math
import shutil
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# 1. Cài đặt thư viện
# ---------------------------------------------------------------------------
os.system("pip uninstall -y torchao")
os.system(
    "pip install -q "
    "'huggingface_hub<0.26.0' "
    "'diffusers>=0.30.0' "
    "'transformers>=4.45.0' "
    "'accelerate>=0.34.0' "
    "'peft>=0.13.0' "
    "safetensors torchvision"
)

try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        from huggingface_hub import hf_hub_download
        huggingface_hub.cached_download = hf_hub_download
except Exception:
    pass

sys.modules['torchao'] = None

import torch
import torch.nn.functional as F
from torchvision import transforms
from peft import PeftModel
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerAncestralDiscreteScheduler
from transformers import CLIPVisionModelWithProjection

# ---------------------------------------------------------------------------
# 2. Tiện ích tính toán chỉ số
# ---------------------------------------------------------------------------
def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * math.log10(255.0 / math.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    gray1 = np.mean(img1.astype(np.float64), axis=2) if img1.ndim == 3 else img1.astype(np.float64)
    gray2 = np.mean(img2.astype(np.float64), axis=2) if img2.ndim == 3 else img2.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = np.mean(gray1), np.mean(gray2)
    sigma1_sq, sigma2_sq = np.var(gray1), np.var(gray2)
    sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
    return float(((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)))


# ---------------------------------------------------------------------------
# 3. Quét Dataset & Checkpoint
# ---------------------------------------------------------------------------
def locate_inputs():
    input_root = Path("/kaggle/input")
    print(f"[Scan] Quét tài nguyên trong {input_root}...")

    # 1. Tìm dataset zip
    dataset_zip = None
    for p in input_root.rglob("*.zip"):
        if "crm_ecommerce_dataset" in p.name:
            dataset_zip = p
            break

    data_dir = Path("/tmp/extracted_dataset")
    if dataset_zip and not data_dir.exists():
        print(f"  -> Giải nén dataset từ: {dataset_zip}")
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        # Nếu có thư mục lồng
        nested = list(data_dir.glob("**/input.png"))
        if nested:
            data_dir = nested[0].parent.parent

    # 2. Tìm LoRA weights
    lora_path = None
    for p in input_root.rglob("adapter_model.safetensors"):
        if "lora_final" in str(p) or "step_8000" in str(p) or "stage1" in str(p):
            lora_path = p.parent
            break

    print(f"  -> Dữ liệu: {data_dir} (tồn tại: {data_dir.exists()})")
    print(f"  -> Trọng số LoRA: {lora_path}")
    return data_dir, lora_path


# ---------------------------------------------------------------------------
# 4. Suy luận đa góc nhìn
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    output_dir = Path("/kaggle/working/sanity_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir, lora_path = locate_inputs()
    if not lora_path:
        raise FileNotFoundError("Không tìm thấy checkpoint LoRA từ kernel source Stage 1!")

    base_model = "sudo-ai/zero123plus-v1.2"
    print(f"[Model] Nạp {base_model} với LoRA từ {lora_path}...")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=dtype).to(device)
    unet = PeftModel.from_pretrained(unet, str(lora_path), torch_dtype=dtype)
    unet.eval()

    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype).to(device).eval()
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained(base_model, subfolder="vision_encoder", torch_dtype=dtype).to(device).eval()
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(base_model, subfolder="scheduler")

    clip_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # Chọn 4 vật thể ngẫu nhiên để đánh giá
    all_objects = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "input.png").exists()])
    test_objects = all_objects[-4:] if len(all_objects) >= 4 else all_objects
    print(f"[Eval] Đánh giá trên {len(test_objects)} vật thể validation...")

    results_summary = []

    for obj_dir in test_objects:
        obj_id = obj_dir.name
        print(f"\n--- Đang xử lý vật thể: {obj_id} ---")
        input_img = Image.open(obj_dir / "input.png").convert("RGB")
        clip_tensor = clip_transform(input_img).unsqueeze(0).to(device, dtype=dtype)

        cond_embeds = vision_encoder(clip_tensor).image_embeds.unsqueeze(1)
        uncond_embeds = vision_encoder(torch.zeros_like(clip_tensor)).image_embeds.unsqueeze(1)
        text_embeddings = torch.cat([uncond_embeds, cond_embeds], dim=0)

        # Khởi tạo latent ngẫu nhiên
        torch.manual_seed(42)
        latents = torch.randn((1, 4, 96, 64), device=device, dtype=dtype)
        scheduler.set_timesteps(40)
        latents = latents * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            noise_pred = unet(sample=latent_model_input, timestep=t, encoder_hidden_states=text_embeddings, return_dict=False)[0]
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + 4.0 * (cond - uncond)
            latents = scheduler.step(noise_pred, t, latents)[0]

        latents = latents / vae.config.scaling_factor
        decoded = vae.decode(latents.to(vae.dtype)).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        grid_np = (decoded.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        grid_img = Image.fromarray(grid_np)

        # Cắt 6 góc nhìn
        coords = [(0, 0), (256, 0), (0, 256), (256, 256), (0, 512), (256, 512)]
        pred_views = [grid_img.crop((x, y, x + 256, y + 256)) for x, y in coords]

        # Đọc Ground Truth
        gt_views = []
        psnr_vals, ssim_vals = [], []
        for i in range(6):
            gt_path = obj_dir / f"view_{i}.png"
            if gt_path.exists():
                gt_im = Image.open(gt_path).convert("RGB").resize((256, 256), Image.BILINEAR)
                gt_views.append(gt_im)
                psnr_vals.append(calculate_psnr(np.array(pred_views[i]), np.array(gt_im)))
                ssim_vals.append(calculate_ssim(np.array(pred_views[i]), np.array(gt_im)))

        # Tạo collage đối chiếu
        sheet = Image.new("RGB", (256 * 7, 256 * 2), color=(240, 240, 240))
        sheet.paste(input_img.resize((256, 256)), (0, 0))
        sheet.paste(input_img.resize((256, 256)), (0, 256))
        for i in range(6):
            sheet.paste(pred_views[i], ((i + 1) * 256, 0))
            if i < len(gt_views):
                sheet.paste(gt_views[i], ((i + 1) * 256, 256))

        sheet_path = output_dir / f"comparison_{obj_id}.png"
        sheet.save(sheet_path)

        obj_summary = {
            "object_id": obj_id,
            "avg_psnr_db": round(float(np.mean(psnr_vals)), 2) if psnr_vals else None,
            "avg_ssim": round(float(np.mean(ssim_vals)), 4) if ssim_vals else None,
            "comparison_image": sheet_path.name
        }
        results_summary.append(obj_summary)
        print(f"  -> Hoàn thành {obj_id}: PSNR = {obj_summary['avg_psnr_db']} dB | SSIM = {obj_summary['avg_ssim']}")

    # Lưu báo cáo tổng
    overall_report = {
        "status": "success",
        "stage": "stage1_sanity_check",
        "evaluated_objects": len(results_summary),
        "mean_dataset_psnr": round(float(np.mean([r["avg_psnr_db"] for r in results_summary if r["avg_psnr_db"]])), 2),
        "mean_dataset_ssim": round(float(np.mean([r["avg_ssim"] for r in results_summary if r["avg_ssim"]])), 4),
        "details": results_summary
    }
    with open(output_dir / "sanity_summary.json", "w") as f:
        json.dump(overall_report, f, indent=2)

    # Đóng gói zip output
    shutil.make_archive("/kaggle/working/sanity_check_outputs", "zip", output_dir)
    print("\n" + "=" * 60)
    print("Sanity Check hoàn tất! Đã nén kết quả thành /kaggle/working/sanity_check_outputs.zip")
    print(f"Mean PSNR: {overall_report['mean_dataset_psnr']} dB | Mean SSIM: {overall_report['mean_dataset_ssim']}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
