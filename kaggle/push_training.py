"""
Kaggle Training Automation - Push & Monitor CRM Training Notebooks
==================================================================
Script tự động:
1. Tạo kernel-metadata.json phù hợp GPU T4x2
2. Đẩy notebook/script lên Kaggle
3. Theo dõi trạng thái kernel (running/complete/error)
4. Tải output (checkpoints, logs) về local

Yêu cầu:
- pip install kaggle pyyaml
- File ~/.kaggle/kaggle.json hoặc env KAGGLE_USERNAME + KAGGLE_KEY
"""

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = KAGGLE_DIR / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "kaggle"

STAGES = {
    "data_prep": {
        "dir": "data_preparation",
        "title": "CRM 3D - Data Preparation (Objaverse)",
        "code_file": "prepare_data.py",
        "enable_gpu": False,
        "enable_internet": True,
        "max_runtime_hours": 6,
        "dataset_sources": [],
        "description": "Download + filter Objaverse, render 6 views via trimesh"
    },
    "stage1": {
        "dir": "stage1_lora_zero123",
        "title": "CRM 3D - Stage 1 LoRA Zero123++ Fine-tune",
        "code_file": "train_stage1.py",
        "enable_gpu": True,
        "enable_internet": True,
        "max_runtime_hours": 12,
        "dataset_sources": [],
        "description": "LoRA fine-tune Zero123++ multi-view diffusion"
    },
    "stage2": {
        "dir": "stage2_crm_finetune",
        "title": "CRM 3D - Stage 2 CRM Reconstruction Fine-tune",
        "code_file": "train_stage2.py",
        "enable_gpu": True,
        "enable_internet": True,
        "max_runtime_hours": 12,
        "dataset_sources": [],
        "description": "Fine-tune CRM U-Net triplane decoder + FlexiCubes"
    }
}


def check_kaggle_cli():
    """Kiểm tra kaggle CLI đã cài và xác thực chưa."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print("[ERROR] Kaggle CLI chưa cài. Chạy: pip install kaggle")
            return False
        print(f"[OK] Kaggle CLI: {result.stdout.strip()}")
    except FileNotFoundError:
        print("[ERROR] Kaggle CLI chưa cài. Chạy: pip install kaggle")
        return False

    # Kiểm tra credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    access_token = Path.home() / ".kaggle" / "access_token"
    if access_token.exists():
        print(f"[OK] Credentials: {access_token} (Kaggle API Token)")
        return True
    elif kaggle_json.exists():
        print(f"[OK] Credentials: {kaggle_json}")
        return True
    elif os.environ.get("KAGGLE_API_TOKEN"):
        print("[OK] Credentials: env KAGGLE_API_TOKEN")
        return True
    elif KAGGLE_USERNAME:
        print(f"[OK] Credentials: env KAGGLE_USERNAME={KAGGLE_USERNAME}")
        return True
    else:
        print("[ERROR] Chưa có credentials.")
        print("  Option 1: Đặt file ~/.kaggle/access_token hoặc ~/.kaggle/kaggle.json")
        print("  Option 2: Set env KAGGLE_API_TOKEN hoặc KAGGLE_USERNAME + KAGGLE_KEY")
        print("  Lấy API token tại: https://www.kaggle.com/settings")
        return False


def get_username():
    """Lấy Kaggle username từ credentials."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    username_file = Path.home() / ".kaggle" / "username"
    if username_file.exists():
        with open(username_file) as f:
            u = f.read().strip()
            if u:
                return u
    if kaggle_json.exists():
        with open(kaggle_json) as f:
            data = json.load(f)
            return data.get("username", KAGGLE_USERNAME)
    return KAGGLE_USERNAME


def generate_metadata(stage_key: str, username: str = None):
    """Tạo kernel-metadata.json cho một stage."""
    if stage_key not in STAGES:
        print(f"[ERROR] Stage '{stage_key}' không hợp lệ. Chọn: {list(STAGES.keys())}")
        return None

    stage = STAGES[stage_key]
    if not username:
        username = get_username()

    if not username:
        print("[ERROR] Không xác định được Kaggle username")
        return None

    slug = f"crm-3d-{stage_key.replace('_', '-')}"

    metadata = {
        "id": f"{username}/{slug}",
        "title": stage["title"],
        "code_file": stage["code_file"],
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": str(stage["enable_gpu"]).lower(),
        "enable_internet": str(stage["enable_internet"]).lower(),
        "dataset_sources": stage["dataset_sources"],
        "competition_sources": [],
        "kernel_sources": []
    }

    # Thêm machine_shape nếu dùng GPU
    if stage["enable_gpu"]:
        metadata["gpu_type"] = "NvidiaTeslaT4"

    # Ghi ra file
    stage_dir = NOTEBOOKS_DIR / stage["dir"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = stage_dir / "kernel-metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] Metadata: {metadata_path}")
    print(f"     Kernel ID: {metadata['id']}")
    print(f"     GPU: {stage['enable_gpu']} | Internet: {stage['enable_internet']}")

    return metadata


def push_kernel(stage_key: str):
    """Đẩy kernel lên Kaggle."""
    stage = STAGES[stage_key]
    stage_dir = NOTEBOOKS_DIR / stage["dir"]

    # Kiểm tra file code tồn tại
    code_path = stage_dir / stage["code_file"]
    if not code_path.exists():
        print(f"[ERROR] Chưa có code file: {code_path}")
        print("  Cần tạo training script trước khi push.")
        return False

    metadata_path = stage_dir / "kernel-metadata.json"
    if not metadata_path.exists():
        print("[INFO] Tạo metadata...")
        generate_metadata(stage_key)

    print(f"\n{'='*60}")
    print(f"  PUSHING: {stage['title']}")
    print(f"  Dir: {stage_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(stage_dir)],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        print(f"[ERROR] Push thất bại:\n{result.stderr}")
        return False

    print(f"[OK] Push thành công!\n{result.stdout}")
    return True


def check_status(stage_key: str):
    """Kiểm tra trạng thái kernel đang chạy."""
    username = get_username()
    slug = f"crm-3d-{stage_key.replace('_', '-')}"
    kernel_id = f"{username}/{slug}"

    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        return {"status": "error", "message": result.stderr.strip()}

    output = result.stdout.strip()
    # Parse status
    status = "unknown"
    for s in ["complete", "running", "queued", "error", "cancelAcknowledged"]:
        if s in output.lower():
            status = s
            break

    return {"status": status, "raw": output}


def monitor_kernel(stage_key: str, poll_interval: int = 60, timeout_hours: float = 12):
    """Theo dõi kernel cho đến khi hoàn thành."""
    print(f"\n[MONITOR] Bắt đầu theo dõi kernel {stage_key}")
    print(f"  Poll interval: {poll_interval}s | Timeout: {timeout_hours}h\n")

    start_time = time.time()
    max_seconds = timeout_hours * 3600

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_seconds:
            print(f"[TIMEOUT] Đã vượt {timeout_hours}h. Dừng monitor.")
            return "timeout"

        status_info = check_status(stage_key)
        status = status_info["status"]
        elapsed_min = elapsed / 60

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"  [{timestamp}] Status: {status} | Elapsed: {elapsed_min:.1f}min")

        if status == "complete":
            print(f"\n[DONE] Kernel hoàn thành sau {elapsed_min:.1f} phút!")
            return "complete"
        elif status == "error":
            print(f"\n[ERROR] Kernel gặp lỗi: {status_info.get('raw', '')}")
            return "error"
        elif status == "cancelAcknowledged":
            print(f"\n[CANCELLED] Kernel đã bị huỷ.")
            return "cancelled"

        time.sleep(poll_interval)


def download_output(stage_key: str):
    """Tải output (checkpoints, logs) từ kernel đã chạy xong."""
    username = get_username()
    slug = f"crm-3d-{stage_key.replace('_', '-')}"
    kernel_id = f"{username}/{slug}"

    output_path = OUTPUT_DIR / stage_key / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] Tải output về: {output_path}")

    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(output_path)],
        capture_output=True, text=True, timeout=300
    )

    if result.returncode != 0:
        print(f"[ERROR] Download thất bại:\n{result.stderr}")
        return None

    print(f"[OK] Download thành công!")
    # Liệt kê files
    files = list(output_path.rglob("*"))
    for f in files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")

    return output_path


def upload_dataset(dataset_dir: str, dataset_slug: str, title: str = None):
    """Upload một thư mục làm Kaggle dataset (để dùng làm input cho notebook)."""
    username = get_username()
    if not title:
        title = dataset_slug.replace("-", " ").title()

    # Tạo dataset-metadata.json
    metadata = {
        "title": title,
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}]
    }

    metadata_path = Path(dataset_dir) / "dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[UPLOAD] Dataset: {username}/{dataset_slug}")
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", dataset_dir],
        capture_output=True, text=True, timeout=600
    )

    if result.returncode != 0:
        # Thử update thay vì create
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", dataset_dir, "-m", "update"],
            capture_output=True, text=True, timeout=600
        )

    if result.returncode != 0:
        print(f"[ERROR] Upload dataset thất bại:\n{result.stderr}")
        return False

    print(f"[OK] Dataset uploaded: {username}/{dataset_slug}")
    return True


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Kaggle CRM Training Automation"
    )
    parser.add_argument(
        "--stage",
        choices=["data_prep", "stage1", "stage2"],
        required=True,
        help="Stage cần chạy"
    )
    parser.add_argument(
        "--action",
        choices=["push", "status", "monitor", "download", "metadata"],
        default="push",
        help="Hành động (mặc định: push)"
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Push xong tự monitor luôn"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="Khoảng cách poll (giây)"
    )
    parser.add_argument(
        "--auto-download", action="store_true",
        help="Tự tải output khi kernel xong"
    )

    args = parser.parse_args()

    # Kiểm tra kaggle CLI
    if not check_kaggle_cli():
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  CRM 3D Kaggle Trainer")
    print(f"  Stage: {args.stage} | Action: {args.action}")
    print(f"{'='*60}\n")

    if args.action == "metadata":
        generate_metadata(args.stage)

    elif args.action == "push":
        success = push_kernel(args.stage)
        if success and args.monitor:
            result = monitor_kernel(args.stage, poll_interval=args.poll_interval)
            if result == "complete" and args.auto_download:
                download_output(args.stage)

    elif args.action == "status":
        info = check_status(args.stage)
        print(f"Status: {info['status']}")
        print(f"Raw: {info.get('raw', '')}")

    elif args.action == "monitor":
        result = monitor_kernel(args.stage, poll_interval=args.poll_interval)
        if result == "complete" and args.auto_download:
            download_output(args.stage)

    elif args.action == "download":
        download_output(args.stage)


if __name__ == "__main__":
    main()
