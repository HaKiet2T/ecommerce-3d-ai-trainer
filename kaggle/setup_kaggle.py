"""
Kaggle Setup Script
===================
Cài đặt và xác thực Kaggle CLI trên máy local.
Chạy một lần trước khi push notebooks.
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def main():
    print("=" * 50)
    print("  Kaggle API Setup for CRM 3D Training")
    print("=" * 50)

    # 1. Check/Install kaggle
    print("\n[1/4] Checking Kaggle CLI...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "kaggle"],
            capture_output=True, text=True
        )
        print("  Kaggle CLI installed/updated.")
    except Exception as e:
        print(f"  Error: {e}")
        return

    # 2. Check credentials
    print("\n[2/4] Checking credentials...")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        with open(kaggle_json) as f:
            creds = json.load(f)
        print(f"  Found: {kaggle_json}")
        print(f"  Username: {creds.get('username', 'N/A')}")
    elif os.environ.get("KAGGLE_USERNAME"):
        print(f"  Using env: KAGGLE_USERNAME={os.environ['KAGGLE_USERNAME']}")
    else:
        print("  No credentials found!")
        print("\n  Cách lấy API token:")
        print("  1. Vào https://www.kaggle.com/settings")
        print("  2. Phần API -> Create New Token")
        print("  3. Download kaggle.json")

        # Hỏi user nhập trực tiếp
        print("\n  Nhập credentials (hoặc Enter để skip):")
        username = input("  Kaggle Username: ").strip()
        if username:
            api_key = input("  Kaggle API Key: ").strip()
            if api_key:
                kaggle_dir.mkdir(exist_ok=True)
                with open(kaggle_json, "w") as f:
                    json.dump({"username": username, "key": api_key}, f)
                # Set permissions (Unix)
                if os.name != "nt":
                    os.chmod(kaggle_json, 0o600)
                print(f"  Saved to {kaggle_json}")
            else:
                print("  Skipped.")
                return
        else:
            print("  Skipped.")
            return

    # 3. Verify
    print("\n[3/4] Verifying connection...")
    result = subprocess.run(
        ["kaggle", "kernels", "list", "--mine", "--page-size", "1"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  Connection OK!")
    else:
        print(f"  Error: {result.stderr}")
        print("  Kiểm tra lại username/key.")
        return

    # 4. Update metadata with real username
    print("\n[4/4] Updating kernel metadata files...")
    kaggle_dir_project = Path(__file__).resolve().parent
    notebooks_dir = kaggle_dir_project / "notebooks"

    # Read username
    if kaggle_json.exists():
        with open(kaggle_json) as f:
            username = json.load(f)["username"]
    else:
        username = os.environ["KAGGLE_USERNAME"]

    # Update all kernel-metadata.json
    for metadata_file in notebooks_dir.rglob("kernel-metadata.json"):
        with open(metadata_file) as f:
            metadata = json.load(f)

        old_id = metadata.get("id", "")
        if "KAGGLE_USERNAME" in old_id or old_id.startswith("/"):
            slug = old_id.split("/")[-1]
            metadata["id"] = f"{username}/{slug}"

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"  Updated: {metadata_file.parent.name}/{metadata_file.name}")
            print(f"    {old_id} -> {metadata['id']}")

    print("\n Setup complete! Bạn có thể push notebooks:")
    print("  python kaggle/push_training.py --stage data_prep")
    print("  python kaggle/push_training.py --stage stage1")
    print("  python kaggle/push_training.py --stage stage2")


if __name__ == "__main__":
    main()
