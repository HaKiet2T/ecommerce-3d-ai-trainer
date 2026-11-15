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
    access_token = kaggle_dir / "access_token"
    username_file = kaggle_dir / "username"

    username = None
    if access_token.exists():
        print(f"  Found API Token: {access_token}")
        if username_file.exists():
            with open(username_file) as f:
                username = f.read().strip()
                print(f"  Username: {username}")
        elif os.environ.get("KAGGLE_USERNAME"):
            username = os.environ["KAGGLE_USERNAME"]
            print(f"  Using env: KAGGLE_USERNAME={username}")
        else:
            print("  Vui long nhap Kaggle Username de lien ket voi API Token:")
            username = input("  Kaggle Username: ").strip()
            if username:
                with open(username_file, "w") as f:
                    f.write(username)
                print(f"  Saved username to {username_file}")
    elif kaggle_json.exists():
        with open(kaggle_json) as f:
            creds = json.load(f)
        print(f"  Found: {kaggle_json}")
        username = creds.get("username", "N/A")
        print(f"  Username: {username}")
    elif os.environ.get("KAGGLE_USERNAME"):
        username = os.environ["KAGGLE_USERNAME"]
        print(f"  Using env: KAGGLE_USERNAME={username}")
    else:
        print("  No credentials found!")
        print("\n  Cách lấy API token:")
        print("  1. Vào https://www.kaggle.com/settings")
        print("  2. Phần API -> Create New Token")
        print("  3. Download kaggle.json hoặc copy API Token")

        # Hỏi user nhập trực tiếp
        print("\n  Nhập credentials (hoặc Enter để skip):")
        username = input("  Kaggle Username: ").strip()
        if username:
            token_or_key = input("  Kaggle API Key/Token: ").strip()
            if token_or_key:
                kaggle_dir.mkdir(exist_ok=True)
                if token_or_key.startswith("KGAT_"):
                    with open(access_token, "w") as f:
                        f.write(token_or_key)
                    with open(username_file, "w") as f:
                        f.write(username)
                    print(f"  Saved access token to {access_token}")
                else:
                    with open(kaggle_json, "w") as f:
                        json.dump({"username": username, "key": token_or_key}, f)
                    print(f"  Saved to {kaggle_json}")
                if os.name != "nt":
                    if access_token.exists():
                        os.chmod(access_token, 0o600)
                    if kaggle_json.exists():
                        os.chmod(kaggle_json, 0o600)
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
    if username:
        pass
    elif username_file.exists():
        with open(username_file) as f:
            username = f.read().strip()
    elif kaggle_json.exists():
        with open(kaggle_json) as f:
            username = json.load(f).get("username")
    elif os.environ.get("KAGGLE_USERNAME"):
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
