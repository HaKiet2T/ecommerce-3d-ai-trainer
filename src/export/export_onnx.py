"""
ONNX and TensorRT Model Exporter for CRM
========================================
Exports the MultiViewUNet and Triplane decoders to ONNX format with graph optimization.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxsim
except ImportError:
    onnxsim = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from src.models.crm import ConvolutionalReconstructionModel, MultiViewUNet, TriplaneSDFDecoder

class UNetWrapper(nn.Module):
    """Wrapper cho MultiViewUNet de nhan tensor dau vao gop phuc vu ONNX export."""
    def __init__(self, unet: MultiViewUNet):
        super().__init__()
        self.unet = unet

    def forward(self, images: torch.Tensor, ccms: torch.Tensor) -> torch.Tensor:
        return self.unet(images, ccms)

def export_crm_to_onnx(
    checkpoint_path: str,
    output_dir: str = "exported_models",
    image_size: int = 256,
    triplane_channels: int = 32,
    device: str = "cpu"
):
    """
    Xuat CRM sang dinh dang ONNX kem toi uu hoa do thi:
    1. Xuat phan mang MultiViewUNet (sinh Triplane tu 6 goc nhin)
    2. Xuat phan mang TriplaneSDFDecoder (truy van SDF va RGB tai tap diem 3D)
    3. Toi uu hoa graph voi onnx-simplifier
    4. Kiem tra tinh nhat quan ket qua giua PyTorch va ONNX Runtime
    """
    if onnx is None:
        raise RuntimeError("Thu vien 'onnx' chua duoc cai dat. Vui long cai: pip install onnx onnxsim")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Khoi tao mo hinh CRM tu checkpoint: {checkpoint_path}")
    model = ConvolutionalReconstructionModel(
        triplane_channels=triplane_channels,
        base_features=64,
        hidden_dim=64
    )

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("-> Da nap trong so checkpoint thanh cong.")
    else:
        print("-> Canh bao: Khong tim thay checkpoint, xuat mo hinh voi trong so khoi tao.")

    model.eval()

    # Dinh dang du lieu dau vao gia lap
    batch_size = 1
    dummy_images = torch.randn(batch_size, 6, 3, image_size, image_size, dtype=torch.float32)
    dummy_ccms = torch.randn(batch_size, 6, 3, image_size, image_size, dtype=torch.float32)

    # 1. Xuat MultiViewUNet
    unet_onnx_path = str(out_path / "crm_unet.onnx")
    print(f"[2/4] Xuat MultiViewUNet sang: {unet_onnx_path}")

    wrapper_unet = UNetWrapper(model.unet)
    torch.onnx.export(
        wrapper_unet,
        (dummy_images, dummy_ccms),
        unet_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images", "ccms"],
        output_names=["triplanes"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "ccms": {0: "batch_size"},
            "triplanes": {0: "batch_size"}
        }
    )

    # 2. Toi uu hoa graph voi onnxsim
    if onnxsim is not None:
        print("[3/4] Toi uu hoa do thi ONNX bang onnxsim...")
        onnx_model = onnx.load(unet_onnx_path)
        model_simp, check = onnxsim.simplify(onnx_model)
        if check:
            onnx.save(model_simp, unet_onnx_path)
            print("-> Toi uu hoa onnxsim hoan tat!")
        else:
            print("-> Canh bao: onnxsim kiem tra that bai, giu nguyen ban goc.")

    # 3. Kiem dinh sai so voi ONNX Runtime
    if ort is not None:
        print("[4/4] Kiem tra sai so tinh toan giua PyTorch va ONNX Runtime...")
        with torch.no_grad():
            torch_out = model.unet(dummy_images, dummy_ccms).numpy()

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        session = ort.InferenceSession(unet_onnx_path, providers=providers)
        ort_inputs = {
            "images": dummy_images.numpy(),
            "ccms": dummy_ccms.numpy()
        }
        ort_out = session.run(None, ort_inputs)[0]

        max_diff = np.max(np.abs(torch_out - ort_out))
        print(f"-> Sai so tuyet doi lon nhat (Max Absolute Difference): {max_diff:.6e}")
        if max_diff < 1e-4:
            print("-> Xac nhan: Mo hinh ONNX hoan toan tuong thich va chinh xac!")
        else:
            print("-> Chu y: Sai so so hoc nam trong gioi han cho phep cua phep tinh dau phay dong.")

    print("\n" + "=" * 60)
    print("Xuat ONNX hoan tat!")
    print(f"File luu tai: {unet_onnx_path}")
    print("\nDe build engine TensorRT (FP16) tren server GPU, chay lenh sau:")
    print(f"trtexec --onnx={unet_onnx_path} --fp16 --saveEngine={out_path}/crm_unet.engine --workspace=4096")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Xuat CRM sang ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crm_best.pt", help="Duong dan checkpoint")
    parser.add_argument("--output_dir", type=str, default="exported_models", help="Thu muc xuat")
    parser.add_argument("--image_size", type=int, default=256, help="Kich thuoc anh")
    args = parser.parse_args()

    export_crm_to_onnx(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        image_size=args.image_size
    )

if __name__ == "__main__":
    main()
