# E-Commerce 3D AI Trainer (Convolutional Reconstruction Model - CRM)

Pipeline huan luyen va phuc vu mo hinh Convolutional Reconstruction Model (CRM) tu 1 anh san pham don le (ban lam viec, ghe cong thai hoc, monitor, gear cong nghe) thanh mo hinh 3D Mesh textured san sang tich hop vao storefront Next.js.

## 1. Tong quan kien truc

Pipeline gom hai giai doan cot loi:
1. **Stage 1 (Multi-view Synthesis)**: Sinh 6 goc nhin truc giao chuan (front, right, back, left, top, bottom) kem Canonical Coordinate Maps (CCM) ma hoa toa do 3D truc giao sang khong gian RGB.
2. **Stage 2 (CRM Reconstruction)**: Mang MultiViewUNet tich chap bien doi 6 goc nhin thanh 3 mat phang truc giao Triplane (XY, XZ, YZ). Implicit Point Decoder giai ma SDF (Signed Distance Function) va mau sac RGB, sau do trich xuat tam giac 3D Mesh bang Marching Cubes / FlexiCubes.

## 2. Cau truc thu muc

```
ecommerce-3d-ai-trainer/
├── dataset.py                # Dataset loader, chuan hoa camera pose, mask, lay mau SDF 3D
├── models/
│   ├── __init__.py
│   └── crm.py               # Kien truc MultiViewUNet + TriplaneSDFDecoder + Mesh Extractor
├── train.py                  # Script huan luyen voi SDF Loss, Chamfer Distance, LPIPS, AMP FP16
├── export_onnx.py            # Xuat mo hinh sang ONNX va huong dan build TensorRT FP16
├── app.py                    # Backend FastAPI cung cap endpoint /generate-3d phuc vu Next.js
├── evaluate.py               # Danh gia dinh luong PSNR, SSIM, LPIPS va Chamfer Distance
├── scripts/
│   └── blender_render.py    # Headless Blender render synthetic 6 views + CCM tu CAD/GLB
├── kaggle/                   # Tu dong hoa day notebook va training tren Kaggle GPU T4x2
│   ├── setup_kaggle.py
│   ├── push_training.py
│   └── notebooks/
├── Dockerfile                # Docker container ho tro CUDA 12.1 + FastAPI serving
├── docker-compose.yml
└── requirements.txt
```

## 3. Huong dan su dung

### Cai dat moi truong:
```bash
pip install -r requirements.txt
```

### Huan luyen cuc bo hoac tren Cloud GPU (T4 / A100):
```bash
python train.py --data_dir data/ecommerce_processed --epochs 30 --batch_size 2 --lr 1e-4
```

### Danh gia dinh luong:
```bash
python evaluate.py --checkpoint checkpoints/crm_best.pt --val_data data/val
```

### Xuat ONNX / TensorRT:
```bash
python export_onnx.py --checkpoint checkpoints/crm_best.pt --output_dir exported_models
```

### Khoi chay Serving API (FastAPI):
```bash
python app.py
# Hoac dung Docker Compose:
docker-compose up --build
```
