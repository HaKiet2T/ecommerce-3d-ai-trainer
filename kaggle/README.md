# Kaggle Training Automation

Hệ thống tự động đẩy notebook huấn luyện CRM 3D lên Kaggle GPU T4x2.

## Cấu trúc

```
kaggle/
├── setup_kaggle.py          # Script cài đặt và xác thực Kaggle API
├── push_training.py         # Script tự động push + monitor notebook
├── notebooks/
│   ├── stage1_lora_zero123/
│   │   ├── kernel-metadata.json
│   │   └── train_stage1.py
│   ├── stage2_crm_finetune/
│   │   ├── kernel-metadata.json
│   │   └── train_stage2.py
│   └── data_preparation/
│       ├── kernel-metadata.json
│       └── prepare_data.py
└── configs/
    └── kaggle_config.yaml
```

## Sử dụng

```bash
# 1. Cài đặt Kaggle API
python kaggle/setup_kaggle.py

# 2. Push notebook data preparation
python kaggle/push_training.py --stage data_prep

# 3. Push Stage 1 (LoRA Zero123++)
python kaggle/push_training.py --stage stage1

# 4. Push Stage 2 (CRM fine-tune)
python kaggle/push_training.py --stage stage2

# 5. Monitor + tải output
python kaggle/push_training.py --stage stage2 --monitor
```
