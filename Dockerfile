# Su dung base image Nvidia CUDA 12.1 phu hop PyTorch 2.1+
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Cai dat Python va cac thu vien he thong can thiet cho do hoa EGL / OpenGL
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cai dat dependencies Python
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Sao chep ma nguon ung dung
COPY . /app

# Tao thu muc chua checkpoints va outputs
RUN mkdir -p /app/checkpoints /app/static/generated_3d

# Expose port FastAPI
EXPOSE 8000

# Bien moi truong
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV CRM_CHECKPOINT=/app/checkpoints/crm_best.pt

# Lenh chay serving
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
