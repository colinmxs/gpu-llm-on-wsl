# Dockerfile for GPU-accelerated LLM inference on WSL2
# Base image: NVIDIA CUDA 12.6 with Ubuntu 22.04
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.6 support - PINNED VERSIONS
RUN pip install --no-cache-dir \
    torch==2.9.1+cu126 \
    torchvision==0.24.1+cu126 \
    torchaudio==2.9.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# Install ML packages for quantized LLM inference - PINNED VERSIONS
RUN pip install --no-cache-dir \
    transformers==4.57.1 \
    accelerate==1.11.0 \
    bitsandbytes==0.48.2 \
    scipy==1.16.3 \
    sentencepiece==0.2.1 \
    protobuf==6.33.1 \
    einops==0.8.1 \
    safetensors==0.7.0 \
    huggingface-hub==0.36.0

# Install additional core utilities - PINNED VERSIONS
RUN pip install --no-cache-dir \
    humanize==4.14.0 \
    pydantic==2.10.5

# Create working directory and model cache directory
WORKDIR /app
RUN mkdir -p /app/models /app/cache

# Install Notebook dependencies
COPY notebooks/requirements.txt /tmp/notebooks-requirements.txt
RUN pip install --no-cache-dir -r /tmp/notebooks-requirements.txt
COPY notebooks /app/notebooks

# Install Frontend dependencies
COPY frontend/requirements.txt /tmp/frontend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/frontend-requirements.txt
COPY frontend /app/frontend

# Set Hugging Face cache directory
ENV HF_HOME=/app/cache

# Expose common ports (Jupyter: 8888, Gradio: 7860)
EXPOSE 8888 7860

# Set the default command
CMD ["/bin/bash"]
