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

# Install PyTorch with CUDA 12.6 support (latest stable)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install latest ML packages for quantized LLM inference
RUN pip install --no-cache-dir \
    transformers==4.57.1 \
    accelerate==1.11.0 \
    bitsandbytes==0.48.2 \
    scipy \
    sentencepiece \
    protobuf \
    einops \
    safetensors \
    huggingface-hub

# Install additional utilities for model management and inference
RUN pip install --no-cache-dir \
    gradio \
    jupyter \
    ipywidgets \
    matplotlib \
    pandas \
    numpy \
    humanize

# Install Strands SDK and its dependencies
COPY requirements-strands.txt /tmp/requirements-strands.txt
RUN pip install --no-cache-dir -r /tmp/requirements-strands.txt

# Create working directory and model cache directory
WORKDIR /app
RUN mkdir -p /app/models /app/cache /app/agents /app/tools

# Copy utility notebooks and frontend into the image
COPY notebooks /app/notebooks
COPY frontend /app/frontend

# Copy example agents and tools to reference directories
COPY frontend/example_agents /app/example_agents
COPY frontend/example_tools /app/example_tools

# Set Hugging Face cache directory
ENV HF_HOME=/app/cache

# Create a simple test script
RUN echo '#!/usr/bin/env python\n\
import torch\n\
import transformers\n\
import bitsandbytes\n\
import accelerate\n\
\n\
print("=" * 50)\n\
print("GPU LLM Environment - Configuration")\n\
print("=" * 50)\n\
print(f"PyTorch version: {torch.__version__}")\n\
print(f"Transformers version: {transformers.__version__}")\n\
print(f"Accelerate version: {accelerate.__version__}")\n\
print(f"Bitsandbytes version: {bitsandbytes.__version__}")\n\
print(f"CUDA available: {torch.cuda.is_available()}")\n\
if torch.cuda.is_available():\n\
    print(f"CUDA version: {torch.version.cuda}")\n\
    print(f"GPU device: {torch.cuda.get_device_name(0)}")\n\
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")\n\
print("=" * 50)\n\
' > /app/test_env.py && chmod +x /app/test_env.py

# Expose common ports (Jupyter: 8888, Gradio: 7860, Agent Playground: 7861)
EXPOSE 8888 7860 7861

# Set the default command
CMD ["/bin/bash"]
