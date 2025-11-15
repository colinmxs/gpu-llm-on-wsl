# gpu-llm-on-wsl

Dockerized environment for running quantized large language models (LLMs) with GPU acceleration on Windows via WSL2. Supports PyTorch, Hugging Face Transformers, and bitsandbytes for efficient local inference with models like Llama 2, Mistral, and CodeLlama.

---

## Pre-requisites
- Windows 10/11 with **WSL2** enabled (Ubuntu recommended)
- **Docker Desktop** with WSL2 integration
- Latest **NVIDIA GPU driver** for RTX cards
- **NVIDIA Container Toolkit** installed in WSL2 (`apt install nvidia-container-toolkit`)
- At least **12 GB VRAM** (RTX 3080 Ti or similar) and **32 GB system RAM**

---

## Setup
1. Clone the repo:
   ```powershell
   git clone https://github.com/yourname/gpu-llm-on-wsl.git
   cd gpu-llm-on-wsl

Build the Docker image:

docker build -t llm-docker .

Run with GPU access:

docker run --gpus all -it llm-docker

Notes

Use quantized 7B models (q4/q5) for smooth inference on 12 GB VRAM.

Mount a local folder for model storage:

docker run --gpus all -v C:\models:/app/models -it llm-docker

Expose ports if running a web UI (e.g., -p 7860:7860).

For larger models (13B+), expect slower inference and possible CPU offload.
