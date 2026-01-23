# GPU LLM on WSL

A simple Dockerized environment for downloading, testing, and serving quantized large language models (LLMs) with GPU acceleration on Windows via WSL2.

---

## Features

- **Model Downloads**: Jupyter notebook interface for downloading quantized models from Hugging Face
- **Inference Testing**: Simple Gradio web UI for testing model inference with GPU acceleration
- **OpenAI-Compatible API**: Serve models via API for integration with agent frameworks (Strands, LangChain, etc.)
- **CUDA 12.6 Support**: Pre-configured with PyTorch and CUDA for optimal performance
- **Quantization**: Built-in support for bitsandbytes 4-bit and 8-bit quantization

---

## Prerequisites

- **System**: Windows 10/11 with WSL2 and Docker Desktop (WSL2 integration enabled).
- **NVIDIA Driver**: Host driver must support **CUDA 12.1 or higher**.
- **NVIDIA Container Toolkit**: Install in your WSL2 distribution to give Docker GPU access.
  ```bash
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```
- **Hardware**:
  - **12 GB+ VRAM** for 7B parameter models.
  - **32 GB+ System RAM**.

---

## Quick Start

1.  **Clone & Build**
    ```powershell
    git clone https://github.com/colinmxs/gpu-llm-on-wsl.git
    cd gpu-llm-on-wsl
    docker build -t llm-docker .
    ```

2.  **Run a Command**

    All commands should mount a local directory to `/app/models` to persist models. Replace `C:\path\to\models` with a directory on your machine.

    -   **Download Models (Jupyter)**: Use a notebook to download models from Hugging Face.
        ```bash
        docker run --gpus all -p 8888:8888 -v C:\path\to\models:/app/models -it llm-docker jupyter notebook --ip=0.0.0.0 --allow-root
        ```
        Navigate to `http://localhost:8888`, open `notebooks/hf-model-manager.ipynb`, and use the interface to download models.

    -   **Test Models (Gradio UI)**: Launch a web UI to test your downloaded models.
        ```bash
        docker run --gpus all -p 7860:7860 -v C:\path\to\models:/app/models -it llm-docker python /app/frontend/gradio_frontend.py
        ```
        Open `http://localhost:7860` in your browser.

    -   **API Server (for Agents)**: Run the OpenAI-compatible API for agent integration.
        ```bash
        docker run --gpus all -p 8000:8000 -v C:\path\to\models:/app/models -it llm-docker python /app/api/server.py
        ```
        Open `http://localhost:8000/docs` for interactive API documentation.
    
    -   **Interactive Shell**: Open a bash shell inside the container for manual control.
        ```bash
        docker run --gpus all -v C:\path\to\models:/app/models -it llm-docker
        ```

---

## Agent Integration

The API server adds OpenAI-compatible `v1/chat/completions` and `v1/models` endpoints to your local models, supporting tool calling and streaming.

**Point your agent framework (OpenAI SDK, LangChain, Strands, etc.) to:**
- **Base URL**: `http://localhost:8000/v1`
- **Reference**: See `http://localhost:8000/docs` for full API details

