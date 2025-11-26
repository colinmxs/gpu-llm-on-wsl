# GPU LLM on WSL

A Dockerized environment for running quantized large language models (LLMs) with GPU acceleration on Windows via WSL2. This setup is optimized for local inference using PyTorch, Hugging Face Transformers, and `bitsandbytes`.

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
    
    **Optional:** Mount a local directory to `/app/agents` to persist agent configurations. Replace `C:\path\to\agents` with a directory on your machine.

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

    -   **FastAPI Server (NEW)**: Run the inference API for local development.
        ```bash
        docker run --gpus all -p 8000:8000 -v C:\path\to\models:/app/models -it llm-docker python -m api.server
        ```
        Open `http://localhost:8000/docs` for API documentation. See [API_SETUP.md](API_SETUP.md) for detailed usage.

    -   **Agent Playground**: Build and test Strands SDK agents with tools.
        ```bash
        docker run --gpus all -p 7861:7861 -v C:\path\to\models:/app/models -v C:\path\to\agents:/app/agents -v C:\path\to\tools:/app/tools -it llm-docker python /app/frontend/agent_playground.py
        ```
        Open `http://localhost:7861` in your browser. Mount `/app/tools` to persist tool definitions.

    -   **Interactive Shell**: Open a bash shell inside the container for manual control.
        ```bash
        docker run --gpus all -v C:\path\to\models:/app/models -v C:\path\to\agents:/app/agents -v C:\path\to\tools:/app/tools -it llm-docker
        ```

---

For advanced validation and sanity checks, see `sanity-test.md`.