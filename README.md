# GPU LLM on WSL

A Dockerized environment for running quantized large language models (LLMs) with GPU acceleration on Windows via WSL2.

This setup is optimized for local inference with popular 4-bit/8-bit quantized models (Llama 2, Mistral, CodeLlama) using PyTorch, Hugging Face Transformers, and `bitsandbytes`.

---

## Key Features

- **CUDA 12.6 Environment**: Built on the `nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04` image.
- **Core Libraries**: Includes a pre-configured set of GPU-ready Python packages:
  - **PyTorch 2.9.1** (CUDA 12.6)
  - **Transformers 4.57.1**
  - **Bitsandbytes 0.48.2**
  - **Accelerate 1.11.0**
- **Ready to Use**: Includes Jupyter and Gradio for interactive model testing and building simple web UIs.
- **Self-Contained**: The environment is fully configured, including Python 3.11, system dependencies, and Hugging Face cache settings.

---

## Pre-requisites

- **Windows 10/11 with WSL2**: An Ubuntu-based distribution is recommended.
- **Docker Desktop**: Must be configured with WSL2 integration enabled.
- **NVIDIA Driver**: Your host system needs an NVIDIA driver that supports **CUDA 12.1 or higher**. Your host driver will be used by the container regardless of the base image CUDA version.
- **NVIDIA Container Toolkit**: Must be installed *inside your WSL2 distribution* to allow Docker to access the GPU. Run:
  ```bash
  sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```
- **Hardware**:
  - **12 GB+ VRAM** (e.g., RTX 3080 Ti or better) for 7B parameter models.
  - **32 GB+ System RAM**.

---

## Setup & Usage

1.  **Clone the repository:**
    ```powershell
    git clone https://github.com/yourname/gpu-llm-on-wsl.git
    cd gpu-llm-on-wsl
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t llm-docker .
    ```

3.  **Verify the environment (Recommended):**
    Run the built-in test script to confirm that the GPU and libraries are recognized correctly.
    ```bash
    docker run --gpus all -it --rm llm-docker python /app/test_env.py
    ```
    You should see output confirming your GPU is available and listing the package versions.
    For additional sanity checks and example inference commands, see `sanity-test.md`.

4.  **Run an interactive session:**
    ```bash
    docker run --gpus all -it llm-docker
    ```

## Common Use Cases

-   **Mount a local folder for model storage:**
    Avoid re-downloading models by mounting a directory from your host machine.
    ```bash
    docker run --gpus all -v C:\path\to\models:/app/models -it llm-docker
    ```

-   **Run a Jupyter Notebook server:**
    Expose the Jupyter port to access it from your browser at `http://localhost:8888`.
    ```bash
    docker run --gpus all -p 8888:8888 -it llm-docker jupyter notebook --ip=0.0.0.0 --allow-root
    ```
