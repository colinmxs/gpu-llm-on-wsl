# GPU LLM on WSL - Setup Guide

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t gpu-llm-api .
```

### 2. Start the API Server

**Option A: Using Docker Compose (Recommended)**

```bash
docker-compose up
```

**Option B: Using Docker Run**

```bash
docker run --gpus all -p 8000:8000 -p 7860:7860 -p 7861:7861 -p 8888:8888 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/agents:/app/agents \
  -v $(pwd)/tools:/app/tools \
  gpu-llm-api \
  python -m api.server
```

### 3. Test the API

Open your browser to:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

Or use the Python client:

```bash
# On your host machine
python api/client_example.py
```

See [api/README.md](api/README.md) for detailed API documentation.

## Environment Variables

Configure the API server with these environment variables:

- `API_PORT` - Port for API (default: 8000)
- `API_HOST` - Host to bind (default: 0.0.0.0)
- `MODELS_DIR` - Models directory (default: /app/models)
- `CACHE_DIR` - HuggingFace cache (default: /app/cache)
