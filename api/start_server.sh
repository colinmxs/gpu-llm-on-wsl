#!/bin/bash
# Start the FastAPI LLM Inference server

# Configuration
export API_PORT="${API_PORT:-8000}"
export API_HOST="${API_HOST:-0.0.0.0}"
export MODELS_DIR="${MODELS_DIR:-/app/models}"
export CACHE_DIR="${CACHE_DIR:-/app/cache}"

echo "=========================================="
echo "Starting LLM Inference API Server"
echo "=========================================="
echo "Host: $API_HOST"
echo "Port: $API_PORT"
echo "Models Directory: $MODELS_DIR"
echo "Cache Directory: $CACHE_DIR"
echo "=========================================="
echo ""

# Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
if [ $? -eq 0 ]; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 
fi

echo ""
echo "Starting server..."
echo "API Documentation: http://localhost:$API_PORT/docs"
echo ""

# Start the server
cd /app
python -m api.server
