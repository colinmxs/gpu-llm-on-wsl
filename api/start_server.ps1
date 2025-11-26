# Start the FastAPI LLM Inference server (PowerShell)

# Configuration
$env:API_PORT = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
$env:API_HOST = if ($env:API_HOST) { $env:API_HOST } else { "0.0.0.0" }
$env:MODELS_DIR = if ($env:MODELS_DIR) { $env:MODELS_DIR } else { "/app/models" }
$env:CACHE_DIR = if ($env:CACHE_DIR) { $env:CACHE_DIR } else { "/app/cache" }

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Starting LLM Inference API Server" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Host: $env:API_HOST"
Write-Host "Port: $env:API_PORT"
Write-Host "Models Directory: $env:MODELS_DIR"
Write-Host "Cache Directory: $env:CACHE_DIR"
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""None""}')"

Write-Host ""
Write-Host "Starting server..."
Write-Host "API Documentation: http://localhost:$env:API_PORT/docs" -ForegroundColor Green
Write-Host ""

# Start the server
Set-Location /app
python -m api.server
