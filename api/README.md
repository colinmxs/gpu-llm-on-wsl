# FastAPI API for LLM Inference

This directory contains the FastAPI server for streaming LLM inferences from the Docker container to the host machine.

## API Endpoints

### Health & Status

- `GET /` - Health check
- `GET /health` - Detailed health check
- `GET /status` - Full system status (model, GPU stats)

### Model Management

- `GET /models` - List all available models
- `GET /models/{model_name}` - Get info about a specific model
- `POST /model/load` - Load a model
- `POST /model/unload` - Unload current model
- `GET /model/current` - Get currently loaded model name

### GPU Stats

- `GET /gpu` - Get GPU memory statistics

### Text Generation

- `POST /generate/stream` - Stream generated text (SSE)
- `POST /generate` - Generate text (non-streaming)

## Using the API from Your Local Python Code

### Install Requirements (Host Machine)

```bash
pip install requests
```

### Basic Usage

```python
from api.client_example import LLMClient

# Initialize client
client = LLMClient("http://localhost:8000")

# Check status
status = client.get_status()
print(f"Model loaded: {status['model_loaded']}")

# Load a model
client.load_model("meta-llama/Llama-3.1-8B-Instruct", "4-bit (NF4)")

# Generate text with streaming
for event in client.generate_stream("Write a haiku about coding"):
    if event['type'] == 'token':
        print(event['text'], end='', flush=True)
    elif event['type'] == 'complete':
        print(f"\n{event['tokens_per_second']:.2f} tokens/sec")
```

## Example Usage

### Load a Model

```python
import requests

response = requests.post("http://localhost:8000/model/load", json={
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "quantization": "4-bit (NF4)"
})
print(response.json())
```

### Stream Generation (SSE)

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/generate/stream",
    json={
        "prompt": "Explain quantum computing in simple terms.",
        "max_tokens": 200,
        "temperature": 0.7
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            event = json.loads(line[6:])
            if event['type'] == 'token':
                print(event['text'], end='', flush=True)
            elif event['type'] == 'complete':
                print(f"\n\nGeneration complete!")
                print(f"Tokens: {event['total_tokens']}")
                print(f"Speed: {event['tokens_per_second']:.2f} tokens/sec")
```

### Non-Streaming Generation

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "Write a haiku about programming.",
    "max_tokens": 100,
    "temperature": 0.9
})

result = response.json()
print(result['text'])
print(f"Generated in {result['metadata']['elapsed_seconds']:.2f}s")
```

## Environment Variables

- `API_PORT` - Port to run the API on (default: 8000)
- `API_HOST` - Host to bind to (default: 0.0.0.0)
- `MODELS_DIR` - Directory containing models (default: /app/models)
- `CACHE_DIR` - HuggingFace cache directory (default: /app/cache)

## CORS

CORS is enabled for all origins by default for development. In production, update the `allow_origins` in `server.py` to specify your frontend URL.

## Interactive Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
