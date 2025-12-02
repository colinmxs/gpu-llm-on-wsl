# FastAPI API for LLM Inference

This directory contains the FastAPI server for streaming LLM inferences from the Docker container to the host machine. It also includes agent and tool management endpoints for creating and managing AI agents with the Strands SDK.

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

### Agent Management

- `POST /agents` - Create a new agent
- `GET /agents` - List all active agents
- `GET /agents/saved` - List all saved agent configurations
- `GET /agents/{agent_name}` - Get detailed info about an agent
- `POST /agents/{agent_name}/load` - Load an agent from disk
- `POST /agents/{agent_name}/save` - Save an agent to disk
- `DELETE /agents/{agent_name}` - Delete an agent (optionally from disk)
- `POST /agents/{agent_name}/chat/stream` - Chat with agent (streaming SSE)
- `POST /agents/{agent_name}/chat` - Chat with agent (non-streaming)

### Tool Management

- `POST /tools` - Create a new tool
- `GET /tools` - List all active tools
- `GET /tools/saved` - List all saved tool configurations
- `GET /tools/{tool_name}` - Get detailed info about a tool
- `POST /tools/{tool_name}/load` - Load a tool from disk
- `POST /tools/{tool_name}/save` - Save a tool to disk
- `DELETE /tools/{tool_name}` - Delete a tool (optionally from disk)

## Using the API from Your Local Python Code

### Install Requirements (Host Machine)

```bash
pip install requests
```

### Basic LLM Usage

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

### Agent and Tool Management

```python
from api.client_agents_tools_example import AgentToolClient

# Initialize client
client = AgentToolClient("http://localhost:8000")

# Create a tool
tool_result = client.create_tool(
    name="calculator",
    description="Adds two numbers",
    function_code="""
import strands

@strands.tool
def add_numbers(a: float, b: float) -> float:
    '''Add two numbers together.'''
    return a + b
"""
)

# Create an agent
agent_result = client.create_agent(
    name="math_tutor",
    description="A helpful math tutor",
    system_prompt="You are a patient math tutor.",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    tools=["calculator"]
)

# Chat with the agent (streaming)
for event in client.chat_with_agent_stream("math_tutor", "What is 5 + 7?"):
    if event['type'] == 'text':
        print(event['text'], end='', flush=True)
```

See `client_agents_tools_example.py` for a complete working example.

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
- `AGENTS_DIR` - Directory for agent configurations (default: ../agents)
- `TOOLS_DIR` - Directory for tool definitions (default: ../tools)

## CORS

CORS is enabled for all origins by default for development. In production, update the `allow_origins` in `server.py` to specify your frontend URL.

## Interactive Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
