# Frontend

This directory contains the Gradio web UI and its backend `ModelManager` engine.

## Architecture & Core Components

The frontend is split into two main parts:

```
frontend/
├── gradio_frontend.py    # Gradio UI (The View)
└── model_manager.py      # Backend Engine (The Controller)
```

- **`model_manager.py`**: A UI-agnostic class that handles model loading, memory management, and text generation. It returns pure data (dictionaries) and can be used by any Python application (CLI, FastAPI, etc.).
- **`gradio_frontend.py`**: A Gradio web interface that consumes the `ModelManager` and formats its output for display.

## Usage

### Launch the Gradio Interface

From the host machine, run the following Docker command:

```bash
docker run --gpus all -p 7860:7860 -v C:\path\to\models:/app/models -it llm-docker python /app/frontend/gradio_frontend.py
```

Then open `http://localhost:7860` in your browser.

## Using the Backend Engine (`ModelManager`)

To use the backend engine in your own scripts, import and use the `ModelManager` class.

```python
from pathlib import Path
from model_manager import ModelManager

# 1. Initialize the manager
manager = ModelManager(models_dir=Path("/app/models"))

# 2. Load a model
manager.load_model("meta-llama/Llama-3.1-8B", "4-bit (NF4)")

# 3. Generate text
for event in manager.generate(prompt="Explain Docker in one sentence.", max_tokens=50):
    if event["type"] == "token":
        print(event["text"], end="", flush=True)

# 4. Unload the model
manager.unload_model()
```

## Extending

- **Add Prompt Formats**: To add a new prompt format, edit the `PROMPT_TEMPLATES` dictionary in `frontend/templates.py`. The UI will update automatically.
- **Create a New Frontend**: Import the `ModelManager` class into your application as shown in the example above. It provides all the necessary backend logic.
