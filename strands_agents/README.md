# Strands Agents Module

**Example:**
```python
from pathlib import Path
from shared.model_manager import ModelManager
from strands_agents.agent_manager import AgentManager

# Initialize
model_manager = ModelManager(models_dir=Path("./models"))
agent_manager = AgentManager(
    agents_dir=Path("./agents"),
    model_manager=model_manager,
    tools_dir=Path("./tools")
)

# Create agent
result = agent_manager.create_agent(
    name="Assistant",
    description="Helpful assistant",
    system_prompt="You are a helpful AI assistant.",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    tools=["calculator", "search"],
    temperature=0.7,
    max_tokens=500
)

# Chat with agent
for event in agent_manager.chat_with_agent(
    agent_name="Assistant",
    message="Hello!",
    history=[]
):
    if event['type'] == 'token':
        print(event['text'], end='', flush=True)
    elif event['type'] == 'complete':
        print(f"\n[{event['tokens_per_second']:.1f} tokens/sec]")
```

### 2. HuggingFaceLocalModel (`huggingface_local_model.py`)

Custom Strands model provider that bridges `ModelManager` with Strands SDK.

**Example:**
```python
from strands_agents.huggingface_local_model import HuggingFaceLocalModel
from strands.agent import Agent

# Create model provider
model = HuggingFaceLocalModel(
    model_manager=model_manager,
    temperature=0.7,
    max_tokens=500
)

# Use with Strands Agent
agent = Agent(
    name="MyAgent",
    model=model,
    system_prompt="You are helpful."
)
```

### 3. ToolManager (`tool_manager.py`)

Manages Strands SDK tools for agents.

**Example:**
```python
from strands_agents.tool_manager import ToolManager, ToolConfig

# Initialize
tool_manager = ToolManager(tools_dir=Path("./tools"))

# Create a tool
config = ToolConfig(
    name="calculator",
    description="Perform basic math calculations",
    function_code="""
def calculate(expression: str) -> float:
    '''Evaluate a mathematical expression'''
    return eval(expression)
""",
    parameters_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        }
    },
    returns_schema={"type": "number"}
)

result = tool_manager.create_tool(config)
# Creates: ./tools/calculator.py (with @strands.tool)
#          ./tools/calculator.json (metadata)
```
## Usage Patterns

### Creating and Using an Agent

```python
from pathlib import Path
from shared.model_manager import ModelManager
from strands_agents.agent_manager import AgentManager

# Setup
model_manager = ModelManager(models_dir=Path("./models"))
agent_manager = AgentManager(
    agents_dir=Path("./agents"),
    model_manager=model_manager,
    tools_dir=Path("./tools")
)

# Load model
model_manager.load_model(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization="4-bit (NF4)"
)

# Create agent
agent_manager.create_agent(
    name="CodeHelper",
    description="Python coding assistant",
    system_prompt="You are an expert Python developer.",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.3,
    max_tokens=1000
)

# Chat
history = []
for event in agent_manager.chat_with_agent(
    agent_name="CodeHelper",
    message="Write a function to calculate fibonacci numbers",
    history=history
):
    if event['type'] == 'token':
        print(event['text'], end='', flush=True)
    elif event['type'] == 'complete':
        print(f"\n\n[{event['tokens_per_second']:.1f} tok/s]")
```

### Agent Persistence

```python
# List saved agents
saved = agent_manager.list_saved_agents()
print(f"Saved agents: {saved}")

# List active agents (in memory)
active = agent_manager.list_active_agents()
print(f"Active agents: {active}")

# Get agent info
info = agent_manager.get_agent_info("CodeHelper")
print(f"Agent: {info['name']}")
print(f"Model: {info['model_name']}")
print(f"Temperature: {info['temperature']}")
print(f"Tools: {info['tools']}")

# Delete agent
agent_manager.delete_agent("CodeHelper", delete_file=True)
```

## File Formats

### Agent Configuration (JSON)

```json
{
  "name": "Assistant",
  "description": "Helpful assistant",
  "system_prompt": "You are a helpful AI assistant.",
  "tools": ["./tools/calculator.py"],
  "_model_config": {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
  }
}
```

### Tool Configuration (JSON)

```json
{
  "name": "calculator",
  "description": "Perform basic math calculations",
  "function_code": "import strands\n\n@strands.tool\ndef calculate(expression: str) -> float:\n    return eval(expression)",
  "parameters_schema": {
    "type": "object",
    "properties": {
      "expression": {"type": "string"}
    }
  },
  "returns_schema": {"type": "number"}
}
```