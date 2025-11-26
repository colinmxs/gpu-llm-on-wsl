# Strands SDK Native Implementation Refactor

## Overview
Complete rewrite of the agent system to be **fully Strands-native**, properly using the Strands Agents SDK instead of bypassing it.

## What Changed

### ✅ NEW: HuggingFace Local Model Adapter
**File**: `frontend/huggingface_local_model.py`

Created custom `HuggingFaceLocalModel` class implementing `strands.models.Model`:
- Wraps `model_manager.generate()` to provide Strands-compatible streaming
- Converts model_manager events to Strands `StreamEvent` format
- Transforms Strands `Messages` format to prompt strings for local models
- Enables `model_manager` to remain SDK-agnostic while working with Strands

**Key Methods**:
- `stream()`: Async streaming compatible with Strands Agent
- `_format_messages_to_prompt()`: Converts Strands Messages to prompt format
- `_convert_to_strands_event()`: Translates model_manager events to Strands events

### ✅ REWRITTEN: Agent Manager
**File**: `frontend/agent_manager.py`

**Before (Wrong)**:
```python
# Created Agent WITHOUT model parameter
agent = Agent(
    name=config.name,
    system_prompt=config.system_prompt,
    tools=tools_list,
    agent_id=config.agent_id,
    # ❌ No model parameter!
)

# Bypassed Agent entirely, called model_manager directly
prompt = build_prompt(...)
for event in self.model_manager.generate(prompt, ...):
    yield event
```

**After (Correct)**:
```python
# Create Agent WITH custom model provider
model_provider = HuggingFaceLocalModel(
    model_manager=self.model_manager,
    temperature=temperature,
    max_tokens=max_tokens,
    ...
)

agent = Agent(
    name=name,
    system_prompt=system_prompt,
    model=model_provider,  # ✅ Proper model parameter
    tools=tools_list
)

# Use Agent's native streaming
async for event in agent.stream_async(messages):
    yield event
```

**Key Changes**:
1. **Removed `AgentConfig` dataclass** - Use Strands Agent's built-in `to_dict()`/`from_dict()` for serialization
2. **Create agents with `model` parameter** - Pass `HuggingFaceLocalModel` instance
3. **Use `Agent.stream_async()`** - Proper Strands streaming with tool execution
4. **Store model config separately** - Keep model parameters (temperature, etc.) outside Strands Agent
5. **Sync wrapper for async** - `chat_with_agent()` wraps `chat_with_agent_async()` for Gradio compatibility

**New API**:
```python
# Create agent
agent_manager.create_agent(
    name="My Agent",
    description="...",
    system_prompt="...",
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    tools=["calculator", "web_search"],
    temperature=0.7,
    max_tokens=500,
    ...
)

# Chat with agent (uses Strands streaming)
for event in agent_manager.chat_with_agent(agent_name, message, history):
    # event is Strands StreamEvent
    if event["type"] == "content_block_delta":
        ...
```

### ✅ UPDATED: Agent Playground
**File**: `frontend/agent_playground.py`

**Changes**:
1. **Removed `AgentConfig` import** - No longer needed
2. **Updated `create_new_agent()`** - Call new `agent_manager.create_agent()` API directly
3. **Updated `populate_agent_form()`** - Use `agent_manager.get_agent_info()` instead of `get_agent_config()`
4. **Enhanced `chat_with_agent_handler()`** - Handle Strands event types:
   - `content_block_delta` - Streaming tokens
   - `message_complete` - Final message with metadata
   - Backward compatible with old event format

**Event Handling**:
```python
for event in agent_manager.chat_with_agent(agent_name, message, history):
    event_type = event.get("type")
    
    if event_type == "content_block_delta":
        # Strands streaming token
        delta = event.get("delta", {})
        if delta.get("type") == "text_delta":
            token = delta.get("text", "")
            assistant_response += token
    
    elif event_type == "message_complete":
        # Final message with stats
        msg = event.get("message", {})
        metadata = event.get("metadata", {})
        ...
```

## Architecture

### Before (Broken)
```
User Input → agent_playground.py → agent_manager.py
                                           ↓
                                    [Create Agent WITHOUT model]
                                           ↓
                                    [Bypass Agent entirely]
                                           ↓
                                    model_manager.generate() ← Direct call
                                           ↓
                                    Response
```

**Problems**:
- Agent created but never used
- Manual prompt building
- No tool execution
- No conversation management
- Only used Strands for tool loading

### After (Correct)
```
User Input → agent_playground.py → agent_manager.py
                                           ↓
                                    [Create Agent WITH HuggingFaceLocalModel]
                                           ↓
                                    Agent.stream_async(messages) ← Proper Strands API
                                           ↓
                                    HuggingFaceLocalModel.stream()
                                           ↓
                                    model_manager.generate()
                                           ↓
                                    Convert to Strands events
                                           ↓
                                    Response with tool execution
```

**Benefits**:
- ✅ Agent properly initialized with model
- ✅ Use Agent's built-in conversation management
- ✅ Native Strands streaming with `StreamEvent` format
- ✅ Tool execution handled by Agent
- ✅ Proper adapter pattern keeps `model_manager` SDK-agnostic
- ✅ Full Strands functionality available

## Files Changed

1. **NEW**: `frontend/huggingface_local_model.py` (271 lines)
   - Custom Strands Model provider for local HuggingFace models

2. **REWRITTEN**: `frontend/agent_manager.py` (415 lines)
   - Removed `AgentConfig` dataclass
   - Use Strands Agent.to_dict/from_dict for serialization
   - Create agents with model parameter
   - Use Agent.stream_async() for chat
   - Async streaming with sync wrapper

3. **UPDATED**: `frontend/agent_playground.py` (3 functions)
   - `create_new_agent()` - Use new API
   - `populate_agent_form()` - Use get_agent_info()
   - `chat_with_agent_handler()` - Handle Strands events

## Unchanged Files

- `frontend/model_manager.py` - Remains SDK-agnostic ✅
- `frontend/tool_manager.py` - Basic @strands.tool decorator validation ✅
- `frontend/gradio_frontend.py` - Simple GPU LLM testing interface ✅

## Testing Plan

1. **Unit Tests** (TODO):
   - Test `HuggingFaceLocalModel.stream()` with mock model_manager
   - Test event conversion in `_convert_to_strands_event()`
   - Test agent serialization with `to_dict()`/`from_dict()`

2. **Integration Tests** (TODO):
   - Create agent with custom model
   - Chat with agent and verify streaming
   - Load/save agents to disk
   - Test with tools

3. **Manual Testing**:
   - Build Docker container
   - Load a model
   - Create an agent
   - Chat and verify streaming works
   - Test with tools

## Running the Code

```bash
# Build Docker container
docker build -t gpu-llm-on-wsl .

# Run with GPU support
docker run --gpus all -p 7860:7860 -v ./models:/app/models gpu-llm-on-wsl

# Access at http://localhost:7860
```

## Migration Notes

### For Existing Agent Configs

Old agent configs (with `AgentConfig` format) will need migration:

**Old format**:
```json
{
  "name": "My Agent",
  "description": "...",
  "system_prompt": "...",
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "temperature": 0.7,
  "tools": ["calculator"]
}
```

**New format** (Strands native):
```json
{
  "name": "My Agent",
  "description": "...",
  "system_prompt": "...",
  "tools": ["/app/tools/calculator.py"],
  "_model_config": {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "temperature": 0.7,
    "max_tokens": 500,
    ...
  }
}
```

The `_model_config` section is our custom addition (not part of Strands) to store model parameters separately.

## Benefits of This Refactor

1. **Proper Strands Usage**: Now actually uses Strands Agent class instead of creating empty agents
2. **Tool Execution**: Strands Agent handles tool execution natively
3. **Conversation Management**: Uses Strands built-in conversation handling
4. **Streaming**: Proper Strands StreamEvent format with metadata
5. **Serialization**: Uses Strands to_dict/from_dict instead of custom dataclass
6. **Maintainability**: Less custom code, more reliance on well-tested SDK
7. **Future-Proof**: Can easily add more Strands features (conversation managers, callbacks, etc.)

## Next Steps

1. ✅ Create HuggingFaceLocalModel adapter
2. ✅ Rewrite agent_manager.py
3. ✅ Update agent_playground.py
4. ⏳ Test integration with Docker
5. 📋 Add callback handlers for better UI updates (optional)
6. 📋 Add ConversationManager support (optional)
7. 📋 Write unit tests (optional)

## Known Limitations

1. **No structured output**: Local models don't natively support structured output (would need parsing)
2. **Async in sync context**: Using `asyncio.new_event_loop()` for Gradio compatibility
3. **Model config separate**: Store model params outside Strands Agent (by design)

## Documentation References

- [Strands SDK Docs](https://docs.strands.ai/)
- [Agent API](https://docs.strands.ai/api/agent)
- [Model Providers](https://docs.strands.ai/api/models)
- [Streaming](https://docs.strands.ai/guides/streaming)
