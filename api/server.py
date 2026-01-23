"""
OpenAI-Compatible API Server for Local LLM Inference

Provides /v1/chat/completions and /v1/models endpoints compatible with
the OpenAI Python SDK, enabling integration with agent frameworks like
Strands SDK, LangChain, LlamaIndex, etc.
"""

import json
import time
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "frontend"))
from model_manager import ModelManager


# =============================================================================
# Pydantic Models (OpenAI API Schema)
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.1
    stream: Optional[bool] = False
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = Path("/app/models")
CACHE_DIR = Path("/app/cache")
DEFAULT_QUANTIZATION = "4-bit (NF4)"

# Global model manager instance
model_manager: Optional[ModelManager] = None


# =============================================================================
# Lifespan & App Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model manager on startup."""
    global model_manager
    
    print("=" * 80)
    print("OpenAI-Compatible API Server")
    print("=" * 80)
    
    model_manager = ModelManager(MODELS_DIR, CACHE_DIR)
    
    models = model_manager.list_models()
    print(f"\n📦 Found {len(models)} model(s) available:")
    for m in models:
        print(f"   - {m}")
    
    print(f"\n🚀 API server ready!")
    print(f"   Docs: http://localhost:8000/docs")
    print(f"   Models: http://localhost:8000/v1/models")
    print("=" * 80)
    
    yield
    
    # Cleanup
    if model_manager and model_manager.is_model_loaded():
        model_manager.unload_model()


app = FastAPI(
    title="Local LLM API",
    description="OpenAI-compatible API for local LLM inference",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

def build_chat_prompt(messages: List[ChatMessage], tools: Optional[List[ToolDefinition]] = None) -> str:
    """
    Convert OpenAI-style messages to a prompt string.
    Supports tool definitions for function calling.
    """
    prompt_parts = []
    
    # Add tool definitions if present (for function calling)
    if tools:
        tool_descriptions = []
        for tool in tools:
            func = tool.function
            tool_desc = {
                "name": func.name,
                "description": func.description or "",
                "parameters": func.parameters or {}
            }
            tool_descriptions.append(tool_desc)
        
        tools_text = json.dumps(tool_descriptions, indent=2)
        prompt_parts.append(f"You have access to the following tools:\n\n{tools_text}\n")
        prompt_parts.append("To use a tool, respond with a JSON object in this format:")
        prompt_parts.append('{"tool_calls": [{"id": "call_xxx", "type": "function", "function": {"name": "tool_name", "arguments": "{...}"}}]}')
        prompt_parts.append("\nOnly use tools when necessary. Respond normally for regular conversation.\n")
    
    # Build conversation
    for msg in messages:
        role = msg.role
        content = msg.content or ""
        
        if role == "system":
            prompt_parts.append(f"System: {content}\n")
        elif role == "user":
            prompt_parts.append(f"User: {content}\n")
        elif role == "assistant":
            if msg.tool_calls:
                # Include tool call in context
                tool_call_str = json.dumps({"tool_calls": msg.tool_calls})
                prompt_parts.append(f"Assistant: {tool_call_str}\n")
            else:
                prompt_parts.append(f"Assistant: {content}\n")
        elif role == "tool":
            # Tool response
            prompt_parts.append(f"Tool Result ({msg.tool_call_id}): {content}\n")
    
    prompt_parts.append("Assistant:")
    return "".join(prompt_parts)


def parse_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Attempt to parse tool calls from model output.
    Returns None if no valid tool calls found.
    """
    # Try to find JSON with tool_calls
    try:
        # Look for JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            parsed = json.loads(json_str)
            if "tool_calls" in parsed:
                return parsed["tool_calls"]
    except (json.JSONDecodeError, KeyError):
        pass
    return None


async def ensure_model_loaded(requested_model: str) -> None:
    """
    Ensure the requested model is loaded, auto-switching if necessary.
    """
    global model_manager
    
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    current_model = model_manager.get_current_model_name()
    
    # Check if we need to load a different model
    if current_model != requested_model:
        available_models = model_manager.list_models()
        
        # Find matching model (exact or partial match)
        matched_model = None
        for m in available_models:
            if m == requested_model or requested_model in m or m in requested_model:
                matched_model = m
                break
        
        if not matched_model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{requested_model}' not found. Available: {available_models}"
            )
        
        print(f"\n🔄 Auto-loading model: {matched_model}")
        result = model_manager.load_model(matched_model, DEFAULT_QUANTIZATION)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to load model"))
        
        print(f"✅ Model loaded: {matched_model}")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_model_loaded() if model_manager else False,
        "current_model": model_manager.get_current_model_name() if model_manager else None
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)."""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    models = model_manager.list_models()
    return ModelsResponse(
        data=[ModelInfo(id=m) for m in models]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Supports streaming and tool/function calling.
    """
    global model_manager
    
    # Auto-load the requested model
    await ensure_model_loaded(request.model)
    
    # Build prompt from messages
    prompt = build_chat_prompt(request.messages, request.tools)
    
    if request.stream:
        return StreamingResponse(
            stream_chat_response(request, prompt),
            media_type="text/event-stream"
        )
    else:
        return await generate_chat_response(request, prompt)


async def generate_chat_response(request: ChatCompletionRequest, prompt: str) -> ChatCompletionResponse:
    """Generate a non-streaming chat response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_time = int(time.time())
    
    full_text = ""
    total_tokens = 0
    
    # Run generation in thread pool to not block
    loop = asyncio.get_event_loop()
    
    def generate():
        nonlocal full_text, total_tokens
        for event in model_manager.generate(
            prompt=prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9,
            top_k=request.top_k or 50,
            repetition_penalty=request.repetition_penalty or 1.1,
            skip_prompt=True
        ):
            if event["type"] == "complete":
                full_text = event["text"]
                total_tokens = event["total_tokens"]
            elif event["type"] == "error":
                raise HTTPException(status_code=500, detail=event["error"])
    
    await loop.run_in_executor(None, generate)
    
    # Check for tool calls in response
    tool_calls = None
    finish_reason = "stop"
    
    if request.tools:
        tool_calls = parse_tool_calls(full_text)
        if tool_calls:
            finish_reason = "tool_calls"
            # Clean up the content if we found tool calls
            full_text = None
    
    return ChatCompletionResponse(
        id=completion_id,
        created=created_time,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=full_text,
                    tool_calls=tool_calls
                ),
                finish_reason=finish_reason
            )
        ],
        usage=Usage(
            prompt_tokens=len(prompt.split()),  # Approximate
            completion_tokens=total_tokens,
            total_tokens=len(prompt.split()) + total_tokens
        )
    )


async def stream_chat_response(request: ChatCompletionRequest, prompt: str):
    """Generate a streaming chat response using SSE."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_time = int(time.time())
    
    # Helper to format SSE
    def format_sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"
    
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    
    def generate():
        try:
            for event in model_manager.generate(
                prompt=prompt,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.9,
                top_k=request.top_k or 50,
                repetition_penalty=request.repetition_penalty or 1.1,
                skip_prompt=True
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "error": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)
    
    # Start generation in background thread
    import threading
    thread = threading.Thread(target=generate)
    thread.start()
    
    accumulated_text = ""
    
    while True:
        event = await queue.get()
        
        if event is None:
            break
        
        if event["type"] == "error":
            error_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n\nError: {event['error']}"},
                    "finish_reason": "error"
                }]
            }
            yield format_sse(error_chunk)
            break
        
        elif event["type"] == "token":
            # Get just the new token (delta)
            new_text = event["cumulative_text"]
            delta = new_text[len(accumulated_text):]
            accumulated_text = new_text
            
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": delta},
                    "finish_reason": None
                }]
            }
            yield format_sse(chunk)
        
        elif event["type"] == "complete":
            # Check for tool calls
            tool_calls = None
            finish_reason = "stop"
            
            if request.tools:
                tool_calls = parse_tool_calls(accumulated_text)
                if tool_calls:
                    finish_reason = "tool_calls"
            
            # Send final chunk
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }]
            }
            yield format_sse(final_chunk)
    
    yield "data: [DONE]\n\n"
    thread.join()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import os
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)
