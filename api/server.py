"""
FastAPI Server for LLM Inference Streaming

This API provides endpoints for managing and streaming inference from LLMs
running in a Docker container, allowing local development against remote GPU resources.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import json
import sys
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Add shared module and strands_agents to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "strands_agents"))

from model_manager import ModelManager
from agent_manager import AgentManager
from tool_manager import ToolManager, ToolConfig

# Initialize FastAPI app
app = FastAPI(
    title="LLM Inference API",
    description="Stream LLM inferences from Docker container to host",
    version="1.0.0"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ModelManager
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/app/cache"))
AGENTS_DIR = Path(os.getenv("AGENTS_DIR", str(Path(__file__).parent.parent / "agents")))
TOOLS_DIR = Path(os.getenv("TOOLS_DIR", str(Path(__file__).parent.parent / "tools")))

model_manager = ModelManager(models_dir=MODELS_DIR, cache_dir=CACHE_DIR)
agent_manager = AgentManager(agents_dir=AGENTS_DIR, model_manager=model_manager, tools_dir=TOOLS_DIR)
tool_manager = ToolManager(tools_dir=TOOLS_DIR)


# Pydantic models for request/response validation
class LoadModelRequest(BaseModel):
    model_name: str = Field(..., description="Model name in 'org/model' format")
    quantization: str = Field(
        default="4-bit (NF4)",
        description="Quantization type: '4-bit (NF4)', '8-bit', or 'Full Precision (FP16)'"
    )


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(default=200, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(default=50, ge=0, le=100, description="Top-k sampling parameter")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    skip_prompt: bool = Field(default=True, description="Skip prompt in output")


class ModelInfo(BaseModel):
    exists: bool
    name: Optional[str] = None
    path: Optional[str] = None
    file_count: Optional[int] = None
    total_size_bytes: Optional[int] = None
    total_size_human: Optional[str] = None
    has_config: Optional[bool] = None
    has_tokenizer: Optional[bool] = None
    safetensors_count: Optional[int] = None
    bin_count: Optional[int] = None
    error: Optional[str] = None


class GPUStats(BaseModel):
    available: bool
    name: Optional[str] = None
    total_gb: Optional[float] = None
    allocated_gb: Optional[float] = None
    reserved_gb: Optional[float] = None
    free_gb: Optional[float] = None
    usage_percent: Optional[float] = None
    error: Optional[str] = None


class LoadModelResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    model_name: Optional[str] = None
    quantization: Optional[str] = None
    device_map: Optional[Dict[str, Any]] = None
    previous_model: Optional[str] = None
    error: Optional[str] = None


class UnloadModelResponse(BaseModel):
    success: bool
    message: str
    previous_model: Optional[str] = None


class StatusResponse(BaseModel):
    model_loaded: bool
    current_model: Optional[str] = None
    available_models: List[str]
    gpu_stats: GPUStats


# Agent Manager Request/Response Models
class CreateAgentRequest(BaseModel):
    name: str = Field(..., description="Agent name")
    description: str = Field(default="", description="Agent description")
    system_prompt: str = Field(..., description="System prompt for the agent")
    model_name: str = Field(..., description="HuggingFace model name")
    tools: Optional[List[str]] = Field(default=None, description="List of tool names")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=100)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class AgentChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    history: Optional[List[List[str]]] = Field(
        default=None, 
        description="Chat history as list of [user_msg, assistant_msg] pairs"
    )


class AgentResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    agent_name: Optional[str] = None
    error: Optional[str] = None


class AgentInfoResponse(BaseModel):
    exists: bool
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    tools: Optional[List[str]] = None
    is_saved: Optional[bool] = None
    model_loaded: Optional[bool] = None
    has_model_provider: Optional[bool] = None
    error: Optional[str] = None


# Tool Manager Request/Response Models
class CreateToolRequest(BaseModel):
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    function_code: str = Field(..., description="Python function code with @strands.tool decorator")
    parameters_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema for parameters"
    )
    returns_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema for return value"
    )


class ToolResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    tool_name: Optional[str] = None
    filepath: Optional[str] = None
    python_file: Optional[str] = None
    error: Optional[str] = None


class ToolInfoResponse(BaseModel):
    exists: bool
    name: Optional[str] = None
    description: Optional[str] = None
    function_code: Optional[str] = None
    parameters_schema: Optional[Dict[str, Any]] = None
    returns_schema: Optional[Dict[str, Any]] = None
    is_saved: Optional[bool] = None
    has_python_file: Optional[bool] = None
    strands_compatible: Optional[bool] = None
    python_filepath: Optional[str] = None
    error: Optional[str] = None


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "LLM Inference API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    gpu_stats = model_manager.get_gpu_stats()
    return {
        "status": "healthy",
        "cuda_available": gpu_stats["available"],
        "model_loaded": model_manager.is_model_loaded()
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status including loaded model and GPU stats"""
    return {
        "model_loaded": model_manager.is_model_loaded(),
        "current_model": model_manager.get_current_model_name(),
        "available_models": model_manager.list_models(),
        "gpu_stats": model_manager.get_gpu_stats()
    }


@app.get("/models", response_model=List[str])
async def list_models():
    """List all available models"""
    return model_manager.list_models()


@app.get("/models/{model_name:path}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    info = model_manager.get_model_info(model_name)
    return info


@app.get("/gpu", response_model=GPUStats)
async def get_gpu_stats():
    """Get current GPU memory statistics"""
    return model_manager.get_gpu_stats()


@app.post("/model/load", response_model=LoadModelResponse)
async def load_model(request: LoadModelRequest):
    """Load a model with specified quantization"""
    result = model_manager.load_model(
        model_name=request.model_name,
        quantization=request.quantization
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to load model"))
    
    return result


@app.post("/model/unload", response_model=UnloadModelResponse)
async def unload_model():
    """Unload the currently loaded model"""
    result = model_manager.unload_model()
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@app.get("/model/current")
async def get_current_model():
    """Get the name of the currently loaded model"""
    model_name = model_manager.get_current_model_name()
    if not model_name:
        raise HTTPException(status_code=404, detail="No model currently loaded")
    
    return {"model_name": model_name}


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Generate text from a prompt with Server-Sent Events (SSE) streaming.
    
    Each event is a JSON object with:
    - type: "token" | "complete" | "error"
    - text: generated text
    - Additional metadata in completion event
    """
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")
    
    async def event_generator():
        """Generate SSE events from model output"""
        for event in model_manager.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            skip_prompt=request.skip_prompt
        ):
            # Format as SSE
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt (non-streaming).
    Returns the complete generated text.
    """
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")
    
    generated_text = ""
    metadata = {}
    
    for event in model_manager.generate(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        skip_prompt=request.skip_prompt
    ):
        if event["type"] == "error":
            raise HTTPException(status_code=500, detail=event["error"])
        elif event["type"] == "complete":
            generated_text = event["text"]
            metadata = {
                "total_tokens": event["total_tokens"],
                "tokens_per_second": event["tokens_per_second"],
                "elapsed_seconds": event["elapsed_seconds"]
            }
    
    return {
        "text": generated_text,
        "metadata": metadata
    }


# Agent Manager Endpoints

@app.post("/agents", response_model=AgentResponse)
async def create_agent(request: CreateAgentRequest):
    """Create a new agent"""
    result = agent_manager.create_agent(
        name=request.name,
        description=request.description,
        system_prompt=request.system_prompt,
        model_name=request.model_name,
        tools=request.tools,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to create agent"))
    
    return result


@app.get("/agents", response_model=List[str])
async def list_agents():
    """List all active agents"""
    return agent_manager.list_active_agents()


@app.get("/agents/saved", response_model=List[str])
async def list_saved_agents():
    """List all saved agent configurations"""
    return agent_manager.list_saved_agents()


@app.get("/agents/{agent_name}", response_model=AgentInfoResponse)
async def get_agent_info(agent_name: str):
    """Get detailed information about a specific agent"""
    return agent_manager.get_agent_info(agent_name)


@app.post("/agents/{agent_name}/load", response_model=AgentResponse)
async def load_agent(agent_name: str):
    """Load an agent from disk"""
    result = agent_manager.load_agent(agent_name)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("error", "Failed to load agent"))
    
    return result


@app.post("/agents/{agent_name}/save", response_model=AgentResponse)
async def save_agent(agent_name: str):
    """Save an agent to disk"""
    result = agent_manager.save_agent(agent_name)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to save agent"))
    
    return result


@app.delete("/agents/{agent_name}", response_model=AgentResponse)
async def delete_agent(agent_name: str, delete_file: bool = False):
    """Delete an agent from memory and optionally from disk"""
    result = agent_manager.delete_agent(agent_name, delete_file=delete_file)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to delete agent"))
    
    return result


@app.post("/agents/{agent_name}/chat/stream")
async def chat_with_agent_stream(agent_name: str, request: AgentChatRequest):
    """
    Chat with an agent with Server-Sent Events (SSE) streaming.
    
    Each event is a JSON object with Strands SDK StreamEvent format.
    """
    # Convert history format from [[user, assistant], ...] to [(user, assistant), ...]
    history = None
    if request.history:
        history = [tuple(pair) for pair in request.history]
    
    async def event_generator():
        """Generate SSE events from agent chat"""
        async for event in agent_manager.chat_with_agent_async(
            agent_name=agent_name,
            message=request.message,
            history=history
        ):
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/agents/{agent_name}/chat")
async def chat_with_agent(agent_name: str, request: AgentChatRequest):
    """
    Chat with an agent (non-streaming).
    Returns the complete response.
    """
    # Convert history format
    history = None
    if request.history:
        history = [tuple(pair) for pair in request.history]
    
    response_text = ""
    metadata = {}
    
    async for event in agent_manager.chat_with_agent_async(
        agent_name=agent_name,
        message=request.message,
        history=history
    ):
        logger.info(f"Chat event received: {json.dumps(event, indent=2, default=str)}")
        
        if event.get("type") == "error":
            raise HTTPException(status_code=500, detail=event.get("error", "Chat failed"))
        elif event.get("type") == "text":
            response_text += event.get("text", "")
        elif event.get("type") == "complete":
            metadata = event
    
    return {
        "text": response_text,
        "metadata": metadata
    }


# Tool Manager Endpoints

@app.post("/tools", response_model=ToolResponse)
async def create_tool(request: CreateToolRequest):
    """Create a new tool"""
    config = ToolConfig(
        name=request.name,
        description=request.description,
        function_code=request.function_code,
        parameters_schema=request.parameters_schema or {"type": "object", "properties": {}},
        returns_schema=request.returns_schema or {"type": "string"}
    )
    
    result = tool_manager.create_tool(config)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to create tool"))
    
    return result


@app.get("/tools", response_model=List[str])
async def list_tools():
    """List all active tools"""
    return tool_manager.list_active_tools()


@app.get("/tools/saved", response_model=List[str])
async def list_saved_tools():
    """List all saved tool configurations"""
    return tool_manager.list_saved_tools()


@app.get("/tools/{tool_name}", response_model=ToolInfoResponse)
async def get_tool_info(tool_name: str):
    """Get detailed information about a specific tool"""
    return tool_manager.get_tool_info(tool_name)


@app.post("/tools/{tool_name}/load", response_model=ToolResponse)
async def load_tool(tool_name: str):
    """Load a tool from disk"""
    result = tool_manager.load_tool(tool_name)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("error", "Failed to load tool"))
    
    return result


@app.post("/tools/{tool_name}/save", response_model=ToolResponse)
async def save_tool(tool_name: str):
    """Save a tool to disk"""
    result = tool_manager.save_tool(tool_name)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to save tool"))
    
    return result


@app.delete("/tools/{tool_name}", response_model=ToolResponse)
async def delete_tool(tool_name: str, delete_file: bool = False):
    """Delete a tool from memory and optionally from disk"""
    result = tool_manager.delete_tool(tool_name, delete_file=delete_file)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to delete tool"))
    
    return result


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    print(f"Starting LLM Inference API on {host}:{port}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Agents directory: {AGENTS_DIR}")
    print(f"Tools directory: {TOOLS_DIR}")
    print(f"Active agents: {len(agent_manager.list_active_agents())}")
    print(f"Active tools: {len(tool_manager.list_active_tools())}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
