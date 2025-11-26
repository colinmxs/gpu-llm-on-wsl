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

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from model_manager import ModelManager

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
model_manager = ModelManager(models_dir=MODELS_DIR, cache_dir=CACHE_DIR)


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


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    print(f"Starting LLM Inference API on {host}:{port}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
