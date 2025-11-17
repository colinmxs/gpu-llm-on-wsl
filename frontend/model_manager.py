"""
Model Manager - Backend logic for LLM loading, unloading, and inference.

This module provides a clean, Gradio-agnostic interface for managing language models,
completely separating model lifecycle management from UI concerns.
"""

import gc
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Generator, List
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import humanize


@dataclass
class ModelState:
    """Encapsulates the current state of a loaded model."""
    model: Any
    tokenizer: Any
    name: str
    quantization: str
    device_map: Optional[Dict[str, Any]] = None


class ModelManager:
    """
    Manages the lifecycle of language models including loading, unloading, and generation.
    
    This class is completely independent of any UI framework and returns data dictionaries
    that can be formatted by any frontend (Gradio, CLI, FastAPI, etc.).
    """
    
    def __init__(self, models_dir: Path, cache_dir: Optional[Path] = None):
        """
        Initialize the ModelManager.
        
        Args:
            models_dir: Directory where models are stored
            cache_dir: Optional cache directory for Hugging Face
        """
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.state: Optional[ModelState] = None
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.state is not None
    
    def get_current_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded model, or None."""
        return self.state.name if self.state else None
    
    def list_models(self) -> List[str]:
        """
        Get a list of available models.
        
        Returns:
            List of model names in "org/model" format
        """
        if not self.models_dir.exists():
            return []
        
        model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
        return sorted([d.name.replace("--", "/", 1) for d in model_dirs])
    
    def get_model_path(self, model_name: str) -> Path:
        """
        Convert model name to filesystem path.
        
        Args:
            model_name: Model name in "org/model" format
            
        Returns:
            Path to the model directory
        """
        safe_name = model_name.replace("/", "--")
        return self.models_dir / safe_name
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model to inspect
            
        Returns:
            Dictionary containing model information:
                - exists: bool
                - name: str
                - path: str
                - file_count: int
                - total_size_bytes: int
                - has_config: bool
                - has_tokenizer: bool
                - safetensors_count: int
                - bin_count: int
                - error: Optional[str]
        """
        if not model_name:
            return {"exists": False, "error": "No model name provided"}
        
        model_path = self.get_model_path(model_name)
        
        if not model_path.exists():
            return {
                "exists": False,
                "name": model_name,
                "path": str(model_path),
                "error": "Model path not found"
            }
        
        try:
            # Count files and calculate size
            file_count = sum(1 for _ in model_path.rglob("*") if _.is_file())
            total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
            
            # Check for specific files
            has_config = (model_path / "config.json").exists()
            has_tokenizer = (model_path / "tokenizer_config.json").exists()
            
            # Count model files
            safetensors_count = len(list(model_path.glob("*.safetensors")))
            bin_count = len(list(model_path.glob("*.bin")))
            
            return {
                "exists": True,
                "name": model_name,
                "path": str(model_path),
                "file_count": file_count,
                "total_size_bytes": total_size,
                "total_size_human": humanize.naturalsize(total_size, binary=True),
                "has_config": has_config,
                "has_tokenizer": has_tokenizer,
                "safetensors_count": safetensors_count,
                "bin_count": bin_count
            }
        except Exception as e:
            return {
                "exists": True,
                "name": model_name,
                "path": str(model_path),
                "error": f"Error reading model info: {str(e)}"
            }
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Get current GPU memory statistics.
        
        Returns:
            Dictionary containing:
                - available: bool
                - name: Optional[str]
                - total_gb: Optional[float]
                - allocated_gb: Optional[float]
                - reserved_gb: Optional[float]
                - free_gb: Optional[float]
                - usage_percent: Optional[float]
                - error: Optional[str]
        """
        if not torch.cuda.is_available():
            return {"available": False, "error": "CUDA not available"}
        
        try:
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / 1024**3
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
            free_gb = total_gb - reserved_gb
            usage_percent = (reserved_gb / total_gb) * 100
            
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "total_gb": total_gb,
                "allocated_gb": allocated_gb,
                "reserved_gb": reserved_gb,
                "free_gb": free_gb,
                "usage_percent": usage_percent
            }
        except Exception as e:
            return {
                "available": True,
                "error": f"Error reading GPU stats: {str(e)}"
            }
    
    def unload_model(self) -> Dict[str, Any]:
        """
        Unload the current model and free GPU memory.
        
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - previous_model: Optional[str]
        """
        if not self.state:
            return {
                "success": False,
                "message": "No model currently loaded"
            }
        
        previous_model = self.state.name
        
        # Clean up
        del self.state.model
        del self.state.tokenizer
        self.state = None
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "success": True,
            "message": "Model unloaded successfully",
            "previous_model": previous_model
        }
    
    def load_model(self, model_name: str, quantization: str = "4-bit (NF4)") -> Dict[str, Any]:
        """
        Load a model with specified quantization.
        
        Args:
            model_name: Name of the model to load
            quantization: One of "4-bit (NF4)", "8-bit", or "Full Precision (FP16)"
            
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - model_name: Optional[str]
                - quantization: Optional[str]
                - device_map: Optional[dict]
                - previous_model: Optional[str]
                - error: Optional[str]
        """
        if not model_name:
            return {"success": False, "error": "No model name provided"}
        
        model_path = self.get_model_path(model_name)
        
        if not model_path.exists():
            return {
                "success": False,
                "error": f"Model not found: {model_path}"
            }
        
        # Unload existing model if any
        previous_model = None
        if self.state:
            previous_model = self.state.name
            self.unload_model()
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True
            )
            
            # Configure quantization and load model
            if quantization == "4-bit (NF4)":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    quantization_config=bnb_config,
                    device_map="auto",
                    local_files_only=True,
                    torch_dtype=torch.float16
                )
            elif quantization == "8-bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    quantization_config=bnb_config,
                    device_map="auto",
                    local_files_only=True
                )
            else:  # Full precision
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    device_map="auto",
                    local_files_only=True,
                    torch_dtype=torch.float16
                )
            
            # Store state
            device_map = model.hf_device_map if hasattr(model, 'hf_device_map') else None
            self.state = ModelState(
                model=model,
                tokenizer=tokenizer,
                name=model_name,
                quantization=quantization,
                device_map=device_map
            )
            
            return {
                "success": True,
                "message": "Model loaded successfully",
                "model_name": model_name,
                "quantization": quantization,
                "device_map": device_map,
                "previous_model": previous_model
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load model: {str(e)}",
                "model_name": model_name
            }
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        skip_prompt: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate text from a prompt with streaming output.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            skip_prompt: Whether to skip the prompt in the output
            
        Yields:
            Dictionary containing:
                - type: "token" | "complete" | "error"
                - text: str (the generated token or full text)
                - total_tokens: Optional[int] (only in "complete" type)
                - tokens_per_second: Optional[float] (only in "complete" type)
                - elapsed_seconds: Optional[float] (only in "complete" type)
                - error: Optional[str] (only in "error" type)
        """
        if not self.state:
            yield {"type": "error", "error": "No model loaded"}
            return
        
        if not prompt.strip():
            yield {"type": "error", "error": "Empty prompt"}
            return
        
        try:
            # Tokenize input
            inputs = self.state.tokenizer(prompt, return_tensors="pt").to(self.state.model.device)
            
            start_time = time.time()
            
            # Setup streaming
            streamer = TextIteratorStreamer(
                self.state.tokenizer,
                skip_prompt=skip_prompt,
                skip_special_tokens=True
            )
            
            # Generation parameters
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.state.tokenizer.eos_token_id,
                streamer=streamer
            )
            
            # Start generation in a separate thread
            thread = threading.Thread(
                target=self.state.model.generate,
                kwargs=generation_kwargs
            )
            thread.start()
            
            # Stream the output
            generated_text = ""
            num_tokens = 0
            
            for new_text in streamer:
                generated_text += new_text
                num_tokens += 1
                yield {
                    "type": "token",
                    "text": new_text,
                    "cumulative_text": generated_text
                }
            
            # Wait for thread to complete
            thread.join()
            
            elapsed = time.time() - start_time
            tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
            
            # Yield completion event
            yield {
                "type": "complete",
                "text": generated_text,
                "total_tokens": num_tokens,
                "tokens_per_second": tokens_per_sec,
                "elapsed_seconds": elapsed
            }
            
        except Exception as e:
            yield {"type": "error", "error": f"Generation failed: {str(e)}"}
