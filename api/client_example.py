"""
Example Python client for the LLM Inference API

This demonstrates how to interact with the FastAPI server from your local machine
while the LLMs run in the Docker container.
"""

import requests
import json
from typing import Generator, Dict, Any


class LLMClient:
    """Client for interacting with the LLM Inference API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the LLM API client.
        
        Args:
            base_url: Base URL of the API server (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status including loaded model and GPU stats"""
        response = requests.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> list:
        """List all available models"""
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        response = requests.get(f"{self.base_url}/models/{model_name}")
        response.raise_for_status()
        return response.json()
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU memory statistics"""
        response = requests.get(f"{self.base_url}/gpu")
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_name: str, quantization: str = "4-bit (NF4)") -> Dict[str, Any]:
        """
        Load a model with specified quantization.
        
        Args:
            model_name: Model name in 'org/model' format
            quantization: One of "4-bit (NF4)", "8-bit", or "Full Precision (FP16)"
        """
        response = requests.post(
            f"{self.base_url}/model/load",
            json={
                "model_name": model_name,
                "quantization": quantization
            }
        )
        response.raise_for_status()
        return response.json()
    
    def unload_model(self) -> Dict[str, Any]:
        """Unload the currently loaded model"""
        response = requests.post(f"{self.base_url}/model/unload")
        response.raise_for_status()
        return response.json()
    
    def get_current_model(self) -> Dict[str, str]:
        """Get the name of the currently loaded model"""
        response = requests.get(f"{self.base_url}/model/current")
        response.raise_for_status()
        return response.json()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        skip_prompt: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt (non-streaming).
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (1-4096)
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            top_k: Top-k sampling parameter (0-100)
            repetition_penalty: Repetition penalty (1.0-2.0)
            skip_prompt: Skip prompt in output
            
        Returns:
            Dictionary with 'text' and 'metadata' keys
        """
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "skip_prompt": skip_prompt
            }
        )
        response.raise_for_status()
        return response.json()
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        skip_prompt: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate text from a prompt with streaming (SSE).
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (1-4096)
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            top_k: Top-k sampling parameter (0-100)
            repetition_penalty: Repetition penalty (1.0-2.0)
            skip_prompt: Skip prompt in output
            
        Yields:
            Dictionary with event data (type: "token" | "complete" | "error")
        """
        response = requests.post(
            f"{self.base_url}/generate/stream",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "skip_prompt": skip_prompt
            },
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    event_data = json.loads(line[6:])
                    yield event_data


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = LLMClient("http://localhost:8000")
    
    # Check health
    print("Checking API health...")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"CUDA Available: {health['cuda_available']}")
    print()
    
    # Get available models
    print("Available models:")
    models = client.list_models()
    for model in models:
        print(f"  - {model}")
    print()
    
    # Load a model (change this to a model you have)
    if models:
        model_name = models[0]
        print(f"Loading model: {model_name}")
        result = client.load_model(model_name, quantization="4-bit (NF4)")
        print(f"Success: {result['success']}")
        print()
        
        # Get GPU stats
        print("GPU Statistics:")
        gpu = client.get_gpu_stats()
        if gpu['available']:
            print(f"  GPU: {gpu['name']}")
            print(f"  Memory: {gpu['reserved_gb']:.2f} GB / {gpu['total_gb']:.2f} GB")
            print(f"  Usage: {gpu['usage_percent']:.1f}%")
        print()
        
        # Generate text with streaming
        prompt = "Explain what a neural network is in simple terms."
        print(f"Prompt: {prompt}")
        print("Response: ", end="", flush=True)
        
        for event in client.generate_stream(prompt, max_tokens=150):
            if event['type'] == 'token':
                print(event['text'], end='', flush=True)
            elif event['type'] == 'complete':
                print("\n")
                print(f"Generated {event['total_tokens']} tokens")
                print(f"Speed: {event['tokens_per_second']:.2f} tokens/sec")
                print(f"Time: {event['elapsed_seconds']:.2f}s")
            elif event['type'] == 'error':
                print(f"\nError: {event['error']}")
        
        print()
        
        # Unload model
        print("Unloading model...")
        result = client.unload_model()
        print(f"Success: {result['success']}")
    else:
        print("No models available. Please download models first.")
