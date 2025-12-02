"""
API Model Client - Simple client for model management operations via API

This client only handles model management operations through the API.
Agent and tool management is handled locally via strands_agents imports.
"""

import requests
import json
from typing import Generator, Dict, Any, List, Optional


class APIModelClient:
    """Client for interacting with Model Management API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """
        Make a generic HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments passed to requests
            
        Returns:
            Parsed JSON response
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    # ============ Model Management Methods ============
    
    def list_models(self) -> List[str]:
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status including loaded model and GPU stats"""
        response = requests.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
