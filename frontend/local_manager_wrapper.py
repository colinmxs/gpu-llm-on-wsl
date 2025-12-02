"""
Local Manager Wrapper - Uses API for model management, local imports for agents/tools

This module provides a unified interface that:
- Uses API client for model management operations
- Uses local strands_agents imports for agent and tool management
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add strands_agents to path
sys.path.insert(0, str(Path(__file__).parent.parent / "strands_agents"))

from agent_manager import AgentManager
from tool_manager import ToolManager, ToolConfig
from api_model_client import APIModelClient


class LocalManagerWrapper:
    """
    Wrapper that combines API model management with local agent/tool management.
    
    This provides a single interface for the frontend to interact with both
    the API (for models) and local code (for agents/tools).
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000",
                 agents_dir: Optional[Path] = None,
                 tools_dir: Optional[Path] = None):
        """
        Initialize the wrapper.
        
        Args:
            api_base_url: Base URL for the model API
            agents_dir: Directory for agent configurations
            tools_dir: Directory for tool configurations
        """
        # API client for model operations
        self.api_client = APIModelClient(api_base_url)
        
        # Set up directories
        if agents_dir is None:
            agents_dir = Path(__file__).parent.parent / "agents"
        if tools_dir is None:
            tools_dir = Path(__file__).parent.parent / "tools"
        
        self.agents_dir = Path(agents_dir)
        self.tools_dir = Path(tools_dir)
        
        # Create directories if they don't exist
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize local managers
        # Pass API base URL to agent_manager so it can use API for model operations
        self.agent_manager = AgentManager(
            agents_dir=self.agents_dir,
            api_base_url=api_base_url,
            tools_dir=self.tools_dir
        )
        self.tool_manager = ToolManager(tools_dir=self.tools_dir)
    
    # ============ Model Management Methods (proxied to API) ============
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return self.api_client.list_models()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        return self.api_client.get_model_info(model_name)
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU memory statistics"""
        return self.api_client.get_gpu_stats()
    
    def load_model(self, model_name: str, quantization: str = "4-bit (NF4)") -> Dict[str, Any]:
        """Load a model with specified quantization"""
        return self.api_client.load_model(model_name, quantization)
    
    def unload_model(self) -> Dict[str, Any]:
        """Unload the currently loaded model"""
        return self.api_client.unload_model()
    
    def get_current_model(self) -> Dict[str, str]:
        """Get the name of the currently loaded model"""
        return self.api_client.get_current_model()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return self.api_client.get_status()
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Pass-through for any direct API requests"""
        return self.api_client._request(method, endpoint, **kwargs)
    
    # ============ Agent Management Methods (local) ============
    
    def create_agent(
        self,
        name: str,
        system_prompt: str,
        model_name: str,
        description: str = "",
        tools: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> Dict[str, Any]:
        """Create a new agent"""
        return self.agent_manager.create_agent(
            name=name,
            description=description,
            system_prompt=system_prompt,
            model_name=model_name,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
    
    def list_agents(self) -> List[str]:
        """List all active agents"""
        return self.agent_manager.list_active_agents()
    
    def list_saved_agents(self) -> List[str]:
        """List all saved agent configurations"""
        return self.agent_manager.list_saved_agents()
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific agent"""
        return self.agent_manager.get_agent_info(agent_name)
    
    def load_agent(self, agent_name: str) -> Dict[str, Any]:
        """Load an agent from disk"""
        return self.agent_manager.load_agent(agent_name)
    
    def save_agent(self, agent_name: str) -> Dict[str, Any]:
        """Save an agent to disk"""
        return self.agent_manager.save_agent(agent_name)
    
    def delete_agent(self, agent_name: str, delete_file: bool = False) -> Dict[str, Any]:
        """Delete an agent"""
        return self.agent_manager.delete_agent(agent_name, delete_file=delete_file)
    
    async def chat_with_agent_async(self, agent_name: str, message: str, history=None):
        """Chat with an agent (async streaming)"""
        async for event in self.agent_manager.chat_with_agent_async(
            agent_name=agent_name,
            message=message,
            history=history
        ):
            yield event
    
    def chat_with_agent_stream(self, agent_name: str, message: str, history=None):
        """Chat with an agent (sync streaming generator for Gradio)"""
        import asyncio
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Create async generator
        async_gen = self.agent_manager.chat_with_agent_async(
            agent_name=agent_name,
            message=message,
            history=history
        )
        
        # Yield from async generator synchronously
        while True:
            try:
                event = loop.run_until_complete(async_gen.__anext__())
                yield event
            except StopAsyncIteration:
                break
    
    # ============ Tool Management Methods (local) ============
    
    def create_tool(
        self,
        name: str,
        description: str,
        function_code: str,
        parameters_schema: Optional[Dict[str, Any]] = None,
        returns_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new tool"""
        config = ToolConfig(
            name=name,
            description=description,
            function_code=function_code,
            parameters_schema=parameters_schema or {"type": "object", "properties": {}},
            returns_schema=returns_schema or {"type": "string"}
        )
        return self.tool_manager.create_tool(config)
    
    def list_tools(self) -> List[str]:
        """List all active tools"""
        return self.tool_manager.list_active_tools()
    
    def list_saved_tools(self) -> List[str]:
        """List all saved tool configurations"""
        return self.tool_manager.list_saved_tools()
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific tool"""
        return self.tool_manager.get_tool_info(tool_name)
    
    def load_tool(self, tool_name: str) -> Dict[str, Any]:
        """Load a tool from disk"""
        return self.tool_manager.load_tool(tool_name)
    
    def save_tool(self, tool_name: str) -> Dict[str, Any]:
        """Save a tool to disk"""
        return self.tool_manager.save_tool(tool_name)
    
    def delete_tool(self, tool_name: str, delete_file: bool = False) -> Dict[str, Any]:
        """Delete a tool"""
        return self.tool_manager.delete_tool(tool_name, delete_file=delete_file)
