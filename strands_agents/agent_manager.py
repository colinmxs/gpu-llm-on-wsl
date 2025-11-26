"""
Agent Manager - Backend logic for Strands SDK agent creation, management, and execution.

This module provides a clean interface for managing Strands SDK agents,
handling agent configuration, persistence, and execution independently of UI concerns.

Fully Strands-native implementation using Agent class, custom Model adapter,
and built-in conversation management. Agent serialization is handled manually
as the Agent class doesn't provide to_dict/from_dict methods.
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Generator, List, Tuple
import sys

from strands.agent import Agent
from strands.types.content import Messages

# Import from shared
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from model_manager import ModelManager

# Import Strands model adapter from local module
from .huggingface_local_model import HuggingFaceLocalModel


class AgentManager:
    """
    Manages Strands SDK agents including creation, loading, saving, and execution.
    
    Fully Strands-native implementation:
    - Uses Agent class properly with custom HuggingFaceLocalModel
    - Uses custom JSON serialization for persistence (Agent lacks to_dict/from_dict)
    - Uses Agent.stream_async() for chat with native tool execution
    - Maintains minimal adapter code between model_manager and Strands
    """
    def __init__(self, agents_dir: Path, model_manager: ModelManager, tools_dir: Optional[Path] = None):
        """
        Initialize the AgentManager.
        
        Args:
            agents_dir: Directory where agent configurations are stored
            model_manager: ModelManager instance for LLM inference
            tools_dir: Directory where tools are stored (for Strands to load)
        """
        self.agents_dir = Path(agents_dir)
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.model_manager = model_manager
        self.tools_dir = Path(tools_dir) if tools_dir else None
        
        # Cache of active Strands Agent instances
        self._agents: Dict[str, Agent] = {}
        
        # Store model configuration separately (not part of Strands Agent)
        self._agent_model_configs: Dict[str, Dict[str, Any]] = {}
        
        # Auto-load all saved agents at startup
        self._load_all_agents()
    
    def _load_all_agents(self):
        """Load all saved agent configurations from the agents directory."""
        if not self.agents_dir.exists():
            return
        
        for agent_file in self.agents_dir.glob("*.json"):
            try:
                with open(agent_file, 'r') as f:
                    saved_data = json.load(f)
                
                # Extract model config and agent data
                model_config = saved_data.pop("_model_config", {})
                
                # Recreate model provider with saved config
                model_provider = self._create_model_provider(model_config)
                
                # Prepare tools list
                tools_list = self._prepare_tools_list(model_config.get("tools", []))
                
                # Reconstruct Agent manually (Agent class doesn't have from_dict)
                agent = Agent(
                    name=saved_data.get("name", agent_file.stem),
                    description=saved_data.get("description", ""),
                    system_prompt=saved_data.get("system_prompt", ""),
                    model=model_provider,
                    tools=tools_list
                )
                
                agent_name = agent.name
                self._agents[agent_name] = agent
                self._agent_model_configs[agent_name] = model_config
                
            except Exception as e:
                print(f"Warning: Failed to load agent {agent_file.name}: {str(e)}")
    
    def _prepare_tools_list(self, tools: List[str]) -> Optional[List[str]]:
        """
        Convert tool names to file paths for Strands.
        
        Args:
            tools: List of tool names or paths
            
        Returns:
            List of tool paths or None
        """
        if not tools or not self.tools_dir:
            return None
        
        tools_list = []
        for tool_name in tools:
            # Handle both tool names and full paths
            if Path(tool_name).exists():
                tools_list.append(str(tool_name))
            else:
                tool_path = self.tools_dir / f"{tool_name}.py"
                if tool_path.exists():
                    tools_list.append(str(tool_path))
        
        return tools_list if tools_list else None
    
    def _create_model_provider(self, model_config: Dict[str, Any]) -> HuggingFaceLocalModel:
        """
        Create HuggingFaceLocalModel provider with configuration.
        
        Args:
            model_config: Dictionary with model parameters
            
        Returns:
            HuggingFaceLocalModel instance
        """
        return HuggingFaceLocalModel(
            model_manager=self.model_manager,
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_tokens", 500),
            top_p=model_config.get("top_p", 0.9),
            top_k=model_config.get("top_k", 50),
            repetition_penalty=model_config.get("repetition_penalty", 1.1)
        )
    
    def create_agent(
        self,
        name: str,
        description: str,
        system_prompt: str,
        model_name: str,
        tools: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> Dict[str, Any]:
        """
        Create a new Strands Agent with custom model provider.
        
        Args:
            name: Agent name
            description: Agent description
            system_prompt: System prompt for the agent
            model_name: Name of HuggingFace model to use
            tools: List of tool names/paths
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty factor
            
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - agent_name: Optional[str]
                - error: Optional[str]
        """
        try:
            if not name:
                return {"success": False, "error": "Agent name is required"}
            
            if not model_name:
                return {"success": False, "error": "Model name is required"}
            
            # Store model configuration and tools
            model_config = {
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "tools": tools or []  # Store original tool list
            }
            
            # Create model provider
            model_provider = self._create_model_provider(model_config)
            
            # Prepare tool list for Strands
            tools_list = self._prepare_tools_list(tools or [])
            
            # Create Strands Agent with model provider
            agent = Agent(
                name=name,
                description=description,
                system_prompt=system_prompt,
                model=model_provider,
                tools=tools_list
            )
            
            # Store agent and model config
            self._agents[name] = agent
            self._agent_model_configs[name] = model_config
            
            # Auto-save to disk
            save_result = self.save_agent(name)
            if not save_result["success"]:
                return save_result
            
            return {
                "success": True,
                "message": f"Agent '{name}' created and saved",
                "agent_name": name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create agent: {str(e)}"
            }
    
    def save_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Save an agent configuration to disk as JSON.
        
        Args:
            agent_name: Name of the agent to save
            
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - filepath: Optional[str]
                - error: Optional[str]
        """
        try:
            if agent_name not in self._agents:
                return {"success": False, "error": f"Agent '{agent_name}' not found"}
            
            agent = self._agents[agent_name]
            model_config = self._agent_model_configs.get(agent_name, {})
            
            # Manually serialize agent (Agent class doesn't have to_dict)
            agent_dict = {
                "name": agent.name,
                "description": agent.description,
                "system_prompt": agent.system_prompt,
                "_model_config": model_config
            }
            
            filepath = self.agents_dir / f"{agent_name}.json"
            with open(filepath, 'w') as f:
                json.dump(agent_dict, f, indent=2)
            
            return {
                "success": True,
                "message": f"Agent '{agent_name}' saved successfully",
                "filepath": str(filepath)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save agent: {str(e)}"
            }
    
    def load_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Load an agent configuration from disk as JSON.
        
        Args:
            agent_name: Name of the agent to load
            
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - error: Optional[str]
        """
        try:
            filepath = self.agents_dir / f"{agent_name}.json"
            
            if not filepath.exists():
                return {"success": False, "error": f"Agent file not found: {filepath}"}
            
            with open(filepath, 'r') as f:
                saved_data = json.load(f)
            
            # Extract model config and agent data
            model_config = saved_data.pop("_model_config", {})
            
            # Recreate model provider with saved config
            model_provider = self._create_model_provider(model_config)
            
            # Prepare tools list
            tools_list = self._prepare_tools_list(model_config.get("tools", []))
            
            # Reconstruct Agent manually (Agent class doesn't have from_dict)
            agent = Agent(
                name=saved_data.get("name", agent_name),
                description=saved_data.get("description", ""),
                system_prompt=saved_data.get("system_prompt", ""),
                model=model_provider,
                tools=tools_list
            )
            
            # Store agent and config
            self._agents[agent_name] = agent
            self._agent_model_configs[agent_name] = model_config
            
            return {
                "success": True,
                "message": f"Agent '{agent_name}' loaded successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load agent: {str(e)}"
            }
    
    def list_saved_agents(self) -> List[str]:
        """
        Get a list of saved agent configurations.
        
        Returns:
            List of agent names
        """
        if not self.agents_dir.exists():
            return []
        
        return sorted([f.stem for f in self.agents_dir.glob("*.json")])
    
    def list_active_agents(self) -> List[str]:
        """
        Get a list of currently active (in-memory) agents.
        
        Returns:
            List of agent names
        """
        return sorted(list(self._agents.keys()))
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """
        Get the Strands Agent instance.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Strands Agent instance if found, None otherwise
        """
        return self._agents.get(agent_name)
    
    def delete_agent(self, agent_name: str, delete_file: bool = False) -> Dict[str, Any]:
        """
        Delete an agent from active memory and optionally from disk.
        
        Args:
            agent_name: Name of the agent to delete
            delete_file: Whether to also delete the saved file
            
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - error: Optional[str]
        """
        try:
            messages = []
            
            # Remove from active agents
            if agent_name in self._agents:
                del self._agents[agent_name]
                messages.append(f"Removed '{agent_name}' from active agents")
            else:
                messages.append(f"Agent '{agent_name}' was not in active memory")
            
            # Remove model config
            if agent_name in self._agent_model_configs:
                del self._agent_model_configs[agent_name]
            
            # Delete file if requested
            if delete_file:
                filepath = self.agents_dir / f"{agent_name}.json"
                if filepath.exists():
                    filepath.unlink()
                    messages.append(f"Deleted agent file: {filepath}")
                else:
                    messages.append("No saved file found to delete")
            
            return {
                "success": True,
                "message": "\n".join(messages)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete agent: {str(e)}"
            }
    
    async def chat_with_agent_async(
        self,
        agent_name: str,
        message: str,
        history: Optional[List[Tuple[str, str]]] = None
    ):
        """
        Chat with agent using native Strands Agent.stream_async().
        
        This is the proper Strands-native implementation that uses:
        - Agent's built-in conversation management
        - Agent's tool execution
        - Agent's streaming API
        
        Args:
            agent_name: Name of the agent to chat with
            message: User message
            history: Chat history as list of (user, assistant) tuples
            
        Yields:
            StreamEvent dictionaries from Strands SDK
        """
        if agent_name not in self._agents:
            yield {"type": "error", "error": f"Agent '{agent_name}' not found"}
            return
        
        agent = self._agents[agent_name]
        model_config = self._agent_model_configs.get(agent_name, {})
        
        # Check if the correct model is loaded
        current_model = self.model_manager.get_current_model_name()
        required_model = model_config.get("model_name")
        
        if current_model != required_model:
            yield {
                "type": "error",
                "error": f"Agent requires model '{required_model}' but '{current_model}' is loaded. Please load the correct model first."
            }
            return
        
        if not self.model_manager.is_model_loaded():
            yield {"type": "error", "error": "No model loaded. Please load the agent's model first."}
            return
        
        # Build Strands Messages from history
        messages: Messages = []
        
        if history:
            for user_msg, assistant_msg in history:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": user_msg}]
                })
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_msg}]
                })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": message}]
        })
        
        # Use Strands Agent streaming
        try:
            async for event in agent.stream_async(messages):
                yield event
        except Exception as e:
            yield {"type": "error", "error": f"Agent streaming failed: {str(e)}"}
    
    def chat_with_agent(
        self,
        agent_name: str,
        message: str,
        history: Optional[List[Tuple[str, str]]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Synchronous wrapper for chat_with_agent_async.
        
        Args:
            agent_name: Name of the agent to chat with
            message: User message
            history: Chat history as list of (user, assistant) tuples
            
        Yields:
            StreamEvent dictionaries from Strands SDK
        """
        # Run async generator in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            async_gen = self.chat_with_agent_async(agent_name, message, history)
            
            while True:
                try:
                    event = loop.run_until_complete(async_gen.__anext__())
                    yield event
                except StopAsyncIteration:
                    break
                    
        finally:
            loop.close()
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """
        Get detailed information about an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dictionary containing agent configuration and status
        """
        if agent_name not in self._agents:
            return {"exists": False, "error": f"Agent '{agent_name}' not found"}
        
        agent = self._agents[agent_name]
        model_config = self._agent_model_configs.get(agent_name, {})
        is_saved = (self.agents_dir / f"{agent_name}.json").exists()
        
        return {
            "exists": True,
            "name": agent.name,
            "description": agent.description,
            "system_prompt": agent.system_prompt,
            "model_name": model_config.get("model_name", "unknown"),
            "temperature": model_config.get("temperature", 0.7),
            "max_tokens": model_config.get("max_tokens", 500),
            "top_p": model_config.get("top_p", 0.9),
            "top_k": model_config.get("top_k", 50),
            "repetition_penalty": model_config.get("repetition_penalty", 1.1),
            "tools": agent.tool_names if hasattr(agent, 'tool_names') else [],
            "is_saved": is_saved,
            "model_loaded": self.model_manager.get_current_model_name() == model_config.get("model_name"),
            "has_model_provider": agent.model is not None
        }
