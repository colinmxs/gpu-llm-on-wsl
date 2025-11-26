"""
Agent Manager - Backend logic for Strands SDK agent creation, management, and execution.

This module provides a clean interface for managing Strands SDK agents,
handling agent configuration, persistence, and execution independently of UI concerns.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Generator, List, Tuple
from dataclasses import dataclass, asdict

from strands.agent import Agent
from strands.models import Model

from model_manager import ModelManager


@dataclass
class AgentConfig:
    """
    Configuration for a Strands SDK agent.
    This stores the parameters needed to initialize a Strands Agent.
    """
    name: str
    description: str
    system_prompt: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    tools: List[str] = None  # List of tool names/paths
    agent_id: Optional[str] = None  # Strands agent ID
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.agent_id is None:
            self.agent_id = self.name.lower().replace(" ", "_")


class AgentManager:
    """
    Manages Strands SDK agents including creation, loading, saving, and execution.
    
    This class leverages the Strands SDK Agent class instead of reimplementing agent logic.
    It maintains configurations for persistence and creates Strands Agent instances on demand.
    """
    
    def __init__(self, agents_dir: Path, model_manager: ModelManager, tools_dir: Optional[Path] = None):
        """
        Initialize the AgentManager.
        
        Args:
            agents_dir: Directory where agent configurations are stored
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
        
        # Store agent configurations
        self.agent_configs: Dict[str, AgentConfig] = {}
        
        # Cache of active Strands Agent instances
        self._agent_instances: Dict[str, Agent] = {}
        
        # Auto-load all saved agent configurations at startup
        self._load_all_agents()_file, 'r') as f:
                    config_dict = json.load(f)
                config = AgentConfig(**config_dict)
                self.agent_configs[config.name] = config
            except Exception as e:
                print(f"Warning: Failed to load agent {agent_file.name}: {str(e)}")
    
    def _get_or_create_agent(self, agent_name: str) -> Optional[Agent]:
        """
        Get or create a Strands Agent instance for the given configuration.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Strands Agent instance or None if config not found
        """
        if agent_name not in self.agent_configs:
            return None
        
        # Return cached instance if available
        if agent_name in self._agent_instances:
            return self._agent_instances[agent_name]
        
        config = self.agent_configs[agent_name]
        
        # Prepare tool list for Strands
        tools_list = None
        if config.tools and self.tools_dir:
            # Convert tool names to file paths
            tools_list = []
            for tool_name in config.tools:
                tool_path = self.tools_dir / f"{tool_name}.py"
                if tool_path.exists():
                    tools_list.append(str(tool_path))
        
        # Create Strands Agent instance
        try:
            agent = Agent(
                name=config.name,
                description=config.description,
                system_prompt=config.system_prompt,
                tools=tools_list,
                agent_id=config.agent_id,
            )
            
            self._agent_instances[agent_name] = agent
            return agent
            
        except Exception as e:
            print(f"Warning: Failed to create Strands Agent instance for '{agent_name}': {str(e)}")
            return None
    
    def create_agent(self, config: AgentConfig) -> Dict[str, Any]:
        """
        Create a new agent configuration and automatically save to disk.
        
        Args:
            config: Agent configuration
            
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - agent_name: Optional[str]
                - filepath: Optional[str]
                - error: Optional[str]
        """
        try:
            if not config.name:
                return {"success": False, "error": "Agent name is required"}
            
            if not config.model_name:
                return {"success": False, "error": "Model name is required"}
            
            # Store configuration
            self.agent_configs[config.name] = config
            
            # Clear any cached instance so it will be recreated with new config
            if config.name in self._agent_instances:
                del self._agent_instances[config.name]
            
            # Auto-save to disk
            filepath = self.agents_dir / f"{config.name}.json"
            with open(filepath, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
            return {
                "success": True,
                "message": f"Agent '{config.name}' created and saved",
                "agent_name": config.name,
                "filepath": str(filepath)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create agent: {str(e)}"
            }
    
    def save_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Save an agent configuration to disk.
        
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
            if agent_name not in self.agent_configs:
                return {"success": False, "error": f"Agent '{agent_name}' not found"}
            
            config = self.agent_configs[agent_name]
            filepath = self.agents_dir / f"{agent_name}.json"
            
            with open(filepath, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
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
        Load an agent configuration from disk.
        
        Args:
            agent_name: Name of the agent to load
            
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - config: Optional[AgentConfig]
                - error: Optional[str]
        """
        try:
            filepath = self.agents_dir / f"{agent_name}.json"
            
            if not filepath.exists():
                return {"success": False, "error": f"Agent file not found: {filepath}"}
            
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            config = AgentConfig(**config_dict)
            self.agent_configs[agent_name] = config
            
            # Clear cached instance so it will be recreated
            if agent_name in self._agent_instances:
                del self._agent_instances[agent_name]
            
            return {
                "success": True,
                "message": f"Agent '{agent_name}' loaded successfully",
                "config": config
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
        return sorted(list(self.agent_configs.keys()))
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """
        Get the configuration of an active agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            AgentConfig if found, None otherwise
        """
        return self.agent_configs.get(agent_name)
    
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
            if agent_name in self.agent_configs:
                del self.agent_configs[agent_name]
                messages.append(f"Removed '{agent_name}' from active agents")
            else:
                messages.append(f"Agent '{agent_name}' was not in active memory")
            
            # Clear cached instance
            if agent_name in self._agent_instances:
                # Cleanup Strands Agent if it has cleanup method
                agent = self._agent_instances[agent_name]
                if hasattr(agent, 'cleanup'):
                    agent.cleanup()
                del self._agent_instances[agent_name]
            
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
    
    def chat_with_agent(
        self,
        agent_name: str,
        message: str,
        history: List[Tuple[str, str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Chat with an agent using Strands SDK Agent class with streaming.
        
        Uses Strands Agent built-in conversation management and tool execution.
        
        Args:
            agent_name: Name of the agent to chat with
            message: User message
            history: Chat history as list of (user, assistant) tuples
            
        Yields:
            Dictionary containing:
                - type: "error" | "token" | "complete"
                - text: str
                - cumulative_text: Optional[str]
                - total_tokens: Optional[int]
                - tokens_per_second: Optional[float]
                - elapsed_seconds: Optional[float]
                - error: Optional[str]
        """
        if agent_name not in self.agent_configs:
            yield {"type": "error", "error": f"Agent '{agent_name}' not found"}
            return
        
        config = self.agent_configs[agent_name]
        
        # Check if the correct model is loaded
        current_model = self.model_manager.get_current_model_name()
        if current_model != config.model_name:
            yield {
                "type": "error",
                "error": f"Agent requires model '{config.model_name}' but '{current_model}' is loaded. Please load the correct model first."
            }
            return
        
        if not self.model_manager.is_model_loaded():
            yield {"type": "error", "error": "No model loaded. Please load the agent's model first."}
            return
        
        # Get or create Strands Agent instance
        agent = self._get_or_create_agent(agent_name)
        if agent is None:
            yield {"type": "error", "error": f"Failed to initialize agent '{agent_name}'"}
            return
        
        # Use model_manager for generation (Strands Agent will use custom model provider in future)
        # Build the prompt with history
        if history is None:
            history = []
        
        prompt = f"System: {config.system_prompt}\n\n"
        for user_msg, assistant_msg in history:
            prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
        prompt += f"User: {message}\nAssistant:"
        
        # Generate using model_manager
        for event in self.model_manager.generate(
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            skip_prompt=True
        ):
            yield event
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """
        Get detailed information about an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dictionary containing agent configuration and status
        """
        if agent_name not in self.agent_configs:
            return {"exists": False, "error": f"Agent '{agent_name}' not found"}
        
        config = self.agent_configs[agent_name]
        is_saved = (self.agents_dir / f"{agent_name}.json").exists()
        
        return {
            "exists": True,
            "name": config.name,
            "description": config.description,
            "system_prompt": config.system_prompt,
            "model_name": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "tools": config.tools,
            "agent_id": config.agent_id,
            "is_saved": is_saved,
            "model_loaded": self.model_manager.get_current_model_name() == config.model_name,
            "has_strands_instance": agent_name in self._agent_instances
        }
