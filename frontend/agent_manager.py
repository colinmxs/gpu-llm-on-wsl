"""
Agent Manager - Backend logic for Strands SDK agent creation, management, and execution.

This module provides a clean interface for managing Strands SDK agents,
handling agent configuration, persistence, and execution independently of UI concerns.
"""

import os
import json
import gc
from pathlib import Path
from typing import Optional, Dict, Any, Generator, List, Tuple
from dataclasses import dataclass, asdict
import time
import threading

from model_manager import ModelManager


@dataclass
class AgentConfig:
    """Configuration for a Strands SDK agent."""
    name: str
    description: str
    system_prompt: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    tools: List[str] = None  # List of tool names associated with this agent
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []


class AgentManager:
    """
    Manages Strands SDK agents including creation, loading, saving, and execution.
    
    This class is UI-agnostic and returns data dictionaries for any frontend.
    """
    
    def __init__(self, agents_dir: Path, model_manager: ModelManager):
        """
        Initialize the AgentManager.
        
        Args:
            agents_dir: Directory where agent configurations are stored
            model_manager: ModelManager instance for LLM inference
        """
        self.agents_dir = Path(agents_dir)
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.model_manager = model_manager
        self.active_agents: Dict[str, AgentConfig] = {}
        
        # Auto-load all saved agents at startup
        self._load_all_agents()
    
    def _load_all_agents(self):
        """Load all saved agent configurations from disk into memory."""
        for agent_file in self.agents_dir.glob("*.json"):
            try:
                with open(agent_file, 'r') as f:
                    config_dict = json.load(f)
                config = AgentConfig(**config_dict)
                self.active_agents[config.name] = config
            except Exception as e:
                print(f"Warning: Failed to load agent {agent_file.name}: {str(e)}")
    
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
            
            # Store in active agents
            self.active_agents[config.name] = config
            
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
            if agent_name not in self.active_agents:
                return {"success": False, "error": f"Agent '{agent_name}' not found"}
            
            config = self.active_agents[agent_name]
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
            self.active_agents[agent_name] = config
            
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
        return sorted(list(self.active_agents.keys()))
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """
        Get the configuration of an active agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            AgentConfig if found, None otherwise
        """
        return self.active_agents.get(agent_name)
    
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
            if agent_name in self.active_agents:
                del self.active_agents[agent_name]
                messages.append(f"Removed '{agent_name}' from active agents")
            else:
                messages.append(f"Agent '{agent_name}' was not in active memory")
            
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
        Chat with an agent using the configured model and parameters.
        
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
        if agent_name not in self.active_agents:
            yield {"type": "error", "error": f"Agent '{agent_name}' not found"}
            return
        
        config = self.active_agents[agent_name]
        
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
        
        # Build the prompt with system prompt and history
        if history is None:
            history = []
        
        # Construct prompt with system message
        prompt = f"System: {config.system_prompt}\n\n"
        
        # Add conversation history
        for user_msg, assistant_msg in history:
            prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
        
        # Add current message
        prompt += f"User: {message}\nAssistant:"
        
        # Generate response using model_manager with agent's configuration
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
        if agent_name not in self.active_agents:
            return {"exists": False, "error": f"Agent '{agent_name}' not found"}
        
        config = self.active_agents[agent_name]
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
            "is_saved": is_saved,
            "model_loaded": self.model_manager.get_current_model_name() == config.model_name
        }
