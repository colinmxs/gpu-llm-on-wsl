"""
Tool Manager - Backend logic for Strands SDK tool creation and management.

This module provides a clean interface for managing tools that can be used by agents,
handling tool configuration, persistence, and instantiation independently of UI concerns.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class ToolConfig:
    """Configuration for a Strands SDK tool."""
    name: str
    description: str
    function_code: str  # Python code that defines the tool function
    parameters_schema: Dict[str, Any]  # JSON schema for tool parameters
    returns_schema: Dict[str, Any]  # JSON schema for return value
    
    def __post_init__(self):
        if not self.parameters_schema:
            self.parameters_schema = {"type": "object", "properties": {}}
        if not self.returns_schema:
            self.returns_schema = {"type": "string"}


class ToolManager:
    """
    Manages Strands SDK tools including creation, loading, saving, and instantiation.
    
    This class is UI-agnostic and returns data dictionaries for any frontend.
    """
    
    def __init__(self, tools_dir: Path):
        """
        Initialize the ToolManager.
        
        Args:
            tools_dir: Directory where tool configurations are stored
        """
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.active_tools: Dict[str, ToolConfig] = {}
        
        # Auto-load all saved tools at startup
        self._load_all_tools()
    
    def _load_all_tools(self):
        """Load all saved tool configurations from disk into memory."""
        for tool_file in self.tools_dir.glob("*.json"):
            try:
                with open(tool_file, 'r') as f:
                    config_dict = json.load(f)
                config = ToolConfig(**config_dict)
                self.active_tools[config.name] = config
            except Exception as e:
                print(f"Warning: Failed to load tool {tool_file.name}: {str(e)}")
    
    def create_tool(self, config: ToolConfig) -> Dict[str, Any]:
        """
        Create a new tool configuration and automatically save to disk.
        
        Args:
            config: Tool configuration
            
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - tool_name: Optional[str]
                - filepath: Optional[str]
                - error: Optional[str]
        """
        try:
            if not config.name:
                return {"success": False, "error": "Tool name is required"}
            
            if not config.function_code:
                return {"success": False, "error": "Function code is required"}
            
            # Store in active tools
            self.active_tools[config.name] = config
            
            # Auto-save to disk
            filepath = self.tools_dir / f"{config.name}.json"
            with open(filepath, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
            return {
                "success": True,
                "message": f"Tool '{config.name}' created and saved",
                "tool_name": config.name,
                "filepath": str(filepath)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create tool: {str(e)}"
            }
    
    def save_tool(self, tool_name: str) -> Dict[str, Any]:
        """
        Save a tool configuration to disk.
        
        Args:
            tool_name: Name of the tool to save
            
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - filepath: Optional[str]
                - error: Optional[str]
        """
        try:
            if tool_name not in self.active_tools:
                return {"success": False, "error": f"Tool '{tool_name}' not found"}
            
            config = self.active_tools[tool_name]
            filepath = self.tools_dir / f"{tool_name}.json"
            
            with open(filepath, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
            return {
                "success": True,
                "message": f"Tool '{tool_name}' saved successfully",
                "filepath": str(filepath)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save tool: {str(e)}"
            }
    
    def load_tool(self, tool_name: str) -> Dict[str, Any]:
        """
        Load a tool configuration from disk.
        
        Args:
            tool_name: Name of the tool to load
            
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - config: Optional[ToolConfig]
                - error: Optional[str]
        """
        try:
            filepath = self.tools_dir / f"{tool_name}.json"
            
            if not filepath.exists():
                return {"success": False, "error": f"Tool file not found: {filepath}"}
            
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            config = ToolConfig(**config_dict)
            self.active_tools[tool_name] = config
            
            return {
                "success": True,
                "message": f"Tool '{tool_name}' loaded successfully",
                "config": config
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load tool: {str(e)}"
            }
    
    def list_saved_tools(self) -> List[str]:
        """
        Get a list of saved tool configurations.
        
        Returns:
            List of tool names
        """
        if not self.tools_dir.exists():
            return []
        
        return sorted([f.stem for f in self.tools_dir.glob("*.json")])
    
    def list_active_tools(self) -> List[str]:
        """
        Get a list of currently active (in-memory) tools.
        
        Returns:
            List of tool names
        """
        return sorted(list(self.active_tools.keys()))
    
    def get_tool_config(self, tool_name: str) -> Optional[ToolConfig]:
        """
        Get the configuration of an active tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolConfig if found, None otherwise
        """
        return self.active_tools.get(tool_name)
    
    def delete_tool(self, tool_name: str, delete_file: bool = False) -> Dict[str, Any]:
        """
        Delete a tool from active memory and optionally from disk.
        
        Args:
            tool_name: Name of the tool to delete
            delete_file: Whether to also delete the saved file
            
        Returns:
            Dictionary containing:
                - success: bool
                - message: str
                - error: Optional[str]
        """
        try:
            messages = []
            
            # Remove from active tools
            if tool_name in self.active_tools:
                del self.active_tools[tool_name]
                messages.append(f"Removed '{tool_name}' from active tools")
            else:
                messages.append(f"Tool '{tool_name}' was not in active memory")
            
            # Delete file if requested
            if delete_file:
                filepath = self.tools_dir / f"{tool_name}.json"
                if filepath.exists():
                    filepath.unlink()
                    messages.append(f"Deleted tool file: {filepath}")
                else:
                    messages.append("No saved file found to delete")
            
            return {
                "success": True,
                "message": "\n".join(messages)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete tool: {str(e)}"
            }
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary containing tool configuration and status
        """
        if tool_name not in self.active_tools:
            return {"exists": False, "error": f"Tool '{tool_name}' not found"}
        
        config = self.active_tools[tool_name]
        is_saved = (self.tools_dir / f"{tool_name}.json").exists()
        
        return {
            "exists": True,
            "name": config.name,
            "description": config.description,
            "function_code": config.function_code,
            "parameters_schema": config.parameters_schema,
            "returns_schema": config.returns_schema,
            "is_saved": is_saved
        }
