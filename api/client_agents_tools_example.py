"""
Example Python client for the Agent and Tool Management API

This demonstrates how to interact with the agent_manager and tool_manager
endpoints from your local machine while the LLMs run in the Docker container.
"""

import requests
import json
from typing import Generator, Dict, Any, List, Optional


class AgentToolClient:
    """Client for interacting with Agent and Tool Management API"""
    
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
    
    # ============ Tool Management Methods ============
    
    def create_tool(
        self,
        name: str,
        description: str,
        function_code: str,
        parameters_schema: Optional[Dict[str, Any]] = None,
        returns_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new tool.
        
        Args:
            name: Tool name
            description: Tool description
            function_code: Python function code with @strands.tool decorator
            parameters_schema: JSON schema for parameters (optional)
            returns_schema: JSON schema for return value (optional)
            
        Returns:
            Dictionary with success status and tool details
        """
        response = requests.post(
            f"{self.base_url}/tools",
            json={
                "name": name,
                "description": description,
                "function_code": function_code,
                "parameters_schema": parameters_schema,
                "returns_schema": returns_schema
            }
        )
        response.raise_for_status()
        return response.json()
    
    def list_tools(self) -> List[str]:
        """List all active tools"""
        response = requests.get(f"{self.base_url}/tools")
        response.raise_for_status()
        return response.json()
    
    def list_saved_tools(self) -> List[str]:
        """List all saved tool configurations"""
        response = requests.get(f"{self.base_url}/tools/saved")
        response.raise_for_status()
        return response.json()
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific tool"""
        response = requests.get(f"{self.base_url}/tools/{tool_name}")
        response.raise_for_status()
        return response.json()
    
    def load_tool(self, tool_name: str) -> Dict[str, Any]:
        """Load a tool from disk"""
        response = requests.post(f"{self.base_url}/tools/{tool_name}/load")
        response.raise_for_status()
        return response.json()
    
    def save_tool(self, tool_name: str) -> Dict[str, Any]:
        """Save a tool to disk"""
        response = requests.post(f"{self.base_url}/tools/{tool_name}/save")
        response.raise_for_status()
        return response.json()
    
    def delete_tool(self, tool_name: str, delete_file: bool = False) -> Dict[str, Any]:
        """Delete a tool from memory and optionally from disk"""
        response = requests.delete(
            f"{self.base_url}/tools/{tool_name}",
            params={"delete_file": delete_file}
        )
        response.raise_for_status()
        return response.json()
    
    # ============ Agent Management Methods ============
    
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
        """
        Create a new agent.
        
        Args:
            name: Agent name
            system_prompt: System prompt for the agent
            model_name: HuggingFace model name
            description: Agent description
            tools: List of tool names (optional)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate (1-4096)
            top_p: Nucleus sampling parameter (0.0-1.0)
            top_k: Top-k sampling parameter (0-100)
            repetition_penalty: Repetition penalty (1.0-2.0)
            
        Returns:
            Dictionary with success status and agent details
        """
        response = requests.post(
            f"{self.base_url}/agents",
            json={
                "name": name,
                "description": description,
                "system_prompt": system_prompt,
                "model_name": model_name,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty
            }
        )
        response.raise_for_status()
        return response.json()
    
    def list_agents(self) -> List[str]:
        """List all active agents"""
        response = requests.get(f"{self.base_url}/agents")
        response.raise_for_status()
        return response.json()
    
    def list_saved_agents(self) -> List[str]:
        """List all saved agent configurations"""
        response = requests.get(f"{self.base_url}/agents/saved")
        response.raise_for_status()
        return response.json()
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific agent"""
        response = requests.get(f"{self.base_url}/agents/{agent_name}")
        response.raise_for_status()
        return response.json()
    
    def load_agent(self, agent_name: str) -> Dict[str, Any]:
        """Load an agent from disk"""
        response = requests.post(f"{self.base_url}/agents/{agent_name}/load")
        response.raise_for_status()
        return response.json()
    
    def save_agent(self, agent_name: str) -> Dict[str, Any]:
        """Save an agent to disk"""
        response = requests.post(f"{self.base_url}/agents/{agent_name}/save")
        response.raise_for_status()
        return response.json()
    
    def delete_agent(self, agent_name: str, delete_file: bool = False) -> Dict[str, Any]:
        """Delete an agent from memory and optionally from disk"""
        response = requests.delete(
            f"{self.base_url}/agents/{agent_name}",
            params={"delete_file": delete_file}
        )
        response.raise_for_status()
        return response.json()
    
    def chat_with_agent(
        self,
        agent_name: str,
        message: str,
        history: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Chat with an agent (non-streaming).
        
        Args:
            agent_name: Name of the agent to chat with
            message: User message
            history: Chat history as list of [user_msg, assistant_msg] pairs
            
        Returns:
            Dictionary with 'text' and 'metadata' keys
        """
        response = requests.post(
            f"{self.base_url}/agents/{agent_name}/chat",
            json={
                "message": message,
                "history": history
            }
        )
        response.raise_for_status()
        return response.json()
    
    def chat_with_agent_stream(
        self,
        agent_name: str,
        message: str,
        history: Optional[List[List[str]]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Chat with an agent with streaming (SSE).
        
        Args:
            agent_name: Name of the agent to chat with
            message: User message
            history: Chat history as list of [user_msg, assistant_msg] pairs
            
        Yields:
            Dictionary with Strands SDK StreamEvent format
        """
        response = requests.post(
            f"{self.base_url}/agents/{agent_name}/chat/stream",
            json={
                "message": message,
                "history": history
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
    client = AgentToolClient("http://localhost:8000")
    
    print("=" * 60)
    print("Tool Management Example")
    print("=" * 60)
    print()
    
    # Create a simple tool
    calculator_code = """
import strands

@strands.tool
def add_numbers(a: float, b: float) -> float:
    '''Add two numbers together.'''
    return a + b
"""
    
    print("Creating a calculator tool...")
    tool_result = client.create_tool(
        name="calculator",
        description="A simple calculator tool that adds numbers",
        function_code=calculator_code,
        parameters_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        },
        returns_schema={"type": "number"}
    )
    print(f"Tool created: {tool_result['success']}")
    if tool_result['success']:
        print(f"Tool name: {tool_result['tool_name']}")
        print(f"Python file: {tool_result.get('python_file', 'N/A')}")
    print()
    
    # List all tools
    print("Available tools:")
    tools = client.list_tools()
    for tool in tools:
        print(f"  - {tool}")
    print()
    
    # Get tool info
    if tools:
        tool_name = tools[0]
        print(f"Tool info for '{tool_name}':")
        tool_info = client.get_tool_info(tool_name)
        print(f"  Description: {tool_info['description']}")
        print(f"  Saved: {tool_info['is_saved']}")
        print(f"  Strands compatible: {tool_info['strands_compatible']}")
        print()
    
    print("=" * 60)
    print("Agent Management Example")
    print("=" * 60)
    print()
    
    # Create an agent (note: model must be downloaded first)
    print("Creating a math tutor agent...")
    agent_result = client.create_agent(
        name="math_tutor",
        description="A helpful math tutor",
        system_prompt="You are a patient and helpful math tutor. Explain concepts clearly and provide examples.",
        model_name="meta-llama/Llama-3.2-1B-Instruct",  # Change to your model
        tools=["calculator"] if tools else None,
        temperature=0.7,
        max_tokens=300
    )
    print(f"Agent created: {agent_result['success']}")
    if agent_result['success']:
        print(f"Agent name: {agent_result['agent_name']}")
    print()
    
    # List all agents
    print("Available agents:")
    agents = client.list_agents()
    for agent in agents:
        print(f"  - {agent}")
    print()
    
    # Get agent info
    if agents:
        agent_name = agents[0]
        print(f"Agent info for '{agent_name}':")
        agent_info = client.get_agent_info(agent_name)
        print(f"  Description: {agent_info['description']}")
        print(f"  Model: {agent_info['model_name']}")
        print(f"  Tools: {agent_info.get('tools', [])}")
        print(f"  Model loaded: {agent_info['model_loaded']}")
        print()
        
        # Chat with agent (if model is loaded)
        if agent_info['model_loaded']:
            print(f"Chatting with '{agent_name}'...")
            message = "What is 5 + 7?"
            print(f"User: {message}")
            print("Agent: ", end="", flush=True)
            
            for event in client.chat_with_agent_stream(agent_name, message):
                if event.get('type') == 'text':
                    print(event.get('text', ''), end='', flush=True)
                elif event.get('type') == 'complete':
                    print("\n")
                elif event.get('type') == 'error':
                    print(f"\nError: {event.get('error')}")
        else:
            print(f"Note: Load the model '{agent_info['model_name']}' first to chat with this agent.")
    else:
        print("No agents available. Create one first!")
    
    print()
    print("=" * 60)
    print("Example complete!")
    print("=" * 60)
