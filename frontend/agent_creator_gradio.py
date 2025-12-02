"""
Gradio Frontend for Strands Agent Creator & Tool Creator

This module provides a comprehensive UI for creating and managing Strands SDK agents and tools.
Includes both Agent Creator and Tool Creator in a single tabbed interface.
"""

import gradio as gr
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.model_manager import ModelManager
from strands_agents.agent_manager import AgentManager
from strands_agents.tool_manager import ToolManager

# Import tab modules
from frontend.agent_creator_tab import AgentCreatorTab
from frontend.tool_creator_tab import ToolCreatorTab
from frontend.agent_playground_tab import AgentPlaygroundTab


# Initialize managers
# When running in Docker, these paths map to mounted volumes from docker-compose.yml:
# - ./agents:/app/agents (host ./agents directory mounted to /app/agents in container)
# - ./tools:/app/tools (host ./tools directory mounted to /app/tools in container)
MODELS_DIR = Path("/app/models")
AGENTS_DIR = Path("/app/agents")
TOOLS_DIR = Path("/app/tools")

# Global manager instances (will be set based on API mode toggle)
model_manager = None
agent_manager = None
tool_manager = None

def initialize_managers(use_api_mode: bool = False, api_url: str = "http://localhost:8000"):
    """Initialize or reinitialize managers with the specified mode."""
    global model_manager, agent_manager, tool_manager
    
    model_manager = ModelManager(models_dir=MODELS_DIR, use_api=use_api_mode, api_base_url=api_url)
    agent_manager = AgentManager(agents_dir=AGENTS_DIR, model_manager=model_manager, tools_dir=TOOLS_DIR)
    tool_manager = ToolManager(tools_dir=TOOLS_DIR)
    
    return model_manager, agent_manager, tool_manager

# Initialize with default (native mode)
initialize_managers(use_api_mode=False)


def create_agent_creator_ui():
    """Create the complete Agent Creator + Tool Creator + Agent Playground UI with tabs."""
    
    # Initialize tab instances
    agent_tab = AgentCreatorTab(model_manager, agent_manager, tool_manager)
    tool_tab = ToolCreatorTab(tool_manager)
    playground_tab = AgentPlaygroundTab(model_manager, agent_manager, tool_manager)
    
    with gr.Blocks(title="Strands SDK - Agent & Tool Creator") as interface:
        gr.Markdown("# 🚀 Strands SDK - Agent & Tool Creator")
        gr.Markdown("Comprehensive interface for creating AI agents and tools")
        
        # API Mode Toggle
        with gr.Accordion("⚙️ Settings", open=False):
            gr.Markdown("### Model Manager Mode")
            with gr.Row():
                api_mode_toggle = gr.Checkbox(
                    label="Use API Mode",
                    value=False,
                    info="Enable to use FastAPI backend for model operations instead of direct loading"
                )
                api_url_input = gr.Textbox(
                    label="API URL",
                    value="http://localhost:8000",
                    placeholder="http://localhost:8000",
                    visible=False
                )
            api_status = gr.Markdown("**Mode:** Native (Direct Model Loading)")
            
            def toggle_api_mode(use_api: bool, api_url: str):
                """Switch between native and API mode."""
                initialize_managers(use_api_mode=use_api, api_url=api_url)
                
                # Update tab instances
                agent_tab.model_manager = model_manager
                agent_tab.agent_manager = agent_manager
                tool_tab.tool_manager = tool_manager
                playground_tab.model_manager = model_manager
                playground_tab.agent_manager = agent_manager
                playground_tab.tool_manager = tool_manager
                
                mode_text = "API Mode" if use_api else "Native (Direct Model Loading)"
                url_visible = use_api
                status = f"**Mode:** {mode_text}"
                if use_api:
                    status += f"\n**API URL:** `{api_url}`"
                
                return status, gr.update(visible=url_visible)
            
            api_mode_toggle.change(
                fn=toggle_api_mode,
                inputs=[api_mode_toggle, api_url_input],
                outputs=[api_status, api_url_input]
            )
            
            api_url_input.change(
                fn=toggle_api_mode,
                inputs=[api_mode_toggle, api_url_input],
                outputs=[api_status, api_url_input]
            )
        
        with gr.Tabs():
            # Agent Creator Tab
            with gr.Tab("🤖 Agent Creator"):
                agent_components = agent_tab.create_ui()
            
            # Tool Creator Tab
            with gr.Tab("🛠️ Tool Creator"):
                tool_components = tool_tab.create_ui()
            
            # Agent Playground Tab
            with gr.Tab("🎮 Agent Playground"):
                playground_components = playground_tab.create_ui()
        
        # Refresh all dropdowns when the page loads
        interface.load(
            fn=lambda: [
                gr.update(choices=agent_tab.get_saved_agents()),
                gr.update(choices=agent_tab.get_available_tools()),
                gr.update(choices=tool_tab.tool_manager.list_saved_tools()),
                gr.update(choices=playground_tab.get_saved_agents())
            ],
            outputs=[
                agent_components['agent_list'],
                agent_components['tools_dropdown'],
                tool_components['tool_list'],
                playground_components['agent_list']
            ]
        )
    
    return interface


def launch_agent_creator(share=False, server_port=7861):
    """Launch the Agent Creator UI."""
    interface = create_agent_creator_ui()
    interface.launch(
        share=share,
        server_port=server_port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    launch_agent_creator()
