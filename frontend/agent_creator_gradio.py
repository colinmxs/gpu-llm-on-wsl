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

from api.client_agents_tools_example import AgentToolClient

# Import tab modules
from frontend.agent_creator_tab import AgentCreatorTab
from frontend.tool_creator_tab import ToolCreatorTab
from frontend.agent_playground_tab import AgentPlaygroundTab


# Initialize managers
# API client will communicate with the backend API server
API_BASE_URL = "http://localhost:8000"

# Global API client instance
api_client = None

def initialize_api_client(api_url: str = API_BASE_URL):
    """Initialize or reinitialize API client with the specified URL."""
    global api_client
    api_client = AgentToolClient(base_url=api_url)
    return api_client

# Initialize with default
initialize_api_client()


def create_agent_creator_ui():
    """Create the complete Agent Creator + Tool Creator + Agent Playground UI with tabs."""
    
    # Initialize tab instances with API client
    agent_tab = AgentCreatorTab(api_client)
    tool_tab = ToolCreatorTab(api_client)
    playground_tab = AgentPlaygroundTab(api_client)
    
    with gr.Blocks(title="Strands SDK - Agent & Tool Creator") as interface:
        gr.Markdown("# 🚀 Strands SDK - Agent & Tool Creator")
        gr.Markdown("Comprehensive interface for creating AI agents and tools")
        
        # API URL Configuration
        with gr.Accordion("⚙️ Settings", open=False):
            gr.Markdown("### API Configuration")
            with gr.Row():
                api_url_input = gr.Textbox(
                    label="API URL",
                    value=API_BASE_URL,
                    placeholder="http://localhost:8000",
                    info="URL of the backend API server"
                )
                update_api_btn = gr.Button("🔄 Update API URL", size="sm")
            api_status = gr.Markdown(f"**API URL:** `{API_BASE_URL}`")
            
            def update_api_url(api_url: str):
                """Update the API client URL."""
                initialize_api_client(api_url)
                
                # Update tab instances with new API client
                agent_tab.api_client = api_client
                tool_tab.api_client = api_client
                playground_tab.api_client = api_client
                
                status = f"**API URL:** `{api_url}`"
                return status
            
            update_api_btn.click(
                fn=update_api_url,
                inputs=[api_url_input],
                outputs=[api_status]
            )
            
            api_url_input.submit(
                fn=update_api_url,
                inputs=[api_url_input],
                outputs=[api_status]
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
                gr.update(choices=playground_tab.get_saved_agents())
            ],
            outputs=[
                agent_components['agent_list'],
                agent_components['tools_dropdown'],
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
