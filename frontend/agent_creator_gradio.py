"""
Gradio Frontend for Strands Agent Creator

This module provides a comprehensive UI for creating and managing Strands SDK agents,
including configuration of models, tools, conversation management, and advanced features.
"""

import gradio as gr
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.model_manager import ModelManager
from strands_agents.agent_manager import AgentManager
from strands_agents.tool_manager import ToolManager


# Initialize managers
MODELS_DIR = Path(__file__).parent.parent / "models"
AGENTS_DIR = Path(__file__).parent.parent / "strands_agents" / "examples" / "agents"
TOOLS_DIR = Path(__file__).parent.parent / "strands_agents" / "examples" / "tools"

model_manager = ModelManager(models_dir=MODELS_DIR)
agent_manager = AgentManager(agents_dir=AGENTS_DIR, model_manager=model_manager, tools_dir=TOOLS_DIR)
tool_manager = ToolManager(tools_dir=TOOLS_DIR)


# Preset configurations for quick start
PRESETS = {
    "Helpful Assistant": {
        "description": "A general-purpose helpful AI assistant",
        "system_prompt": "You are a helpful, harmless, and honest AI assistant. You provide clear, accurate, and concise responses. When you don't know something, you admit it. You are friendly and professional in your communication style.",
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "conv_manager_type": "Sliding Window",
        "window_size": 40,
    },
    "Code Assistant": {
        "description": "A coding assistant specialized in Python and debugging",
        "system_prompt": "You are an expert Python programmer and debugging assistant. You help users write clean, efficient, and well-documented code. You explain your reasoning, suggest best practices, and provide code examples when helpful. You're patient and break down complex concepts into understandable parts.",
        "temperature": 0.3,
        "max_tokens": 800,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.15,
        "conv_manager_type": "Sliding Window",
        "window_size": 40,
    },
    "Creative Writer": {
        "description": "A creative writing assistant for storytelling and content creation",
        "system_prompt": "You are a creative writing assistant with expertise in storytelling, narrative development, and engaging content creation. You help users brainstorm ideas, develop characters, craft compelling plots, and refine their writing. You're imaginative, supportive, and provide constructive feedback.",
        "temperature": 0.9,
        "max_tokens": 1000,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "conv_manager_type": "Sliding Window",
        "window_size": 40,
    },
    "Research Agent": {
        "description": "An agent specialized in research and information gathering",
        "system_prompt": "You are a research assistant skilled at gathering, analyzing, and synthesizing information. You help users find relevant information, compare sources, and draw insightful conclusions. You cite your reasoning and acknowledge limitations in available data.",
        "temperature": 0.4,
        "max_tokens": 800,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "conv_manager_type": "Summarizing",
        "summary_ratio": 0.3,
    },
    "Custom": {
        "description": "",
        "system_prompt": "",
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "conv_manager_type": "Sliding Window",
        "window_size": 40,
    }
}


def get_available_models() -> List[str]:
    """Get list of available models from model manager."""
    models = model_manager.list_models()
    return models if models else ["No models found"]


def get_available_tools() -> List[str]:
    """Get list of available tools from tool manager."""
    tools = tool_manager.list_saved_tools()
    return tools if tools else []


def get_saved_agents() -> List[str]:
    """Get list of saved agents."""
    agents = agent_manager.list_saved_agents()
    return agents if agents else ["No agents saved"]


def apply_preset(preset_name: str) -> Tuple:
    """
    Apply a preset configuration.
    
    Returns tuple of all configuration values.
    """
    preset = PRESETS.get(preset_name, PRESETS["Custom"])
    
    # Return all configuration values in the correct order
    return (
        preset.get("description", ""),
        preset.get("system_prompt", ""),
        preset.get("temperature", 0.7),
        preset.get("max_tokens", 500),
        preset.get("top_p", 0.9),
        preset.get("top_k", 50),
        preset.get("repetition_penalty", 1.1),
        preset.get("conv_manager_type", "Sliding Window"),
        preset.get("window_size", 40),
        preset.get("truncate_results", True),
        preset.get("summary_ratio", 0.3),
        preset.get("preserve_recent", 10),
        preset.get("custom_summary_prompt", ""),
    )


def load_agent_config(agent_name: str) -> Tuple:
    """
    Load an agent configuration and return all field values.
    
    Returns tuple of all configuration values.
    """
    if not agent_name or agent_name == "No agents saved":
        return apply_preset("Custom")
    
    agent_info = agent_manager.get_agent_info(agent_name)
    
    if not agent_info.get("exists"):
        return apply_preset("Custom")
    
    # Extract configuration
    return (
        agent_name,  # name
        agent_info.get("description", ""),
        agent_info.get("system_prompt", ""),
        agent_info.get("model_name", ""),
        agent_info.get("temperature", 0.7),
        agent_info.get("max_tokens", 500),
        agent_info.get("top_p", 0.9),
        agent_info.get("top_k", 50),
        agent_info.get("repetition_penalty", 1.1),
        agent_info.get("tools", []),
        "Sliding Window",  # Default conversation manager
        40,  # Default window size
        True,  # Default truncate results
        0.3,  # Default summary ratio
        10,  # Default preserve recent
        "",  # Default custom summary prompt
    )


def save_agent(
    name: str,
    description: str,
    system_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    tools: List[str],
    conv_manager_type: str,
    window_size: int,
    truncate_results: bool,
    summary_ratio: float,
    preserve_recent: int,
    custom_summary_prompt: str,
) -> Tuple[str, str]:
    """
    Save agent configuration.
    
    Returns (status_message, updated_agent_list_choices).
    """
    if not name:
        return "❌ Error: Agent name is required", gr.update()
    
    if not model_name or model_name == "No models found":
        return "❌ Error: Please select a model", gr.update()
    
    # Create agent
    result = agent_manager.create_agent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        model_name=model_name,
        tools=tools or None,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    
    if result.get("success"):
        # Update agent list
        agent_list = get_saved_agents()
        return f"✅ {result.get('message')}", gr.update(choices=agent_list, value=name)
    else:
        return f"❌ Error: {result.get('error')}", gr.update()


def delete_agent(agent_name: str) -> Tuple[str, str]:
    """
    Delete an agent.
    
    Returns (status_message, updated_agent_list_choices).
    """
    if not agent_name or agent_name == "No agents saved":
        return "❌ Error: Please select an agent to delete", gr.update()
    
    result = agent_manager.delete_agent(agent_name, delete_file=True)
    
    if result.get("success"):
        agent_list = get_saved_agents()
        return f"✅ Agent '{agent_name}' deleted successfully", gr.update(choices=agent_list, value=None)
    else:
        return f"❌ Error: {result.get('error')}", gr.update()


def export_agent_json(agent_name: str) -> str:
    """
    Export agent configuration as JSON.
    
    Returns JSON string for display.
    """
    if not agent_name or agent_name == "No agents saved":
        return json.dumps({"error": "No agent selected"}, indent=2)
    
    agent_info = agent_manager.get_agent_info(agent_name)
    
    if not agent_info.get("exists"):
        return json.dumps({"error": "Agent not found"}, indent=2)
    
    # Create export configuration
    export_config = {
        "name": agent_info.get("name"),
        "description": agent_info.get("description"),
        "system_prompt": agent_info.get("system_prompt"),
        "model_name": agent_info.get("model_name"),
        "temperature": agent_info.get("temperature"),
        "max_tokens": agent_info.get("max_tokens"),
        "top_p": agent_info.get("top_p"),
        "top_k": agent_info.get("top_k"),
        "repetition_penalty": agent_info.get("repetition_penalty"),
        "tools": agent_info.get("tools", []),
    }
    
    return json.dumps(export_config, indent=2)


def get_model_status() -> str:
    """Get current model loading status."""
    if model_manager.is_model_loaded():
        current_model = model_manager.get_current_model_name()
        gpu_stats = model_manager.get_gpu_stats()
        
        status = f"✅ **Model Loaded:** `{current_model}`\n\n"
        
        if gpu_stats.get("available"):
            status += f"**GPU:** {gpu_stats.get('name', 'Unknown')}\n"
            status += f"**Memory Usage:** {gpu_stats.get('usage_percent', 0):.1f}% "
            status += f"({gpu_stats.get('reserved_gb', 0):.2f}GB / {gpu_stats.get('total_gb', 0):.2f}GB)"
        
        return status
    else:
        return "⚠️ **No model loaded**\n\nPlease load a model before testing agents."


def get_tool_info(tool_names: List[str]) -> str:
    """Get information about selected tools."""
    if not tool_names:
        return "No tools selected"
    
    info_parts = []
    for tool_name in tool_names:
        tool_info = tool_manager.get_tool_info(tool_name)
        if tool_info.get("exists"):
            info_parts.append(f"### {tool_name}")
            info_parts.append(f"**Description:** {tool_info.get('description', 'N/A')}")
            info_parts.append(f"**Strands Compatible:** {'✅' if tool_info.get('strands_compatible') else '❌'}")
            info_parts.append("")
    
    return "\n".join(info_parts) if info_parts else "No valid tools selected"


def update_conv_manager_visibility(conv_type: str) -> Tuple:
    """Update visibility of conversation manager options based on type."""
    if conv_type == "Sliding Window":
        return (
            gr.update(visible=True),   # window_size
            gr.update(visible=True),   # truncate_results
            gr.update(visible=False),  # summary_ratio
            gr.update(visible=False),  # preserve_recent
            gr.update(visible=False),  # custom_summary_prompt
        )
    elif conv_type == "Summarizing":
        return (
            gr.update(visible=False),  # window_size
            gr.update(visible=False),  # truncate_results
            gr.update(visible=True),   # summary_ratio
            gr.update(visible=True),   # preserve_recent
            gr.update(visible=True),   # custom_summary_prompt
        )
    else:  # Null
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def create_agent_creator_ui():
    """Create the complete Agent Creator UI."""
    
    with gr.Blocks(title="Strands Agent Creator") as interface:
        gr.Markdown("# 🤖 Strands Agent Creator")
        gr.Markdown("Create and manage AI agents with custom configurations, tools, and conversation management.")
        
        with gr.Row():
            # Left sidebar - Agent Library
            with gr.Column(scale=1):
                gr.Markdown("## 📚 Agent Library")
                
                agent_list = gr.Dropdown(
                    choices=get_saved_agents(),
                    label="Saved Agents",
                    interactive=True,
                    value=None
                )
                
                with gr.Row():
                    load_btn = gr.Button("📂 Load", size="sm")
                    delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                
                gr.Markdown("### Quick Presets")
                preset_dropdown = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    label="Apply Preset",
                    value="Custom",
                    interactive=True
                )
                
                apply_preset_btn = gr.Button("✨ Apply Preset", size="sm")
                
                gr.Markdown("---")
                
                # Model Status
                gr.Markdown("### 🖥️ Model Status")
                model_status = gr.Markdown(get_model_status())
                refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")
            
            # Main configuration area
            with gr.Column(scale=3):
                status_message = gr.Markdown("")
                
                # Basic Configuration
                with gr.Group():
                    gr.Markdown("## 🎯 Basic Configuration")
                    
                    agent_name = gr.Textbox(
                        label="Agent Name",
                        placeholder="e.g., my-helpful-assistant",
                        info="Unique identifier for your agent"
                    )
                    
                    agent_description = gr.Textbox(
                        label="Description",
                        placeholder="Brief description of your agent's purpose",
                        lines=2
                    )
                    
                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        placeholder="Enter detailed instructions for your agent's behavior and personality...",
                        lines=6,
                        info="This guides how your agent responds to users"
                    )
                
                # Model Configuration
                with gr.Group():
                    gr.Markdown("## 🔧 Model Configuration")
                    
                    model_dropdown = gr.Dropdown(
                        choices=get_available_models(),
                        label="Model",
                        info="Select the LLM model for this agent"
                    )
                    
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                            info="Higher = more creative, Lower = more focused"
                        )
                        
                        max_tokens = gr.Slider(
                            minimum=50,
                            maximum=2000,
                            value=500,
                            step=50,
                            label="Max Tokens",
                            info="Maximum response length"
                        )
                    
                    with gr.Row():
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-P",
                            info="Nucleus sampling threshold"
                        )
                        
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Top-K",
                            info="Top-k sampling parameter"
                        )
                    
                    repetition_penalty = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.1,
                        step=0.05,
                        label="Repetition Penalty",
                        info="Penalty for repeating tokens"
                    )
                
                # Tools Configuration
                with gr.Group():
                    gr.Markdown("## 🛠️ Tools")
                    
                    tools_dropdown = gr.Dropdown(
                        choices=get_available_tools(),
                        label="Select Tools",
                        multiselect=True,
                        info="Tools available to this agent"
                    )
                    
                    tool_info_display = gr.Markdown("No tools selected")
                
                # Conversation Management
                with gr.Group():
                    gr.Markdown("## 💬 Conversation Management")
                    
                    conv_manager_type = gr.Radio(
                        choices=["Sliding Window", "Summarizing", "Null"],
                        value="Sliding Window",
                        label="Conversation Manager Type",
                        info="How the agent manages conversation history"
                    )
                    
                    # Sliding Window Options
                    with gr.Group() as sliding_options:
                        window_size = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=40,
                            step=5,
                            label="Window Size",
                            info="Maximum number of messages to keep"
                        )
                        
                        truncate_results = gr.Checkbox(
                            value=True,
                            label="Truncate Tool Results",
                            info="Truncate large tool results when they exceed context limits"
                        )
                    
                    # Summarizing Options
                    with gr.Group(visible=False) as summarizing_options:
                        summary_ratio = gr.Slider(
                            minimum=0.1,
                            maximum=0.8,
                            value=0.3,
                            step=0.1,
                            label="Summary Ratio",
                            info="Percentage of messages to summarize when reducing context"
                        )
                        
                        preserve_recent = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=10,
                            step=5,
                            label="Preserve Recent Messages",
                            info="Minimum number of recent messages to always keep"
                        )
                        
                        custom_summary_prompt = gr.Textbox(
                            label="Custom Summarization Prompt (Optional)",
                            placeholder="Leave empty to use default summarization prompt...",
                            lines=4
                        )
                
                # Advanced Configuration (Collapsible)
                with gr.Accordion("⚙️ Advanced Configuration", open=False):
                    gr.Markdown("### Agent State")
                    gr.Markdown("*Key-value pairs for stateful information (JSON format)*")
                    agent_state_json = gr.Textbox(
                        label="Agent State (JSON)",
                        placeholder='{"key": "value"}',
                        lines=3
                    )
                    
                    record_direct_tool_call = gr.Checkbox(
                        value=True,
                        label="Record Direct Tool Calls",
                        info="Whether to record direct tool calls in message history"
                    )
                
                # Action Buttons
                with gr.Row():
                    save_btn = gr.Button("💾 Save Agent", variant="primary", size="lg")
                    duplicate_btn = gr.Button("📋 Duplicate", size="lg")
                    export_btn = gr.Button("📤 Export JSON", size="lg")
                
                # Export Display
                with gr.Accordion("📄 Agent Configuration (JSON)", open=False):
                    export_display = gr.Code(language="json", label="Configuration")
        
        # Event Handlers
        
        # Apply preset
        apply_preset_btn.click(
            fn=apply_preset,
            inputs=[preset_dropdown],
            outputs=[
                agent_description,
                system_prompt,
                temperature,
                max_tokens,
                top_p,
                top_k,
                repetition_penalty,
                conv_manager_type,
                window_size,
                truncate_results,
                summary_ratio,
                preserve_recent,
                custom_summary_prompt,
            ]
        )
        
        # Load agent
        load_btn.click(
            fn=load_agent_config,
            inputs=[agent_list],
            outputs=[
                agent_name,
                agent_description,
                system_prompt,
                model_dropdown,
                temperature,
                max_tokens,
                top_p,
                top_k,
                repetition_penalty,
                tools_dropdown,
                conv_manager_type,
                window_size,
                truncate_results,
                summary_ratio,
                preserve_recent,
                custom_summary_prompt,
            ]
        )
        
        # Save agent
        save_btn.click(
            fn=save_agent,
            inputs=[
                agent_name,
                agent_description,
                system_prompt,
                model_dropdown,
                temperature,
                max_tokens,
                top_p,
                top_k,
                repetition_penalty,
                tools_dropdown,
                conv_manager_type,
                window_size,
                truncate_results,
                summary_ratio,
                preserve_recent,
                custom_summary_prompt,
            ],
            outputs=[status_message, agent_list]
        )
        
        # Delete agent
        delete_btn.click(
            fn=delete_agent,
            inputs=[agent_list],
            outputs=[status_message, agent_list]
        )
        
        # Export agent
        export_btn.click(
            fn=export_agent_json,
            inputs=[agent_list],
            outputs=[export_display]
        )
        
        # Refresh model status
        refresh_status_btn.click(
            fn=get_model_status,
            outputs=[model_status]
        )
        
        # Update tool info when selection changes
        tools_dropdown.change(
            fn=get_tool_info,
            inputs=[tools_dropdown],
            outputs=[tool_info_display]
        )
        
        # Update conversation manager visibility
        conv_manager_type.change(
            fn=update_conv_manager_visibility,
            inputs=[conv_manager_type],
            outputs=[window_size, truncate_results, summary_ratio, preserve_recent, custom_summary_prompt]
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
