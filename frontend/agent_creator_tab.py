"""
Agent Creator Tab for Gradio Frontend

This module provides the UI components and functionality for creating and managing Strands SDK agents.
"""

import gradio as gr
import json
from typing import List, Tuple


PROMPT_TEMPLATES = {
    "Llama 3": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    
    "Mistral Instruct": """<s>[INST] You are a helpful AI assistant.

{user_message} [/INST]""",
    
    "Cohere Command": """<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a helpful AI assistant.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_message}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>""",
}


class AgentCreatorTab:
    """Manages the Agent Creator tab UI and functionality."""
    
    def __init__(self, manager):
        self.manager = manager
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from API."""
        try:
            models = self.manager.list_models()
            return models if models else ["No models found"]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return ["No models found"]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        try:
            tools = self.manager.list_saved_tools()
            return tools if tools else []
        except Exception as e:
            print(f"Error fetching tools: {e}")
            return []
    
    def get_saved_agents(self) -> List[str]:
        """Get list of saved agents."""
        try:
            agents = self.manager.list_saved_agents()
            return agents if agents else ["No agents saved"]
        except Exception as e:
            print(f"Error fetching agents: {e}")
            return ["No agents saved"]
    
    def apply_prompt_template(self, template_name: str) -> str:
        """Apply a prompt template. Returns the prompt template text."""
        return PROMPT_TEMPLATES.get(template_name, "")
    
    def load_agent_config(self, agent_name: str) -> Tuple:
        """Load an agent configuration and return all field values."""
        # Default empty values
        default_values = (
            "",  # name
            "",  # description
            "",  # system_prompt
            "",  # model_name
            0.7,  # temperature
            500,  # max_tokens
            0.9,  # top_p
            50,  # top_k
            1.1,  # repetition_penalty
            [],  # tools
            "Sliding Window",  # conv_manager_type
            40,  # window_size
            True,  # truncate_results
            0.3,  # summary_ratio
            10,  # preserve_recent
            "",  # custom_summary_prompt
        )
        
        if not agent_name or agent_name == "No agents saved":
            return default_values
        
        try:
            agent_info = self.manager.get_agent_info(agent_name)
        except Exception as e:
            print(f"Error loading agent: {e}")
            return default_values
        
        if not agent_info.get("exists"):
            return default_values
        
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
        self,
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
        """Save agent configuration. Returns (status_message, updated_agent_list_choices)."""
        if not name:
            return "❌ Error: Agent name is required", gr.update()
        
        if not model_name or model_name == "No models found":
            return "❌ Error: Please select a model", gr.update()
        
        # Create agent via API
        try:
            result = self.manager.create_agent(
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
        except Exception as e:
            return f"❌ Error: {str(e)}", gr.update()
        
        if result.get("success"):
            # Update agent list
            agent_list = self.get_saved_agents()
            return f"✅ {result.get('message')}", gr.update(choices=agent_list, value=name)
        else:
            return f"❌ Error: {result.get('error')}", gr.update()
    
    def delete_agent(self, agent_name: str) -> Tuple[str, str]:
        """Delete an agent. Returns (status_message, updated_agent_list_choices)."""
        if not agent_name or agent_name == "No agents saved":
            return "❌ Error: Please select an agent to delete", gr.update()
        
        try:
            result = self.manager.delete_agent(agent_name, delete_file=True)
        except Exception as e:
            return f"❌ Error: {str(e)}", gr.update()
        
        if result.get("success"):
            agent_list = self.get_saved_agents()
            return f"✅ Agent '{agent_name}' deleted successfully", gr.update(choices=agent_list, value=None)
        else:
            return f"❌ Error: {result.get('error')}", gr.update()
    
    def export_agent_json(self, agent_name: str) -> str:
        """Export agent configuration as JSON. Returns JSON string for display."""
        if not agent_name or agent_name == "No agents saved":
            return json.dumps({"error": "No agent selected"}, indent=2)
        
        try:
            agent_info = self.manager.get_agent_info(agent_name)
        except Exception as e:
            return json.dumps({"error": f"API error: {str(e)}"}, indent=2)
        
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
    
    def get_model_status(self) -> str:
        """Get current model loading status."""
        try:
            status_response = self.manager._request("GET", "/status")
            model_loaded = status_response.get("model_loaded", False)
            
            if model_loaded:
                current_model = status_response.get("current_model")
                gpu_stats = status_response.get("gpu_stats", {})
        except Exception as e:
            return f"❌ Error fetching model status: {str(e)}"
        
        if model_loaded:
            
            status = f"✅ **Model Loaded:** `{current_model}`\n\n"
            
            if gpu_stats.get("available"):
                status += f"**GPU:** {gpu_stats.get('name', 'Unknown')}\n"
                status += f"**Memory Usage:** {gpu_stats.get('usage_percent', 0):.1f}% "
                status += f"({gpu_stats.get('reserved_gb', 0):.2f}GB / {gpu_stats.get('total_gb', 0):.2f}GB)"
            
            return status
        else:
            return "⚠️ **No model loaded**\n\nPlease load a model before testing agents."
    
    def get_tool_info(self, tool_names: List[str]) -> str:
        """Get information about selected tools."""
        if not tool_names:
            return "No tools selected"
        
        info_parts = []
        for tool_name in tool_names:
            try:
                tool_info = self.manager.get_tool_info(tool_name)
            except Exception as e:
                continue
            if tool_info.get("exists"):
                info_parts.append(f"### {tool_name}")
                info_parts.append(f"**Description:** {tool_info.get('description', 'N/A')}")
                info_parts.append(f"**Strands Compatible:** {'✅' if tool_info.get('strands_compatible') else '❌'}")
                info_parts.append("")
        
        return "\n".join(info_parts) if info_parts else "No valid tools selected"
    
    def update_conv_manager_visibility(self, conv_type: str) -> Tuple:
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
    
    def create_ui(self):
        """Create the Agent Creator tab UI."""
        with gr.Row():
            # Left sidebar - Agent Library
            with gr.Column(scale=1):
                gr.Markdown("## 📚 Agent Library")
                
                agent_list = gr.Dropdown(
                    choices=self.get_saved_agents(),
                    label="Saved Agents",
                    interactive=True,
                    value=None
                )
                
                with gr.Row():
                    refresh_agents_btn = gr.Button("🔄", size="sm")
                    load_btn = gr.Button("📂 Load", size="sm")
                    delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                
                gr.Markdown("### 📝 Prompt Templates")
                gr.Markdown("*Model-specific prompt formats*")
                template_dropdown = gr.Dropdown(
                    choices=list(PROMPT_TEMPLATES.keys()),
                    label="Select Template",
                    value=None,
                    interactive=True
                )
                
                apply_template_btn = gr.Button("✨ Apply Template", size="sm")
                
                gr.Markdown("---")
                
                # Model Status
                gr.Markdown("### 🖥️ Model Status")
                model_status = gr.Markdown(self.get_model_status())
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
                        choices=self.get_available_models(),
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
                    
                    with gr.Row():
                        tools_dropdown = gr.Dropdown(
                            choices=self.get_available_tools(),
                            label="Select Tools",
                            multiselect=True,
                            info="Tools available to this agent",
                            scale=9
                        )
                        refresh_tools_btn = gr.Button("🔄", size="sm", scale=1)
                    
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
        
        # Apply prompt template
        apply_template_btn.click(
            fn=self.apply_prompt_template,
            inputs=[template_dropdown],
            outputs=[system_prompt]
        )
        
        # Load agent
        load_btn.click(
            fn=self.load_agent_config,
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
            fn=self.save_agent,
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
            fn=self.delete_agent,
            inputs=[agent_list],
            outputs=[status_message, agent_list]
        )
        
        # Export agent
        export_btn.click(
            fn=self.export_agent_json,
            inputs=[agent_list],
            outputs=[export_display]
        )
        
        # Refresh model status
        refresh_status_btn.click(
            fn=self.get_model_status,
            outputs=[model_status]
        )
        
        # Update tool info when selection changes
        tools_dropdown.change(
            fn=self.get_tool_info,
            inputs=[tools_dropdown],
            outputs=[tool_info_display]
        )
        
        # Update conversation manager visibility
        conv_manager_type.change(
            fn=self.update_conv_manager_visibility,
            inputs=[conv_manager_type],
            outputs=[window_size, truncate_results, summary_ratio, preserve_recent, custom_summary_prompt]
        )
        
        # Refresh agent list
        refresh_agents_btn.click(
            fn=lambda: gr.update(choices=self.get_saved_agents()),
            outputs=[agent_list]
        )
        
        # Refresh tools list
        refresh_tools_btn.click(
            fn=lambda: gr.update(choices=self.get_available_tools()),
            outputs=[tools_dropdown]
        )
        
        # Return components that might need refreshing from other tabs
        return {
            'agent_list': agent_list,
            'tools_dropdown': tools_dropdown
        }
