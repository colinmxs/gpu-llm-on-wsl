"""
Agent Playground Tab for Gradio Frontend

This module provides a streaming chat interface for interacting with Strands agents.
Features real-time streaming, tool call visualization, and conversation history.
"""

import gradio as gr
import json
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

class AgentPlaygroundTab:
    """Manages the Agent Playground tab UI and functionality."""
    
    def __init__(self, manager):
        self.manager = manager
    
    def get_saved_agents(self) -> List[str]:
        """Get list of saved agents from API."""
        try:
            agents = self.manager.list_saved_agents()
            return agents if agents else ["No agents saved"]
        except Exception as e:
            print(f"Error fetching agents: {e}")
            return ["No agents saved"]
    
    def load_agent_for_chat(self, agent_name: str) -> Tuple[str, str, str]:
        """
        Load an agent for chatting and return its info.
        
        Returns:
            Tuple of (agent_info_md, system_prompt, status_message)
        """
        if not agent_name or agent_name == "No agents saved":
            return "No agent selected", "", "⚠️ Please select an agent"
        
        agent_info = self.manager.get_agent_info(agent_name)
        
        if not agent_info.get("exists"):
            return f"❌ Agent '{agent_name}' not found", "", f"❌ Agent not found"
        
        # Build info display
        info_lines = [
            f"## 🤖 {agent_info['name']}",
            f"**Description:** {agent_info.get('description', 'N/A')}",
            "",
            f"**Model:** `{agent_info.get('model_name', 'Unknown')}`",
            f"**Temperature:** {agent_info.get('temperature', 0.7)}",
            f"**Max Tokens:** {agent_info.get('max_tokens', 500)}",
            "",
        ]
        
        # Show tools
        tools = agent_info.get('tools', [])
        if tools:
            info_lines.append(f"**Tools ({len(tools)}):**")
            for tool in tools:
                info_lines.append(f"  • `{tool}`")
        else:
            info_lines.append("**Tools:** None")
        
        info_lines.append("")
        
        # Model status
        model_loaded = agent_info.get('model_loaded', False)
        try:
            status_response = self.manager._request("GET", "/status")
            current_model = status_response.get("current_model")
        except:
            current_model = None
        required_model = agent_info.get('model_name')
        
        if model_loaded:
            info_lines.append(f"✅ **Model Status:** Loaded and ready")
        else:
            info_lines.append(f"⚠️ **Model Status:** Not loaded")
            info_lines.append(f"   Current: `{current_model}`")
            info_lines.append(f"   Required: `{required_model}`")
            info_lines.append("   *Click 'Load Model' to load the required model*")
        
        agent_info_md = "\n".join(info_lines)
        system_prompt = agent_info.get('system_prompt', '')
        
        status = "✅ Agent loaded and ready!" if model_loaded else "⚠️ Agent loaded but model not ready"
        
        return agent_info_md, system_prompt, status
    
    def load_agent_model(self, agent_name: str) -> Tuple[str, str, str]:
        """
        Load the model required by the selected agent.
        
        Returns:
            Tuple of (agent_info_md, system_prompt, status_message)
        """
        if not agent_name or agent_name == "No agents saved":
            return "No agent selected", "", "⚠️ Please select an agent first"
        
        try:
            agent_info = self.manager.get_agent_info(agent_name)
        except Exception as e:
            return f"❌ Error: {str(e)}", "", "❌ API error"
        
        if not agent_info.get("exists"):
            return "❌ Agent not found", "", "❌ Agent not found"
        
        required_model = agent_info.get('model_name')
        if not required_model:
            return "❌ Agent has no model configured", "", "❌ No model configured"
        
        # Check if correct model is already loaded
        try:
            status_response = self.manager._request("GET", "/status")
            current_model = status_response.get("current_model")
        except:
            current_model = None
            
        if current_model == required_model:
            return self.load_agent_for_chat(agent_name)
        
        # Load the required model (default to 4-bit quantization)
        status_msg = f"🔄 Loading model `{required_model}`... This may take a minute..."
        
        # Attempt to load the model via API
        try:
            result = self.manager._request("POST", "/model/load", json={
                "model_name": required_model,
                "quantization": "4-bit (NF4)"
            })
        except Exception as e:
            return self.load_agent_for_chat(agent_name)[0], self.load_agent_for_chat(agent_name)[1], f"❌ Failed to load model: {str(e)}"
        
        if result.get("success"):
            # Reload agent info with updated model status
            return self.load_agent_for_chat(agent_name)[0], self.load_agent_for_chat(agent_name)[1], f"✅ Model `{required_model}` loaded successfully!"
        else:
            error_msg = result.get("error", "Unknown error")
            return self.load_agent_for_chat(agent_name)[0], self.load_agent_for_chat(agent_name)[1], f"❌ Failed to load model: {error_msg}"
    
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
            return "⚠️ **No model loaded**\n\nPlease load a model before chatting with agents."
    
    def format_tool_call_display(self, tool_name: str, tool_input: Dict[str, Any], tool_result: Any) -> str:
        """Format tool call information for display."""
        lines = [
            f"### 🛠️ Tool Call: `{tool_name}`",
            "",
            "**Input:**",
            "```json",
            json.dumps(tool_input, indent=2),
            "```",
            "",
            "**Result:**",
            "```",
            str(tool_result),
            "```"
        ]
        return "\n".join(lines)
    
    def chat_with_agent(
        self,
        message: str,
        history: List[Dict[str, str]],
        agent_name: str
    ):
        """
        Stream chat with the selected agent via API.
        
        This uses the API client's streaming implementation.
        
        Args:
            message: User message
            history: Chat history as list of (user, assistant) tuples
            agent_name: Name of the agent to chat with
            
        Yields:
            Updated history for Gradio Chatbot
        """
        if not agent_name or agent_name == "No agents saved":
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "❌ Error: No agent selected. Please select an agent first."})
            yield history
            return
        
        if not message.strip():
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "❌ Error: Empty message."})
            yield history
            return
        
        # Add user message to history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        
        # Initialize response accumulator
        response_text = ""
        tool_calls_info = []
        current_tool_call = None
        
        try:
            # Convert messages history to tuple format for agent_manager
            tuple_history = []
            for i in range(0, len(history)-2, 2):
                if i+1 < len(history):
                    tuple_history.append((history[i]["content"], history[i+1]["content"]))
            
            # Stream from agent - using local manager with sync streaming
            for event in self.manager.chat_with_agent_stream(agent_name, message, tuple_history):
                # Error handling
                if "error" in event:
                    error_msg = event.get("error", "Unknown error")
                    response_text = f"❌ Error: {error_msg}"
                    history[-1]["content"] = response_text
                    yield history
                    return
                
                # Text generation - Strands uses "data" key for text chunks
                if "data" in event:
                    delta = event["data"]
                    response_text += delta
                    history[-1]["content"] = response_text
                    yield history
                
                # Tool usage - Strands uses "current_tool_use" with name and input
                if "current_tool_use" in event:
                    tool_info = event["current_tool_use"]
                    tool_name = tool_info.get("name")
                    
                    if tool_name and (not current_tool_call or current_tool_call["name"] != tool_name):
                        # New tool call starting
                        current_tool_call = {
                            "name": tool_name,
                            "input": tool_info.get("input", {}),
                            "result": None
                        }
                        
                        # Show tool call in progress
                        tool_indicator = f"\n\n🛠️ *Calling tool: `{tool_name}`...*"
                        history[-1]["content"] = response_text + tool_indicator
                        yield history
                    elif tool_name and current_tool_call:
                        # Update input as it accumulates during streaming
                        current_tool_call["input"] = tool_info.get("input", {})
                
                # Completion event
                if event.get("complete", False):
                    # If we had a tool call, mark it as completed
                    if current_tool_call:
                        tool_calls_info.append(current_tool_call)
                        tool_indicator = f"\n\n✅ *Tool `{current_tool_call['name']}` completed*"
                        history[-1]["content"] = response_text + tool_indicator
                        current_tool_call = None
                        yield history
            
            # Final update with complete response
            final_response = response_text
            
            # Add tool calls summary if any
            if tool_calls_info:
                final_response += "\n\n---\n**Tool Calls:**\n"
                for i, tool_call in enumerate(tool_calls_info, 1):
                    final_response += f"\n{i}. `{tool_call['name']}`"
                    if tool_call['result']:
                        result_preview = str(tool_call['result'])[:100]
                        if len(str(tool_call['result'])) > 100:
                            result_preview += "..."
                        final_response += f" → {result_preview}"
            
            history[-1]["content"] = final_response
            yield history
            
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            history[-1]["content"] = error_msg
            yield history
    
    def clear_conversation(self) -> Tuple[List, str]:
        """Clear the conversation history."""
        return [], "🆕 Conversation cleared"
    
    def export_conversation(self, history: List[Dict[str, str]], agent_name: str) -> str:
        """Export conversation history as JSON."""
        if not history:
            return json.dumps({"message": "No conversation to export"}, indent=2)
        
        export_data = {
            "agent": agent_name,
            "conversation": history,
            "message_count": len(history)
        }
        
        return json.dumps(export_data, indent=2)
    
    def create_ui(self):
        """Create the Agent Playground tab UI."""
        gr.Markdown("### Chat with your Strands agents in real-time")
        
        with gr.Row():
            # Left sidebar - Agent selection and info
            with gr.Column(scale=1):
                gr.Markdown("## 🎮 Agent Selection")
                
                playground_agent_list = gr.Dropdown(
                    choices=self.get_saved_agents(),
                    label="Select Agent",
                    interactive=True,
                    value=None
                )
                
                with gr.Row():
                    refresh_playground_btn = gr.Button("🔄", size="sm")
                    load_agent_btn = gr.Button("📂 Load Agent", variant="primary", size="sm")
                
                load_model_btn = gr.Button("🖥️ Load Model", variant="secondary", size="sm")
                
                playground_status = gr.Markdown("⚠️ No agent loaded")
                
                gr.Markdown("---")
                
                # Agent info display
                gr.Markdown("## 📋 Agent Info")
                agent_info_display = gr.Markdown("*Select an agent to view details*")
                
                # System prompt display
                with gr.Accordion("💬 System Prompt", open=False):
                    system_prompt_display = gr.Textbox(
                        label="System Prompt",
                        lines=8,
                        interactive=False,
                        show_label=False
                    )
                
                gr.Markdown("---")
                
                # Model status
                gr.Markdown("## 🖥️ Model Status")
                playground_model_status = gr.Markdown(self.get_model_status())
                refresh_model_status_btn = gr.Button("🔄 Refresh Status", size="sm")
            
            # Main chat area
            with gr.Column(scale=3):
                gr.Markdown("## 💬 Chat Interface")
                
                # Chat display
                chatbot = gr.Chatbot(
                    [],
                    label="Conversation",
                    height=500,
                    show_label=False
                )
                
                # Input area
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        lines=3,
                        scale=9,
                        show_label=False
                    )
                    send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                
                # Action buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", size="sm")
                    export_btn = gr.Button("📤 Export", size="sm")
                    retry_btn = gr.Button("🔄 Retry Last", size="sm")
                
                # Export display
                with gr.Accordion("📄 Export Conversation", open=False):
                    export_display = gr.Code(language="json", label="Conversation JSON")
                
                # Tips
                with gr.Accordion("💡 Tips", open=False):
                    gr.Markdown("""
### Using the Agent Playground

1. **Select an Agent:** Choose from your saved agents in the sidebar
2. **Load the Agent:** Click "Load Agent" to prepare it for chatting
3. **Check Model Status:** Ensure the correct model is loaded for your agent
4. **Start Chatting:** Type your message and click Send or press Enter
5. **Watch Tool Calls:** See real-time updates when agents use tools
6. **Export Conversations:** Save your chat history as JSON

**Streaming Features:**
- 🔄 Real-time text streaming as the agent responds
- 🛠️ Live tool call notifications and results
- ✅ Completion indicators
- 📊 Tool usage summaries

**Pro Tips:**
- Load your agent's required model before starting a conversation
- Use the system prompt view to understand your agent's behavior
- Export conversations for analysis or record-keeping
- Clear history for a fresh start with the same agent
                    """)
        
        # Event Handlers
        
        # Load agent when selected
        load_agent_btn.click(
            fn=self.load_agent_for_chat,
            inputs=[playground_agent_list],
            outputs=[agent_info_display, system_prompt_display, playground_status]
        )
        
        # Load model for selected agent
        load_model_btn.click(
            fn=self.load_agent_model,
            inputs=[playground_agent_list],
            outputs=[agent_info_display, system_prompt_display, playground_status]
        )
        
        # Refresh agent list
        refresh_playground_btn.click(
            fn=lambda: gr.update(choices=self.get_saved_agents()),
            outputs=[playground_agent_list]
        )
        
        # Refresh model status
        refresh_model_status_btn.click(
            fn=self.get_model_status,
            outputs=[playground_model_status]
        )
        
        # Send message (button click)
        send_btn.click(
            fn=self.chat_with_agent,
            inputs=[msg_input, chatbot, playground_agent_list],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",  # Clear input after sending
            outputs=[msg_input]
        )
        
        # Send message (Enter key)
        msg_input.submit(
            fn=self.chat_with_agent,
            inputs=[msg_input, chatbot, playground_agent_list],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",  # Clear input after sending
            outputs=[msg_input]
        )
        
        # Clear conversation
        clear_btn.click(
            fn=self.clear_conversation,
            outputs=[chatbot, playground_status]
        )
        
        # Export conversation
        export_btn.click(
            fn=self.export_conversation,
            inputs=[chatbot, playground_agent_list],
            outputs=[export_display]
        )
        
        # Retry last message
        def retry_last(history):
            """Retry the last user message."""
            if not history or len(history) < 2:
                return history, ""
            # Get last user message and remove last exchange
            last_user_msg = ""
            for i in range(len(history)-1, -1, -1):
                if history[i]["role"] == "user":
                    last_user_msg = history[i]["content"]
                    history = history[:i]
                    break
            return history, last_user_msg
        
        retry_btn.click(
            fn=retry_last,
            inputs=[chatbot],
            outputs=[chatbot, msg_input]
        )
        
        # Return components that might need refreshing from other tabs
        return {
            'agent_list': playground_agent_list
        }
