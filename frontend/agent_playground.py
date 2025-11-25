#!/usr/bin/env python3
"""
Agent Playground - Gradio interface for creating and testing Strands SDK agents.
A dedicated interface to build, test, and manage AI agents with custom configurations.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import gradio as gr

from model_manager import ModelManager
from agent_manager import AgentManager, AgentConfig


# Configuration
MODELS_DIR = Path("/app/models")
AGENTS_DIR = Path("/app/agents")
HF_CACHE = Path(os.getenv("HF_HOME", "/app/cache"))

# Initialize managers
model_manager = ModelManager(MODELS_DIR, HF_CACHE)
agent_manager = AgentManager(AGENTS_DIR, model_manager)


# Helper functions

def format_agent_info(info: dict) -> str:
    """Format agent info dictionary as Markdown."""
    if not info.get("exists"):
        return info.get("error", "No agent selected")
    
    result = f"## 🤖 {info['name']}\n\n"
    result += f"**Description:** {info['description']}\n\n"
    result += f"**Model:** {info['model_name']}\n\n"
    
    if info['model_loaded']:
        result += "✅ Model is loaded\n\n"
    else:
        result += "⚠️ Model not loaded\n\n"
    
    result += "### Configuration\n\n"
    result += f"- **Temperature:** {info['temperature']}\n"
    result += f"- **Max Tokens:** {info['max_tokens']}\n"
    result += f"- **Top-p:** {info['top_p']}\n"
    result += f"- **Top-k:** {info['top_k']}\n"
    result += f"- **Repetition Penalty:** {info['repetition_penalty']}\n\n"
    
    result += "### System Prompt\n\n"
    result += f"```\n{info['system_prompt']}\n```\n\n"
    
    if info['is_saved']:
        result += "💾 Agent is saved to disk\n"
    else:
        result += "⚠️ Agent not saved (in memory only)\n"
    
    return result


def format_gpu_stats(stats: dict) -> str:
    """Format GPU stats dictionary as Markdown."""
    if not stats.get("available"):
        return f"❌ {stats.get('error', 'CUDA not available')}"
    
    if "error" in stats:
        return f"❌ {stats['error']}"
    
    result = f"**GPU:** {stats['name']}\n\n"
    result += f"**Total:** {stats['total_gb']:.2f} GB\n\n"
    result += f"**Allocated:** {stats['allocated_gb']:.2f} GB\n\n"
    result += f"**Free:** {stats['free_gb']:.2f} GB\n\n"
    result += f"**Usage:** {stats['usage_percent']:.1f}%"
    
    return result


# UI Action Handlers

def create_new_agent(
    name: str,
    description: str,
    system_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    repetition_penalty: float
) -> Tuple[str, str, str]:
    """Create a new agent with the specified configuration."""
    if not name.strip():
        return "❌ Agent name is required", "", ""
    
    if not model_name or model_name == "None":
        return "❌ Please select a model", "", ""
    
    config = AgentConfig(
        name=name.strip(),
        description=description.strip(),
        system_prompt=system_prompt.strip(),
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty
    )
    
    result = agent_manager.create_agent(config)
    
    if result["success"]:
        status_msg = f"✅ {result['message']}\n\n"
        status_msg += "ℹ️ Agent created in memory. Use 'Save Agent' to persist to disk."
        
        # Update agent list dropdown
        active_agents = agent_manager.list_active_agents()
        agent_dropdown_update = gr.Dropdown(choices=["None"] + active_agents, value=name)
        
        return status_msg, name, agent_dropdown_update
    else:
        return f"❌ {result['error']}", "", gr.Dropdown()


def save_agent_config(agent_name: str) -> str:
    """Save an agent configuration to disk."""
    if not agent_name or agent_name == "None":
        return "❌ Please select an agent to save"
    
    result = agent_manager.save_agent(agent_name)
    
    if result["success"]:
        return f"✅ {result['message']}\n\n📁 Saved to: `{result['filepath']}`"
    else:
        return f"❌ {result['error']}"


def load_agent_config(agent_name: str) -> Tuple[str, str, str, str, str, float, int, float, int, float, str]:
    """Load an agent configuration from disk."""
    if not agent_name or agent_name == "None":
        return ("❌ Please select an agent to load", "", "", "", "", 0.7, 500, 0.9, 50, 1.1, "")
    
    result = agent_manager.load_agent(agent_name)
    
    if result["success"]:
        config = result["config"]
        status_msg = f"✅ {result['message']}"
        
        # Return all config fields to populate the form
        return (
            status_msg,
            config.name,
            config.description,
            config.system_prompt,
            config.model_name,
            config.temperature,
            config.max_tokens,
            config.top_p,
            config.top_k,
            config.repetition_penalty,
            config.name  # For agent selector dropdown
        )
    else:
        return (f"❌ {result['error']}", "", "", "", "", 0.7, 500, 0.9, 50, 1.1, "")


def delete_agent_handler(agent_name: str, delete_file: bool) -> Tuple[str, str]:
    """Delete an agent from memory and optionally from disk."""
    if not agent_name or agent_name == "None":
        return "❌ Please select an agent to delete", gr.Dropdown()
    
    result = agent_manager.delete_agent(agent_name, delete_file)
    
    if result["success"]:
        active_agents = agent_manager.list_active_agents()
        agent_dropdown_update = gr.Dropdown(choices=["None"] + active_agents, value="None")
        return f"✅ {result['message']}", agent_dropdown_update
    else:
        return f"❌ {result['error']}", gr.Dropdown()


def get_agent_info_display(agent_name: str) -> str:
    """Get and format agent information for display."""
    if not agent_name or agent_name == "None":
        return "No agent selected"
    
    info = agent_manager.get_agent_info(agent_name)
    return format_agent_info(info)


def chat_with_agent_handler(
    agent_name: str,
    message: str,
    history: List[Tuple[str, str]]
):
    """Handle chat interaction with an agent."""
    if not agent_name or agent_name == "None":
        yield history + [(message, "⚠️ Please select an agent first!")], ""
        return
    
    if not message.strip():
        yield history, message
        return
    
    assistant_response = ""
    
    for event in agent_manager.chat_with_agent(agent_name, message, history):
        if event["type"] == "error":
            yield history + [(message, f"❌ {event['error']}")], ""
            return
        
        elif event["type"] == "token":
            assistant_response = event["cumulative_text"]
            yield history + [(message, assistant_response)], ""
        
        elif event["type"] == "complete":
            # Add stats to final response
            response_with_stats = event['text'] + f"\n\n*[{event['total_tokens']} tokens, {event['tokens_per_second']:.1f} tok/s, {event['elapsed_seconds']:.2f}s]*"
            yield history + [(message, response_with_stats)], ""


def load_model_for_agent(agent_name: str, quantization: str) -> Tuple[str, str]:
    """Load the model required by the selected agent."""
    if not agent_name or agent_name == "None":
        return "❌ Please select an agent first", ""
    
    config = agent_manager.get_agent_config(agent_name)
    if not config:
        return f"❌ Agent '{agent_name}' not found", ""
    
    model_name = config.model_name
    result = model_manager.load_model(model_name, quantization)
    
    status_msg = ""
    if result.get("previous_model"):
        status_msg += f"🧹 Unloading previous model: {result['previous_model']}\n\n"
    
    if result["success"]:
        status_msg += f"⏳ Loading model for agent '{agent_name}': {model_name}\n\n"
        status_msg += f"📊 Quantization: {result['quantization']}\n\n"
        status_msg += "✅ Model loaded successfully!\n\n"
    else:
        status_msg += f"❌ {result.get('error', 'Unknown error')}\n\n"
    
    return status_msg, format_gpu_stats(model_manager.get_gpu_stats())


def unload_current_model() -> Tuple[str, str]:
    """Unload the currently loaded model."""
    result = model_manager.unload_model()
    
    if result["success"]:
        status_msg = f"✅ Model unloaded: {result['previous_model']}\n\n"
        status_msg += "🧹 GPU memory cleared"
    else:
        status_msg = f"⚠️ {result['message']}"
    
    return status_msg, format_gpu_stats(model_manager.get_gpu_stats())


def refresh_all_lists() -> Tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown]:
    """Refresh all dropdown lists (models, active agents, saved agents)."""
    models = model_manager.list_models()
    active_agents = agent_manager.list_active_agents()
    saved_agents = agent_manager.list_saved_agents()
    
    return (
        gr.Dropdown(choices=["None"] + models),
        gr.Dropdown(choices=["None"] + active_agents),
        gr.Dropdown(choices=["None"] + saved_agents)
    )


def create_interface():
    """Create the Agent Playground Gradio interface."""
    
    # Get available options
    available_models = model_manager.list_models()
    model_choices = ["None"] + available_models
    active_agents = agent_manager.list_active_agents()
    saved_agents = agent_manager.list_saved_agents()
    
    with gr.Blocks(title="Agent Playground", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎮 Agent Playground")
        gr.Markdown("Build, test, and manage Strands SDK agents with custom configurations")
        
        with gr.Tabs():
            # Agent Builder Tab
            with gr.Tab("🛠️ Agent Builder"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("## Create or Edit Agent")
                        
                        agent_name_input = gr.Textbox(
                            label="Agent Name",
                            placeholder="my-assistant",
                            info="Unique identifier for this agent"
                        )
                        
                        agent_description_input = gr.Textbox(
                            label="Description",
                            placeholder="A helpful AI assistant that...",
                            lines=2,
                            info="Brief description of the agent's purpose"
                        )
                        
                        system_prompt_input = gr.Textbox(
                            label="System Prompt",
                            placeholder="You are a helpful AI assistant...",
                            lines=6,
                            info="Instructions that define the agent's behavior and personality"
                        )
                        
                        model_selector = gr.Dropdown(
                            choices=model_choices,
                            value="None",
                            label="Model",
                            info="Select the LLM model for this agent"
                        )
                        
                        gr.Markdown("### Generation Parameters")
                        
                        with gr.Row():
                            temperature_input = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                            max_tokens_input = gr.Slider(10, 2048, value=500, step=10, label="Max Tokens")
                        
                        with gr.Row():
                            top_p_input = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                            top_k_input = gr.Slider(1, 100, value=50, step=1, label="Top-k")
                        
                        repetition_penalty_input = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
                        
                        with gr.Row():
                            create_btn = gr.Button("🆕 Create Agent", variant="primary", size="lg")
                            refresh_btn = gr.Button("♻️ Refresh Lists", variant="secondary")
                        
                        builder_status = gr.Markdown("Ready to create an agent")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("## 💾 Save/Load")
                        
                        active_agent_selector = gr.Dropdown(
                            choices=["None"] + active_agents,
                            value="None",
                            label="Active Agents",
                            info="Agents in memory"
                        )
                        
                        save_btn = gr.Button("💾 Save Agent", variant="primary")
                        
                        saved_agent_selector = gr.Dropdown(
                            choices=["None"] + saved_agents,
                            value="None",
                            label="Saved Agents",
                            info="Agents saved to disk"
                        )
                        
                        load_btn = gr.Button("📂 Load Agent", variant="primary")
                        
                        gr.Markdown("## 🗑️ Delete")
                        
                        delete_file_checkbox = gr.Checkbox(
                            label="Also delete saved file",
                            value=False,
                            info="Remove from disk permanently"
                        )
                        
                        delete_btn = gr.Button("🗑️ Delete Agent", variant="stop")
                        
                        save_load_status = gr.Markdown("")
            
            # Agent Testing Tab
            with gr.Tab("🧪 Agent Testing"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("## Test Your Agent")
                        
                        test_agent_selector = gr.Dropdown(
                            choices=["None"] + active_agents,
                            value="None",
                            label="Select Agent to Test",
                            info="Choose an agent from active agents"
                        )
                        
                        agent_info_display = gr.Markdown("No agent selected")
                        
                        with gr.Row():
                            quantization_radio = gr.Radio(
                                choices=["4-bit (NF4)", "8-bit", "Full Precision (FP16)"],
                                value="4-bit (NF4)",
                                label="Quantization",
                                info="4-bit recommended"
                            )
                        
                        with gr.Row():
                            load_model_btn = gr.Button("🔄 Load Model", variant="primary")
                            unload_model_btn = gr.Button("🗑️ Unload Model", variant="secondary")
                        
                        model_status = gr.Markdown("")
                        
                        gr.Markdown("### 💬 Chat with Agent")
                        
                        chatbot = gr.Chatbot(
                            height=400,
                            label="Agent Response",
                            show_label=True
                        )
                        
                        chat_input = gr.Textbox(
                            label="Your Message",
                            placeholder="Type your message here...",
                            lines=3
                        )
                        
                        with gr.Row():
                            send_btn = gr.Button("📤 Send", variant="primary", size="lg")
                            clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("## 🎮 GPU Status")
                        gpu_status_display = gr.Markdown(format_gpu_stats(model_manager.get_gpu_stats()))
                        
                        gr.Markdown("---")
                        
                        gr.Markdown("## ℹ️ Tips")
                        gr.Markdown("""
                        **Getting Started:**
                        1. Select an agent to test
                        2. Load the required model
                        3. Start chatting!
                        
                        **Multi-Agent Future:**
                        This interface is designed to be extensible for multi-agent collaboration scenarios.
                        """)
        
        # Event Handlers
        
        # Builder Tab Events
        create_btn.click(
            fn=create_new_agent,
            inputs=[
                agent_name_input,
                agent_description_input,
                system_prompt_input,
                model_selector,
                temperature_input,
                max_tokens_input,
                top_p_input,
                top_k_input,
                repetition_penalty_input
            ],
            outputs=[builder_status, agent_name_input, active_agent_selector]
        ).then(
            fn=refresh_all_lists,
            outputs=[model_selector, active_agent_selector, saved_agent_selector]
        ).then(
            fn=refresh_all_lists,
            outputs=[model_selector, test_agent_selector, saved_agent_selector]
        )
        
        save_btn.click(
            fn=save_agent_config,
            inputs=[active_agent_selector],
            outputs=[save_load_status]
        ).then(
            fn=refresh_all_lists,
            outputs=[model_selector, saved_agent_selector, active_agent_selector]
        )
        
        load_btn.click(
            fn=load_agent_config,
            inputs=[saved_agent_selector],
            outputs=[
                save_load_status,
                agent_name_input,
                agent_description_input,
                system_prompt_input,
                model_selector,
                temperature_input,
                max_tokens_input,
                top_p_input,
                top_k_input,
                repetition_penalty_input,
                active_agent_selector
            ]
        ).then(
            fn=refresh_all_lists,
            outputs=[model_selector, active_agent_selector, test_agent_selector]
        )
        
        delete_btn.click(
            fn=delete_agent_handler,
            inputs=[active_agent_selector, delete_file_checkbox],
            outputs=[save_load_status, active_agent_selector]
        ).then(
            fn=refresh_all_lists,
            outputs=[model_selector, active_agent_selector, test_agent_selector]
        )
        
        refresh_btn.click(
            fn=refresh_all_lists,
            outputs=[model_selector, active_agent_selector, saved_agent_selector]
        ).then(
            fn=refresh_all_lists,
            outputs=[model_selector, test_agent_selector, saved_agent_selector]
        )
        
        # Testing Tab Events
        test_agent_selector.change(
            fn=get_agent_info_display,
            inputs=[test_agent_selector],
            outputs=[agent_info_display]
        )
        
        load_model_btn.click(
            fn=load_model_for_agent,
            inputs=[test_agent_selector, quantization_radio],
            outputs=[model_status, gpu_status_display]
        )
        
        unload_model_btn.click(
            fn=unload_current_model,
            outputs=[model_status, gpu_status_display]
        )
        
        chat_input.submit(
            fn=chat_with_agent_handler,
            inputs=[test_agent_selector, chat_input, chatbot],
            outputs=[chatbot, chat_input],
            show_progress="hidden"
        )
        
        send_btn.click(
            fn=chat_with_agent_handler,
            inputs=[test_agent_selector, chat_input, chatbot],
            outputs=[chatbot, chat_input],
            show_progress="hidden"
        )
        
        clear_chat_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, chat_input]
        )
    
    return demo


def main():
    """Main entry point."""
    print("=" * 80)
    print("Agent Playground - Strands SDK Agent Builder")
    print("=" * 80)
    print()
    
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"✅ VRAM: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  WARNING: CUDA not available!")
    print()
    
    # Check directories
    if MODELS_DIR.exists():
        models = model_manager.list_models()
        print(f"📦 Found {len(models)} model(s) in {MODELS_DIR}")
        for model in models:
            print(f"   - {model}")
    else:
        print(f"⚠️  Models directory not found: {MODELS_DIR}")
    print()
    
    if AGENTS_DIR.exists():
        saved_agents = agent_manager.list_saved_agents()
        print(f"🤖 Found {len(saved_agents)} saved agent(s) in {AGENTS_DIR}")
        for agent in saved_agents:
            print(f"   - {agent}")
    else:
        print(f"📁 Creating agents directory: {AGENTS_DIR}")
        AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    print()
    
    print("🚀 Starting Agent Playground...")
    print("=" * 80)
    print()
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from main frontend
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
