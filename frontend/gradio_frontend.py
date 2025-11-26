#!/usr/bin/env python3
"""
Gradio Frontend for GPU LLM Testing
A simple web interface to test all installed models with GPU acceleration.
"""

import os
from pathlib import Path
from typing import Tuple, List

import torch
import gradio as gr
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from model_manager import ModelManager
from templates import PROMPT_TEMPLATES
from utils import apply_prompt_template


# Configuration
MODELS_DIR = Path("/app/models")
HF_CACHE = Path(os.getenv("HF_HOME", "/app/cache"))

# Initialize model manager
model_manager = ModelManager(MODELS_DIR, HF_CACHE)


# Formatting helpers - convert data dictionaries to Markdown

def format_model_info(info: dict) -> str:
    """Format model info dictionary as Markdown."""
    if not info.get("exists"):
        return info.get("error", "No model selected")
    
    if "error" in info:
        return f"❌ {info['error']}"
    
    result = f"**Model:** {info['name']}\n\n"
    result += f"**Path:** `{info['path']}`\n\n"
    result += f"**Files:** {info['file_count']}\n\n"
    result += f"**Total Size:** {info['total_size_human']}\n\n"
    
    if info['has_config']:
        result += "✅ config.json found\n\n"
    if info['has_tokenizer']:
        result += "✅ tokenizer_config.json found\n\n"
    if info['safetensors_count'] > 0:
        result += f"✅ {info['safetensors_count']} safetensors file(s)\n\n"
    if info['bin_count'] > 0:
        result += f"✅ {info['bin_count']} .bin file(s)\n\n"
    
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
    result += f"**Reserved:** {stats['reserved_gb']:.2f} GB\n\n"
    result += f"**Free:** {stats['free_gb']:.2f} GB\n\n"
    result += f"**Usage:** {stats['usage_percent']:.1f}%"
    
    return result


# Wrapper functions that call model_manager and format results

def get_model_info(model_name: str) -> str:
    """Get and format model information."""
    info = model_manager.get_model_info(model_name)
    return format_model_info(info)


def get_gpu_memory_info() -> str:
    """Get and format GPU memory information."""
    stats = model_manager.get_gpu_stats()
    return format_gpu_stats(stats)


def load_model(model_name: str, quantization: str) -> Tuple[str, str, str]:
    """Load a model with specified quantization."""
    result = model_manager.load_model(model_name, quantization)
    
    # Build status message
    status_msg = ""
    
    if result.get("previous_model"):
        status_msg += f"🧹 Unloading previous model: {result['previous_model']}\n\n"
    
    if result["success"]:
        status_msg += f"⏳ Loading model: {result['model_name']}\n\n"
        status_msg += f"📊 Quantization: {result['quantization']}\n\n"
        status_msg += "✅ Model loaded successfully!\n\n"
        
        if result.get("device_map"):
            status_msg += f"📊 Device map: {result['device_map']}\n\n"
    else:
        status_msg += f"❌ {result.get('error', 'Unknown error')}\n\n"
    
    return status_msg, get_model_info(model_name), get_gpu_memory_info()


def unload_model() -> Tuple[str, str]:
    """Unload the current model and free GPU memory."""
    result = model_manager.unload_model()
    
    if result["success"]:
        status_msg = f"✅ Model unloaded: {result['previous_model']}\n\n"
        status_msg += "🧹 GPU memory cleared"
    else:
        status_msg = f"⚠️ {result['message']}"
    
    return status_msg, get_gpu_memory_info()


def generate_text(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float
):
    """Generate text using the loaded model with streaming."""
    if not model_manager.is_model_loaded():
        yield "⚠️ Please load a model first!"
        return
    
    if not prompt.strip():
        yield "⚠️ Please enter a prompt"
        return
    
    generated_text = ""
    
    for event in model_manager.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        skip_prompt=False
    ):
        if event["type"] == "error":
            yield f"❌ {event['error']}"
            return
        
        elif event["type"] == "token":
            generated_text = event["cumulative_text"]
            yield f"**Generated Text:**\n\n{generated_text}"
        
        elif event["type"] == "complete":
            # Final output with stats
            result = f"**Generated Text:**\n\n{event['text']}\n\n"
            result += "---\n\n"
            result += f"⏱️ **Time:** {event['elapsed_seconds']:.2f}s | "
            result += f"📝 **Tokens:** {event['total_tokens']} | "
            result += f"⚡ **Speed:** {event['tokens_per_second']:.2f} tokens/s"
            yield result


def chat_generate(
    message: str,
    history: list,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float
):
    """Generate response in chat interface with streaming."""
    if not model_manager.is_model_loaded():
        yield history + [(message, "⚠️ Please load a model first!")], ""
        return
    
    if not message.strip():
        yield history, message
        return
    
    # Build prompt from history
    prompt = ""
    for user_msg, assistant_msg in history:
        prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
    prompt += f"User: {message}\nAssistant:"
    
    assistant_response = ""
    
    for event in model_manager.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        skip_prompt=True
    ):
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





def create_interface():
    """Create the Gradio interface."""
    
    # Get available models
    available_models = model_manager.list_models()
    model_choices = ["None"] + available_models
    
    with gr.Blocks(title="GPU LLM Testing") as demo:
        gr.Markdown("# 🚀 GPU LLM Testing Interface")
        gr.Markdown("Test your installed language models with GPU acceleration and quantization")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Model Management Section
                gr.Markdown("## 📦 Model Management")
                
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value="None",
                    label="Select Model",
                    info="Choose a model from /app/models"
                )
                
                quantization_radio = gr.Radio(
                    choices=["4-bit (NF4)", "8-bit", "Full Precision (FP16)"],
                    value="4-bit (NF4)",
                    label="Quantization",
                    info="4-bit recommended for most GPUs"
                )
                
                with gr.Row():
                    load_btn = gr.Button("🔄 Load Model", variant="primary")
                    unload_btn = gr.Button("🗑️ Unload Model", variant="secondary")
                    refresh_btn = gr.Button("♻️ Refresh Models", variant="secondary")
                
                status_output = gr.Markdown("Ready to load a model")
                
            with gr.Column(scale=1):
                # Info panels
                gr.Markdown("## 📊 Model Info")
                model_info_output = gr.Markdown(get_model_info("None"))
                
                gr.Markdown("## 🎮 GPU Status")
                gpu_info_output = gr.Markdown(get_gpu_memory_info())
        
        # Tabs for different interfaces
        with gr.Tabs():
            # Simple Generation Tab
            with gr.Tab("📝 Text Generation"):
                gr.Markdown("### Single prompt generation")
                
                with gr.Row():
                    template_dropdown = gr.Dropdown(
                        choices=list(PROMPT_TEMPLATES.keys()),
                        value="None",
                        label="Prompt Template",
                        info="Select a template to format your prompt",
                        scale=2
                    )
                    apply_template_btn = gr.Button("📋 Apply Template", scale=1)
                
                raw_prompt_input = gr.Textbox(
                    label="Your Input",
                    placeholder="Enter your question or instruction here...",
                    lines=3
                )
                
                prompt_input = gr.Textbox(
                    label="Final Prompt (with template applied)",
                    placeholder="This will show the formatted prompt...",
                    lines=6
                )
                
                with gr.Accordion("⚙️ Generation Parameters", open=False):
                    with gr.Row():
                        max_tokens_gen = gr.Slider(10, 2048, value=200, step=10, label="Max Tokens")
                        temperature_gen = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                    with gr.Row():
                        top_p_gen = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                        top_k_gen = gr.Slider(1, 100, value=50, step=1, label="Top-k")
                    repetition_penalty_gen = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
                
                generate_btn = gr.Button("✨ Generate", variant="primary", size="lg")
                
                output_text = gr.Markdown(label="Output")
            
            # Chat Interface Tab
            with gr.Tab("💬 Chat Interface"):
                gr.Markdown("### Interactive chat with your model")
                
                chatbot = gr.Chatbot(height=400)
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    lines=2
                )
                
                with gr.Accordion("⚙️ Generation Parameters", open=False):
                    with gr.Row():
                        max_tokens_chat = gr.Slider(10, 1024, value=150, step=10, label="Max Tokens")
                        temperature_chat = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                    with gr.Row():
                        top_p_chat = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                        top_k_chat = gr.Slider(1, 100, value=50, step=1, label="Top-k")
                    repetition_penalty_chat = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
                
                with gr.Row():
                    submit_btn = gr.Button("📤 Send", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
        
        # Event handlers
        def refresh_models():
            models = model_manager.list_models()
            return gr.Dropdown(choices=["None"] + models)
        
        refresh_btn.click(
            fn=refresh_models,
            outputs=model_dropdown
        )
        
        load_btn.click(
            fn=load_model,
            inputs=[model_dropdown, quantization_radio],
            outputs=[status_output, model_info_output, gpu_info_output]
        )
        
        unload_btn.click(
            fn=unload_model,
            outputs=[status_output, gpu_info_output]
        )
        
        model_dropdown.change(
            fn=get_model_info,
            inputs=model_dropdown,
            outputs=model_info_output
        )
        
        # Template application handler
        apply_template_btn.click(
            fn=apply_prompt_template,
            inputs=[template_dropdown, raw_prompt_input],
            outputs=prompt_input
        )
        
        # Auto-apply template when raw prompt changes
        raw_prompt_input.change(
            fn=apply_prompt_template,
            inputs=[template_dropdown, raw_prompt_input],
            outputs=prompt_input
        )
        
        # Auto-apply template when template selection changes
        template_dropdown.change(
            fn=apply_prompt_template,
            inputs=[template_dropdown, raw_prompt_input],
            outputs=prompt_input
        )
        
        generate_btn.click(
            fn=generate_text,
            inputs=[
                prompt_input,
                max_tokens_gen,
                temperature_gen,
                top_p_gen,
                top_k_gen,
                repetition_penalty_gen
            ],
            outputs=output_text,
            show_progress="hidden"
        )
        
        msg.submit(
            fn=chat_generate,
            inputs=[
                msg,
                chatbot,
                max_tokens_chat,
                temperature_chat,
                top_p_chat,
                top_k_chat,
                repetition_penalty_chat
            ],
            outputs=[chatbot, msg],
            show_progress="hidden"
        )
        
        submit_btn.click(
            fn=chat_generate,
            inputs=[
                msg,
                chatbot,
                max_tokens_chat,
                temperature_chat,
                top_p_chat,
                top_k_chat,
                repetition_penalty_chat
            ],
            outputs=[chatbot, msg],
            show_progress="hidden"
        )
        
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, msg]
        )
    
    return demo


def main():
    """Main entry point."""
    print("=" * 80)
    print("GPU LLM Testing Interface")
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
    
    # Check models directory
    if MODELS_DIR.exists():
        models = model_manager.list_models()
        print(f"📦 Found {len(models)} model(s) in {MODELS_DIR}")
        for model in models:
            print(f"   - {model}")
    else:
        print(f"⚠️  Models directory not found: {MODELS_DIR}")
        print("   You can download models using the Jupyter notebook")
    print()
    
    print("🚀 Starting Gradio interface...")
    print("=" * 80)
    print()
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
