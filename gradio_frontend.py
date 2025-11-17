#!/usr/bin/env python3
"""
Gradio Frontend for GPU LLM Testing
A simple web interface to test all installed models with GPU acceleration.
"""

import os
import gc
from pathlib import Path
from typing import Optional, Tuple, List
import threading

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import humanize


# Configuration
MODELS_DIR = Path("/app/models")
HF_CACHE = Path(os.getenv("HF_HOME", "/app/cache"))

# Global state
current_model = None
current_tokenizer = None
current_model_name = None
current_quantization = None


def get_available_models() -> List[str]:
    """Scan the models directory and return list of available models."""
    if not MODELS_DIR.exists():
        return []
    
    model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir()]
    # Convert back to readable format (e.g., "meta-llama--Llama-3.1-8B" -> "meta-llama/Llama-3.1-8B")
    return sorted([d.name.replace("--", "/", 1) for d in model_dirs])


def get_model_path(model_name: str) -> Path:
    """Convert model name to file system path."""
    # Convert "meta-llama/Llama-3.1-8B" -> "meta-llama--Llama-3.1-8B"
    safe_name = model_name.replace("/", "--")
    return MODELS_DIR / safe_name


def get_model_info(model_name: str) -> str:
    """Get information about a model."""
    if not model_name or model_name == "None":
        return "No model selected"
    
    model_path = get_model_path(model_name)
    
    if not model_path.exists():
        return f"Model path not found: {model_path}"
    
    # Count files and calculate size
    file_count = sum(1 for _ in model_path.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    
    info = f"**Model:** {model_name}\n\n"
    info += f"**Path:** `{model_path}`\n\n"
    info += f"**Files:** {file_count}\n\n"
    info += f"**Total Size:** {humanize.naturalsize(total_size, binary=True)}\n\n"
    
    # Check for specific files
    config_file = model_path / "config.json"
    if config_file.exists():
        info += "✅ config.json found\n\n"
    
    tokenizer_file = model_path / "tokenizer_config.json"
    if tokenizer_file.exists():
        info += "✅ tokenizer_config.json found\n\n"
    
    # Check for model files
    safetensors_files = list(model_path.glob("*.safetensors"))
    bin_files = list(model_path.glob("*.bin"))
    
    if safetensors_files:
        info += f"✅ {len(safetensors_files)} safetensors file(s)\n\n"
    if bin_files:
        info += f"✅ {len(bin_files)} .bin file(s)\n\n"
    
    return info


def get_gpu_memory_info() -> str:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return "❌ CUDA not available"
    
    info = f"**GPU:** {torch.cuda.get_device_name(0)}\n\n"
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
    free_gb = total_gb - reserved_gb
    
    info += f"**Total:** {total_gb:.2f} GB\n\n"
    info += f"**Allocated:** {allocated_gb:.2f} GB\n\n"
    info += f"**Reserved:** {reserved_gb:.2f} GB\n\n"
    info += f"**Free:** {free_gb:.2f} GB\n\n"
    
    # Calculate percentage
    usage_percent = (reserved_gb / total_gb) * 100
    info += f"**Usage:** {usage_percent:.1f}%"
    
    return info


def load_model(model_name: str, quantization: str) -> Tuple[str, str, str]:
    """Load a model with specified quantization."""
    global current_model, current_tokenizer, current_model_name, current_quantization
    
    if not model_name or model_name == "None":
        return "⚠️ Please select a model", get_model_info("None"), get_gpu_memory_info()
    
    model_path = get_model_path(model_name)
    
    if not model_path.exists():
        return f"❌ Model not found: {model_path}", get_model_info(model_name), get_gpu_memory_info()
    
    # Unload existing model if any
    if current_model is not None:
        status_msg = f"🧹 Unloading previous model: {current_model_name}\n\n"
        del current_model
        del current_tokenizer
        current_model = None
        current_tokenizer = None
        current_model_name = None
        current_quantization = None
        gc.collect()
        torch.cuda.empty_cache()
    else:
        status_msg = ""
    
    status_msg += f"⏳ Loading model: {model_name}\n\n"
    status_msg += f"📊 Quantization: {quantization}\n\n"
    
    try:
        # Load tokenizer
        status_msg += "🔧 Loading tokenizer...\n\n"
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        
        # Configure quantization
        status_msg += "🔧 Loading model...\n\n"
        
        if quantization == "4-bit (NF4)":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                quantization_config=bnb_config,
                device_map="auto",
                local_files_only=True,
                torch_dtype=torch.float16
            )
        elif quantization == "8-bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                quantization_config=bnb_config,
                device_map="auto",
                local_files_only=True
            )
        else:  # Full precision
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                local_files_only=True,
                torch_dtype=torch.float16
            )
        
        current_model = model
        current_tokenizer = tokenizer
        current_model_name = model_name
        current_quantization = quantization
        
        status_msg += "✅ Model loaded successfully!\n\n"
        status_msg += f"📊 Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}\n\n"
        
        return status_msg, get_model_info(model_name), get_gpu_memory_info()
        
    except Exception as e:
        error_msg = f"❌ Failed to load model: {str(e)}"
        return error_msg, get_model_info(model_name), get_gpu_memory_info()


def unload_model() -> Tuple[str, str]:
    """Unload the current model and free GPU memory."""
    global current_model, current_tokenizer, current_model_name, current_quantization
    
    if current_model is None:
        return "⚠️ No model is currently loaded", get_gpu_memory_info()
    
    model_name = current_model_name
    
    del current_model
    del current_tokenizer
    current_model = None
    current_tokenizer = None
    current_model_name = None
    current_quantization = None
    
    gc.collect()
    torch.cuda.empty_cache()
    
    status_msg = f"✅ Model unloaded: {model_name}\n\n"
    status_msg += "🧹 GPU memory cleared"
    
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
    global current_model, current_tokenizer, current_model_name
    
    if current_model is None or current_tokenizer is None:
        yield "⚠️ Please load a model first!"
        return
    
    if not prompt.strip():
        yield "⚠️ Please enter a prompt"
        return
    
    try:
        # Tokenize input
        inputs = current_tokenizer(prompt, return_tensors="pt").to(current_model.device)
        
        # Setup streaming
        import time
        start_time = time.time()
        
        streamer = TextIteratorStreamer(
            current_tokenizer,
            skip_prompt=False,
            skip_special_tokens=True
        )
        
        # Generation parameters
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=current_tokenizer.eos_token_id,
            streamer=streamer
        )
        
        # Start generation in a separate thread
        thread = threading.Thread(target=current_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the output
        generated_text = ""
        num_tokens = 0
        
        for new_text in streamer:
            generated_text += new_text
            num_tokens += 1
            
            # Yield with formatted output
            result = f"**Generated Text:**\n\n{generated_text}"
            yield result
        
        # Wait for thread to complete
        thread.join()
        
        elapsed = time.time() - start_time
        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
        
        # Final output with stats
        result = f"**Generated Text:**\n\n{generated_text}\n\n"
        result += "---\n\n"
        result += f"⏱️ **Time:** {elapsed:.2f}s | "
        result += f"📝 **Tokens:** {num_tokens} | "
        result += f"⚡ **Speed:** {tokens_per_sec:.2f} tokens/s"
        
        yield result
        
    except Exception as e:
        yield f"❌ Generation failed: {str(e)}"


def chat_generate(
    message: str,
    history: List[Tuple[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float
):
    """Generate response in chat interface with streaming."""
    global current_model, current_tokenizer, current_model_name
    
    if current_model is None or current_tokenizer is None:
        yield history + [(message, "⚠️ Please load a model first!")], ""
        return
    
    if not message.strip():
        yield history, message
        return
    
    try:
        # Build prompt from history
        prompt = ""
        for user_msg, assistant_msg in history:
            prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
        prompt += f"User: {message}\nAssistant:"
        
        # Tokenize
        inputs = current_tokenizer(prompt, return_tensors="pt").to(current_model.device)
        
        # Setup streaming
        import time
        start_time = time.time()
        
        streamer = TextIteratorStreamer(
            current_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generation parameters
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=current_tokenizer.eos_token_id,
            streamer=streamer
        )
        
        # Start generation in a separate thread
        thread = threading.Thread(target=current_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the output
        assistant_response = ""
        num_tokens = 0
        
        for new_text in streamer:
            assistant_response += new_text
            num_tokens += 1
            
            # Yield intermediate result
            yield history + [(message, assistant_response)], ""
        
        # Wait for thread to complete
        thread.join()
        
        elapsed = time.time() - start_time
        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
        
        # Add stats to final response
        response_with_stats = assistant_response + f"\n\n*[{num_tokens} tokens, {tokens_per_sec:.1f} tok/s, {elapsed:.2f}s]*"
        
        yield history + [(message, response_with_stats)], ""
        
    except Exception as e:
        yield history + [(message, f"❌ Error: {str(e)}")], ""


def create_interface():
    """Create the Gradio interface."""
    
    # Get available models
    available_models = get_available_models()
    model_choices = ["None"] + available_models
    
    with gr.Blocks(title="GPU LLM Testing", theme=gr.themes.Soft()) as demo:
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
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=4
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
            models = get_available_models()
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
        models = get_available_models()
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
