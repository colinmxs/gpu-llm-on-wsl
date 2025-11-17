"""
Prompt templates for various LLM instruction formats.
"""

PROMPT_TEMPLATES = {
    "None": "",
    "Llama 3 - Chat": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    "Llama 3 - Instruction": """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    "Alpaca": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
""",
    "ChatML": """<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
""",
    "Vicuna": """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {prompt}
ASSISTANT:""",
    "Mistral Instruct": """<s>[INST] {prompt} [/INST]""",
    "Zephyr": """<|system|>
You are a helpful AI assistant.</s>
<|user|>
{prompt}</s>
<|assistant|>
""",
    "Plain": "{prompt}"
}
