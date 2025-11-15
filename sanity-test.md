# Sanity Tests for GPU LLM Container

This file contains small, self-contained tests to verify that the container, GPU, and libraries work as expected.

---

## 1. Basic GPU and PyTorch Check

Run this inside the container:

```bash
python /app/test_env.py
```

You should see:

- CUDA version
- PyTorch / Transformers / Accelerate / Bitsandbytes versions
- `CUDA available: True`
- Your GPU name and memory

---

## 2. Manual PyTorch CUDA Sanity Check

Run:

```bash
python - << 'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))

x = torch.randn(1000, 1000, device="cuda")
print("Tensor on:", x.device)
PY
```

Expected:

- `CUDA available: True`
- Device name matches your GPU
- `Tensor on: cuda:0`

---

## 3. Small Public Model (GPT-2) on GPU

This test uses a public model that does not require authentication.

```bash
python - << 'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")

prompt = "Write a short sentence about GPUs:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(out[0], skip_special_tokens=True))
PY
```

If this runs and prints a coherent sentence, your basic Hugging Face + transformers + GPU path is working.

---

## 4. 4-bit Quantized LLM: `meta-llama/Meta-Llama-3-8B-Instruct`

This example uses `bitsandbytes` 4-bit loading and requires a Hugging Face account and token.

### 4.1. Hugging Face Account and Access

1. Create an account:  
   https://huggingface.co/join

2. On the model page, accept the license and request access:  
   https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

3. Create a read-access token:  
   https://huggingface.co/settings/tokens  
   Save the token in a secure place.

### 4.2. Using the Token in the Container

**Option A: Login interactively inside the container**

From inside the running container:

```bash
huggingface-cli login
# or, depending on the CLI version
hf auth login
```

Paste your token when prompted. The token is stored in the container's home directory and used automatically by `transformers`.

**Option B: Pass the token as an environment variable**

Start the container from the host with:

```bash
docker run --gpus all -it \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  llm-docker
```

Inside the container, `transformers` will pick up the token from the environment.

### 4.3. Run the 4-bit Llama 3 Inference

Assuming you have access and a token configured:

```bash
python - << 'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
)

prompt = "You are a helpful assistant. Explain what a Docker image is for."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
PY
```

If this runs and produces a reasonable explanation, your full 4-bit LLM stack (PyTorch, CUDA, bitsandbytes, transformers, Hugging Face authentication) is working correctly.
