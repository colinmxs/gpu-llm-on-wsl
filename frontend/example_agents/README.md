# Example Agents

This directory contains example agent configurations to help you get started with the Agent Playground.

## Available Examples

### 1. helpful-assistant
A general-purpose AI assistant configured for balanced, helpful responses.
- **Temperature**: 0.7 (balanced creativity)
- **Use Case**: General questions, advice, information lookup

### 2. code-assistant
A specialized coding assistant optimized for Python programming and debugging.
- **Temperature**: 0.3 (more focused/deterministic)
- **Use Case**: Code generation, debugging, best practices

### 3. creative-writer
A creative writing assistant for storytelling and content creation.
- **Temperature**: 0.9 (high creativity)
- **Use Case**: Story writing, brainstorming, creative content

## How to Use

### Option 1: Copy to Agents Directory
Copy any example agent to your mounted agents directory:
```bash
# On your host machine (Windows)
copy frontend\example_agents\helpful-assistant.json C:\path\to\agents\
```

Then load it in the Agent Playground UI.

### Option 2: Create New Agent from Template
1. Open an example JSON file
2. Use it as a template in the Agent Builder
3. Modify the name, system prompt, and parameters
4. Create and save your custom agent

## Customization Tips

### Temperature
- **0.1-0.3**: Focused, deterministic (good for code, facts)
- **0.4-0.7**: Balanced (good for general use)
- **0.8-1.0**: Creative, varied (good for writing, brainstorming)

### System Prompt
The system prompt is the most important part of an agent. It defines:
- The agent's role and expertise
- How it should respond (tone, style, format)
- Any constraints or guidelines

**Good System Prompt Example:**
```
You are a Python expert specializing in data science. You provide clear explanations with code examples. You always consider performance and best practices. When suggesting code, you explain why you chose that approach.
```

### Max Tokens
- **200-500**: Short, concise responses
- **500-1000**: Medium-length explanations
- **1000+**: Detailed, comprehensive responses

## Model Compatibility

All example agents use `meta-llama/Llama-2-7b-chat-hf` as the default model. You can change the `model_name` field to any model you have downloaded:
- `mistralai/Mistral-7B-Instruct-v0.2`
- `microsoft/phi-2`
- Any other compatible model in your `/app/models` directory

Just update the JSON file or change it in the UI when creating/editing an agent.
