"""
Gradio Frontend for Strands Agent Creator & Tool Creator

This module provides a comprehensive UI for creating and managing Strands SDK agents and tools.
Includes both Agent Creator and Tool Creator in a single tabbed interface.
"""

import gradio as gr
import sys
import ast
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.model_manager import ModelManager
from strands_agents.agent_manager import AgentManager
from strands_agents.tool_manager import ToolManager, ToolConfig


# Initialize managers
MODELS_DIR = Path(__file__).parent.parent / "models"
AGENTS_DIR = Path(__file__).parent.parent / "strands_agents" / "examples" / "agents"
TOOLS_DIR = Path(__file__).parent.parent / "strands_agents" / "examples" / "tools"

model_manager = ModelManager(models_dir=MODELS_DIR)
agent_manager = AgentManager(agents_dir=AGENTS_DIR, model_manager=model_manager, tools_dir=TOOLS_DIR)
tool_manager = ToolManager(tools_dir=TOOLS_DIR)


PROMPT_TEMPLATES = {
    "Llama 3": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    
    "Mistral Instruct": """<s>[INST] You are a helpful AI assistant.

{user_message} [/INST]""",
    
    "Cohere Command": """<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a helpful AI assistant.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_message}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>""",
}

# Tool Creator type annotations
TYPE_ANNOTATIONS = [
    "str", "int", "float", "bool", "list", "dict", "Any",
    "Optional[str]", "Optional[int]", "Optional[float]", "Optional[bool]",
    "List[str]", "List[int]", "Dict[str, Any]",
]

# Tool Creator code templates
CODE_TEMPLATES = {
    "Basic Template": """try:
    # Your implementation here
    result = "Hello, World!"
    return result
except Exception as e:
    return f"Error: {str(e)}"
""",
    "Safe Math Evaluation": """import ast
import operator

operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow
}

def eval_expr(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
    else:
        raise ValueError(f"Unsupported operation: {type(node)}")

try:
    tree = ast.parse(expression, mode='eval')
    result = eval_expr(tree.body)
    return str(result)
except Exception as e:
    return f"Error: {str(e)}"
""",
    "String Processing": """try:
    result = text.strip().lower()
    result = result.replace(old_value, new_value)
    return result
except Exception as e:
    return f"Error: {str(e)}"
""",
    "JSON Data Processing": """import json

try:
    data = json.loads(json_string)
    result = {"processed": True, "data": data}
    return json.dumps(result, indent=2)
except json.JSONDecodeError as e:
    return f"Invalid JSON: {str(e)}"
except Exception as e:
    return f"Error: {str(e)}"
""",
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


def apply_prompt_template(template_name: str) -> str:
    """
    Apply a prompt template.
    
    Returns the prompt template text.
    """
    return PROMPT_TEMPLATES.get(template_name, "")


def load_agent_config(agent_name: str) -> Tuple:
    """
    Load an agent configuration and return all field values.
    
    Returns tuple of all configuration values.
    """
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
    
    agent_info = agent_manager.get_agent_info(agent_name)
    
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


# ============================================================================
# TOOL CREATOR FUNCTIONS
# ============================================================================

def to_snake_case(text: str) -> str:
    """Convert text to snake_case for valid Python identifiers."""
    text = re.sub(r'[^a-zA-Z0-9_]', '_', text)
    text = re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()
    text = re.sub(r'_+', '_', text)
    return text.strip('_')


def refresh_tool_list_tc() -> gr.Dropdown:
    """Refresh tool list for tool creator."""
    tools = tool_manager.list_saved_tools()
    return gr.Dropdown(choices=tools, value=None)


def auto_populate_function_name(tool_name: str) -> str:
    """Auto-generate function name from tool name."""
    return to_snake_case(tool_name) if tool_name else ""


def generate_function_skeleton(function_name: str, parameters_json: str, return_type: str, 
                               enable_context: bool, context_param_name: str, template_choice: str) -> str:
    """Generate a function skeleton based on parameters."""
    try:
        params = json.loads(parameters_json) if parameters_json else []
        param_list = []
        for p in params:
            param_str = f"{p['name']}: {p['type_annotation']}"
            if not p['required'] and p['default_value']:
                param_str += f" = {p['default_value']}"
            param_list.append(param_str)
        
        if enable_context:
            ctx_name = context_param_name if context_param_name else "tool_context"
            param_list.append(f"{ctx_name}: dict")
        
        params_str = ", ".join(param_list)
        signature = f"def {function_name}({params_str}) -> {return_type}:"
        template_body = CODE_TEMPLATES.get(template_choice, CODE_TEMPLATES["Basic Template"])
        body_lines = template_body.split('\n')
        indented_body = '\n'.join('    ' + line if line.strip() else '' for line in body_lines)
        return f"{signature}\n{indented_body}"
    except Exception as e:
        return f"# Error: {str(e)}\n\npass"


def build_docstring(description: str, parameters_json: str, return_description: str) -> str:
    """Build a properly formatted docstring."""
    try:
        params = json.loads(parameters_json) if parameters_json else []
        lines = ['"""', description, ""]
        if params:
            lines.append("Args:")
            for p in params:
                lines.append(f"    {p['name']}: {p.get('description', 'No description')}")
            lines.append("")
        if return_description:
            lines.append("Returns:")
            lines.append(f"    {return_description}")
            lines.append("")
        lines.append('"""')
        return '\n'.join(lines)
    except Exception as e:
        return f'"""Error: {str(e)}"""'


def generate_complete_tool_code(tool_name: str, description: str, function_name: str, parameters_json: str,
                                return_type: str, return_description: str, function_body: str,
                                enable_context: bool, context_param_name: str, custom_name: str,
                                custom_description: str, enable_custom_schema: bool, custom_schema: str) -> Tuple[str, str]:
    """Generate the complete tool code with decorator and docstring."""
    try:
        params = json.loads(parameters_json) if parameters_json else []
        imports = ["from strands import tool"]
        all_types = [p['type_annotation'] for p in params] + [return_type]
        if any('Optional' in t or 'List' in t or 'Dict' in t for t in all_types):
            imports.append("from typing import Optional, List, Dict, Any")
        
        decorator_args = []
        if custom_name:
            decorator_args.append(f'name="{custom_name}"')
        if custom_description:
            decorator_args.append(f'description="{custom_description}"')
        if enable_custom_schema and custom_schema:
            decorator_args.append(f'inputSchema={custom_schema}')
        if enable_context:
            ctx_name = context_param_name if context_param_name else "tool_context"
            decorator_args.append(f'context="{ctx_name}"')
        
        decorator = f"@tool({', '.join(decorator_args)})" if decorator_args else "@tool"
        
        param_list = []
        for p in params:
            param_str = f"{p['name']}: {p['type_annotation']}"
            if not p['required'] and p['default_value']:
                param_str += f" = {p['default_value']}"
            param_list.append(param_str)
        
        if enable_context:
            ctx_name = context_param_name if context_param_name else "tool_context"
            param_list.append(f"{ctx_name}: dict")
        
        docstring = build_docstring(description, parameters_json, return_description)
        docstring_lines = docstring.split('\n')
        indented_docstring = '\n'.join('    ' + line if line.strip() or i == 0 else '' 
                                      for i, line in enumerate(docstring_lines))
        body_lines = function_body.split('\n')
        indented_body = '\n'.join('    ' + line if line.strip() else '' for line in body_lines)
        
        complete_code = f"""{chr(10).join(imports)}


{decorator}
def {function_name}({", ".join(param_list)}) -> {return_type}:
{indented_docstring}
{indented_body}
"""
        return complete_code, "✅ Code generated successfully!"
    except Exception as e:
        return "", f"❌ Error: {str(e)}"


def validate_tool_code(code: str) -> str:
    """Validate Python syntax and Strands requirements."""
    if not code.strip():
        return "❌ No code provided"
    issues = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append(f"Syntax Error (line {e.lineno}): {e.msg}")
    if "@tool" not in code:
        issues.append("Missing @tool decorator")
    if "@tool" in code and "from strands import tool" not in code and "import strands" not in code:
        issues.append("Missing strands import")
    if "def " not in code:
        issues.append("No function definition")
    if '"""' not in code and "'''" not in code:
        issues.append("Missing docstring (recommended)")
    if issues:
        return "⚠️ Issues:\n" + "\n".join(f"  • {issue}" for issue in issues)
    return "✅ All validation checks passed!"


def preview_tool_spec(code: str) -> str:
    """Generate a preview of how the tool will appear to agents."""
    try:
        tree = ast.parse(code)
        tool_info = {"tool_name": "unknown", "description": "", "parameters": [], "return_type": "Any"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                tool_info["tool_name"] = node.name
                if ast.get_docstring(node):
                    tool_info["description"] = ast.get_docstring(node).split('\n')[0]
                for arg in node.args.args:
                    param_type = ast.unparse(arg.annotation) if arg.annotation else "Any"
                    tool_info["parameters"].append({"name": arg.arg, "type": param_type})
                if node.returns:
                    tool_info["return_type"] = ast.unparse(node.returns)
                break
        
        preview = f"**Tool:** `{tool_info['tool_name']}`\n**Description:** {tool_info['description']}\n**Parameters:**\n"
        if tool_info["parameters"]:
            for p in tool_info["parameters"]:
                preview += f"  • `{p['name']}`: {p['type']}\n"
        else:
            preview += "  • No parameters\n"
        preview += f"\n**Returns:** `{tool_info['return_type']}`"
        return preview
    except Exception as e:
        return f"❌ Error: {str(e)}"


def save_tool_tc(tool_name: str, description: str, complete_code: str, parameters_json: str) -> Tuple[str, gr.Dropdown]:
    """Save the tool to disk."""
    try:
        if not tool_name:
            return "❌ Tool name required", gr.Dropdown()
        if not complete_code:
            return "❌ No code to save", gr.Dropdown()
        
        validation = validate_tool_code(complete_code)
        if not validation.startswith("✅"):
            return f"❌ Validation failed:\n{validation}", gr.Dropdown()
        
        params = json.loads(parameters_json) if parameters_json else []
        params_schema = {"type": "object", "properties": {}, "required": []}
        for p in params:
            type_map = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "dict": "object"}
            json_type = type_map.get(p["type_annotation"].lower().split("[")[0], "string")
            params_schema["properties"][p["name"]] = {"type": json_type, "description": p.get("description", "")}
            if p["required"]:
                params_schema["required"].append(p["name"])
        
        config = ToolConfig(
            name=tool_name,
            description=description,
            function_code=complete_code,
            parameters_schema=params_schema,
            returns_schema={"type": "string"}
        )
        
        result = tool_manager.create_tool(config)
        if result["success"]:
            return f"✅ Tool saved!\n**JSON:** `{result['filepath']}`\n**Python:** `{result['python_file']}`", refresh_tool_list_tc()
        return f"❌ {result['error']}", gr.Dropdown()
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.Dropdown()


def load_tool_tc(tool_name: str) -> Tuple:
    """Load a tool from disk into the editor."""
    try:
        if not tool_name:
            return ("", "", "", "[]", "str", "", "", False, "tool_context", "", "", False, "{}", "No tool selected")
        
        result = tool_manager.load_tool(tool_name)
        if not result["success"]:
            return ("", "", "", "[]", "str", "", "", False, "tool_context", "", "", False, "{}", f"❌ {result['error']}")
        
        config = result["config"]
        tree = ast.parse(config.function_code)
        function_name, parameters, return_type, function_body = "", [], "str", "pass"
        enable_context, context_param = False, "tool_context"
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                for arg in node.args.args:
                    param_type = ast.unparse(arg.annotation) if arg.annotation else "str"
                    if arg.arg in ["tool_context", "context"] and "dict" in param_type.lower():
                        enable_context, context_param = True, arg.arg
                    else:
                        parameters.append({"name": arg.arg, "type_annotation": param_type, "description": "", "required": True, "default_value": ""})
                if node.returns:
                    return_type = ast.unparse(node.returns)
                body_lines = [ast.unparse(stmt) for stmt in node.body if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))]
                function_body = '\n'.join(body_lines)
                break
        
        return (config.name, config.description, function_name, json.dumps(parameters, indent=2), return_type, "", 
                function_body, enable_context, context_param, "", "", False, "{}", f"✅ Loaded: {tool_name}")
    except Exception as e:
        return ("", "", "", "[]", "str", "", "", False, "tool_context", "", "", False, "{}", f"❌ Error: {str(e)}")


def delete_tool_tc(tool_name: str) -> Tuple[str, gr.Dropdown]:
    """Delete a tool from disk."""
    if not tool_name:
        return "❌ No tool selected", gr.Dropdown()
    result = tool_manager.delete_tool(tool_name, delete_file=True)
    if result["success"]:
        return f"✅ '{tool_name}' deleted", refresh_tool_list_tc()
    return f"❌ {result['error']}", gr.Dropdown()


# ============================================================================
# UI CREATION
# ============================================================================

def create_agent_creator_ui():
    """Create the complete Agent Creator + Tool Creator UI with tabs."""
    
    with gr.Blocks(title="Strands SDK - Agent & Tool Creator") as interface:
        gr.Markdown("# 🚀 Strands SDK - Agent & Tool Creator")
        gr.Markdown("Comprehensive interface for creating AI agents and tools")
        
        with gr.Tabs():
            # ========================= AGENT CREATOR TAB =========================
            with gr.Tab("🤖 Agent Creator"):
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
        
        # Apply prompt template
        apply_template_btn.click(
            fn=apply_prompt_template,
            inputs=[template_dropdown],
            outputs=[system_prompt]
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
    
        # ========================= TOOL CREATOR TAB =========================
        with gr.Tab("🛠️ Tool Creator"):
            gr.Markdown("### Create production-ready Strands SDK tools")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 📚 Tool Library")
                    tool_list_tc = gr.Dropdown(choices=tool_manager.list_saved_tools(), label="Saved Tools")
                    with gr.Row():
                        refresh_tc_btn = gr.Button("🔄", size="sm")
                        load_tc_btn = gr.Button("📂 Load", size="sm", variant="primary")
                    clear_tc_btn = gr.Button("🆕 Clear", size="sm")
                    delete_tc_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                    tc_status = gr.Markdown("Ready!")
                
                with gr.Column(scale=3):
                    with gr.Group():
                        gr.Markdown("### 🎯 Tool Identity")
                        with gr.Row():
                            tc_tool_name = gr.Textbox(label="Tool Name", placeholder="my_tool")
                            tc_function_name = gr.Textbox(label="Function Name", placeholder="my_tool")
                        tc_description = gr.Textbox(label="Description", lines=2)
                    
                    with gr.Group():
                        gr.Markdown("### 📋 Parameters (JSON)")
                        tc_parameters = gr.Textbox(label="Parameters", value="[]", lines=6)
                        gr.Markdown('**Example:** `[{"name": "text", "type_annotation": "str", "description": "Input text", "required": true, "default_value": ""}]`')
                    
                    with gr.Group():
                        gr.Markdown("### 🔄 Return Type")
                        with gr.Row():
                            tc_return_type = gr.Dropdown(choices=TYPE_ANNOTATIONS, value="str", label="Return Type")
                            tc_return_desc = gr.Textbox(label="Return Description")
                    
                    with gr.Accordion("⚙️ Advanced", open=False):
                        with gr.Row():
                            tc_enable_context = gr.Checkbox(label="Enable Context Param")
                            tc_context_param = gr.Textbox(label="Context Param Name", value="tool_context")
                        with gr.Row():
                            tc_custom_name = gr.Textbox(label="Custom Name Override")
                            tc_custom_desc = gr.Textbox(label="Custom Description Override")
                        with gr.Row():
                            tc_enable_schema = gr.Checkbox(label="Custom Input Schema")
                            tc_custom_schema = gr.Textbox(label="Schema JSON", value="{}", lines=3)
                    
                    with gr.Group():
                        gr.Markdown("### 💻 Implementation")
                        tc_template = gr.Dropdown(choices=list(CODE_TEMPLATES.keys()), value="Basic Template", label="Template")
                        tc_gen_skeleton_btn = gr.Button("🏗️ Generate Skeleton", variant="secondary")
                        tc_function_body = gr.Code(label="Function Body", language="python", lines=12, value=CODE_TEMPLATES["Basic Template"])
                    
                    with gr.Group():
                        gr.Markdown("### ✨ Generate Complete Code")
                        tc_gen_code_btn = gr.Button("Generate Complete Tool Code", variant="primary", size="lg")
                        tc_gen_status = gr.Textbox(label="Status")
                        tc_complete_code = gr.Code(label="Complete Tool Code", language="python", lines=15)
                    
                    with gr.Row():
                        with gr.Column():
                            tc_validate_btn = gr.Button("🔍 Validate")
                            tc_validation = gr.Textbox(label="Validation", lines=4)
                        with gr.Column():
                            tc_preview_btn = gr.Button("👀 Preview")
                            tc_preview = gr.Markdown("No preview")
                    
                    with gr.Group():
                        gr.Markdown("### 💾 Save Tool")
                        tc_save_btn = gr.Button("💾 Save Tool", variant="primary", size="lg")
                        tc_save_output = gr.Markdown("Not saved yet")
            
            # Tool Creator Event Handlers
            tc_tool_name.change(fn=auto_populate_function_name, inputs=[tc_tool_name], outputs=[tc_function_name])
            refresh_tc_btn.click(fn=refresh_tool_list_tc, outputs=[tool_list_tc])
            tc_gen_skeleton_btn.click(fn=generate_function_skeleton, inputs=[tc_function_name, tc_parameters, tc_return_type, tc_enable_context, tc_context_param, tc_template], outputs=[tc_function_body])
            tc_gen_code_btn.click(fn=generate_complete_tool_code, inputs=[tc_tool_name, tc_description, tc_function_name, tc_parameters, tc_return_type, tc_return_desc, tc_function_body, tc_enable_context, tc_context_param, tc_custom_name, tc_custom_desc, tc_enable_schema, tc_custom_schema], outputs=[tc_complete_code, tc_gen_status])
            tc_validate_btn.click(fn=validate_tool_code, inputs=[tc_complete_code], outputs=[tc_validation])
            tc_preview_btn.click(fn=preview_tool_spec, inputs=[tc_complete_code], outputs=[tc_preview])
            tc_save_btn.click(fn=save_tool_tc, inputs=[tc_tool_name, tc_description, tc_complete_code, tc_parameters], outputs=[tc_save_output, tool_list_tc])
            load_tc_btn.click(fn=load_tool_tc, inputs=[tool_list_tc], outputs=[tc_tool_name, tc_description, tc_function_name, tc_parameters, tc_return_type, tc_return_desc, tc_function_body, tc_enable_context, tc_context_param, tc_custom_name, tc_custom_desc, tc_enable_schema, tc_custom_schema, tc_status])
            clear_tc_btn.click(fn=lambda: ("", "", "", "[]", "str", "", CODE_TEMPLATES["Basic Template"], False, "tool_context", "", "", False, "{}", "", "Cleared"), outputs=[tc_tool_name, tc_description, tc_function_name, tc_parameters, tc_return_type, tc_return_desc, tc_function_body, tc_enable_context, tc_context_param, tc_custom_name, tc_custom_desc, tc_enable_schema, tc_custom_schema, tc_complete_code, tc_status])
            delete_tc_btn.click(fn=delete_tool_tc, inputs=[tool_list_tc], outputs=[tc_status, tool_list_tc])
    
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
