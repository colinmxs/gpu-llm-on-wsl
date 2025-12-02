"""
Tool Creator Tab for Gradio Frontend

This module provides the UI components and functionality for creating and managing Strands SDK tools.
"""

import gradio as gr
import ast
import re
import json
from typing import Tuple, List

from strands_agents.tool_manager import ToolManager, ToolConfig


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


class ToolCreatorTab:
    """Manages the Tool Creator tab UI and functionality."""
    
    def __init__(self, api_client: AgentToolClient):
        self.api_client = api_client
    
    def to_snake_case(self, text: str) -> str:
        """Convert text to snake_case for valid Python identifiers."""
        text = re.sub(r'[^a-zA-Z0-9_]', '_', text)
        text = re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()
        text = re.sub(r'_+', '_', text)
        return text.strip('_')
    
    def refresh_tool_list(self) -> gr.Dropdown:
        """Refresh tool list for tool creator."""
        try:
            tools = self.api_client.list_saved_tools()
        except Exception as e:
            print(f"Error fetching tools: {e}")
            tools = []
        return gr.Dropdown(choices=tools, value=None)
    
    def auto_populate_function_name(self, tool_name: str) -> str:
        """Auto-generate function name from tool name."""
        return self.to_snake_case(tool_name) if tool_name else ""
    
    def generate_function_skeleton(self, function_name: str, parameters_json: str, return_type: str, 
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
    
    def build_docstring(self, description: str, parameters_json: str, return_description: str) -> str:
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
    
    def generate_complete_tool_code(self, tool_name: str, description: str, function_name: str, parameters_json: str,
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
            
            docstring = self.build_docstring(description, parameters_json, return_description)
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
    
    def validate_tool_code(self, code: str) -> str:
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
    
    def preview_tool_spec(self, code: str) -> str:
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
    
    def save_tool(self, tool_name: str, description: str, complete_code: str, parameters_json: str) -> Tuple[str, gr.Dropdown]:
        """Save the tool to disk."""
        try:
            if not tool_name:
                return "❌ Tool name required", gr.Dropdown()
            if not complete_code:
                return "❌ No code to save", gr.Dropdown()
            
            validation = self.validate_tool_code(complete_code)
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
            
            try:
                result = self.api_client.create_tool(
                    name=tool_name,
                    description=description,
                    function_code=complete_code,
                    parameters_schema=params_schema,
                    returns_schema={"type": "string"}
                )
            except Exception as e:
                return f"❌ Error: {str(e)}", gr.Dropdown()
            if result["success"]:
                return f"✅ Tool saved!\n**JSON:** `{result['filepath']}`\n**Python:** `{result['python_file']}`", self.refresh_tool_list()
            return f"❌ {result['error']}", gr.Dropdown()
        except Exception as e:
            return f"❌ Error: {str(e)}", gr.Dropdown()
    
    def load_tool(self, tool_name: str) -> Tuple:
        """Load a tool from disk into the editor."""
        try:
            if not tool_name:
                return ("", "", "", "[]", "str", "", "", False, "tool_context", "", "", False, "{}", "No tool selected")
            
            try:
                result = self.api_client.load_tool(tool_name)
            except Exception as e:
                return ("", "", "", "[]", "str", "", "", False, "tool_context", "", "", False, "{}", f"❌ {str(e)}")
                
            if not result["success"]:
                return ("", "", "", "[]", "str", "", "", False, "tool_context", "", "", False, "{}", f"❌ {result['error']}")
            
            # Get tool info to get the config
            try:
                tool_info = self.api_client.get_tool_info(tool_name)
                config_dict = {
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "function_code": tool_info["function_code"],
                    "parameters_schema": tool_info.get("parameters_schema", {}),
                    "returns_schema": tool_info.get("returns_schema", {})
                }
                # Create a simple object to hold the config
                class Config:
                    def __init__(self, d):
                        for k, v in d.items():
                            setattr(self, k, v)
                config = Config(config_dict)
            except Exception as e:
                return ("", "", "", "[]", "str", "", "", False, "tool_context", "", "", False, "{}", f"❌ {str(e)}")
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
    
    def delete_tool(self, tool_name: str) -> Tuple[str, gr.Dropdown]:
        """Delete a tool from disk."""
        if not tool_name:
            return "❌ No tool selected", gr.Dropdown()
        try:
            result = self.api_client.delete_tool(tool_name, delete_file=True)
        except Exception as e:
            return f"❌ Error: {str(e)}", gr.Dropdown()
        if result["success"]:
            return f"✅ '{tool_name}' deleted", self.refresh_tool_list()
        return f"❌ {result['error']}", gr.Dropdown()
    
    def create_ui(self):
        """Create the Tool Creator tab UI."""
        gr.Markdown("### Create production-ready Strands SDK tools")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📚 Tool Library")
                try:
                    initial_tools = self.api_client.list_saved_tools()
                except:
                    initial_tools = []
                tool_list_tc = gr.Dropdown(choices=initial_tools, label="Saved Tools")
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
        tc_tool_name.change(fn=self.auto_populate_function_name, inputs=[tc_tool_name], outputs=[tc_function_name])
        refresh_tc_btn.click(fn=self.refresh_tool_list, outputs=[tool_list_tc])
        tc_gen_skeleton_btn.click(fn=self.generate_function_skeleton, inputs=[tc_function_name, tc_parameters, tc_return_type, tc_enable_context, tc_context_param, tc_template], outputs=[tc_function_body])
        tc_gen_code_btn.click(fn=self.generate_complete_tool_code, inputs=[tc_tool_name, tc_description, tc_function_name, tc_parameters, tc_return_type, tc_return_desc, tc_function_body, tc_enable_context, tc_context_param, tc_custom_name, tc_custom_desc, tc_enable_schema, tc_custom_schema], outputs=[tc_complete_code, tc_gen_status])
        tc_validate_btn.click(fn=self.validate_tool_code, inputs=[tc_complete_code], outputs=[tc_validation])
        tc_preview_btn.click(fn=self.preview_tool_spec, inputs=[tc_complete_code], outputs=[tc_preview])
        tc_save_btn.click(fn=self.save_tool, inputs=[tc_tool_name, tc_description, tc_complete_code, tc_parameters], outputs=[tc_save_output, tool_list_tc])
        load_tc_btn.click(fn=self.load_tool, inputs=[tool_list_tc], outputs=[tc_tool_name, tc_description, tc_function_name, tc_parameters, tc_return_type, tc_return_desc, tc_function_body, tc_enable_context, tc_context_param, tc_custom_name, tc_custom_desc, tc_enable_schema, tc_custom_schema, tc_status])
        clear_tc_btn.click(fn=lambda: ("", "", "", "[]", "str", "", CODE_TEMPLATES["Basic Template"], False, "tool_context", "", "", False, "{}", "", "Cleared"), outputs=[tc_tool_name, tc_description, tc_function_name, tc_parameters, tc_return_type, tc_return_desc, tc_function_body, tc_enable_context, tc_context_param, tc_custom_name, tc_custom_desc, tc_enable_schema, tc_custom_schema, tc_complete_code, tc_status])
        delete_tc_btn.click(fn=self.delete_tool, inputs=[tool_list_tc], outputs=[tc_status, tool_list_tc])
        
        # Return components that might need refreshing from other tabs
        return {
            'tool_list': tool_list_tc
        }
