# Example Tools

Example tool configurations for the Agent Playground Tool Builder.

## Available Examples

### calculator
Safe mathematical expression evaluator using Python's AST module.

### web_search  
Placeholder for web search functionality (shows schema pattern).

## Usage

Copy to your tools directory:
```bash
copy frontend\example_tools\calculator.json C:\path\to\tools\
```

Then load in the Tool Builder tab.

## Creating Custom Tools

Tools are defined with:
- **function_code**: Python function implementation
- **parameters_schema**: JSON schema for inputs
- **returns_schema**: JSON schema for outputs

See examples for patterns.
