"""
Utility functions for LLM management.
"""

from pathlib import Path
from typing import List
from .templates import PROMPT_TEMPLATES


def get_model_path(model_name: str, models_dir: Path) -> Path:
    """
    Convert model name to file system path.
    
    Args:
        model_name: Model name in format "org/model" (e.g., "meta-llama/Llama-3.1-8B")
        models_dir: Base directory where models are stored
        
    Returns:
        Path object pointing to the model directory
    """
    # Convert "meta-llama/Llama-3.1-8B" -> "meta-llama--Llama-3.1-8B"
    safe_name = model_name.replace("/", "--")
    return models_dir / safe_name


def get_available_models(models_dir: Path) -> List[str]:
    """
    Scan the models directory and return list of available models.
    
    Args:
        models_dir: Base directory where models are stored
        
    Returns:
        Sorted list of model names in "org/model" format
    """
    if not models_dir.exists():
        return []
    
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    # Convert back to readable format (e.g., "meta-llama--Llama-3.1-8B" -> "meta-llama/Llama-3.1-8B")
    return sorted([d.name.replace("--", "/", 1) for d in model_dirs])


def apply_prompt_template(template_name: str, user_prompt: str) -> str:
    """
    Apply a prompt template to user input.
    
    Args:
        template_name: Name of the template from PROMPT_TEMPLATES
        user_prompt: The user's raw input text
        
    Returns:
        Formatted prompt with template applied, or original prompt if no template
    """
    if not template_name or template_name == "None" or not user_prompt:
        return user_prompt
    
    template = PROMPT_TEMPLATES.get(template_name, "")
    if "{prompt}" in template:
        return template.format(prompt=user_prompt)
    return user_prompt
