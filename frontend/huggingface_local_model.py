"""
HuggingFace Local Model Provider for Strands SDK.

This adapter allows Strands SDK to work with locally hosted HuggingFace models
managed by the model_manager, maintaining SDK agnosticism while providing
full Strands Agent compatibility.
"""

from typing import Optional, Any, AsyncGenerator, Type
from strands.models import Model
from strands.types import Messages, ToolSpec, StreamEvent, ToolChoice
from strands.types.content_blocks import SystemContentBlock
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from model_manager import ModelManager


class HuggingFaceLocalModel(Model):
    """
    Strands Model adapter for locally hosted HuggingFace models.
    
    This adapter bridges the gap between model_manager (SDK-agnostic) and 
    Strands SDK requirements, enabling local GPU-accelerated inference
    within the Strands Agent framework.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ):
        """
        Initialize the HuggingFace Local Model adapter.
        
        Args:
            model_manager: ModelManager instance for local model inference
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty factor
        """
        self.model_manager = model_manager
        self.config = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        }
    
    def get_config(self) -> dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Dictionary containing model configuration parameters
        """
        return self.config.copy()
    
    def update_config(self, **model_config: Any) -> None:
        """
        Update the model configuration.
        
        Args:
            **model_config: Configuration parameters to update
        """
        self.config.update(model_config)
    
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: Optional[ToolChoice] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream conversation with the local HuggingFace model.
        
        Converts Strands Message format to a prompt string and streams
        the response using model_manager.
        
        Args:
            messages: Strands Messages list
            tool_specs: Tool specifications (not used for local models)
            system_prompt: System prompt string
            tool_choice: Tool choice strategy (not used for local models)
            system_prompt_content: System prompt content blocks
            **kwargs: Additional arguments
            
        Yields:
            StreamEvent dictionaries compatible with Strands SDK
        """
        # Build prompt from Strands Messages
        prompt = self._format_messages_to_prompt(messages, system_prompt)
        
        # Stream using model_manager
        for event in self.model_manager.generate(
            prompt=prompt,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            top_k=self.config["top_k"],
            repetition_penalty=self.config["repetition_penalty"],
            skip_prompt=True
        ):
            # Convert model_manager events to Strands StreamEvent format
            strands_event = self._convert_to_strands_event(event)
            if strands_event:
                yield strands_event
    
    def _format_messages_to_prompt(
        self,
        messages: Messages,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Convert Strands Messages to a prompt string.
        
        Args:
            messages: Strands Messages list
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add system prompt if provided
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")
        
        # Process messages
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", [])
            
            # Extract text content from content blocks
            if isinstance(content, list):
                text_content = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        text_content.append(block["text"])
                text = " ".join(text_content)
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)
            
            # Format based on role
            if role == "user":
                prompt_parts.append(f"User: {text}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {text}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _convert_to_strands_event(self, model_manager_event: dict) -> Optional[dict]:
        """
        Convert model_manager event format to Strands StreamEvent format.
        
        Args:
            model_manager_event: Event from model_manager.generate()
            
        Returns:
            Strands-compatible StreamEvent or None
        """
        event_type = model_manager_event.get("type")
        
        if event_type == "error":
            # Error event
            return {
                "type": "error",
                "error": model_manager_event.get("error", "Unknown error")
            }
        
        elif event_type == "token":
            # Token streaming event
            return {
                "type": "content_block_delta",
                "delta": {
                    "type": "text_delta",
                    "text": model_manager_event.get("token", "")
                },
                "cumulative_text": model_manager_event.get("cumulative_text", "")
            }
        
        elif event_type == "complete":
            # Completion event
            return {
                "type": "message_complete",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": model_manager_event.get("text", "")
                        }
                    ]
                },
                "metadata": {
                    "total_tokens": model_manager_event.get("total_tokens", 0),
                    "tokens_per_second": model_manager_event.get("tokens_per_second", 0),
                    "elapsed_seconds": model_manager_event.get("elapsed_seconds", 0)
                }
            }
        
        return None
    
    async def structured_output(
        self,
        output_model: Type[BaseModel],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Get structured output from the model.
        
        Note: Local HuggingFace models don't natively support structured output.
        This would require additional parsing/validation logic.
        
        Args:
            output_model: Pydantic model for structured output
            prompt: Messages to process
            system_prompt: Optional system prompt
            **kwargs: Additional arguments
            
        Yields:
            Dictionary containing structured output events
            
        Raises:
            NotImplementedError: Structured output not yet supported for local models
        """
        raise NotImplementedError(
            "Structured output is not yet implemented for local HuggingFace models. "
            "This would require additional parsing and validation logic."
        )
