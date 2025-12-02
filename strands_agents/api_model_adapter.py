"""
API Model Adapter for Strands SDK.

This adapter allows Strands SDK to work with models hosted via API,
avoiding local dependencies on torch, transformers, pydantic, etc.
"""

from typing import Optional, Any, AsyncGenerator
import requests
import json


class APIModelAdapter:
    """
    Strands Model adapter that uses API for inference.
    
    This adapter communicates with the API server for all model operations,
    keeping heavy dependencies (torch, transformers, etc.) in the container.
    """
    
    def __init__(
        self,
        api_base_url: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ):
        """
        Initialize the API Model adapter.
        
        Args:
            api_base_url: Base URL of the API server
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty factor
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.config = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        }
    
    def get_config(self) -> dict[str, Any]:
        """Get the model configuration."""
        return self.config.copy()
    
    def update_config(self, **model_config: Any) -> None:
        """Update the model configuration."""
        self.config.update(model_config)
    
    async def stream(
        self,
        messages,
        tool_specs: Optional[list] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: Optional[Any] = None,
        system_prompt_content: Optional[list] = None,
        **kwargs: Any
    ) -> AsyncGenerator[dict, None]:
        """
        Stream conversation with the API-hosted model.
        
        Converts Strands Message format to a prompt string and streams
        the response using the API.
        
        Args:
            messages: Strands Messages list
            tool_specs: Tool specifications (not used for API models)
            system_prompt: System prompt string
            tool_choice: Tool choice strategy (not used)
            system_prompt_content: System prompt content blocks
            **kwargs: Additional arguments
            
        Yields:
            StreamEvent dictionaries compatible with Strands SDK
        """
        import asyncio
        
        # Build prompt from Strands Messages
        prompt = self._format_messages_to_prompt(messages, system_prompt)
        
        # Stream using API
        try:
            response = requests.post(
                f"{self.api_base_url}/generate/stream",
                json={
                    "prompt": prompt,
                    "max_tokens": self.config["max_tokens"],
                    "temperature": self.config["temperature"],
                    "top_p": self.config["top_p"],
                    "top_k": self.config["top_k"],
                    "repetition_penalty": self.config["repetition_penalty"],
                    "skip_prompt": True
                },
                stream=True
            )
            response.raise_for_status()
            
            # Parse SSE stream
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str.strip():
                            try:
                                event = json.loads(data_str)
                                # Convert API event to Strands StreamEvent format
                                strands_event = self._convert_to_strands_event(event)
                                if strands_event:
                                    yield strands_event
                                    # Yield control to allow other async operations
                                    await asyncio.sleep(0)
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def _format_messages_to_prompt(
        self,
        messages,
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
    
    def _convert_to_strands_event(self, api_event: dict) -> Optional[dict]:
        """
        Convert API event format to Strands StreamEvent format.
        
        Args:
            api_event: Event from API /generate/stream endpoint
            
        Returns:
            Strands-compatible StreamEvent or None
        """
        event_type = api_event.get("type")
        
        if event_type == "error":
            # Error event
            return {
                "type": "error",
                "error": api_event.get("error", "Unknown error")
            }
        
        elif event_type == "token":
            # Token streaming event
            return {
                "type": "content_block_delta",
                "delta": {
                    "type": "text_delta",
                    "text": api_event.get("text", "")
                },
                "cumulative_text": api_event.get("cumulative_text", "")
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
                            "text": api_event.get("text", "")
                        }
                    ]
                },
                "metadata": {
                    "total_tokens": api_event.get("total_tokens", 0),
                    "tokens_per_second": api_event.get("tokens_per_second", 0),
                    "elapsed_seconds": api_event.get("elapsed_seconds", 0)
                }
            }
        
        return None
    
    async def structured_output(
        self,
        output_model,
        prompt,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Get structured output from the model.
        
        Note: API models don't natively support structured output.
        This would require additional parsing/validation logic.
        
        Raises:
            NotImplementedError: Structured output not yet supported
        """
        raise NotImplementedError(
            "Structured output is not yet implemented for API-based models. "
            "This would require additional parsing and validation logic."
        )
