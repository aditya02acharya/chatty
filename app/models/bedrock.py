"""
AWS Bedrock model provider using strands SDK.

Uses the official strands.models.BedrockModel for native Bedrock integration.
"""

import asyncio
from collections.abc import AsyncIterator
from functools import lru_cache

from strands.models import BedrockModel

from app.core.config import settings
from app.core.exceptions import BedrockError


class BedrockProvider:
    """AWS Bedrock model provider using strands SDK.

    Provides async model inference with streaming support.
    """

    def __init__(self, model_id: str | None = None, region: str | None = None):
        """Initialize the Bedrock provider.

        Args:
            model_id: Model ID to use (defaults to settings.bedrock.default_model)
            region: AWS region (defaults to settings.bedrock.region)
        """
        self._model_id = model_id or settings.bedrock.default_model
        self._region = region or settings.bedrock.region
        self._model: BedrockModel | None = None

    @property
    def model(self) -> BedrockModel:
        """Get or create the BedrockModel instance."""
        if self._model is None:
            self._model = BedrockModel(
                model_id=self._model_id,
                region_name=self._region,
            )
        return self._model

    async def invoke(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **inference_config,
    ) -> str:
        """Invoke the model with a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **inference_config: Inference parameters (max_tokens, temperature, etc.)

        Returns:
            Model response text
        """
        # Merge with default inference config
        config = settings.bedrock.inference_config.copy()
        config.update(inference_config)

        try:
            # Build messages
            messages = [{"role": "user", "content": [{"text": prompt}]}]

            # Use structured output for non-streaming
            response = await asyncio.to_thread(
                self.model.structured_output,
                messages=messages,
                system_prompt=system_prompt or settings.agent.system_prompt,
                **config,
            )

            return response.content
        except Exception as e:
            raise BedrockError(f"Bedrock invocation failed: {e}") from e

    async def invoke_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **inference_config,
    ) -> AsyncIterator[str]:
        """Invoke the model with streaming response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **inference_config: Inference parameters (max_tokens, temperature, etc.)

        Yields:
            Response text chunks as they arrive
        """
        # Merge with default inference config
        config = settings.bedrock.inference_config.copy()
        config.update(inference_config)

        try:
            # Build messages
            messages = [{"role": "user", "content": [{"text": prompt}]}]

            # Stream the response
            async for chunk in self.model.stream(
                messages=messages,
                system_prompt=system_prompt or settings.agent.system_prompt,
                **config,
            ):
                yield chunk
        except Exception as e:
            raise BedrockError(f"Bedrock streaming failed: {e}") from e

    def update_config(self, model_id: str | None = None, **config) -> None:
        """Update model configuration.

        Args:
            model_id: New model ID (optional)
            **config: Other configuration parameters
        """
        if model_id:
            self._model_id = model_id
            # Recreate model with new ID
            self._model = BedrockModel(
                model_id=self._model_id,
                region_name=self._region,
            )

        if config:
            self.model.update_config(**config)


@lru_cache
def create_bedrock_model(
    model_id: str | None = None,
    region: str | None = None,
) -> BedrockProvider:
    """Factory function to create a Bedrock provider instance.

    Args:
        model_id: Model ID to use
        region: AWS region

    Returns:
        BedrockProvider instance
    """
    return BedrockProvider(model_id=model_id, region=region)
