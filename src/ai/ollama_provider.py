"""
Ollama implementation of the AI provider interface.

Uses Ollama's OpenAI-compatible API (v1/chat/completions) via the
openai SDK, allowing local LLM inference for patient data processing
without sending data to external APIs.

Ollama must be running locally (default: http://localhost:11434).
This is the ONLY file that configures Ollama-specific settings.

Note: Embeddings are handled by PubMedBERT (via the Embedder service),
not by the LLM provider. The embed() method raises a clear error.
"""

import logging

from openai import AsyncOpenAI, APIConnectionError, APIStatusError

from src.ai.base import AIProvider, ChatMessage, ChatResponse, EmbeddingResponse
from src.ai.exceptions import (
    AIConnectionError as AIConnError,
    AIInvalidResponseError,
    AIProviderError,
)
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class OllamaProvider(AIProvider):
    """Ollama local LLM via OpenAI-compatible API."""

    def __init__(self) -> None:
        settings = get_settings()
        # Ollama exposes an OpenAI-compatible endpoint at /v1
        self._client = AsyncOpenAI(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",  # Ollama doesn't need a real key
            timeout=settings.ollama_request_timeout,
            max_retries=1,
        )
        self._default_model = settings.ollama_model
        self._default_max_tokens = settings.ollama_max_tokens
        self._default_temperature = settings.ollama_temperature
        logger.info(
            "Ollama provider initialised (base_url=%s, model=%s)",
            settings.ollama_base_url,
            self._default_model,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to the local Ollama server."""
        model = model or self._default_model
        temperature = temperature if temperature is not None else self._default_temperature
        max_tokens = max_tokens or self._default_max_tokens

        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APIConnectionError as exc:
            raise AIConnError(
                f"Cannot connect to Ollama at {self._client.base_url}. "
                f"Is Ollama running? (ollama serve): {exc}",
                provider="ollama",
            ) from exc
        except APIStatusError as exc:
            raise AIInvalidResponseError(
                f"Ollama error (status {exc.status_code}): {exc.message}",
                provider="ollama",
            ) from exc

        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logger.debug(
            "Ollama chat completed (model=%s, tokens=%s)",
            model,
            usage.get("total_tokens", "?"),
        )

        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=usage,
            raw_response=response.model_dump(),
        )

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> EmbeddingResponse:
        """
        Not supported — GuidelineGuard uses PubMedBERT for embeddings.

        Raises:
            AIProviderError: Always, with a helpful message.
        """
        raise AIProviderError(
            "Ollama embed() is not used. GuidelineGuard uses PubMedBERT "
            "for medical embeddings (via the Embedder service). "
            "The LLM provider is only used for chat/scoring.",
            provider="ollama",
        )
