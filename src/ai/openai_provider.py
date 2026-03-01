"""
OpenAI implementation of the AI provider interface.

This is the ONLY file that imports the openai SDK.
All other application code uses the abstract AIProvider interface.
"""

import logging

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from src.ai.base import AIProvider, ChatMessage, ChatResponse, EmbeddingResponse
from src.ai.exceptions import (
    AIAuthenticationError,
    AIConnectionError,
    AIInvalidResponseError,
    AIRateLimitError,
)
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class OpenAIProvider(AIProvider):
    """OpenAI API implementation."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise AIAuthenticationError(
                "OPENAI_API_KEY is not set. Add it to your .env file.",
                provider="openai",
            )
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._default_model = settings.openai_model
        self._default_embedding_model = settings.openai_embedding_model
        self._default_max_tokens = settings.openai_max_tokens
        self._default_temperature = settings.openai_temperature
        logger.info(
            "OpenAI provider initialised (model=%s, embedding_model=%s)",
            self._default_model,
            self._default_embedding_model,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to OpenAI."""
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
        except AuthenticationError as exc:
            raise AIAuthenticationError(
                f"Invalid API key: {exc.message}", provider="openai"
            ) from exc
        except RateLimitError as exc:
            raise AIRateLimitError(
                f"Rate limit exceeded: {exc.message}", provider="openai"
            ) from exc
        except APIConnectionError as exc:
            raise AIConnectionError(
                f"Connection failed: {exc}", provider="openai"
            ) from exc
        except APIStatusError as exc:
            raise AIInvalidResponseError(
                f"API error (status {exc.status_code}): {exc.message}",
                provider="openai",
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
            "OpenAI chat completed (model=%s, tokens=%s)",
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
        """Generate embeddings using OpenAI's API."""
        model = model or self._default_embedding_model

        try:
            response = await self._client.embeddings.create(
                model=model,
                input=texts,
            )
        except AuthenticationError as exc:
            raise AIAuthenticationError(
                f"Invalid API key: {exc.message}", provider="openai"
            ) from exc
        except RateLimitError as exc:
            raise AIRateLimitError(
                f"Rate limit exceeded: {exc.message}", provider="openai"
            ) from exc
        except APIConnectionError as exc:
            raise AIConnectionError(
                f"Connection failed: {exc}", provider="openai"
            ) from exc
        except APIStatusError as exc:
            raise AIInvalidResponseError(
                f"API error (status {exc.status_code}): {exc.message}",
                provider="openai",
            ) from exc

        embeddings = [item.embedding for item in response.data]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logger.debug(
            "OpenAI embedding completed (model=%s, texts=%d)",
            model,
            len(texts),
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=response.model,
            usage=usage,
        )
