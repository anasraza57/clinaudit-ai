"""
Abstract base class for AI providers.

This defines the contract that ALL AI providers must implement.
Application code imports ONLY from this module (or the factory) —
never directly from openai, anthropic, or any specific SDK.

This is the Strategy Pattern: the interface stays the same,
the implementation is swapped at runtime based on configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ChatMessage:
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ChatResponse:
    """The result of a chat completion request."""

    content: str
    model: str
    usage: dict = field(default_factory=dict)  # {"prompt_tokens": X, "completion_tokens": Y}
    raw_response: dict = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class EmbeddingResponse:
    """The result of an embedding request."""

    embeddings: list[list[float]]
    model: str
    usage: dict = field(default_factory=dict)


class AIProvider(ABC):
    """
    Abstract interface for AI/LLM providers.

    Every provider (OpenAI, Anthropic, local, etc.) must implement
    these methods. Application code uses this interface exclusively.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider (e.g., 'openai', 'anthropic')."""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of conversation messages.
            model: Override the default model for this request.
            temperature: Override the default temperature (0.0 = deterministic).
            max_tokens: Override the default max output tokens.

        Returns:
            ChatResponse with the model's reply.

        Raises:
            AIAuthenticationError: Invalid or missing API key.
            AIRateLimitError: Provider rate limit exceeded.
            AIConnectionError: Cannot reach the provider API.
            AIInvalidResponseError: Unexpected response format.
        """
        ...

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            model: Override the default embedding model.

        Returns:
            EmbeddingResponse with embedding vectors.

        Raises:
            AIProviderError: On any provider error.
        """
        ...

    async def chat_simple(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """
        Convenience method for simple single-turn prompts.

        Args:
            prompt: The user's message.
            system_prompt: Optional system instruction.
            **kwargs: Passed through to chat().

        Returns:
            The model's reply as a plain string.
        """
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        response = await self.chat(messages, **kwargs)
        return response.content
