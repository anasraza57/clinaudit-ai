"""
AI provider factory.

This is the single entry point for obtaining an AI provider instance.
Application code calls get_ai_provider() and gets back whatever
provider is configured via the AI_PROVIDER environment variable.

To add a new provider:
  1. Create a new file (e.g., anthropic_provider.py) implementing AIProvider
  2. Register it in the _PROVIDERS dict below
  3. Set AI_PROVIDER=anthropic in .env
  Done. Zero changes to any other application code.
"""

import logging
from functools import lru_cache

from src.ai.base import AIProvider
from src.ai.exceptions import AIProviderError
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def _create_openai_provider() -> AIProvider:
    from src.ai.openai_provider import OpenAIProvider
    return OpenAIProvider()


def _create_ollama_provider() -> AIProvider:
    from src.ai.ollama_provider import OllamaProvider
    return OllamaProvider()


# Registry of available providers.
# Keys must match the AI_PROVIDER env var values.
_PROVIDERS: dict[str, callable] = {
    "openai": _create_openai_provider,
    "ollama": _create_ollama_provider,
    "local": _create_ollama_provider,  # Alias for ollama
    # "anthropic": _create_anthropic_provider,  # Add when needed
}


# Model-name prefixes that map to the OpenAI provider.
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-", "text-embedding", "chatgpt")


def get_ai_provider_for_model(model: str) -> AIProvider:
    """
    Create an AI provider appropriate for the given model name.

    Detects the provider type from the model name prefix:
    - ``gpt-*``, ``o1-*``, ``o3-*`` → OpenAI
    - Everything else → Ollama

    Unlike ``get_ai_provider()`` this is **not** cached — it creates
    a fresh instance each time so the caller can use it independently.
    """
    if any(model.startswith(p) for p in _OPENAI_PREFIXES):
        logger.info("Creating OpenAI provider for model: %s", model)
        return _create_openai_provider()
    logger.info("Creating Ollama provider for model: %s", model)
    return _create_ollama_provider()


@lru_cache()
def get_ai_provider() -> AIProvider:
    """
    Return the configured AI provider (cached singleton).

    The provider is determined by the AI_PROVIDER environment variable.
    Call get_ai_provider.cache_clear() in tests to reset.

    Returns:
        An AIProvider implementation.

    Raises:
        AIProviderError: If the configured provider is not registered.
    """
    settings = get_settings()
    provider_name = settings.ai_provider.lower()

    if provider_name not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS.keys()))
        raise AIProviderError(
            f"Unknown AI provider '{provider_name}'. "
            f"Available providers: {available}. "
            f"Set AI_PROVIDER in your .env file.",
            provider=provider_name,
        )

    logger.info("Initialising AI provider: %s", provider_name)
    return _PROVIDERS[provider_name]()
