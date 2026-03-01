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


# Registry of available providers.
# Keys must match the AI_PROVIDER env var values.
_PROVIDERS: dict[str, callable] = {
    "openai": _create_openai_provider,
    # "anthropic": _create_anthropic_provider,  # Add when needed
    # "local": _create_local_provider,          # Add when needed
}


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
