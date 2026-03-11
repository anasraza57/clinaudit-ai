"""
Tests for the Ollama AI provider.

Uses mocked OpenAI client to test the Ollama provider without a real
Ollama server running.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai.base import ChatMessage, ChatResponse, EmbeddingResponse
from src.ai.exceptions import AIConnectionError, AIInvalidResponseError, AIProviderError


# ── Provider initialisation ─────────────────────────────────────────


class TestOllamaProviderInit:
    def test_provider_name(self):
        """Provider should identify itself as 'ollama'."""
        with patch.dict(os.environ, {
            "AI_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "OLLAMA_MODEL": "llama3.1",
        }):
            from src.config.settings import get_settings
            get_settings.cache_clear()
            from src.ai.ollama_provider import OllamaProvider
            provider = OllamaProvider()
            assert provider.provider_name == "ollama"
            get_settings.cache_clear()

    def test_custom_base_url(self):
        """Provider should use the configured base URL."""
        with patch.dict(os.environ, {
            "AI_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://gpu-server:11434",
            "OLLAMA_MODEL": "mistral",
        }):
            from src.config.settings import get_settings
            get_settings.cache_clear()
            from src.ai.ollama_provider import OllamaProvider
            provider = OllamaProvider()
            assert "gpu-server" in str(provider._client.base_url)
            get_settings.cache_clear()


# ── Chat completion ─────────────────────────────────────────────────


class TestOllamaChat:
    @pytest.mark.asyncio
    async def test_chat_success(self):
        """Successful chat should return a ChatResponse."""
        with patch.dict(os.environ, {
            "AI_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }):
            from src.config.settings import get_settings
            get_settings.cache_clear()
            from src.ai.ollama_provider import OllamaProvider
            provider = OllamaProvider()

            # Mock the internal OpenAI client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Score: +2\nExplanation: Good."
            mock_response.model = "llama3.1"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_response.model_dump.return_value = {}

            provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await provider.chat(
                messages=[ChatMessage(role="user", content="Hello")],
                temperature=0.0,
            )

            assert isinstance(result, ChatResponse)
            assert "Score: +2" in result.content
            assert result.model == "llama3.1"
            assert result.usage["total_tokens"] == 150
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_chat_connection_error(self):
        """Connection error should raise AIConnectionError."""
        from openai import APIConnectionError

        with patch.dict(os.environ, {
            "AI_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }):
            from src.config.settings import get_settings
            get_settings.cache_clear()
            from src.ai.ollama_provider import OllamaProvider
            provider = OllamaProvider()

            provider._client.chat.completions.create = AsyncMock(
                side_effect=APIConnectionError(request=MagicMock())
            )

            with pytest.raises(AIConnectionError, match="Cannot connect to Ollama"):
                await provider.chat(
                    messages=[ChatMessage(role="user", content="Hello")]
                )
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_chat_simple_convenience(self):
        """chat_simple should work via the base class method."""
        with patch.dict(os.environ, {
            "AI_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }):
            from src.config.settings import get_settings
            get_settings.cache_clear()
            from src.ai.ollama_provider import OllamaProvider
            provider = OllamaProvider()

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.model = "llama3.1"
            mock_response.usage = None
            mock_response.model_dump.return_value = {}

            provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await provider.chat_simple("Hello")
            assert result == "Test response"
            get_settings.cache_clear()


# ── Embed (not supported) ──────────────────────────────────────────


class TestOllamaEmbed:
    @pytest.mark.asyncio
    async def test_embed_raises_error(self):
        """embed() should raise AIProviderError (PubMedBERT is used instead)."""
        with patch.dict(os.environ, {
            "AI_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }):
            from src.config.settings import get_settings
            get_settings.cache_clear()
            from src.ai.ollama_provider import OllamaProvider
            provider = OllamaProvider()

            with pytest.raises(AIProviderError, match="PubMedBERT"):
                await provider.embed(["test text"])
            get_settings.cache_clear()


# ── Factory registration ────────────────────────────────────────────


class TestOllamaFactory:
    def test_ollama_registered_in_factory(self):
        """'ollama' should be a valid provider in the factory."""
        from src.ai.factory import _PROVIDERS
        assert "ollama" in _PROVIDERS

    def test_local_alias_registered(self):
        """'local' should be an alias for ollama in the factory."""
        from src.ai.factory import _PROVIDERS
        assert "local" in _PROVIDERS
        assert _PROVIDERS["local"] is _PROVIDERS["ollama"]
