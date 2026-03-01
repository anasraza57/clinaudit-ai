"""Tests for AI abstraction layer base classes."""

from src.ai.base import ChatMessage, ChatResponse, EmbeddingResponse


def test_chat_message_creation():
    """ChatMessage should be created with role and content."""
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_chat_message_immutable():
    """ChatMessage should be frozen (immutable)."""
    msg = ChatMessage(role="user", content="Hello")
    try:
        msg.content = "Changed"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass  # Expected — dataclass is frozen


def test_chat_response_creation():
    """ChatResponse should store content and metadata."""
    resp = ChatResponse(
        content="The answer is 42",
        model="gpt-4o-mini",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    assert resp.content == "The answer is 42"
    assert resp.model == "gpt-4o-mini"
    assert resp.usage["prompt_tokens"] == 10
    assert resp.usage["completion_tokens"] == 5


def test_embedding_response_creation():
    """EmbeddingResponse should store embedding vectors."""
    resp = EmbeddingResponse(
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        model="text-embedding-3-small",
    )
    assert len(resp.embeddings) == 2
    assert len(resp.embeddings[0]) == 3
