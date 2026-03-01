"""
Custom exceptions for AI provider operations.

Having specific exception types allows calling code to handle
different failure modes appropriately (e.g., retry on rate limit,
alert on auth failure, fallback on connection error).
"""


class AIProviderError(Exception):
    """Base exception for all AI provider errors."""

    def __init__(self, message: str, provider: str = "unknown"):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class AIAuthenticationError(AIProviderError):
    """Raised when API key is invalid or missing."""
    pass


class AIRateLimitError(AIProviderError):
    """Raised when the provider's rate limit is exceeded."""

    def __init__(self, message: str, provider: str = "unknown", retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(message, provider)


class AIConnectionError(AIProviderError):
    """Raised when unable to connect to the provider's API."""
    pass


class AIInvalidResponseError(AIProviderError):
    """Raised when the provider returns an unexpected or unparseable response."""
    pass
