"""Custom exceptions for the chatbot application."""


class ChatbotError(Exception):
    """Base exception for chatbot errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ConfigurationError(ChatbotError):
    """Raised when configuration is invalid or missing."""

    pass


class BedrockError(ChatbotError):
    """Raised when Bedrock API call fails."""

    pass


class MCPError(ChatbotError):
    """Raised when MCP client operation fails."""

    pass


class AgentError(ChatbotError):
    """Raised when agent execution fails."""

    pass


class ToolExecutionError(ChatbotError):
    """Raised when tool execution fails."""

    pass


class StreamDisconnectedError(ChatbotError):
    """Raised when SSE stream is disconnected unexpectedly."""

    pass
