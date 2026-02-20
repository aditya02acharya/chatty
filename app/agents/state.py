"""
Agent state models for the chatbot.

Uses pydantic for structured state management.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentState(BaseModel):
    """Current state of the agent."""

    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(description="Unique request identifier")
    user_message: str = Field(description="Original user message")
    iteration: int = Field(default=0, description="Current iteration count")
    max_iterations: int = Field(
        default=10, description="Maximum allowed iterations"
    )

    # Decision tracking
    last_decision: str | None = Field(
        default=None, description="Last decision made"
    )
    last_tool_used: str | None = Field(
        default=None, description="Last tool called"
    )

    # Response building
    response_parts: list[str] = Field(
        default_factory=list, description="Accumulated response"
    )

    # Timing
    started_at: datetime = Field(
        default_factory=datetime.now, description="Start time"
    )
    finished_at: datetime | None = Field(
        default=None, description="Finish time"
    )


class ToolCallResult(BaseModel):
    """Result of a tool call."""

    tool_name: str = Field(description="Name of the tool")
    arguments: dict[str, Any] = Field(description="Arguments passed to tool")
    result: Any = Field(description="Tool result")
    error: str | None = Field(default=None, description="Error if tool failed")
    duration_ms: float = Field(description="Execution time in milliseconds")
    is_remote: bool = Field(
        default=False, description="Whether tool is remote (MCP)"
    )


class ConversationMessage(BaseModel):
    """A message in the conversation."""

    role: str = Field(
        description="Message role (user, assistant, system, tool)"
    )
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_call_id: str | None = Field(
        default=None, description="Tool call ID if applicable"
    )
    tool_name: str | None = Field(
        default=None, description="Tool name if applicable"
    )


class ConversationHistory(BaseModel):
    """Conversation history with messages."""

    messages: list[ConversationMessage] = Field(default_factory=list)

    def add_message(
        self,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Add a message to the conversation."""
        self.messages.append(
            ConversationMessage(
                role=role,
                content=content,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
        )

    def get_strands_messages(self) -> list[dict[str, Any]]:
        """Convert to strands message format."""
        strands_messages = []
        for msg in self.messages:
            strands_msg = {
                "role": msg.role,
                "content": [{"text": msg.content}],
            }
            strands_messages.append(strands_msg)
        return strands_messages
