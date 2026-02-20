"""Shared execution mode enum used across agent and API layers."""

from enum import Enum


class ExecutionMode(str, Enum):
    """Agent execution mode.

    Used by both the agent layer and the API request/response models.
    - FAST: direct questions, simple lookups, quick answers.
    - AGENTIC: research, multi-step investigation, data synthesis.
    - AUTO: agent decides based on query analysis.
    """

    FAST = "fast"
    AGENTIC = "agentic"
    AUTO = "auto"
