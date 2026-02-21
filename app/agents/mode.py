"""Execution mode enum."""

from enum import Enum


class ExecutionMode(str, Enum):
    """Agent execution mode.

    FAST    - User explicitly requests fast mode.
    AGENTIC - User explicitly requests agentic mode.
    AUTO    - Supervisor decides based on query analysis.
    """

    FAST = "fast"
    AGENTIC = "agentic"
    AUTO = "auto"
