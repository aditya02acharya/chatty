"""Agents module for chatbot orchestration.

Public API:
    ChatGraph / create_chat_graph — multi-agent graph entry point.
    ExecutionMode                 — FAST | AGENTIC | AUTO.
    SmartAgentHook                — tool-result compaction hook.
    SessionStore / SessionLifecycle — session filesystem.
    MCPManager / create_mcp_manager — MCP integration.
"""

from app.agents.graph import ChatGraph, create_chat_graph
from app.agents.hook import SmartAgentHook
from app.agents.mode import ExecutionMode
from app.agents.session_fs import (
    CleanupPolicy,
    IndexEntry,
    SessionLifecycle,
    SessionStore,
    create_session_store,
)
from app.agents.tools import MCPManager, create_mcp_manager

__all__ = [
    "ChatGraph",
    "CleanupPolicy",
    "ExecutionMode",
    "IndexEntry",
    "MCPManager",
    "SessionLifecycle",
    "SessionStore",
    "SmartAgentHook",
    "create_chat_graph",
    "create_mcp_manager",
    "create_session_store",
]
