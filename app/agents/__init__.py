"""Agents module for chatbot orchestration."""

from app.agents.agent import ExecutionMode, HybridChatbotAgent, create_chatbot_agent
from app.agents.nodes import (
    ReflectionNode,
    RoutingNode,
    SynthesisNode,
    ToolNode,
    ToolNodeConfig,
    ValidationNode,
)
from app.agents.session_fs import (
    CleanupPolicy,
    IndexEntry,
    SessionLifecycle,
    SessionStore,
    create_session_store,
)
from app.agents.state import (
    AgentState,
    ConversationHistory,
    ConversationMessage,
    ToolCallResult,
)
from app.agents.tools import MCPManager, ToolWrapper, create_mcp_manager, with_hooks

# Alias for backwards compatibility
ChatbotAgent = HybridChatbotAgent

__all__ = [
    "ChatbotAgent",
    "CleanupPolicy",
    "ExecutionMode",
    "HybridChatbotAgent",
    "IndexEntry",
    "SessionLifecycle",
    "SessionStore",
    "create_chatbot_agent",
    "create_session_store",
    "ReflectionNode",
    "RoutingNode",
    "SynthesisNode",
    "ToolNode",
    "ToolNodeConfig",
    "ValidationNode",
    "AgentState",
    "ConversationHistory",
    "ConversationMessage",
    "ToolCallResult",
    "MCPManager",
    "ToolWrapper",
    "create_mcp_manager",
    "with_hooks",
]
