"""Agents module for chatbot orchestration."""

from app.agents.agent import (
    HybridChatbotAgent,
    create_chatbot_agent,
)
from app.agents.execution_mode import ExecutionMode
from app.agents.hook import SmartAgentHook, extract_text
from app.agents.mode_analyzer import ModeAnalyzer
from app.agents.planner import ToolCall, ToolCallPlanner
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
from app.agents.tools import (
    DiscoveredTool,
    MCPManager,
    ToolDiscoveryClient,
    ToolWrapper,
    create_mcp_manager,
    with_hooks,
)

__all__ = [
    "CleanupPolicy",
    "DiscoveredTool",
    "ExecutionMode",
    "HybridChatbotAgent",
    "IndexEntry",
    "ModeAnalyzer",
    "SessionLifecycle",
    "SessionStore",
    "SmartAgentHook",
    "ToolCall",
    "ToolCallPlanner",
    "ToolDiscoveryClient",
    "create_chatbot_agent",
    "create_session_store",
    "extract_text",
    "AgentState",
    "ConversationHistory",
    "ConversationMessage",
    "ToolCallResult",
    "MCPManager",
    "ToolWrapper",
    "create_mcp_manager",
    "with_hooks",
]
