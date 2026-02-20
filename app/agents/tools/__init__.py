"""Tools module for local and MCP remote tools."""

from app.agents.tools.discovery import DiscoveredTool, ToolDiscoveryClient
from app.agents.tools.mcp_manager import MCPManager, create_mcp_manager
from app.agents.tools.tool_wrapper import ToolWrapper, with_hooks

__all__ = [
    "DiscoveredTool",
    "MCPManager",
    "ToolDiscoveryClient",
    "ToolWrapper",
    "create_mcp_manager",
    "with_hooks",
]
