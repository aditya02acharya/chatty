"""
Tools endpoints for tool discovery and listing.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents import create_chatbot_agent
from app.agents.tools import create_mcp_manager

router = APIRouter(prefix="/tools", tags=["tools"])


class ToolInfo(BaseModel):
    """Information about a tool."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    is_remote: bool = False
    server_name: str | None = None


@router.get("", response_model=list[ToolInfo])
async def list_tools() -> list[ToolInfo]:
    """List all available tools (local and MCP remote).

    Returns:
        List of tool definitions
    """
    try:
        async with create_chatbot_agent(enable_mcp=False) as agent:
            tools = []

            # Local tools – use the agent's public factory
            for tool_func in agent._create_local_tools(None):
                name = (
                    tool_func.name
                    if hasattr(tool_func, "name")
                    else tool_func.__name__
                )
                tools.append(
                    ToolInfo(
                        name=name,
                        description=tool_func.__doc__,
                        is_remote=False,
                    )
                )

            # MCP tools – use the managed context manager
            async with create_mcp_manager() as mcp:
                mcp_tools = await mcp.list_tools()
                for tool_name, (server_name, tool_spec) in mcp_tools.items():
                    tools.append(
                        ToolInfo(
                            name=tool_name,
                            description=tool_spec.get("description"),
                            input_schema=tool_spec.get("inputSchema"),
                            is_remote=True,
                            server_name=server_name,
                        )
                    )

            return tools
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list tools: {e}"
        )


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "tools-api"}
