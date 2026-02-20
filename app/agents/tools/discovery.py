"""
Tool discovery client using an MCP-hosted discovery tool.

Calls an external MCP tool that accepts a query description and returns
the best-matching tools. The agent uses these results to load only a
relevant subset of tools at runtime.
"""

import logging
from dataclasses import dataclass
from typing import Any

from app.agents.tools.mcp_manager import MCPManager
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredTool:
    """A tool returned by the discovery service."""

    name: str
    description: str = ""
    score: float = 0.0
    server_name: str | None = None


class ToolDiscoveryClient:
    """Calls the MCP-hosted discovery tool to find relevant tools for a query.

    The discovery tool is itself an MCP tool exposed by an external server.
    It accepts a query/description and returns the top-K matching tools.
    """

    def __init__(self, mcp_manager: MCPManager):
        self._mcp = mcp_manager
        self._config = settings.agent.tool_discovery

    async def discover(self, query: str) -> list[DiscoveredTool]:
        """Call the discovery MCP tool and return matching tools.

        Args:
            query: User query or description of what tools are needed.

        Returns:
            List of DiscoveredTool sorted by relevance score (highest first).
            Returns empty list if discovery is disabled or fails.
        """
        if not self._config.enabled:
            return []

        try:
            result = await self._mcp.call_tool(
                tool_name=self._config.tool_name,
                server_name=self._config.server_name,
                arguments={
                    "query": query,
                    "top_k": self._config.top_k,
                },
            )
            return self._parse_result(result)
        except Exception:
            logger.warning(
                "Tool discovery failed for query: %s", query, exc_info=True
            )
            return []

    def _parse_result(self, result: Any) -> list[DiscoveredTool]:
        """Parse the MCP tool response into DiscoveredTool objects."""
        tools: list[DiscoveredTool] = []

        # Handle various response shapes from the MCP tool
        raw_tools = self._extract_tools_list(result)

        for item in raw_tools:
            if isinstance(item, dict):
                tools.append(
                    DiscoveredTool(
                        name=item.get("name", ""),
                        description=item.get("description", ""),
                        score=float(item.get("score", 0.0)),
                        server_name=item.get("server_name"),
                    )
                )
            elif isinstance(item, str):
                tools.append(DiscoveredTool(name=item))

        # Sort by score descending
        tools.sort(key=lambda t: t.score, reverse=True)
        return tools[: self._config.top_k]

    @staticmethod
    def _extract_tools_list(result: Any) -> list:
        """Extract the list of tools from various MCP result formats."""
        if isinstance(result, list):
            return result

        if isinstance(result, dict):
            # Try common keys: "tools", "results", "content"
            for key in ("tools", "results", "matches"):
                if key in result and isinstance(result[key], list):
                    return result[key]

            # Strands MCP result: {"content": [{"text": "..."}]}
            content = result.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if "json" in block and isinstance(
                            block["json"], list
                        ):
                            return block["json"]
                        if "json" in block and isinstance(
                            block["json"], dict
                        ):
                            return ToolDiscoveryClient._extract_tools_list(
                                block["json"]
                            )

        return []
