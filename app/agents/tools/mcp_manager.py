"""
MCP (Model Context Protocol) client manager using strands SDK.

Manages connections to multiple MCP servers using strands.tools.mcp.MCPClient.
Implements full MCP spec: tools, resources, and prompts.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from mcp.client.streamable_http import streamablehttp_client
from strands.hooks import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookRegistry,
)
from strands.tools.mcp.mcp_client import MCPClient

from app.core.config import settings
from app.core.exceptions import MCPError, ToolExecutionError


def _create_streamable_transport(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float = 30,
) -> Callable[[], AsyncIterator]:
    """Create a transport callable for Streamable HTTP.

    Args:
        url: MCP server URL
        headers: Optional HTTP headers
        timeout: Request timeout in seconds

    Returns:
        Transport callable compatible with strands MCPClient
    """

    @asynccontextmanager
    async def _transport() -> AsyncIterator:
        async with streamablehttp_client(
            url=url,
            headers=headers,
            timeout=timeout,
        ) as (recv_stream, send_stream, get_session_id):
            yield (recv_stream, send_stream)

    return _transport


class MCPServerConnection:
    """Connection to a single MCP server using strands MCPClient.

    Implements full MCP spec: tools, resources, and prompts.
    """

    name: str
    url: str
    headers: dict[str, str]
    _hook_registry: HookRegistry | None
    _client: MCPClient | None
    _transport: Callable | None

    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize MCP server connection.

        Args:
            name: Server identifier
            url: Server URL
            headers: Optional HTTP headers
            hook_registry: Optional hook registry for tool call tracking
        """
        self.name = name
        self.url = url
        self.headers = headers or {}
        self._hook_registry = hook_registry
        self._client = None
        self._transport = None

    async def start(self) -> None:
        """Start the MCP connection."""
        self._transport = _create_streamable_transport(
            self.url,
            self.headers,
            settings.mcp.timeout,
        )

        self._client = MCPClient(
            transport_callable=self._transport,
            startup_timeout=settings.mcp.timeout,
        )

        await self._client.start()

    async def stop(self) -> None:
        """Stop the MCP connection."""
        if self._client:
            await self._client.stop()

    # ===== Tools =====

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from this server.

        Returns:
            List of tool definitions
        """
        if not self._client:
            await self.start()

        try:
            result = await asyncio.to_thread(self._client.list_tools_sync)
            return result.get("tools", [])
        except Exception as e:
            raise MCPError(
                f"Failed to list tools from {self.name}: {e}"
            ) from e

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call a tool on this server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if not self._client:
            await self.start()

        # Pre-hook
        if self._hook_registry:
            await self._hook_registry.execute(
                BeforeToolCallEvent(
                    tool_name=tool_name,
                    tool_args=arguments,
                )
            )

        try:
            result = await asyncio.to_thread(
                self._client.call_tool_sync,
                tool_name,
                arguments,
            )

            # Post-hook (success)
            if self._hook_registry:
                await self._hook_registry.execute(
                    AfterToolCallEvent(
                        tool_name=tool_name,
                        tool_args=arguments,
                        result=result,
                    )
                )

            return result

        except Exception as e:
            # Post-hook (error)
            if self._hook_registry:
                await self._hook_registry.execute(
                    AfterToolCallEvent(
                        tool_name=tool_name,
                        tool_args=arguments,
                        result=None,
                        error=e,
                    )
                )

            raise ToolExecutionError(f"Tool call failed: {e}") from e

    # ===== Resources =====

    async def list_resources(self) -> list[dict[str, Any]]:
        """List available resources from this server.

        Returns:
            List of resource definitions
        """
        if not self._client:
            await self.start()

        try:
            result = await asyncio.to_thread(self._client.list_resources_sync)
            return result.get("resources", [])
        except Exception as e:
            raise MCPError(
                f"Failed to list resources from {self.name}: {e}"
            ) from e

    async def list_resource_templates(self) -> list[dict[str, Any]]:
        """List available resource templates from this server.

        Returns:
            List of resource template definitions
        """
        if not self._client:
            await self.start()

        try:
            result = await asyncio.to_thread(
                self._client.list_resource_templates_sync
            )
            return result.get("resourceTemplates", [])
        except Exception as e:
            raise MCPError(
                f"Failed to list resource templates from {self.name}: {e}"
            ) from e

    async def read_resource(self, uri: str) -> Any:
        """Read a resource from this server.

        Args:
            uri: Resource URI

        Returns:
            Resource contents
        """
        if not self._client:
            await self.start()

        try:
            result = await asyncio.to_thread(
                self._client.read_resource_sync,
                uri,
            )
            return result
        except Exception as e:
            raise MCPError(
                f"Failed to read resource {uri} from {self.name}: {e}"
            ) from e

    # ===== Prompts =====

    async def list_prompts(self) -> list[dict[str, Any]]:
        """List available prompts from this server.

        Returns:
            List of prompt definitions
        """
        if not self._client:
            await self.start()

        try:
            result = await asyncio.to_thread(self._client.list_prompts_sync)
            return result.get("prompts", [])
        except Exception as e:
            raise MCPError(
                f"Failed to list prompts from {self.name}: {e}"
            ) from e

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Get a prompt from this server.

        Args:
            prompt_name: Name of the prompt
            arguments: Optional prompt arguments

        Returns:
            Prompt content
        """
        if not self._client:
            await self.start()

        try:
            result = await asyncio.to_thread(
                self._client.get_prompt_sync,
                prompt_name,
                arguments or {},
            )
            return result
        except Exception as e:
            raise MCPError(
                f"Failed to get prompt {prompt_name} from {self.name}: {e}"
            ) from e


class MCPManager:
    """Manages multiple MCP server connections using strands MCPClient.

    Implements full MCP spec:
    - Tools: List and call tools
    - Resources: List and read resources
    - Prompts: List and get prompts

    Features:
    - Multiple server support
    - Tool discovery across all servers
    - Pre/post hooks for metadata tracking
    """

    _connections: dict[str, MCPServerConnection]
    _hook_registry: HookRegistry | None

    def __init__(self, hook_registry: HookRegistry | None = None):
        """Initialize the MCP manager.

        Args:
            hook_registry: Optional strands hook registry for pre/post hooks
        """
        self._connections = {}
        self._hook_registry = hook_registry

        # Register servers from config
        for endpoint in settings.mcp.endpoints:
            self.add_server(
                name=endpoint["name"],
                url=endpoint["url"],
                headers=endpoint.get("headers"),
            )

    def add_server(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Add an MCP server.

        Args:
            name: Server identifier
            url: Server URL
            headers: Optional HTTP headers
        """
        self._connections[name] = MCPServerConnection(
            name,
            url,
            headers,
            self._hook_registry,
        )

    def get_server(self, name: str) -> MCPServerConnection:
        """Get a server connection by name.

        Args:
            name: Server identifier

        Returns:
            MCPServerConnection instance

        Raises:
            KeyError: If server not found
        """
        if name not in self._connections:
            raise KeyError(f"MCP server not found: {name}")
        return self._connections[name]

    def list_servers(self) -> list[str]:
        """List all registered server names.

        Returns:
            List of server identifiers
        """
        return list(self._connections.keys())

    # ===== Discovery =====

    async def discover_all(self) -> dict[str, Any]:
        """Discover all resources (tools, resources, prompts) from all servers.

        Returns:
            Dict with 'tools', 'resources', 'prompts' keys
        """
        # Start all connections
        start_tasks = [conn.start() for conn in self._connections.values()]
        await asyncio.gather(*start_tasks, return_exceptions=True)

        result = {
            "tools": {},
            "resources": {},
            "prompts": {},
        }

        # Discover from each server
        for server_name, conn in self._connections.items():
            try:
                # Tools
                tools = await conn.list_tools()
                for t in tools:
                    t["_server"] = server_name
                    result["tools"][t["name"]] = t

                # Resources
                resources = await conn.list_resources()
                for r in resources:
                    r["_server"] = server_name
                    result["resources"][r["uri"]] = r

                # Prompts
                prompts = await conn.list_prompts()
                for p in prompts:
                    p["_server"] = server_name
                    result["prompts"][p["name"]] = p
            except Exception:
                # Log but continue with other servers
                pass

        return result

    # ===== Tools =====

    async def list_tools(self) -> dict[str, tuple[str, dict]]:
        """Discover all tools from all servers.

        Returns:
            Dict mapping tool names to (server_name, tool_spec) tuples
        """
        all_tools = {}

        for server_name in self._connections:
            conn = self._connections[server_name]
            try:
                await conn.start()
                tools = await conn.list_tools()
                for t in tools:
                    all_tools[t["name"]] = (server_name, t)
            except Exception:
                pass

        return all_tools

    async def call_tool(
        self,
        tool_name: str,
        server_name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Call a tool on a specific server.

        Args:
            tool_name: Name of the tool to call
            server_name: Name of the server (required
                if tool exists on multiple servers)
            arguments: Tool arguments

        Returns:
            Tool result
        """
        # Find server if not specified
        if server_name is None:
            tools = await self.list_tools()
            if tool_name not in tools:
                raise ToolExecutionError(f"Tool not found: {tool_name}")
            server_name = tools[tool_name][0]

        conn = self._connections[server_name]
        return await conn.call_tool(tool_name, arguments or {})

    # ===== Resources =====

    async def list_resources(self) -> dict[str, tuple[str, dict]]:
        """List all resources from all servers.

        Returns:
            Dict mapping resource URIs to (server_name, resource_spec) tuples
        """
        all_resources = {}

        for server_name in self._connections:
            conn = self._connections[server_name]
            try:
                await conn.start()
                resources = await conn.list_resources()
                for r in resources:
                    all_resources[r["uri"]] = (server_name, r)
            except Exception:
                pass

        return all_resources

    async def read_resource(
        self,
        uri: str,
        server_name: str | None = None,
    ) -> Any:
        """Read a resource from a server.

        Args:
            uri: Resource URI
            server_name: Optional server name (auto-detected if not specified)

        Returns:
            Resource contents
        """
        # Find server if not specified
        if server_name is None:
            resources = await self.list_resources()
            if uri not in resources:
                raise MCPError(f"Resource not found: {uri}")
            server_name = resources[uri][0]

        conn = self._connections[server_name]
        return await conn.read_resource(uri)

    # ===== Prompts =====

    async def list_prompts(self) -> dict[str, tuple[str, dict]]:
        """List all prompts from all servers.

        Returns:
            Dict mapping prompt names to (server_name, prompt_spec) tuples
        """
        all_prompts = {}

        for server_name in self._connections:
            conn = self._connections[server_name]
            try:
                await conn.start()
                prompts = await conn.list_prompts()
                for p in prompts:
                    all_prompts[p["name"]] = (server_name, p)
            except Exception:
                pass

        return all_prompts

    async def get_prompt(
        self,
        prompt_name: str,
        server_name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Get a prompt from a server.

        Args:
            prompt_name: Name of the prompt
            server_name: Optional server name (auto-detected if not specified)
            arguments: Optional prompt arguments

        Returns:
            Prompt content
        """
        # Find server if not specified
        if server_name is None:
            prompts = await self.list_prompts()
            if prompt_name not in prompts:
                raise MCPError(f"Prompt not found: {prompt_name}")
            server_name = prompts[prompt_name][0]

        conn = self._connections[server_name]
        return await conn.get_prompt(prompt_name, arguments)

    # ===== Strands Integration =====

    async def load_strands_tools(
        self, tool_names: list[str] | None = None
    ) -> list:
        """Load MCP tools as strands ToolProvider objects.

        Args:
            tool_names: Optional whitelist of tool names to load.
                        If None, loads all tools from all servers.

        Returns:
            List of strands tool providers
        """
        tools = []

        for conn in self._connections.values():
            await conn.start()

            mcp_tools = await asyncio.to_thread(conn._client.load_tools)

            if tool_names is not None:
                allowed = set(tool_names)
                mcp_tools = [
                    t
                    for t in mcp_tools
                    if (
                        (hasattr(t, "tool_name") and t.tool_name in allowed)
                        or (hasattr(t, "name") and t.name in allowed)
                        or (
                            hasattr(t, "__name__") and t.__name__ in allowed
                        )
                    )
                ]

            tools.extend(mcp_tools)

        return tools

    # ===== Lifecycle =====

    async def cleanup(self) -> None:
        """Cleanup all connections."""
        stop_tasks = [conn.stop() for conn in self._connections.values()]
        await asyncio.gather(*stop_tasks, return_exceptions=True)


@asynccontextmanager
async def create_mcp_manager(
    hook_registry: HookRegistry | None = None,
) -> AsyncIterator[MCPManager]:
    """Create an MCP manager with automatic cleanup.

    Args:
        hook_registry: Optional strands hook registry

    Yields:
        MCPManager instance
    """
    manager = MCPManager(hook_registry=hook_registry)
    try:
        yield manager
    finally:
        await manager.cleanup()
