"""
Hybrid chatbot agent with intelligent mode selection and execution.

The agent orchestrates mode analysis, tool planning, session filesystem
lifecycle, and Strands SDK integration into a single streaming entry point.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.agent.conversation_manager import (
    NullConversationManager,
    SlidingWindowConversationManager,
)
from strands.models import BedrockModel
from strands.tools.executors import SequentialToolExecutor

from app.agents.execution_mode import ExecutionMode
from app.agents.hook import SmartAgentHook
from app.agents.mode_analyzer import ModeAnalyzer
from app.agents.planner import ToolCallPlanner
from app.agents.session_fs import (
    CleanupPolicy,
    SessionLifecycle,
    SessionStore,
)
from app.agents.tools import (
    ToolDiscoveryClient,
    create_mcp_manager,
)
from app.agents.tools.mcp_manager import MCPManager
from app.core.config import settings
from app.core.exceptions import AgentError
from app.streaming import create_streamer

logger = logging.getLogger(__name__)


def _cleanup_policy_for_mode(mode: ExecutionMode) -> CleanupPolicy:
    """Map execution mode to the appropriate cleanup policy (Strategy)."""
    if mode == ExecutionMode.FAST:
        return CleanupPolicy.ALWAYS
    return (
        CleanupPolicy.ALWAYS
    )  # agentic/auto: still cleanup after each request


class HybridChatbotAgent(BaseModel):
    """Hybrid agent with intelligent mode selection and execution.

    Features:
    - Auto mode selection (agent decides)
    - Parallel execution for independent tools
    - Sequential/iterative for dependent tools
    - Filesystem-based working memory with lifecycle cleanup
    - Complete, curated responses only
    """

    model_id: str = Field(
        default_factory=lambda: settings.bedrock.default_model
    )
    mode: ExecutionMode = Field(default=ExecutionMode.AUTO)
    session_id: str | None = Field(default=None)
    enable_mcp: bool = Field(default=True)

    max_parallel_tools: int = Field(
        default=5, description="Max parallel tools"
    )
    filesystem_enabled: bool = Field(default=True)

    model_config = {"populate_by_name": True}

    def __init__(self, **data):
        super().__init__(**data)

        self._bedrock_model = BedrockModel(
            model_id=self.model_id,
            region_name=settings.bedrock.region,
        )
        self._actual_mode: ExecutionMode | None = None
        # Lifecycle is created per-request in chat_stream / chat_complete
        self._lifecycle: SessionLifecycle | None = None

    # -- factory methods (Template Method helpers) --------------------------

    def _create_mode_analyzer(self) -> ModeAnalyzer:
        return ModeAnalyzer(model=self._bedrock_model)

    def _create_planner(self) -> ToolCallPlanner:
        return ToolCallPlanner(model=self._bedrock_model)

    def _create_local_tools(self, store: SessionStore | None) -> list:
        """Create local tools, binding them to the current session store."""

        @tool
        async def get_current_time() -> str:
            """Get the current time."""
            return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        @tool
        async def session_grep(
            pattern: str, case_sensitive: bool = False
        ) -> str:
            """Search session filesystem for pattern.

            Args:
                pattern: Regex pattern to search
                case_sensitive: Whether case sensitive
            """
            if not store:
                return "No session filesystem available"

            matches = await store.grep(pattern, case_sensitive)
            if not matches:
                return f"No matches for: {pattern}"

            results = [f"Found {len(matches)} matches:"]
            for m in matches[:20]:
                results.append(f"\n- Entry: {m['entry_id']}")
                results.append(f"  Tool: {m['tool']}")
                results.append(f"  Match: {m['match']}")

            return "\n".join(results)

        @tool
        async def session_summary() -> str:
            """Get summary of session filesystem data."""
            if not store:
                return "No session filesystem available"

            s = store.summary()
            per_tool_lines = []
            for name, info in s.get("per_tool", {}).items():
                per_tool_lines.append(
                    f"  - {name}: "
                    f"{info['count']} results, "
                    f"{info['total_bytes']} bytes"
                )

            return (
                f"Session: {s['session_id']}\n"
                f"Total entries: {s['total_entries']}\n"
                f"Tools used:\n" + "\n".join(per_tool_lines)
            )

        @tool
        async def read_session_file(entry_id: str) -> str:
            """Read a file from session filesystem.

            Args:
                entry_id: Entry identifier (e.g. 'search/001')
            """
            if not store:
                return "No session filesystem available"

            try:
                data = await store.read(entry_id)
                return f"Contents of {entry_id}:\n\n{data}"
            except FileNotFoundError:
                return f"Entry not found: {entry_id}"

        return [
            get_current_time,
            session_grep,
            session_summary,
            read_session_file,
        ]

    async def _discover_tools(
        self, message: str, mcp: MCPManager | None
    ) -> list[str] | None:
        """Use the discovery MCP tool to find relevant tools for a query.

        Returns:
            List of tool names if discovery is enabled and succeeds,
            or None to indicate "load all tools" (discovery disabled/failed).
        """
        if not settings.agent.tool_discovery.enabled or mcp is None:
            return None

        discovery = ToolDiscoveryClient(mcp)
        matches = await discovery.discover(message)
        if not matches:
            return None

        return [m.name for m in matches]

    # -- streaming entry point -----------------------------------------------

    async def chat_stream(
        self,
        message: str,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Process chat with streaming response.

        Handles three exit paths:
        1. Success  – stream completes, lifecycle cleans up.
        2. Error    – exception propagates, lifecycle cleans up.
        3. Cancel   – asyncio.CancelledError caught, lifecycle cleans up.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        async with create_streamer(request_id) as streamer:
            await streamer.run_started()

            # Step 1: Resolve mode
            analyzer = self._create_mode_analyzer()
            self._actual_mode, analysis = await analyzer.analyze_query(message)
            if self.mode != ExecutionMode.AUTO:
                self._actual_mode = self.mode

            await streamer.thinking_start(
                f"Mode: {self._actual_mode.value}"
                f" - {analysis.get('reasoning', '')}"
            )

            # Determine cleanup policy for this run
            policy = _cleanup_policy_for_mode(self._actual_mode)
            needs_fs = (
                self._actual_mode
                in (ExecutionMode.AGENTIC, ExecutionMode.AUTO)
                and self.filesystem_enabled
                and self.session_id is not None
            )

            # Step 2: Enter lifecycle (creates + guarantees cleanup)
            lifecycle = (
                SessionLifecycle(self.session_id or request_id, policy)
                if needs_fs
                else None
            )

            try:
                store: SessionStore | None = None
                if lifecycle:
                    store = await lifecycle.__aenter__()

                # Step 3: Discover relevant tools and plan
                tools = self._create_local_tools(store)
                mcp_tool_names: list[str] | None = None

                if self.enable_mcp:
                    async with create_mcp_manager() as mcp:
                        # Ask the discovery service which MCP tools
                        # are relevant for this query
                        mcp_tool_names = await self._discover_tools(
                            message, mcp
                        )

                        # Load only the discovered subset (or all
                        # if discovery is disabled / returned nothing)
                        mcp_tools = await mcp.load_strands_tools(
                            tool_names=mcp_tool_names
                        )
                        tools.extend(mcp_tools)

                available_names = [
                    t.name if hasattr(t, "name") else t.__name__
                    for t in tools
                ]

                planner = self._create_planner()
                planned_calls = await planner.plan_execution(
                    message, available_names, self._actual_mode
                )

                conv_manager = (
                    NullConversationManager()
                    if self._actual_mode == ExecutionMode.FAST
                    else SlidingWindowConversationManager(max_messages=100)
                )
                executor = (
                    None
                    if self._actual_mode == ExecutionMode.FAST
                    or not planned_calls
                    else SequentialToolExecutor()
                )

                hook = SmartAgentHook(streamer, self._actual_mode, store)

                agent = Agent(
                    model=self._bedrock_model,
                    tools=tools,
                    system_prompt=self._get_system_prompt(),
                    tool_executor=executor,
                    conversation_manager=conv_manager,
                    hooks=[hook],
                )

                # Step 5: Execute and stream complete response
                full_response = ""
                async for chunk in agent.stream_async(message):
                    full_response += chunk

                await streamer.content_start()
                await streamer.content_delta(full_response)
                await streamer.content_end()
                await streamer.done()

                yield full_response

            except asyncio.CancelledError:
                # Client disconnected – ensure cleanup runs
                logger.info(
                    "Request %s cancelled (client disconnect)", request_id
                )
                if lifecycle:
                    lifecycle.mark_error()
                raise

            except Exception as e:
                if lifecycle:
                    lifecycle.mark_error()
                await streamer.error("AGENT_ERROR", str(e))
                raise AgentError(f"Agent execution failed: {e}") from e

            finally:
                # Guarantee lifecycle cleanup regardless of exit path
                if lifecycle:
                    await lifecycle.__aexit__(None, None, None)

    # -- non-streaming entry point -------------------------------------------

    async def chat_complete(self, message: str, request_id: str) -> str:
        """Non-streaming chat – collects the full response and returns it."""
        full_response = ""
        async for chunk in self.chat_stream(message, request_id):
            full_response += chunk
        return full_response

    # -- prompt builder -------------------------------------------------------

    def _get_system_prompt(self) -> str:
        base = settings.agent.system_prompt

        if self._actual_mode == ExecutionMode.FAST:
            return f"""{base}

You are in FAST mode:
- Respond quickly and accurately
- Use tools only when necessary
- Provide direct, concise answers
- Maximum 1-2 tool calls
"""
        else:
            return f"""{base}

You are in AGENTIC mode:
- Be thorough and methodical
- Use multiple sources when needed
- Synthesize comprehensive, accurate answers
- Cite your sources
- Parallelize independent tool calls when beneficial

## Session filesystem

Tool results are automatically stored on a session filesystem. Instead of the
full result you receive a **compact receipt** with:
- A short preview (2-3 lines)
- A gap analysis describing what information is NOT in the preview

Use the gap analysis to decide whether you need more detail. If so, drill in:
- session_grep(pattern)              – regex search across all stored results
- read_session_file(entry_id)        – read a specific stored result in full
- session_summary()                  – see what data has been collected so far

Only retrieve what you actually need. The receipt often contains enough signal
to answer without a follow-up read.
"""

    # -- cleanup --------------------------------------------------------------

    async def cleanup(self) -> None:
        """Cleanup agent-level resources (lifecycle handled per-request)."""
        pass


@asynccontextmanager
async def create_chatbot_agent(**kwargs):
    """Create agent with automatic cleanup."""
    agent = HybridChatbotAgent(**kwargs)
    try:
        yield agent
    finally:
        await agent.cleanup()
