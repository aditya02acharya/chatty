"""
Agent mode selection and execution strategy.

Design Patterns:
- Strategy Pattern: ExecutionMode + CleanupPolicy determine runtime behaviour.
- Template Method: chat_stream defines the skeleton (analyse → plan → execute →
  stream → cleanup); subclasses or config change the individual steps.
- Observer/Interceptor Pattern: SmartAgentHook intercepts tool results via the
  Strands AfterToolCallEvent, stores full payloads to the session filesystem,
  and replaces the conversation result with a compact receipt (preview + gap
  analysis) so the context window stays lean.
- Factory Method: _create_* helpers let subclasses override component creation.

Modes:
- fast: User explicitly requests fast mode
- agentic: User explicitly requests agentic mode
- auto: Agent decides based on query analysis

Execution:
- Parallel for independent tools
- Sequential/iterative for dependent tools
- Hybrid as needed
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.agent.conversation_manager import (
    NullConversationManager,
    SlidingWindowConversationManager,
)
from strands.hooks import HookProvider
from strands.hooks.events import AfterToolCallEvent
from strands.models import BedrockModel
from strands.tools.executors import SequentialToolExecutor

from app.agents.result_compactor import PASSTHROUGH_TOOLS, ResultCompactor
from app.agents.session_fs import (
    CleanupPolicy,
    SessionLifecycle,
    SessionStore,
)
from app.agents.tools import MCPManager, create_mcp_manager
from app.core.config import settings
from app.core.exceptions import AgentError
from app.streaming import AGUIStreamer

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """Agent execution mode."""

    FAST = "fast"
    AGENTIC = "agentic"
    AUTO = "auto"


@dataclass
class ToolCall:
    """A tool call with metadata."""

    name: str
    args: dict[str, Any]
    dependencies: list[str]  # Names of tools this depends on
    independent: bool = True
    estimated_duration: float = 1.0  # seconds


# ---------------------------------------------------------------------------
# Mode analyser
# ---------------------------------------------------------------------------

class ModeAnalyzer(BaseModel):
    """Analyzes query to determine appropriate mode."""

    model: BedrockModel = Field(description="Model for analysis")

    model_config = {"arbitrary_types_allowed": True}

    async def analyze_query(self, query: str) -> tuple[ExecutionMode, dict[str, Any]]:
        """Analyze query to determine mode and strategy."""
        prompt = f"""Analyze this query and determine the best mode:

Query: "{query}"

Consider:
- FAST mode: Direct questions, simple lookups, quick answers
- AGENTIC mode: Research, multi-step investigation, data synthesis

Return JSON:
{{
    "mode": "fast|agentic",
    "reasoning": "...",
    "estimated_tools": 0-10,
    "requires_research": true/false,
    "complexity": "low|medium|high"
}}
"""

        try:
            response = await self.model.structured_output_async(
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                system_prompt="You analyze queries to determine processing strategy.",
            )
            result = response.content

            mode_str = result.get("mode", "fast").lower()
            mode = ExecutionMode.AGENTIC if mode_str == "agentic" else ExecutionMode.FAST

            return mode, result
        except Exception:
            # Fallback: simple heuristic
            query_lower = query.lower()
            keywords = ["research", "investigate", "analyze", "compare", "detailed", "comprehensive"]
            if any(kw in query_lower for kw in keywords):
                return ExecutionMode.AGENTIC, {"reasoning": "keyword_match", "complexity": "high"}
            return ExecutionMode.FAST, {"reasoning": "default", "complexity": "low"}


# ---------------------------------------------------------------------------
# Tool-call planner
# ---------------------------------------------------------------------------

class ToolCallPlanner(BaseModel):
    """Plans tool execution with dependency analysis."""

    model: BedrockModel = Field(description="Model for planning")

    model_config = {"arbitrary_types_allowed": True}

    async def plan_execution(
        self,
        query: str,
        available_tools: list[str],
        mode: ExecutionMode,
    ) -> list[ToolCall]:
        """Plan tool execution with dependency analysis."""
        if mode == ExecutionMode.FAST:
            return await self._plan_fast(query, available_tools)
        return await self._plan_agentic(query, available_tools)

    async def _plan_fast(self, query: str, available_tools: list[str]) -> list[ToolCall]:
        prompt = f"""Does this query require multiple tools?

Query: "{query}"
Available tools: {', '.join(available_tools[:10])}

Return JSON:
{{
    "tools_needed": [],
    "reasoning": "..."
}}

If only 1 tool or no tools needed, return that. If multiple explicitly requested (e.g. "compare X and Y"), return all.
"""

        try:
            response = await self.model.structured_output_async(
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                system_prompt="Identify required tools.",
            )
            result = response.content
            tools_requested = result.get("tools_needed", [])
            return [
                ToolCall(name=t, args={}, dependencies=[], independent=True)
                for t in tools_requested[:3]
            ]
        except Exception:
            return []

    async def _plan_agentic(self, query: str, available_tools: list[str]) -> list[ToolCall]:
        tools_str = "\n".join(f"- {t}" for t in available_tools)

        prompt = f"""Plan the execution for this query:

Query: "{query}"

Available tools:
{tools_str}

Create a plan. Return JSON:
{{
    "steps": [
        {{"tool": "tool_name", "args": {{...}}, "depends_on": [], "reasoning": "..."}},
        ...
    ]
}}

Rules:
- Tools with no dependencies can run in parallel
- Tools that depend on other tool results must run after
- Be thorough - use multiple sources to verify information
"""

        try:
            response = await self.model.structured_output_async(
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                system_prompt="You are a planning assistant. Create detailed plans.",
            )
            result = response.content
            steps = result.get("steps", [])

            return [
                ToolCall(
                    name=s["tool"],
                    args=s.get("args", {}),
                    dependencies=s.get("depends_on", []),
                    independent=len(s.get("depends_on", [])) == 0,
                )
                for s in steps
            ]
        except Exception:
            return []

    def group_for_execution(self, calls: list[ToolCall]) -> list[list[ToolCall]]:
        """Group tool calls into waves for optimal execution.

        Each wave contains calls whose dependencies are already satisfied.
        Calls within a wave can run in parallel.
        """
        groups: list[list[ToolCall]] = []
        completed: set[str] = set()
        remaining = list(calls)

        while remaining:
            ready = [
                c for c in remaining
                if all(dep in completed for dep in c.dependencies)
            ]

            if not ready:
                # Circular dependency or unresolvable — run remaining sequentially
                for c in remaining:
                    groups.append([c])
                break

            groups.append(ready)
            completed.update(c.name for c in ready)
            remaining = [c for c in remaining if c not in ready]

        return groups


# ---------------------------------------------------------------------------
# Observer: SmartAgentHook
# ---------------------------------------------------------------------------

class SmartAgentHook(HookProvider):
    """Intercepts tool results, stores full payloads to the session filesystem,
    and replaces the conversation result with a compact receipt so context
    stays lean.

    For results below the compaction threshold or from passthrough tools
    (session_grep, read_session_file, …) the original result passes through
    unmodified.
    """

    def __init__(
        self,
        streamer: AGUIStreamer,
        mode: ExecutionMode = ExecutionMode.AUTO,
        session_store: SessionStore | None = None,
    ):
        self._streamer = streamer
        self._mode = mode
        self._session_store = session_store
        self._compactor = ResultCompactor()
        self._completed_calls: list[str] = []

    # -- Strands HookProvider protocol --------------------------------------

    def register_hooks(self, registry, **kwargs) -> None:  # type: ignore[override]
        """Register with the Strands hook system."""
        registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)

    # -- event handler -------------------------------------------------------

    async def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """After tool call – stream UI events, store full result, compact."""
        tool_name: str = event.tool_use["name"]
        tool_args: dict[str, Any] = event.tool_use.get("input", {})
        result = event.result
        error = event.exception

        # 1. Emit streaming events for the frontend
        if isinstance(self._streamer, AGUIStreamer):
            await self._streamer.tool_call_end(tool_name)
            if error:
                await self._streamer.tool_result(tool_name, "", error=str(error))
            else:
                text = _extract_text(result)
                await self._streamer.tool_result(tool_name, text[:1000])

        if error:
            return

        self._completed_calls.append(tool_name)

        # 2. Decide whether to offload to the filesystem
        should_offload = (
            self._mode in (ExecutionMode.AGENTIC, ExecutionMode.AUTO)
            and self._session_store is not None
            and tool_name not in PASSTHROUGH_TOOLS
        )
        if not should_offload:
            return

        # 3. Store the full result on the session filesystem
        text = _extract_text(result)
        entry = await self._session_store.store_tool_result(
            tool_name=tool_name,
            tool_args=tool_args,
            result=text,
        )

        # 4. Replace the conversation result with a compact receipt
        if self._compactor.should_compact(tool_name, text):
            receipt = self._compactor.compact(entry.entry_id, text)
            compact_result: dict[str, Any] = {
                "content": [{"text": receipt.format()}],
            }
            # Preserve status and toolUseId when the original result is a dict
            if isinstance(result, dict):
                compact_result["status"] = result.get("status", "success")
                if "toolUseId" in result:
                    compact_result["toolUseId"] = result["toolUseId"]
            event.result = compact_result


def _extract_text(result: Any) -> str:
    """Pull plain text out of a Strands ToolResult (or fall back to str)."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content", [])
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
                elif "json" in block:
                    parts.append(json.dumps(block["json"], default=str))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(result)
    return str(result)


# ---------------------------------------------------------------------------
# Hybrid agent
# ---------------------------------------------------------------------------

def _cleanup_policy_for_mode(mode: ExecutionMode) -> CleanupPolicy:
    """Map execution mode to the appropriate cleanup policy (Strategy)."""
    if mode == ExecutionMode.FAST:
        return CleanupPolicy.ALWAYS
    return CleanupPolicy.ALWAYS  # agentic/auto: still cleanup after each request


class HybridChatbotAgent(BaseModel):
    """Hybrid agent with intelligent mode selection and execution.

    Features:
    - Auto mode selection (agent decides)
    - Parallel execution for independent tools
    - Sequential/iterative for dependent tools
    - Filesystem-based working memory with lifecycle cleanup
    - Complete, curated responses only
    """

    model_id: str = Field(default_factory=lambda: settings.bedrock.default_model)
    mode: ExecutionMode = Field(default=ExecutionMode.AUTO)
    session_id: str | None = Field(default=None)
    enable_mcp: bool = Field(default=True)

    max_parallel_tools: int = Field(default=5, description="Max parallel tools")
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
            import time
            return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        @tool
        async def session_grep(pattern: str, case_sensitive: bool = False) -> str:
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
                    f"  - {name}: {info['count']} results, {info['total_bytes']} bytes"
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

    async def _load_tools(self) -> list[str]:
        """Get list of available tool names."""
        tools = [
            t.name if hasattr(t, "name") else t.__name__
            for t in self._create_local_tools(None)
        ]

        if self.enable_mcp:
            async with create_mcp_manager() as mcp:
                all_tools = await mcp.discover_all()
                tools.extend(all_tools["tools"].keys())

        return tools

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

        from app.streaming import create_streamer

        async with create_streamer(request_id) as streamer:
            await streamer.run_started()

            # Step 1: Resolve mode
            analyzer = self._create_mode_analyzer()
            self._actual_mode, analysis = await analyzer.analyze_query(message)
            if self.mode != ExecutionMode.AUTO:
                self._actual_mode = self.mode

            await streamer.thinking_start(
                f"Mode: {self._actual_mode.value} - {analysis.get('reasoning', '')}"
            )

            # Determine cleanup policy for this run
            policy = _cleanup_policy_for_mode(self._actual_mode)
            needs_fs = (
                self._actual_mode in (ExecutionMode.AGENTIC, ExecutionMode.AUTO)
                and self.filesystem_enabled
                and self.session_id is not None
            )

            # Step 2: Enter lifecycle (creates + guarantees cleanup)
            lifecycle = SessionLifecycle(
                self.session_id or request_id, policy
            ) if needs_fs else None

            try:
                store: SessionStore | None = None
                if lifecycle:
                    store = await lifecycle.__aenter__()

                # Step 3: Plan tool calls
                available_tools = await self._load_tools()
                planner = self._create_planner()
                planned_calls = await planner.plan_execution(
                    message, available_tools, self._actual_mode
                )

                # Step 4: Build the Strands agent
                tools = self._create_local_tools(store)
                if self.enable_mcp:
                    async with create_mcp_manager() as mcp:
                        mcp_tools = await mcp.load_strands_tools()
                        tools.extend(mcp_tools)

                conv_manager = (
                    NullConversationManager()
                    if self._actual_mode == ExecutionMode.FAST
                    else SlidingWindowConversationManager(max_messages=100)
                )
                executor = (
                    None
                    if self._actual_mode == ExecutionMode.FAST or not planned_calls
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
                logger.info("Request %s cancelled (client disconnect)", request_id)
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
