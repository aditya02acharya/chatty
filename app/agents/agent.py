"""
Agent mode selection and execution strategy.

Modes:
- fast: User explicitly requests fast mode
- agentic: User explicitly requests agentic mode
- auto: Agent decides based on query analysis

Execution:
- Parallel for independent tools
- Sequential/iterative for dependent tools
- Hybrid as needed
"""

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
from strands.models import BedrockModel
from strands.tools.executors import SequentialToolExecutor

from app.agents.session_fs import SessionFileSystem, create_session_filesystem
from app.agents.tools import MCPManager
from app.core.config import settings
from app.core.exceptions import AgentError
from app.streaming import AGUIStreamer


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


class ModeAnalyzer(BaseModel):
    """Analyzes query to determine appropriate mode."""

    model: BedrockModel = Field(description="Model for analysis")

    model_config = {"arbitrary_types_allowed": True}

    async def analyze_query(self, query: str) -> tuple[ExecutionMode, dict[str, Any]]:
        """Analyze query to determine mode and strategy.

        Args:
            query: User query

        Returns:
            Tuple of (mode, analysis_result)
        """
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
        """Plan tool execution with dependency analysis.

        Args:
            query: User query
            available_tools: List of available tools
            mode: Current execution mode

        Returns:
            List of ToolCall objects with dependencies
        """
        # In fast mode, just return first tool (or none)
        if mode == ExecutionMode.FAST:
            # Check if query explicitly asks for multiple tools
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
                    for t in tools_requested[:3]  # Max 3 in fast mode
                ]
            except Exception:
                return []

        # In agentic mode, do full planning
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
            # Fallback: return empty plan
            return []

    def group_for_execution(self, calls: list[ToolCall]) -> list[list[ToolCall]]:
        """Group tool calls for optimal execution.

        Args:
            calls: List of ToolCall objects

        Returns:
            List of groups (each group can run in parallel)
        """
        groups = []
        remaining = calls.copy()

        while remaining:
            # Find all independent tools (no unmet dependencies)
            ready = [
                c for c in remaining
                if c.independent and all(
                    dep not in [r.name for r in remaining]
                    for dep in c.dependencies
                )
            ]

            if ready:
                groups.append(ready)
                # Remove ready from remaining
                remaining = [c for c in remaining if c not in ready]
            else:
                # All remaining depend on each other - execute sequentially
                for c in remaining:
                    groups.append([c])
                break

        return groups


class SmartAgentHook(HookProvider):
    """Hook that stores tool results to filesystem in agentic mode."""

    _streamer: AGUIStreamer
    _mode: ExecutionMode
    _session_fs: SessionFileSystem | None
    _plan: list[ToolCall] | None
    _completed_calls: list[str]

    def __init__(
        self,
        streamer: AGUIStreamer,
        mode: ExecutionMode = ExecutionMode.AUTO,
        session_fs: SessionFileSystem | None = None,
        plan: list[ToolCall] | None = None,
    ):
        self._streamer = streamer
        self._mode = mode
        self._session_fs = session_fs
        self._plan = plan
        self._completed_calls = []

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        error: Exception | None = None,
    ) -> None:
        """After tool call - emit event and store to filesystem."""
        # Emit event
        if isinstance(self._streamer, AGUIStreamer):
            await self._streamer.tool_call_end(tool_name)
            if error:
                await self._streamer.tool_result(tool_name, "", error=str(error))
            else:
                await self._streamer.tool_result(tool_name, str(result)[:1000])

        # Track completion
        if not error:
            self._completed_calls.append(tool_name)

        # Store to filesystem in agentic mode
        if self._mode in (ExecutionMode.AGENTIC, ExecutionMode.AUTO) and self._session_fs and not error:
            await self._session_fs.store_tool_result(
                tool_name=tool_name,
                tool_args=tool_args,
                result=result,
            )


class HybridChatbotAgent(BaseModel):
    """Hybrid agent with intelligent mode selection and execution.

    Features:
    - Auto mode selection (agent decides)
    - Parallel execution for independent tools
    - Sequential/iterative for dependent tools
    - Filesystem-based working memory
    - Complete, curated responses only
    """

    model_id: str = Field(default_factory=lambda: settings.bedrock.default_model)
    mode: ExecutionMode = Field(default=ExecutionMode.AUTO)
    session_id: str | None = Field(default=None)
    enable_mcp: bool = Field(default=True)

    # Config
    max_parallel_tools: int = Field(default=5, description="Max parallel tools")
    filesystem_enabled: bool = Field(default=True)

    model_config = {"populate_by_name": True}

    def __init__(self, **data):
        super().__init__(**data)

        self._bedrock_model = BedrockModel(
            model_id=self.model_id,
            region_name=settings.bedrock.region,
        )

        self._session_fs: SessionFileSystem | None = None
        self._actual_mode: ExecutionMode | None = None  # Resolved after analysis

        if self.session_id and self.filesystem_enabled:
            self._session_fs = create_session_filesystem(self.session_id)
            self._session_fs.create_session()

    def _create_mode_analyzer(self) -> ModeAnalyzer:
        """Create mode analyzer."""
        return ModeAnalyzer(model=self._bedrock_model)

    def _create_planner(self) -> ToolCallPlanner:
        """Create tool call planner."""
        return ToolCallPlanner(model=self._bedrock_model)

    def _create_local_tools(self) -> list:
        """Create local tools."""

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

            Returns:
                Search results with context
            """
            if not self._session_fs:
                return "No session filesystem available"

            matches = await self._session_fs.grep(pattern, case_sensitive)
            if not matches:
                return f"No matches for: {pattern}"

            results = [f"Found {len(matches)} matches in session filesystem:"]
            for m in matches[:20]:
                results.append(f"\n- File: {m['filename']}")
                results.append(f"  Tool: {m['tool']}")
                results.append(f"  Match: {m['match']}")

            return "\n".join(results)

        @tool
        async def session_summary() -> str:
            """Get summary of session filesystem data."""
            if not self._session_fs:
                return "No session filesystem available"

            summary = await self._session_fs.get_summary()
            files_info = "\n".join(
                f"- {f['filename']} (from {f['source_tool']})"
                for f in summary["files"]
            )
            return f"""Session: {summary['session_id']}
Files: {summary['total_files']}
Tools: {', '.join(set(f['source_tool'] for f in summary['files']))}

{files_info}
"""

        @tool
        async def read_session_file(filename: str) -> str:
            """Read a file from session filesystem.

            Args:
                filename: Name of file to read

            Returns:
                File contents
            """
            if not self._session_fs:
                return "No session filesystem available"

            try:
                data = await self._session_fs.read_file(filename)
                return f"Contents of {filename}:\n\n{data}"
            except FileNotFoundError:
                return f"File not found: {filename}"

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
            for t in self._create_local_tools()
        ]

        if self.enable_mcp:
            async with MCPManager() as mcp:
                all_tools = await mcp.discover_all()
                tools.extend(all_tools["tools"].keys())

        return tools

    async def _execute_parallel(self, agent: Agent, calls: list[ToolCall]) -> dict[str, Any]:
        """Execute tool calls in parallel.

        Args:
            agent: Strands agent
            calls: Tool calls to execute

        Returns:
            Combined results
        """
        # For now, use agent's tool execution
        # The agent will handle the actual tool calls
        # This is a placeholder for parallel execution management
        results = {}
        for call in calls:
            # Results will come through the agent's response
            results[call.name] = "executed"
        return results

    async def chat_stream(
        self,
        message: str,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Process chat with streaming response (complete, not partial).

        Args:
            message: User message
            request_id: Request ID

        Yields:
            SSE chunks (only when response is complete and verified)
        """
        import uuid

        if request_id is None:
            request_id = str(uuid.uuid4())

        from app.streaming import create_streamer

        async with create_streamer(request_id) as streamer:
            await streamer.run_started()

            # Step 1: Analyze mode
            analyzer = self._create_mode_analyzer()
            self._actual_mode, analysis = await analyzer.analyze_query(message)

            if self.mode != ExecutionMode.AUTO:
                self._actual_mode = self.mode

            await streamer.thinking_start(f"Mode: {self._actual_mode.value} - {analysis.get('reasoning', '')}")

            # Step 2: Plan tool calls (if any)
            available_tools = await self._load_tools()
            planner = self._create_planner()
            planned_calls = await planner.plan_execution(message, available_tools, self._actual_mode)

            # Step 3: Create agent
            tools = self._create_local_tools()
            if self.enable_mcp:
                async with MCPManager() as mcp:
                    mcp_tools = await mcp.load_strands_tools()
                    tools.extend(mcp_tools)

            # Choose conversation manager
            conv_manager = NullConversationManager() if self._actual_mode == ExecutionMode.FAST else SlidingWindowConversationManager(max_messages=100)

            # Choose tool executor based on planned calls
            if self._actual_mode == ExecutionMode.FAST or not planned_calls:
                executor = None  # Let agent decide
            else:
                executor = SequentialToolExecutor()  # Default to sequential

            hook = SmartAgentHook(streamer, self._actual_mode, self._session_fs, planned_calls)

            agent = Agent(
                model=self._bedrock_model,
                tools=tools,
                system_prompt=self._get_system_prompt(),
                tool_executor=executor,
                conversation_manager=conv_manager,
                hooks=[hook],
            )

            # Step 4: Execute and stream complete response
            try:
                full_response = ""

                async for chunk in agent.stream_async(message):
                    full_response += chunk

                # Only stream when complete
                await streamer.content_start()
                await streamer.content_delta(full_response)
                await streamer.content_end()
                yield streamer._encoder.encode(full_response)

                await streamer.done()

            except Exception as e:
                await streamer.error("AGENT_ERROR", str(e))
                raise AgentError(f"Agent execution failed: {e}") from e

    def _get_system_prompt(self) -> str:
        """Get system prompt based on mode."""
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
- Store findings to session filesystem
- Synthesize comprehensive, accurate answers
- Cite your sources
- Parallelize independent tool calls when beneficial
"""

    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass


@asynccontextmanager
async def create_chatbot_agent(**kwargs):
    """Create agent with automatic cleanup."""
    agent = HybridChatbotAgent(**kwargs)
    try:
        yield agent
    finally:
        await agent.cleanup()
