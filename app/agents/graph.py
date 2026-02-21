"""Multi-agent chat graph using strands GraphBuilder.

Architecture
------------
The graph routes queries through specialised agents based on a
supervisor's classification:

    AUTO mode:
        supervisor ──ANSWER/CLARIFY──► responder ──► END
        supervisor ──CALL_TOOL──────► researcher ──► END

    FAST mode  :  responder only  (single-node graph)
    AGENTIC mode: researcher only (single-node graph)

The supervisor is a lightweight LLM call that returns one word
(ANSWER, CALL_TOOL, or CLARIFY).  The responder handles direct
answers and clarifications.  The researcher uses tools, session
filesystem, and the SmartAgentHook for result compaction.  The
Strands Agent's native tool loop handles the internal
execute → reflect → need-more cycle inside the researcher.

Conversation history is persisted to PostgreSQL when a session_id
is provided, using ``PostgresConversationManager``.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from strands.models import BedrockModel
from strands.multiagent import GraphBuilder
from strands.multiagent.base import Status
from strands.multiagent.graph import GraphState

from app.agents.agents import (
    create_researcher,
    create_responder,
    create_supervisor,
)
from app.agents.hook import SmartAgentHook
from app.agents.local_tools import create_session_tools
from app.agents.mode import ExecutionMode
from app.agents.postgres_conversation_manager import (
    PostgresConversationManager,
)
from app.agents.session_fs import (
    CleanupPolicy,
    SessionLifecycle,
    SessionStore,
)
from app.agents.tools import create_mcp_manager
from app.core.config import settings
from app.core.exceptions import AgentError
from app.streaming import create_streamer

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Condition helpers for graph edges
# -------------------------------------------------------------------


def _wants_responder(state: GraphState) -> bool:
    """Route to responder when supervisor says ANSWER or CLARIFY."""
    result = state.results.get("supervisor")
    if result and result.status == Status.COMPLETED:
        text = str(result.result).strip().upper()
        return "ANSWER" in text or "CLARIFY" in text
    return False


def _wants_researcher(state: GraphState) -> bool:
    """Route to researcher when supervisor says CALL_TOOL."""
    result = state.results.get("supervisor")
    if result and result.status == Status.COMPLETED:
        text = str(result.result).strip().upper()
        return "CALL_TOOL" in text
    # Default to researcher when classification is unclear
    return True


# -------------------------------------------------------------------
# Graph builder
# -------------------------------------------------------------------


def build_chat_graph(mode, supervisor, responder, researcher):
    """Construct the strands Graph for the given execution mode.

    AUTO    → supervisor routes to responder or researcher.
    FAST    → responder only (no supervisor overhead).
    AGENTIC → researcher only (no supervisor overhead).
    """
    builder = GraphBuilder()

    if mode == ExecutionMode.AUTO:
        builder.add_node(supervisor, "supervisor")
        builder.add_node(responder, "responder")
        builder.add_node(researcher, "researcher")
        builder.add_edge(
            "supervisor", "responder", condition=_wants_responder
        )
        builder.add_edge(
            "supervisor", "researcher", condition=_wants_researcher
        )
        builder.set_entry_point("supervisor")

    elif mode == ExecutionMode.FAST:
        builder.add_node(responder, "responder")
        builder.set_entry_point("responder")

    else:  # AGENTIC
        builder.add_node(researcher, "researcher")
        builder.set_entry_point("researcher")
        builder.set_execution_timeout(120)

    return builder.build()


# -------------------------------------------------------------------
# Response extraction
# -------------------------------------------------------------------


def _extract_response(result) -> str:
    """Extract the text response from the terminal graph node."""
    for node_id in ("researcher", "responder"):
        node_result = result.results.get(node_id)
        if node_result and node_result.status == Status.COMPLETED:
            return str(node_result.result).strip()

    # Fallback: last completed node
    if result.execution_order:
        last_id = result.execution_order[-1].node_id
        if last_id in result.results:
            return str(result.results[last_id].result).strip()

    return "Unable to generate a response."


# -------------------------------------------------------------------
# ChatGraph — public interface
# -------------------------------------------------------------------


class ChatGraph:
    """Multi-agent chat system backed by a strands Graph.

    This replaces HybridChatbotAgent with a cleaner multi-agent
    setup.  The public API (``chat_stream`` / ``chat_complete``)
    stays the same so the FastAPI routes need minimal changes.
    """

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.AUTO,
        model_id: str | None = None,
        session_id: str | None = None,
        enable_mcp: bool = True,
        filesystem_enabled: bool = True,
    ):
        self.mode = mode
        self.model_id = model_id or settings.bedrock.default_model
        self.session_id = session_id
        self.enable_mcp = enable_mcp
        self.filesystem_enabled = filesystem_enabled
        self._model = BedrockModel(
            model_id=self.model_id,
            region_name=settings.bedrock.region,
        )

    # -- streaming entry point -------------------------------------------

    async def chat_stream(
        self,
        message: str,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Process a chat message and yield the response.

        Handles three exit paths:
        1. Success  — stream completes, lifecycle cleans up.
        2. Error    — exception propagates, lifecycle cleans up.
        3. Cancel   — CancelledError caught, lifecycle cleans up.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        async with create_streamer(request_id) as streamer:
            await streamer.run_started()
            await streamer.status(
                "initialising", f"Mode: {self.mode.value}"
            )
            await streamer.thinking_start(
                f"Mode: {self.mode.value}"
            )

            # Determine whether the session filesystem is needed
            needs_fs = (
                self.mode
                in (ExecutionMode.AGENTIC, ExecutionMode.AUTO)
                and self.filesystem_enabled
                and self.session_id is not None
            )
            lifecycle = (
                SessionLifecycle(
                    self.session_id or request_id,
                    CleanupPolicy.ALWAYS,
                )
                if needs_fs
                else None
            )

            try:
                store: SessionStore | None = None
                if lifecycle:
                    store = await lifecycle.__aenter__()

                # --- build tools --------------------------------
                await streamer.status(
                    "loading_tools", "Loading tools"
                )
                tools = create_session_tools(store)
                if self.enable_mcp:
                    async with create_mcp_manager() as mcp:
                        mcp_tools = await mcp.load_strands_tools()
                        tools.extend(mcp_tools)

                # --- build conversation manager -----------------
                conv_manager = None
                if self.session_id:
                    await streamer.status(
                        "loading_history",
                        "Loading conversation history",
                    )
                    conv_manager = PostgresConversationManager(
                        session_id=self.session_id,
                        window_size=100,
                    )

                # --- build agents -------------------------------
                await streamer.status(
                    "building_agents", "Building agents"
                )
                hook = SmartAgentHook(
                    streamer, self.mode, store
                )
                supervisor = create_supervisor(self._model)
                responder = create_responder(self._model, tools)
                researcher = create_researcher(
                    self._model,
                    tools,
                    [hook],
                    conversation_manager=conv_manager,
                )

                # --- build & run graph --------------------------
                await streamer.status(
                    "processing", "Processing query"
                )
                graph = build_chat_graph(
                    self.mode,
                    supervisor,
                    responder,
                    researcher,
                )
                result = await graph.invoke_async(message)
                response = _extract_response(result)

                # --- stream the response to the client ----------
                await streamer.status(
                    "responding", "Streaming response"
                )
                await streamer.content_start()
                await streamer.content_delta(response)
                await streamer.content_end()
                await streamer.done()

                yield response

            except asyncio.CancelledError:
                logger.info(
                    "Request %s cancelled (client disconnect)",
                    request_id,
                )
                if lifecycle:
                    lifecycle.mark_error()
                raise

            except Exception as e:
                if lifecycle:
                    lifecycle.mark_error()
                await streamer.error("AGENT_ERROR", str(e))
                raise AgentError(
                    f"Agent execution failed: {e}"
                ) from e

            finally:
                if lifecycle:
                    await lifecycle.__aexit__(None, None, None)

    # -- non-streaming entry point ---------------------------------------

    async def chat_complete(
        self, message: str, request_id: str
    ) -> str:
        """Non-streaming chat — collects and returns the full response."""
        full = ""
        async for chunk in self.chat_stream(message, request_id):
            full += chunk
        return full

    # -- cleanup ---------------------------------------------------------

    async def cleanup(self) -> None:
        """Cleanup agent-level resources."""
        pass


@asynccontextmanager
async def create_chat_graph(**kwargs):
    """Create a ChatGraph with automatic cleanup."""
    graph = ChatGraph(**kwargs)
    try:
        yield graph
    finally:
        await graph.cleanup()
