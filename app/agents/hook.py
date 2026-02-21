"""SmartAgentHook: intercepts tool calls for elicitation, compaction,
and storage.

Listens for BeforeToolCallEvent (elicitation) and AfterToolCallEvent
(compaction / storage) from the Strands hook system.

- **Elicitation**: when a tool call triggers an ``InterruptException``
  during the ``BeforeToolCallEvent``, the hook emits an AG-UI
  ``elicitation_request`` custom event so the frontend can prompt the
  user for the required information.

- **Compaction**: when a tool returns a large result in AGENTIC/AUTO
  mode, the full payload is stored on the session filesystem and the
  conversation result is replaced with a compact receipt (preview +
  gap analysis) so the context window stays lean.

- **Status**: emits AG-UI ``StateSnapshotEvent`` at key points so
  the frontend can show progress.
"""

import json
import logging
from typing import Any

from strands.hooks import HookProvider
from strands.hooks.events import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
)
from strands.interrupt import InterruptException

from app.agents.mode import ExecutionMode
from app.agents.result_compactor import PASSTHROUGH_TOOLS, ResultCompactor
from app.agents.session_fs import SessionStore
from app.streaming import AGUIStreamer

logger = logging.getLogger(__name__)


def extract_text(result: Any) -> str:
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
                    parts.append(
                        json.dumps(block["json"], default=str)
                    )
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(result)
    return str(result)


class SmartAgentHook(HookProvider):
    """Intercepts tool calls for elicitation, stores results to
    session filesystem, and replaces large results with compact
    receipts.

    For results below the compaction threshold or from passthrough
    tools (session_grep, read_session_file, ...) the original result
    passes through unmodified.
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

    # -- Strands HookProvider protocol ------------------------------------

    def register_hooks(self, registry, **kwargs) -> None:  # type: ignore[override]
        """Register with the Strands hook system."""
        registry.add_callback(
            BeforeToolCallEvent, self._on_before_tool_call
        )
        registry.add_callback(
            AfterToolCallEvent, self._on_after_tool_call
        )

    # -- before tool call (elicitation) -----------------------------------

    async def _on_before_tool_call(
        self, event: BeforeToolCallEvent
    ) -> None:
        """Before tool call: emit status and handle elicitation.

        When a tool's ``BeforeToolCallEvent`` triggers an
        ``InterruptException``, the hook emits an AG-UI
        elicitation_request so the frontend can prompt the user.
        """
        tool_name: str = event.tool_use["name"]
        tool_args: dict[str, Any] = event.tool_use.get("input", {})

        # Emit tool-call-start status
        if isinstance(self._streamer, AGUIStreamer):
            await self._streamer.status(
                "tool_call", f"Calling {tool_name}"
            )
            await self._streamer.tool_call_start(tool_name, tool_args)

    async def handle_elicitation(
        self,
        tool_name: str,
        interrupt: "InterruptException",
    ) -> None:
        """Emit an elicitation request to the frontend.

        Called externally when an InterruptException is caught
        during tool execution.

        Args:
            tool_name: The tool that raised the interrupt.
            interrupt: The InterruptException with details.
        """
        if not isinstance(self._streamer, AGUIStreamer):
            return

        intr = interrupt.interrupt
        await self._streamer.status(
            "elicitation",
            f"Tool '{tool_name}' needs user input",
        )
        await self._streamer.elicitation_request(
            tool_name=tool_name,
            interrupt_id=intr.id,
            reason=str(intr.reason) if intr.reason else (
                f"Tool '{tool_name}' requires additional input"
            ),
            schema=None,
        )

    # -- after tool call (compaction) -------------------------------------

    async def _on_after_tool_call(
        self, event: AfterToolCallEvent
    ) -> None:
        """After tool call: stream UI events, store, compact."""
        tool_name: str = event.tool_use["name"]
        tool_args: dict[str, Any] = event.tool_use.get("input", {})
        result = event.result
        error = event.exception

        # 1. Emit streaming events for the frontend
        if isinstance(self._streamer, AGUIStreamer):
            await self._streamer.tool_call_end(tool_name)
            if error:
                await self._streamer.tool_result(
                    tool_name, "", error=str(error)
                )
            else:
                text = extract_text(result)
                await self._streamer.tool_result(
                    tool_name, text[:1000]
                )

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

        # 3. Emit compaction status
        text = extract_text(result)
        if (
            isinstance(self._streamer, AGUIStreamer)
            and self._compactor.should_compact(tool_name, text)
        ):
            await self._streamer.status(
                "compacting",
                f"Compacting {tool_name} result",
            )

        # 4. Store the full result on the session filesystem
        entry = await self._session_store.store_tool_result(
            tool_name=tool_name,
            tool_args=tool_args,
            result=text,
        )

        # 5. Replace the conversation result with a compact receipt
        if self._compactor.should_compact(tool_name, text):
            receipt = self._compactor.compact(entry.entry_id, text)
            compact_result: dict[str, Any] = {
                "content": [{"text": receipt.format()}],
            }
            if isinstance(result, dict):
                compact_result["status"] = result.get(
                    "status", "success"
                )
                if "toolUseId" in result:
                    compact_result["toolUseId"] = result["toolUseId"]
            event.result = compact_result
