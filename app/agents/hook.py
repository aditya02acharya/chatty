"""
Smart agent hook for result interception and compaction.

Intercepts tool results via the Strands AfterToolCallEvent, stores full
payloads to the session filesystem, and replaces the conversation result
with a compact receipt so the context window stays lean.
"""

import json
from typing import Any

from strands.hooks import HookProvider
from strands.hooks.events import AfterToolCallEvent

from app.agents.execution_mode import ExecutionMode
from app.agents.result_compactor import PASSTHROUGH_TOOLS, ResultCompactor
from app.agents.session_fs import SessionStore
from app.streaming import AGUIStreamer


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
                await self._streamer.tool_result(
                    tool_name, "", error=str(error)
                )
            else:
                text = extract_text(result)
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
        text = extract_text(result)
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
                    parts.append(json.dumps(block["json"], default=str))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(result)
    return str(result)
