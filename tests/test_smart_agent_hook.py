"""
Tests for SmartAgentHook: result interception, filesystem storage, and
compact receipt replacement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.agent import ExecutionMode, SmartAgentHook, _extract_text
from app.agents.result_compactor import MIN_COMPACT_SIZE, PASSTHROUGH_TOOLS
from app.agents.session_fs import SessionStore

# ---------------------------------------------------------------------------
# Helpers: lightweight stand-ins for Strands types
# ---------------------------------------------------------------------------


def _make_tool_result(text: str, tool_use_id: str = "tu-1") -> dict:
    """Build a minimal ToolResult dict."""
    return {
        "content": [{"text": text}],
        "status": "success",
        "toolUseId": tool_use_id,
    }


@dataclass
class FakeAfterToolCallEvent:
    """Mimics strands.hooks.events.AfterToolCallEvent for testing."""

    tool_use: dict[str, Any]
    result: dict[str, Any]
    invocation_state: dict[str, Any] = field(default_factory=dict)
    selected_tool: Any = None
    exception: Exception | None = None
    cancel_message: str | None = None
    retry: bool = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def streamer():
    """A mock AGUIStreamer."""
    s = AsyncMock()
    # Make isinstance check pass for AGUIStreamer
    s.__class__ = type("AGUIStreamer", (), {})
    return s


@pytest.fixture
def store(tmp_path):
    """A real SessionStore backed by tmp_path."""
    s = SessionStore("hook-test", base_path=tmp_path)
    s.create()
    return s


def _make_hook(streamer, mode=ExecutionMode.AGENTIC, store=None):
    """Create a SmartAgentHook with the given configuration."""
    return SmartAgentHook(streamer, mode=mode, session_store=store)


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_string_passthrough(self):
        assert _extract_text("hello") == "hello"

    def test_tool_result_text_block(self):
        tr = _make_tool_result("some output")
        assert _extract_text(tr) == "some output"

    def test_tool_result_json_block(self):
        tr = {
            "content": [{"json": {"key": "val"}}],
            "status": "success",
            "toolUseId": "x",
        }
        assert '"key"' in _extract_text(tr)

    def test_tool_result_multiple_blocks(self):
        tr = {
            "content": [{"text": "line1"}, {"text": "line2"}],
            "status": "success",
            "toolUseId": "x",
        }
        text = _extract_text(tr)
        assert "line1" in text
        assert "line2" in text

    def test_fallback_to_str(self):
        assert _extract_text(42) == "42"

    def test_empty_content_list(self):
        tr = {"content": [], "status": "success", "toolUseId": "x"}
        # Falls back to str(result) since no blocks
        result = _extract_text(tr)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# SmartAgentHook.register_hooks
# ---------------------------------------------------------------------------


class TestRegisterHooks:
    def test_registers_callback(self, streamer):
        hook = _make_hook(streamer)
        registry = MagicMock()
        hook.register_hooks(registry)
        registry.add_callback.assert_called_once()


# ---------------------------------------------------------------------------
# SmartAgentHook._on_after_tool_call – error path
# ---------------------------------------------------------------------------


class TestHookErrorPath:
    @pytest.mark.asyncio
    async def test_error_does_not_store(self, streamer, store):
        hook = _make_hook(streamer, store=store)
        event = FakeAfterToolCallEvent(
            tool_use={"name": "search", "input": {}, "toolUseId": "tu-1"},
            result=_make_tool_result(""),
            exception=RuntimeError("boom"),
        )
        await hook._on_after_tool_call(event)

        assert store.get_all() == []
        assert len(hook._completed_calls) == 0


# ---------------------------------------------------------------------------
# SmartAgentHook._on_after_tool_call – passthrough tools
# ---------------------------------------------------------------------------


class TestHookPassthrough:
    @pytest.mark.asyncio
    async def test_passthrough_tools_not_stored(self, streamer, store):
        hook = _make_hook(streamer, store=store)
        for tool_name in PASSTHROUGH_TOOLS:
            event = FakeAfterToolCallEvent(
                tool_use={"name": tool_name, "input": {}, "toolUseId": "tu-1"},
                result=_make_tool_result("x" * 1000),
            )
            await hook._on_after_tool_call(event)

        assert store.get_all() == []

    @pytest.mark.asyncio
    async def test_passthrough_result_unchanged(self, streamer, store):
        hook = _make_hook(streamer, store=store)
        original_result = _make_tool_result("original data")
        event = FakeAfterToolCallEvent(
            tool_use={
                "name": "session_grep",
                "input": {},
                "toolUseId": "tu-1",
            },
            result=original_result,
        )
        await hook._on_after_tool_call(event)
        assert event.result is original_result  # not replaced


# ---------------------------------------------------------------------------
# SmartAgentHook._on_after_tool_call – fast mode
# ---------------------------------------------------------------------------


class TestHookFastMode:
    @pytest.mark.asyncio
    async def test_fast_mode_does_not_store(self, streamer, store):
        hook = _make_hook(streamer, mode=ExecutionMode.FAST, store=store)
        event = FakeAfterToolCallEvent(
            tool_use={
                "name": "search",
                "input": {"q": "test"},
                "toolUseId": "tu-1",
            },
            result=_make_tool_result("x" * 1000),
        )
        await hook._on_after_tool_call(event)
        assert store.get_all() == []


# ---------------------------------------------------------------------------
# SmartAgentHook._on_after_tool_call – compaction
# ---------------------------------------------------------------------------


class TestHookCompaction:
    @pytest.mark.asyncio
    async def test_large_result_stored_and_compacted(self, streamer, store):
        hook = _make_hook(streamer, store=store)
        big_text = "x" * (MIN_COMPACT_SIZE + 100)
        event = FakeAfterToolCallEvent(
            tool_use={
                "name": "web_search",
                "input": {"q": "test"},
                "toolUseId": "tu-99",
            },
            result=_make_tool_result(big_text, tool_use_id="tu-99"),
        )

        await hook._on_after_tool_call(event)

        # Full result is on the filesystem
        entries = store.get_all()
        assert len(entries) == 1
        assert entries[0].tool_name == "web_search"

        # Conversation result was replaced with a compact receipt
        replaced_text = event.result["content"][0]["text"]
        assert "[session_fs:" in replaced_text
        assert "Preview:" in replaced_text
        assert "Gap:" in replaced_text
        assert "Retrieve:" in replaced_text

        # toolUseId preserved
        assert event.result["toolUseId"] == "tu-99"

    @pytest.mark.asyncio
    async def test_small_result_stored_but_not_compacted(
        self, streamer, store
    ):
        hook = _make_hook(streamer, store=store)
        small_text = "small result"
        original_result = _make_tool_result(small_text)
        event = FakeAfterToolCallEvent(
            tool_use={"name": "lookup", "input": {}, "toolUseId": "tu-1"},
            result=original_result,
        )

        await hook._on_after_tool_call(event)

        # Stored on filesystem
        assert len(store.get_all()) == 1

        # But the conversation result is NOT replaced (too small to compact)
        assert event.result is original_result

    @pytest.mark.asyncio
    async def test_no_store_means_no_compaction(self, streamer):
        hook = _make_hook(streamer, store=None)
        big_text = "x" * (MIN_COMPACT_SIZE + 100)
        original_result = _make_tool_result(big_text)
        event = FakeAfterToolCallEvent(
            tool_use={"name": "search", "input": {}, "toolUseId": "tu-1"},
            result=original_result,
        )

        await hook._on_after_tool_call(event)
        assert event.result is original_result


# ---------------------------------------------------------------------------
# SmartAgentHook._on_after_tool_call – completed_calls tracking
# ---------------------------------------------------------------------------


class TestCompletedCallsTracking:
    @pytest.mark.asyncio
    async def test_successful_call_tracked(self, streamer):
        hook = _make_hook(streamer)
        event = FakeAfterToolCallEvent(
            tool_use={"name": "search", "input": {}, "toolUseId": "tu-1"},
            result=_make_tool_result("ok"),
        )
        await hook._on_after_tool_call(event)
        assert "search" in hook._completed_calls

    @pytest.mark.asyncio
    async def test_errored_call_not_tracked(self, streamer):
        hook = _make_hook(streamer)
        event = FakeAfterToolCallEvent(
            tool_use={"name": "search", "input": {}, "toolUseId": "tu-1"},
            result=_make_tool_result(""),
            exception=RuntimeError("fail"),
        )
        await hook._on_after_tool_call(event)
        assert hook._completed_calls == []


# ---------------------------------------------------------------------------
# End-to-end: store → compact → retrieve via session tools
# ---------------------------------------------------------------------------


class TestEndToEndRetrieval:
    @pytest.mark.asyncio
    async def test_stored_data_retrievable_via_grep(self, streamer, store):
        """After compaction, the full data is on disk and searchable."""
        hook = _make_hook(streamer, store=store)

        weather_data = json.dumps(
            {
                "city": "London",
                "temperature": 20,
                "conditions": "partly cloudy",
                "humidity": 65,
                "wind_speed": 12,
                "forecast": [
                    {"day": "Mon", "high": 22},
                    {"day": "Tue", "high": 19},
                ],
            },
            indent=2,
        )
        # Pad to exceed compaction threshold
        weather_data += "\n" + " " * max(
            0, MIN_COMPACT_SIZE - len(weather_data) + 100
        )

        event = FakeAfterToolCallEvent(
            tool_use={
                "name": "get_weather",
                "input": {"city": "London"},
                "toolUseId": "tu-1",
            },
            result=_make_tool_result(weather_data),
        )
        await hook._on_after_tool_call(event)

        # Conversation got a compact receipt
        assert "[session_fs:" in event.result["content"][0]["text"]

        # But the full data is retrievable
        matches = await store.grep("London")
        assert len(matches) >= 1

        entry_id = store.get_all()[0].entry_id
        data = await store.read(entry_id)
        assert "London" in json.dumps(data)
