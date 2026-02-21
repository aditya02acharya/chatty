"""
Tests for AG-UI status and elicitation events.
"""

import asyncio

import pytest

from app.streaming.agui import AGUIStreamer


@pytest.fixture
def streamer():
    """Create a streamer for testing."""
    return AGUIStreamer("test-request-id", buffer_size=100)


class TestStatusEvents:
    @pytest.mark.asyncio
    async def test_status_emits_state_snapshot(self, streamer):
        await streamer.status("loading_history", "Loading data")

        # Pull the event from the queue
        event = await asyncio.wait_for(
            streamer._queue.get(), timeout=1.0
        )

        assert event is not None
        assert hasattr(event, "snapshot")
        # snapshot is a plain dict (State is typing.Any in ag-ui)
        assert event.snapshot["key"] == "status"
        assert event.snapshot["value"]["phase"] == "loading_history"
        assert event.snapshot["value"]["detail"] == "Loading data"

    @pytest.mark.asyncio
    async def test_status_with_empty_detail(self, streamer):
        await streamer.status("initialising")

        event = await asyncio.wait_for(
            streamer._queue.get(), timeout=1.0
        )

        assert event.snapshot["value"]["phase"] == "initialising"
        assert event.snapshot["value"]["detail"] == ""


class TestElicitationEvents:
    @pytest.mark.asyncio
    async def test_elicitation_request(self, streamer):
        await streamer.elicitation_request(
            tool_name="confirm_action",
            interrupt_id="int-001",
            reason="Please confirm deletion",
            schema={"type": "boolean"},
        )

        event = await asyncio.wait_for(
            streamer._queue.get(), timeout=1.0
        )

        assert hasattr(event, "name")
        assert event.name == "elicitation_request"
        assert event.value["tool_name"] == "confirm_action"
        assert event.value["interrupt_id"] == "int-001"
        assert event.value["reason"] == "Please confirm deletion"
        assert event.value["schema"] == {"type": "boolean"}

    @pytest.mark.asyncio
    async def test_elicitation_without_schema(self, streamer):
        await streamer.elicitation_request(
            tool_name="get_input",
            interrupt_id="int-002",
            reason="Need more info",
        )

        event = await asyncio.wait_for(
            streamer._queue.get(), timeout=1.0
        )

        assert event.value["schema"] is None
