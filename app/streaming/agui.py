"""
ag-ui protocol SSE encoder and streamer.

Uses the official ag-ui-protocol library for event models and encoding.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ag_ui.core.events import (
    BaseEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    State,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageRole,
    TextMessageStartEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ThinkingTextMessageContentEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder.encoder import EventEncoder

from app.core.config import settings
from app.core.exceptions import StreamDisconnectedError


class _HeartbeatSentinel:
    """Internal marker pushed to the queue to emit an SSE heartbeat comment."""

    __slots__ = ("ts",)

    def __init__(self, ts: int):
        self.ts = ts


class AGUIStreamer:
    """ag-ui protocol event streamer.

    Buffers events and provides async iteration over SSE-encoded events.
    """

    def __init__(self, request_id: str, buffer_size: int = 8192):
        """Initialize the streamer.

        Args:
            request_id: Unique request identifier
                (maps to ag-ui thread_id/run_id)
            buffer_size: Maximum buffer size for events
        """
        self.request_id = request_id
        self.buffer_size = buffer_size
        self._queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue(
            maxsize=buffer_size
        )
        self._closed = False
        self._start_time = time.time()
        self._encoder = EventEncoder()
        self._iterations = 0
        self._last_event_time = time.time()

    async def emit(self, event: BaseEvent) -> None:
        """Emit an event to the stream.

        Args:
            event: Event to emit

        Raises:
            StreamDisconnectedError: If stream is closed
        """
        if self._closed:
            raise StreamDisconnectedError("Stream is closed")

        try:
            await asyncio.wait_for(self._queue.put(event), timeout=5.0)
        except asyncio.TimeoutError:
            raise StreamDisconnectedError(
                "Stream buffer full, client may be disconnected"
            )

    async def run_started(self) -> None:
        """Emit a run started event."""
        await self.emit(RunStartedEvent(thread_id=self.request_id))

    async def thinking_start(self, message: str | None = None) -> None:
        """Emit a thinking start event, optionally followed by content."""
        await self.emit(ThinkingStartEvent())
        if message:
            await self.emit(ThinkingTextMessageContentEvent(content=message))

    async def thinking_end(self) -> None:
        """Emit a thinking end event."""
        await self.emit(ThinkingEndEvent())

    async def step_started(self, step_name: str) -> None:
        """Emit a step started event."""
        await self.emit(StepStartedEvent(name=step_name))

    async def step_finished(self, step_name: str) -> None:
        """Emit a step finished event."""
        await self.emit(StepFinishedEvent(name=step_name))

    async def tool_call_start(self, tool_name: str, tool_args: dict) -> None:
        """Emit a tool call start event."""
        await self.emit(ToolCallStartEvent(tool_name=tool_name))
        await self.emit(ToolCallArgsEvent(args=tool_args))

    async def tool_call_end(self, tool_name: str) -> None:
        """Emit a tool call end event."""
        await self.emit(ToolCallEndEvent(tool_name=tool_name))

    async def tool_result(
        self, tool_name: str, result: str, error: str | None = None
    ) -> None:
        """Emit a tool result event."""
        await self.emit(
            ToolCallResultEvent(
                tool_name=tool_name, result=result, error=error
            )
        )

    async def content_start(self) -> None:
        """Emit a text message start event."""
        await self.emit(TextMessageStartEvent(role=TextMessageRole.AGENT))

    async def content_delta(self, content: str) -> None:
        """Emit a text message content event (delta)."""
        await self.emit(TextMessageContentEvent(content=content))

    async def content_end(self, full_content: str | None = None) -> None:
        """Emit a text message end event."""
        await self.emit(TextMessageEndEvent(role=TextMessageRole.AGENT))

    async def error(self, error_code: str, error_message: str) -> None:
        """Emit a run error event."""
        await self.emit(
            RunErrorEvent(error_message=f"{error_code}: {error_message}")
        )

    async def done(self, final_response: str | None = None) -> None:
        """Emit a done event and close the stream.

        Args:
            final_response: Optional final response content
        """
        duration_ms = (time.time() - self._start_time) * 1000

        # Emit final content if provided
        if final_response:
            await self.content_start()
            await self.content_delta(final_response)
            await self.content_end()

        # Create final state
        final_state = State(
            key="final",
            value={"iterations": self._iterations, "duration_ms": duration_ms},
        )

        await self.emit(
            RunFinishedEvent(
                thread_id=self.request_id,
                final_state=final_state,
                is_success=True,
            )
        )

        # Sentinel to signal end of stream
        await self._queue.put(None)
        self._closed = True

    def increment_iterations(self) -> None:
        """Increment the iteration counter."""
        self._iterations += 1

    @asynccontextmanager
    async def stream_context(self):
        """Context manager for the stream lifecycle.

        Yields:
            Self for method chaining
        """
        await self.run_started()
        try:
            yield self
        finally:
            if not self._closed:
                await self.done()

    async def __aiter__(self) -> AsyncIterator[str]:
        """Iterate over SSE-encoded events.

        Yields:
            SSE-formatted event strings
        """
        heartbeat_task = asyncio.create_task(self._heartbeat_generator())

        try:
            while True:
                item = await self._queue.get()

                # Sentinel for end of stream
                if item is None:
                    break

                # Heartbeat sentinel – emit SSE comment, not an encoded event
                if isinstance(item, _HeartbeatSentinel):
                    yield f": heartbeat {item.ts}\n\n"
                    continue

                self._last_event_time = time.time()
                yield self._encoder.encode(item)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_generator(self) -> None:
        """Push heartbeat sentinels to the queue when idle."""
        interval = settings.agui.heartbeat_interval

        while not self._closed:
            await asyncio.sleep(interval)

            # Only send heartbeat if no events have been sent recently
            idle_time = time.time() - self._last_event_time
            if idle_time >= interval:
                try:
                    await self._queue.put(
                        _HeartbeatSentinel(ts=int(time.time()))
                    )
                except asyncio.CancelledError:
                    break


@asynccontextmanager
async def create_streamer(
    request_id: str, buffer_size: int | None = None
) -> AsyncIterator[AGUIStreamer]:
    """Create a streamer with automatic cleanup.

    Args:
        request_id: Unique request identifier
        buffer_size: Optional buffer size override

    Yields:
        AGUIStreamer instance
    """
    if buffer_size is None:
        buffer_size = settings.agui.buffer_size

    streamer = AGUIStreamer(request_id, buffer_size)
    try:
        yield streamer
    finally:
        if not streamer._closed:
            await streamer.done()


class EventCollector:
    """Collects events in memory for non-streaming responses."""

    def __init__(self, request_id: str):
        """Initialize the collector.

        Args:
            request_id: Unique request identifier
        """
        self.request_id = request_id
        self.events: list[BaseEvent] = []
        self._start_time = time.time()
        self._iterations = 0
        self._encoder = EventEncoder()

    def collect(self, event: BaseEvent) -> BaseEvent:
        """Collect an event.

        Args:
            event: Event to collect

        Returns:
            The same event for chaining
        """
        self.events.append(event)
        return event

    def increment_iterations(self) -> None:
        """Increment the iteration counter."""
        self._iterations += 1

    @property
    def final_content(self) -> str:
        """Get the final concatenated content from all content events."""
        content_parts = []
        for e in self.events:
            if isinstance(e, TextMessageContentEvent):
                content_parts.append(e.content)
        return "".join(content_parts)

    @property
    def duration_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self._start_time) * 1000

    def to_sse_chunks(self) -> list[str]:
        """Convert all collected events to SSE chunks.

        Returns:
            List of SSE-formatted event strings
        """
        return [self._encoder.encode(e) for e in self.events]
