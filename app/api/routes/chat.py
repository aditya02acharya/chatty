"""
Chat endpoints for the chatbot API.

Provides streaming and non-streaming chat responses via SSE.
Handles client disconnects gracefully – the agent's session lifecycle
ensures filesystem cleanup regardless of how the request ends.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from enum import Enum

from ag_ui.core.events import RunErrorEvent
from ag_ui.encoder.encoder import EventEncoder
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents import ExecutionMode, create_chat_graph
from app.core.exceptions import AgentError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _graph_kwargs(request: "ChatRequest", **overrides) -> dict:
    """Build kwargs for create_chat_graph."""
    kwargs: dict = {
        "mode": ExecutionMode(request.mode.value),
        "enable_mcp": True,
        **overrides,
    }
    if request.model_id is not None:
        kwargs["model_id"] = request.model_id
    if request.session_id is not None:
        kwargs["session_id"] = request.session_id
    return kwargs


class ChatMode(str, Enum):
    """Chat execution mode."""

    FAST = "fast"
    AGENTIC = "agentic"
    AUTO = "auto"


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(description="User message", min_length=1)
    mode: ChatMode = Field(
        default=ChatMode.AUTO,
        description="Execution mode: fast, agentic, or auto",
    )
    model_id: str | None = Field(
        default=None, description="Optional model ID override"
    )
    session_id: str | None = Field(
        default=None, description="Session ID for agentic mode storage"
    )

    model_config = {"use_enum_values": True}


class ChatResponse(BaseModel):
    """Response model for non-streaming chat endpoint."""

    request_id: str = Field(description="Unique request identifier")
    response: str = Field(description="Agent response")
    mode: str = Field(description="Mode used for processing")
    iterations: int = Field(
        default=0, description="Number of agent iterations"
    )
    duration_ms: float = Field(description="Execution time in milliseconds")
    session_id: str | None = Field(
        default=None, description="Session ID if created"
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint using Server-Sent Events (SSE).

    Returns ag-ui protocol events as SSE chunks for real-time updates.
    The underlying agent manages its own session lifecycle – cleanup
    runs on success, error, *and* client disconnect (CancelledError).
    """
    request_id = str(uuid.uuid4())

    async def generate() -> AsyncIterator[str]:
        """Generate SSE chunks for the chat response."""
        encoder = EventEncoder()
        try:
            async with create_chat_graph(
                **_graph_kwargs(request)
            ) as graph:
                async for chunk in graph.chat_stream(
                    request.message, request_id
                ):
                    yield chunk

        except asyncio.CancelledError:
            # Starlette cancels the generator when the client disconnects.
            # The agent's finally block has already cleaned up the session.
            logger.info("Stream %s cancelled (client disconnect)", request_id)
            return

        except AgentError as e:
            yield encoder.encode(RunErrorEvent(error_message=str(e)))

        except Exception as e:
            # Cannot raise HTTPException inside a streaming generator
            # (headers already sent). Emit an SSE error event instead.
            logger.exception("Unexpected error in stream %s", request_id)
            yield encoder.encode(
                RunErrorEvent(error_message=f"Internal error: {e}")
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
    )


@router.post("/complete", response_model=ChatResponse)
async def chat_complete(request: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint.

    Returns the complete response after all processing is done.
    Session cleanup is handled by the agent lifecycle.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        kwargs = _graph_kwargs(request)
        kwargs.setdefault("session_id", request_id)
        async with create_chat_graph(**kwargs) as graph:
            response = await graph.chat_complete(
                request.message, request_id
            )

        duration_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            request_id=request_id,
            response=response,
            mode=request.mode.value,
            iterations=0,
            duration_ms=duration_ms,
            session_id=request.session_id or request_id,
        )
    except AgentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in complete %s", request_id)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "chatbot-api"}
