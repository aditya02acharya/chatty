"""
Chat endpoints for the chatbot API.

Provides streaming and non-streaming chat responses via SSE.
"""

import time
import uuid
from collections.abc import AsyncIterator
from enum import Enum

from ag_ui.core.events import RunErrorEvent
from ag_ui.encoder.encoder import EventEncoder
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents import ExecutionMode, create_chatbot_agent
from app.core.exceptions import AgentError

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMode(str, Enum):
    """Chat execution mode."""

    FAST = "fast"
    AGENTIC = "agentic"
    AUTO = "auto"


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(description="User message", min_length=1)
    mode: ChatMode = Field(default=ChatMode.AUTO, description="Execution mode: fast, agentic, or auto")
    model_id: str | None = Field(default=None, description="Optional model ID override")
    session_id: str | None = Field(default=None, description="Session ID for agentic mode storage")

    model_config = {"use_enum_values": True}


class ChatResponse(BaseModel):
    """Response model for non-streaming chat endpoint."""

    request_id: str = Field(description="Unique request identifier")
    response: str = Field(description="Agent response")
    mode: str = Field(description="Mode used for processing")
    iterations: int = Field(default=0, description="Number of agent iterations")
    duration_ms: float = Field(description="Execution time in milliseconds")
    session_id: str | None = Field(default=None, description="Session ID if created")


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint using Server-Sent Events (SSE).

    Returns ag-ui protocol events as SSE chunks for real-time updates.

    Args:
        request: Chat request with message and optional configuration

    Returns:
        StreamingResponse with SSE content type
    """
    request_id = str(uuid.uuid4())

    async def generate() -> AsyncIterator[str]:
        """Generate SSE chunks for the chat response."""
        try:
            async with create_chatbot_agent(
                mode=ExecutionMode(request.mode.value),
                model_id=request.model_id,
                session_id=request.session_id,
                enable_mcp=True,
            ) as agent:
                async for chunk in agent.chat_stream(request.message, request_id):
                    yield chunk
        except AgentError as e:
            error_event = RunErrorEvent(error_message=str(e))
            encoder = EventEncoder()
            yield encoder.encode(error_event)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

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

    Args:
        request: Chat request with message and optional configuration

    Returns:
        ChatResponse with complete response and metadata
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        async with create_chatbot_agent(
            mode=ExecutionMode(request.mode.value),
            model_id=request.model_id,
            session_id=request.session_id or request_id,  # Use request_id as session if not provided
            enable_mcp=True,
        ) as agent:
            response = await agent.chat_complete(request.message, request_id)

        duration_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            request_id=request_id,
            response=response,
            mode=request.mode.value,
            iterations=0,
            duration_ms=duration_ms,
            session_id=getattr(agent, "_session_id", None),
        )
    except AgentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "chatbot-api"}
