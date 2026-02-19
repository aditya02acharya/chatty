"""
Pytest configuration and fixtures.
"""

import asyncio
from collections.abc import Generator
from typing import Any

import pytest


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_bedrock_response() -> dict[str, Any]:
    """Mock Bedrock API response."""
    return {
        "content": [{"text": "This is a mock response from Bedrock."}],
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
    }


@pytest.fixture
def sample_mcp_tools() -> list[dict[str, Any]]:
    """Sample MCP tools for testing."""
    return [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
        {
            "name": "search",
            "description": "Search the web",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    ]
