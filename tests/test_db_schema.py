"""
Tests for database schema initialization.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from app.db.schema import ensure_schema


def _make_mock_pool(mock_conn):
    """Create a mock pool with a proper async context manager."""
    mock_pool = AsyncMock()

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    mock_pool.acquire = _acquire
    return mock_pool


class TestEnsureSchema:
    @pytest.mark.asyncio
    async def test_executes_schema_sql(self):
        mock_conn = AsyncMock()
        mock_pool = _make_mock_pool(mock_conn)

        await ensure_schema(mock_pool)

        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS conversation_messages" in sql
        assert "CREATE INDEX IF NOT EXISTS" in sql

    @pytest.mark.asyncio
    async def test_schema_contains_required_columns(self):
        mock_conn = AsyncMock()
        mock_pool = _make_mock_pool(mock_conn)

        await ensure_schema(mock_pool)

        sql = mock_conn.execute.call_args[0][0]
        assert "session_id" in sql
        assert "role" in sql
        assert "content" in sql
        assert "JSONB" in sql
