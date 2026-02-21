"""
Tests for PostgresConversationManager.

Uses mock asyncpg pool/connection to avoid requiring a real database.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.postgres_conversation_manager import (
    PostgresConversationManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pool(rows=None):
    """Create a mock asyncpg pool."""
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows or [])
    pool.execute = AsyncMock()
    return pool


def _make_row(row_id, role, content):
    """Create a mock database row."""
    return {
        "id": row_id,
        "role": role,
        "content": json.dumps(content),
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestPostgresConversationManagerInit:
    def test_creates_with_session_id(self):
        mgr = PostgresConversationManager("sess-001")
        assert mgr.session_id == "sess-001"
        assert mgr._history_loaded is False

    def test_inherits_window_size(self):
        mgr = PostgresConversationManager(
            "sess-001", window_size=50
        )
        assert mgr.window_size == 50


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------


class TestHistoryLoading:
    @pytest.mark.asyncio
    async def test_load_history_from_db(self):
        rows = [
            _make_row(1, "user", [{"text": "Hello"}]),
            _make_row(
                2, "assistant", [{"text": "Hi there!"}]
            ),
        ]
        mock_pool = _make_mock_pool(rows)

        mgr = PostgresConversationManager("sess-load")

        # Simulate agent with messages
        agent = MagicMock()
        agent.messages = [
            {
                "role": "user",
                "content": [{"text": "New message"}],
            }
        ]

        event = MagicMock()
        event.agent = agent

        with patch(
            "app.agents.postgres_conversation_manager.get_pool",
            return_value=mock_pool,
        ):
            await mgr._load_history_once(event)

        assert mgr._history_loaded is True
        # History should be prepended
        assert len(agent.messages) == 3
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == [
            {"text": "Hello"}
        ]
        assert agent.messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_load_history_only_once(self):
        mock_pool = _make_mock_pool([])

        mgr = PostgresConversationManager("sess-once")
        mgr._history_loaded = True

        event = MagicMock()
        event.agent = MagicMock()
        event.agent.messages = []

        with patch(
            "app.agents.postgres_conversation_manager.get_pool",
            return_value=mock_pool,
        ):
            await mgr._load_history_once(event)

        # Pool should not be called since history was already loaded
        mock_pool.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_history_empty_session(self):
        mock_pool = _make_mock_pool([])

        mgr = PostgresConversationManager("sess-empty")

        agent = MagicMock()
        agent.messages = [
            {
                "role": "user",
                "content": [{"text": "Hello"}],
            }
        ]

        event = MagicMock()
        event.agent = agent

        with patch(
            "app.agents.postgres_conversation_manager.get_pool",
            return_value=mock_pool,
        ):
            await mgr._load_history_once(event)

        assert mgr._history_loaded is True
        assert len(agent.messages) == 1  # No history prepended

    @pytest.mark.asyncio
    async def test_load_history_handles_db_error(self):
        mock_pool = AsyncMock()
        mock_pool.fetch = AsyncMock(
            side_effect=Exception("DB down")
        )

        mgr = PostgresConversationManager("sess-err")

        agent = MagicMock()
        agent.messages = []
        event = MagicMock()
        event.agent = agent

        with patch(
            "app.agents.postgres_conversation_manager.get_pool",
            return_value=mock_pool,
        ):
            # Should not raise
            await mgr._load_history_once(event)

        assert mgr._history_loaded is True


# ---------------------------------------------------------------------------
# Message persistence
# ---------------------------------------------------------------------------


class TestMessagePersistence:
    @pytest.mark.asyncio
    async def test_persist_message(self):
        mock_pool = _make_mock_pool()

        mgr = PostgresConversationManager("sess-persist")
        message = {
            "role": "user",
            "content": [{"text": "Hello"}],
        }

        with patch(
            "app.agents.postgres_conversation_manager.get_pool",
            return_value=mock_pool,
        ):
            await mgr.persist_message(message)

        mock_pool.execute.assert_called_once()
        call_args = mock_pool.execute.call_args
        assert call_args[0][1] == "sess-persist"
        assert call_args[0][2] == "user"

    @pytest.mark.asyncio
    async def test_persist_handles_error(self):
        mock_pool = AsyncMock()
        mock_pool.execute = AsyncMock(
            side_effect=Exception("DB write failed")
        )

        mgr = PostgresConversationManager("sess-perr")
        message = {
            "role": "user",
            "content": [{"text": "test"}],
        }

        with patch(
            "app.agents.postgres_conversation_manager.get_pool",
            return_value=mock_pool,
        ):
            # Should not raise
            await mgr.persist_message(message)


# ---------------------------------------------------------------------------
# State serialisation
# ---------------------------------------------------------------------------


class TestStateSerialization:
    def test_get_state_includes_session_id(self):
        mgr = PostgresConversationManager("sess-state")
        state = mgr.get_state()
        assert state["session_id"] == "sess-state"
        assert "__name__" in state

    def test_restore_from_session(self):
        mgr = PostgresConversationManager("sess-restore")
        state = {
            "__name__": "PostgresConversationManager",
            "removed_message_count": 5,
            "model_call_count": 3,
            "session_id": "sess-restored",
        }
        mgr.restore_from_session(state)
        assert mgr.session_id == "sess-restored"
        assert mgr.removed_message_count == 5


# ---------------------------------------------------------------------------
# Static helper
# ---------------------------------------------------------------------------


class TestLoadMessages:
    @pytest.mark.asyncio
    async def test_load_messages_static(self):
        rows = [
            _make_row(1, "user", [{"text": "Hi"}]),
            _make_row(2, "assistant", [{"text": "Hello"}]),
        ]
        mock_pool = _make_mock_pool(rows)

        messages = await PostgresConversationManager.load_messages(
            "sess-static", pool=mock_pool
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
