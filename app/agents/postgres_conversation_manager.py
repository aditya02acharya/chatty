"""PostgreSQL-backed conversation history manager.

Persists every message to PostgreSQL and restores history when
a session is resumed.  Extends `SlidingWindowConversationManager`
so the in-memory window logic (trimming, tool-pair safety) is
inherited -- this class adds the persistence layer on top.

The write path is fire-and-forget (runs in a background task) so
it never blocks the agent loop.  The read path loads from the
database into `agent.messages` before the first model call.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import asyncpg
from strands.agent.conversation_manager import (
    SlidingWindowConversationManager,
)
from strands.hooks import BeforeModelCallEvent, HookRegistry
from strands.types.content import Message

from app.db.pool import get_pool

if TYPE_CHECKING:
    from strands.agent.agent import Agent

logger = logging.getLogger(__name__)


class PostgresConversationManager(SlidingWindowConversationManager):
    """Conversation manager that persists messages to PostgreSQL.

    On construction, pass the ``session_id`` that identifies the
    conversation.  When the agent starts, any prior messages for
    that session are loaded from the database and prepended to
    ``agent.messages``.

    Every new message added during the agent loop is asynchronously
    written to the database so that history survives restarts.
    """

    def __init__(
        self,
        session_id: str,
        *,
        window_size: int = 100,
        should_truncate_results: bool = True,
        per_turn: bool | int = False,
    ):
        super().__init__(
            window_size=window_size,
            should_truncate_results=should_truncate_results,
            per_turn=per_turn,
        )
        self.session_id = session_id
        self._history_loaded = False
        self._seen_ids: set[int] = set()

    # -- hook registration ------------------------------------------------

    def register_hooks(
        self, registry: HookRegistry, **kwargs: Any
    ) -> None:
        """Register hooks for persistence and history loading."""
        super().register_hooks(registry, **kwargs)
        registry.add_callback(
            BeforeModelCallEvent,
            self._load_history_once,
        )

    # -- history loading --------------------------------------------------

    async def _load_history_once(
        self, event: BeforeModelCallEvent
    ) -> None:
        """Load prior messages from the DB before the first model call."""
        if self._history_loaded:
            return

        self._history_loaded = True
        agent = event.agent
        try:
            pool = await get_pool()
            rows = await pool.fetch(
                "SELECT id, role, content "
                "FROM conversation_messages "
                "WHERE session_id = $1 "
                "ORDER BY id",
                self.session_id,
            )
            if not rows:
                return

            restored: list[Message] = []
            for row in rows:
                self._seen_ids.add(row["id"])
                content = json.loads(row["content"])
                restored.append(
                    {"role": row["role"], "content": content}
                )

            # Prepend history before the current user message
            agent.messages[:0] = restored
            logger.info(
                "Loaded %d messages for session %s",
                len(restored),
                self.session_id,
            )
        except Exception:
            logger.exception(
                "Failed to load history for session %s",
                self.session_id,
            )

    # -- persistence -------------------------------------------------------

    async def persist_message(self, message: Message) -> None:
        """Write a single message to the database."""
        try:
            pool = await get_pool()
            await pool.execute(
                "INSERT INTO conversation_messages "
                "(session_id, role, content) "
                "VALUES ($1, $2, $3)",
                self.session_id,
                message["role"],
                json.dumps(
                    message["content"],
                    default=str,
                    ensure_ascii=False,
                ),
            )
        except Exception:
            logger.exception(
                "Failed to persist message for session %s",
                self.session_id,
            )

    def apply_management(
        self, agent: Agent, **kwargs: Any
    ) -> None:
        """Persist new messages then apply the sliding window."""
        # Persist any messages that haven't been written yet.
        # We detect new messages by checking the length vs what
        # we've already seen.
        for msg in agent.messages:
            key = id(msg)
            if key not in self._seen_ids:
                self._seen_ids.add(key)
                asyncio.ensure_future(self.persist_message(msg))

        super().apply_management(agent, **kwargs)

    # -- state serialisation -----------------------------------------------

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["session_id"] = self.session_id
        return state

    def restore_from_session(
        self, state: dict[str, Any]
    ) -> list[Message] | None:
        result = super().restore_from_session(state)
        self.session_id = state.get(
            "session_id", self.session_id
        )
        return result

    # -- class-level helpers -----------------------------------------------

    @staticmethod
    async def load_messages(
        session_id: str,
        pool: asyncpg.Pool | None = None,
    ) -> list[Message]:
        """Load all messages for a session (useful outside the agent)."""
        if pool is None:
            pool = await get_pool()
        rows = await pool.fetch(
            "SELECT role, content "
            "FROM conversation_messages "
            "WHERE session_id = $1 "
            "ORDER BY id",
            session_id,
        )
        return [
            {
                "role": row["role"],
                "content": json.loads(row["content"]),
            }
            for row in rows
        ]
