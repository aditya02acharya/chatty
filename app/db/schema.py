"""Database schema initialisation.

Creates the conversation_messages table on first run.
Uses IF NOT EXISTS so it is safe to call on every startup.
"""

import logging

import asyncpg

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT        NOT NULL,
    role            TEXT        NOT NULL,
    content         JSONB       NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conv_session
    ON conversation_messages (session_id, id);

CREATE INDEX IF NOT EXISTS idx_conv_session_created
    ON conversation_messages (session_id, created_at);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Create tables and indexes if they don't exist."""
    async with pool.acquire() as conn:
        await conn.execute(_SCHEMA_SQL)
    logger.info("Database schema ensured")
