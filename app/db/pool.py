"""asyncpg connection pool management."""

import logging

import asyncpg

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    """Initialise the global connection pool.

    Safe to call multiple times -- returns the existing pool if
    already created.
    """
    global _pool
    if _pool is not None:
        return _pool

    logger.info("Creating PostgreSQL pool: %s", settings.postgres.dsn)
    _pool = await asyncpg.create_pool(
        dsn=settings.postgres.dsn,
        min_size=settings.postgres.min_pool_size,
        max_size=settings.postgres.max_pool_size,
        statement_cache_size=settings.postgres.statement_cache_size,
    )
    return _pool


async def get_pool() -> asyncpg.Pool:
    """Return the global pool, initialising it if needed."""
    if _pool is None:
        return await init_pool()
    return _pool


async def close_pool() -> None:
    """Gracefully close the pool (call on shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL pool closed")
