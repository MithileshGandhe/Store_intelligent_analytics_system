"""
Database layer for the Store Intelligence API.

Supports two backends:
  • PostgreSQL via asyncpg  (production / Docker)
  • SQLite    via aiosqlite (local development — default)

Set env var USE_SQLITE=true (or omit DATABASE_URL) to use SQLite.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, List, Optional, Tuple

logger = logging.getLogger("store_intelligence.database")

# ─── SQL DDL ────────────────────────────────────────────────────────────────────

_EVENTS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    store_id        TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    visitor_id      TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    timestamp       TIMESTAMP NOT NULL,
    zone_id         TEXT,
    dwell_ms        INTEGER DEFAULT 0,
    is_staff        BOOLEAN DEFAULT FALSE,
    confidence      REAL NOT NULL,
    meta_queue_depth INTEGER,
    meta_sku_zone   TEXT,
    meta_session_seq INTEGER DEFAULT 0
);
"""

_POS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS pos_transactions (
    transaction_id  TEXT PRIMARY KEY,
    store_id        TEXT NOT NULL,
    timestamp       TIMESTAMP NOT NULL,
    basket_value_inr REAL NOT NULL
);
"""

_INDEXES_PG = [
    "CREATE INDEX IF NOT EXISTS idx_events_store_ts ON events (store_id, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_events_store_type ON events (store_id, event_type);",
    "CREATE INDEX IF NOT EXISTS idx_events_visitor ON events (visitor_id);",
    "CREATE INDEX IF NOT EXISTS idx_pos_store_ts ON pos_transactions (store_id, timestamp);",
]

# SQLite uses the same DDL (BOOLEAN stored as 0/1, TIMESTAMP as TEXT).
_EVENTS_TABLE_SQLITE = _EVENTS_TABLE_PG
_POS_TABLE_SQLITE = _POS_TABLE_PG
_INDEXES_SQLITE = _INDEXES_PG


# ─── Abstract interface ─────────────────────────────────────────────────────────

class DatabaseBackend:
    """Minimal async interface that both PG and SQLite backends implement."""

    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def health_check(self) -> bool: ...
    async def execute(self, query: str, *args: Any) -> str: ...
    async def executemany(self, query: str, args_list: List[Tuple]) -> None: ...
    async def fetch(self, query: str, *args: Any) -> List[dict]: ...
    async def fetchrow(self, query: str, *args: Any) -> Optional[dict]: ...
    async def fetchval(self, query: str, *args: Any) -> Any: ...
    async def create_tables(self) -> None: ...


# ─── PostgreSQL backend ─────────────────────────────────────────────────────────

class PostgresBackend(DatabaseBackend):
    """asyncpg-based PostgreSQL backend with connection pooling."""

    def __init__(self, dsn: str, min_size: int = 5, max_size: int = 20):
        self._dsn = dsn
        self._min = min_size
        self._max = max_size
        self._pool = None

    async def connect(self) -> None:
        import asyncpg
        try:
            self._pool = await asyncpg.create_pool(
                self._dsn, min_size=self._min, max_size=self._max
            )
            logger.info("PostgreSQL pool created (%s–%s connections)", self._min, self._max)
        except Exception as exc:
            logger.error("Failed to connect to PostgreSQL: %s", exc)
            raise

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("PostgreSQL pool closed")

    async def health_check(self) -> bool:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def execute(self, query: str, *args: Any) -> str:
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def executemany(self, query: str, args_list: List[Tuple]) -> None:
        async with self._pool.acquire() as conn:
            await conn.executemany(query, args_list)

    async def fetch(self, query: str, *args: Any) -> List[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(r) for r in rows]

    async def fetchrow(self, query: str, *args: Any) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetchval(self, query: str, *args: Any) -> Any:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def create_tables(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_EVENTS_TABLE_PG)
            await conn.execute(_POS_TABLE_PG)
            for idx in _INDEXES_PG:
                await conn.execute(idx)
        logger.info("PostgreSQL tables & indexes ready")


# ─── SQLite backend ─────────────────────────────────────────────────────────────

class SQLiteBackend(DatabaseBackend):
    """aiosqlite-based SQLite backend for local development."""

    def __init__(self, db_path: str = "store_intelligence.db"):
        self._path = db_path
        self._db = None

    async def connect(self) -> None:
        import aiosqlite
        try:
            self._db = await aiosqlite.connect(self._path)
            self._db.row_factory = sqlite3.Row
            await self._db.execute("PRAGMA journal_mode=WAL;")
            await self._db.execute("PRAGMA foreign_keys=ON;")
            logger.info("SQLite connected: %s", self._path)
        except Exception as exc:
            logger.error("Failed to connect to SQLite: %s", exc)
            raise

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            logger.info("SQLite connection closed")

    async def health_check(self) -> bool:
        try:
            async with self._db.execute("SELECT 1") as cur:
                await cur.fetchone()
            return True
        except Exception:
            return False

    async def execute(self, query: str, *args: Any) -> str:
        # Convert $1, $2 … placeholders to ? for SQLite
        q = _pg_to_sqlite(query)
        await self._db.execute(q, args)
        await self._db.commit()
        return "OK"

    async def executemany(self, query: str, args_list: List[Tuple]) -> None:
        q = _pg_to_sqlite(query)
        await self._db.executemany(q, args_list)
        await self._db.commit()

    async def fetch(self, query: str, *args: Any) -> List[dict]:
        q = _pg_to_sqlite(query)
        async with self._db.execute(q, args) as cur:
            rows = await cur.fetchall()
            if not rows:
                return []
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in rows]

    async def fetchrow(self, query: str, *args: Any) -> Optional[dict]:
        q = _pg_to_sqlite(query)
        async with self._db.execute(q, args) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))

    async def fetchval(self, query: str, *args: Any) -> Any:
        q = _pg_to_sqlite(query)
        async with self._db.execute(q, args) as cur:
            row = await cur.fetchone()
            return row[0] if row else None

    async def create_tables(self) -> None:
        await self._db.execute(_EVENTS_TABLE_SQLITE)
        await self._db.execute(_POS_TABLE_SQLITE)
        for idx in _INDEXES_SQLITE:
            await self._db.execute(idx)
        await self._db.commit()
        logger.info("SQLite tables & indexes ready")


# ─── Placeholder conversion helper ──────────────────────────────────────────────

import re

def _pg_to_sqlite(query: str) -> str:
    """Convert PostgreSQL $1, $2 … placeholders to SQLite ? placeholders."""
    return re.sub(r'\$\d+', '?', query)


# ─── Singleton + factory ────────────────────────────────────────────────────────

_db: Optional[DatabaseBackend] = None


def _build_pg_dsn() -> Optional[str]:
    """Build a PostgreSQL DSN from environment variables."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("POSTGRES_HOST")
    if not host:
        return None
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "store_intelligence")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


async def init_db() -> DatabaseBackend:
    """Create and connect the database backend (called once at startup)."""
    global _db

    use_sqlite = os.getenv("USE_SQLITE", "true").lower() in ("true", "1", "yes")
    pg_dsn = _build_pg_dsn()

    if not use_sqlite and pg_dsn:
        _db = PostgresBackend(pg_dsn)
    else:
        db_path = os.getenv("SQLITE_PATH", "store_intelligence.db")
        _db = SQLiteBackend(db_path)

    await _db.connect()
    await _db.create_tables()
    return _db


async def close_db() -> None:
    """Shut down the database backend (called once at shutdown)."""
    global _db
    if _db:
        await _db.close()
        _db = None


def get_db() -> DatabaseBackend:
    """Return the current database backend (raises if not initialised)."""
    if _db is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _db


@asynccontextmanager
async def db_session() -> AsyncIterator[DatabaseBackend]:
    """Async context manager that yields the singleton backend.

    Useful for dependency injection in FastAPI routes.
    """
    backend = get_db()
    try:
        yield backend
    except Exception:
        logger.exception("Database operation failed")
        raise
