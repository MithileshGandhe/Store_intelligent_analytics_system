"""
Health-check router.

GET /health — system health status including database connectivity,
uptime, per-store event freshness, and stale-feed warnings.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter

from app.database import get_db
from app.models import HealthResponse

logger = logging.getLogger("store_intelligence.health")

router = APIRouter(tags=["health"])

# Set at import time; will be overwritten by main.py lifespan if desired
_start_time: float = time.time()

APP_VERSION = "1.0.0"


def set_start_time(t: float):
    global _start_time
    _start_time = t


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return system health.

    • Checks DB connectivity (SELECT 1).
    • Reports per-store last-event timestamps.
    • Warns about stores with stale feeds (>10 min lag).
    """
    db = get_db()
    uptime = time.time() - _start_time

    # Database health
    try:
        db_ok = await db.health_check()
    except Exception:
        db_ok = False

    db_status = "connected" if db_ok else "disconnected"

    # Per-store last event
    stores_map: dict[str, str | None] = {}
    warnings: list[str] = []

    if db_ok:
        try:
            rows = await db.fetch(
                """
                SELECT store_id, MAX(timestamp) AS last_ts
                FROM events
                GROUP BY store_id
                """
            )
            now = datetime.now(timezone.utc)
            for row in rows:
                sid = row["store_id"]
                last_ts = row["last_ts"]
                stores_map[sid] = str(last_ts) if last_ts else None

                # Stale feed warning
                if last_ts:
                    try:
                        if isinstance(last_ts, str):
                            dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                        else:
                            dt = last_ts
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        gap = (now - dt).total_seconds()
                        if gap > 600:
                            warnings.append(
                                f"STALE_FEED: store '{sid}' — last event {int(gap)}s ago"
                            )
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("Failed to query store status: %s", exc)

    # Overall status
    if not db_ok:
        status = "unhealthy"
    elif warnings:
        status = "degraded"
    else:
        status = "healthy"

    return HealthResponse(
        status=status,
        database=db_status,
        uptime_seconds=round(uptime, 2),
        stores=stores_map,
        warnings=warnings,
        version=APP_VERSION,
    )
