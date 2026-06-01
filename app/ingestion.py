"""
Event ingestion router.

POST /events/ingest — accepts batches of up to 500 events, inserts them
into the database with idempotency (ON CONFLICT DO NOTHING on event_id).
Supports partial success: valid events are accepted even if some are malformed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.database import get_db
from app.models import Event, IngestResponse

logger = logging.getLogger("store_intelligence.ingestion")

router = APIRouter(prefix="/events", tags=["ingestion"])

# Reference to the WebSocket manager — will be set by main.py at startup
_ws_manager = None


def set_ws_manager(manager):
    """Allow main.py to inject the WebSocket connection manager."""
    global _ws_manager
    _ws_manager = manager


_INSERT_EVENT = """
INSERT INTO events (
    event_id, store_id, camera_id, visitor_id, event_type,
    timestamp, zone_id, dwell_ms, is_staff, confidence,
    meta_queue_depth, meta_sku_zone, meta_session_seq
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
ON CONFLICT (event_id) DO NOTHING
"""


@router.post("/ingest", response_model=IngestResponse)
async def ingest_events(request: Request):
    """Ingest a batch of store events.

    • Max 500 events per request.
    • Each event is validated independently — one bad event does NOT
      fail the entire batch (partial success).
    • Duplicate event_ids are silently ignored (idempotent).
    """
    trace_id = request.state.trace_id if hasattr(request.state, "trace_id") else str(uuid.uuid4())

    # Parse raw JSON body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_json", "message": "Request body must be valid JSON", "trace_id": trace_id},
        )

    raw_events = body.get("events", [])
    if not isinstance(raw_events, list):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_format", "message": "'events' must be a list", "trace_id": trace_id},
        )

    if len(raw_events) > 500:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "batch_too_large",
                "message": f"Maximum 500 events per batch, got {len(raw_events)}",
                "trace_id": trace_id,
            },
        )

    db = get_db()
    accepted = 0
    rejected = 0
    errors: List[dict] = []
    affected_stores: set = set()

    for idx, raw in enumerate(raw_events):
        # Step 1: Validate event against Pydantic model
        try:
            event = Event.model_validate(raw)
        except (ValidationError, Exception) as exc:
            rejected += 1
            errors.append({
                "index": idx,
                "event_id": raw.get("event_id", "unknown"),
                "error": str(exc),
            })
            continue

        # Step 2: Insert into database
        try:
            ts = event.timestamp
            ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)

            await db.execute(
                _INSERT_EVENT,
                event.event_id,
                event.store_id,
                event.camera_id,
                event.visitor_id,
                event.event_type,
                ts_str,
                event.zone_id,
                event.dwell_ms,
                int(event.is_staff),  # SQLite stores bool as int
                event.confidence,
                event.metadata.queue_depth,
                event.metadata.sku_zone,
                event.metadata.session_seq,
            )
            accepted += 1
            affected_stores.add(event.store_id)
        except Exception as exc:
            rejected += 1
            errors.append({
                "index": idx,
                "event_id": event.event_id,
                "error": str(exc),
            })
            logger.warning(
                "Event rejected: idx=%d event_id=%s error=%s",
                idx, event.event_id, exc,
            )

    logger.info(
        "Ingest complete: trace_id=%s events=%d accepted=%d rejected=%d",
        trace_id, len(raw_events), accepted, rejected,
    )

    # Broadcast update to WebSocket clients for affected stores
    if _ws_manager and affected_stores:
        for store_id in affected_stores:
            try:
                await _ws_manager.broadcast_update(store_id)
            except Exception:
                logger.debug("WebSocket broadcast skipped for store %s", store_id)

    return IngestResponse(accepted=accepted, rejected=rejected, errors=errors)

