"""
WebSocket support for live dashboard updates.

WS /ws/dashboard/{store_id} — real-time metrics stream.

On connect:  sends a full metrics snapshot.
On events:   the ingestion layer calls broadcast_update() to push
             refreshed metrics to all clients watching that store.
Every 30 s:  heartbeat ping to keep the connection alive.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.database import get_db
from app.models import DashboardMessage

logger = logging.getLogger("store_intelligence.websocket")

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Track WebSocket clients grouped by store_id."""

    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}
        self._heartbeat_tasks: Dict[int, asyncio.Task] = {}

    async def connect(self, store_id: str, ws: WebSocket):
        await ws.accept()
        self._connections.setdefault(store_id, []).append(ws)
        logger.info("WS client connected: store=%s (total=%d)", store_id, len(self._connections[store_id]))

        # Start heartbeat
        task = asyncio.create_task(self._heartbeat(store_id, ws))
        self._heartbeat_tasks[id(ws)] = task

    def disconnect(self, store_id: str, ws: WebSocket):
        if store_id in self._connections:
            self._connections[store_id] = [
                c for c in self._connections[store_id] if c is not ws
            ]
            if not self._connections[store_id]:
                del self._connections[store_id]

        # Cancel heartbeat
        task = self._heartbeat_tasks.pop(id(ws), None)
        if task:
            task.cancel()
        logger.info("WS client disconnected: store=%s", store_id)

    async def broadcast_update(self, store_id: str):
        """Push a metrics snapshot to all clients watching *store_id*."""
        clients = self._connections.get(store_id, [])
        if not clients:
            return

        payload = await _build_snapshot(store_id)
        msg = DashboardMessage(
            type="update",
            store_id=store_id,
            payload=payload,
        )
        data = msg.model_dump_json()

        dead: List[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(store_id, ws)

    async def _heartbeat(self, store_id: str, ws: WebSocket):
        """Send a heartbeat message every 30 seconds."""
        try:
            while True:
                await asyncio.sleep(30)
                msg = DashboardMessage(
                    type="heartbeat",
                    store_id=store_id,
                    payload={"ts": datetime.now(timezone.utc).isoformat()},
                )
                await ws.send_text(msg.model_dump_json())
        except (asyncio.CancelledError, WebSocketDisconnect, Exception):
            pass


# Module-level singleton
manager = ConnectionManager()


@router.websocket("/ws/dashboard/{store_id}")
async def dashboard_ws(ws: WebSocket, store_id: str):
    """WebSocket endpoint for the live store dashboard."""
    await manager.connect(store_id, ws)

    # Send initial snapshot
    try:
        payload = await _build_snapshot(store_id)
        initial = DashboardMessage(
            type="snapshot",
            store_id=store_id,
            payload=payload,
        )
        await ws.send_text(initial.model_dump_json())
    except Exception as exc:
        logger.warning("Failed to send initial snapshot: %s", exc)

    # Keep connection open, listen for client messages (ignored for now)
    try:
        while True:
            _ = await ws.receive_text()  # client can send, we just keep alive
    except WebSocketDisconnect:
        manager.disconnect(store_id, ws)
    except Exception:
        manager.disconnect(store_id, ws)


async def _build_snapshot(store_id: str) -> dict:
    """Build a lightweight metrics snapshot for the WebSocket payload."""
    try:
        db = get_db()

        visitors = await db.fetchval(
            """
            SELECT COUNT(DISTINCT visitor_id) FROM events
            WHERE store_id = $1 AND event_type = 'ENTRY' AND is_staff = 0
            """,
            store_id,
        ) or 0

        queue_row = await db.fetchrow(
            """
            SELECT meta_queue_depth FROM events
            WHERE store_id = $1 AND event_type = 'BILLING_QUEUE_JOIN'
              AND meta_queue_depth IS NOT NULL
            ORDER BY timestamp DESC LIMIT 1
            """,
            store_id,
        )
        queue = int(queue_row["meta_queue_depth"]) if queue_row and queue_row.get("meta_queue_depth") else 0

        txn_count = await db.fetchval(
            "SELECT COUNT(*) FROM pos_transactions WHERE store_id = $1",
            store_id,
        ) or 0

        return {
            "unique_visitors": int(visitors),
            "current_queue_depth": queue,
            "total_transactions": int(txn_count),
            "snapshot_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.warning("Snapshot build failed: %s", exc)
        return {"error": str(exc)}
