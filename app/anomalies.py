"""
Anomaly detection router.

GET /stores/{store_id}/anomalies — scans for operational anomalies:
  • BILLING_QUEUE_SPIKE  — queue depth > 2× average
  • CONVERSION_DROP      — today's conversion < 70 % of 7-day avg
  • DEAD_ZONE            — zone with 0 visits in last 30 min
  • STALE_FEED           — no events from any camera in > 10 min
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.database import get_db
from app.models import Anomaly, AnomalyResponse

logger = logging.getLogger("store_intelligence.anomalies")

router = APIRouter(prefix="/stores", tags=["anomalies"])


@router.get("/{store_id}/anomalies", response_model=AnomalyResponse)
async def get_anomalies(
    store_id: str,
    date_str: Optional[str] = Query(None, alias="date", description="YYYY-MM-DD, default today"),
):
    """Detect operational anomalies for the given store."""
    db = get_db()
    target_date = date_str or date.today().isoformat()
    day_start = f"{target_date}T00:00:00"
    day_end = f"{target_date}T23:59:59"
    now = datetime.now(timezone.utc)

    # Check store existence
    exists = await db.fetchval(
        "SELECT 1 FROM events WHERE store_id = $1 LIMIT 1", store_id
    )
    if not exists:
        raise HTTPException(status_code=404, detail={"error": "store_not_found", "message": f"No data for store '{store_id}'"})

    anomalies: List[Anomaly] = []

    # ── 1. BILLING_QUEUE_SPIKE ──────────────────────────────────────────────
    await _check_queue_spike(db, store_id, day_start, day_end, anomalies)

    # ── 2. CONVERSION_DROP ──────────────────────────────────────────────────
    await _check_conversion_drop(db, store_id, target_date, day_start, day_end, anomalies)

    # ── 3. DEAD_ZONE ───────────────────────────────────────────────────────
    await _check_dead_zones(db, store_id, day_start, day_end, now, anomalies)

    # ── 4. STALE_FEED ──────────────────────────────────────────────────────
    await _check_stale_feed(db, store_id, now, anomalies)

    return AnomalyResponse(store_id=store_id, anomalies=anomalies)


# ─── Internal checks ────────────────────────────────────────────────────────────


async def _check_queue_spike(db, store_id, day_start, day_end, anomalies):
    """BILLING_QUEUE_SPIKE: current queue_depth > 2× day average."""
    # Average queue depth today
    avg_depth = await db.fetchval(
        """
        SELECT AVG(meta_queue_depth)
        FROM events
        WHERE store_id = $1
          AND event_type = 'BILLING_QUEUE_JOIN'
          AND meta_queue_depth IS NOT NULL
          AND timestamp BETWEEN $2 AND $3
        """,
        store_id, day_start, day_end,
    )
    if avg_depth is None or float(avg_depth) == 0:
        return

    avg_depth = float(avg_depth)

    # Latest queue depth
    latest = await db.fetchval(
        """
        SELECT meta_queue_depth
        FROM events
        WHERE store_id = $1
          AND event_type = 'BILLING_QUEUE_JOIN'
          AND meta_queue_depth IS NOT NULL
          AND timestamp BETWEEN $2 AND $3
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        store_id, day_start, day_end,
    )
    if latest is None:
        return

    latest = float(latest)
    ratio = latest / avg_depth if avg_depth > 0 else 0

    if ratio > 3:
        anomalies.append(Anomaly(
            type="BILLING_QUEUE_SPIKE",
            severity="CRITICAL",
            message=f"Queue depth {int(latest)} is {ratio:.1f}× the daily average ({avg_depth:.1f})",
            suggested_action="Open additional billing counters immediately",
            details={"current_depth": latest, "avg_depth": avg_depth, "ratio": round(ratio, 2)},
        ))
    elif ratio > 2:
        anomalies.append(Anomaly(
            type="BILLING_QUEUE_SPIKE",
            severity="WARN",
            message=f"Queue depth {int(latest)} is {ratio:.1f}× the daily average ({avg_depth:.1f})",
            suggested_action="Consider opening an additional billing counter",
            details={"current_depth": latest, "avg_depth": avg_depth, "ratio": round(ratio, 2)},
        ))


async def _check_conversion_drop(db, store_id, target_date, day_start, day_end, anomalies):
    """CONVERSION_DROP: today's conversion rate < 70% of 7-day average."""
    # Today's conversion rate
    visitors_today = await db.fetchval(
        """
        SELECT COUNT(DISTINCT visitor_id) FROM events
        WHERE store_id = $1 AND event_type = 'ENTRY' AND is_staff = 0
          AND timestamp BETWEEN $2 AND $3
        """,
        store_id, day_start, day_end,
    ) or 0

    if visitors_today == 0:
        return

    txn_today = await db.fetchval(
        """
        SELECT COUNT(*) FROM pos_transactions
        WHERE store_id = $1 AND timestamp BETWEEN $2 AND $3
        """,
        store_id, day_start, day_end,
    ) or 0

    conv_today = txn_today / visitors_today if visitors_today > 0 else 0

    # 7-day average conversion (simplified: txn count / visitor count)
    try:
        target = date.fromisoformat(target_date)
    except (ValueError, TypeError):
        return

    week_start = (target - timedelta(days=7)).isoformat() + "T00:00:00"
    week_end = (target - timedelta(days=1)).isoformat() + "T23:59:59"

    visitors_week = await db.fetchval(
        """
        SELECT COUNT(DISTINCT visitor_id) FROM events
        WHERE store_id = $1 AND event_type = 'ENTRY' AND is_staff = 0
          AND timestamp BETWEEN $2 AND $3
        """,
        store_id, week_start, week_end,
    ) or 0

    txn_week = await db.fetchval(
        """
        SELECT COUNT(*) FROM pos_transactions
        WHERE store_id = $1 AND timestamp BETWEEN $2 AND $3
        """,
        store_id, week_start, week_end,
    ) or 0

    if visitors_week == 0:
        return

    conv_week_avg = txn_week / visitors_week
    if conv_week_avg == 0:
        return

    if conv_today < 0.7 * conv_week_avg:
        anomalies.append(Anomaly(
            type="CONVERSION_DROP",
            severity="WARN",
            message=(
                f"Today's conversion rate ({conv_today:.2%}) is below 70% "
                f"of the 7-day average ({conv_week_avg:.2%})"
            ),
            suggested_action="Review store layout, staffing, and promotions",
            details={
                "conversion_today": round(conv_today, 4),
                "conversion_7day_avg": round(conv_week_avg, 4),
            },
        ))


async def _check_dead_zones(db, store_id, day_start, day_end, now, anomalies):
    """DEAD_ZONE: a known zone with 0 visits in the last 30 minutes."""
    # All known zones for the store
    known_zones = await db.fetch(
        """
        SELECT DISTINCT zone_id FROM events
        WHERE store_id = $1 AND zone_id IS NOT NULL
        """,
        store_id,
    )
    if not known_zones:
        return

    cutoff = (now - timedelta(minutes=30)).isoformat()
    now_str = now.isoformat()

    for row in known_zones:
        zone_id = row["zone_id"]
        recent = await db.fetchval(
            """
            SELECT COUNT(*) FROM events
            WHERE store_id = $1
              AND zone_id = $2
              AND event_type = 'ZONE_ENTER'
              AND timestamp BETWEEN $3 AND $4
            """,
            store_id, zone_id, cutoff, now_str,
        ) or 0

        if recent == 0:
            anomalies.append(Anomaly(
                type="DEAD_ZONE",
                severity="INFO",
                message=f"Zone '{zone_id}' has had 0 visits in the last 30 minutes",
                suggested_action=f"Check displays and signage in zone '{zone_id}'",
                details={"zone_id": zone_id},
            ))


async def _check_stale_feed(db, store_id, now, anomalies):
    """STALE_FEED: no events from any camera in > 10 minutes."""
    cutoff = (now - timedelta(minutes=10)).isoformat()

    latest = await db.fetchval(
        """
        SELECT MAX(timestamp) FROM events
        WHERE store_id = $1
        """,
        store_id,
    )
    if latest is None:
        return

    # Parse and compare
    try:
        if isinstance(latest, str):
            latest_dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
        else:
            latest_dt = latest
        if latest_dt.tzinfo is None:
            latest_dt = latest_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return

    gap = (now - latest_dt).total_seconds()
    if gap > 600:  # 10 minutes
        anomalies.append(Anomaly(
            type="STALE_FEED",
            severity="CRITICAL",
            message=f"No events received for {int(gap)}s (last event: {latest})",
            suggested_action="Check camera connectivity and network status",
            details={"last_event": str(latest), "gap_seconds": round(gap, 1)},
        ))
