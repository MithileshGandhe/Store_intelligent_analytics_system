"""
Store metrics router.

GET /stores/{store_id}/metrics — returns KPIs for a given store,
including unique visitors, conversion rate, dwell times, queue depth,
abandonment rate, transactions, and basket value.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.database import get_db
from app.models import MetricsResponse

logger = logging.getLogger("store_intelligence.metrics")

router = APIRouter(prefix="/stores", tags=["metrics"])


async def _store_exists(db, store_id: str) -> bool:
    """Return True if the store has at least one event on record."""
    val = await db.fetchval(
        "SELECT 1 FROM events WHERE store_id = $1 LIMIT 1", store_id
    )
    return val is not None


@router.get("/{store_id}/metrics", response_model=MetricsResponse)
async def get_metrics(
    store_id: str,
    date_str: Optional[str] = Query(None, alias="date", description="YYYY-MM-DD, default today"),
    start_time: Optional[str] = Query(None, description="HH:MM:SS"),
    end_time: Optional[str] = Query(None, description="HH:MM:SS"),
):
    """Compute store-level KPIs for the requested day / time window.

    All customer metrics exclude staff (is_staff = true).
    Zero-traffic scenarios return 0s rather than nulls or errors.
    """
    db = get_db()

    # Resolve target date
    target_date = date_str or date.today().isoformat()

    # Build time boundaries
    day_start = f"{target_date}T00:00:00"
    day_end = f"{target_date}T23:59:59"
    if start_time:
        day_start = f"{target_date}T{start_time}"
    if end_time:
        day_end = f"{target_date}T{end_time}"

    # Check if store exists
    exists = await _store_exists(db, store_id)
    if not exists:
        raise HTTPException(status_code=404, detail={"error": "store_not_found", "message": f"No data for store '{store_id}'"})

    # 1. Unique visitors (non-staff ENTRY events)
    unique_visitors = await db.fetchval(
        """
        SELECT COUNT(DISTINCT visitor_id)
        FROM events
        WHERE store_id = $1
          AND event_type = 'ENTRY'
          AND is_staff = 0
          AND timestamp BETWEEN $2 AND $3
        """,
        store_id, day_start, day_end,
    ) or 0

    # 2. Average dwell by zone
    dwell_rows = await db.fetch(
        """
        SELECT zone_id, AVG(dwell_ms) AS avg_dwell
        FROM events
        WHERE store_id = $1
          AND event_type = 'ZONE_DWELL'
          AND is_staff = 0
          AND timestamp BETWEEN $2 AND $3
          AND zone_id IS NOT NULL
        GROUP BY zone_id
        """,
        store_id, day_start, day_end,
    )
    avg_dwell_by_zone = {row["zone_id"]: round(float(row["avg_dwell"]), 2) for row in dwell_rows}

    # 3. Current queue depth — latest BILLING_QUEUE_JOIN meta_queue_depth
    queue_row = await db.fetchrow(
        """
        SELECT meta_queue_depth
        FROM events
        WHERE store_id = $1
          AND event_type = 'BILLING_QUEUE_JOIN'
          AND timestamp BETWEEN $2 AND $3
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        store_id, day_start, day_end,
    )
    current_queue_depth = 0
    if queue_row and queue_row.get("meta_queue_depth") is not None:
        current_queue_depth = int(queue_row["meta_queue_depth"])

    # 4. Abandonment rate
    joins = await db.fetchval(
        """
        SELECT COUNT(*) FROM events
        WHERE store_id = $1
          AND event_type = 'BILLING_QUEUE_JOIN'
          AND is_staff = 0
          AND timestamp BETWEEN $2 AND $3
        """,
        store_id, day_start, day_end,
    ) or 0

    abandons = await db.fetchval(
        """
        SELECT COUNT(*) FROM events
        WHERE store_id = $1
          AND event_type = 'BILLING_QUEUE_ABANDON'
          AND is_staff = 0
          AND timestamp BETWEEN $2 AND $3
        """,
        store_id, day_start, day_end,
    ) or 0

    abandonment_rate = round(abandons / joins, 4) if joins > 0 else 0.0

    # 5. POS transactions
    total_transactions = await db.fetchval(
        """
        SELECT COUNT(*) FROM pos_transactions
        WHERE store_id = $1
          AND timestamp BETWEEN $2 AND $3
        """,
        store_id, day_start, day_end,
    ) or 0

    avg_basket = await db.fetchval(
        """
        SELECT AVG(basket_value_inr) FROM pos_transactions
        WHERE store_id = $1
          AND timestamp BETWEEN $2 AND $3
        """,
        store_id, day_start, day_end,
    )
    avg_basket_value = round(float(avg_basket), 2) if avg_basket else 0.0

    # 6. Conversion rate — visitors who joined billing queue within 5 min before a POS txn
    conversion_rate = 0.0
    if unique_visitors > 0 and total_transactions > 0:
        converted = await db.fetchval(
            """
            SELECT COUNT(DISTINCT e.visitor_id)
            FROM events e
            JOIN pos_transactions p
              ON e.store_id = p.store_id
            WHERE e.store_id = $1
              AND e.event_type = 'BILLING_QUEUE_JOIN'
              AND e.is_staff = 0
              AND e.timestamp BETWEEN $2 AND $3
              AND p.timestamp BETWEEN $2 AND $3
              AND e.timestamp <= p.timestamp
              AND e.timestamp >= datetime(p.timestamp, '-5 minutes')
            """,
            store_id, day_start, day_end,
        ) or 0
        conversion_rate = round(converted / unique_visitors, 4)

    return MetricsResponse(
        store_id=store_id,
        date=target_date,
        unique_visitors=int(unique_visitors),
        conversion_rate=conversion_rate,
        avg_dwell_by_zone=avg_dwell_by_zone,
        current_queue_depth=current_queue_depth,
        abandonment_rate=abandonment_rate,
        total_transactions=int(total_transactions),
        avg_basket_value=avg_basket_value,
    )
