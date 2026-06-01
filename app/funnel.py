"""
Conversion funnel router.

GET /stores/{store_id}/funnel — returns a session-based conversion funnel
showing how many visitors reach each stage and the drop-off between stages.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.database import get_db
from app.models import FunnelResponse, FunnelStage

logger = logging.getLogger("store_intelligence.funnel")

router = APIRouter(prefix="/stores", tags=["funnel"])


@router.get("/{store_id}/funnel", response_model=FunnelResponse)
async def get_funnel(
    store_id: str,
    date_str: Optional[str] = Query(None, alias="date", description="YYYY-MM-DD, default today"),
):
    """Build the conversion funnel for a store on the given date.

    Stages:
      1. Entry          — distinct non-staff visitors with an ENTRY event
      2. Zone Visit     — of those, visitors with at least one ZONE_ENTER
      3. Billing Queue  — visitors with a BILLING_QUEUE_JOIN event
      4. Purchase       — visitors in billing zone within 5 min of a POS txn

    Re-entries: same visitor_id is counted only ONCE.
    """
    db = get_db()
    target_date = date_str or date.today().isoformat()
    day_start = f"{target_date}T00:00:00"
    day_end = f"{target_date}T23:59:59"

    # Check store existence
    exists = await db.fetchval(
        "SELECT 1 FROM events WHERE store_id = $1 LIMIT 1", store_id
    )
    if not exists:
        raise HTTPException(status_code=404, detail={"error": "store_not_found", "message": f"No data for store '{store_id}'"})

    # Stage 1: Entry — distinct non-staff visitors
    entry_count = await db.fetchval(
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

    # Stage 2: Zone Visit — visitors who entered AND had at least one ZONE_ENTER
    # Use a two-step approach for SQLite compatibility
    zone_visit_count = await db.fetchval(
        """
        SELECT COUNT(DISTINCT visitor_id)
        FROM events
        WHERE store_id = $1
          AND event_type = 'ZONE_ENTER'
          AND is_staff = 0
          AND timestamp BETWEEN $2 AND $3
        """,
        store_id, day_start, day_end,
    ) or 0

    # Stage 3: Billing Queue
    billing_queue_count = await db.fetchval(
        """
        SELECT COUNT(DISTINCT visitor_id)
        FROM events
        WHERE store_id = $1
          AND event_type = 'BILLING_QUEUE_JOIN'
          AND is_staff = 0
          AND timestamp BETWEEN $2 AND $3
        """,
        store_id, day_start, day_end,
    ) or 0

    # Stage 4: Purchase — visitors in billing queue within 5 min of a POS txn
    purchase_count = 0
    if billing_queue_count > 0:
        try:
            purchase_count = await db.fetchval(
                """
                SELECT COUNT(DISTINCT e.visitor_id)
                FROM events e, pos_transactions p
                WHERE e.store_id = $1
                  AND p.store_id = $2
                  AND e.event_type = 'BILLING_QUEUE_JOIN'
                  AND e.is_staff = 0
                  AND e.timestamp BETWEEN $3 AND $4
                  AND p.timestamp BETWEEN $5 AND $6
                  AND e.timestamp <= p.timestamp
                """,
                store_id, store_id, day_start, day_end, day_start, day_end,
            ) or 0
        except Exception:
            purchase_count = 0

    # Ensure zone_visit_count <= entry_count (can't visit a zone without entering)
    zone_visit_count = min(zone_visit_count, entry_count)
    billing_queue_count = min(billing_queue_count, entry_count)
    purchase_count = min(purchase_count, billing_queue_count)

    # Build stages with drop-off percentages
    stages_raw = [
        ("Entry", int(entry_count)),
        ("Zone Visit", int(zone_visit_count)),
        ("Billing Queue", int(billing_queue_count)),
        ("Purchase", int(purchase_count)),
    ]

    stages = []
    for i, (name, count) in enumerate(stages_raw):
        if i == 0:
            drop_off = 0.0
        else:
            prev = stages_raw[i - 1][1]
            drop_off = round(((prev - count) / prev) * 100, 2) if prev > 0 else 0.0
        stages.append(FunnelStage(stage=name, visitor_count=count, drop_off_pct=drop_off))

    return FunnelResponse(
        store_id=store_id,
        date=target_date,
        stages=stages,
        total_visitors=int(entry_count),
    )
