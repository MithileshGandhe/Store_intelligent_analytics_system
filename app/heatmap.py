"""
Zone heatmap router.

GET /stores/{store_id}/heatmap — returns per-zone traffic intensity,
average dwell time, and a normalised 0-100 score.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.database import get_db
from app.models import HeatmapResponse, ZoneHeatmapEntry

logger = logging.getLogger("store_intelligence.heatmap")

router = APIRouter(prefix="/stores", tags=["heatmap"])


@router.get("/{store_id}/heatmap", response_model=HeatmapResponse)
async def get_heatmap(
    store_id: str,
    date_str: Optional[str] = Query(None, alias="date", description="YYYY-MM-DD, default today"),
):
    """Generate a zone-level heatmap for the store.

    For each zone:
      • visit_count     — distinct non-staff visitors with ZONE_ENTER
      • avg_dwell_ms    — average dwell from ZONE_DWELL events
      • normalized_score — 0-100, proportional to busiest zone
      • data_confidence — HIGH if ≥20 sessions, LOW otherwise
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

    # Per-zone visit counts (distinct visitors)
    visit_rows = await db.fetch(
        """
        SELECT zone_id, COUNT(DISTINCT visitor_id) AS visit_count
        FROM events
        WHERE store_id = $1
          AND event_type = 'ZONE_ENTER'
          AND is_staff = 0
          AND timestamp BETWEEN $2 AND $3
          AND zone_id IS NOT NULL
        GROUP BY zone_id
        """,
        store_id, day_start, day_end,
    )

    # Per-zone average dwell
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

    # Per-zone total ZONE_ENTER event count (not distinct — for confidence)
    session_rows = await db.fetch(
        """
        SELECT zone_id, COUNT(*) AS session_count
        FROM events
        WHERE store_id = $1
          AND event_type = 'ZONE_ENTER'
          AND is_staff = 0
          AND timestamp BETWEEN $2 AND $3
          AND zone_id IS NOT NULL
        GROUP BY zone_id
        """,
        store_id, day_start, day_end,
    )

    # Merge into lookup dicts
    dwell_map = {row["zone_id"]: float(row["avg_dwell"]) for row in dwell_rows}
    session_map = {row["zone_id"]: int(row["session_count"]) for row in session_rows}

    # Collect all zone_ids
    all_zones = {row["zone_id"] for row in visit_rows}
    all_zones.update(dwell_map.keys())

    if not all_zones:
        return HeatmapResponse(store_id=store_id, date=target_date, zones=[])

    visit_map = {row["zone_id"]: int(row["visit_count"]) for row in visit_rows}
    max_visits = max(visit_map.values()) if visit_map else 1

    zones = []
    for zone_id in sorted(all_zones):
        visits = visit_map.get(zone_id, 0)
        avg_dwell = round(dwell_map.get(zone_id, 0.0), 2)
        normalized = round((visits / max_visits) * 100, 2) if max_visits > 0 else 0.0
        sessions = session_map.get(zone_id, 0)
        confidence = "HIGH" if sessions >= 20 else "LOW"

        zones.append(
            ZoneHeatmapEntry(
                zone_id=zone_id,
                visit_count=visits,
                avg_dwell_ms=avg_dwell,
                normalized_score=normalized,
                data_confidence=confidence,
            )
        )

    return HeatmapResponse(store_id=store_id, date=target_date, zones=zones)
