"""
event_converter.py — Convert between pipeline internal and production event schemas.

Pipeline internal events (from tracker.py):
    ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL,
    BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, REENTRY

Production events (from sample_eventsbe42122.jsonl):
    Type A: entry / exit  (with demographics)
    Type B: zone_entered / zone_exited  (with hotspot coords)
    Type C: queue_completed / queue_abandoned  (with wait/position)

Usage:
    converter = EventConverter(zone_lookup={"SKINCARE": {"name": "Skincare", "type": "SHELF"}})
    prod_event = converter.pipeline_to_production(pipeline_event, store_config)
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


# Mapping from pipeline event_type to production event_type
_PIPELINE_TO_PROD = {
    "ENTRY":                 "entry",
    "EXIT":                  "exit",
    "ZONE_ENTER":            "zone_entered",
    "ZONE_EXIT":             "zone_exited",
    "ZONE_DWELL":            "zone_dwell",
    "BILLING_QUEUE_JOIN":    "queue_joined",
    "BILLING_QUEUE_ABANDON": "queue_abandoned",
    "REENTRY":               "entry",  # Re-entry maps to entry in production
}

_PROD_TO_PIPELINE = {
    "entry":           "ENTRY",
    "exit":            "EXIT",
    "zone_entered":    "ZONE_ENTER",
    "zone_exited":     "ZONE_EXIT",
    "zone_dwell":      "ZONE_DWELL",
    "queue_completed": "BILLING_QUEUE_JOIN",  # approximation
    "queue_abandoned": "BILLING_QUEUE_ABANDON",
    "queue_joined":    "BILLING_QUEUE_JOIN",
}


class EventConverter:
    """Bidirectional event schema converter.

    Args:
        zone_lookup: dict mapping zone_id -> {"name": str, "type": str, "is_revenue": bool}
    """

    def __init__(self, zone_lookup: Optional[Dict[str, dict]] = None):
        self.zone_lookup = zone_lookup or {}

    @classmethod
    def from_layout(cls, layout: dict) -> "EventConverter":
        """Create converter from a store_layout.json dict."""
        zone_lookup = {}
        for zone in layout.get("zones", []):
            zid = zone["zone_id"]
            zone_lookup[zid] = {
                "name": zone.get("zone_name", zid),
                "type": zone.get("zone_type", "SHELF").upper(),
                "is_revenue": zone.get("is_revenue_zone", "No") == "Yes"
                              or zone.get("zone_type", "") in ("product", "billing"),
                "sku_zone": zone.get("sku_zone"),
            }
        return cls(zone_lookup=zone_lookup)

    def _get_zone_info(self, zone_id: Optional[str]) -> dict:
        """Look up zone metadata."""
        if zone_id and zone_id in self.zone_lookup:
            return self.zone_lookup[zone_id]
        return {"name": zone_id or "", "type": "UNKNOWN", "is_revenue": False, "sku_zone": None}

    # ------------------------------------------------------------------ #
    #  Pipeline → Production                                              #
    # ------------------------------------------------------------------ #

    def pipeline_to_production(self, event: dict, store_config: Optional[dict] = None) -> dict:
        """Convert a pipeline TrackerEvent dict to production schema.

        Args:
            event: Pipeline event dict with keys like event_type, visitor_id, etc.
            store_config: Optional store config with store_code, store_id, etc.

        Returns:
            Production-schema event dict.
        """
        etype = event.get("event_type", "")
        prod_type = _PIPELINE_TO_PROD.get(etype, etype.lower())

        store_config = store_config or {}
        store_code = store_config.get("store_code", event.get("store_id", ""))
        store_id = event.get("store_id", store_config.get("store_id", ""))
        camera_id = event.get("camera_id", "")
        timestamp = event.get("timestamp", "")
        visitor_id = event.get("visitor_id", "")
        zone_id = event.get("zone_id")
        confidence = event.get("confidence", 0.0)
        is_staff = event.get("is_staff", False)
        dwell_ms = event.get("dwell_ms")
        queue_depth = event.get("queue_depth")
        session_seq = event.get("session_seq", 0)

        # Format timestamp
        if isinstance(timestamp, (int, float)):
            ts_str = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%S.%f")
        else:
            ts_str = str(timestamp)

        zone_info = self._get_zone_info(zone_id)

        # Build event based on type category
        if prod_type in ("entry", "exit"):
            return {
                "event_type": prod_type,
                "id_token": visitor_id,
                "store_code": store_code,
                "camera_id": camera_id,
                "event_timestamp": ts_str,
                "is_staff": is_staff,
                "gender_pred": None,
                "age_pred": None,
                "age_bucket": None,
                "is_face_hidden": False,
                "group_id": None,
                "group_size": None,
            }

        elif prod_type in ("zone_entered", "zone_exited", "zone_dwell"):
            return {
                "event_type": prod_type,
                "track_id": session_seq,
                "store_id": store_id,
                "camera_id": camera_id,
                "zone_id": zone_id,
                "zone_name": zone_info["name"],
                "zone_type": zone_info["type"],
                "is_revenue_zone": "Yes" if zone_info.get("is_revenue") else "No",
                "event_time": ts_str,
                "zone_hotspot_x": None,
                "zone_hotspot_y": None,
                "gender": None,
                "age": None,
                "age_bucket": None,
                "dwell_ms": dwell_ms,
            }

        elif prod_type in ("queue_joined", "queue_completed", "queue_abandoned"):
            return {
                "queue_event_id": str(uuid.uuid4()),
                "event_type": prod_type,
                "track_id": session_seq,
                "store_id": store_id,
                "camera_id": camera_id,
                "zone_id": zone_id,
                "zone_name": zone_info["name"],
                "zone_type": "BILLING",
                "is_revenue_zone": "Yes",
                "queue_join_ts": ts_str if prod_type == "queue_joined" else None,
                "queue_served_ts": None,
                "queue_exit_ts": ts_str if prod_type != "queue_joined" else None,
                "wait_seconds": (dwell_ms // 1000) if dwell_ms else None,
                "queue_position_at_join": queue_depth,
                "abandoned": prod_type == "queue_abandoned",
                "zone_hotspot_x": None,
                "zone_hotspot_y": None,
                "gender": None,
                "age": None,
                "age_bucket": None,
            }

        else:
            # Fallback: pass through with minimal mapping
            return {
                "event_type": prod_type,
                "store_id": store_id,
                "camera_id": camera_id,
                "timestamp": ts_str,
                "visitor_id": visitor_id,
                "raw": event,
            }

    # ------------------------------------------------------------------ #
    #  Production → Pipeline                                              #
    # ------------------------------------------------------------------ #

    def production_to_pipeline(self, event: dict) -> dict:
        """Convert a production event dict to pipeline internal schema."""
        etype = event.get("event_type", "")
        pipeline_type = _PROD_TO_PIPELINE.get(etype, etype.upper())

        # Determine timestamp
        timestamp = (
            event.get("event_timestamp")
            or event.get("event_time")
            or event.get("queue_join_ts")
            or event.get("queue_exit_ts")
            or ""
        )

        # Determine visitor/track id
        visitor_id = event.get("id_token", event.get("track_id", ""))
        store_id = event.get("store_code", event.get("store_id", ""))

        result = {
            "event_type": pipeline_type,
            "visitor_id": str(visitor_id),
            "store_id": store_id,
            "camera_id": event.get("camera_id", ""),
            "timestamp": timestamp,
            "zone_id": event.get("zone_id"),
            "is_staff": event.get("is_staff", False),
            "confidence": 0.9,  # production events don't carry confidence
            "dwell_ms": event.get("dwell_ms") or (
                (event.get("wait_seconds", 0) or 0) * 1000 if "wait_seconds" in event else None
            ),
            "session_seq": event.get("track_id", 0),
        }

        # Queue-specific fields
        if "queue_position_at_join" in event:
            result["queue_depth"] = event["queue_position_at_join"]

        return result

    # ------------------------------------------------------------------ #
    #  Batch conversion                                                   #
    # ------------------------------------------------------------------ #

    def convert_batch(
        self,
        events: List[dict],
        direction: str = "pipeline_to_production",
        store_config: Optional[dict] = None,
    ) -> List[dict]:
        """Convert a batch of events.

        Args:
            events: List of event dicts.
            direction: "pipeline_to_production" or "production_to_pipeline".
            store_config: Store config for pipeline→production conversion.
        """
        if direction == "pipeline_to_production":
            return [self.pipeline_to_production(e, store_config) for e in events]
        elif direction == "production_to_pipeline":
            return [self.production_to_pipeline(e) for e in events]
        else:
            raise ValueError(f"Unknown direction: {direction}")
