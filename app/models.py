"""
Pydantic models for the Store Intelligence API.

Defines request/response schemas for event ingestion, metrics,
funnel analysis, heatmap, anomaly detection, health checks, and WebSocket messages.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Event Ingestion Models ────────────────────────────────────────────────────


class EventMetadata(BaseModel):
    """Additional metadata attached to an event."""
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int = 0


class Event(BaseModel):
    """A single camera/sensor event from a store."""
    event_id: str = Field(..., description="UUID v4 identifying this event (idempotency key)")
    store_id: str = Field(..., min_length=1)
    camera_id: str = Field(..., min_length=1)
    visitor_id: str = Field(..., min_length=1)
    event_type: Literal[
        "ENTRY", "EXIT",
        "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
        "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON",
        "REENTRY",
    ]
    timestamp: datetime
    zone_id: Optional[str] = None
    dwell_ms: int = Field(default=0, ge=0)
    is_staff: bool = False
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: EventMetadata = EventMetadata()

    @field_validator("event_type")
    @classmethod
    def zone_events_need_zone_id(cls, v: str, info) -> str:
        """Zone-scoped events should carry a zone_id (warning only)."""
        return v


class IngestRequest(BaseModel):
    """Batch of events to ingest (max 500)."""
    events: List[Event] = Field(..., max_length=500)


class IngestResponse(BaseModel):
    """Result of an ingestion batch."""
    accepted: int
    rejected: int
    errors: List[dict] = []


# ─── Metrics Models ────────────────────────────────────────────────────────────


class MetricsResponse(BaseModel):
    """Store-level KPIs for a given date/time range."""
    store_id: str
    date: str
    unique_visitors: int = 0
    conversion_rate: float = 0.0
    avg_dwell_by_zone: Dict[str, float] = {}
    current_queue_depth: int = 0
    abandonment_rate: float = 0.0
    total_transactions: int = 0
    avg_basket_value: float = 0.0
    staff_excluded: bool = True
    computed_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Funnel Models ──────────────────────────────────────────────────────────────


class FunnelStage(BaseModel):
    """One stage in the conversion funnel."""
    stage: str
    visitor_count: int = 0
    drop_off_pct: float = 0.0  # % lost from previous stage


class FunnelResponse(BaseModel):
    """Full conversion funnel for a store."""
    store_id: str
    date: str
    stages: List[FunnelStage] = []
    total_visitors: int = 0


# ─── Heatmap Models ─────────────────────────────────────────────────────────────


class ZoneHeatmapEntry(BaseModel):
    """Per-zone traffic and dwell summary."""
    zone_id: str
    visit_count: int = 0
    avg_dwell_ms: float = 0.0
    normalized_score: float = 0.0  # 0-100
    data_confidence: Literal["HIGH", "LOW"] = "LOW"


class HeatmapResponse(BaseModel):
    """Zone-level heatmap for a store."""
    store_id: str
    date: str
    zones: List[ZoneHeatmapEntry] = []


# ─── Anomaly Models ─────────────────────────────────────────────────────────────


class Anomaly(BaseModel):
    """A detected anomaly in store operations."""
    type: Literal[
        "BILLING_QUEUE_SPIKE",
        "CONVERSION_DROP",
        "DEAD_ZONE",
        "STALE_FEED",
    ]
    severity: Literal["INFO", "WARN", "CRITICAL"]
    message: str
    suggested_action: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, object] = {}


class AnomalyResponse(BaseModel):
    """List of detected anomalies for a store."""
    store_id: str
    anomalies: List[Anomaly] = []
    checked_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Health Models ───────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """System health status."""
    status: Literal["healthy", "degraded", "unhealthy"]
    database: Literal["connected", "disconnected"]
    uptime_seconds: float
    stores: Dict[str, Optional[str]] = {}  # store_id → last_event ISO
    warnings: List[str] = []
    version: str = "1.0.0"


# ─── WebSocket Models ───────────────────────────────────────────────────────────


class DashboardMessage(BaseModel):
    """A message pushed to the live dashboard WebSocket."""
    type: Literal["snapshot", "update", "heartbeat", "error"]
    store_id: str
    payload: dict = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Generic Error Model ────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Structured error returned by the API."""
    error: str
    message: str
    trace_id: Optional[str] = None
    retry_after: Optional[int] = None
