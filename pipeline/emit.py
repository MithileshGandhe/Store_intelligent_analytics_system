"""
emit.py — EventEmitter: converts tracker events into structured JSON events.

Validates events with Pydantic, writes to JSONL files, and optionally
POSTs batches to an API endpoint.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    # Fallback: minimal validation if pydantic is not installed
    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> dict:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    class Field:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

    def field_validator(*a: Any, **kw: Any):  # type: ignore[no-redef]
        def wrapper(fn: Any) -> Any:
            return fn
        return wrapper

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from .tracker import TrackerEvent

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Event types                                                                #
# --------------------------------------------------------------------------- #

VALID_EVENT_TYPES = {
    "ENTRY",
    "EXIT",
    "ZONE_ENTER",
    "ZONE_EXIT",
    "ZONE_DWELL",
    "BILLING_QUEUE_JOIN",
    "BILLING_QUEUE_ABANDON",
    "REENTRY",
}


# --------------------------------------------------------------------------- #
#  Pydantic models for validation                                             #
# --------------------------------------------------------------------------- #

class EventMetadata(BaseModel):
    """Nested metadata within a behavioral event."""
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int = 0


class BehavioralEvent(BaseModel):
    """A fully structured behavioral event matching the API schema.

    This schema is validated before writing to JSONL or POSTing to API.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: str
    timestamp: str  # ISO-8601 UTC
    zone_id: Optional[str] = None
    dwell_ms: Optional[int] = None
    is_staff: bool = False
    confidence: float = 0.0
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        if v not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event_type: {v}. Must be one of {VALID_EVENT_TYPES}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {v}")
        return round(v, 4)


# --------------------------------------------------------------------------- #
#  Event Emitter                                                              #
# --------------------------------------------------------------------------- #

class EventEmitter:
    """Converts TrackerEvents into validated JSON events and outputs them.

    Outputs:
      - JSONL file (one JSON object per line)
      - Optional HTTP POST to API endpoint in batches

    Usage:
        emitter = EventEmitter(
            store_id="STORE_BLR_002",
            camera_id="CAM_ENTRY_01",
            output_path="events.jsonl",
            api_url="http://localhost:8000",
        )
        emitter.emit(tracker_events)
        emitter.flush()  # Flush remaining batch to API
    """

    API_BATCH_SIZE = 50  # POST events in batches of this size

    def __init__(
        self,
        store_id: str,
        camera_id: str,
        output_path: str = "events.jsonl",
        api_url: Optional[str] = None,
    ) -> None:
        """Initialize the emitter.

        Args:
            store_id: Store identifier (e.g. "STORE_BLR_002").
            camera_id: Camera identifier (e.g. "CAM_ENTRY_01").
            output_path: Path for JSONL output file.
            api_url: Optional API base URL for HTTP ingestion.
        """
        self._store_id = store_id
        self._camera_id = camera_id
        self._output_path = Path(output_path)
        self._api_url = api_url.rstrip("/") if api_url else None
        self._batch: List[dict] = []
        self._total_emitted: int = 0

        # Ensure output directory exists
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file handle (append mode)
        self._file_handle = open(self._output_path, "a", encoding="utf-8")

        logger.info(
            "EventEmitter initialized: store=%s camera=%s output=%s api=%s",
            store_id, camera_id, output_path, api_url or "disabled",
        )

    def __del__(self) -> None:
        """Clean up file handle on garbage collection."""
        self.close()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def emit(self, tracker_events: List[TrackerEvent]) -> List[dict]:
        """Convert and emit a list of TrackerEvents.

        Args:
            tracker_events: Events from the VisitorTracker.

        Returns:
            List of validated event dictionaries that were emitted.
        """
        emitted: List[dict] = []

        for te in tracker_events:
            try:
                event = self._build_event(te)
                event_dict = self._validate_and_serialize(event)
                if event_dict is not None:
                    self._write_jsonl(event_dict)
                    self._batch.append(event_dict)
                    emitted.append(event_dict)
                    self._total_emitted += 1
            except Exception as e:
                logger.error("Failed to emit event: %s — %s", te, e)

        # Flush batch to API if large enough
        if len(self._batch) >= self.API_BATCH_SIZE:
            self._post_batch()

        return emitted

    def flush(self) -> None:
        """Flush any remaining events in the batch to the API."""
        if self._batch:
            self._post_batch()
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.flush()

    def close(self) -> None:
        """Close the output file handle and flush remaining events."""
        self.flush()
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.close()

    @property
    def total_emitted(self) -> int:
        """Total number of events emitted so far."""
        return self._total_emitted

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_event(self, te: TrackerEvent) -> BehavioralEvent:
        """Convert a TrackerEvent to a BehavioralEvent."""
        # Convert epoch timestamp to ISO-8601 UTC
        if te.timestamp is not None:
            ts_str = datetime.fromtimestamp(te.timestamp, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        else:
            ts_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        metadata = EventMetadata(
            queue_depth=te.queue_depth,
            sku_zone=te.sku_zone,
            session_seq=te.session_seq,
        )

        return BehavioralEvent(
            store_id=self._store_id,
            camera_id=self._camera_id,
            visitor_id=te.visitor_id,
            event_type=te.event_type,
            timestamp=ts_str,
            zone_id=te.zone_id,
            dwell_ms=te.dwell_ms,
            is_staff=te.is_staff,
            confidence=te.confidence,
            metadata=metadata,
        )

    def _validate_and_serialize(self, event: BehavioralEvent) -> Optional[dict]:
        """Validate an event and convert to dict.

        Returns:
            Dict representation of the event, or None if validation fails.
        """
        try:
            if hasattr(event, "model_dump"):
                return event.model_dump()
            else:
                # Fallback for non-pydantic BaseModel
                d = event.__dict__.copy()
                if hasattr(event.metadata, "model_dump"):
                    d["metadata"] = event.metadata.model_dump()
                elif hasattr(event.metadata, "__dict__"):
                    d["metadata"] = {
                        k: v for k, v in event.metadata.__dict__.items()
                        if not k.startswith("_")
                    }
                return d
        except Exception as e:
            logger.error("Event validation failed: %s", e)
            return None

    def _write_jsonl(self, event_dict: dict) -> None:
        """Write a single event as a JSON line to the output file."""
        try:
            line = json.dumps(event_dict, default=str, ensure_ascii=False)
            self._file_handle.write(line + "\n")
        except Exception as e:
            logger.error("Failed to write JSONL: %s", e)

    def _post_batch(self) -> None:
        """POST the current batch of events to the API endpoint."""
        if not self._api_url or not self._batch:
            self._batch.clear()
            return

        if not _HAS_REQUESTS:
            logger.warning("requests library not installed — skipping API POST")
            self._batch.clear()
            return

        url = f"{self._api_url}/events/ingest"
        payload = {"events": list(self._batch)}
        self._batch.clear()

        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code in (200, 201, 202):
                logger.info("POSTed %d events to %s — %d", len(payload["events"]), url, resp.status_code)
            else:
                logger.warning(
                    "API POST failed: %d %s", resp.status_code, resp.text[:200]
                )
        except requests.RequestException as e:
            logger.warning("API POST error: %s", e)
