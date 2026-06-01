"""
tracker.py — VisitorTracker: stateful cross-frame person tracking.

Maintains visitor sessions across video frames by:
  - Assigning unique visitor IDs (VIS_ + 6 hex chars)
  - Detecting ENTRY / EXIT via centroid threshold crossing
  - Mapping centroids to store zones (point-in-polygon)
  - Re-identifying visitors after exit via feature similarity
  - Tracking dwell time per zone (emits ZONE_DWELL every 30s)
  - Monitoring billing queue depth
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .detector_base import Detection

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Data structures                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class VisitorState:
    """Tracked state for a single visitor across frames."""
    visitor_id: str
    track_id: int                         # Detector-assigned track ID
    is_staff: bool
    first_seen_ts: float                  # Epoch seconds
    last_seen_ts: float
    last_centroid: Tuple[float, float]     # (cx, cy) normalized
    prev_centroid: Optional[Tuple[float, float]] = None
    current_zone: Optional[str] = None
    previous_zone: Optional[str] = None
    entered: bool = False                 # Has crossed entry line into store
    exited: bool = False                  # Has crossed exit line out of store
    session_seq: int = 0                  # Monotonic event counter
    features: Optional[np.ndarray] = None # Latest Re-ID feature vector
    zone_enter_ts: Optional[float] = None # When they entered current zone
    last_dwell_emit_ts: Optional[float] = None  # Last ZONE_DWELL emission time
    in_billing_queue: bool = False        # Currently in billing queue


@dataclass
class TrackerEvent:
    """An event produced by the tracker for the emitter to format."""
    event_type: str                       # ENTRY, EXIT, ZONE_ENTER, etc.
    visitor_id: str
    is_staff: bool
    confidence: float
    zone_id: Optional[str] = None
    dwell_ms: Optional[int] = None
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int = 0
    timestamp: Optional[float] = None     # Epoch seconds


# --------------------------------------------------------------------------- #
#  Zone geometry helpers                                                      #
# --------------------------------------------------------------------------- #

def _point_in_polygon(px: float, py: float, polygon: List[List[float]]) -> bool:
    """Ray-casting algorithm to test if point (px, py) is inside a polygon.

    Args:
        px, py: Point coordinates (normalized 0-1).
        polygon: List of [x, y] vertices (normalized 0-1).

    Returns:
        True if the point is inside the polygon.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _load_zones(layout_path: str) -> Tuple[List[dict], dict]:
    """Load zone definitions from store_layout.json.

    Returns:
        Tuple of (zones_list, entry_line_config).
    """
    path = Path(layout_path)
    if not path.exists():
        logger.warning("Store layout not found at %s — using empty zones", layout_path)
        return [], {"y_threshold": 0.15, "direction": "top_to_bottom_is_entry"}

    with open(path, "r", encoding="utf-8") as f:
        layout = json.load(f)

    zones = layout.get("zones", [])
    entry_line = layout.get("entry_line", {
        "y_threshold": 0.15,
        "direction": "top_to_bottom_is_entry"
    })
    return zones, entry_line


# --------------------------------------------------------------------------- #
#  Visitor Tracker                                                            #
# --------------------------------------------------------------------------- #

class VisitorTracker:
    """Stateful tracker that converts raw detections into behavioral events.

    Usage:
        tracker = VisitorTracker(store_layout_path="data/store_layout.json")
        for frame_detections, timestamp in detection_stream:
            events = tracker.update(frame_detections, timestamp)
            # events: List[TrackerEvent]
    """

    # Thresholds
    REID_SIMILARITY_THRESHOLD = 0.75   # Cosine similarity for re-identification
    DWELL_EMIT_INTERVAL_S = 30.0       # Emit ZONE_DWELL every N seconds
    LOST_TIMEOUT_S = 5.0               # Seconds before a lost track triggers EXIT

    def __init__(
        self,
        store_layout_path: str = "data/store_layout.json",
        entry_direction: str = "top_to_bottom_is_entry",
    ) -> None:
        """Initialize the tracker.

        Args:
            store_layout_path: Path to the store_layout.json file.
            entry_direction: How to interpret centroid movement for entry/exit.
                'top_to_bottom_is_entry' = moving downward = ENTRY.
        """
        self._zones, self._entry_line = _load_zones(store_layout_path)
        self._entry_threshold: float = self._entry_line.get("y_threshold", 0.15)
        self._entry_direction: str = entry_direction

        # Active visitors: track_id -> VisitorState
        self._active: Dict[int, VisitorState] = {}
        # Exited visitors for Re-ID: visitor_id -> VisitorState
        self._exited_pool: Dict[str, VisitorState] = {}
        # Visitor ID counter for uniqueness
        self._visitor_counter: int = 0
        # Billing queue set
        self._billing_queue: Set[str] = set()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def update(
        self,
        detections: List[Detection],
        timestamp: float,
    ) -> List[TrackerEvent]:
        """Process a frame's detections and return behavioral events.

        Args:
            detections: List of Detection objects from the detector.
            timestamp: Frame timestamp in epoch seconds.

        Returns:
            List of TrackerEvent objects generated by this frame.
        """
        events: List[TrackerEvent] = []

        # Set of track_ids seen this frame
        seen_track_ids: Set[int] = set()

        for det in detections:
            if det.track_id is None:
                continue

            seen_track_ids.add(det.track_id)
            cx, cy = det.centroid

            if det.track_id in self._active:
                # ---- Update existing visitor ----
                visitor = self._active[det.track_id]
                visitor.prev_centroid = visitor.last_centroid
                visitor.last_centroid = (cx, cy)
                visitor.last_seen_ts = timestamp
                visitor.confidence = det.confidence
                if det.features is not None:
                    visitor.features = det.features

                # Check zone transitions
                new_zone = self._get_zone(cx, cy)
                zone_events = self._handle_zone_transition(visitor, new_zone, det.confidence, timestamp)
                events.extend(zone_events)

                # Check dwell
                dwell_events = self._check_dwell(visitor, det.confidence, timestamp)
                events.extend(dwell_events)

                # Check exit
                exit_events = self._check_exit_crossing(visitor, det.confidence, timestamp)
                events.extend(exit_events)

            else:
                # ---- New detection: check Re-ID first ----
                reentry_visitor = self._try_reidentify(det)

                if reentry_visitor is not None:
                    # Re-entry! Reactivate the visitor
                    reentry_visitor.track_id = det.track_id
                    reentry_visitor.last_centroid = (cx, cy)
                    reentry_visitor.last_seen_ts = timestamp
                    reentry_visitor.exited = False
                    reentry_visitor.entered = True
                    if det.features is not None:
                        reentry_visitor.features = det.features
                    self._active[det.track_id] = reentry_visitor

                    reentry_visitor.session_seq += 1
                    events.append(TrackerEvent(
                        event_type="REENTRY",
                        visitor_id=reentry_visitor.visitor_id,
                        is_staff=reentry_visitor.is_staff,
                        confidence=det.confidence,
                        zone_id=self._get_zone(cx, cy),
                        session_seq=reentry_visitor.session_seq,
                        timestamp=timestamp,
                    ))
                else:
                    # Brand new visitor
                    visitor_id = self._mint_visitor_id()
                    visitor = VisitorState(
                        visitor_id=visitor_id,
                        track_id=det.track_id,
                        is_staff=det.is_staff,
                        first_seen_ts=timestamp,
                        last_seen_ts=timestamp,
                        last_centroid=(cx, cy),
                        features=det.features,
                    )
                    self._active[det.track_id] = visitor

                    # Check if entering
                    entry_events = self._check_entry_crossing(visitor, det.confidence, timestamp)
                    events.extend(entry_events)

        # ---- Handle lost tracks (not seen this frame) ----
        lost_events = self._handle_lost_tracks(seen_track_ids, timestamp)
        events.extend(lost_events)

        return events

    @property
    def billing_queue_depth(self) -> int:
        """Current number of people in the billing queue."""
        return len(self._billing_queue)

    @property
    def active_visitor_count(self) -> int:
        """Number of currently tracked visitors."""
        return len(self._active)

    # ------------------------------------------------------------------ #
    #  Visitor ID generation                                              #
    # ------------------------------------------------------------------ #

    def _mint_visitor_id(self) -> str:
        """Generate a unique visitor ID in format VIS_ + 6 hex chars."""
        self._visitor_counter += 1
        raw = hashlib.md5(f"visitor_{self._visitor_counter}_{time.time_ns()}".encode()).hexdigest()
        return f"VIS_{raw[:6]}"

    # ------------------------------------------------------------------ #
    #  Zone mapping                                                       #
    # ------------------------------------------------------------------ #

    def _get_zone(self, cx: float, cy: float) -> Optional[str]:
        """Map a centroid coordinate to a zone using point-in-polygon.

        Args:
            cx, cy: Centroid in normalized [0, 1] coordinates.

        Returns:
            Zone ID string, or None if outside all zones.
        """
        for zone in self._zones:
            polygon = zone.get("polygon", [])
            if _point_in_polygon(cx, cy, polygon):
                return zone["zone_id"]
        return None

    def _get_zone_info(self, zone_id: Optional[str]) -> Optional[dict]:
        """Look up full zone metadata by zone_id."""
        if zone_id is None:
            return None
        for zone in self._zones:
            if zone["zone_id"] == zone_id:
                return zone
        return None

    # ------------------------------------------------------------------ #
    #  Entry / Exit detection                                             #
    # ------------------------------------------------------------------ #

    def _check_entry_crossing(
        self, visitor: VisitorState, confidence: float, timestamp: float
    ) -> List[TrackerEvent]:
        """Check if a new visitor's position indicates an ENTRY."""
        events = []
        cx, cy = visitor.last_centroid

        # For "top_to_bottom_is_entry": if centroid is near/below the entry
        # threshold line, they've entered the store.
        if self._entry_direction == "top_to_bottom_is_entry":
            if cy >= self._entry_threshold:
                visitor.entered = True
                visitor.session_seq += 1
                zone = self._get_zone(cx, cy)
                visitor.current_zone = zone
                visitor.zone_enter_ts = timestamp
                visitor.last_dwell_emit_ts = timestamp
                events.append(TrackerEvent(
                    event_type="ENTRY",
                    visitor_id=visitor.visitor_id,
                    is_staff=visitor.is_staff,
                    confidence=confidence,
                    zone_id=zone,
                    session_seq=visitor.session_seq,
                    timestamp=timestamp,
                ))

        return events

    def _check_exit_crossing(
        self, visitor: VisitorState, confidence: float, timestamp: float
    ) -> List[TrackerEvent]:
        """Check if a tracked visitor's movement indicates an EXIT."""
        events = []
        if not visitor.entered or visitor.exited:
            return events

        cx, cy = visitor.last_centroid
        prev = visitor.prev_centroid

        if prev is None:
            return events

        _, prev_cy = prev

        # Exit: moving from below threshold to above (bottom → top)
        if self._entry_direction == "top_to_bottom_is_entry":
            if prev_cy >= self._entry_threshold and cy < self._entry_threshold:
                events.extend(self._do_exit(visitor, confidence, timestamp))

        return events

    def _do_exit(
        self, visitor: VisitorState, confidence: float, timestamp: float
    ) -> List[TrackerEvent]:
        """Handle a visitor exiting the store."""
        events = []

        # If they were in a zone, emit ZONE_EXIT first
        if visitor.current_zone is not None:
            dwell_ms = int((timestamp - (visitor.zone_enter_ts or timestamp)) * 1000)
            zone_info = self._get_zone_info(visitor.current_zone)
            visitor.session_seq += 1
            events.append(TrackerEvent(
                event_type="ZONE_EXIT",
                visitor_id=visitor.visitor_id,
                is_staff=visitor.is_staff,
                confidence=confidence,
                zone_id=visitor.current_zone,
                dwell_ms=dwell_ms,
                sku_zone=zone_info.get("sku_zone") if zone_info else None,
                session_seq=visitor.session_seq,
                timestamp=timestamp,
            ))

        # Handle billing queue abandonment
        if visitor.in_billing_queue:
            visitor.in_billing_queue = False
            self._billing_queue.discard(visitor.visitor_id)
            visitor.session_seq += 1
            events.append(TrackerEvent(
                event_type="BILLING_QUEUE_ABANDON",
                visitor_id=visitor.visitor_id,
                is_staff=visitor.is_staff,
                confidence=confidence,
                zone_id="BILLING",
                queue_depth=self.billing_queue_depth,
                session_seq=visitor.session_seq,
                timestamp=timestamp,
            ))

        # EXIT event
        visitor.session_seq += 1
        visitor.exited = True
        visitor.current_zone = None
        events.append(TrackerEvent(
            event_type="EXIT",
            visitor_id=visitor.visitor_id,
            is_staff=visitor.is_staff,
            confidence=confidence,
            session_seq=visitor.session_seq,
            timestamp=timestamp,
        ))

        # Move to exited pool for Re-ID
        self._exited_pool[visitor.visitor_id] = visitor
        if visitor.track_id in self._active:
            del self._active[visitor.track_id]

        return events

    # ------------------------------------------------------------------ #
    #  Zone transitions                                                   #
    # ------------------------------------------------------------------ #

    def _handle_zone_transition(
        self,
        visitor: VisitorState,
        new_zone: Optional[str],
        confidence: float,
        timestamp: float,
    ) -> List[TrackerEvent]:
        """Detect and handle transitions between zones."""
        events = []

        if not visitor.entered:
            return events

        old_zone = visitor.current_zone

        if new_zone == old_zone:
            return events  # No transition

        # ZONE_EXIT from old zone
        if old_zone is not None:
            dwell_ms = int((timestamp - (visitor.zone_enter_ts or timestamp)) * 1000)
            zone_info = self._get_zone_info(old_zone)
            visitor.session_seq += 1
            events.append(TrackerEvent(
                event_type="ZONE_EXIT",
                visitor_id=visitor.visitor_id,
                is_staff=visitor.is_staff,
                confidence=confidence,
                zone_id=old_zone,
                dwell_ms=dwell_ms,
                sku_zone=zone_info.get("sku_zone") if zone_info else None,
                session_seq=visitor.session_seq,
                timestamp=timestamp,
            ))

            # Billing queue logic
            if old_zone == "BILLING" and visitor.in_billing_queue:
                visitor.in_billing_queue = False
                self._billing_queue.discard(visitor.visitor_id)
                visitor.session_seq += 1
                events.append(TrackerEvent(
                    event_type="BILLING_QUEUE_ABANDON",
                    visitor_id=visitor.visitor_id,
                    is_staff=visitor.is_staff,
                    confidence=confidence,
                    zone_id="BILLING",
                    queue_depth=self.billing_queue_depth,
                    session_seq=visitor.session_seq,
                    timestamp=timestamp,
                ))

        # ZONE_ENTER into new zone
        if new_zone is not None:
            visitor.current_zone = new_zone
            visitor.zone_enter_ts = timestamp
            visitor.last_dwell_emit_ts = timestamp
            zone_info = self._get_zone_info(new_zone)
            visitor.session_seq += 1
            events.append(TrackerEvent(
                event_type="ZONE_ENTER",
                visitor_id=visitor.visitor_id,
                is_staff=visitor.is_staff,
                confidence=confidence,
                zone_id=new_zone,
                sku_zone=zone_info.get("sku_zone") if zone_info else None,
                session_seq=visitor.session_seq,
                timestamp=timestamp,
            ))

            # Billing queue join
            if new_zone == "BILLING" and not visitor.is_staff:
                visitor.in_billing_queue = True
                self._billing_queue.add(visitor.visitor_id)
                visitor.session_seq += 1
                events.append(TrackerEvent(
                    event_type="BILLING_QUEUE_JOIN",
                    visitor_id=visitor.visitor_id,
                    is_staff=visitor.is_staff,
                    confidence=confidence,
                    zone_id="BILLING",
                    queue_depth=self.billing_queue_depth,
                    session_seq=visitor.session_seq,
                    timestamp=timestamp,
                ))
        else:
            visitor.current_zone = None
            visitor.zone_enter_ts = None
            visitor.last_dwell_emit_ts = None

        return events

    # ------------------------------------------------------------------ #
    #  Dwell tracking                                                     #
    # ------------------------------------------------------------------ #

    def _check_dwell(
        self, visitor: VisitorState, confidence: float, timestamp: float
    ) -> List[TrackerEvent]:
        """Emit ZONE_DWELL events every DWELL_EMIT_INTERVAL_S seconds."""
        events = []

        if (
            visitor.current_zone is None
            or visitor.zone_enter_ts is None
            or visitor.last_dwell_emit_ts is None
        ):
            return events

        elapsed_since_last = timestamp - visitor.last_dwell_emit_ts
        if elapsed_since_last >= self.DWELL_EMIT_INTERVAL_S:
            total_dwell_ms = int((timestamp - visitor.zone_enter_ts) * 1000)
            zone_info = self._get_zone_info(visitor.current_zone)
            visitor.session_seq += 1
            visitor.last_dwell_emit_ts = timestamp
            events.append(TrackerEvent(
                event_type="ZONE_DWELL",
                visitor_id=visitor.visitor_id,
                is_staff=visitor.is_staff,
                confidence=confidence,
                zone_id=visitor.current_zone,
                dwell_ms=total_dwell_ms,
                sku_zone=zone_info.get("sku_zone") if zone_info else None,
                queue_depth=self.billing_queue_depth if visitor.current_zone == "BILLING" else None,
                session_seq=visitor.session_seq,
                timestamp=timestamp,
            ))

        return events

    # ------------------------------------------------------------------ #
    #  Re-identification                                                  #
    # ------------------------------------------------------------------ #

    def _try_reidentify(self, detection: Detection) -> Optional[VisitorState]:
        """Attempt to match a new detection to a previously exited visitor.

        Uses cosine similarity between Re-ID feature vectors.

        Returns:
            VisitorState if a match is found above threshold, else None.
        """
        if detection.features is None or len(self._exited_pool) == 0:
            return None

        best_sim = -1.0
        best_vid: Optional[str] = None

        feat = detection.features
        feat_norm = np.linalg.norm(feat)
        if feat_norm < 1e-8:
            return None

        for vid, visitor in self._exited_pool.items():
            if visitor.features is None:
                continue
            pool_norm = np.linalg.norm(visitor.features)
            if pool_norm < 1e-8:
                continue
            sim = float(np.dot(feat, visitor.features) / (feat_norm * pool_norm))
            if sim > best_sim:
                best_sim = sim
                best_vid = vid

        if best_vid is not None and best_sim >= self.REID_SIMILARITY_THRESHOLD:
            visitor = self._exited_pool.pop(best_vid)
            logger.info(
                "Re-ID match: %s (sim=%.3f)", visitor.visitor_id, best_sim
            )
            return visitor

        return None

    # ------------------------------------------------------------------ #
    #  Lost track handling                                                #
    # ------------------------------------------------------------------ #

    def _handle_lost_tracks(
        self, seen_ids: Set[int], timestamp: float
    ) -> List[TrackerEvent]:
        """Handle tracks that were not seen in this frame.

        After LOST_TIMEOUT_S seconds without a detection, the visitor
        is considered to have exited.
        """
        events = []
        lost_track_ids = []

        for track_id, visitor in list(self._active.items()):
            if track_id not in seen_ids:
                time_since_seen = timestamp - visitor.last_seen_ts
                if time_since_seen >= self.LOST_TIMEOUT_S:
                    lost_track_ids.append(track_id)
                    if visitor.entered and not visitor.exited:
                        exit_events = self._do_exit(visitor, 0.5, timestamp)
                        events.extend(exit_events)

        # Clean up (can't modify dict during iteration above when _do_exit
        # already removes from _active, but handle leftovers)
        for tid in lost_track_ids:
            self._active.pop(tid, None)

        return events
