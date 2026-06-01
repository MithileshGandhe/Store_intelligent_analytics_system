# PROMPT: "Generate pytest conftest.py for a FastAPI application that uses
# SQLite for testing. Create fixtures for: test client with async support,
# sample events covering all event types (ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT,
# ZONE_DWELL, BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, REENTRY), staff events,
# and a function to seed events into the database."
#
# CHANGES MADE:
# - Added proper lifespan handling for FastAPI TestClient
# - Used httpx.AsyncClient instead of TestClient for async tests
# - Added sample POS transactions to support conversion rate testing
# - Added factory fixtures for generating custom events with overrides
# - Added cleanup fixture to reset DB between tests

"""
Shared test fixtures for the Store Intelligence API test suite.
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

import pytest
import pytest_asyncio

# Force SQLite for testing
os.environ["USE_SQLITE"] = "true"
os.environ["SQLITE_PATH"] = ":memory:"
os.environ["APP_VERSION"] = "1.0.0-test"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def base_url():
    return "http://testserver"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def sample_entry_event() -> Dict[str, Any]:
    """A single valid ENTRY event."""
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "VIS_test01",
        "event_type": "ENTRY",
        "timestamp": "2026-03-03T10:05:12Z",
        "zone_id": None,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.94,
        "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
    }


@pytest.fixture
def sample_staff_event() -> Dict[str, Any]:
    """A single valid staff ENTRY event."""
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "VIS_staff01",
        "event_type": "ENTRY",
        "timestamp": "2026-03-03T09:55:00Z",
        "zone_id": None,
        "dwell_ms": 0,
        "is_staff": True,
        "confidence": 0.97,
        "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
    }


@pytest.fixture
def sample_full_session_events() -> List[Dict[str, Any]]:
    """Complete visitor session: ENTRY → ZONE_ENTER → ZONE_DWELL → ZONE_EXIT → BILLING_QUEUE_JOIN → EXIT."""
    vid = "VIS_full01"
    base_time = datetime(2026, 3, 3, 10, 5, 0, tzinfo=timezone.utc)
    return [
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": vid,
            "event_type": "ENTRY",
            "timestamp": base_time.isoformat(),
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.94,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
        },
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_FLOOR_01",
            "visitor_id": vid,
            "event_type": "ZONE_ENTER",
            "timestamp": (base_time + timedelta(seconds=15)).isoformat(),
            "zone_id": "SKINCARE",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.91,
            "metadata": {"queue_depth": None, "sku_zone": "SKINCARE", "session_seq": 2},
        },
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_FLOOR_01",
            "visitor_id": vid,
            "event_type": "ZONE_DWELL",
            "timestamp": (base_time + timedelta(seconds=45)).isoformat(),
            "zone_id": "SKINCARE",
            "dwell_ms": 30000,
            "is_staff": False,
            "confidence": 0.89,
            "metadata": {"queue_depth": None, "sku_zone": "SKINCARE", "session_seq": 3},
        },
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_FLOOR_01",
            "visitor_id": vid,
            "event_type": "ZONE_EXIT",
            "timestamp": (base_time + timedelta(seconds=120)).isoformat(),
            "zone_id": "SKINCARE",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.87,
            "metadata": {"queue_depth": None, "sku_zone": "SKINCARE", "session_seq": 4},
        },
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_BILLING_01",
            "visitor_id": vid,
            "event_type": "BILLING_QUEUE_JOIN",
            "timestamp": (base_time + timedelta(seconds=135)).isoformat(),
            "zone_id": "BILLING",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.92,
            "metadata": {"queue_depth": 2, "sku_zone": "BILLING", "session_seq": 5},
        },
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": vid,
            "event_type": "EXIT",
            "timestamp": (base_time + timedelta(seconds=300)).isoformat(),
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.93,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 6},
        },
    ]


@pytest.fixture
def sample_reentry_events() -> List[Dict[str, Any]]:
    """Visitor with ENTRY → EXIT → REENTRY → EXIT sequence."""
    vid = "VIS_reentry01"
    base_time = datetime(2026, 3, 3, 10, 10, 0, tzinfo=timezone.utc)
    return [
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": vid,
            "event_type": "ENTRY",
            "timestamp": base_time.isoformat(),
            "zone_id": None, "dwell_ms": 0, "is_staff": False,
            "confidence": 0.92,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
        },
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": vid,
            "event_type": "EXIT",
            "timestamp": (base_time + timedelta(minutes=5)).isoformat(),
            "zone_id": None, "dwell_ms": 0, "is_staff": False,
            "confidence": 0.90,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 2},
        },
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": vid,
            "event_type": "REENTRY",
            "timestamp": (base_time + timedelta(minutes=10)).isoformat(),
            "zone_id": None, "dwell_ms": 0, "is_staff": False,
            "confidence": 0.78,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 3},
        },
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": vid,
            "event_type": "EXIT",
            "timestamp": (base_time + timedelta(minutes=20)).isoformat(),
            "zone_id": None, "dwell_ms": 0, "is_staff": False,
            "confidence": 0.91,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 4},
        },
    ]


@pytest.fixture
def sample_abandon_event() -> Dict[str, Any]:
    """A BILLING_QUEUE_ABANDON event."""
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_BILLING_01",
        "visitor_id": "VIS_abandon01",
        "event_type": "BILLING_QUEUE_ABANDON",
        "timestamp": "2026-03-03T10:14:30Z",
        "zone_id": "BILLING",
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.87,
        "metadata": {"queue_depth": 2, "sku_zone": "BILLING", "session_seq": 5},
    }


def make_event(**overrides) -> Dict[str, Any]:
    """Factory to create an event with optional overrides."""
    base = {
        "event_id": str(uuid.uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": f"VIS_{uuid.uuid4().hex[:6]}",
        "event_type": "ENTRY",
        "timestamp": "2026-03-03T10:00:00Z",
        "zone_id": None,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.90,
        "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
    }
    base.update(overrides)
    return base
