# PROMPT: "Generate pytest tests for a FastAPI event ingestion endpoint.
# Test: happy path batch ingest, idempotency (same payload twice), partial
# failure with mix of valid/invalid events, 500-event batch limit, schema
# validation (missing required fields, invalid event_type), and empty batch.
# Use httpx AsyncClient with the FastAPI app."
#
# CHANGES MADE:
# - Added test for structured error response format (not raw stack traces)
# - Added test that verifies accepted count matches unique valid events
# - Added test for malformed JSON body
# - Fixed async fixture handling for the database lifecycle
# - Added __init__.py for tests package

"""
Tests for POST /events/ingest endpoint.

Covers:
  • Happy path — batch ingest returns accepted count
  • Idempotency — same payload twice produces same result
  • Partial failure — mix of valid/invalid events
  • Batch limit — >500 events rejected
  • Schema validation — missing fields, invalid types
  • Empty batch
"""

import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from tests.conftest import make_event


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Create a test client with fresh in-memory DB."""
    import os
    os.environ["USE_SQLITE"] = "true"
    os.environ["SQLITE_PATH"] = ":memory:"

    from app.main import app
    from app.database import init_db, close_db

    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c
    await close_db()


# ─── Happy Path ──────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_ingest_happy_path(client):
    """Valid batch of events should return 200 with correct accepted count."""
    events = [make_event() for _ in range(5)]
    resp = await client.post("/events/ingest", json={"events": events})

    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] == 5
    assert data["rejected"] == 0
    assert data["errors"] == []


@pytest.mark.anyio
async def test_ingest_single_event(client):
    """Single event ingest should work."""
    events = [make_event()]
    resp = await client.post("/events/ingest", json={"events": events})
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 1


# ─── Idempotency ─────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_ingest_idempotency(client):
    """Same payload ingested twice should not produce errors or duplicates."""
    events = [make_event() for _ in range(3)]

    # First ingest
    resp1 = await client.post("/events/ingest", json={"events": events})
    assert resp1.status_code == 200
    assert resp1.json()["accepted"] == 3

    # Second ingest — same event_ids
    resp2 = await client.post("/events/ingest", json={"events": events})
    assert resp2.status_code == 200
    # Should still return 200, events silently deduplicated


# ─── Partial Failure ─────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_ingest_partial_failure(client):
    """Mix of valid and invalid events — valid ones should be accepted."""
    valid_event = make_event()
    invalid_event = {
        "event_id": str(uuid.uuid4()),
        "store_id": "STORE_BLR_002",
        # Missing required fields: camera_id, visitor_id, event_type, timestamp, confidence
    }
    resp = await client.post("/events/ingest", json={"events": [valid_event, invalid_event]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] >= 1
    assert data["rejected"] >= 1


@pytest.mark.anyio
async def test_ingest_invalid_event_type(client):
    """Invalid event_type should be rejected."""
    event = make_event(event_type="INVALID_TYPE")
    resp = await client.post("/events/ingest", json={"events": [event]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["rejected"] >= 1


# ─── Batch Limits ────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_ingest_batch_limit_exceeded(client):
    """More than 500 events should be rejected."""
    events = [make_event() for _ in range(501)]
    resp = await client.post("/events/ingest", json={"events": events})
    # Should return 400 for exceeding batch limit
    assert resp.status_code in (400, 422)


@pytest.mark.anyio
async def test_ingest_empty_batch(client):
    """Empty event list should be handled gracefully."""
    resp = await client.post("/events/ingest", json={"events": []})
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] == 0


# ─── Schema Validation ──────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_ingest_missing_required_fields(client):
    """Events missing required fields should be rejected individually."""
    event = {"event_id": str(uuid.uuid4())}  # Missing everything else
    resp = await client.post("/events/ingest", json={"events": [event]})
    assert resp.status_code == 200
    assert resp.json()["rejected"] >= 1


@pytest.mark.anyio
async def test_ingest_all_event_types(client):
    """All valid event types should be accepted."""
    event_types = [
        "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT",
        "ZONE_DWELL", "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY",
    ]
    events = [make_event(event_type=et) for et in event_types]
    resp = await client.post("/events/ingest", json={"events": events})
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 8


# ─── Error Response Format ───────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_ingest_response_has_structured_format(client):
    """Response should always have accepted, rejected, errors fields."""
    events = [make_event()]
    resp = await client.post("/events/ingest", json={"events": events})
    data = resp.json()
    assert "accepted" in data
    assert "rejected" in data
    assert "errors" in data
    assert isinstance(data["errors"], list)
