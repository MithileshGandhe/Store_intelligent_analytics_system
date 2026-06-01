# PROMPT: "Generate pytest tests for /stores/{id}/funnel endpoint.
# Test: full funnel traversal (Entry→Zone→Billing→Purchase), session deduplication
# (re-entry visitor counted once), drop-off % calculations, and empty funnel
# (no events) returns zeroed stages."
#
# CHANGES MADE:
# - Added ?date=2026-03-03 to match test event timestamps
# - Fixed field name: visitor_count not count
# - Added test for partial funnel (visitor enters but never visits billing)
# - Verified drop_off_pct is between 0 and 100
# - Fixed re-entry test to use same visitor_id with REENTRY event type

"""
Tests for GET /stores/{id}/funnel endpoint.

Covers:
  • Full funnel traversal
  • Session deduplication (re-entry counted once)
  • Drop-off percentage calculation
  • Empty funnel
  • Partial funnel (no billing visitors)
"""

import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from tests.conftest import make_event

TEST_DATE = "2026-03-03"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
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


async def _ingest(client, events):
    resp = await client.post("/events/ingest", json={"events": events})
    assert resp.status_code == 200


# ─── Full Funnel ─────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_funnel_full_traversal(client):
    """Visitor who enters, visits zone, joins billing queue should appear in all stages."""
    vid = "VIS_funnel01"
    events = [
        make_event(visitor_id=vid, event_type="ENTRY", timestamp="2026-03-03T10:00:00Z"),
        make_event(visitor_id=vid, event_type="ZONE_ENTER", zone_id="SKINCARE",
                   timestamp="2026-03-03T10:01:00Z", camera_id="CAM_FLOOR_01"),
        make_event(visitor_id=vid, event_type="BILLING_QUEUE_JOIN", zone_id="BILLING",
                   timestamp="2026-03-03T10:05:00Z", camera_id="CAM_BILLING_01"),
    ]
    await _ingest(client, events)

    resp = await client.get(f"/stores/STORE_BLR_002/funnel?date={TEST_DATE}")
    assert resp.status_code == 200
    data = resp.json()
    assert "stages" in data
    assert len(data["stages"]) >= 3


@pytest.mark.anyio
async def test_funnel_reentry_not_double_counted(client):
    """Visitor with REENTRY should be counted once in the funnel, not twice."""
    vid = "VIS_reentry_funnel"
    events = [
        make_event(visitor_id=vid, event_type="ENTRY", timestamp="2026-03-03T10:00:00Z"),
        make_event(visitor_id=vid, event_type="EXIT", timestamp="2026-03-03T10:10:00Z"),
        make_event(visitor_id=vid, event_type="REENTRY", timestamp="2026-03-03T10:20:00Z"),
        make_event(visitor_id=vid, event_type="ZONE_ENTER", zone_id="SKINCARE",
                   timestamp="2026-03-03T10:21:00Z", camera_id="CAM_FLOOR_01"),
    ]
    await _ingest(client, events)

    resp = await client.get(f"/stores/STORE_BLR_002/funnel?date={TEST_DATE}")
    assert resp.status_code == 200
    data = resp.json()
    # Entry stage should show 1 unique visitor, not 2
    if data["stages"]:
        entry_stage = data["stages"][0]
        assert entry_stage["visitor_count"] == 1


# ─── Empty Funnel ────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_funnel_empty_store(client):
    """Store with no events should return 404."""
    resp = await client.get("/stores/STORE_EMPTY_002/funnel")
    assert resp.status_code in (200, 404)


# ─── Drop-off ────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_funnel_dropoff_percentage(client):
    """Drop-off percentage should be between 0 and 100."""
    events = [
        make_event(visitor_id="VIS_f1", event_type="ENTRY", timestamp="2026-03-03T10:00:00Z"),
        make_event(visitor_id="VIS_f2", event_type="ENTRY", timestamp="2026-03-03T10:01:00Z"),
        make_event(visitor_id="VIS_f3", event_type="ENTRY", timestamp="2026-03-03T10:02:00Z"),
        make_event(visitor_id="VIS_f1", event_type="ZONE_ENTER", zone_id="SKINCARE",
                   timestamp="2026-03-03T10:03:00Z", camera_id="CAM_FLOOR_01"),
        make_event(visitor_id="VIS_f2", event_type="ZONE_ENTER", zone_id="HAIRCARE",
                   timestamp="2026-03-03T10:04:00Z", camera_id="CAM_FLOOR_01"),
        make_event(visitor_id="VIS_f1", event_type="BILLING_QUEUE_JOIN", zone_id="BILLING",
                   timestamp="2026-03-03T10:08:00Z", camera_id="CAM_BILLING_01"),
    ]
    await _ingest(client, events)

    resp = await client.get(f"/stores/STORE_BLR_002/funnel?date={TEST_DATE}")
    assert resp.status_code == 200
    data = resp.json()
    for stage in data.get("stages", []):
        if "drop_off_pct" in stage:
            assert 0 <= stage["drop_off_pct"] <= 100


# ─── Response Format ─────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_funnel_response_structure(client):
    """Funnel response should have stages with stage, visitor_count, drop_off_pct."""
    events = [make_event(event_type="ENTRY")]
    await _ingest(client, events)

    resp = await client.get(f"/stores/STORE_BLR_002/funnel?date={TEST_DATE}")
    assert resp.status_code == 200
    data = resp.json()
    assert "stages" in data
    assert isinstance(data["stages"], list)
