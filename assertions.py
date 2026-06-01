"""
Assertions test suite — 10 example test assertions the API must pass.
Run with: python assertions.py [--api-url http://localhost:8000]

These are NOT the full scoring test suite. They verify the acceptance gate.
"""
import argparse
import json
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install with: pip install requests")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run API acceptance gate assertions")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    
    base = args.api_url.rstrip("/")
    passed = 0
    failed = 0
    total = 10
    
    def assert_test(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ PASS: {name}")
            passed += 1
        else:
            print(f"  ❌ FAIL: {name} — {detail}")
            failed += 1

    print(f"\n🔍 Running {total} acceptance gate assertions against {base}\n")

    # -------------------------------------------------------------------
    # 1. Health endpoint responds
    # -------------------------------------------------------------------
    try:
        r = requests.get(f"{base}/health", timeout=10)
        assert_test(
            "GET /health returns 200",
            r.status_code == 200,
            f"got {r.status_code}"
        )
    except Exception as e:
        assert_test("GET /health returns 200", False, str(e))

    # -------------------------------------------------------------------
    # 2. Health response has required fields
    # -------------------------------------------------------------------
    try:
        data = r.json()
        has_fields = all(k in data for k in ("status", "database", "version"))
        assert_test(
            "Health response has status, database, version",
            has_fields,
            f"keys: {list(data.keys())}"
        )
    except Exception as e:
        assert_test("Health response has required fields", False, str(e))

    # -------------------------------------------------------------------
    # 3. Event ingest accepts valid events
    # -------------------------------------------------------------------
    sample_events = [
        {
            "event_id": "test-assertion-0001-0001-000000000001",
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_test01",
            "event_type": "ENTRY",
            "timestamp": "2026-03-03T10:05:12Z",
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.94,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1}
        },
        {
            "event_id": "test-assertion-0001-0001-000000000002",
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_FLOOR_01",
            "visitor_id": "VIS_test01",
            "event_type": "ZONE_ENTER",
            "timestamp": "2026-03-03T10:05:25Z",
            "zone_id": "SKINCARE",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.91,
            "metadata": {"queue_depth": None, "sku_zone": "SKINCARE", "session_seq": 2}
        },
        {
            "event_id": "test-assertion-0001-0001-000000000003",
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_BILLING_01",
            "visitor_id": "VIS_test01",
            "event_type": "BILLING_QUEUE_JOIN",
            "timestamp": "2026-03-03T10:08:45Z",
            "zone_id": "BILLING",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.92,
            "metadata": {"queue_depth": 2, "sku_zone": "BILLING", "session_seq": 3}
        },
        {
            "event_id": "test-assertion-0001-0001-000000000004",
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_test01",
            "event_type": "EXIT",
            "timestamp": "2026-03-03T10:12:30Z",
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.93,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 4}
        },
        {
            "event_id": "test-assertion-0001-0001-000000000005",
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_staff_t1",
            "event_type": "ENTRY",
            "timestamp": "2026-03-03T09:55:00Z",
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": True,
            "confidence": 0.97,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1}
        }
    ]
    try:
        r = requests.post(f"{base}/events/ingest", json={"events": sample_events}, timeout=10)
        assert_test(
            "POST /events/ingest returns 200",
            r.status_code == 200,
            f"got {r.status_code}: {r.text[:200]}"
        )
    except Exception as e:
        assert_test("POST /events/ingest returns 200", False, str(e))

    # -------------------------------------------------------------------
    # 4. Ingest response has accepted/rejected counts
    # -------------------------------------------------------------------
    try:
        data = r.json()
        has_counts = "accepted" in data and "rejected" in data
        assert_test(
            "Ingest response has accepted/rejected counts",
            has_counts and data["accepted"] > 0,
            f"data: {data}"
        )
    except Exception as e:
        assert_test("Ingest response has accepted/rejected counts", False, str(e))

    # -------------------------------------------------------------------
    # 5. Idempotency — same payload ingested twice gives same result
    # -------------------------------------------------------------------
    try:
        r2 = requests.post(f"{base}/events/ingest", json={"events": sample_events}, timeout=10)
        data2 = r2.json()
        assert_test(
            "Idempotent ingest — duplicate accepted without error",
            r2.status_code == 200,
            f"got {r2.status_code}"
        )
    except Exception as e:
        assert_test("Idempotent ingest", False, str(e))

    # -------------------------------------------------------------------
    # 6. Metrics endpoint returns valid JSON
    # -------------------------------------------------------------------
    try:
        r = requests.get(f"{base}/stores/STORE_BLR_002/metrics", timeout=10)
        assert_test(
            "GET /stores/STORE_BLR_002/metrics returns 200 with JSON",
            r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"),
            f"status={r.status_code}, ct={r.headers.get('content-type')}"
        )
    except Exception as e:
        assert_test("GET /stores/{id}/metrics returns 200", False, str(e))

    # -------------------------------------------------------------------
    # 7. Metrics exclude staff from unique_visitors
    # -------------------------------------------------------------------
    try:
        data = r.json()
        assert_test(
            "Metrics unique_visitors excludes staff",
            "unique_visitors" in data and isinstance(data["unique_visitors"], int),
            f"data: {json.dumps(data, indent=2)[:200]}"
        )
    except Exception as e:
        assert_test("Metrics unique_visitors field exists", False, str(e))

    # -------------------------------------------------------------------
    # 8. Funnel endpoint returns stages
    # -------------------------------------------------------------------
    try:
        r = requests.get(f"{base}/stores/STORE_BLR_002/funnel", timeout=10)
        data = r.json()
        has_stages = "stages" in data and len(data["stages"]) > 0
        assert_test(
            "GET /stores/{id}/funnel returns stages",
            r.status_code == 200 and has_stages,
            f"status={r.status_code}, data={json.dumps(data)[:200]}"
        )
    except Exception as e:
        assert_test("GET /stores/{id}/funnel returns stages", False, str(e))

    # -------------------------------------------------------------------
    # 9. Heatmap endpoint returns zone data
    # -------------------------------------------------------------------
    try:
        r = requests.get(f"{base}/stores/STORE_BLR_002/heatmap", timeout=10)
        data = r.json()
        assert_test(
            "GET /stores/{id}/heatmap returns zone data",
            r.status_code == 200 and "zones" in data,
            f"status={r.status_code}, keys={list(data.keys()) if isinstance(data, dict) else 'not dict'}"
        )
    except Exception as e:
        assert_test("GET /stores/{id}/heatmap returns zone data", False, str(e))

    # -------------------------------------------------------------------
    # 10. Anomalies endpoint returns list
    # -------------------------------------------------------------------
    try:
        r = requests.get(f"{base}/stores/STORE_BLR_002/anomalies", timeout=10)
        data = r.json()
        assert_test(
            "GET /stores/{id}/anomalies returns anomaly list",
            r.status_code == 200 and "anomalies" in data and isinstance(data["anomalies"], list),
            f"status={r.status_code}"
        )
    except Exception as e:
        assert_test("GET /stores/{id}/anomalies returns list", False, str(e))

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("🎉 All acceptance gate assertions passed!")
    else:
        print(f"⚠️  {failed} assertion(s) failed — review above.")
    print(f"{'='*50}\n")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
