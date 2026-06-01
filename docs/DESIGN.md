# Store Intelligence System — Architecture Design

## Overview

The Store Intelligence system transforms raw CCTV footage from retail stores into actionable analytics via a four-stage pipeline:

```
📹 Raw CCTV → 🔍 Detection Layer → ⚡ Event Stream → 🧠 Intelligence API → 📊 Live Dashboard
```

The system is designed for **Apex Retail's 40-store network**, providing real-time visibility into offline store performance — specifically the **Offline Store Conversion Rate**, which is the north star metric.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                          │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Video    │───▶│ Detector │───▶│ Tracker  │───▶│ Emitter  │ │
│  │  Input    │    │ (YOLO/   │    │ (Re-ID,  │    │ (Schema  │ │
│  │          │    │  Dummy)  │    │  Zones)  │    │  Valid.) │ │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘ │
└──────────────────────────────────────────────────────────┼──────┘
                                                          │
                                              JSONL file / HTTP POST
                                                          │
┌─────────────────────────────────────────────────────────┼──────┐
│                    INTELLIGENCE API (FastAPI)            │      │
│                                                          ▼      │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────────┐  │
│  │ Ingest   │───▶│PostgreSQL│◀───│  Metric Computation      │  │
│  │ Endpoint │    │   DB     │    │  • Metrics  • Funnel     │  │
│  └──────────┘    └──────────┘    │  • Heatmap  • Anomalies  │  │
│                       │          └──────────────────────────┘  │
│                       │                     │                   │
│                  ┌────┴────┐          ┌─────┴─────┐            │
│                  │  Health │          │ WebSocket  │            │
│                  │  Check  │          │ Broadcast  │            │
│                  └─────────┘          └─────┬─────┘            │
└──────────────────────────────────────────────┼─────────────────┘
                                               │
                                         WebSocket + REST
                                               │
┌──────────────────────────────────────────────┼─────────────────┐
│                    LIVE DASHBOARD             │                  │
│                                               ▼                  │
│  ┌───────────┬──────────┬──────────┬──────────┐                 │
│  │ Visitors  │ Convert. │  Queue   │ Abandon. │  Metric Cards   │
│  │  Count    │  Rate    │  Depth   │  Rate    │                 │
│  └───────────┴──────────┴──────────┴──────────┘                 │
│  ┌──────────────────┐  ┌──────────────────────┐                 │
│  │ Conversion Funnel│  │    Zone Heatmap      │                 │
│  │ Entry→Zone→Bill  │  │  Color-coded grid    │                 │
│  └──────────────────┘  └──────────────────────┘                 │
│  ┌──────────────────┐  ┌──────────────────────┐                 │
│  │ Active Anomalies │  │  Live Event Feed     │                 │
│  └──────────────────┘  └──────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. Detection Pipeline (`pipeline/`)

**Purpose**: Process CCTV video frames into structured behavioural events.

**Architecture**: Pluggable detector model behind an abstract base class (`DetectorBase`). The current implementation uses a `DummyDetector` that generates synthetic detections, designed to be swapped for a real model (YOLOv8, RT-DETR, etc.) by implementing a single class.

**Flow**:
1. **Frame Extraction**: OpenCV reads video at configurable FPS (default: 5 fps from 15 fps source)
2. **Detection**: Detector identifies people in each frame with bounding boxes and confidence scores
3. **Tracking**: `VisitorTracker` maintains state across frames — assigns visitor IDs, tracks zone transitions, detects entry/exit direction, handles Re-ID for re-entries
4. **Emission**: `EventEmitter` converts state transitions into schema-compliant events, validates with Pydantic, outputs to JSONL and/or POSTs to API

**Edge Case Handling**:
- **Group entry**: Detector returns individual bounding boxes; each gets a unique visitor_id
- **Staff exclusion**: `classify_staff()` method flags staff based on appearance features
- **Re-entry**: Feature vector comparison against recent EXIT visitors
- **Partial occlusion**: Low-confidence detections are emitted (not suppressed) with `confidence` field reflecting certainty
- **Empty periods**: Pipeline correctly handles zero-detection frames without errors

### 2. Intelligence API (`app/`)

**Framework**: FastAPI (async) with PostgreSQL via asyncpg.

**Design Principles**:
- **Idempotent ingestion**: `event_id` is the primary key; duplicate ingests are safely ignored via `ON CONFLICT DO NOTHING`
- **Real-time computation**: All metrics are computed from current data, not cached aggregates
- **Structured errors**: No raw stack traces; all errors return JSON with `error`, `message`, `detail` fields
- **Graceful degradation**: DB failures → HTTP 503 with `retry_after` header

**Endpoints**:
| Endpoint | Purpose |
|----------|---------|
| `POST /events/ingest` | Batch ingest up to 500 events with partial success |
| `GET /stores/{id}/metrics` | Real-time daily KPIs (visitors, conversion, dwell, queue) |
| `GET /stores/{id}/funnel` | Session-based conversion funnel with drop-off % |
| `GET /stores/{id}/heatmap` | Zone visit frequency normalised 0-100 |
| `GET /stores/{id}/anomalies` | Active anomalies with severity and suggested actions |
| `GET /health` | Service health with per-store feed freshness |

### 3. Live Dashboard (`dashboard/`)

**Technology**: Pure HTML/CSS/JS — no build step required.

**Connection**: WebSocket to `/ws/dashboard/{store_id}` for live metric streaming, with REST polling fallback every 30 seconds.

**Design**: Dark glassmorphism theme with animated metric cards, conversion funnel visualisation, zone heatmap grid, and real-time event feed.

### 4. Data Layer

**Storage**: PostgreSQL 16 (Docker) with fallback to SQLite for local development.

**Schema**:
- `events` table — all event fields, `event_id` as PRIMARY KEY for idempotency
- `pos_transactions` table — POS records loaded on startup for conversion correlation

---

## AI-Assisted Decisions

### Decision 1: Pluggable Detector Architecture

**AI Suggestion**: Claude suggested creating a factory pattern with a registry of detector implementations, where each detector registers itself at import time.

**My Decision**: I agreed with the abstraction principle but simplified to a direct abstract base class + concrete implementations. The factory pattern added unnecessary complexity for a system with only 2-3 detector variants. The current approach — `DetectorBase` → `DummyDetector` / `YoloDetector` — is simpler to understand and equally extensible.

**Outcome**: The pluggable architecture works well. Swapping detectors requires implementing 2 methods (`detect`, `classify_staff`) and changing one CLI flag.

### Decision 2: Event Schema Design — Flat vs Nested

**AI Suggestion**: An LLM proposed a deeply nested event schema with separate objects for `location`, `tracking`, `classification`, and `temporal` data — arguing it was more "semantically clean."

**My Decision**: I overrode this in favour of a flatter schema with a single `metadata` object for optional fields. Reasoning:
1. The scoring harness expects the flat schema from `sample_events.jsonl`
2. Flat schemas are faster to query in SQL (no JSON path gymnastics for common fields)
3. The nested approach increased payload size by ~35% for no analytical benefit
4. Every consumer (API, dashboard, tests) would need to destructure the nested objects

**Outcome**: The flat schema with `metadata` for edge-case fields (queue_depth, sku_zone) is the right balance. Common query patterns (`WHERE event_type = 'ENTRY' AND is_staff = false`) work directly on top-level columns.

### Decision 3: Real-Time Metrics vs Pre-Computed Aggregates

**AI Suggestion**: GPT-4 suggested using materialised views or a pre-computation layer (like a CRON job that runs every minute) to cache metrics, arguing it would be faster at scale.

**My Decision**: I chose real-time computation for this challenge, with the understanding that pre-computation would be the right move at 40 live stores. Reasoning:
1. With the challenge dataset (5 stores, ~1 hour of data), real-time queries complete in <50ms
2. Pre-computed aggregates introduce staleness — contradicting the "real-time, not cached from yesterday" requirement
3. The WebSocket dashboard expects immediate updates on new events
4. Adding caching later is straightforward (query result cache with TTL) without architectural change

**Outcome**: Real-time computation meets the challenge requirements. The API includes a note in the code where a caching layer would be inserted for production scale.

---

## Production Considerations

### What Would Change at Scale (40 Stores, Real-Time)

1. **Database**: Move from single PostgreSQL to a time-series database (TimescaleDB) or event store (Kafka + ClickHouse) for high write throughput
2. **Metrics**: Pre-compute 1-minute and 5-minute rollups to avoid query pressure on raw events
3. **Detection Pipeline**: Run as a Kubernetes job per store with GPU nodes; stream events via Kafka rather than HTTP POST
4. **Dashboard**: Add per-store caching, CDN for static assets, and WebSocket fanout via Redis pub/sub
5. **Monitoring**: Prometheus + Grafana for pipeline lag, detection accuracy drift, and API latency percentiles

### Security (Not Implemented — Challenge Scope)

- API authentication (JWT/API key) for production
- Rate limiting on ingest endpoint
- TLS for all connections
- POS data encryption at rest

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Detection | Python + OpenCV | Industry standard for CV pipelines |
| Model | YOLOv8 (pluggable, dummy for testing) | Best speed/accuracy for retail person detection |
| Tracking | Custom centroid tracker | Sufficient for fixed-camera retail; avoids heavy dependencies |
| API | FastAPI (async) | High performance, automatic OpenAPI docs, native async |
| Database | PostgreSQL 16 | ACID compliance, concurrent writes, rich aggregation |
| Dashboard | HTML/CSS/JS + WebSocket | No build step, instant load, live updates |
| Container | Docker Compose | Single-command startup, reproducible |
| Testing | pytest + httpx | Async test support, coverage reporting |
