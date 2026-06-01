# Technical Choices — Store Intelligence System

This document covers three key design decisions made during the build, including the options considered, what AI tools suggested, and the final rationale.

---

## Choice 1: Detection Model Selection

### The Problem

The detection pipeline needs to identify and track individual people in retail CCTV footage at 1080p/15fps. Requirements include: individual person detection (not group blobs), staff vs customer classification, and Re-ID for re-entry handling.

### Options Considered

| Model | Pros | Cons |
|-------|------|------|
| **YOLOv8 (nano/small)** | Fast inference (~5ms/frame), excellent person detection AP, native tracking support via ultralytics | Requires GPU for real-time; nano variant loses accuracy on partial occlusion |
| **YOLOv9** | Improved architecture (GELAN + PGI), better accuracy on occluded targets | Newer, less community support; heavier inference cost |
| **RT-DETR** | Transformer-based; better global context for crowded scenes | Slower inference; overkill for fixed-camera retail where spatial priors are strong |
| **MediaPipe** | Lightweight, CPU-friendly, includes pose estimation | Person detection less robust in crowded/occluded scenarios; no native Re-ID |

### What AI Suggested

**Claude's recommendation**: "YOLOv8-small is the optimal choice for retail CCTV. It balances inference speed (~8ms on GPU) with strong person detection AP (44.9 on COCO). For Re-ID, pair it with ByteTrack for online tracking and OSNet for appearance features."

**GPT-4's recommendation**: "Consider RT-DETR for its superior handling of occlusion through self-attention. The transformer architecture captures global context that YOLO's convolutional approach misses, which is important for the billing queue buildup edge case."

### My Decision: YOLOv8-small + ByteTrack + Centroid-Based Re-ID

I agreed with Claude's recommendation over GPT-4's, with modifications:

1. **YOLOv8-small over RT-DETR**: For fixed retail cameras, spatial priors are extremely strong — the entry is always in the same location, zones don't move. YOLO's speed advantage (3-4x faster inference) matters more than RT-DETR's global context, which helps in dynamic scenes but adds little for static cameras.

2. **ByteTrack over DeepSORT**: ByteTrack's two-stage association (high-confidence first, then low-confidence) handles the partial occlusion edge case more gracefully. DeepSORT's appearance model adds latency and requires a separate Re-ID model.

3. **Centroid-based Re-ID over OSNet**: For the challenge scope (20-minute clips per camera), a simple approach works: cache the exit position + bounding box dimensions + appearance histogram of the last N exited visitors. When a new entry occurs within T seconds, compare against the cache. Full OSNet Re-ID would be needed for production (longer time horizons, more visitors), but adds dependency complexity for the challenge.

4. **Pluggable architecture**: Regardless of model choice, the `DetectorBase` abstraction means swapping models requires zero changes to the tracker, emitter, or API. This is the most important decision — the model will change; the interface should not.

### What I Would Change for Production

At 40 stores in production, I would:
- Upgrade to YOLOv8-medium for better accuracy on occluded targets
- Replace centroid Re-ID with OSNet features for cross-camera deduplication
- Add a VLM (GPT-4V or Gemini Vision) as a secondary classifier for staff detection, prompted with "Does this person appear to be wearing a store uniform? Respond with confidence 0-1." — validated against manual labels

---

## Choice 2: Event Schema Design

### The Problem

Design an event schema that is: (1) compliant with the challenge specification, (2) efficient to query for metrics/funnel/heatmap/anomalies, (3) extensible for future event types.

### Options Considered

**Option A — Flat Schema (Selected)**:
All common fields at the top level. Optional/edge-case fields in a `metadata` object.

```json
{
  "event_id": "uuid",
  "store_id": "STORE_BLR_002",
  "event_type": "ZONE_DWELL",
  "zone_id": "SKINCARE",
  "dwell_ms": 8400,
  "confidence": 0.91,
  "metadata": {"queue_depth": null, "sku_zone": "MOISTURISER", "session_seq": 5}
}
```

**Option B — Fully Normalised**:
Separate tables for visitors, sessions, zones, with events referencing foreign keys.

**Option C — Deeply Nested**:
Nested objects for location, tracking, classification, temporal data.

### What AI Suggested

Claude recommended Option A (flat) with the reasoning: "Flat schemas with a metadata sidecar for optional fields are the standard pattern in event-driven architectures. They optimise for the common read path (SQL queries on top-level columns) while keeping extensibility via the metadata JSON field."

GPT-4 recommended Option C (nested), arguing for "semantic clarity and self-documenting structure."

### My Decision: Option A — Flat with Metadata Sidecar

Agreed with Claude. Key reasons:

1. **Schema compliance**: The challenge specification shows a flat schema in `sample_events.jsonl`. Deviating risks failing automated correctness tests.

2. **Query efficiency**: The most common queries (`COUNT DISTINCT visitor_id WHERE event_type = 'ENTRY' AND is_staff = false`) operate on top-level columns. No JSON path extraction needed.

3. **Storage efficiency**: Flat events are ~30% smaller than nested equivalents. At 40 stores × 3 cameras × hundreds of events per hour, this compounds.

4. **Extensibility**: The `metadata` object handles event-type-specific fields (queue_depth for billing events, sku_zone for zone events) without schema migration. New event types can add new metadata fields without changing the table schema.

**What I disagreed with**: AI suggested making `visitor_id` a UUID. I kept the `VIS_` prefix format (`VIS_c8a2f1`) because:
- It's immediately identifiable as a visitor token in logs and debugging
- The prefix makes cross-table joins more readable
- It matches the challenge specification examples

---

## Choice 3: API Architecture — Synchronous Computation vs Event Sourcing

### The Problem

The API needs to ingest events and serve real-time metrics. Two architectural approaches:

**Option A — Synchronous Query (Selected)**: Store events in PostgreSQL, compute metrics on-demand from raw events using SQL aggregation.

**Option B — Event Sourcing + CQRS**: Use an event log as the source of truth, maintain separate read models (materialised views) for each endpoint, update projections on each ingest.

### What AI Suggested

GPT-4 strongly recommended CQRS: "For real-time analytics, event sourcing with read-model projections is the canonical architecture. Each endpoint maintains its own denormalised view, updated on write. This gives O(1) read latency regardless of event volume."

Claude recommended synchronous queries with PostgreSQL: "For the challenge scope, CQRS is over-engineering. PostgreSQL can aggregate 10K events in <50ms. Add a query result cache if needed. CQRS introduces eventual consistency, projection rebuild logic, and doubles your storage — none of which is justified for 5 stores and 1 hour of data."

### My Decision: Synchronous Queries with PostgreSQL

I agreed with Claude and went with the simpler approach:

1. **Scope-appropriate**: With 5 stores and ~1 hour of data per store, PostgreSQL aggregation queries complete in 10-50ms. CQRS would add complexity with zero performance benefit at this scale.

2. **Correctness over speed**: Synchronous computation guarantees that `/metrics` always reflects the latest ingested events. CQRS introduces an eventual consistency window where metrics could lag behind recent ingests — directly contradicting the "real-time, not cached" requirement.

3. **Simpler debugging**: When a metric looks wrong, I can inspect the raw events and the SQL query. With CQRS, I'd need to trace through projection logic, replay events, and verify the denormalised state.

4. **Clear upgrade path**: The code is structured so that adding a caching layer (Redis with 5-second TTL) or materialised views requires zero endpoint changes — only the metric computation functions need modification.

5. **Production caveat**: At 40 stores with continuous event streams, I would move to CQRS with Kafka as the event log and ClickHouse for analytics queries. But I'd only do this when PostgreSQL query latency exceeds the 200ms SLA — premature optimisation is the root of all evil.

### Database Choice: PostgreSQL over SQLite

While SQLite is simpler, I chose PostgreSQL because:
- **Concurrent writes**: The pipeline may ingest events while the API serves read queries. SQLite's write lock would serialise these operations.
- **Rich aggregation**: Window functions, CTEs, and JSON operators make metric computation cleaner.
- **Docker-native**: PostgreSQL runs as a standard Docker service with health checks. SQLite requires file volume mounting and has no built-in health check.
- **Fallback**: The code includes an SQLite fallback (`USE_SQLITE=true`) for local development without Docker.

---

## Summary

| Decision | AI Recommendation | My Choice | Agreed? |
|----------|-------------------|-----------|---------|
| Detection model | Claude: YOLOv8, GPT-4: RT-DETR | YOLOv8-small + ByteTrack | Agreed with Claude, modified Re-ID approach |
| Event schema | Claude: Flat, GPT-4: Nested | Flat with metadata sidecar | Agreed with Claude |
| API architecture | GPT-4: CQRS, Claude: Synchronous | Synchronous PostgreSQL | Agreed with Claude |

The common thread: **I favoured simplicity-at-scale over theoretical purity.** The system is designed to be simple enough to debug at 3am, with clear upgrade paths documented for when scale demands it.
