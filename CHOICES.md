# Technical Choices — Store Intelligence System

This document covers the key design decisions made during the build, including the options considered, what AI tools suggested, and the final rationale.

---

## Choice 1: Detection Model Selection

### The Problem
The detection pipeline needs to identify and track individual people in retail CCTV footage at 1080p/15fps. Requirements include: individual person detection (not group blobs), staff vs customer classification, and Re-ID for re-entry handling.

### Options Considered
| Model | Pros | Cons |
|-------|------|------|
| **YOLOv8 (nano/small)** | Fast inference (~5ms/frame), excellent person detection AP | Requires GPU for real-time; nano variant loses accuracy on partial occlusion |
| **YOLOv9** | Improved architecture (GELAN + PGI), better accuracy on occluded targets | Newer, less community support; heavier inference cost |
| **RT-DETR** | Transformer-based; better global context for crowded scenes | Slower inference; overkill for fixed-camera retail where spatial priors are strong |
| **MediaPipe** | Lightweight, CPU-friendly, includes pose estimation | Person detection less robust in crowded/occluded scenarios; no native Re-ID |

### My Decision: Pre-trained YOLOv8 + Modular Classifiers (MobileNetV3 + OSNet)
Instead of heavily fine-tuning a monolithic object detection model to predict "staff" vs. "customer", I opted for a composite, modular pipeline:
1. **Person Detection (YOLOv8s):** I use a COCO pre-trained YOLO model filtered to class `0` (person). COCO person detection is incredibly robust off-the-shelf. Fine-tuning YOLO on a limited dataset of store footage risks severe overfitting, yielding minimal gain for maximum effort.
2. **Staff Classification (MobileNetV3-Small):** We extract the bounding box crop of each detected person and pass it through a lightweight, custom-trained MobileNet classifier (binary classification). Because staff uniforms vary significantly store-by-store, this allows us to quickly train and hot-swap store-specific classifiers without retraining the entire detection backbone.
3. **Person Re-ID (OSNet-x0.25):** We use an OSNet architecture specifically optimized for Person Re-Identification to generate a 128-dimensional embedding from the person crop. This correctly handles cross-camera tracking and re-entries.
4. **Pluggable architecture (`DetectorBase`):** The main pipeline orchestrator (`pipeline/detect.py`) communicates with these models through an abstract `DetectorBase` class. Swapping models requires zero changes to the tracker, emitter, or API.

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

### My Decision: Option A — Flat with Metadata Sidecar
1. **Schema compliance**: The challenge specification shows a flat schema in `sample_events.jsonl`. Deviating risks failing automated correctness tests.
2. **Query efficiency**: The most common queries operate on top-level columns. No JSON path extraction needed.
3. **Storage efficiency**: Flat events are ~30% smaller than nested equivalents. At 40 stores × 3 cameras × hundreds of events per hour, this compounds.
4. **Extensibility**: The `metadata` object handles event-type-specific fields without schema migration.

---

## Choice 3: API Architecture — Synchronous Computation vs Event Sourcing

### The Problem
The API needs to ingest events and serve real-time metrics. Two architectural approaches:

**Option A — Synchronous Query (Selected)**: Store events in PostgreSQL, compute metrics on-demand from raw events using SQL aggregation.
**Option B — Event Sourcing + CQRS**: Use an event log as the source of truth, maintain separate read models (materialised views) for each endpoint.

### My Decision: Synchronous Queries with PostgreSQL
1. **Scope-appropriate**: With 5 stores and ~1 hour of data per store, PostgreSQL aggregation queries complete in 10-50ms. CQRS would add complexity with zero performance benefit at this scale.
2. **Correctness over speed**: Synchronous computation guarantees that `/metrics` always reflects the latest ingested events, meeting the "real-time, not cached" requirement.
3. **Production caveat**: At 40 stores with continuous event streams, I would move to CQRS with Kafka as the event log and ClickHouse for analytics queries. But I'd only do this when PostgreSQL query latency exceeds the 200ms SLA.

### Database Choice: PostgreSQL over SQLite
- **Concurrent writes**: The pipeline may ingest events while the API serves read queries.
- **Rich aggregation**: Window functions, CTEs, and JSON operators make metric computation cleaner.
- **Docker-native**: PostgreSQL runs as a standard Docker service with health checks. SQLite requires file volume mounting and has no built-in health check.
