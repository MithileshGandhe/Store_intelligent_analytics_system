# Store Intelligence API — System Design

This document details the high-level architecture and data flow of the Store Intelligence system, built to process raw CCTV footage into actionable, real-time analytics.

## 1. High-Level Architecture

The system is separated into three distinct, decoupled components:

1.  **Computer Vision Pipeline (Edge Inference):** Processes video frames, detects people, tracks their identities, and emits standardized events.
2.  **FastAPI Backend (Intelligence API):** Ingests events from the pipeline, processes metrics, and serves REST/WebSocket endpoints.
3.  **Live Dashboard (Client UI):** A premium, glassmorphism-themed frontend that connects to the API via WebSockets for real-time visualization.

---

## 2. The Vision Pipeline Data Flow

The `pipeline.detect` module orchestrates the vision analysis. The flow for every video frame is as follows:

### A. Detection (`YOLODetector`)
*   **Frame Capture:** Video frames are extracted via OpenCV.
*   **Bounding Boxes:** YOLOv8s (pre-trained on COCO) runs inference to find all persons in the frame. We filter for `class 0` (person) and apply Non-Maximum Suppression (NMS).
*   **Staff Classification:** Each person's bounding box is cropped and passed through a custom MobileNetV3-Small classifier to determine if they are `Staff` or `Customer`.
*   **Re-Identification:** The crop is passed through a custom OSNet-x0.25 model to extract a 128-dimensional embedding (feature vector) representing the person's unique appearance.

### B. Tracking (`VisitorTracker`)
*   **Spatial Tracking (IoU):** Within a single camera view, we use Intersection-over-Union (IoU) matching to track individuals from frame to frame.
*   **Global Tracking (Re-ID):** The 128-d Re-ID embeddings allow us to match identities across different cameras. If a person disappears from the entry camera and appears at the billing camera, the system recognizes the identical embeddings.
*   **Zone Intersection:** We project the centroid of each person's bounding box onto the calibrated store layout polygons. If a person's centroid enters the "billing" polygon, a zone interaction is registered.

### C. Event Emission (`EventEmitter`)
*   The tracker generates internal pipeline events (e.g., `ENTRY`, `ZONE_ENTER`). 
*   The `EventConverter` translates these into the strict schema required by the API (`Type A`, `Type B`, `Type C` as seen in `events.jsonl`).
*   Events are either written to a local `.jsonl` file or HTTP POSTed directly to the live API endpoint.

---

## 3. API & Backend Design

The backend is built with **FastAPI** for maximum concurrency and performance.

### Data Ingestion
*   **`POST /events/ingest`**: Receives batches of translated events from the pipeline. Validates the JSON schema using Pydantic.
*   **Database:** Events are persisted in an SQLite database (easily swappable to PostgreSQL) using SQLAlchemy ORM.

### Analytics & Endpoints
*   **`GET /metrics/live/{store_id}`**: Computes live occupancy, current staff count, and active queue sizes based on the most recent events.
*   **`GET /metrics/funnel/{store_id}`**: Computes the conversion funnel (Total Visitors -> Zonal Engagement -> Checkout Queue).
*   **`GET /metrics/heatmap/{store_id}`**: Aggregates spatial data (x, y coordinates) from Type A events to generate activity heatmaps.

### Real-Time Delivery (WebSocket)
*   **`WS /ws/dashboard/{store_id}`**: The API maintains active WebSocket connections. When new events are ingested, a background task recalculates live metrics and broadcasts a JSON payload to all connected dashboard clients simultaneously.

---

## 4. Training Infrastructure (`/training`)

The system includes a fully self-contained training suite:
1.  **Local Data Preparation:** `prepare_frames.py` and GUI-based `build_staff_dataset.py` allow users to extract and manually label custom staff uniforms locally.
2.  **Google Colab Training:** The provided Jupyter notebooks load the local datasets to train the MobileNet and OSNet models using free cloud GPUs.
3.  **Plug-and-Play Inference:** The resulting `.pth` files are simply dropped into the local machine for CPU-based edge inference, ensuring the heavy training process never blocks the production deployment.
