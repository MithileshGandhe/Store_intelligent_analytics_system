# Store Intelligence System

Real-time store analytics pipeline — from raw CCTV footage to live conversion metrics.

Built for the Apex Retail Engineering Hiring Challenge.

## Quick Start (5 Commands)

```bash
# 1. Clone the repository
git clone https://github.com/MithileshGandhe/Store_intelligent_analytics_system && cd store-intelligence

# 2. Start the API + database
docker compose up -d

# 3. Verify the API is running
curl http://localhost:8000/health

# 4. Run the detection pipeline (generates synthetic events and ingests them)
pip install -r requirements-pipeline.txt
python -m pipeline.detect --synthetic 500 --store STORE_BLR_002 --camera CAM_ENTRY_01 --api-url http://localhost:8000

# 5. Open the dashboard
# Navigate to http://localhost:8000/dashboard in your browser
```

## Project Structure

```
store-intelligence/
├── pipeline/                  # Detection pipeline
│   ├── detector_base.py       # Abstract detector interface
│   ├── tracker.py             # Visitor tracking + Re-ID
│   ├── emit.py                # Event schema + emission
│   ├── detect.py              # Main orchestrator (CLI)
│   ├── run.sh                 # Process all clips (Linux/Mac)
│   └── run.ps1                # Process all clips (Windows)
├── app/                       # FastAPI Intelligence API
│   ├── main.py                # App entrypoint + middleware
│   ├── models.py              # Pydantic schemas
│   ├── database.py            # PostgreSQL + SQLite backends
│   ├── ingestion.py           # POST /events/ingest
│   ├── metrics.py             # GET /stores/{id}/metrics
│   ├── funnel.py              # GET /stores/{id}/funnel
│   ├── heatmap.py             # GET /stores/{id}/heatmap
│   ├── anomalies.py           # GET /stores/{id}/anomalies
│   ├── health.py              # GET /health
│   └── websocket.py           # WebSocket live updates
├── dashboard/                 # Live web dashboard
│   ├── index.html
│   ├── styles.css
│   └── dashboard.js
├── tests/                     # Test suite (>70% coverage)
│   ├── conftest.py
│   ├── test_pipeline.py
│   ├── test_ingestion.py
│   ├── test_metrics.py
│   ├── test_funnel.py
│   └── test_anomalies.py
├── data/                      # Dataset
│   ├── store_layout.json
│   ├── pos_transactions.csv
│   └── sample_events.jsonl
├── docs/
│   ├── DESIGN.md              # Architecture + AI-assisted decisions
│   └── CHOICES.md             # 3 key technical decisions
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.pipeline
├── requirements.txt
├── requirements-pipeline.txt
├── assertions.py              # 10 acceptance gate assertions
└── README.md
```

## Running the Detection Pipeline

### Option A: Synthetic Mode (No Video Required)

Generate synthetic events for testing:

```bash
# Generate 500 frames of synthetic events and ingest into API
python -m pipeline.detect --synthetic 500 \
    --store STORE_BLR_002 \
    --camera CAM_ENTRY_01 \
    --output output/events.jsonl \
    --api-url http://localhost:8000
```

### Option B: Process Video Files

```bash
# Single video
python -m pipeline.detect \
    --input clips/store_blr_002_entry.mp4 \
    --store STORE_BLR_002 \
    --camera CAM_ENTRY_01 \
    --output output/events.jsonl \
    --api-url http://localhost:8000

# All clips in a directory
python -m pipeline.detect \
    --input clips/ \
    --store STORE_BLR_002 \
    --camera CAM_ENTRY_01 \
    --output output/events.jsonl
```

### Option C: Using the Script

```bash
# Linux/Mac
./pipeline/run.sh --input-dir ./clips --output-dir ./output/events --api-url http://localhost:8000

# Windows
.\pipeline\run.ps1 -InputDir .\clips -OutputDir .\output\events -ApiUrl http://localhost:8000
```

### Option D: Docker Pipeline

```bash
# Mount clips directory and run pipeline in container
docker compose --profile pipeline run pipeline
```

### Swapping the Detector

The pipeline uses a pluggable detector architecture. To use a real model:

1. Create a new file (e.g., `pipeline/yolo_detector.py`) implementing `DetectorBase`
2. Register it in `pipeline/detect.py` → `DETECTOR_REGISTRY`
3. Run with `--detector yolo`

See `pipeline/detector_base.py` for the interface contract.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/events/ingest` | POST | Batch ingest up to 500 events (idempotent) |
| `/stores/{id}/metrics` | GET | Today's KPIs: visitors, conversion, dwell, queue |
| `/stores/{id}/funnel` | GET | Conversion funnel: Entry → Zone → Billing → Purchase |
| `/stores/{id}/heatmap` | GET | Zone visit frequency, normalised 0–100 |
| `/stores/{id}/anomalies` | GET | Active anomalies with severity and suggested actions |
| `/health` | GET | Service health, DB status, feed freshness |
| `/docs` | GET | Interactive OpenAPI documentation |
| `/dashboard` | GET | Live web dashboard |

### Example: Ingest Events

```bash
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{"events": [{"event_id": "uuid-here", "store_id": "STORE_BLR_002", "camera_id": "CAM_ENTRY_01", "visitor_id": "VIS_abc123", "event_type": "ENTRY", "timestamp": "2026-03-03T10:05:12Z", "zone_id": null, "dwell_ms": 0, "is_staff": false, "confidence": 0.94, "metadata": {"queue_depth": null, "sku_zone": null, "session_seq": 1}}]}'
```

### Example: Get Metrics

```bash
curl http://localhost:8000/stores/STORE_BLR_002/metrics
```

## 📊 Live Dashboard

The dashboard is available at `http://localhost:8000/dashboard` when the API is running.

Features:
- **Real-time metrics** via WebSocket (auto-updates when events are ingested)
- **Conversion funnel** with drop-off percentages
- **Zone heatmap** with color-coded visit frequency
- **Active anomalies** with severity badges
- **Live event feed** showing the latest 20 events

## Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt pytest pytest-asyncio anyio httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov=pipeline --cov-report=term-missing

# Run acceptance gate assertions (requires running API)
python assertions.py --api-url http://localhost:8000
```

## Docker

```bash
# Start everything (API + PostgreSQL)
docker compose up -d

# Check logs
docker compose logs -f api

# Stop
docker compose down

# Full reset (including data)
docker compose down -v
```

## Architecture

See [DESIGN.md](docs/DESIGN.md) for full architecture overview and AI-assisted decisions.

See [CHOICES.md](docs/CHOICES.md) for key technical decisions with reasoning.

## Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment for SQLite mode
export USE_SQLITE=true  # Linux/Mac
$env:USE_SQLITE="true"  # Windows PowerShell

# Run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, run the pipeline
python -m pipeline.detect --synthetic 300 --api-url http://localhost:8000
```

## North Star Metric

**Offline Store Conversion Rate** = Visitors who completed a purchase ÷ Total unique visitors

Every component serves this metric:
- **Detection Pipeline** → accurate visitor counting
- **Metrics API** → real-time conversion rate
- **Funnel** → where customers drop off
- **Heatmap** → which zones attract attention
- **Anomalies** → when something breaks
