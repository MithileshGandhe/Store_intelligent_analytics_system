"""
FastAPI application entrypoint for the Store Intelligence API.

• Includes all routers (ingestion, metrics, funnel, heatmap, anomalies, health, ws)
• Mounts the static dashboard at /dashboard
• Middleware: Trace-ID, structured logging, CORS, global exception handler
• Lifespan: DB init → table creation → seed POS data → cleanup
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.database import close_db, get_db, init_db
from app.models import ErrorResponse

# ─── Routers ────────────────────────────────────────────────────────────────────

from app.ingestion import router as ingestion_router, set_ws_manager
from app.metrics import router as metrics_router
from app.funnel import router as funnel_router
from app.heatmap import router as heatmap_router
from app.anomalies import router as anomalies_router
from app.health import router as health_router, set_start_time
from app.websocket import router as ws_router, manager as ws_manager

# ─── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger("store_intelligence")

# ─── Paths ───────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DASHBOARD_DIR = BASE_DIR / "dashboard"

# Data files — check data/ subdir first, then project root, then /app/data (Docker)
def _find_data_file(name: str) -> Path:
    candidates = [
        BASE_DIR / "data" / name,
        BASE_DIR / name,
        Path("/app/data") / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # default even if missing

STORE_LAYOUT_PATH = _find_data_file("store_layout.json")
POS_CSV_PATH = _find_data_file("pos_transactions.csv")


# ─── Lifespan ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    start = time.time()
    set_start_time(start)
    logger.info("Starting Store Intelligence API …")

    # 1. Initialise database
    try:
        await init_db()
        logger.info("Database ready")
    except Exception as exc:
        logger.error("Database init failed: %s", exc)
        # Continue anyway — health endpoint will report unhealthy

    # 2. Load store layout (informational — stored in app state)
    app.state.store_layout = {}
    if STORE_LAYOUT_PATH.exists():
        try:
            with open(STORE_LAYOUT_PATH) as f:
                app.state.store_layout = json.load(f)
            logger.info("Loaded store layout from %s", STORE_LAYOUT_PATH)
        except Exception as exc:
            logger.warning("Could not load store_layout.json: %s", exc)

    # 3. Seed POS transactions from CSV
    if POS_CSV_PATH.exists():
        try:
            await _seed_pos_transactions(POS_CSV_PATH)
        except Exception as exc:
            logger.warning("Could not seed POS data: %s", exc)

    # 4. Wire WebSocket manager into ingestion
    set_ws_manager(ws_manager)

    yield  # ← app is running

    # Shutdown
    logger.info("Shutting down …")
    await close_db()
    logger.info("Goodbye.")


async def _seed_pos_transactions(csv_path: Path):
    """Load pos_transactions.csv into the database (skip duplicates)."""
    db = get_db()
    count = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                await db.execute(
                    """
                    INSERT INTO pos_transactions (transaction_id, store_id, timestamp, basket_value_inr)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (transaction_id) DO NOTHING
                    """,
                    row["transaction_id"],
                    row["store_id"],
                    row["timestamp"],
                    float(row["basket_value_inr"]),
                )
                count += 1
            except Exception as exc:
                logger.debug("POS row skip: %s", exc)
    logger.info("Seeded %d POS transactions from %s", count, csv_path)


# ─── App ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Store Intelligence API",
    description="Real-time store analytics — footfall, conversion, heatmaps, anomalies.",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── Middleware ──────────────────────────────────────────────────────────────────

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_id_middleware(request: Request, call_next):
    """Attach a unique trace_id to every request and response."""
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    request.state.trace_id = trace_id

    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Structured request/response logging with latency."""
    start = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start) * 1000, 2)

    # Extract store_id from path if present
    store_id = "-"
    parts = request.url.path.strip("/").split("/")
    if "stores" in parts:
        idx = parts.index("stores")
        if idx + 1 < len(parts):
            store_id = parts[idx + 1]

    trace_id = getattr(request.state, "trace_id", "-")

    logger.info(
        "request: trace_id=%s method=%s path=%s store=%s status=%d latency_ms=%.2f",
        trace_id,
        request.method,
        request.url.path,
        store_id,
        response.status_code,
        latency_ms,
    )
    return response


# ─── Global exception handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: never expose raw stack traces to clients."""
    trace_id = getattr(request.state, "trace_id", None)

    # Database unreachable → 503
    if isinstance(exc, RuntimeError) and "Database" in str(exc):
        return JSONResponse(
            status_code=503,
            content={
                "error": "service_unavailable",
                "message": "Database connection failed",
                "trace_id": trace_id,
                "retry_after": 30,
            },
        )

    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An internal error occurred",
            "trace_id": trace_id,
        },
    )


# ─── Include routers ────────────────────────────────────────────────────────────

app.include_router(ingestion_router)
app.include_router(metrics_router)
app.include_router(funnel_router)
app.include_router(heatmap_router)
app.include_router(anomalies_router)
app.include_router(health_router)
app.include_router(ws_router)


# ─── Static dashboard ───────────────────────────────────────────────────────────

if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


# ─── Root ────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the dashboard (or return API info if dashboard not mounted)."""
    if DASHBOARD_DIR.exists():
        return RedirectResponse(url="/dashboard")
    return {
        "service": "Store Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
