"""
detect.py — Main orchestrator for the detection pipeline.

Processes video files (or generates synthetic frames for testing) through
the detector → tracker → emitter pipeline.

Usage:
    python -m pipeline.detect --input video.mp4 --store STORE_BLR_002 --camera CAM_ENTRY_01 --output events.jsonl
    python -m pipeline.detect --input clips/ --store STORE_BLR_002 --camera CAM_ENTRY_01 --output events.jsonl
    python -m pipeline.detect --synthetic 500 --store STORE_BLR_002 --camera CAM_ENTRY_01  # 500 synthetic frames

CLI Arguments:
    --input          Path to video file or directory of videos
    --synthetic      Number of synthetic frames to generate (no video file needed)
    --store          Store ID (default: STORE_BLR_002)
    --camera         Camera ID (default: CAM_ENTRY_01)
    --output         Output JSONL file path (default: events.jsonl)
    --detector       Detector backend: dummy | yolo (default: dummy)
    --fps            Processing frame rate (default: 5)
    --api-url        Optional API endpoint for event ingestion
    --store-layout   Path to store_layout.json
    --start-time     Clip start time ISO-8601 (default: now)
    --verbose        Enable debug logging
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running as `python pipeline/detect.py` or `python -m pipeline.detect`
_PIPELINE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _PIPELINE_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from pipeline.detector_base import DetectorBase
from pipeline.dummy_detector import DummyDetector
from pipeline.tracker import VisitorTracker
from pipeline.emit import EventEmitter

logger = logging.getLogger("pipeline.detect")


# --------------------------------------------------------------------------- #
#  Detector registry                                                          #
# --------------------------------------------------------------------------- #

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TO ADD A NEW DETECTOR:                                                 ║
# ║  1. Create your detector class (subclass DetectorBase)                  ║
# ║  2. Import it here                                                      ║
# ║  3. Add an entry to DETECTOR_REGISTRY below                             ║
# ║  4. Use it via CLI: --detector <your_key>                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

DETECTOR_REGISTRY: dict = {
    "dummy": DummyDetector,
    # "yolo": YOLODetector,       # Uncomment when YOLOv8 detector is ready
    # "rtdetr": RTDETRDetector,   # Uncomment when RT-DETR detector is ready
}


# --------------------------------------------------------------------------- #
#  Video source abstraction                                                   #
# --------------------------------------------------------------------------- #

class VideoSource:
    """Wraps cv2.VideoCapture or generates synthetic frames.

    Handles gracefully when OpenCV is not installed or video files
    are missing — falls back to synthetic frame generation.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        synthetic_frames: int = 0,
        target_fps: float = 5.0,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> None:
        self._path = path
        self._synthetic_frames = synthetic_frames
        self._target_fps = target_fps
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._cap = None
        self._native_fps: float = 30.0
        self._total_frames: int = 0
        self._current_frame: int = 0
        self._is_synthetic: bool = False

        if path and not synthetic_frames:
            self._try_open_video(path)
        else:
            self._is_synthetic = True
            self._total_frames = synthetic_frames or 300  # default 1 minute at 5fps

    def _try_open_video(self, path: str) -> None:
        """Attempt to open a video file with OpenCV."""
        try:
            import cv2
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                logger.warning("Cannot open video '%s' — using synthetic frames", path)
                self._is_synthetic = True
                self._total_frames = 300
                return
            self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(
                "Opened video: %s — %.1f fps, %d frames, %dx%d",
                path, self._native_fps, self._total_frames,
                self._frame_width, self._frame_height,
            )
        except ImportError:
            logger.warning("OpenCV not installed — using synthetic frames")
            self._is_synthetic = True
            self._total_frames = 300

    def __iter__(self):
        """Yield (frame_index, frame_ndarray) tuples at the target FPS."""
        if self._is_synthetic:
            yield from self._synthetic_generator()
        else:
            yield from self._video_generator()

    def _video_generator(self):
        """Read frames from a real video file, sampling at target FPS."""
        import cv2
        skip = max(1, int(self._native_fps / self._target_fps))
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % skip == 0:
                yield frame_idx, frame
        self._cap.release()

    def _synthetic_generator(self):
        """Generate synthetic frames for testing."""
        rng = np.random.default_rng(12345)
        for i in range(self._total_frames):
            # Simple synthetic frame with store-like appearance
            frame = np.full(
                (self._frame_height, self._frame_width, 3),
                fill_value=175, dtype=np.uint8,
            )
            # Add subtle noise
            noise = rng.integers(0, 15, size=frame.shape, dtype=np.uint8)
            frame = np.clip(
                frame.astype(np.int16) + noise.astype(np.int16), 0, 255
            ).astype(np.uint8)
            yield i + 1, frame

    @property
    def is_synthetic(self) -> bool:
        return self._is_synthetic

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def native_fps(self) -> float:
        return self._native_fps

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


# --------------------------------------------------------------------------- #
#  Pipeline runner                                                            #
# --------------------------------------------------------------------------- #

def run_pipeline(
    video_source: VideoSource,
    detector: DetectorBase,
    tracker: VisitorTracker,
    emitter: EventEmitter,
    start_time: float,
    target_fps: float = 5.0,
) -> int:
    """Run the detection → tracking → emission pipeline.

    Args:
        video_source: Frame source (video or synthetic).
        detector: Person detector instance.
        tracker: Visitor tracker instance.
        emitter: Event emitter instance.
        start_time: Clip start time as epoch seconds.
        target_fps: Target processing frame rate.

    Returns:
        Total number of events emitted.
    """
    frame_interval = 1.0 / target_fps
    total_events = 0
    processed = 0

    logger.info("Starting pipeline — target FPS: %.1f", target_fps)

    for frame_idx, frame in video_source:
        # Compute timestamp from clip start + frame offset
        timestamp = start_time + (frame_idx * frame_interval)

        # 1. Detect people in the frame
        detections = detector.detect(frame)

        # 2. Update tracker with new detections
        events = tracker.update(detections, timestamp)

        # 3. Emit events
        if events:
            emitted = emitter.emit(events)
            total_events += len(emitted)

        processed += 1

        # Progress logging
        if processed % 100 == 0:
            logger.info(
                "Progress: %d frames processed, %d events emitted, "
                "%d active visitors, queue depth=%d",
                processed, total_events,
                tracker.active_visitor_count,
                tracker.billing_queue_depth,
            )

    # Final flush
    emitter.flush()

    logger.info(
        "Pipeline complete: %d frames → %d events",
        processed, total_events,
    )

    return total_events


# --------------------------------------------------------------------------- #
#  CLI entry point                                                            #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Store Intelligence Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video
  python -m pipeline.detect --input store_cam1.mp4 --store STORE_BLR_002 --camera CAM_ENTRY_01

  # Generate synthetic test events (no video needed)
  python -m pipeline.detect --synthetic 500 --store STORE_BLR_002 --camera CAM_ENTRY_01

  # Process with API ingestion
  python -m pipeline.detect --synthetic 300 --api-url http://localhost:8000
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to video file or directory containing video files",
    )
    parser.add_argument(
        "--synthetic", "-S",
        type=int,
        default=0,
        help="Generate N synthetic frames for testing (no video file needed)",
    )
    parser.add_argument(
        "--store",
        type=str,
        default="STORE_BLR_002",
        help="Store ID (default: STORE_BLR_002)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="CAM_ENTRY_01",
        help="Camera ID (default: CAM_ENTRY_01)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="events.jsonl",
        help="Output JSONL file path (default: events.jsonl)",
    )
    parser.add_argument(
        "--detector", "-d",
        type=str,
        default="dummy",
        choices=list(DETECTOR_REGISTRY.keys()),
        help="Detector backend (default: dummy)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Processing frame rate (default: 5.0)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="API endpoint for event ingestion (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--store-layout",
        type=str,
        default=None,
        help="Path to store_layout.json (default: auto-detect)",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Clip start time in ISO-8601 format (default: current time)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def resolve_store_layout(explicit_path: Optional[str]) -> str:
    """Find the store_layout.json file.

    Search order:
      1. Explicit --store-layout path
      2. ../data/store_layout.json (relative to pipeline/)
      3. data/store_layout.json (relative to CWD)
    """
    if explicit_path:
        return explicit_path

    candidates = [
        _PROJECT_DIR / "data" / "store_layout.json",
        Path("data") / "store_layout.json",
        Path("store_layout.json"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    logger.warning("store_layout.json not found — tracker will use empty zones")
    return str(candidates[0])


def main() -> None:
    """Main entry point for the detection pipeline."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 70)
    logger.info("Store Intelligence Detection Pipeline")
    logger.info("=" * 70)

    # Validate inputs
    if not args.input and not args.synthetic:
        logger.info("No --input or --synthetic specified — defaulting to 300 synthetic frames")
        args.synthetic = 300

    # Parse start time
    if args.start_time:
        try:
            start_dt = datetime.fromisoformat(args.start_time)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            start_time = start_dt.timestamp()
        except ValueError:
            logger.error("Invalid --start-time format: %s", args.start_time)
            sys.exit(1)
    else:
        start_time = time.time()

    # Resolve store layout
    layout_path = resolve_store_layout(args.store_layout)
    logger.info("Store layout: %s", layout_path)

    # Build detector
    detector_cls = DETECTOR_REGISTRY[args.detector]
    detector = detector_cls()
    detector.initialize({
        "camera_type": "overhead" if "ENTRY" in args.camera or "BILLING" in args.camera else "angled",
        "seed": 42,
    })
    logger.info("Detector: %s (%s)", args.detector, detector_cls.__name__)

    # Build tracker
    tracker = VisitorTracker(
        store_layout_path=layout_path,
    )

    # Collect video files to process
    video_files = []
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            video_files = [input_path]
        elif input_path.is_dir():
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
            video_files = sorted(
                f for f in input_path.iterdir()
                if f.suffix.lower() in video_exts
            )
            if not video_files:
                logger.warning("No video files found in %s — using synthetic frames", input_path)
                args.synthetic = 300
        else:
            logger.warning("Input path '%s' not found — using synthetic frames", args.input)
            args.synthetic = 300

    # Process each video (or synthetic)
    if video_files:
        for video_path in video_files:
            logger.info("Processing: %s", video_path)
            # Build output filename based on video
            output_name = video_path.stem + "_events.jsonl"
            output_path = Path(args.output).parent / output_name if len(video_files) > 1 else args.output

            emitter = EventEmitter(
                store_id=args.store,
                camera_id=args.camera,
                output_path=str(output_path),
                api_url=args.api_url,
            )

            source = VideoSource(
                path=str(video_path),
                target_fps=args.fps,
            )

            total = run_pipeline(source, detector, tracker, emitter, start_time, args.fps)
            logger.info("Emitted %d events → %s", total, output_path)
            emitter.close()
            source.close()
    else:
        # Synthetic mode
        n_frames = args.synthetic or 300
        logger.info("Synthetic mode: generating %d frames", n_frames)

        emitter = EventEmitter(
            store_id=args.store,
            camera_id=args.camera,
            output_path=args.output,
            api_url=args.api_url,
        )

        source = VideoSource(
            synthetic_frames=n_frames,
            target_fps=args.fps,
        )

        total = run_pipeline(source, detector, tracker, emitter, start_time, args.fps)
        logger.info("Emitted %d events → %s", total, args.output)
        emitter.close()
        source.close()

    logger.info("=" * 70)
    logger.info("Pipeline finished successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
