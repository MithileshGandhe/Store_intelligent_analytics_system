"""
dummy_detector.py — Synthetic detector for testing the pipeline end-to-end.

Generates realistic-looking person detections WITHOUT any ML model.
Uses deterministic seeding so results are reproducible across runs.

╔══════════════════════════════════════════════════════════════════════╗
║  HOW TO SWAP IN A REAL MODEL:                                       ║
║                                                                      ║
║  1. Create a new file, e.g. `pipeline/yolo_detector.py`              ║
║  2. Subclass DetectorBase (from detector_base.py)                    ║
║  3. In initialize(): load your YOLOv8/RT-DETR weights               ║
║  4. In detect(): run inference and return List[Detection]            ║
║  5. In classify_staff(): use uniform/appearance classifier           ║
║  6. Register in detect.py DETECTOR_REGISTRY:                         ║
║       DETECTOR_REGISTRY["yolo"] = YOLODetector                       ║
║                                                                      ║
║  That's it — zero changes to tracker.py, emit.py, or the pipeline.  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import math
from typing import List, Optional

import numpy as np

from .detector_base import Detection, DetectorBase


# --------------------------------------------------------------------------- #
#  Realistic distribution parameters                                          #
# --------------------------------------------------------------------------- #
# Average number of people visible per frame by time-of-day "feel"
_PEOPLE_LAMBDA = 3.5          # Poisson λ for person count per frame
_MAX_PEOPLE = 8               # Hard cap
_STAFF_RATIO = 0.15           # ~15% of detections are staff
_CONF_MIN = 0.55              # Minimum confidence
_CONF_MAX = 0.98              # Maximum confidence
_FEATURE_DIM = 128            # Re-ID feature vector dimension
_NUM_PERSISTENT_IDS = 20      # Pool of "people" the dummy cycles through


class DummyDetector(DetectorBase):
    """Synthetic detector that generates realistic but fake detections.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  SWAP POINT: Replace this entire class with a real detector.       │
    │  See docstring at module top for step-by-step instructions.        │
    └─────────────────────────────────────────────────────────────────────┘

    The dummy maintains a pool of "persistent identities" so that the
    tracker's Re-ID logic can link detections across frames. Each identity
    has a stable feature vector that drifts slightly per frame (simulating
    appearance changes due to lighting / angle).
    """

    def __init__(self) -> None:
        self._frame_idx: int = 0
        self._identity_features: Optional[np.ndarray] = None
        self._identity_is_staff: Optional[np.ndarray] = None
        self._camera_type: str = "overhead"  # "overhead" | "angled"
        self._rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------ #
    #  Initialization                                                     #
    # ------------------------------------------------------------------ #
    def initialize(self, config: dict) -> None:
        """Set up the persistent identity pool.

        Config keys (all optional):
            seed (int): master random seed, default 42
            camera_type (str): 'overhead' or 'angled', affects bbox layout
            num_identities (int): pool size, default 20
            staff_ratio (float): fraction of identities that are staff
        """
        seed = config.get("seed", 42)
        self._camera_type = config.get("camera_type", "overhead")
        n_ids = config.get("num_identities", _NUM_PERSISTENT_IDS)
        staff_ratio = config.get("staff_ratio", _STAFF_RATIO)

        self._rng = np.random.default_rng(seed)

        # Pre-generate stable feature vectors for each identity
        self._identity_features = self._rng.standard_normal(
            (n_ids, _FEATURE_DIM)
        ).astype(np.float32)
        # Normalize to unit sphere
        norms = np.linalg.norm(self._identity_features, axis=1, keepdims=True)
        self._identity_features /= np.clip(norms, 1e-6, None)

        # Assign staff status to identities
        n_staff = max(1, int(n_ids * staff_ratio))
        self._identity_is_staff = np.zeros(n_ids, dtype=bool)
        self._identity_is_staff[:n_staff] = True
        # Shuffle so staff aren't always the first IDs
        self._rng.shuffle(self._identity_is_staff)

    # ------------------------------------------------------------------ #
    #  Core detection                                                     #
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Generate synthetic detections for a single frame.

        Args:
            frame: Video frame (used only for shape; content is ignored).

        Returns:
            List of Detection objects with realistic bounding boxes,
            confidence scores, staff labels, and Re-ID feature vectors.
        """
        if self._rng is None:
            self.initialize({})

        self._frame_idx += 1

        # Deterministic per-frame RNG (reproducible given frame index)
        frame_seed = int(
            hashlib.md5(f"frame_{self._frame_idx}".encode()).hexdigest()[:8], 16
        )
        frng = np.random.default_rng(frame_seed)

        # Number of people follows a Poisson distribution
        n_people = min(frng.poisson(_PEOPLE_LAMBDA), _MAX_PEOPLE)
        if n_people == 0:
            return []

        # Pick which identities appear in this frame
        n_ids = len(self._identity_features)
        active_ids = frng.choice(n_ids, size=n_people, replace=False)

        detections: List[Detection] = []
        for i, identity_idx in enumerate(active_ids):
            bbox = self._generate_bbox(frng, i, n_people)
            confidence = self._generate_confidence(frng)
            is_staff = bool(self._identity_is_staff[identity_idx])

            # Feature vector: stable identity + small per-frame noise
            base_feat = self._identity_features[identity_idx].copy()
            noise = frng.standard_normal(_FEATURE_DIM).astype(np.float32) * 0.05
            feat = base_feat + noise
            feat /= np.linalg.norm(feat) + 1e-8  # re-normalize

            detections.append(
                Detection(
                    bbox=tuple(bbox),
                    confidence=confidence,
                    is_staff=is_staff,
                    track_id=int(identity_idx),  # hint for tracker
                    features=feat,
                )
            )

        return detections

    def classify_staff(self, bbox: tuple, frame: np.ndarray) -> bool:
        """Classify whether a person is staff (dummy: random with staff_ratio).

        In a real detector this would crop the bbox region and run a
        uniform/appearance classifier.
        """
        if self._rng is None:
            self.initialize({})
        return bool(self._rng.random() < _STAFF_RATIO)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #
    def _generate_bbox(
        self, rng: np.random.Generator, idx: int, total: int
    ) -> tuple:
        """Generate a realistic bounding box based on camera type.

        For overhead cameras: people appear roughly circular, spread across frame.
        For angled cameras: people are taller rectangles, clustered near bottom.
        """
        if self._camera_type == "overhead":
            # Spread people across the frame in a grid-ish pattern
            cols = max(1, int(math.ceil(math.sqrt(total))))
            row, col = divmod(idx, cols)
            cell_w = 1.0 / cols
            cell_h = 1.0 / max(1, int(math.ceil(total / cols)))

            cx = (col + 0.5) * cell_w + rng.normal(0, 0.03)
            cy = (row + 0.5) * cell_h + rng.normal(0, 0.03)

            # People from above: roughly square bbox
            w = rng.uniform(0.04, 0.10)
            h = rng.uniform(0.04, 0.10)
        else:
            # Angled camera: people appear as tall rectangles
            cx = rng.uniform(0.08, 0.92)
            cy = rng.uniform(0.35, 0.90)  # mostly lower half

            w = rng.uniform(0.05, 0.12)
            h = rng.uniform(0.15, 0.40)
            # People further from camera (higher up) are smaller
            scale = 0.4 + 0.6 * cy  # larger when closer to bottom
            w *= scale
            h *= scale

        x1 = float(np.clip(cx - w / 2, 0.0, 1.0))
        y1 = float(np.clip(cy - h / 2, 0.0, 1.0))
        x2 = float(np.clip(cx + w / 2, 0.0, 1.0))
        y2 = float(np.clip(cy + h / 2, 0.0, 1.0))

        return (x1, y1, x2, y2)

    def _generate_confidence(self, rng: np.random.Generator) -> float:
        """Generate a realistic confidence score.

        Distribution is skewed toward high confidence (most detections
        are confident; a few near the boundary are lower).
        """
        # Beta distribution α=5, β=1.5 gives a nice right-skew
        raw = rng.beta(5.0, 1.5)
        return float(_CONF_MIN + raw * (_CONF_MAX - _CONF_MIN))
    
    def _generate_synthetic_frame(
        self, width: int = 1920, height: int = 1080
    ) -> np.ndarray:
        """Generate a synthetic video frame for testing without real video.

        Creates a frame with a store-like background pattern.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            BGR image as numpy array.
        """
        if self._rng is None:
            self.initialize({})

        # Base color: warm store lighting
        frame = np.full((height, width, 3), fill_value=180, dtype=np.uint8)

        # Add some variation
        noise = self._rng.integers(0, 20, size=(height, width, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(
            np.uint8
        )

        # Draw grid lines to simulate shelving
        for y in range(0, height, height // 6):
            frame[max(0, y - 1) : y + 1, :] = [120, 120, 130]
        for x in range(0, width, width // 8):
            frame[:, max(0, x - 1) : x + 1] = [120, 120, 130]

        return frame
