"""
detector_base.py — Abstract base class for all person detectors.

This module defines the contract that any detector (dummy, YOLOv8, RT-DETR, etc.)
must implement. The pipeline is detector-agnostic: swap implementations by
subclassing DetectorBase and registering in detect.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    """A single person detection in a video frame.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in normalized [0, 1] coordinates.
        confidence: Detection confidence score in [0, 1].
        is_staff: Whether the detected person is identified as store staff.
        track_id: Optional tracker-assigned ID (populated by the tracker, not detector).
        features: Optional Re-ID feature vector (128-d float32) for re-identification.
    """
    bbox: tuple  # (x1, y1, x2, y2) normalized 0-1
    confidence: float
    is_staff: bool
    track_id: Optional[int] = None
    features: Optional[np.ndarray] = None  # Re-ID feature vector (128-d)

    @property
    def centroid(self) -> tuple:
        """Compute the centroid of the bounding box.

        Returns:
            Tuple (cx, cy) in normalized coordinates.
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        """Compute the area of the bounding box in normalized coordinates."""
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def iou(self, other: "Detection") -> float:
        """Compute Intersection over Union with another Detection."""
        x1 = max(self.bbox[0], other.bbox[0])
        y1 = max(self.bbox[1], other.bbox[1])
        x2 = min(self.bbox[2], other.bbox[2])
        y2 = min(self.bbox[3], other.bbox[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class DetectorBase(ABC):
    """Abstract base class for person detectors.

    To implement a new detector:
      1. Subclass DetectorBase
      2. Implement detect() and classify_staff()
      3. Optionally override initialize() for model loading
      4. Register the class in detect.py's DETECTOR_REGISTRY

    Example:
        class MyYOLODetector(DetectorBase):
            def initialize(self, config):
                self.model = YOLO(config['weights_path'])

            def detect(self, frame):
                results = self.model(frame)
                return [Detection(...) for r in results]

            def classify_staff(self, bbox, frame):
                # Use uniform classifier or secondary model
                return self.uniform_model.predict(crop)
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect people in a single video frame.

        Args:
            frame: BGR image as numpy array, shape (H, W, 3), dtype uint8.

        Returns:
            List of Detection objects found in the frame.
        """
        pass

    @abstractmethod
    def classify_staff(self, bbox: tuple, frame: np.ndarray) -> bool:
        """Determine if a detected person is store staff.

        Args:
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates.
            frame: The full video frame (for cropping and classification).

        Returns:
            True if the person is classified as staff, False otherwise.
        """
        pass

    def initialize(self, config: dict) -> None:
        """Optional initialization hook for loading models, weights, etc.

        Args:
            config: Dictionary of configuration parameters.
        """
        pass
