"""
yolo_detector.py — Production YOLOv8 detector implementing the DetectorBase interface.

This module provides a real person detector using:
  - YOLOv8 (pre-trained, COCO) for person detection
  - MobileNetV3-Small for staff/customer classification (optional)
  - OSNet-x0.25 for 128-d Re-ID embeddings (optional)
  - Simple IoU-based frame-to-frame tracking

Works standalone (no pipeline imports required). Compatible with the
existing DETECTOR_REGISTRY — add {"yolo": YOLODetector} to detect.py.

Usage:
    detector = YOLODetector()
    detector.initialize({
        "weights_path": "yolov8s.pt",
        "staff_model_path": "staff_classifier.pth",   # optional
        "reid_model_path": "reid_osnet.pth",           # optional
        "device": "cuda",
    })
    detections = detector.detect(frame)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Standalone copies of Detection & DetectorBase (works without pipeline pkg)
# ---------------------------------------------------------------------------
try:
    from pipeline.detector_base import Detection, DetectorBase
except ImportError:

    @dataclass
    class Detection:
        """A single person detection in a video frame."""
        bbox: tuple          # (x1, y1, x2, y2) normalized 0-1
        confidence: float
        is_staff: bool
        track_id: Optional[int] = None
        features: Optional[np.ndarray] = None  # Re-ID 128-d

        @property
        def centroid(self) -> tuple:
            x1, y1, x2, y2 = self.bbox
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        @property
        def area(self) -> float:
            x1, y1, x2, y2 = self.bbox
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        def iou(self, other: "Detection") -> float:
            x1 = max(self.bbox[0], other.bbox[0])
            y1 = max(self.bbox[1], other.bbox[1])
            x2 = min(self.bbox[2], other.bbox[2])
            y2 = min(self.bbox[3], other.bbox[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            union = self.area + other.area - inter
            return inter / union if union > 0 else 0.0

    class DetectorBase(ABC):
        """Abstract base class for person detectors."""

        @abstractmethod
        def detect(self, frame: np.ndarray) -> List[Detection]:
            pass

        @abstractmethod
        def classify_staff(self, bbox: tuple, frame: np.ndarray) -> bool:
            pass

        def initialize(self, config: dict) -> None:
            pass


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _box_iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# OSNet Architecture for Re-ID
# ---------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid()
        )
    def forward(self, x): return x * self.gate(x).unsqueeze(-1).unsqueeze(-1)

class LightBottleneck(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        mid = out_c // 4
        self.conv1 = ConvBNReLU(in_c, mid, 1, 1, 0)
        self.conv2 = ConvBNReLU(mid, mid, 3, 1, 1, groups=mid)
        self.conv3 = nn.Sequential(nn.Conv2d(mid, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        self.gate = ChannelGate(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        out = self.gate(out)
        return self.relu(out + self.shortcut(x))

class OSNet025(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 16, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = nn.Sequential(LightBottleneck(16, 64), LightBottleneck(64, 64))
        self.pool1 = nn.Sequential(ConvBNReLU(64, 64, 1, 1, 0), nn.AvgPool2d(2, 2))
        self.layer2 = nn.Sequential(LightBottleneck(64, 96), LightBottleneck(96, 96))
        self.pool2 = nn.Sequential(ConvBNReLU(96, 96, 1, 1, 0), nn.AvgPool2d(2, 2))
        self.layer3 = nn.Sequential(LightBottleneck(96, 128), LightBottleneck(128, 128))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.layer3(x)
        x = self.gap(x).flatten(1)
        x = self.bn(self.fc(x))
        return F.normalize(x, p=2, dim=1)

# ---------------------------------------------------------------------------
# YOLODetector
# ---------------------------------------------------------------------------

class YOLODetector(DetectorBase):
    """Person detector using pre-trained YOLOv8 + optional staff/Re-ID models.

    Config keys:
        weights_path (str): Path to YOLOv8 weights (default: "yolov8s.pt")
        staff_model_path (str): Path to staff classifier .pth (optional)
        reid_model_path (str): Path to Re-ID model .pth (optional)
        conf_threshold (float): Detection confidence threshold (default: 0.35)
        iou_threshold (float): NMS IoU threshold (default: 0.45)
        device (str): "auto" | "cpu" | "cuda" | "cuda:0"
    """

    def __init__(self):
        self._yolo = None
        self._staff_model = None
        self._reid_model = None
        self._device = "cpu"
        self._conf_threshold = 0.35
        self._iou_threshold = 0.45

        # Simple IoU tracker state
        self._tracks: Dict[int, Tuple[float, ...]] = {}  # track_id -> last bbox
        self._next_track_id: int = 0
        self._track_iou_threshold = 0.3

        # Image preprocessing (lazily initialized)
        self._staff_transform = None
        self._reid_transform = None

    def initialize(self, config: dict) -> None:
        """Load models from config."""
        import torch

        device_str = config.get("device", "auto")
        if device_str == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device_str

        self._conf_threshold = config.get("conf_threshold", 0.35)
        self._iou_threshold = config.get("iou_threshold", 0.45)

        # --- YOLO ---
        from ultralytics import YOLO
        weights = config.get("weights_path", "yolov8s.pt")
        self._yolo = YOLO(weights)
        print(f"[YOLODetector] YOLO loaded: {weights} on {self._device}")

        # --- Staff classifier (optional) ---
        staff_path = config.get("staff_model_path")
        if staff_path and os.path.exists(staff_path):
            self._load_staff_model(staff_path)
        else:
            print("[YOLODetector] Staff classifier not loaded (is_staff defaults to False)")

        # --- Re-ID model (optional) ---
        reid_path = config.get("reid_model_path")
        if reid_path and os.path.exists(reid_path):
            self._load_reid_model(reid_path)
        else:
            print("[YOLODetector] Re-ID model not loaded (features will be None)")

    def _load_staff_model(self, path: str):
        """Load MobileNetV3-Small staff classifier."""
        import torch
        import timm
        from torchvision import transforms

        self._staff_model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=2)
        state = torch.load(path, map_location=self._device)
        self._staff_model.load_state_dict(state if isinstance(state, dict) and "model_state_dict" not in state
                                          else state.get("model_state_dict", state))
        self._staff_model.to(self._device).eval()

        self._staff_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print(f"[YOLODetector] Staff classifier loaded: {path}")

    def _load_reid_model(self, path: str):
        """Load Re-ID embedding model."""
        import torch
        from torchvision import transforms

        self._reid_model = OSNet025(embed_dim=128)
        state = torch.load(path, map_location=self._device)
        self._reid_model.load_state_dict(state if isinstance(state, dict) and "model_state_dict" not in state
                                          else state.get("model_state_dict", state))
        self._reid_model.to(self._device).eval()

        self._reid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print(f"[YOLODetector] Re-ID model loaded: {path}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect persons in a video frame.

        Args:
            frame: BGR image (H, W, 3), dtype uint8.

        Returns:
            List of Detection objects.
        """
        if self._yolo is None:
            raise RuntimeError("YOLODetector not initialized. Call initialize() first.")

        h, w = frame.shape[:2]
        results = self._yolo(
            frame, verbose=False,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
        )

        raw_detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:  # person class only
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                # Normalize to [0, 1]
                bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)
                raw_detections.append((bbox_norm, conf, (int(x1), int(y1), int(x2), int(y2))))

        # Assign track IDs via IoU matching
        new_tracks = {}
        matched_track_ids = set()
        detections: List[Detection] = []

        for bbox_norm, conf, bbox_px in raw_detections:
            # Staff classification
            is_staff = self._classify_crop(frame, bbox_px, "staff")

            # Re-ID features
            features = self._extract_reid(frame, bbox_px)

            # Track matching
            track_id = self._match_track(bbox_norm, matched_track_ids)
            new_tracks[track_id] = bbox_norm
            matched_track_ids.add(track_id)

            detections.append(Detection(
                bbox=bbox_norm,
                confidence=conf,
                is_staff=is_staff,
                track_id=track_id,
                features=features,
            ))

        self._tracks = new_tracks
        return detections

    def _match_track(self, bbox: tuple, already_matched: set) -> int:
        """Match a detection to an existing track by IoU."""
        best_iou = 0.0
        best_tid = None

        for tid, prev_bbox in self._tracks.items():
            if tid in already_matched:
                continue
            iou = _box_iou(bbox, prev_bbox)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid

        if best_tid is not None and best_iou >= self._track_iou_threshold:
            return best_tid

        # New track
        tid = self._next_track_id
        self._next_track_id += 1
        return tid

    def _classify_crop(self, frame: np.ndarray, bbox_px: tuple, model_type: str) -> bool:
        """Classify a person crop as staff or customer."""
        if self._staff_model is None:
            return False

        import torch
        x1, y1, x2, y2 = bbox_px
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return False

        crop = frame[y1:y2, x1:x2]
        crop_rgb = crop[:, :, ::-1].copy()  # BGR -> RGB
        tensor = self._staff_transform(crop_rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._staff_model(tensor)
            pred = torch.argmax(logits, dim=1).item()

        return pred == 1  # 0=customer, 1=staff

    def _extract_reid(self, frame: np.ndarray, bbox_px: tuple) -> Optional[np.ndarray]:
        """Extract 128-d Re-ID feature vector from a person crop."""
        if self._reid_model is None:
            return None

        import torch
        x1, y1, x2, y2 = bbox_px
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_rgb = crop[:, :, ::-1].copy()
        tensor = self._reid_transform(crop_rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._reid_model(tensor)
            if isinstance(embedding, tuple):
                embedding = embedding[0]
            embedding = embedding.cpu().numpy().flatten()
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm

        return embedding.astype(np.float32)

    def classify_staff(self, bbox: tuple, frame: np.ndarray) -> bool:
        """Standalone staff classification."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bbox_px = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        return self._classify_crop(frame, bbox_px, "staff")


import os  # noqa: E402 — needed for os.path.exists in initialize()
