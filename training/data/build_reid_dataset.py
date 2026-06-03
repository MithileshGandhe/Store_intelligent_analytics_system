#!/usr/bin/env python3
"""
build_reid_dataset.py — Build person Re-ID dataset using IoU-based tracking.

Detects persons across consecutive frames, tracks them via IoU overlap,
and groups crops by identity for Re-ID model training.

Usage:
    python build_reid_dataset.py --frames-dir training/data/frames
"""

import argparse
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleIoUTracker:
    """Frame-to-frame IoU-based person tracker."""

    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = {}       # track_id -> {"bbox": [...], "lost": int, "crops": [...]}
        self.next_id = 0
        self.finalized = []    # completed tracks

    def update(self, detections, frame):
        """Update tracks with new detections.

        Args:
            detections: list of [x1, y1, x2, y2, conf]
            frame: current BGR frame

        Returns:
            dict of track_id -> bbox for active tracks
        """
        h, w = frame.shape[:2]
        matched_tracks = set()
        matched_dets = set()

        # Match detections to existing tracks by IoU
        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_tid = None
            for tid, track in self.tracks.items():
                iou = compute_iou(det[:4], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_iou >= self.iou_threshold and best_tid is not None:
                self.tracks[best_tid]["bbox"] = det[:4]
                self.tracks[best_tid]["lost"] = 0

                # Save crop
                x1, y1, x2, y2 = map(int, det[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) > 10 and (y2 - y1) > 15:
                    crop = frame[y1:y2, x1:x2].copy()
                    self.tracks[best_tid]["crops"].append(crop)

                matched_tracks.add(best_tid)
                matched_dets.add(det_idx)

        # Start new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_dets:
                continue
            x1, y1, x2, y2 = map(int, det[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) < 10 or (y2 - y1) < 15:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            self.tracks[self.next_id] = {
                "bbox": det[:4],
                "lost": 0,
                "crops": [crop],
            }
            self.next_id += 1

        # Age out lost tracks
        to_remove = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track["lost"] += 1
                if track["lost"] > self.max_lost:
                    to_remove.append(tid)

        for tid in to_remove:
            self.finalized.append(self.tracks.pop(tid))

    def finalize_all(self):
        """Finalize all remaining active tracks."""
        for tid in list(self.tracks.keys()):
            self.finalized.append(self.tracks.pop(tid))


def process_camera_frames(frame_dir: str, yolo_model, conf_threshold: float = 0.35) -> list:
    """Process all frames from a camera directory and return tracked identities."""
    frame_files = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if not frame_files:
        return []

    tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost=5)

    for fpath in frame_files:
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        results = yolo_model(frame, verbose=False, conf=conf_threshold)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])

        tracker.update(detections, frame)

    tracker.finalize_all()
    return tracker.finalized


def main():
    parser = argparse.ArgumentParser(description="Build Re-ID dataset from tracked persons")
    parser.add_argument("--frames-dir", default="training/data/frames", help="Directory with extracted frames")
    parser.add_argument("--output-dir", default="training/data/reid_dataset", help="Output Re-ID dataset directory")
    parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLOv8 model for detection")
    parser.add_argument("--min-track-length", type=int, default=3, help="Minimum crops per identity")
    parser.add_argument("--max-crops-per-id", type=int, default=20, help="Maximum crops per identity")
    parser.add_argument("--conf-threshold", type=float, default=0.35, help="Detection confidence threshold")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.yolo_model)

    frames_root = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Discover camera directories
    camera_dirs = []
    for store_dir in sorted(frames_root.iterdir()):
        if not store_dir.is_dir():
            continue
        for cam_dir in sorted(store_dir.iterdir()):
            if cam_dir.is_dir():
                camera_dirs.append((store_dir.name, cam_dir.name, str(cam_dir)))

    if not camera_dirs:
        logger.error("No camera frame directories found in %s", frames_root)
        return

    logger.info("Found %d camera directories", len(camera_dirs))

    identity_idx = 0
    total_tracks = 0
    total_saved = 0

    for store_id, cam_id, cam_path in camera_dirs:
        logger.info("Processing %s/%s...", store_id, cam_id)
        tracks = process_camera_frames(cam_path, model, args.conf_threshold)
        total_tracks += len(tracks)

        for track in tracks:
            crops = track["crops"]
            if len(crops) < args.min_track_length:
                continue

            # Sample evenly if too many
            if len(crops) > args.max_crops_per_id:
                indices = np.linspace(0, len(crops) - 1, args.max_crops_per_id, dtype=int)
                crops = [crops[i] for i in indices]

            # Save crops
            id_dir = output_dir / f"identity_{identity_idx:04d}"
            os.makedirs(id_dir, exist_ok=True)

            for ci, crop in enumerate(crops):
                crop_path = id_dir / f"crop_{ci:03d}.jpg"
                cv2.imwrite(str(crop_path), crop)

            identity_idx += 1
            total_saved += len(crops)

    print("\n" + "=" * 60)
    print("Re-ID Dataset Build Complete")
    print("=" * 60)
    print(f"  Camera dirs processed: {len(camera_dirs)}")
    print(f"  Total tracks detected: {total_tracks}")
    print(f"  Identities saved (>= {args.min_track_length} crops): {identity_idx}")
    print(f"  Total crops saved: {total_saved}")
    print(f"  Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
