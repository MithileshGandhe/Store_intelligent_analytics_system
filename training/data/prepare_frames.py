#!/usr/bin/env python3
"""
prepare_frames.py — Extract frames from store CCTV footage.

Reads video files from training/data/raw/{store}/ and saves individual frames
at a configurable FPS. Skips near-duplicate frames to save storage.

Usage:
    python prepare_frames.py --config training/config.yaml
    python prepare_frames.py --fps 2.0 --store store1
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_frame_similarity(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute similarity between two frames using mean absolute difference."""
    if frame_a is None or frame_b is None:
        return 0.0
    if frame_a.shape != frame_b.shape:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(gray_a - gray_b).mean()
    return 1.0 - (diff / 255.0)


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    store_id: str,
    camera_id: str,
    target_fps: float = 1.0,
    skip_duplicates: bool = True,
    similarity_threshold: float = 0.95,
    quality: int = 95,
) -> list:
    """Extract frames from a single video file.

    Returns list of dicts with frame metadata for the manifest.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(video_fps / target_fps)))

    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        "Processing %s — %d total frames, extracting every %d (target %.1f FPS)",
        video_path, total_frames, frame_interval, target_fps,
    )

    manifest_entries = []
    prev_frame = None
    saved_count = 0
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=f"  {camera_id}", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if skip_duplicates and prev_frame is not None:
                sim = compute_frame_similarity(frame, prev_frame)
                if sim >= similarity_threshold:
                    frame_idx += 1
                    pbar.update(1)
                    continue

            timestamp_s = frame_idx / video_fps
            fname = f"frame_{saved_count:06d}.jpg"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            manifest_entries.append({
                "frame_path": fpath,
                "store_id": store_id,
                "camera_id": camera_id,
                "timestamp_s": round(timestamp_s, 3),
                "frame_idx": saved_count,
            })
            prev_frame = frame.copy()
            saved_count += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    logger.info("  Saved %d frames to %s", saved_count, output_dir)
    return manifest_entries


def main():
    parser = argparse.ArgumentParser(description="Extract frames from store CCTV footage")
    parser.add_argument("--config", default="training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--raw-dir", default="training/data/raw", help="Root directory with raw footage")
    parser.add_argument("--output-dir", default="training/data/frames", help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=None, help="Override extraction FPS (default from config)")
    parser.add_argument("--store", default=None, help="Process only this store (e.g., store1)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Config not found at %s, using defaults", config_path)
        config = {"frame_extraction": {"fps": 1.0, "skip_duplicates": True, "similarity_threshold": 0.95, "quality": 95}}

    fe_config = config.get("frame_extraction", {})
    target_fps = args.fps or fe_config.get("fps", 1.0)
    skip_dupes = fe_config.get("skip_duplicates", True)
    sim_thresh = fe_config.get("similarity_threshold", 0.95)
    quality = fe_config.get("quality", 95)

    stores = config.get("stores", {})
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    all_manifest = []

    for store_name, store_cfg in stores.items():
        if args.store and store_name != args.store:
            continue

        store_id = store_cfg.get("store_id", store_name)
        cameras = store_cfg.get("cameras", {})
        logger.info("=== Processing store: %s (%s) ===", store_name, store_id)

        for cam_name, cam_cfg in cameras.items():
            video_file = cam_cfg.get("file", f"{cam_name}.mp4")
            video_path = raw_dir / store_name / video_file

            if not video_path.exists():
                logger.warning("Video not found: %s — skipping", video_path)
                continue

            cam_output = output_dir / store_id / cam_name
            entries = extract_frames_from_video(
                str(video_path), str(cam_output),
                store_id, cam_name,
                target_fps, skip_dupes, sim_thresh, quality,
            )
            all_manifest.extend(entries)

    # Write manifest CSV
    if all_manifest:
        manifest_path = output_dir / "manifest.csv"
        os.makedirs(output_dir, exist_ok=True)
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_path", "store_id", "camera_id", "timestamp_s", "frame_idx"])
            writer.writeheader()
            writer.writerows(all_manifest)
        logger.info("Manifest written to %s (%d entries)", manifest_path, len(all_manifest))

    # Summary
    print("\n" + "=" * 60)
    print("Frame Extraction Summary")
    print("=" * 60)
    print(f"  Total frames extracted: {len(all_manifest)}")
    stores_seen = set(e["store_id"] for e in all_manifest)
    cams_seen = set(e["camera_id"] for e in all_manifest)
    print(f"  Stores: {len(stores_seen)} — {', '.join(sorted(stores_seen))}")
    print(f"  Cameras: {len(cams_seen)} — {', '.join(sorted(cams_seen))}")
    print(f"  Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
