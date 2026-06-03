#!/usr/bin/env python3
"""
build_staff_dataset.py — Build staff-vs-customer classification dataset.

Detects persons in extracted frames using pre-trained YOLOv8n, crops them,
and provides interactive labeling (matplotlib) or auto-labeling from CSV.
Outputs an ImageFolder structure for PyTorch training.

Usage:
    python build_staff_dataset.py --frames-dir training/data/frames
    python build_staff_dataset.py --auto-label labels.csv
"""

import argparse
import csv
import logging
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def detect_and_crop_persons(frames_dir: str, crops_dir: str, yolo_model: str = "yolov8n.pt",
                            max_crops: int = 500, conf_threshold: float = 0.35) -> list:
    """Run YOLOv8 on extracted frames and save person crops.

    Returns list of crop file paths.
    """
    from ultralytics import YOLO
    model = YOLO(yolo_model)
    os.makedirs(crops_dir, exist_ok=True)

    frame_files = []
    for root, _, files in os.walk(frames_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                frame_files.append(os.path.join(root, f))

    if not frame_files:
        logger.error("No frames found in %s", frames_dir)
        return []

    logger.info("Found %d frames. Running YOLOv8 detection...", len(frame_files))
    crop_paths = []
    crop_idx = 0

    for fpath in tqdm(frame_files, desc="Detecting persons"):
        if crop_idx >= max_crops:
            break

        frame = cv2.imread(fpath)
        if frame is None:
            continue

        results = model(frame, verbose=False, conf=conf_threshold)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:  # person class
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Ensure valid crop
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) < 20 or (y2 - y1) < 30:
                    continue

                crop = frame[y1:y2, x1:x2]
                crop_name = f"crop_{crop_idx:05d}.jpg"
                crop_path = os.path.join(crops_dir, crop_name)
                cv2.imwrite(crop_path, crop)
                crop_paths.append(crop_path)
                crop_idx += 1

                if crop_idx >= max_crops:
                    break

    logger.info("Saved %d person crops to %s", len(crop_paths), crops_dir)
    return crop_paths


def interactive_label(crop_paths: list) -> dict:
    """Show crops one-by-one and ask user to label as staff or customer.

    Returns dict mapping crop_path -> label ('staff' or 'customer').
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    labels = {}
    print("\n" + "=" * 60)
    print("Interactive Labeling")
    print("Press 'z' for STAFF, 'c' for CUSTOMER, 'q' to QUIT")
    print("=" * 60)

    for i, cpath in enumerate(crop_paths):
        img = cv2.imread(cpath)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        ax.imshow(img_rgb)
        ax.set_title(f"[{i+1}/{len(crop_paths)}] Press 's'=staff, 'c'=customer, 'q'=quit")
        ax.axis("off")

        label = [None]

        def on_key(event):
            if event.key == "z":
                label[0] = "staff"
                plt.close()
            elif event.key == "c":
                label[0] = "customer"
                plt.close()
            elif event.key == "q":
                label[0] = "QUIT"
                plt.close()

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

        if label[0] == "QUIT":
            break
        if label[0] is not None:
            labels[cpath] = label[0]

    return labels


def auto_label_from_csv(crop_paths: list, csv_path: str) -> dict:
    """Load labels from a CSV file (crop_path,label)."""
    labels = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("crop_path", row.get("path", ""))
            l = row.get("label", row.get("class", ""))
            if p and l:
                labels[p] = l.strip().lower()
    # Match to crop_paths by basename
    basename_map = {os.path.basename(p): p for p in crop_paths}
    resolved = {}
    for p, l in labels.items():
        bn = os.path.basename(p)
        if bn in basename_map:
            resolved[basename_map[bn]] = l
    return resolved


def organize_dataset(labels: dict, output_dir: str, val_ratio: float = 0.2, seed: int = 42):
    """Organize labeled crops into ImageFolder structure.

    Output: output_dir/{train,val}/{staff,customer}/
    """
    random.seed(seed)
    items = list(labels.items())
    random.shuffle(items)

    split_idx = int(len(items) * (1 - val_ratio))
    train_items = items[:split_idx]
    val_items = items[split_idx:]

    for split_name, split_items in [("train", train_items), ("val", val_items)]:
        for cls_name in ["staff", "customer"]:
            os.makedirs(os.path.join(output_dir, split_name, cls_name), exist_ok=True)

        for cpath, label in split_items:
            if label not in ("staff", "customer"):
                continue
            dest_dir = os.path.join(output_dir, split_name, label)
            dest = os.path.join(dest_dir, os.path.basename(cpath))
            shutil.copy2(cpath, dest)

    train_staff = len([l for _, l in train_items if l == "staff"])
    train_cust = len([l for _, l in train_items if l == "customer"])
    val_staff = len([l for _, l in val_items if l == "staff"])
    val_cust = len([l for _, l in val_items if l == "customer"])

    print(f"\nDataset organized at: {output_dir}")
    print(f"  Train: {train_staff} staff, {train_cust} customer ({train_staff + train_cust} total)")
    print(f"  Val:   {val_staff} staff, {val_cust} customer ({val_staff + val_cust} total)")


def main():
    parser = argparse.ArgumentParser(description="Build staff-vs-customer classification dataset")
    parser.add_argument("--frames-dir", default="training/data/frames", help="Directory with extracted frames")
    parser.add_argument("--output-dir", default="training/data/staff_dataset", help="Output dataset directory")
    parser.add_argument("--crops-dir", default="training/data/staff_crops", help="Temporary crops directory")
    parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLOv8 model for detection")
    parser.add_argument("--max-crops", type=int, default=500, help="Maximum number of person crops")
    parser.add_argument("--auto-label", default=None, help="CSV file for automatic labeling")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    # Step 1: Detect and crop persons
    logger.info("Step 1: Detecting persons in frames...")
    crop_paths = detect_and_crop_persons(
        args.frames_dir, args.crops_dir,
        args.yolo_model, args.max_crops,
    )

    if not crop_paths:
        logger.error("No crops generated. Check frames directory.")
        return

    # Step 2: Label crops
    if args.auto_label:
        logger.info("Step 2: Auto-labeling from %s...", args.auto_label)
        labels = auto_label_from_csv(crop_paths, args.auto_label)
    else:
        logger.info("Step 2: Interactive labeling (%d crops)...", len(crop_paths))
        labels = interactive_label(crop_paths)

    if not labels:
        logger.error("No labels collected. Exiting.")
        return

    logger.info("Collected %d labels", len(labels))

    # Step 3: Organize into ImageFolder
    logger.info("Step 3: Organizing dataset...")
    organize_dataset(labels, args.output_dir, args.val_ratio)

    print("\n" + "=" * 60)
    print("Staff Dataset Build Complete")
    print("=" * 60)
    print(f"  Crops detected: {len(crop_paths)}")
    print(f"  Labels collected: {len(labels)}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
