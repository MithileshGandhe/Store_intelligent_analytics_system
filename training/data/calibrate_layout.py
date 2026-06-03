#!/usr/bin/env python3
"""
calibrate_layout.py — Interactive store layout calibration tool.

Loads a store's layout.png image and lets you define zone polygons
interactively using matplotlib. Outputs a store_layout.json file
compatible with the pipeline's tracker.

NOTE: This script requires a GUI (matplotlib TkAgg backend).
      Run locally, NOT on Google Colab.

Usage:
    python calibrate_layout.py --store store1
    python calibrate_layout.py --store store2 --raw-dir training/data/raw
"""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ZONE_TYPES = ["ENTRY", "BILLING", "SHELF", "DISPLAY", "PROMO", "TRANSITION"]


def load_reference_frame(video_path: str) -> np.ndarray:
    """Extract the first frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None


def interactive_zone_editor(layout_img: np.ndarray, store_id: str, cameras: dict) -> dict:
    """Launch an interactive matplotlib editor for defining zones.

    Returns a store_layout dict.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    layout_rgb = cv2.cvtColor(layout_img, cv2.COLOR_BGR2RGB)
    h, w = layout_img.shape[:2]

    zones = []
    current_polygon = []

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(layout_rgb)
    ax.set_title(f"Store: {store_id} — Click to define zone polygons\n"
                 "Left-click: add point | Right-click: finish polygon | 'q': done")

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # Left click — add point
            current_polygon.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, "ro", markersize=5)
            if len(current_polygon) > 1:
                pts = current_polygon[-2:]
                ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], "r-", linewidth=1)
            fig.canvas.draw()

        elif event.button == 3:  # Right click — finish polygon
            if len(current_polygon) < 3:
                print("Need at least 3 points for a polygon.")
                return

            # Close the polygon visually
            pts = current_polygon + [current_polygon[0]]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "g-", linewidth=2)
            ax.fill([p[0] for p in current_polygon], [p[1] for p in current_polygon],
                    alpha=0.2, color="green")

            # Prompt for zone info
            zone_num = len(zones) + 1
            print(f"\n--- Zone {zone_num} ({len(current_polygon)} points) ---")
            zone_name = input("  Zone name (e.g., 'Skincare Section'): ").strip() or f"Zone_{zone_num}"
            zone_id = input(f"  Zone ID (e.g., 'SKINCARE') [{zone_name.upper().replace(' ', '_')}]: ").strip()
            if not zone_id:
                zone_id = zone_name.upper().replace(" ", "_")

            print(f"  Zone types: {', '.join(ZONE_TYPES)}")
            zone_type = input(f"  Zone type [SHELF]: ").strip().upper() or "SHELF"

            sku_zone = None
            if zone_type in ("SHELF", "DISPLAY", "PROMO"):
                sku_zone = input("  SKU zone (e.g., 'LIPSTICK') [None]: ").strip() or None

            is_revenue = zone_type in ("SHELF", "DISPLAY", "PROMO", "BILLING")

            # Normalize polygon to [0, 1]
            normalized = [[round(x / w, 4), round(y / h, 4)] for x, y in current_polygon]

            zone_def = {
                "zone_id": zone_id,
                "zone_name": zone_name,
                "polygon": normalized,
                "zone_type": zone_type.lower(),
            }
            if sku_zone:
                zone_def["sku_zone"] = sku_zone
            if is_revenue:
                zone_def["is_revenue_zone"] = "Yes"

            zones.append(zone_def)
            ax.text(np.mean([p[0] for p in current_polygon]),
                    np.mean([p[1] for p in current_polygon]),
                    zone_id, fontsize=8, ha="center", color="white",
                    bbox=dict(boxstyle="round", facecolor="green", alpha=0.7))
            fig.canvas.draw()

            current_polygon.clear()
            print(f"  Zone '{zone_id}' saved. Click to start next zone or press 'q' to finish.")

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    # Entry line threshold
    print("\n--- Entry Line Configuration ---")
    y_thresh_str = input("  Entry line y_threshold (0-1, fraction from top) [0.15]: ").strip()
    y_threshold = float(y_thresh_str) if y_thresh_str else 0.15

    # Build camera definitions
    camera_defs = []
    for cam_name, cam_cfg in cameras.items():
        cam_def = {
            "camera_id": cam_name,
            "type": cam_cfg.get("type", "angled"),
            "covers_zones": cam_cfg.get("covers_zones", []),
        }
        if not cam_def["covers_zones"]:
            print(f"\n  Camera '{cam_name}' ({cam_cfg.get('role', 'zone')}):")
            zone_ids = [z["zone_id"] for z in zones]
            print(f"  Available zones: {', '.join(zone_ids)}")
            covers = input("  Covers zones (comma-separated): ").strip()
            cam_def["covers_zones"] = [z.strip() for z in covers.split(",") if z.strip()]
        camera_defs.append(cam_def)

    layout = {
        "store_id": store_id,
        "zones": zones,
        "cameras": camera_defs,
        "entry_line": {
            "y_threshold": y_threshold,
            "direction": "top_to_bottom_is_entry",
        },
    }

    return layout


def main():
    parser = argparse.ArgumentParser(description="Interactive store layout calibration")
    parser.add_argument("--store", required=True, help="Store name (e.g., store1)")
    parser.add_argument("--raw-dir", default="training/data/raw", help="Raw data directory")
    parser.add_argument("--output-dir", default="training/data/layouts", help="Output directory for layouts")
    parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"stores": {}}

    store_cfg = config.get("stores", {}).get(args.store, {})
    store_id = store_cfg.get("store_id", args.store.upper())
    cameras = store_cfg.get("cameras", {})

    raw_dir = Path(args.raw_dir) / args.store
    layout_path = raw_dir / store_cfg.get("layout_image", "layout.png")

    if not layout_path.exists():
        logger.error("Layout image not found: %s", layout_path)
        return

    layout_img = cv2.imread(str(layout_path))
    if layout_img is None:
        logger.error("Cannot read layout image: %s", layout_path)
        return

    logger.info("Layout image loaded: %s (%dx%d)", layout_path, layout_img.shape[1], layout_img.shape[0])

    # Run interactive editor
    layout = interactive_zone_editor(layout_img, store_id, cameras)

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.store}_layout.json")
    with open(output_path, "w") as f:
        json.dump(layout, f, indent=2)

    print(f"\nLayout saved to: {output_path}")
    print(f"  Zones: {len(layout['zones'])}")
    print(f"  Cameras: {len(layout['cameras'])}")


if __name__ == "__main__":
    main()
