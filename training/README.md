# 🧠 Store Intelligence — Training Pipeline

A self-contained training folder for building ML models from raw CCTV footage.
**Does NOT modify the main pipeline code** — trained models plug in via the `DETECTOR_REGISTRY`.

---

## 📋 Overview

| Component | Purpose | Runs On |
|-----------|---------|---------|
| `prepare_frames.py` | Extract frames from video clips | Local (CPU) |
| `build_staff_dataset.py` | Build staff/customer classification dataset | Local (CPU) |
| `build_reid_dataset.py` | Build person Re-ID identity dataset | Local (CPU) |
| `calibrate_layout.py` | Interactive zone polygon editor | Local (GUI) |
| `01_detection_and_tracking.ipynb` | Demo YOLOv8 detection + tracking | Colab |
| `02_staff_classifier.ipynb` | Train staff/customer classifier | Colab (GPU) |
| `03_reid_embedding.ipynb` | Train Re-ID embedding model | Colab (GPU) |
| `04_full_pipeline.ipynb` | End-to-end pipeline integration | Colab (GPU) |

---

## 🚀 Quick Start

### Step 1: Place Your Footage

Copy your store footage into `training/data/raw/`:

```
training/data/raw/
├── store1/
│   ├── cam1_zone.mp4
│   ├── cam2_zone.mp4
│   ├── cam3_entry.mp4
│   ├── cam4_billing.mp4
│   └── layout.png
└── store2/
    ├── cam1_zone.mp4
    ├── cam2_entry1.mp4
    ├── cam3_entry2.mp4
    ├── cam4_billing.mp4
    └── layout.png
```

### Step 2: Extract Frames (Local)

```bash
cd purplle_hackathon
python training/data/prepare_frames.py --config training/config.yaml
```

This extracts frames at 1 FPS, skips duplicates, and creates a manifest CSV.
Output goes to `training/data/frames/{store_id}/{camera_id}/`.

### Step 3: Build Datasets (Local)

**Staff Dataset** (interactive labeling):
```bash
python training/data/build_staff_dataset.py --frames-dir training/data/frames
```
- Shows person crops one-by-one
- Press `s` for staff, `c` for customer, `q` to quit
- Aim for ~50+ labeled crops per class

**Re-ID Dataset** (automatic):
```bash
python training/data/build_reid_dataset.py --frames-dir training/data/frames
```
- Automatically tracks persons across frames using IoU matching
- Groups crops by identity

### Step 4: Calibrate Layout (Local, Optional)

```bash
python training/data/calibrate_layout.py --store store1
```
- Opens layout.png in a matplotlib window
- Left-click to add polygon points, right-click to finish a zone
- Saves `training/data/layouts/store1_layout.json`

### Step 5: Upload to Google Drive

Upload the entire `training/` folder to Google Drive:
```
My Drive/
└── purplle_hackathon/
    └── training/    ← upload this folder
```

### Step 6: Run Colab Notebooks

Open each notebook in Google Colab (in order):

1. **01_detection_and_tracking.ipynb** — Verify YOLOv8 works on your footage
2. **02_staff_classifier.ipynb** — Train staff/customer classifier (~15 min)
3. **03_reid_embedding.ipynb** — Train Re-ID embeddings (~20 min)
4. **04_full_pipeline.ipynb** — Run full end-to-end pipeline

> **Important:** Set Runtime → Change runtime type → **GPU (T4)** in Colab.

### Step 7: Integrate with Main Pipeline

After training, download model weights and register the detector:

```python
# In pipeline/detect.py, add:
from training.models.yolo_detector import YOLODetector
DETECTOR_REGISTRY["yolo"] = YOLODetector

# Run with:
python -m pipeline.detect --detector yolo \
    --weights training/models/weights/ \
    --video path/to/video.mp4
```

---

## 📁 Directory Structure

After running all steps, the directory looks like:

```
training/
├── config.yaml                           # Central configuration
├── requirements.txt                      # Training dependencies
├── README.md                             # This file
│
├── data/
│   ├── prepare_frames.py                 # Frame extraction
│   ├── build_staff_dataset.py            # Staff dataset builder
│   ├── build_reid_dataset.py             # Re-ID dataset builder
│   ├── calibrate_layout.py              # Zone calibration tool
│   │
│   ├── raw/                              # Raw footage (you provide)
│   │   ├── store1/  (cam*.mp4, layout.png)
│   │   └── store2/  (cam*.mp4, layout.png)
│   │
│   ├── frames/                           # Extracted frames (auto-generated)
│   │   ├── manifest.csv
│   │   ├── STORE_01/
│   │   │   ├── cam1_zone/  (frame_000000.jpg, ...)
│   │   │   ├── cam2_zone/
│   │   │   ├── cam3_entry/
│   │   │   └── cam4_billing/
│   │   └── STORE_02/ (...)
│   │
│   ├── staff_dataset/                    # Staff classifier training data
│   │   ├── train/ {staff/, customer/}
│   │   └── val/   {staff/, customer/}
│   │
│   ├── reid_dataset/                     # Re-ID training data
│   │   ├── identity_0000/ (crop_000.jpg, ...)
│   │   ├── identity_0001/
│   │   └── ...
│   │
│   └── layouts/                          # Calibrated zone layouts
│       ├── store1_layout.json
│       └── store2_layout.json
│
├── notebooks/                            # Google Colab notebooks
│   ├── 01_detection_and_tracking.ipynb
│   ├── 02_staff_classifier.ipynb
│   ├── 03_reid_embedding.ipynb
│   └── 04_full_pipeline.ipynb
│
└── models/                               # Production model code
    ├── __init__.py
    ├── yolo_detector.py                  # YOLODetector(DetectorBase)
    ├── event_converter.py                # Schema translation
    └── weights/                          # Trained model weights
        ├── staff_classifier.pth          # (from notebook 02)
        └── reid_osnet.pth                # (from notebook 03)
```

---

## 🔧 Configuration

Edit `config.yaml` to customize:

- **Stores & cameras**: Adjust to match your footage structure
- **Frame extraction**: Change FPS, duplicate detection threshold
- **YOLOv8**: Model size (n/s/m), confidence threshold
- **Staff classifier**: Epochs, learning rate, augmentation
- **Re-ID**: Embedding dimension, triplet margin, epochs

---

## 🔌 Model Architecture

### Person Detection
- **Model**: YOLOv8s (pre-trained on COCO, class 0 = person)
- **No fine-tuning** — COCO pre-training already excellent for person detection
- **Inference**: ~640px, conf ≥ 0.35, NMS IoU ≤ 0.45

### Staff Classifier
- **Model**: MobileNetV3-Small (~2.5M params)
- **Training**: Transfer learning from ImageNet
  - Phase 1: Freeze backbone, train head (10 epochs, lr=1e-3)
  - Phase 2: Unfreeze all, fine-tune (20 epochs, lr=1e-4)
- **Input**: 224×224 RGB crop of detected person

### Re-ID Embeddings
- **Model**: OSNet-x0.25 (~0.2M params)
- **Training**: Triplet loss with hard mining (30 epochs)
- **Output**: 128-d L2-normalized embedding vector
- **Input**: 256×128 RGB crop of detected person

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| `No frames found` | Check video paths in `training/data/raw/` |
| `CUDA out of memory` | Reduce batch_size in config.yaml or use smaller model |
| `Staff classifier low accuracy` | Label more crops (aim for 100+ per class) |
| `Re-ID t-SNE looks random` | Need more identities (15+) with more crops each |
| `Layout calibration crashes` | Needs GUI — run locally, not on Colab |
| `Import errors on Colab` | Make sure project root is in sys.path |

---

## 📌 Key Design Decisions

1. **No YOLO fine-tuning**: COCO person class works well on store footage. Fine-tuning on ~1000 frames risks overfitting with minimal gain.

2. **Store-specific staff models**: Uniforms differ per store. If accuracy is important, train separate classifiers per store.

3. **Lightweight Re-ID**: OSNet-x0.25 chosen for real-time inference. For better accuracy with slower speed, consider OSNet-x1.0.

4. **IoU tracking**: Simple but effective for short clips. For production, consider ByteTrack or BoT-SORT.
