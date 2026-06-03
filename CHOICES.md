# Engineering Choices & Trade-offs

This document outlines the key architectural and modeling decisions made for the Store Intelligence API pipeline.

## 1. Vision Models & Annotation Strategy

**Choice: Pre-trained YOLOv8 (No Fine-Tuning) + Modular Classifiers**
Instead of heavily fine-tuning a monolithic object detection model to predict "staff" vs. "customer", we opted for a composite, modular pipeline:
*   **Person Detection (YOLOv8s):** We use a COCO pre-trained YOLO model filtered to class `0` (person). COCO person detection is incredibly robust off-the-shelf. Fine-tuning YOLO on a limited dataset of store footage risks severe overfitting and catastrophic forgetting, yielding minimal gain for maximum effort.
*   **Staff Classification (MobileNetV3-Small):** We extract the bounding box crop of each detected person and pass it through a lightweight, custom-trained MobileNet classifier (binary classification: staff vs. customer). Because staff uniforms vary significantly store-by-store, this allows us to quickly train and hot-swap store-specific classifiers without retraining the entire detection backbone.
*   **Person Re-ID (OSNet-x0.25):** We use an OSNet architecture specifically optimized for Person Re-Identification. It generates a 128-dimensional embedding from the person crop, allowing us to accurately track individuals as they move across disjoint camera views (e.g., from Entry to Billing).

**Trade-offs:** 
*   *Pros:* Highly modular. Training the classifier and Re-ID models takes minutes on a GPU (compared to hours for YOLO fine-tuning). Excellent generalization. Allows CPU-friendly inference in production.
*   *Cons:* Requires running three separate inference passes per frame. We mitigate this by using ultra-lightweight architectures (MobileNet-Small and OSNet-x0.25) which run comfortably on edge CPUs.

## 2. Infrastructure & Training Isolation

**Choice: Google Colab for Training / Local Edge for Inference**
Given the constraint of running on a potentially resource-limited Windows machine without a heavy local GPU:
*   We separated the entire training suite into an isolated `training/` folder.
*   The raw data preparation (frame extraction, GUI polygon zone calibration) is done locally, generating lightweight datasets that can be easily uploaded.
*   Resource-heavy tasks (training the staff classifier and triplet-loss Re-ID embedding) are designed to run entirely on Google Colab using free T4 GPUs.
*   The resulting `.pth` weights are seamlessly plugged back into the local `pipeline.detect` orchestrator.

## 3. Software Architecture (Detector Registry)

**Choice: The `DetectorBase` Interface**
The main pipeline orchestrator (`pipeline/detect.py`) communicates with the computer vision models entirely through an abstract `DetectorBase` class.
*   **Why:** This allows the pipeline to be built, tested, and containerized long before the actual machine learning models are finished. During early development, the API was tested using a `DummyDetector` that generated synthetic events.
*   **Result:** When the Colab training finished, we simply implemented `YOLODetector(DetectorBase)` and dropped it into the `DETECTOR_REGISTRY`. No core pipeline logic had to be rewritten.

## 4. Storage & API

**Choice: SQLite & FastAPI**
*   **FastAPI:** Chosen for its asynchronous capabilities, built-in validation (Pydantic), and seamless WebSocket support for the live dashboard.
*   **SQLite:** Selected as the default database for its zero-configuration portability, ensuring the local pipeline runs out-of-the-box. The application is built using SQLAlchemy, making it trivial to switch to PostgreSQL when containerized in production (e.g., via `docker-compose`).
