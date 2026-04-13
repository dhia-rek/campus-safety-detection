"""
Step 1 — Run YOLOv8 object detection on every frame.

For each input frame, writes a JSON sidecar file containing the list of
detected bounding boxes (label, confidence, xyxy coords).  Empty frames
receive an empty-list JSON so downstream steps can rely on every frame
having a sidecar.

Environment variables
---------------------
DATASET_SPLIT : "training" (default) or "testing"
SUBDIR_FILTER : process only this scene subdirectory (e.g. "05_0019")

Tunable constants (edit at the top of this file)
-------------------------------------------------
MODEL_WEIGHTS : YOLOv8 checkpoint — "yolov8n.pt" (fast) … "yolov8l.pt" (accurate)
CONF_THRES    : minimum detection confidence to keep
KEEP_LABELS   : set of YOLO class names to retain (others are discarded)

Usage
-----
    # Testing split, single scene
    DATASET_SPLIT=testing SUBDIR_FILTER=05_0019 python -m src.pipeline.detect

    # Full training split
    DATASET_SPLIT=training python -m src.pipeline.detect
"""

import json
import os

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from src.paths import TRAIN_FRAMES_ROOT, TEST_FRAMES_ROOT, DETECTIONS_ROOT


# ── Tunable settings ──────────────────────────────────────────────────────────
MODEL_WEIGHTS = "yolov8m.pt"   # swap to yolov8n.pt for speed, yolov8l.pt for accuracy
CONF_THRES    = 0.15
KEEP_LABELS   = {"person", "bicycle", "motorbike", "car", "bus", "truck"}
KEEP_ALL      = False          # set True to retain every YOLO class (debug mode)
# ─────────────────────────────────────────────────────────────────────────────

DATASET_SPLIT  = os.getenv("DATASET_SPLIT", "training").strip().lower()
SUBDIR_FILTER  = os.getenv("SUBDIR_FILTER", "").strip()
FRAMES_ROOT    = str(TEST_FRAMES_ROOT if DATASET_SPLIT == "testing" else TRAIN_FRAMES_ROOT)
DETS_OUT_ROOT  = str(DETECTIONS_ROOT / DATASET_SPLIT)


def is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def detect_frame(model: YOLO, frame_path: str) -> list[dict]:
    """Run YOLO on a single frame and return a list of detection dicts."""
    img = cv2.imread(frame_path)
    if img is None:
        return []

    results = model.predict(img, conf=CONF_THRES, verbose=False)
    dets = []
    for r in results:
        if getattr(r, "boxes", None) is None:
            continue
        boxes   = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs   = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), cls, cf in zip(boxes, classes, confs):
            label = model.names[int(cls)]
            if KEEP_ALL or label in KEEP_LABELS:
                dets.append({
                    "label": label,
                    "conf":  float(cf),
                    "bbox":  [int(x1), int(y1), int(x2), int(y2)],
                })
    return dets


def main() -> None:
    print(f"Split:              {DATASET_SPLIT}")
    print(f"Frames root:        {FRAMES_ROOT}")
    print(f"Detections output:  {DETS_OUT_ROOT}")
    if SUBDIR_FILTER:
        print(f"Subdir filter:      {SUBDIR_FILTER}")

    if not os.path.isdir(FRAMES_ROOT):
        raise FileNotFoundError(f"Frames root not found: {FRAMES_ROOT}")

    os.makedirs(DETS_OUT_ROOT, exist_ok=True)
    model = YOLO(MODEL_WEIGHTS)

    total = written = 0
    for dirpath, _, files in os.walk(FRAMES_ROOT):
        imgs = [f for f in files if is_image(f)]
        if not imgs:
            continue

        rel = os.path.relpath(dirpath, FRAMES_ROOT)
        if SUBDIR_FILTER and rel != SUBDIR_FILTER:
            continue

        out_dir = os.path.join(DETS_OUT_ROOT, rel)
        os.makedirs(out_dir, exist_ok=True)

        for fname in tqdm(imgs, desc=f"Detect {rel}", leave=False):
            total += 1
            frame_path = os.path.join(dirpath, fname)
            json_path  = os.path.join(out_dir, os.path.splitext(fname)[0] + ".json")

            dets = detect_frame(model, frame_path)
            with open(json_path, "w") as f:
                json.dump(dets, f)
            if dets:
                written += 1

    print("Done detection.")
    print(f"Frames visited: {total} | Frames with ≥1 detection: {written}")
    print(f"JSONs saved under: {DETS_OUT_ROOT}")


if __name__ == "__main__":
    main()
