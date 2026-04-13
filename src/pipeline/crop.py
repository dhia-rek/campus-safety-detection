"""
Step 2 — Crop detected objects out of every frame.

Reads the YOLO JSON sidecars produced by detect.py, pads each bounding
box by PADDING_RATIO, and saves a 224×224 JPEG crop.  Also writes an
index CSV (<split>_crops_index.csv) listing every crop with its metadata.

Environment variables
---------------------
DATASET_SPLIT : "training" (default) or "testing"
SUBDIR_FILTER : process only this scene subdirectory (e.g. "05_0019")

Usage
-----
    DATASET_SPLIT=testing SUBDIR_FILTER=05_0019 python -m src.pipeline.crop
"""

import json
import math
import os

import cv2
from tqdm import tqdm

from src.paths import TRAIN_FRAMES_ROOT, TEST_FRAMES_ROOT, DETECTIONS_ROOT, CROPS_ROOT


# ── Settings ──────────────────────────────────────────────────────────────────
PADDING_RATIO = 0.15   # fractional padding added around each bounding box
OUTPUT_SIZE   = 224    # crop is resized to OUTPUT_SIZE × OUTPUT_SIZE (pixels)
MIN_BOX_WH    = 6      # boxes smaller than this (px) are skipped
# ─────────────────────────────────────────────────────────────────────────────

DATASET_SPLIT  = os.getenv("DATASET_SPLIT", "training").strip().lower()
SUBDIR_FILTER  = os.getenv("SUBDIR_FILTER", "").strip()
FRAMES_ROOT    = str(TEST_FRAMES_ROOT if DATASET_SPLIT == "testing" else TRAIN_FRAMES_ROOT)
DETS_ROOT      = str(DETECTIONS_ROOT / DATASET_SPLIT)
CROPS_OUT_ROOT = str(CROPS_ROOT / DATASET_SPLIT)


def is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def clamp(v: float, lo: int, hi: int) -> int:
    return max(lo, min(int(v), hi))


def pad_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    """Expand a bounding box by PADDING_RATIO and clamp to image bounds."""
    bw, bh = x2 - x1, y2 - y1
    px, py  = int(math.floor(bw * PADDING_RATIO)), int(math.floor(bh * PADDING_RATIO))
    return (
        clamp(x1 - px, 0, w - 1),
        clamp(y1 - py, 0, h - 1),
        clamp(x2 + px, 0, w - 1),
        clamp(y2 + py, 0, h - 1),
    )


def crop_frame(img: "np.ndarray", dets: list[dict], out_dir: str, stem: str) -> list[dict]:
    """Crop every detection in *dets* from *img* and save to *out_dir*.

    Returns a list of metadata rows for the index CSV.
    """
    import numpy as np  # lazy import — only needed here

    h, w = img.shape[:2]
    rows = []

    for obj_id, det in enumerate(dets):
        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
        label = det.get("label", "")
        conf  = float(det.get("conf", 0.0))

        x1p, y1p, x2p, y2p = pad_box(int(x1), int(y1), int(x2), int(y2), w, h)
        if x2p <= x1p or y2p <= y1p:
            continue
        if (x2p - x1p) < MIN_BOX_WH or (y2p - y1p) < MIN_BOX_WH:
            continue

        crop = img[y1p:y2p, x1p:x2p]
        try:
            crop = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        except Exception:
            continue

        crop_name = f"{stem}_{obj_id:03d}.jpg"
        crop_path = os.path.join(out_dir, crop_name)
        cv2.imwrite(crop_path, crop)

        rows.append({
            "frame_file": stem,
            "crop_path":  crop_path,
            "label":      label,
            "conf":       conf,
            "bbox":       [int(x1), int(y1), int(x2), int(y2)],
        })

    return rows


def main() -> None:
    print(f"Frames root:     {FRAMES_ROOT}")
    print(f"Detections root: {DETS_ROOT}")
    print(f"Crops out root:  {CROPS_OUT_ROOT}")
    if SUBDIR_FILTER:
        print(f"Subdir filter:   {SUBDIR_FILTER}")

    if not os.path.isdir(FRAMES_ROOT):
        raise FileNotFoundError(f"Frames root not found: {FRAMES_ROOT}")
    if not os.path.isdir(DETS_ROOT):
        raise FileNotFoundError(f"Detections root not found: {DETS_ROOT}")

    total_frames = frames_with_json = total_crops = 0
    all_rows: list[dict] = []

    for dirpath, _, files in os.walk(FRAMES_ROOT):
        imgs = [f for f in files if is_image(f)]
        if not imgs:
            continue

        rel = os.path.relpath(dirpath, FRAMES_ROOT)
        if SUBDIR_FILTER and rel != SUBDIR_FILTER:
            continue

        det_dir = os.path.join(DETS_ROOT, rel)
        out_dir = os.path.join(CROPS_OUT_ROOT, rel)
        os.makedirs(out_dir, exist_ok=True)

        for fname in tqdm(sorted(imgs), desc=f"Cropping {rel}", leave=False):
            total_frames += 1
            frame_path = os.path.join(dirpath, fname)
            json_path  = os.path.join(det_dir, os.path.splitext(fname)[0] + ".json")

            if not os.path.isfile(json_path):
                continue

            frames_with_json += 1
            img = cv2.imread(frame_path)
            if img is None:
                continue

            with open(json_path) as f:
                try:
                    dets = json.load(f)
                except Exception:
                    dets = []

            if not dets:
                continue

            stem = os.path.splitext(fname)[0]
            rows = crop_frame(img, dets, out_dir, stem)
            total_crops += len(rows)
            all_rows.extend(rows)

    if all_rows:
        import pandas as pd
        csv_out = os.path.join(CROPS_OUT_ROOT, f"{DATASET_SPLIT}_crops_index.csv")
        pd.DataFrame(all_rows).to_csv(csv_out, index=False)
        print(f"Saved index CSV: {csv_out}")

    print("Summary:")
    print(f"  Total frames scanned:    {total_frames}")
    print(f"  Frames with JSON sidecar: {frames_with_json}")
    print(f"  Total crops saved:        {total_crops}")
    print(f"  Crops root:               {CROPS_OUT_ROOT}")


if __name__ == "__main__":
    main()
