"""
Step 7 — Frame-level ROC-AUC evaluation against ECE campus ground-truth masks.

Loads every `<scene>.npy` mask under `<dataset>/testing/test_frame_mask/`,
aligns it with the matching scored CSV, and reports both per-scene and
pooled frame-level ROC-AUC.

Environment variables
---------------------
DATASET_SPLIT : "testing" (default)
SCORE_KIND    : "frames" (default), "fused", or "audio"
                — determines which CSV folder under runtime_data/scores/ is read,
                  and what filename suffix is expected (e.g. <scene>_fused.csv).
SCORE_COL     : column name to read; defaults based on SCORE_KIND
                (smoothed_score / fused_score / audio_score).

Usage
-----
    python -m src.pipeline.evaluate                # visual only
    SCORE_KIND=fused python -m src.pipeline.evaluate
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.paths import DATASET_ROOT, SCORES_ROOT


DATASET_SPLIT = os.getenv("DATASET_SPLIT", "testing").strip().lower()
SCORE_KIND    = os.getenv("SCORE_KIND",    "frames").strip().lower()
SCORE_COL_ENV = os.getenv("SCORE_COL",     "").strip()

DEFAULT_COL = {
    "frames": "smoothed_score",
    "fused":  "fused_score",
    "audio":  "audio_score",
}


def main() -> None:
    masks_dir  = Path(str(DATASET_ROOT)) / DATASET_SPLIT / "test_frame_mask"
    scores_dir = Path(str(SCORES_ROOT))  / DATASET_SPLIT / SCORE_KIND
    col        = SCORE_COL_ENV or DEFAULT_COL.get(SCORE_KIND, "smoothed_score")

    if not masks_dir.is_dir():
        raise FileNotFoundError(
            f"Frame-mask directory not found: {masks_dir}\n"
            "Ground-truth frame masks expected under testing/test_frame_mask/."
        )
    if not scores_dir.is_dir():
        raise FileNotFoundError(f"Score directory not found: {scores_dir}")

    print(f"Masks dir:  {masks_dir}")
    print(f"Scores dir: {scores_dir}")
    print(f"Score col:  {col}")
    print()

    per_scene_auc: list[tuple[str, float, int, int]] = []
    y_true_all:  list[int]   = []
    y_score_all: list[float] = []

    for mask_path in sorted(masks_dir.glob("*.npy")):
        scene = mask_path.stem
        score_csv = scores_dir / f"{scene}_{SCORE_KIND}.csv"

        if not score_csv.is_file():
            print(f"  Skip {scene}: no score CSV at {score_csv.name}")
            continue

        mask = np.load(mask_path).astype(int)   # (n_frames,)
        df   = pd.read_csv(score_csv)

        if col not in df.columns:
            print(f"  Skip {scene}: column '{col}' not in {score_csv.name}")
            continue

        # Build a per-frame score vector aligned to mask length.
        # Frames absent from the CSV are treated as 0 (no anomaly signal).
        scores = np.zeros(len(mask), dtype=float)
        valid  = df["frame"].values < len(mask)
        scores[df["frame"].values[valid]] = df[col].values[valid]

        if mask.sum() == 0 or mask.sum() == len(mask):
            print(f"  Skip {scene}: degenerate ground truth (all 0s or all 1s)")
            continue

        auc = roc_auc_score(mask, scores)
        per_scene_auc.append((scene, auc, len(mask), int(mask.sum())))
        y_true_all.extend(mask.tolist())
        y_score_all.extend(scores.tolist())
        print(f"  {scene}: AUC={auc:.4f}  frames={len(mask)}  pos={int(mask.sum())}")

    if not per_scene_auc:
        print("\nNo scenes evaluated.")
        return

    overall_auc = roc_auc_score(y_true_all, y_score_all)
    mean_auc    = float(np.mean([a for _, a, _, _ in per_scene_auc]))

    print()
    print(f"Scenes evaluated:      {len(per_scene_auc)}")
    print(f"Mean per-scene AUC:    {mean_auc:.4f}")
    print(f"Pooled (overall) AUC:  {overall_auc:.4f}")


if __name__ == "__main__":
    main()
