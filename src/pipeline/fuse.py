"""
Step 6 — Fuse visual and audio scores into a single per-frame CSV.

Reads:
  - runtime_data/scores/<split>/frames/<scene>_frames.csv   (visual, smoothed)
  - runtime_data/scores/<split>/audio/<scene>_audio.csv     (audio, per-frame)

Writes:
  - runtime_data/scores/<split>/fused/<scene>_fused.csv

Both modalities are z-normalised against their own clip-level statistics,
then fused as `max(visual_z, audio_z)` — either modality firing flags the frame.

Environment variables
---------------------
SCENE         : scene name, e.g. "05_0019" (required)
DATASET_SPLIT : "testing" (default) or "training"

Usage
-----
    SCENE=05_0019 python -m src.pipeline.fuse
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import SCORES_ROOT


EPS = 1e-6
SCENE         = os.getenv("SCENE", "").strip()
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "testing").strip().lower()


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + EPS)


def main():
    if not SCENE:
        raise RuntimeError("SCENE environment variable is required.")

    root       = Path(str(SCORES_ROOT)) / DATASET_SPLIT
    visual_csv = root / "frames" / f"{SCENE}_frames.csv"
    audio_csv  = root / "audio"  / f"{SCENE}_audio.csv"

    if not visual_csv.is_file():
        raise FileNotFoundError(f"Visual scores not found: {visual_csv}")
    if not audio_csv.is_file():
        raise FileNotFoundError(f"Audio scores not found: {audio_csv}")

    visual = pd.read_csv(visual_csv)
    audio  = pd.read_csv(audio_csv)

    merged = pd.merge(visual, audio, on="frame", how="outer").sort_values("frame").reset_index(drop=True)
    merged["smoothed_score"] = merged["smoothed_score"].fillna(0.0)
    merged["audio_score"]    = merged["audio_score"].fillna(0.0)

    merged["visual_z"]    = zscore(merged["smoothed_score"].values)
    merged["audio_z"]     = zscore(merged["audio_score"].values)
    merged["fused_score"] = np.maximum(merged["visual_z"].values, merged["audio_z"].values)

    out_dir = root / "fused"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{SCENE}_fused.csv"
    merged[["frame", "smoothed_score", "audio_score", "visual_z", "audio_z", "fused_score"]].to_csv(
        out_csv, index=False
    )
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
