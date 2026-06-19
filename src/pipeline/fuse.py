"""
Step 6 — Fuse visual and audio scores into a single per-frame CSV.

Reads:
  - runtime_data/scores/<split>/frames/<scene>_frames.csv   (visual, smoothed)
  - runtime_data/scores/<split>/audio/<scene>_audio.csv     (audio, per-frame)

Writes:
  - runtime_data/scores/<split>/fused/<scene>_fused.csv

Fusion is `max(visual, audio_z, verbal_z)` — any modality firing flags the frame.
By default the VISUAL signal is the RAW contrastive score (globally comparable,
so a violent clip scores higher than a calm one); set VISUAL_ZSCORE=1 to z-score
it instead (per-clip anomaly localisation). Audio and verbal are z-normalised
against their own clip-level statistics.

Note: with the raw visual default, the `visual_z` column is NOT a z-score and is
compared against the same `threshold` as the z-scored audio/verbal signals.

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

    visual = pd.read_csv(visual_csv)

    # Audio is optional: video-only datasets (e.g. ShanghaiTech) have no .wav.
    if audio_csv.is_file():
        audio  = pd.read_csv(audio_csv)
        merged = pd.merge(visual, audio, on="frame", how="outer").sort_values("frame").reset_index(drop=True)
        merged["audio_score"] = merged["audio_score"].fillna(0.0)
        print("Audio modality found — fusing visual + audio.")
    else:
        merged = visual.sort_values("frame").reset_index(drop=True)
        merged["audio_score"] = 0.0
        print("No audio CSV — fusing visual only (video-only dataset).")

    merged["smoothed_score"] = merged["smoothed_score"].fillna(0.0)

    # Visual signal. Per-clip z-scoring localizes the abnormal *moment within* a
    # scene (good for ShanghaiTech), but it erases the difference between a
    # violent clip and a calm one (both get a relative max). For violence
    # classification we use the RAW contrastive score, which is globally
    # comparable. Toggle with VISUAL_ZSCORE=1.
    if os.getenv("VISUAL_ZSCORE", "0").lower() in {"1", "true", "yes"}:
        merged["visual_z"] = zscore(merged["smoothed_score"].values)
    else:
        merged["visual_z"] = merged["smoothed_score"].values
    merged["audio_z"]  = zscore(merged["audio_score"].values)

    # Optional third modality: verbal abuse from the voice recognizer (transcribe.py).
    verbal_csv = root / "verbal" / f"{SCENE}_verbal.csv"
    has_verbal = verbal_csv.is_file()
    if has_verbal:
        verbal = pd.read_csv(verbal_csv)[["frame", "verbal_score"]]
        merged = pd.merge(merged, verbal, on="frame", how="left")
        merged["verbal_score"] = merged["verbal_score"].fillna(0.0)
        merged["verbal_z"]     = zscore(merged["verbal_score"].values)
        print("Verbal modality found — fusing visual + audio + verbal.")
    else:
        merged["verbal_score"] = 0.0
        merged["verbal_z"]     = 0.0
        print("No verbal CSV — fusing visual + audio only.")

    merged["fused_score"] = np.maximum.reduce([
        merged["visual_z"].values,
        merged["audio_z"].values,
        merged["verbal_z"].values,
    ])

    out_dir = root / "fused"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{SCENE}_fused.csv"
    merged[[
        "frame", "smoothed_score", "audio_score", "verbal_score",
        "visual_z", "audio_z", "verbal_z", "fused_score",
    ]].to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
