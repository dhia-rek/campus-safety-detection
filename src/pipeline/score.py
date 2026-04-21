"""
Step 4 — Compute per-frame bullying scores from CLIP embeddings.

Algorithm
---------
1. For each object crop in a scene, compute a contrastive CLIP score:
       score = max_sim(abnormal_prompts) − max_sim(normal_prompts)
2. Aggregate to frame level by taking the MAX object score per frame.
3. (Optional) Z-score normalise using training-split statistics.
4. Apply Gaussian temporal smoothing (σ = SMOOTH_SIGMA frames).

Outputs two CSVs per scene:
  - scores/<split>/objects/<scene>_objects.csv  — per-object scores
  - scores/<split>/frames/<scene>_frames.csv   — per-frame smoothed scores

The testing split is normalised using calibration statistics derived from
the training split (if training features are available).

Usage
-----
    python -m src.pipeline.score
"""

import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from src.paths import FEATURES_ROOT, PROMPTS_DIR, SCORES_ROOT


# ── Config ────────────────────────────────────────────────────────────────────
SMOOTH_SIGMA = 21   # Gaussian smoothing kernel width (frames)
EPS          = 1e-6
TEXT_NPZ     = str(PROMPTS_DIR / "text_features.npz")
# ─────────────────────────────────────────────────────────────────────────────


def load_text_features() -> tuple[np.ndarray, np.ndarray]:
    """Load normal and abnormal CLIP text embeddings from disk.

    Returns (normal_feats, abnormal_feats) each of shape (N, 512).
    """
    fallback = os.path.join(str(FEATURES_ROOT), "text_features.npz")
    path = TEXT_NPZ if os.path.isfile(TEXT_NPZ) else fallback

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Text features not found at {TEXT_NPZ} or {fallback}. "
            "Run `python -m src.pipeline.text_features` first."
        )

    data = np.load(path)
    feats      = data["feats"]
    n_normal   = int(data["n_normal"])
    n_abnormal = int(data["n_abnormal"])
    return feats[:n_normal], feats[n_normal : n_normal + n_abnormal]


def compute_object_scores(
    img_feats: np.ndarray,
    normal_feats: np.ndarray,
    abnormal_feats: np.ndarray,
) -> np.ndarray:
    """Return a contrastive bullying score for each object crop embedding.

    img_feats : (N_obj, D) — L2-normalised CLIP image embeddings
    Returns   : (N_obj,)  — higher = more anomalous
    """
    sim_normal   = img_feats @ normal_feats.T    # (N_obj, N_normal)
    sim_abnormal = img_feats @ abnormal_feats.T  # (N_obj, N_abnormal)
    return sim_abnormal.max(axis=1) - sim_normal.max(axis=1)


def process_split(
    split: str,
    norm_mu: float | None = None,
    norm_std: float | None = None,
) -> np.ndarray:
    """Score all scenes in *split* and write CSV outputs.

    If *norm_mu* / *norm_std* are given, frame scores are Z-score normalised
    before smoothing (used for the testing split after calibration on training).

    Returns a flat array of all raw frame-level scores (used for calibration).
    """
    in_dir      = os.path.join(str(FEATURES_ROOT), split)
    out_obj_dir = os.path.join(str(SCORES_ROOT), split, "objects")
    out_frm_dir = os.path.join(str(SCORES_ROOT), split, "frames")
    os.makedirs(out_obj_dir, exist_ok=True)
    os.makedirs(out_frm_dir, exist_ok=True)

    normal_feats, abnormal_feats = load_text_features()
    all_frame_scores: list[float] = []

    for fname in tqdm(sorted(os.listdir(in_dir)), desc=f"Scoring {split}"):
        if not fname.endswith(".npz"):
            continue

        data  = np.load(os.path.join(in_dir, fname), allow_pickle=True)
        feats = data["feats"]   # (N_obj, D)
        paths = data["paths"]   # crop paths

        frame_indices = [int(os.path.basename(str(p)).split("_")[0]) for p in paths]
        obj_scores    = compute_object_scores(feats, normal_feats, abnormal_feats)

        obj_df = pd.DataFrame({"frame": frame_indices, "object_score": obj_scores})
        obj_df.to_csv(
            os.path.join(out_obj_dir, fname.replace(".npz", "_objects.csv")),
            index=False,
        )

        # Frame-level aggregation — take the maximum object score per frame
        frame_scores = obj_df.groupby("frame")["object_score"].max()

        if norm_mu is not None:
            frame_scores = (frame_scores - norm_mu) / (norm_std + EPS)

        smoothed = gaussian_filter1d(frame_scores.values, sigma=SMOOTH_SIGMA)

        frm_df = pd.DataFrame({
            "frame":          frame_scores.index.values,
            "raw_score":      frame_scores.values,
            "smoothed_score": smoothed,
        })
        frm_df.to_csv(
            os.path.join(out_frm_dir, fname.replace(".npz", "_frames.csv")),
            index=False,
        )

        all_frame_scores.extend(frame_scores.values.tolist())

    return np.array(all_frame_scores)


def main() -> None:
    training_dir = os.path.join(str(FEATURES_ROOT), "training")
    has_training = os.path.isdir(training_dir) and any(
        f.endswith(".npz") for f in os.listdir(training_dir)
    )

    if has_training:
        print("Scoring TRAINING split (calibration)…")
        train_scores = process_split("training")
        mu, std = train_scores.mean(), train_scores.std()
        print(f"Calibration stats → mean: {mu:.4f}, std: {std:.4f}")

        print("Scoring TESTING split (Z-score normalised)…")
        process_split("testing", norm_mu=mu, norm_std=std)
    else:
        print("Training features not found — scoring testing split without calibration.")
        process_split("testing")

    print("Scoring complete.")


if __name__ == "__main__":
    main()
