"""
Step 3a — Encode text prompts with CLIP and save to prompts/text_features.npz.

Reads normal.txt and abnormal.txt from the prompts directory and produces
a single .npz file containing:
  - feats      : float32 array (N_prompts, 512) of L2-normalised embeddings
  - n_normal   : number of normal prompts
  - n_abnormal : number of abnormal prompts

This file is consumed by embed.py (via embed_texts_once) and by score.py.

Usage
-----
    python -m src.pipeline.text_features
"""

import os

import numpy as np
import torch

from src.paths import PROMPTS_DIR


MODEL_NAME = "ViT-B/32"   # must match the model used in embed.py
BATCH_SIZE = 64
OUT_PATH   = str(PROMPTS_DIR / "text_features.npz")


def read_lines(path: str) -> list[str]:
    """Return non-empty stripped lines from *path*."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    import clip  # openai/CLIP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    normal_path   = str(PROMPTS_DIR / "normal.txt")
    abnormal_path = str(PROMPTS_DIR / "abnormal.txt")

    normals   = read_lines(normal_path)
    abnormals = read_lines(abnormal_path)
    print(f"Normal prompts: {len(normals)} | Abnormal prompts: {len(abnormals)}")

    model, _ = clip.load(MODEL_NAME, device=device)
    model.eval()

    texts    = normals + abnormals
    all_feats: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch  = texts[i : i + BATCH_SIZE]
            tokens = clip.tokenize(batch).to(device)
            feats  = model.encode_text(tokens).float()
            feats  = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())

    text_feats = np.vstack(all_feats).astype(np.float32)
    np.savez_compressed(
        OUT_PATH,
        feats=text_feats,
        n_normal=len(normals),
        n_abnormal=len(abnormals),
    )
    print(f"Saved text features → {OUT_PATH}")


if __name__ == "__main__":
    main()
