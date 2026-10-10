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


import numpy as np
import torch

from src.paths import PROMPTS_DIR

BATCH_SIZE = 64
OUT_PATH   = str(PROMPTS_DIR / "text_features.npz")


def read_lines(path: str) -> list[str]:
    """Return non-empty stripped lines from *path*."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def encode_prompts(model, tokenizer, device: str) -> tuple[np.ndarray, int, int]:
    """Encode normal.txt + abnormal.txt → (L2-normalised feats, n_normal, n_abnormal)."""
    normals   = read_lines(str(PROMPTS_DIR / "normal.txt"))
    abnormals = read_lines(str(PROMPTS_DIR / "abnormal.txt"))
    print(f"Normal prompts: {len(normals)} | Abnormal prompts: {len(abnormals)}")

    texts = normals + abnormals
    all_feats: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            tokens = tokenizer(texts[i : i + BATCH_SIZE]).to(device)
            feats  = model.encode_text(tokens).float()
            feats  = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
    return np.vstack(all_feats).astype(np.float32), len(normals), len(abnormals)


def save_text_features(model, tokenizer, device: str) -> None:
    feats, n_normal, n_abnormal = encode_prompts(model, tokenizer, device)
    np.savez_compressed(OUT_PATH, feats=feats, n_normal=n_normal, n_abnormal=n_abnormal)
    print(f"Saved text features → {OUT_PATH}")


def main() -> None:
    from src.pipeline.clip_model import load_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, _, tokenizer = load_clip(device)
    save_text_features(model, tokenizer, device)


if __name__ == "__main__":
    main()
