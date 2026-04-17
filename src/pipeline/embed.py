"""
Step 3b — Embed object crops with CLIP and save per-scene .npz feature files.

For each scene sub-directory under crops/<split>/, produces one .npz file
under features/<split>/ containing:
  - feats : float32 array (N_crops, 512) of L2-normalised image embeddings
  - paths : string array of the absolute paths to the source crop images

Also calls text_features.py logic once to ensure prompts/text_features.npz
exists before the scoring step.

Environment variables
---------------------
DATASET_SPLIT : "training" (default) or "testing"
SUBDIR_FILTER : process only this scene subdirectory (e.g. "05_0019")

Usage
-----
    DATASET_SPLIT=testing SUBDIR_FILTER=05_0019 python -m src.pipeline.embed
"""

import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.paths import CROPS_ROOT, PROMPTS_DIR, FEATURES_ROOT


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "ViT-B/32"
BATCH_SIZE    = 128
IMG_EXTS      = {".jpg", ".jpeg", ".png"}
SPLIT         = os.getenv("DATASET_SPLIT", "training").strip().lower()
SUBDIR_FILTER = os.getenv("SUBDIR_FILTER", "").strip()
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def embed_texts_once(model, device: str) -> None:
    """Write prompts/text_features.npz if it does not already exist."""
    out_npz = str(PROMPTS_DIR / "text_features.npz")
    if os.path.isfile(out_npz):
        print("text_features.npz already exists — skipping.")
        return

    import clip

    print("Encoding text prompts…")
    normals   = _read_lines(str(PROMPTS_DIR / "normal.txt"))
    abnormals = _read_lines(str(PROMPTS_DIR / "abnormal.txt"))
    texts     = normals + abnormals

    with torch.no_grad():
        feats_list = []
        toks = clip.tokenize(texts, truncate=True).to(device)
        for i in range(0, len(texts), BATCH_SIZE):
            f = model.encode_text(toks[i : i + BATCH_SIZE])
            f = f / f.norm(dim=-1, keepdim=True)
            feats_list.append(f.float().cpu().numpy())

    feats = np.concatenate(feats_list, axis=0)
    np.savez(out_npz, feats=feats, n_normal=len(normals), n_abnormal=len(abnormals), model=MODEL_NAME)
    print(f"Saved: {out_npz}")


def iter_image_folders(split_root: str):
    """Yield (rel_subdir, [sorted image paths]) for every sub-directory with images."""
    for dirpath, _, files in os.walk(split_root):
        imgs = [
            os.path.join(dirpath, f)
            for f in files
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]
        if not imgs:
            continue
        rel = os.path.relpath(dirpath, split_root)
        if SUBDIR_FILTER and rel != SUBDIR_FILTER:
            continue
        yield rel, sorted(imgs)


def embed_split(model, preprocess, device: str) -> None:
    """Embed all crops for SPLIT and write one .npz per scene folder."""
    split_crops     = os.path.join(str(CROPS_ROOT),    SPLIT)
    split_feats_dir = os.path.join(str(FEATURES_ROOT), SPLIT)
    os.makedirs(split_feats_dir, exist_ok=True)

    if not os.path.isdir(split_crops):
        raise FileNotFoundError(f"Crop folder not found: {split_crops}")

    folders = list(iter_image_folders(split_crops))
    if not folders:
        print(f"No images found under: {split_crops}")
        print("Check that your crops exist and have .jpg/.jpeg/.png extensions")
        return

    print(f"Found {len(folders)} crop folder(s) in '{SPLIT}' split.")
    total_imgs = written_files = 0

    for rel, imgs in tqdm(folders, desc=f"Embedding {SPLIT}"):
        total_imgs += len(imgs)

        out_name = (rel.replace("/", "_") or "root") + ".npz"
        out_npz  = os.path.join(split_feats_dir, out_name)
        if os.path.isfile(out_npz):
            continue  # resume-safe: skip already-embedded folders

        feats_list: list[np.ndarray] = []
        paths: list[str]             = []

        with torch.no_grad():
            for i in range(0, len(imgs), BATCH_SIZE):
                batch_paths = imgs[i : i + BATCH_SIZE]
                batch_imgs  = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
                batch_t     = torch.stack(batch_imgs).to(device)
                f = model.encode_image(batch_t)
                f = f / f.norm(dim=-1, keepdim=True)
                feats_list.append(f.float().cpu().numpy())
                paths.extend(batch_paths)

        feats = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, 512), dtype=np.float32)
        np.savez(out_npz, feats=feats, paths=np.array(paths))
        written_files += 1

    print(f"Done — {SPLIT}: {len(folders)} folders, {total_imgs} images, {written_files} files written")
    print(f"Outputs in: {split_feats_dir}")


def main() -> None:
    import clip

    print(f"Device: {DEVICE}")
    os.makedirs(str(FEATURES_ROOT), exist_ok=True)

    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    embed_texts_once(model, DEVICE)
    embed_split(model, preprocess, DEVICE)


if __name__ == "__main__":
    main()
