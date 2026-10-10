"""Step 3b — Embed object crops with CLIP and save per-scene .npz feature files.

For each scene sub-directory under crops/<split>/, produces one .npz file
under features/<split>/ containing:
  - feats : float32 array (N_crops, 512) of L2-normalised image embeddings
  - paths : string array of the absolute paths to the source crop images

Also writes prompts/text_features.npz (via text_features.py) if it is missing,
so the scoring step has its text embeddings.

Environment variables
---------------------
DATASET_SPLIT : "training" (default) or "testing"
SUBDIR_FILTER : process only this scene subdirectory (e.g. "05_0019")

Usage
-----
    DATASET_SPLIT=testing SUBDIR_FILTER=05_0019 python -m src.pipeline.embed
"""

import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.paths import CROPS_ROOT, FEATURES_ROOT, PROMPTS_DIR

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 128
IMG_EXTS      = {".jpg", ".jpeg", ".png"}
SPLIT         = os.getenv("DATASET_SPLIT", "training").strip().lower()
SUBDIR_FILTER = os.getenv("SUBDIR_FILTER", "").strip()
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def embed_texts_once(model: Any, tokenizer: Any, device: str) -> None:
    """Write prompts/text_features.npz if it does not already exist.

    Args:
        model: CLIP model from ``load_clip``.
        tokenizer: CLIP tokenizer from ``load_clip``.
        device: Torch device string.
    """
    if (PROMPTS_DIR / "text_features.npz").is_file():
        print("text_features.npz already exists — skipping.")
        return
    from src.pipeline.text_features import save_text_features

    save_text_features(model, tokenizer, device)


def iter_image_folders(split_root: str) -> Iterator[tuple[str, list[str]]]:
    """Yield (rel_subdir, [sorted image paths]) for every sub-directory with images.

    Honours SUBDIR_FILTER (exact match on the relative path).

    Args:
        split_root: Root folder to walk, e.g. ``crops/<split>``.
    """
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


def embed_split(model: Any, preprocess: Any, device: str) -> None:
    """Embed all crops for SPLIT and write one .npz per scene folder.

    Embeddings are L2-normalised so a dot product equals cosine similarity in
    the score step. Scenes whose .npz already exists are skipped, making
    reruns resumable. The output name is the relative folder path with ``/``
    replaced by ``_`` (``root`` for the top level).

    Args:
        model: CLIP model from ``load_clip``.
        preprocess: CLIP image preprocessing transform.
        device: Torch device string.

    Raises:
        FileNotFoundError: If the crops folder for SPLIT does not exist.
    """
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
    """Load CLIP, ensure text features exist, then embed the split's crops."""
    from src.pipeline.clip_model import load_clip

    print(f"Device: {DEVICE}")
    os.makedirs(str(FEATURES_ROOT), exist_ok=True)

    model, preprocess, tokenizer = load_clip(DEVICE)
    embed_texts_once(model, tokenizer, DEVICE)
    embed_split(model, preprocess, DEVICE)


if __name__ == "__main__":
    main()
