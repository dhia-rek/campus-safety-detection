#!/usr/bin/env python3
"""
Batch pipeline runner — processes scenes one by one, deleting frames
immediately after embedding to keep disk usage low (~2 GB peak vs 36 GB).
Runs N_WORKERS scenes in parallel to saturate all CPU cores.

Usage
-----
    python scripts/run_batch_pipeline.py              # training + testing + score
    python scripts/run_batch_pipeline.py --testing    # testing + score only
    python scripts/run_batch_pipeline.py --score      # score only
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON       = str(PROJECT_ROOT / ".venv" / "bin" / "python")

# Resolve dataset root the same way src/paths.py does
def _dataset_root() -> Path:
    for candidate in [
        PROJECT_ROOT / "Videos_bullying" / "ece_campus",
        PROJECT_ROOT / "Videos_bullying " / "ece_campus",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Dataset root not found.")

DATASET_ROOT = _dataset_root()
TRAIN_VIDEOS = DATASET_ROOT / "training" / "videos"
TRAIN_FRAMES = DATASET_ROOT / "training" / "frames"
TEST_FRAMES  = DATASET_ROOT / "testing"  / "frames"

N_WORKERS    = 4  # scenes processed in parallel (each uses ~2 CPU cores)
FEATURES_ROOT = PROJECT_ROOT / "runtime_data" / "features"


def is_done(scene: str, split: str) -> bool:
    """Return True if this scene's embedding .npz already exists (resume-safe)."""
    npz = FEATURES_ROOT / split / f"{scene}.npz"
    return npz.exists()


def run(args: list[str], env: dict | None = None) -> None:
    full_env = {**os.environ, **(env or {})}
    result = subprocess.run(args, env=full_env, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Failed: {' '.join(args)}")


def process_training_scene(scene: str) -> None:
    frames_dir = TRAIN_FRAMES / scene
    base_env   = {"DATASET_SPLIT": "training", "SUBDIR_FILTER": scene}

    run([PYTHON, "-m", "src.pipeline.extract_frames"], env={"VIDEO_FILTER": scene})
    run([PYTHON, "-m", "src.pipeline.detect"],         env=base_env)
    run([PYTHON, "-m", "src.pipeline.crop"],           env=base_env)
    run([PYTHON, "-m", "src.pipeline.embed"],          env=base_env)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)


def process_testing_scene(scene: str) -> None:
    base_env = {"DATASET_SPLIT": "testing", "SUBDIR_FILTER": scene}
    run([PYTHON, "-m", "src.pipeline.detect"], env=base_env)
    run([PYTHON, "-m", "src.pipeline.crop"],   env=base_env)
    run([PYTHON, "-m", "src.pipeline.embed"],  env=base_env)


def run_parallel(fn, scenes: list[str], split: str) -> list[str]:
    """Run fn(scene) for all scenes using N_WORKERS threads. Returns failed scenes."""
    already_done = [s for s in scenes if is_done(s, split)]
    pending      = [s for s in scenes if not is_done(s, split)]
    failed       = []

    if already_done:
        print(f"  Skipping {len(already_done)} already-completed scene(s).", flush=True)

    bar = tqdm(
        total=len(scenes),
        initial=len(already_done),
        desc=f"  {split}",
        unit="scene",
        dynamic_ncols=True,
    )

    start = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(fn, s): s for s in pending}
        for future in as_completed(futures):
            scene = futures[future]
            try:
                future.result()
                bar.update(1)
            except Exception as exc:
                failed.append(scene)
                bar.write(f"  ✗ {scene}: {exc}", file=sys.stderr)
                bar.update(1)

    bar.close()

    elapsed = time.time() - start
    completed = len(scenes) - len(failed)
    print(f"  Done — {completed}/{len(scenes)} scenes in {elapsed/60:.1f} min", flush=True)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing", action="store_true", help="Skip training, run testing + score")
    parser.add_argument("--score",   action="store_true", help="Run scoring only")
    args = parser.parse_args()

    if not args.testing and not args.score:
        # ── Text features ──────────────────────────────────────────────────────
        print("=== [1/4] Building text features ===")
        run([PYTHON, "-m", "src.pipeline.text_features"])

        # ── Training scenes ────────────────────────────────────────────────────
        training_scenes = sorted(v.stem for v in TRAIN_VIDEOS.glob("*.avi"))
        print(f"\n=== [2/4] Training — {len(training_scenes)} scenes, {N_WORKERS} workers ===")
        print("  (frames are deleted after each scene to save ~27 GB)\n")
        failed = run_parallel(process_training_scene, training_scenes, "training")
        if failed:
            print(f"\n  Warning: {len(failed)} scene(s) failed — {failed}", file=sys.stderr)

    if not args.score:
        # ── Testing scenes ─────────────────────────────────────────────────────
        testing_scenes = sorted(d.name for d in TEST_FRAMES.iterdir() if d.is_dir())
        print(f"\n=== [3/4] Testing — {len(testing_scenes)} scenes, {N_WORKERS} workers ===\n")
        failed = run_parallel(process_testing_scene, testing_scenes, "testing")
        if failed:
            print(f"\n  Warning: {len(failed)} scene(s) failed — {failed}", file=sys.stderr)

    # ── Scoring ────────────────────────────────────────────────────────────────
    print("\n=== [4/4] Scoring ===")
    run([PYTHON, "-m", "src.pipeline.score"])

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
