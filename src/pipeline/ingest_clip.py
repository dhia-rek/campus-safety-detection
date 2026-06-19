"""
Step −1 — Ingest a film/video clip (.mp4, .avi, .mov…) into the pipeline.

Splits a single video file into the two aligned inputs the pipeline expects:
  1. JPEG frames        → <dataset>/<split>/frames/<scene>/000000.jpg …
  2. A mono .wav track   → runtime_data/clips/<scene>.wav

Both are sampled so that frame index t corresponds to time t / FPS — exactly
the temporal alignment the fusion step relies on. This lets you demo the full
audio + video pipeline on a movie clip (e.g. a school-bullying scene) without a
dedicated dataset.

Requires ffmpeg on PATH.

Environment variables
---------------------
VIDEO_PATH    : path to the source clip (required)
SCENE         : scene name to create, e.g. "movie01" (required)
DATASET_SPLIT : "testing" (default) or "training"
FPS           : target frame rate (default: the clip's native FPS, else 24)

Usage
-----
    VIDEO_PATH=bully_scene.mp4 SCENE=movie01 python -m src.pipeline.ingest_clip

Then run the pipeline on that scene:
    DATASET_SPLIT=testing SUBDIR_FILTER=movie01 python -m src.pipeline.detect
    DATASET_SPLIT=testing SUBDIR_FILTER=movie01 python -m src.pipeline.crop
    python -m src.pipeline.text_features
    DATASET_SPLIT=testing SUBDIR_FILTER=movie01 python -m src.pipeline.embed
    python -m src.pipeline.score
    WAV_PATH=runtime_data/clips/movie01.wav SCENE=movie01 FPS=<fps> python -m src.pipeline.audio_score
    WAV_PATH=runtime_data/clips/movie01.wav SCENE=movie01 FPS=<fps> python -m src.pipeline.transcribe
    SCENE=movie01 python -m src.pipeline.fuse
"""

import os
import shutil
import subprocess
from pathlib import Path

import cv2

from src.paths import TRAIN_FRAMES_ROOT, TEST_FRAMES_ROOT, RUNTIME_ROOT


VIDEO_PATH    = os.getenv("VIDEO_PATH", "").strip()
SCENE         = os.getenv("SCENE", "").strip()
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "testing").strip().lower()
FPS_ENV       = os.getenv("FPS", "").strip()

AUDIO_SR = 32000   # PANNs rate; librosa resamples to 16 kHz for Whisper


def native_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 0.0


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    if not VIDEO_PATH or not SCENE:
        raise SystemExit("VIDEO_PATH and SCENE are required.")
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH — install it (brew install ffmpeg).")

    fps = float(FPS_ENV) if FPS_ENV else (native_fps(VIDEO_PATH) or 24.0)
    fps = round(fps, 3)

    frames_root = TEST_FRAMES_ROOT if DATASET_SPLIT == "testing" else TRAIN_FRAMES_ROOT
    frames_dir  = Path(str(frames_root)) / SCENE
    frames_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = Path(str(RUNTIME_ROOT)) / "clips"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_dir / f"{SCENE}.wav"

    print(f"Source : {VIDEO_PATH}")
    print(f"Scene  : {SCENE}  (split: {DATASET_SPLIT})")
    print(f"FPS    : {fps}")

    # 1 — frames (0-based numbering so frame t aligns with audio time t/FPS)
    print("Extracting frames…")
    run([
        "ffmpeg", "-y", "-i", VIDEO_PATH,
        "-vf", f"fps={fps}", "-start_number", "0",
        str(frames_dir / "%06d.jpg"),
    ])
    n_frames = len(list(frames_dir.glob("*.jpg")))

    # 2 — audio track (mono)
    print("Extracting audio…")
    run([
        "ffmpeg", "-y", "-i", VIDEO_PATH,
        "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
        str(wav_path),
    ])
    has_audio = wav_path.is_file() and wav_path.stat().st_size > 1024

    print()
    print(f"✓ Frames : {n_frames} → {frames_dir}")
    print(f"{'✓' if has_audio else '✗'} Audio  : {'→ ' + str(wav_path) if has_audio else 'no audio track in clip'}")
    print()
    print("Next — run the pipeline on this scene:")
    sub = f"DATASET_SPLIT={DATASET_SPLIT} SUBDIR_FILTER={SCENE}"
    print(f"  {sub} python -m src.pipeline.detect")
    print(f"  {sub} python -m src.pipeline.crop")
    print(f"  python -m src.pipeline.text_features")
    print(f"  {sub} python -m src.pipeline.embed")
    print(f"  python -m src.pipeline.score")
    if has_audio:
        print(f"  WAV_PATH={wav_path} SCENE={SCENE} FPS={fps} python -m src.pipeline.audio_score")
        print(f"  WAV_PATH={wav_path} SCENE={SCENE} FPS={fps} python -m src.pipeline.transcribe")
        print(f"  SCENE={SCENE} python -m src.pipeline.fuse")


if __name__ == "__main__":
    main()
