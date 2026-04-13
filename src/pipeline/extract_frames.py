"""
Step 0 — Extract frames from training videos.

Reads every .avi file under DATASET_ROOT/training/videos and writes
individual JPEG frames to DATASET_ROOT/training/frames/<video_stem>/.

Environment variables
---------------------
VIDEO_FILTER : if set, process only the video whose stem matches (e.g. "01_001")
MAX_VIDEOS   : cap the number of videos processed (0 = no cap)

Usage
-----
    python -m src.pipeline.extract_frames
"""

import os
from pathlib import Path

import cv2
from tqdm import tqdm

from src.paths import DATASET_ROOT


VIDEOS_DIR  = DATASET_ROOT / "training" / "videos"
FRAMES_DIR  = DATASET_ROOT / "training" / "frames"
IMAGE_EXT   = ".jpg"
VIDEO_FILTER = os.getenv("VIDEO_FILTER", "").strip()
MAX_VIDEOS   = int(os.getenv("MAX_VIDEOS", "0") or "0")


def extract_video(video_path: Path, output_dir: Path) -> int:
    """Write every frame of *video_path* as a JPEG into *output_dir*.

    Returns the number of frames written.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    idx = written = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = output_dir / f"{idx:06d}{IMAGE_EXT}"
        if cv2.imwrite(str(frame_path), frame):
            written += 1
        idx += 1

    cap.release()
    return written


def main() -> None:
    if not VIDEOS_DIR.is_dir():
        raise FileNotFoundError(f"Training videos not found: {VIDEOS_DIR}")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(VIDEOS_DIR.glob("*.avi"))

    if VIDEO_FILTER:
        videos = [v for v in videos if v.stem == VIDEO_FILTER]
    if MAX_VIDEOS > 0:
        videos = videos[:MAX_VIDEOS]

    if not videos:
        print(f"No .avi files found under: {VIDEOS_DIR}")
        return

    print(f"Extracting {len(videos)} video(s) from: {VIDEOS_DIR}")
    total = 0
    for video in tqdm(videos, desc="Extracting training frames"):
        written = extract_video(video, FRAMES_DIR / video.stem)
        total += written

    print(f"Done. Extracted {total} frames into: {FRAMES_DIR}")


if __name__ == "__main__":
    main()
