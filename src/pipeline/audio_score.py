"""
Step 5 — Score audio for violence using pretrained PANNs CNN14.

Reads a .wav file, runs PANNs in 1-second windows (0.5s hop), sums the
probabilities of violence-related AudioSet classes listed in
prompts/violence_audio_classes.txt, and writes a per-frame CSV aligned
with the visual pipeline.

Environment variables
---------------------
WAV_PATH      : path to the .wav file (required)
SCENE         : scene name, e.g. "05_0019" (required)
FPS           : video frame rate to align with (default 24)
DATASET_SPLIT : "testing" (default) or "training"

Usage
-----
    WAV_PATH=clip.wav SCENE=05_0019 python -m src.pipeline.audio_score
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import PROMPTS_DIR, SCORES_ROOT


SAMPLE_RATE = 32000   # PANNs CNN14 expects 32 kHz mono
WIN_SEC     = 1.0
HOP_SEC     = 0.5
BATCH_SIZE  = 32
CLASS_FILE  = PROMPTS_DIR / "violence_audio_classes.txt"

WAV_PATH      = os.getenv("WAV_PATH", "").strip()
SCENE         = os.getenv("SCENE", "").strip()
FPS           = float(os.getenv("FPS", "24"))
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "testing").strip().lower()


def load_target_class_indices(label_list):
    """Map names in violence_audio_classes.txt to PANNs label indices."""
    wanted = [l.strip() for l in CLASS_FILE.read_text().splitlines() if l.strip()]
    name_to_idx = {name: i for i, name in enumerate(label_list)}
    matched, missing = [], []
    for w in wanted:
        if w in name_to_idx:
            matched.append(name_to_idx[w])
        else:
            missing.append(w)
    if missing:
        print(f"Warning — these class names are not in AudioSet labels: {missing}")
    return matched, [label_list[i] for i in matched]


def slice_windows(audio, sr):
    """Slice mono audio into (N, win_samples) with HOP_SEC stride; tail-pad with zeros."""
    win = int(WIN_SEC * sr)
    hop = int(HOP_SEC * sr)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    n = 1 + (len(audio) - win) // hop
    out = np.zeros((n, win), dtype=np.float32)
    for i in range(n):
        s = i * hop
        out[i] = audio[s : s + win]
    return out


def main():
    if not WAV_PATH or not SCENE:
        raise RuntimeError("WAV_PATH and SCENE environment variables are required.")
    if not os.path.isfile(WAV_PATH):
        raise FileNotFoundError(f"WAV not found: {WAV_PATH}")

    import librosa
    from panns_inference import AudioTagging, labels as audioset_labels

    print(f"Loading audio: {WAV_PATH}")
    audio, _ = librosa.load(WAV_PATH, sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / SAMPLE_RATE
    print(f"Duration: {duration:.2f}s")

    windows = slice_windows(audio, SAMPLE_RATE)
    print(f"Windows: {len(windows)} × {WIN_SEC}s (hop {HOP_SEC}s)")

    print("Loading PANNs CNN14 (downloads ~318 MB checkpoint on first run)…")
    tagger = AudioTagging(checkpoint_path=None, device="cpu")

    clipwise_all = []
    for i in range(0, len(windows), BATCH_SIZE):
        clipwise, _ = tagger.inference(windows[i : i + BATCH_SIZE])
        clipwise_all.append(clipwise)
    clipwise = np.concatenate(clipwise_all, axis=0)   # (N, 527)

    matched_idx, matched_names = load_target_class_indices(list(audioset_labels))
    if not matched_idx:
        raise RuntimeError(
            "No violence-class names matched AudioSet labels — "
            "check prompts/violence_audio_classes.txt"
        )
    print(f"Violence classes matched: {matched_names}")

    window_scores = clipwise[:, matched_idx].sum(axis=1)   # (N,)

    # Map each video frame to the window whose centre is closest in time.
    n_frames    = int(np.ceil(duration * FPS))
    centres     = np.arange(len(windows)) * HOP_SEC + WIN_SEC / 2
    frame_idx   = np.arange(n_frames)
    frame_times = frame_idx / FPS
    win_for_frame = np.abs(frame_times[:, None] - centres[None, :]).argmin(axis=1)
    frame_scores  = window_scores[win_for_frame]

    out_dir = Path(str(SCORES_ROOT)) / DATASET_SPLIT / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{SCENE}_audio.csv"
    pd.DataFrame({"frame": frame_idx, "audio_score": frame_scores}).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
