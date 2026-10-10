"""Scan RLVS violence clips and rank them by violence-sound (PANNs) and loudness.

This lets us pick clips whose own audio is loud/clear (screams, impacts, shouting).

Usage:
    python scripts/scan_audio_clips.py /path/to/Violence   # folder of .mp4 clips
    RLVS_VIOLENCE_DIR=/path/to/Violence python scripts/scan_audio_clips.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import librosa
import numpy as np
from panns_inference import AudioTagging
from panns_inference import labels as audioset_labels

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline.audio_score import SAMPLE_RATE, load_target_class_indices, slice_windows  # noqa: E402

N_SAMPLE_CLIPS = 50  # spread this many clips evenly over the folder
MIN_AUDIO_SEC = 0.5  # shorter than this counts as "no audio"
SILENT_RMS = 1e-4  # RMS below this is a silent track


def clip_number(path: str) -> int:
    """Return the numeric part of a ``V_<n>.mp4`` clip name, or 0 if it has none."""
    stem = os.path.basename(path)[2:-4]
    return int(stem) if stem.isdigit() else 0


def main(argv: list[str] | None = None) -> int:
    """Rank a spread of clips by violence-sound score and print the top ones.

    Args:
        argv: Command-line arguments; ``None`` means ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success, 2 if the clip folder is not given or missing.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clips_dir", nargs="?", default=os.getenv("RLVS_VIOLENCE_DIR", ""),
                    help="folder with the RLVS Violence/*.mp4 clips (or set RLVS_VIOLENCE_DIR)")
    args = ap.parse_args(argv)
    if not args.clips_dir or not os.path.isdir(args.clips_dir):
        ap.error("clips folder missing: pass it as an argument or set RLVS_VIOLENCE_DIR")

    files = sorted(glob.glob(os.path.join(args.clips_dir, "*.mp4")), key=clip_number)
    sample = files[:: max(1, len(files) // N_SAMPLE_CLIPS)][:N_SAMPLE_CLIPS]

    tagger = AudioTagging(checkpoint_path=None, device="cpu")
    idx, _names = load_target_class_indices(list(audioset_labels))

    rows = []
    for f in sample:
        try:
            y, _ = librosa.load(f, sr=SAMPLE_RATE, mono=True)
            if len(y) < SAMPLE_RATE * MIN_AUDIO_SEC:
                continue
            rms = float(np.sqrt((y**2).mean()))
            if rms < SILENT_RMS:
                continue
            w = slice_windows(y, SAMPLE_RATE)
            cw, _ = tagger.inference(w)
            vscore = float(cw[:, idx].sum(axis=1).max())
            rows.append((os.path.basename(f), vscore, rms))
        except Exception as exc:  # one unreadable clip must not abort the scan
            print(f"skip {os.path.basename(f)}: {exc}")

    rows.sort(key=lambda r: -r[1])
    print(f"\nscanned {len(rows)} clips with audio. Top by violence-sound:")
    print(f"{'clip':12} {'violence_snd':>12} {'loudness':>10}")
    for name, vs, rms in rows[:12]:
        print(f"  {name:12} {vs:>12.3f} {rms:>10.3f}")
    print("\nSCAN_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
