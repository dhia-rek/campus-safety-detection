"""
Step 5b — Voice recognizer: transcribe speech and flag verbal abuse.

Complements the PANNs audio scorer (step 5). Where audio_score.py detects the
*sound* of violence (screams, shouting, impacts), this step recognises the
*words* being spoken — using OpenAI Whisper — and flags bullying language
(threats, insults) listed in prompts/bad_words.txt.

It produces two things:
  1. A saved transcript file (the "save the recognized text to a file" goal):
       runtime_data/transcripts/<scene>_transcript.json   (segments + flags)
       runtime_data/transcripts/<scene>_transcript.txt    (human-readable)
  2. A per-frame verbal-abuse score CSV, aligned to the visual/audio pipeline:
       runtime_data/scores/<split>/verbal/<scene>_verbal.csv
       columns: frame, verbal_score, text

The verbal_score for a frame is the number of flagged bad words spoken in the
Whisper segment covering that frame's timestamp (0 if no speech / no match).
This CSV is fused alongside visual and sound scores in fuse.py.

Environment variables
---------------------
WAV_PATH      : path to the .wav file (required)
SCENE         : scene name, e.g. "05_0019" (required)
FPS           : video frame rate to align with (default 24)
DATASET_SPLIT : "testing" (default) or "training"
WHISPER_MODEL : Whisper size — tiny|base|small|medium|large (default "base")
WHISPER_LANG  : force a language code (e.g. "en", "fr"); default = auto-detect

Usage
-----
    WAV_PATH=clip.wav SCENE=05_0019 FPS=24 python -m src.pipeline.transcribe
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import PROMPTS_DIR, SCORES_ROOT, RUNTIME_ROOT


# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000   # Whisper expects 16 kHz mono
BAD_WORDS_FILE = PROMPTS_DIR / "bad_words.txt"

WAV_PATH      = os.getenv("WAV_PATH", "").strip()
SCENE         = os.getenv("SCENE", "").strip()
FPS           = float(os.getenv("FPS", "24"))
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "testing").strip().lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base").strip()
WHISPER_LANG  = os.getenv("WHISPER_LANG", "").strip() or None
# ─────────────────────────────────────────────────────────────────────────────


def load_bad_words() -> list[str]:
    """Read bad_words.txt (skipping comments/blank lines), lower-cased."""
    if not BAD_WORDS_FILE.is_file():
        raise FileNotFoundError(f"Bad-words list not found: {BAD_WORDS_FILE}")
    terms = []
    for line in BAD_WORDS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line.lower())
    return terms


def build_matcher(terms: list[str]) -> re.Pattern:
    """Compile one whole-word, case-insensitive regex for all terms/phrases."""
    # Sort longest-first so multi-word phrases win over their sub-words.
    parts = [re.escape(t) for t in sorted(terms, key=len, reverse=True)]
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


def find_bad_words(text: str, matcher: re.Pattern) -> list[str]:
    """Return the list of flagged terms found in *text* (may repeat)."""
    return [m.group(0).lower() for m in matcher.finditer(text)]


def main() -> None:
    if not WAV_PATH or not SCENE:
        raise RuntimeError("WAV_PATH and SCENE environment variables are required.")
    if not os.path.isfile(WAV_PATH):
        raise FileNotFoundError(f"WAV not found: {WAV_PATH}")

    import librosa
    import whisper

    terms   = load_bad_words()
    matcher = build_matcher(terms)
    print(f"Loaded {len(terms)} bad-word terms.")

    # Load audio ourselves (librosa) and hand Whisper a numpy array, so we do
    # not depend on a system ffmpeg install.
    print(f"Loading audio: {WAV_PATH}")
    audio, _ = librosa.load(WAV_PATH, sr=SAMPLE_RATE, mono=True)
    audio    = audio.astype(np.float32)
    duration = len(audio) / SAMPLE_RATE
    print(f"Duration: {duration:.2f}s")

    print(f"Loading Whisper '{WHISPER_MODEL}' (downloads on first run)…")
    model  = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio, language=WHISPER_LANG, fp16=False, verbose=False)

    # ── Build the saved transcript + per-segment flags ────────────────────────
    segments = []
    for seg in result.get("segments", []):
        text  = seg["text"].strip()
        flags = find_bad_words(text, matcher)
        segments.append({
            "start":     round(float(seg["start"]), 2),
            "end":       round(float(seg["end"]), 2),
            "text":      text,
            "bad_words": flags,
            "score":     len(flags),
        })

    n_flagged = sum(1 for s in segments if s["bad_words"])
    print(f"Transcribed {len(segments)} segments — {n_flagged} contain flagged words.")

    # ── 1. Save the transcript to a file (JSON + readable TXT) ────────────────
    tr_dir = Path(str(RUNTIME_ROOT)) / "transcripts"
    tr_dir.mkdir(parents=True, exist_ok=True)

    json_path = tr_dir / f"{SCENE}_transcript.json"
    json_path.write_text(
        json.dumps(
            {"scene": SCENE, "language": result.get("language"),
             "duration": round(duration, 2), "segments": segments},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    txt_path = tr_dir / f"{SCENE}_transcript.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for s in segments:
            mark = f"  ⚠ {s['bad_words']}" if s["bad_words"] else ""
            f.write(f"[{s['start']:>7.2f}–{s['end']:>7.2f}] {s['text']}{mark}\n")
    print(f"Saved transcript: {json_path}")
    print(f"Saved transcript: {txt_path}")

    # ── 2. Per-frame verbal-abuse score, aligned to the video FPS ─────────────
    n_frames     = int(np.ceil(duration * FPS)) or 1
    frame_idx    = np.arange(n_frames)
    frame_times  = frame_idx / FPS
    verbal_score = np.zeros(n_frames, dtype=np.float32)
    frame_text   = [""] * n_frames

    for s in segments:
        in_seg = (frame_times >= s["start"]) & (frame_times < s["end"])
        verbal_score[in_seg] = np.maximum(verbal_score[in_seg], s["score"])
        for fi in np.where(in_seg)[0]:
            frame_text[fi] = s["text"]

    out_dir = Path(str(SCORES_ROOT)) / DATASET_SPLIT / "verbal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{SCENE}_verbal.csv"
    pd.DataFrame({
        "frame":        frame_idx,
        "verbal_score": verbal_score,
        "text":         frame_text,
    }).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
