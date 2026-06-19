"""Scan RLVS violence clips and rank by REAL violence-sound (PANNs) + loudness,
so we can pick clips whose own audio is loud/clear (screams, impacts, shouting)."""
import glob, os
import numpy as np
import librosa
from panns_inference import AudioTagging, labels as audioset_labels
from src.pipeline.audio_score import slice_windows, load_target_class_indices, SAMPLE_RATE

V = "/Users/majd/Downloads/archive/Real Life Violence Dataset/Violence"
files = sorted(glob.glob(os.path.join(V, "*.mp4")),
               key=lambda p: int(os.path.basename(p)[2:-4]) if os.path.basename(p)[2:-4].isdigit() else 0)
sample = files[::max(1, len(files)//50)][:50]   # spread of ~50 clips

tagger = AudioTagging(checkpoint_path=None, device="cpu")
idx, names = load_target_class_indices(list(audioset_labels))

rows = []
for f in sample:
    try:
        y, _ = librosa.load(f, sr=SAMPLE_RATE, mono=True)
        if len(y) < SAMPLE_RATE * 0.5:          # <0.5s of audio = effectively none
            continue
        rms = float(np.sqrt((y ** 2).mean()))
        if rms < 1e-4:                          # silent track
            continue
        w = slice_windows(y, SAMPLE_RATE)
        cw, _ = tagger.inference(w)
        vscore = float(cw[:, idx].sum(axis=1).max())
        rows.append((os.path.basename(f), vscore, rms))
    except Exception:
        continue

rows.sort(key=lambda r: -r[1])
print(f"\nscanned {len(rows)} clips with audio. Top by violence-sound:")
print(f"{'clip':12} {'violence_snd':>12} {'loudness':>10}")
for name, vs, rms in rows[:12]:
    print(f"  {name:12} {vs:>12.3f} {rms:>10.3f}")
print("\nSCAN_DONE")
