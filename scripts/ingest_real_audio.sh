#!/usr/bin/env bash
# Ingest RLVS clips that have loud REAL audio (ranked by PANNs violence-sound):
# real video + real audio, full pipeline (vision + sound + speech). No TTS.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export DATASET_SPLIT=testing
V="/Users/majd/Downloads/archive/Real Life Violence Dataset/Violence"
N="/Users/majd/Downloads/archive/Real Life Violence Dataset/NonViolence"

names=(fight_1 fight_2 fight_3 fight_4 calm_1)
srcs=("$V/V_921.mp4" "$V/V_961.mp4" "$V/V_241.mp4" "$V/V_221.mp4" "$N/NV_1.mp4")

echo "=== clean old fight/calm scenes ==="
for s in fight_1 fight_2 fight_3 fight_4 calm_1; do
  rm -rf shanghaitech/testing/frames/$s runtime_data/detections/testing/$s runtime_data/crops/testing/$s
  rm -f runtime_data/features/testing/$s.npz
  rm -f runtime_data/scores/testing/{frames,audio,verbal,fused}/${s}_*.csv
  rm -f runtime_data/clips/${s}.wav runtime_data/transcripts/${s}_transcript.*
done

"$PY" -m src.pipeline.text_features 2>&1 | tail -1

for i in "${!names[@]}"; do
  scene="${names[$i]}"; clip="${srcs[$i]}"
  fps=$("$PY" -c "import cv2;c=cv2.VideoCapture('$clip');print(round(c.get(cv2.CAP_PROP_FPS) or 24,3))")
  echo "=== [$scene] V=$(basename "$clip") fps=$fps ==="
  VIDEO_PATH="$clip" SCENE="$scene" "$PY" -m src.pipeline.ingest_clip 2>&1 | grep -E "Frames|Audio" || true
  wav="runtime_data/clips/${scene}.wav"
  # boost loudness so it's audible in the dashboard
  if [ -f "$wav" ]; then
    ffmpeg -y -i "$wav" -af "loudnorm=I=-12:TP=-1.0" -ar 32000 -ac 1 "${wav%.wav}_l.wav" 2>/dev/null && mv "${wav%.wav}_l.wav" "$wav"
  fi
  SUBDIR_FILTER="$scene" "$PY" -m src.pipeline.detect 2>&1 | tail -1
  SUBDIR_FILTER="$scene" "$PY" -m src.pipeline.crop  2>&1 | tail -1
  SUBDIR_FILTER="$scene" "$PY" -m src.pipeline.embed 2>&1 | tail -1
  if [ -f "$wav" ]; then
    WAV_PATH="$wav" SCENE="$scene" FPS="$fps" "$PY" -m src.pipeline.audio_score 2>&1 | tail -1
    WAV_PATH="$wav" SCENE="$scene" FPS="$fps" "$PY" -m src.pipeline.transcribe 2>&1 | tail -1
  fi
done

"$PY" -m src.pipeline.score 2>&1 | tail -1
for scene in "${names[@]}"; do SCENE="$scene" "$PY" -m src.pipeline.fuse >/dev/null 2>&1; done
echo "REAL_AUDIO_DONE"
