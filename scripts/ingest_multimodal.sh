#!/usr/bin/env bash
# Full multimodal ingest: vision (CLIP) + sound (PANNs) + speech (Whisper) on a
# curated set of RLVS violence clips (with audio) + a calm control.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export DATASET_SPLIT=testing
V="/Users/majd/Downloads/archive/Real Life Violence Dataset/Violence"
N="/Users/majd/Downloads/archive/Real Life Violence Dataset/NonViolence"

names=(fight_1 fight_2 fight_3 calm_1)
srcs=("$V/V_1.mp4" "$V/V_151.mp4" "$V/V_251.mp4" "$N/NV_1.mp4")

echo "=== clean previous clip data ==="
for s in fight_1 fight_2 fight_3 fight_4 calm_1; do
  rm -rf shanghaitech/testing/frames/$s runtime_data/detections/testing/$s runtime_data/crops/testing/$s
  rm -f runtime_data/features/testing/$s.npz
  rm -f runtime_data/scores/testing/frames/${s}_frames.csv runtime_data/scores/testing/audio/${s}_audio.csv
  rm -f runtime_data/scores/testing/verbal/${s}_verbal.csv runtime_data/scores/testing/fused/${s}_fused.csv
  rm -f runtime_data/clips/${s}.wav runtime_data/transcripts/${s}_transcript.*
done

"$PY" -m src.pipeline.text_features 2>&1 | tail -1

for i in "${!names[@]}"; do
  scene="${names[$i]}"; clip="${srcs[$i]}"
  fps=$("$PY" -c "import cv2;c=cv2.VideoCapture('$clip');print(round(c.get(cv2.CAP_PROP_FPS) or 24,3))")
  echo "=== [$scene] (fps=$fps) ingest + vision ==="
  VIDEO_PATH="$clip" SCENE="$scene" "$PY" -m src.pipeline.ingest_clip 2>&1 | grep -E "Frames|Audio" || true
  SUBDIR_FILTER="$scene" "$PY" -m src.pipeline.detect 2>&1 | tail -1
  SUBDIR_FILTER="$scene" "$PY" -m src.pipeline.crop  2>&1 | tail -1
  SUBDIR_FILTER="$scene" "$PY" -m src.pipeline.embed 2>&1 | tail -1
  wav="runtime_data/clips/${scene}.wav"
  if [ -f "$wav" ]; then
    echo "--- [$scene] sound (PANNs) ---"
    WAV_PATH="$wav" SCENE="$scene" FPS="$fps" "$PY" -m src.pipeline.audio_score 2>&1 | tail -1
    echo "--- [$scene] speech (Whisper) ---"
    WAV_PATH="$wav" SCENE="$scene" FPS="$fps" "$PY" -m src.pipeline.transcribe 2>&1 | tail -1
  else
    echo "--- [$scene] no audio track ---"
  fi
done

echo "=== score + fuse ==="
"$PY" -m src.pipeline.score 2>&1 | tail -1
for scene in "${names[@]}"; do SCENE="$scene" "$PY" -m src.pipeline.fuse 2>&1 | tail -1; done
echo "MULTIMODAL_DONE"
