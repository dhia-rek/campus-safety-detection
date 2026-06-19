#!/usr/bin/env bash
# Ingest a few RLVS clips (real fights + a calm control) and run them through the
# visual pipeline so they appear on the dashboard as violence-detection demos.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export DATASET_SPLIT=testing

V="/Users/majd/Downloads/archive/Real Life Violence Dataset/Violence"
N="/Users/majd/Downloads/archive/Real Life Violence Dataset/NonViolence"

# scene_name : source clip
names=(fight_1 fight_2 fight_3 calm_1)
paths=("$V/V_1.mp4" "$V/V_2.mp4" "$V/V_11.mp4" "$N/NV_1.mp4")

echo "=== [1] text features (violence prompts) ==="
"$PY" -m src.pipeline.text_features 2>&1 | tail -1

for i in "${!names[@]}"; do
  scene="${names[$i]}"; clip="${paths[$i]}"
  echo "=== [$scene] ingest + detect + crop + embed ==="
  VIDEO_PATH="$clip" SCENE="$scene" "$PY" -m src.pipeline.ingest_clip 2>&1 | grep -E "Frames|Audio" || true
  SUBDIR_FILTER="$scene" "$PY" -m src.pipeline.detect 2>&1 | tail -1
  SUBDIR_FILTER="$scene" "$PY" -m src.pipeline.crop   2>&1 | tail -1
  SUBDIR_FILTER="$scene" "$PY" -m src.pipeline.embed  2>&1 | tail -1
done

echo "=== [scoring all] ==="
"$PY" -m src.pipeline.score 2>&1 | tail -1
for scene in "${names[@]}"; do
  SCENE="$scene" "$PY" -m src.pipeline.fuse >/dev/null 2>&1 || true
done
echo "VIOLENCE_CLIPS_DONE"
