#!/usr/bin/env bash
# Incrementally score + fuse scenes as the batch embeds them, so they appear in
# the dashboard one by one instead of all at the end. Scoring is cheap (reads
# precomputed image features), so we re-run it periodically and fuse any scene
# that has a frames CSV but not yet a fused CSV. Exits after the batch finishes.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FR=runtime_data/scores/testing/frames
FU=runtime_data/scores/testing/fused

pass() {
  "$PY" -m src.pipeline.score >/dev/null 2>&1 || true
  for f in "$FR"/*_frames.csv; do
    [ -e "$f" ] || continue
    scene=$(basename "$f" _frames.csv)
    [ -f "$FU/${scene}_fused.csv" ] && continue
    SCENE="$scene" DATASET_SPLIT=testing "$PY" -m src.pipeline.fuse >/dev/null 2>&1 || true
  done
}

while pgrep -f run_batch_pipeline >/dev/null 2>&1; do
  pass
  ready=$(ls "$FU"/*_fused.csv 2>/dev/null | wc -l | tr -d ' ')
  echo "incremental: ${ready} scenes fused so far"
  sleep 90
done

# final pass after the batch is fully done
pass
ready=$(ls "$FU"/*_fused.csv 2>/dev/null | wc -l | tr -d ' ')
echo "ALL_FUSED_DONE: ${ready} scenes ready"
