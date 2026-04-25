#!/usr/bin/env bash
# run_pipeline.sh — end-to-end bullying-detection pipeline runner
#
# Usage
# -----
#   bash scripts/run_pipeline.sh quick   # single test scene (~2-3 min)
#   bash scripts/run_pipeline.sh full    # full dataset (hours)
#
# Environment variables
# ---------------------
#   PYTHON_BIN : Python executable to use (default: .venv/bin/python)
#   SCENE      : scene name for quick mode (default: 05_0019)
set -euo pipefail

PY="${PYTHON_BIN:-.venv/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "Python executable not found: $PY"
  echo "Set PYTHON_BIN or activate your virtual environment first."
  exit 1
fi

MODE="${1:-quick}"
SCENE="${SCENE:-05_0019}"

run_quick() {
  echo "=== Quick run — scene: $SCENE ==="

  echo "[1/5] Build text features"
  "$PY" -m src.pipeline.text_features

  echo "[2/5] Detect objects (testing / $SCENE)"
  DATASET_SPLIT=testing SUBDIR_FILTER="$SCENE" "$PY" -m src.pipeline.detect

  echo "[3/5] Crop objects"
  DATASET_SPLIT=testing SUBDIR_FILTER="$SCENE" "$PY" -m src.pipeline.crop

  echo "[4/5] Embed crops with CLIP"
  DATASET_SPLIT=testing SUBDIR_FILTER="$SCENE" "$PY" -m src.pipeline.embed

  echo "[5/5] Score bullying timeline"
  "$PY" -m src.pipeline.score

  echo "Done. Scores CSV → runtime_data/scores/testing/frames/${SCENE}_frames.csv"
}

run_full() {
  echo "=== Full run ==="

  echo "[1/7] Build text features"
  "$PY" -m src.pipeline.text_features

  echo "[2/7] Extract training frames"
  "$PY" -m src.pipeline.extract_frames

  echo "[3/7] Detect objects — training"
  DATASET_SPLIT=training "$PY" -m src.pipeline.detect

  echo "[4/7] Crop objects — training"
  DATASET_SPLIT=training "$PY" -m src.pipeline.crop

  echo "[5/7] Embed crops — training"
  DATASET_SPLIT=training "$PY" -m src.pipeline.embed

  echo "[6/7] Detect / crop / embed — testing"
  DATASET_SPLIT=testing "$PY" -m src.pipeline.detect
  DATASET_SPLIT=testing "$PY" -m src.pipeline.crop
  DATASET_SPLIT=testing "$PY" -m src.pipeline.embed

  echo "[7/7] Final scoring (training calibration + testing)"
  "$PY" -m src.pipeline.score

  echo "Full run complete."
}

case "$MODE" in
  quick) run_quick ;;
  full)  run_full  ;;
  *)
    echo "Usage: bash scripts/run_pipeline.sh [quick|full]"
    exit 1
    ;;
esac
