"""
Centralised path configuration for the Bullying Detection project.

All other modules import path constants from here.
Paths can be overridden via environment variables — useful for running
on a server or a different data drive without modifying source code.

Environment variables
---------------------
ECE_DATASET_ROOT  : root of the ECE campus dataset (videos + frames)
PROJECT_DATA_ROOT : root for all generated runtime artefacts
DETECTIONS_ROOT   : override for YOLO JSON outputs
CROPS_ROOT        : override for cropped object images
FEATURES_ROOT     : override for CLIP .npz feature files
SCORES_ROOT       : override for bullying score CSVs
PROMPTS_DIR       : override for normal/abnormal prompt text files
"""

from pathlib import Path
import os


# Project root is two levels above this file: src/paths.py → src/ → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    """Return the path from environment variable *name*, or *default* if unset."""
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def _default_dataset_root() -> Path:
    """Locate the ECE campus dataset folder, trying both common spellings."""
    candidates = [
        PROJECT_ROOT / "Videos_bullying" / "ece_campus",
        PROJECT_ROOT / "Videos_bullying " / "ece_campus",  # trailing space variant
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[1]


# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_ROOT = _env_path("ECE_DATASET_ROOT", _default_dataset_root())

TRAIN_FRAMES_ROOT = DATASET_ROOT / "training" / "frames"
TEST_FRAMES_ROOT  = DATASET_ROOT / "testing"  / "frames"

# ── Runtime outputs (all generated, safe to delete and re-run) ────────────────
RUNTIME_ROOT = _env_path("PROJECT_DATA_ROOT", PROJECT_ROOT / "runtime_data")

DETECTIONS_ROOT = _env_path("DETECTIONS_ROOT", RUNTIME_ROOT / "detections")
CROPS_ROOT      = _env_path("CROPS_ROOT",      RUNTIME_ROOT / "crops")
FEATURES_ROOT   = _env_path("FEATURES_ROOT",   RUNTIME_ROOT / "features")
SCORES_ROOT     = _env_path("SCORES_ROOT",     RUNTIME_ROOT / "scores")

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPTS_DIR = _env_path("PROMPTS_DIR", PROJECT_ROOT / "prompts")
