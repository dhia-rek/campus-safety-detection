"""Shared scene context — the BLACKBOARD the agents read from.

Perception agents do NOT re-run the heavy models; they consume the per-frame
CSVs and transcripts the pipeline already produced (fuse.py, transcribe.py).
This module loads that shared state once so every agent reads the same picture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.paths import RUNTIME_ROOT, SCORES_ROOT


def load_scene_context(scene: str, split: str = "testing", threshold: float = 1.0) -> dict:
    """Load the fused score CSV + transcript once, shared by all agents.

    The transcript is optional: a scene without one simply has no speech segments.

    Args:
        scene: Scene id, e.g. ``"05_0019"``.
        split: Dataset split under the scores root (default ``"testing"``).
        threshold: Z-score alert threshold stored for the agents to compare against.

    Returns:
        The blackboard dict with keys ``scene``, ``split``, ``df`` (fused scores
        indexed by frame), ``segments``, ``threshold`` and ``frame`` (None until
        the runtime sets it).

    Raises:
        FileNotFoundError: If the fused CSV for the scene does not exist.
    """
    fused_csv = Path(str(SCORES_ROOT)) / split / "fused" / f"{scene}_fused.csv"
    if not fused_csv.is_file():
        raise FileNotFoundError(f"Fused scores not found: {fused_csv}. Run fuse.py first.")

    df = pd.read_csv(fused_csv).set_index("frame")

    transcript_path = Path(str(RUNTIME_ROOT)) / "transcripts" / f"{scene}_transcript.json"
    segments = []
    if transcript_path.is_file():
        segments = json.loads(transcript_path.read_text(encoding="utf-8")).get("segments", [])

    return {
        "scene":     scene,
        "split":     split,
        "df":        df,
        "segments":  segments,
        "threshold": threshold,
        "frame":     None,   # filled in by the runtime for the frame under analysis
    }


def frame_value(ctx: dict, frame: int, col: str) -> float:
    """Read one signal column at one frame from the blackboard (0.0 if absent).

    Args:
        ctx: Scene context from ``load_scene_context``.
        frame: Frame index (a label in the dataframe index, not a position).
        col: Column name, e.g. ``"visual_z"``.

    Returns:
        The value as a float, or 0.0 when the column or frame is missing.
    """
    df = ctx["df"]
    if col in df.columns and frame in df.index:
        return float(df.at[frame, col])
    return 0.0


def peak_frame(ctx: dict) -> int:
    """The frame with the highest fused score — the most likely incident.

    Args:
        ctx: Scene context from ``load_scene_context``.

    Returns:
        Frame index of the max ``fused_score``; if that column is missing, the
        smallest frame index; 0 if the dataframe is empty.
    """
    df = ctx["df"]
    if "fused_score" in df.columns and len(df):
        return int(df["fused_score"].idxmax())
    return int(df.index.min()) if len(df) else 0
