"""Backward-compatibility shim.

The perception agents moved into ``src/agents/plugins/`` (vision.py, sound.py,
speech.py) under the registry. The scene-context helpers moved to
``src/agents/context.py``. This module re-exports the old names so existing
imports keep working.

Prefer the new entry points:
    from src.agents.context import load_scene_context, peak_frame
    from src.agents.runtime import analyze_frame
"""

from __future__ import annotations

from src.agents.base import Observation
from src.agents.context import frame_value as _val  # noqa: F401  (legacy name)
from src.agents.context import load_scene_context, peak_frame
from src.agents import registry

__all__ = ["load_scene_context", "peak_frame", "observe_frame", "Observation"]


def observe_frame(ctx: dict, frame: int) -> list[Observation]:
    """Run all enabled perception agents on one frame (legacy helper)."""
    out = []
    for agent in registry.perception_agents():
        obs = agent.observe(ctx, frame)
        if obs is not None:
            out.append(obs)
    return out
