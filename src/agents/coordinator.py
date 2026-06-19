"""CLI demo for the multi-agent layer (kept for backward compatibility).

The coordinator agent itself now lives in
``src/agents/plugins/coordinator_llm.py`` under the registry. This module is a
thin command-line entry point that runs the full pipeline on a scene:

    python -m src.agents.coordinator 05_0019
"""

from __future__ import annotations

import json
import os
import sys

from src.agents.context import load_scene_context, peak_frame
from src.agents.runtime import analyze_frame


def main() -> None:
    scene = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SCENE", "")
    if not scene:
        raise SystemExit("Usage: python -m src.agents.coordinator <scene>")

    ctx    = load_scene_context(scene)
    frame  = peak_frame(ctx)
    result = analyze_frame(ctx, frame)

    print(f"Scene {scene} — peak frame {frame}")
    for o in result["observations"]:
        flag = "⚠" if o["triggered"] else " "
        print(f"  {o['modality']}: z={o['score']:.2f} {flag} {o['detail']}")

    print("\n── Coordinator decision ──")
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))

    print(f"\n── Action agents ── {result['actions']}")


if __name__ == "__main__":
    main()
