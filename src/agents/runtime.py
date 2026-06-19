"""
The RUNTIME — the agent-agnostic core loop.

This is the whole orchestration, and notice it names ZERO concrete agents:
it asks the registry for whoever is registered + enabled, runs them in the
fixed pipeline order, and returns the result.

    perception agents  →  coordinator (brain)  →  action agents

Adding/removing an agent changes behaviour here without changing this code.
"""

from __future__ import annotations

from src.agents import registry
from src.agents.base import Decision, Observation
from src.agents.context import load_scene_context, peak_frame

__all__ = ["analyze_frame", "load_scene_context", "peak_frame"]


def analyze_frame(ctx: dict, frame: int) -> dict:
    """Run the full multi-agent pipeline on one frame of a loaded scene.

    Returns a JSON-ready dict: the frame, each agent's observation, the brain's
    decision, and what each action agent did.
    """
    ctx = {**ctx, "frame": frame}

    # 1 — Perception: every enabled sensing agent reads the blackboard.
    observations: list[Observation] = []
    for agent in registry.perception_agents():
        obs = agent.observe(ctx, frame)
        if obs is not None:
            observations.append(obs)

    # 2 — Coordinator (the brain): fuse observations into one decision.
    decision: Decision = registry.coordinator().decide(observations, ctx["threshold"])

    # 3 — Actions: react to the decision (alert, log, webhook…).
    actions = {
        agent.name: agent.handle(decision, observations, ctx)
        for agent in registry.action_agents()
    }

    return {
        "scene":        ctx["scene"],
        "frame":        frame,
        "observations": [o.to_dict() for o in observations],
        "decision":     decision.to_dict(),
        "actions":      actions,
    }


def analyze_peak(scene: str, split: str = "testing", threshold: float = 1.0) -> dict:
    """Convenience: load a scene and analyse its peak (most-likely-incident) frame."""
    ctx = load_scene_context(scene, split, threshold)
    return analyze_frame(ctx, peak_frame(ctx))
