"""Pluggable multi-agent layer.

Architecture (see docs/Architecture_Agents.md):

    Perception agents  →  Coordinator (brain)  →  Action agents
    (plugins/)            (plugins/)              (plugins/)

The core never names a concrete agent. Drop a file in ``plugins/`` to add a
capability; delete it to remove one.

    from src.agents import registry, runtime
    result = runtime.analyze_peak("05_0019")
"""

from src.agents.base import (
    ActionAgent,
    Agent,
    Coordinator,
    Decision,
    Observation,
    PerceptionAgent,
)

__all__ = [
    "Agent",
    "PerceptionAgent",
    "Coordinator",
    "ActionAgent",
    "Observation",
    "Decision",
]
