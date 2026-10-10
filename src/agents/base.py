"""The agent CONTRACT — the small, stable core every plugin agent implements.

The multi-agent layer has three roles, and nothing in the core ever names a
concrete agent — it only depends on these abstract base classes:

    PerceptionAgent  — senses ONE modality of a frame  → Observation
    Coordinator      — fuses all observations (the brain) → Decision
    ActionAgent      — reacts to a Decision (alert, log, webhook) → side-effect

Concrete agents live in ``src/agents/plugins/`` and register themselves with the
registry. Add a file to add a capability; delete the file to remove it — the
core (registry + runtime + API) is never touched.

Each agent can be toggled at runtime without deleting it, via an environment
variable ``AGENT_<NAME>_ENABLED=0`` (default: enabled). This is the seam the
per-client manifest plugs into for productisation.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


# ── Data carried between the roles ───────────────────────────────────────────
@dataclass
class Observation:
    """One perception agent's read of a single video frame."""
    agent:     str          # "vision" | "sound" | "speech"
    modality:  str          # human label, e.g. "Visual (CLIP+YOLO)"
    score:     float        # z-scored signal strength at this frame
    triggered: bool         # did it cross the alert threshold?
    detail:    str = ""     # free-text context (e.g. the transcribed words)

    def to_dict(self) -> dict:
        """Return the observation as a plain JSON-serialisable dict."""
        return asdict(self)


@dataclass
class Decision:
    """The coordinator agent's final judgement for a frame."""
    is_incident: bool
    severity:    str               # "none" | "low" | "medium" | "high"
    modalities:  list[str] = field(default_factory=list)
    reasoning:   str = ""
    action:      str = ""
    report:      str = ""
    source:      str = "llm"       # "llm" or "heuristic" (fallback)

    def to_dict(self) -> dict:
        """Return the decision as a plain JSON-serialisable dict."""
        return asdict(self)


# ── The base agent (shared metadata + enable/disable seam) ────────────────────
def _env_enabled(name: str) -> bool:
    """Return whether ``AGENT_<NAME>_ENABLED`` leaves the agent on.

    Args:
        name: Agent name (upper-cased to build the variable name).

    Returns:
        True unless the variable is set to 0/false/no/off (case-insensitive);
        an unset variable means enabled.
    """
    raw = os.getenv(f"AGENT_{name.upper()}_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class Agent(ABC):  # noqa: B024  (metadata base; subclasses add the abstract method)
    """Common metadata for every agent — used by the registry and the UI grid."""

    name:        str = "agent"     # unique id, also the env-toggle key
    label:       str = ""          # human-friendly name for the dashboard
    role:        str = "perception"  # "perception" | "coordinator" | "action"
    tier:        str = "starter"   # "starter" | "pro" | "enterprise" (sellable unit)
    description: str = ""          # one line, shown in the admin catalog

    @property
    def enabled(self) -> bool:
        """Whether this agent runs. Toggle with AGENT_<NAME>_ENABLED=0."""
        return _env_enabled(self.name)

    def info(self) -> dict:
        """Self-description for /api/agents and the admin catalog."""
        return {
            "name":        self.name,
            "label":       self.label or self.name,
            "role":        self.role,
            "tier":        self.tier,
            "description": self.description,
            "enabled":     self.enabled,
        }


class PerceptionAgent(Agent):
    """Reads one modality's pipeline output → emits an Observation (or None)."""
    role = "perception"

    @abstractmethod
    def observe(self, ctx: dict, frame: int) -> Observation | None:
        """Read this agent's modality at *frame*.

        Args:
            ctx: Shared scene context (see ``src.agents.context``).
            frame: Frame index to inspect.

        Returns:
            An Observation, or None if the agent has nothing to report.
        """
        ...


class Coordinator(Agent):
    """The brain — fuses all observations into a single Decision."""
    role = "coordinator"

    @abstractmethod
    def decide(self, observations: list[Observation], threshold: float = 1.0) -> Decision:
        """Fuse all observations into one Decision.

        Args:
            observations: Observations from every enabled perception agent.
            threshold: Z-score alert threshold the observations were judged against.

        Returns:
            The final Decision for the frame.
        """
        ...


class ActionAgent(Agent):
    """Reacts to a Decision (Telegram, e-mail, log, webhook…)."""
    role = "action"

    @abstractmethod
    def handle(self, decision: Decision, observations: list[Observation], ctx: dict) -> dict:
        """React to a Decision (send an alert, call a webhook, ...).

        Args:
            decision: The coordinator's judgement.
            observations: The observations it was based on.
            ctx: Shared scene context (``scene``, ``frame``, ``split``...).

        Returns:
            A JSON-ready dict describing what the agent did or why it skipped.
        """
        ...
