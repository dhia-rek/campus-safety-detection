"""The MICROKERNEL — agent registration + auto-discovery.

This is the only "magic" in the system, and it is tiny:

  • Concrete agents in ``plugins/`` decorate their class with ``@register``.
  • ``discover()`` imports every module in ``plugins/`` once, so simply dropping
    a file there registers its agent — no edit to any core file.
  • The core (runtime, API) asks this registry for agents by ROLE; it never
    imports a concrete agent.

Add a capability  →  add a plugin file.
Remove one        →  delete the file (or set AGENT_<NAME>_ENABLED=0).
"""

from __future__ import annotations

import importlib
import os
import pkgutil
from typing import cast

from src.agents.base import ActionAgent, Agent, Coordinator, PerceptionAgent

# role → {name → class}
_CLASSES: dict[str, dict[str, type[Agent]]] = {
    "perception":  {},
    "coordinator": {},
    "action":      {},
}
_discovered = False


def register(cls: type[Agent]) -> type[Agent]:
    """Class decorator: add an agent to the registry under its declared role.

    Args:
        cls: Agent subclass with a valid ``role`` and a non-empty ``name``.
            A later registration with the same role and name replaces the earlier one.

    Returns:
        *cls* unchanged, so it can be used as ``@register``.

    Raises:
        ValueError: If the role is unknown or the name is empty.
    """
    role = getattr(cls, "role", None)
    if role not in _CLASSES:
        raise ValueError(f"{cls.__name__}: unknown role {role!r} (perception|coordinator|action)")
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__}: agent must define a unique `name`")
    _CLASSES[role][cls.name] = cls
    return cls


def discover(force: bool = False) -> None:
    """Import every module under plugins/ so each agent self-registers.

    Args:
        force: Re-run the imports even if discovery already happened. Modules
            already in ``sys.modules`` are not re-executed, so this is cheap.
    """
    global _discovered
    if _discovered and not force:
        return
    from src.agents import plugins
    for mod in pkgutil.iter_modules(plugins.__path__):
        importlib.import_module(f"{plugins.__name__}.{mod.name}")
    _discovered = True


def _instances(role: str, only_enabled: bool = True) -> list[Agent]:
    """Instantiate the registered agents of one role.

    Args:
        role: ``"perception"``, ``"coordinator"`` or ``"action"``.
        only_enabled: Skip agents switched off via ``AGENT_<NAME>_ENABLED``.

    Returns:
        New agent instances (agents are created fresh on every call).
    """
    discover()
    out = []
    for cls in _CLASSES[role].values():
        inst = cls()
        if not only_enabled or inst.enabled:
            out.append(inst)
    return out


def perception_agents(only_enabled: bool = True) -> list[PerceptionAgent]:
    """Return instances of the perception agents (enabled only by default)."""
    return cast("list[PerceptionAgent]", _instances("perception", only_enabled))


def action_agents(only_enabled: bool = True) -> list[ActionAgent]:
    """Return instances of the action agents (enabled only by default)."""
    return cast("list[ActionAgent]", _instances("action", only_enabled))


def coordinator() -> Coordinator:
    """The active brain. Pick one with COORDINATOR_AGENT=<name>, else first enabled.

    An unknown or disabled ``COORDINATOR_AGENT`` silently falls back to the
    first enabled coordinator.

    Raises:
        RuntimeError: If no coordinator is registered and enabled.
    """
    insts = cast("list[Coordinator]", _instances("coordinator"))
    if not insts:
        raise RuntimeError("No coordinator agent registered/enabled.")
    preferred = os.getenv("COORDINATOR_AGENT", "").strip()
    if preferred:
        for inst in insts:
            if inst.name == preferred:
                return inst
    return insts[0]


def catalog() -> list[dict]:
    """Full installed catalog (including disabled agents) for the admin UI.

    Returns:
        One ``Agent.info()`` dict per registered agent, ordered by role
        (perception, coordinator, action).
    """
    discover()
    rows = []
    for role in ("perception", "coordinator", "action"):
        for cls in _CLASSES[role].values():
            rows.append(cls().info())
    return rows
