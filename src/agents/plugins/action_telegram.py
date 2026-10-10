"""Telegram action agent: push an annotated frame to subscribers on a real incident.

Gated by severity, a cooldown, and the existence of subscribers.

The low-level photo send is exposed as ``send_frame_photo`` so the dashboard's
manual "alert" button (src/dashboard/api.py) can reuse the exact same path.

Environment
-----------
TELEGRAM_TOKEN      : Telegram Bot API token (required to send; DEMO_MODE=1 uses a mock instead)
ESCALATE_SEVERITIES : severities that trigger an automatic push (default "high")
"""

from __future__ import annotations

import os
import time

from src.agents.base import ActionAgent, Decision, Observation
from src.agents.registry import register
from src.alerts.notifier import get_notifier
from src.paths import TEST_FRAMES_ROOT

ALERT_COOLDOWN_SEC  = 10
ESCALATE_SEVERITIES = {
    s.strip() for s in os.getenv("ESCALATE_SEVERITIES", "high").split(",") if s.strip()
}

_last_alert_ts = 0.0


def _frame_path(scene: str, idx: int, split: str = "testing") -> str | None:
    """Return the path of the *idx*-th frame (sorted by name) of a scene.

    Args:
        scene: Scene id.
        idx: Zero-based position in the sorted frame list.
        split: Unused; frames are always read from the testing frames root.

    Returns:
        The JPEG path, or None if *idx* is out of range.
    """
    frames = sorted((TEST_FRAMES_ROOT / scene).glob("*.jpg"))
    if 0 <= idx < len(frames):
        return str(frames[idx])
    return None


def send_frame_photo(scene: str, frame: int, caption: str, split: str = "testing") -> dict:
    """Push one frame image to every subscriber. Shared with the manual UI button.

    Args:
        scene: Scene id.
        frame: Zero-based frame position.
        caption: Photo caption.
        split: Dataset split (passed through; see ``_frame_path``).

    Returns:
        The notifier's result dict, or ``{"sent": 0, "skipped": reason}`` when
        no token is configured or the frame is out of range.
    """
    notifier = get_notifier()
    if not notifier.configured:
        return {"sent": 0, "skipped": "no TELEGRAM_TOKEN"}
    path = _frame_path(scene, frame, split)
    if path is None:
        return {"sent": 0, "skipped": "frame out of range"}
    return notifier.send_photo(path, caption)


@register
class TelegramAction(ActionAgent):
    """Action agent that pushes a frame photo for high-severity incidents."""

    name        = "telegram"
    label       = "Telegram alert"
    tier        = "starter"
    description = "Pushes an annotated frame + the LLM report to subscribers on a real incident."

    def handle(self, decision: Decision, observations: list[Observation], ctx: dict) -> dict:
        """Send a Telegram photo alert if the incident is severe enough.

        Skipped when the decision is not an incident, its severity is not in
        ``ESCALATE_SEVERITIES``, or the last successful alert was less than
        ``ALERT_COOLDOWN_SEC`` ago. The cooldown only restarts after a send
        that actually reached someone, so failures can be retried at once.

        Returns:
            The send result, or ``{"sent": 0, "skipped": reason}``.
        """
        global _last_alert_ts
        if not decision.is_incident or decision.severity not in ESCALATE_SEVERITIES:
            return {"sent": 0, "skipped": "below escalation threshold"}

        now = time.time()
        if now - _last_alert_ts < ALERT_COOLDOWN_SEC:
            return {"sent": 0, "skipped": "cooldown"}

        caption = (
            "🚨 CCTV ANOMALY ALERT 🚨\n\n"
            f"Scene: {ctx.get('scene', '?')}\n"
            f"Frame: {ctx.get('frame', -1)}\n"
            f"Severity: {decision.severity}\n"
            f"{decision.report}"
        )
        result = send_frame_photo(ctx.get("scene", ""), ctx.get("frame", -1), caption, ctx.get("split", "testing"))
        if result.get("sent"):
            _last_alert_ts = now
        return result
