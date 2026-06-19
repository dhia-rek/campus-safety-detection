"""
Telegram action agent — pushes an annotated frame to subscribers on a real
incident, gated by severity, a cooldown, and the existence of subscribers.

The low-level photo send is exposed as ``send_frame_photo`` so the dashboard's
manual "alert" button (src/dashboard/api.py) can reuse the exact same path.

Environment
-----------
TELEGRAM_TOKEN      : Telegram Bot API token (required to send)
ESCALATE_SEVERITIES : severities that trigger an automatic push (default "high")
"""

from __future__ import annotations

import json
import os
import time

import requests

from src.agents.base import ActionAgent, Decision, Observation
from src.agents.registry import register
from src.paths import TEST_FRAMES_ROOT

TELEGRAM_TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
SUBSCRIBER_FILE     = "subscribers.json"   # relative to project root (CWD)
ALERT_COOLDOWN_SEC  = 10
ESCALATE_SEVERITIES = {
    s.strip() for s in os.getenv("ESCALATE_SEVERITIES", "high").split(",") if s.strip()
}

_last_alert_ts = 0.0


def _subscribers() -> list:
    try:
        with open(SUBSCRIBER_FILE) as f:
            return json.load(f).get("subscribers", [])
    except Exception:
        return []


def _frame_path(scene: str, idx: int, split: str = "testing") -> str | None:
    frames = sorted((TEST_FRAMES_ROOT / scene).glob("*.jpg"))
    if 0 <= idx < len(frames):
        return str(frames[idx])
    return None


def send_frame_photo(scene: str, frame: int, caption: str, split: str = "testing") -> dict:
    """Push one frame image to every subscriber. Shared with the manual UI button."""
    token = os.getenv("TELEGRAM_TOKEN", TELEGRAM_TOKEN)
    if not token:
        return {"sent": 0, "skipped": "no TELEGRAM_TOKEN"}
    path = _frame_path(scene, frame, split)
    if path is None:
        return {"sent": 0, "skipped": "frame out of range"}
    subs = _subscribers()
    if not subs:
        return {"sent": 0, "skipped": "no subscribers"}

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    sent = 0
    for chat_id in subs:
        try:
            with open(path, "rb") as img:
                requests.post(
                    url,
                    data={"chat_id": chat_id, "caption": caption},
                    files={"photo": img},
                    timeout=15,
                )
            sent += 1
        except Exception as exc:
            print(f"[telegram] error for {chat_id}: {exc}")
    return {"sent": sent}


@register
class TelegramAction(ActionAgent):
    name        = "telegram"
    label       = "Telegram alert"
    tier        = "starter"
    description = "Pushes an annotated frame + the LLM report to subscribers on a real incident."

    def handle(self, decision: Decision, observations: list[Observation], ctx: dict) -> dict:
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
