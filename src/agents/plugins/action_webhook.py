"""
Webhook action agent — automation hook (n8n / Make / Zapier).

When an incident reaches one of ESCALATE_SEVERITIES, the decision is POSTed to
INCIDENT_WEBHOOK_URL so a no-code workflow tool can fan out the response
(Telegram + e-mail + incident log + escalation) without changing this code.

No-ops safely when no webhook is configured or severity is below threshold.

Environment
-----------
INCIDENT_WEBHOOK_URL : workflow webhook (empty = disabled)
ESCALATE_SEVERITIES  : comma list of severities that fire it (default "high")
"""

from __future__ import annotations

import os

import requests

from src.agents.base import ActionAgent, Decision, Observation
from src.agents.registry import register

INCIDENT_WEBHOOK_URL = os.getenv("INCIDENT_WEBHOOK_URL", "").strip()
ESCALATE_SEVERITIES  = {
    s.strip() for s in os.getenv("ESCALATE_SEVERITIES", "high").split(",") if s.strip()
}


@register
class WebhookAction(ActionAgent):
    name        = "webhook"
    label       = "Automation webhook (n8n)"
    tier        = "pro"
    description = "POSTs high-severity incidents to an external no-code automation flow."

    def handle(self, decision: Decision, observations: list[Observation], ctx: dict) -> dict:
        if not INCIDENT_WEBHOOK_URL:
            return {"dispatched": False, "reason": "no INCIDENT_WEBHOOK_URL set"}
        if decision.severity not in ESCALATE_SEVERITIES:
            return {"dispatched": False, "reason": f"severity '{decision.severity}' below threshold"}

        payload = {
            "event":        "campus_safety_incident",
            "scene":        ctx.get("scene", ""),
            "frame":        ctx.get("frame", -1),
            "decision":     decision.to_dict(),
            "observations": [o.to_dict() for o in observations],
        }
        try:
            requests.post(INCIDENT_WEBHOOK_URL, json=payload, timeout=10)
            return {"dispatched": True}
        except Exception as exc:
            print(f"[webhook] automation webhook failed: {exc}")
            return {"dispatched": False, "reason": str(exc)}
