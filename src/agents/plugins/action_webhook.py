"""Webhook action agent — automation hook (n8n / Make / Zapier).

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
    """Action agent that POSTs escalated incidents to a workflow webhook."""

    name        = "webhook"
    label       = "Automation webhook (n8n)"
    tier        = "pro"
    description = "POSTs high-severity incidents to an external no-code automation flow."

    def handle(self, decision: Decision, observations: list[Observation], ctx: dict) -> dict:
        """POST the decision and observations to ``INCIDENT_WEBHOOK_URL``.

        Returns:
            ``{"dispatched": True}`` on success, otherwise ``{"dispatched": False,
            "reason": ...}`` (unset URL, severity not escalated, or request
            error). A non-2xx HTTP response counts as a failure: it is logged and
            not reported as dispatched.
        """
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
            resp = requests.post(INCIDENT_WEBHOOK_URL, json=payload, timeout=10)
            resp.raise_for_status()
            return {"dispatched": True}
        except requests.RequestException as exc:  # includes HTTPError from raise_for_status()
            print(f"[webhook] automation webhook failed: {exc}")
            return {"dispatched": False, "reason": str(exc)}
