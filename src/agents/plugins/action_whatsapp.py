"""
WhatsApp action agent — sends a WhatsApp message on a real incident, via the free
CallMeBot API (no paid account needed; it only messages YOUR own registered number).

Drop-in action plugin: adding this file adds a "whatsapp" capability to the runtime
with no core change. No-ops safely when not configured.

One-time CallMeBot set-up (2 minutes)
-------------------------------------
1. Add the CallMeBot number to your phone contacts:  +34 644 51 95 23
2. Send it this exact WhatsApp message:  "I allow callmebot to send me messages"
3. You receive an API key by reply. Put your phone + key in the env vars below.
   (Your phone must include the country code, e.g. +33XXXXXXXXX.)

Environment
-----------
WHATSAPP_PHONE     : your number with country code, e.g. +33612345678 (required)
CALLMEBOT_APIKEY   : the key CallMeBot replied with (required)
WHATSAPP_SEVERITIES: severities that fire it (default "low,medium,high" = any incident)
"""

from __future__ import annotations

import os
import time
import urllib.parse

import requests

from src.agents.base import ActionAgent, Decision, Observation
from src.agents.registry import register

WHATSAPP_PHONE     = os.getenv("WHATSAPP_PHONE", "").strip()
CALLMEBOT_APIKEY   = os.getenv("CALLMEBOT_APIKEY", "").strip()
WHATSAPP_SEVERITIES = {
    s.strip() for s in os.getenv("WHATSAPP_SEVERITIES", "low,medium,high").split(",") if s.strip()
}
COOLDOWN_SEC = 30

_last_wa_ts = 0.0


def _send_whatsapp(text: str) -> dict:
    if not (WHATSAPP_PHONE and CALLMEBOT_APIKEY):
        return {"sent": False, "skipped": "WHATSAPP_PHONE/CALLMEBOT_APIKEY not set"}
    url = (
        "https://api.callmebot.com/whatsapp.php"
        f"?phone={urllib.parse.quote(WHATSAPP_PHONE)}"
        f"&text={urllib.parse.quote(text)}"
        f"&apikey={urllib.parse.quote(CALLMEBOT_APIKEY)}"
    )
    try:
        r = requests.get(url, timeout=20)
        return {"sent": r.ok, "status": r.status_code}
    except Exception as exc:
        print(f"[whatsapp] send failed: {exc}")
        return {"sent": False, "reason": str(exc)}


@register
class WhatsAppAction(ActionAgent):
    name        = "whatsapp"
    label       = "WhatsApp alert"
    tier        = "starter"
    description = "Sends a WhatsApp message to your number on a real incident (via CallMeBot)."

    def handle(self, decision: Decision, observations: list[Observation], ctx: dict) -> dict:
        global _last_wa_ts
        if not decision.is_incident or decision.severity not in WHATSAPP_SEVERITIES:
            return {"sent": False, "skipped": "no incident / below threshold"}

        now = time.time()
        if now - _last_wa_ts < COOLDOWN_SEC:
            return {"sent": False, "skipped": "cooldown"}

        scene = ctx.get("scene", "?")
        frame = ctx.get("frame", -1)
        text = (
            "🚨 Campus Safety AI — ANOMALY ALERT\n"
            f"Scene: {scene} · Frame: {frame}\n"
            f"Severity: {decision.severity}\n"
            f"{decision.report}"
        )
        result = _send_whatsapp(text)
        if result.get("sent"):
            _last_wa_ts = now
        return result
