"""
Email action agent — sends a rich HTML e-mail alert (danger icons + the incident
frame inline + a short video clip of the footage attached) whenever the
coordinator declares an incident, gated by a cooldown.

Drop-in action plugin: dropping this file adds an "email" capability to the
multi-agent runtime with no core change. No-ops safely when SMTP is not set.

Set-up for Gmail
----------------
Gmail blocks normal passwords for SMTP, so create an App Password:
  Google Account -> Security -> 2-Step Verification -> App passwords.
Put it in SMTP_PASS (16 characters, no spaces).

Environment
-----------
SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS : SMTP credentials
EMAIL_TO         : recipient            (default "dhia.rekik11@gmail.com")
EMAIL_FROM       : sender               (default = SMTP_USER)
EMAIL_SEVERITIES : severities that fire it (default "low,medium,high")
EMAIL_CLIP       : attach a video clip of the footage (default "1")
"""

from __future__ import annotations

import os
import shutil
import smtplib
import subprocess
import tempfile
import threading
import time
from email.message import EmailMessage
from pathlib import Path

from src.agents.base import ActionAgent, Decision, Observation
from src.agents.registry import register
from src.paths import TEST_FRAMES_ROOT

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()
EMAIL_TO  = os.getenv("EMAIL_TO", "dhia.rekik11@gmail.com").strip()
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER or EMAIL_TO).strip()
EMAIL_SEVERITIES = {
    s.strip() for s in os.getenv("EMAIL_SEVERITIES", "low,medium,high").split(",") if s.strip()
}
EMAIL_CLIP = os.getenv("EMAIL_CLIP", "1").lower() not in {"0", "false", "no"}
COOLDOWN_SEC = 30

# severity -> (emoji icon, accent colour)
SEV_STYLE = {
    "high":   ("🚨", "#c0392b"),
    "medium": ("⚠️", "#e67e22"),
    "low":    ("🔔", "#b8860b"),
    "none":   ("✅", "#27ae60"),
}

_last_email_ts = 0.0


def _frames(scene: str):
    return sorted((TEST_FRAMES_ROOT / scene).glob("*.jpg"))


def _frame_path(scene: str, idx: int) -> str | None:
    fs = _frames(scene)
    return str(fs[idx]) if 0 <= idx < len(fs) else None


def _build_clip(scene: str, frame: int, window: int = 75, fps: int = 15) -> str | None:
    """Build a short MP4 of the footage around the incident frame (ffmpeg)."""
    fs = _frames(scene)
    if not fs:
        return None
    lo, hi = max(0, frame - window), min(len(fs), frame + window)
    sel = fs[lo:hi] or fs
    tmp = Path(tempfile.mkdtemp(prefix="csclip_"))
    for i, fp in enumerate(sel):
        shutil.copy(fp, tmp / f"{i:04d}.jpg")
    out = tmp / f"{scene}_incident.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp / "%04d.jpg"),
             "-vf", "scale=854:-2,format=yuv420p", "-c:v", "libx264", str(out)],
            capture_output=True, timeout=60,
        )
    except Exception as exc:
        print(f"[email] clip build failed: {exc}")
        return None
    return str(out) if out.exists() else None


def _html(scene: str, frame: int, decision: Decision, has_frame: bool) -> str:
    icon, color = SEV_STYLE.get(decision.severity, SEV_STYLE["medium"])
    img = (f'<img src="cid:frame" alt="incident frame" '
           f'style="width:100%;max-width:520px;border-radius:8px;border:2px solid {color};margin-top:12px"/>'
           ) if has_frame else ""
    return f"""\
<div style="font-family:Arial,sans-serif;max-width:560px;margin:auto;color:#1a1a1a">
  <div style="background:{color};color:#fff;padding:16px 20px;border-radius:10px 10px 0 0">
    <span style="font-size:26px">{icon}</span>
    <span style="font-size:20px;font-weight:bold;vertical-align:middle">  CCTV ANOMALY ALERT</span>
  </div>
  <div style="border:1px solid #e3e3e3;border-top:none;padding:18px 20px;border-radius:0 0 10px 10px">
    <p style="margin:0 0 10px">
      <span style="background:{color};color:#fff;padding:3px 12px;border-radius:20px;
      font-weight:bold;text-transform:uppercase;font-size:13px">{icon} {decision.severity}</span>
    </p>
    <table style="font-size:14px;line-height:1.7">
      <tr><td><b>Scene</b></td><td style="padding-left:14px">{scene}</td></tr>
      <tr><td><b>Frame</b></td><td style="padding-left:14px">{frame}</td></tr>
      <tr><td><b>Incident</b></td><td style="padding-left:14px">{decision.is_incident}</td></tr>
    </table>
    <p style="font-size:14px"><b>🧠 Report.</b> {decision.report}</p>
    <p style="font-size:14px"><b>➡️ Action.</b> {decision.action}</p>
    {img}
    <p style="font-size:12px;color:#888;margin-top:14px">🎞️ Footage clip attached · Campus Safety AI — ECE</p>
  </div>
</div>"""


def _send_email(scene: str, frame: int, decision: Decision) -> dict:
    if not (SMTP_USER and SMTP_PASS):
        return {"sent": False, "skipped": "SMTP_USER/SMTP_PASS not set"}
    icon, _ = SEV_STYLE.get(decision.severity, SEV_STYLE["medium"])
    fpath = _frame_path(scene, frame)

    msg = EmailMessage()
    msg["Subject"] = f"{icon} [Campus Safety] {decision.severity.upper()} incident — {scene} frame {frame}"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.set_content(
        f"CCTV ANOMALY ALERT\nScene: {scene} | Frame: {frame} | Severity: {decision.severity}\n"
        f"Report: {decision.report}\nAction: {decision.action}"
    )
    msg.add_alternative(_html(scene, frame, decision, has_frame=bool(fpath)), subtype="html")

    # inline frame image (referenced by cid:frame in the HTML)
    if fpath:
        with open(fpath, "rb") as f:
            msg.get_payload()[1].add_related(f.read(), maintype="image", subtype="jpeg", cid="frame")

    # attach a short video clip of the footage around the incident
    if EMAIL_CLIP:
        clip = _build_clip(scene, frame)
        if clip:
            with open(clip, "rb") as f:
                msg.add_attachment(f.read(), maintype="video", subtype="mp4",
                                   filename=f"{scene}_incident.mp4")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return {"sent": True, "to": EMAIL_TO}
    except Exception as exc:
        print(f"[email] send failed: {exc}")
        return {"sent": False, "reason": str(exc)}


@register
class EmailAction(ActionAgent):
    name        = "email"
    label       = "Email alert"
    tier        = "starter"
    description = "E-mails an HTML alert (danger icons + frame + footage clip) when an incident is detected."

    def handle(self, decision: Decision, observations: list[Observation], ctx: dict) -> dict:
        global _last_email_ts
        if not decision.is_incident or decision.severity not in EMAIL_SEVERITIES:
            return {"sent": False, "skipped": "no incident / below threshold"}
        now = time.time()
        if now - _last_email_ts < COOLDOWN_SEC:
            return {"sent": False, "skipped": "cooldown"}
        # Send in the background so the dashboard never freezes while the
        # frame + footage clip are built and the SMTP handshake runs.
        _last_email_ts = now
        scene, frame = ctx.get("scene", "?"), ctx.get("frame", -1)
        threading.Thread(target=_send_email, args=(scene, frame, decision), daemon=True).start()
        return {"queued": True, "to": EMAIL_TO}
