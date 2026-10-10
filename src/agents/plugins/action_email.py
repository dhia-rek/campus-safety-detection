"""Email action agent: rich HTML e-mail alert on incidents, gated by a cooldown.

The e-mail has danger icons, the incident frame inline and a short video clip of
the footage attached. It is sent whenever the coordinator declares an incident.

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
EMAIL_TO         : recipient (required; the email action is disabled when unset)
EMAIL_FROM       : sender               (default = SMTP_USER)
EMAIL_SEVERITIES : severities that fire it (default "low,medium,high")
EMAIL_CLIP       : attach a video clip of the footage (default "1")
"""

from __future__ import annotations

import html
import os
import shutil
import smtplib
import subprocess
import tempfile
import threading
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any, cast

from src.agents.base import ActionAgent, Decision, Observation
from src.agents.registry import register
from src.paths import TEST_FRAMES_ROOT

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()
EMAIL_TO  = os.getenv("EMAIL_TO", "").strip()  # no default: never send to a baked-in address
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

_last_email_ts = 0.0  # time of the last *successful* send
_in_flight = False  # a send is running; blocks duplicates until its outcome is known
_state_lock = threading.Lock()
_warned_unconfigured = False


def _frames(scene: str) -> list[Path]:
    """Sorted JPEG frame paths of a scene (empty if the folder is missing)."""
    return sorted((TEST_FRAMES_ROOT / scene).glob("*.jpg"))


def _frame_path(scene: str, idx: int) -> str | None:
    """Path of the *idx*-th sorted frame of a scene, or None if out of range."""
    fs = _frames(scene)
    return str(fs[idx]) if 0 <= idx < len(fs) else None


CLIP_WINDOW_FRAMES = 75  # frames of footage in the attached clip
CLIP_FPS = 15  # playback rate of the attached clip
CLIP_WIDTH = 854  # clip width in pixels; height follows the aspect ratio
NETWORK_TIMEOUT_SEC = 60  # ceiling for ffmpeg and the SMTP connection


def _build_clip(scene: str, frame: int, window: int = CLIP_WINDOW_FRAMES, fps: int = CLIP_FPS) -> str | None:
    """Build a short MP4 of the footage around the incident frame (ffmpeg).

    Args:
        scene: Scene id.
        frame: Incident frame position.
        window: Frames of footage before *frame* and after it (the end is exclusive).
        fps: Frame rate of the resulting clip.

    Returns:
        Path of the MP4 in a fresh temp directory (never cleaned up here), or
        None if the scene has no frames, ffmpeg failed to launch/timed out, or
        no output file was produced. If the window is empty (frame beyond the
        end) all frames are used instead.
    """
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
             "-vf", f"scale={CLIP_WIDTH}:-2,format=yuv420p", "-c:v", "libx264", str(out)],
            capture_output=True, timeout=NETWORK_TIMEOUT_SEC,
        )
    except Exception as exc:
        print(f"[email] clip build failed: {exc}")
        return None
    return str(out) if out.exists() else None


def _html(scene: str, frame: int, decision: Decision, has_frame: bool) -> str:
    """Render the HTML body of the alert e-mail.

    Args:
        scene: Scene id.
        frame: Frame position shown in the table.
        decision: Decision whose severity picks the icon/colour (unknown
            severities use the "medium" style).
        has_frame: Include an ``<img src="cid:frame">`` placeholder; the image
            itself is attached by the caller.

    Returns:
        HTML string. All interpolated text (scene, severity, report, action) is
        HTML-escaped, since the report can come from an LLM.
    """
    icon, color = SEV_STYLE.get(decision.severity, SEV_STYLE["medium"])
    esc = html.escape
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
      font-weight:bold;text-transform:uppercase;font-size:13px">{icon} {esc(str(decision.severity))}</span>
    </p>
    <table style="font-size:14px;line-height:1.7">
      <tr><td><b>Scene</b></td><td style="padding-left:14px">{esc(str(scene))}</td></tr>
      <tr><td><b>Frame</b></td><td style="padding-left:14px">{esc(str(frame))}</td></tr>
      <tr><td><b>Incident</b></td><td style="padding-left:14px">{esc(str(decision.is_incident))}</td></tr>
    </table>
    <p style="font-size:14px"><b>🧠 Report.</b> {esc(str(decision.report))}</p>
    <p style="font-size:14px"><b>➡️ Action.</b> {esc(str(decision.action))}</p>
    {img}
    <p style="font-size:12px;color:#888;margin-top:14px">🎞️ Footage clip attached · Campus Safety AI — ECE</p>
  </div>
</div>"""


def _send_and_record(scene: str, frame: int, decision: Decision) -> dict:
    """Run ``_send_email`` and update the cooldown state from its outcome.

    The cooldown starts only after a successful send, so a failed send can be
    retried immediately. ``_in_flight`` stops a second thread from starting
    while this one is still running.
    """
    global _last_email_ts, _in_flight
    try:
        result = _send_email(scene, frame, decision)
        with _state_lock:
            if result.get("sent"):
                _last_email_ts = time.time()
        return result
    finally:
        with _state_lock:
            _in_flight = False


def _send_email(scene: str, frame: int, decision: Decision) -> dict:
    """Build and send the alert e-mail (blocking; run it in a thread).

    The message has plain-text and HTML parts, the frame inline, and, when
    ``EMAIL_CLIP`` is on, an MP4 clip attached.

    Returns:
        ``{"sent": True, "to": ...}``; ``{"sent": False, "skipped": ...}`` if SMTP
        credentials are unset; ``{"sent": False, "reason": ...}`` on SMTP error.
    """
    if not (SMTP_USER and SMTP_PASS):
        return {"sent": False, "skipped": "SMTP_USER/SMTP_PASS not set"}
    try:
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
                cast(Any, msg.get_payload())[1].add_related(f.read(), maintype="image", subtype="jpeg", cid="frame")

        # attach a short video clip of the footage around the incident
        if EMAIL_CLIP:
            clip = _build_clip(scene, frame)
            if clip:
                with open(clip, "rb") as f:
                    msg.add_attachment(f.read(), maintype="video", subtype="mp4",
                                       filename=f"{scene}_incident.mp4")

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=NETWORK_TIMEOUT_SEC) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return {"sent": True, "to": EMAIL_TO}
    except Exception as exc:  # message-build, attachment, or SMTP failure must not crash the worker thread
        print(f"[email] send failed: {exc}")
        return {"sent": False, "reason": str(exc)}


@register
class EmailAction(ActionAgent):
    """Action agent that e-mails an HTML alert with frame and clip."""

    name        = "email"
    label       = "Email alert"
    tier        = "starter"
    description = "E-mails an HTML alert (danger icons + frame + footage clip) when an incident is detected."

    def handle(self, decision: Decision, observations: list[Observation], ctx: dict) -> dict:
        """Queue an alert e-mail for an incident in ``EMAIL_SEVERITIES``.

        The cooldown starts only after a *successful* send, so a failed send can
        be retried at once. While a send is in flight further e-mails are skipped.
        If ``EMAIL_TO`` is unset the action is disabled (one log line, then a
        ``skipped`` result).

        Returns:
            ``{"queued": True, "to": ...}`` or ``{"sent": False, "skipped": reason}``.
        """
        global _in_flight, _warned_unconfigured
        if not EMAIL_TO:
            if not _warned_unconfigured:
                _warned_unconfigured = True
                print("[email] disabled: EMAIL_TO is not set (set EMAIL_TO, SMTP_USER and SMTP_PASS to enable it)")
            return {"sent": False, "skipped": "EMAIL_TO not set"}
        if not decision.is_incident or decision.severity not in EMAIL_SEVERITIES:
            return {"sent": False, "skipped": "no incident / below threshold"}
        with _state_lock:
            if _in_flight:
                return {"sent": False, "skipped": "send in progress"}
            if time.time() - _last_email_ts < COOLDOWN_SEC:
                return {"sent": False, "skipped": "cooldown"}
            _in_flight = True
        # Send in the background so the dashboard never freezes while the
        # frame + footage clip are built and the SMTP handshake runs.
        scene, frame = ctx.get("scene", "?"), ctx.get("frame", -1)
        threading.Thread(target=_send_and_record, args=(scene, frame, decision), daemon=True).start()
        return {"queued": True, "to": EMAIL_TO}
