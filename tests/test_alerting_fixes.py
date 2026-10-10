"""Regression tests for the alerting bug fixes (HTTP status, escaping, cooldown, dry run, EMAIL_TO, scan script)."""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

from src.agents import registry
from src.agents.base import Decision
from src.agents.plugins import action_email, action_webhook
from src.alerts import notifier
from src.dashboard import api

ROOT = Path(__file__).resolve().parents[1]


def _decision(**kw) -> Decision:
    base = dict(is_incident=True, severity="high", modalities=["visual"], reasoning="r",
                action="Notify operator.", report="Fight detected.", source="heuristic")
    base.update(kw)
    return Decision(**base)


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status
        self.ok = 200 <= status < 300

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code} error")


# 1. HTTP status checks -------------------------------------------------------------------------
def test_webhook_non_2xx_is_not_dispatched(monkeypatch):
    monkeypatch.setattr(action_webhook, "INCIDENT_WEBHOOK_URL", "https://hook.example/x")
    monkeypatch.setattr(action_webhook.requests, "post", lambda *a, **k: _Resp(500))
    res = action_webhook.WebhookAction().handle(_decision(), [], {"scene": "s", "frame": 1})
    assert res["dispatched"] is False
    assert "500" in res["reason"]


def test_webhook_2xx_is_dispatched(monkeypatch):
    monkeypatch.setattr(action_webhook, "INCIDENT_WEBHOOK_URL", "https://hook.example/x")
    monkeypatch.setattr(action_webhook.requests, "post", lambda *a, **k: _Resp(200))
    res = action_webhook.WebhookAction().handle(_decision(), [], {"scene": "s", "frame": 1})
    assert res == {"dispatched": True}


def test_telegram_http_error_is_not_counted_as_sent(monkeypatch, tmp_path):
    subs = tmp_path / "subs.json"
    subs.write_text(json.dumps({"subscribers": [1, 2]}), encoding="utf-8")
    img = tmp_path / "f.jpg"
    img.write_bytes(b"x")
    statuses = iter([200, 403])  # second subscriber blocked the bot
    monkeypatch.setattr(notifier.requests, "post", lambda *a, **k: _Resp(next(statuses)))
    n = notifier.TelegramNotifier(token="t", subscriber_file=str(subs))
    assert n.send_photo(str(img), "cap") == {"sent": 1}


# 2. HTML escaping ------------------------------------------------------------------------------
def test_email_html_escapes_report_and_action():
    d = _decision(report='<script>alert("x")</script> & <b>bold</b>', action="<img src=x onerror=1>")
    out = action_email._html("<scene>", 3, d, has_frame=False)
    assert "<script>" not in out and "<img src=x" not in out
    assert "&lt;script&gt;" in out and "&amp;" in out and "&lt;scene&gt;" in out


# 3. Cooldown only after success ----------------------------------------------------------------
def _run_email_handle(monkeypatch, result: dict) -> tuple[dict, threading.Thread]:
    """Call EmailAction.handle with _send_email stubbed; return handle result and the worker thread."""
    monkeypatch.setattr(action_email, "EMAIL_TO", "ops@example.com")
    monkeypatch.setattr(action_email, "_send_email", lambda *a, **k: result)
    started: list[threading.Thread] = []
    real_thread = threading.Thread

    def spy(*a, **k):
        t = real_thread(*a, **k)
        started.append(t)
        return t

    monkeypatch.setattr(action_email.threading, "Thread", spy)
    out = action_email.EmailAction().handle(_decision(), [], {"scene": "s", "frame": 1})
    for t in started:
        t.join(5)
    return out, started[0] if started else None


def test_email_cooldown_not_started_by_failed_send(monkeypatch):
    monkeypatch.setattr(action_email, "_last_email_ts", 0.0)
    monkeypatch.setattr(action_email, "_in_flight", False)
    out, _ = _run_email_handle(monkeypatch, {"sent": False, "reason": "smtp down"})
    assert out["queued"] is True
    assert action_email._last_email_ts == 0.0 and action_email._in_flight is False
    # a retry is allowed immediately
    out2, _ = _run_email_handle(monkeypatch, {"sent": False, "reason": "smtp down"})
    assert out2.get("queued") is True


def test_email_cooldown_starts_after_successful_send(monkeypatch):
    monkeypatch.setattr(action_email, "_last_email_ts", 0.0)
    monkeypatch.setattr(action_email, "_in_flight", False)
    out, _ = _run_email_handle(monkeypatch, {"sent": True, "to": "ops@example.com"})
    assert out["queued"] is True
    assert action_email._last_email_ts > 0
    out2 = action_email.EmailAction().handle(_decision(), [], {"scene": "s", "frame": 1})
    assert out2 == {"sent": False, "skipped": "cooldown"}


# 4. GET decision is a dry run ------------------------------------------------------------------
class _SpyAction:
    name = "spy"
    calls = 0

    def handle(self, *a, **k):
        _SpyAction.calls += 1
        return {"sent": 1}


@pytest.fixture
def spy_registry(monkeypatch):
    from src.agents.plugins.coordinator_llm import LLMCoordinator
    from src.agents.plugins.vision import VisionAgent

    monkeypatch.setattr(registry, "discover", lambda force=False: None)
    monkeypatch.setattr(registry, "perception_agents", lambda only_enabled=True: [VisionAgent()])
    monkeypatch.setattr(registry, "coordinator", lambda: LLMCoordinator())
    monkeypatch.setattr(registry, "action_agents", lambda only_enabled=True: [_SpyAction()])
    monkeypatch.setattr("src.agents.plugins.coordinator_llm.requests.post",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    _SpyAction.calls = 0
    api._scene_frames.cache_clear()


def test_get_decision_runs_no_actions(spy_registry):
    res = TestClient(api.app).get("/api/decision/demo_incident")
    assert res.status_code == 200
    assert res.json()["actions"] == {}
    assert _SpyAction.calls == 0


def test_post_execute_runs_actions(spy_registry):
    res = TestClient(api.app).post("/api/decision/demo_incident/execute")
    assert res.status_code == 200
    assert res.json()["actions"] == {"spy": {"sent": 1}}
    assert _SpyAction.calls == 1


def test_get_execute_not_allowed(spy_registry):
    assert TestClient(api.app).get("/api/decision/demo_incident/execute").status_code in (404, 405)


# 5. No hard-coded EMAIL_TO ---------------------------------------------------------------------
def test_email_disabled_without_email_to(monkeypatch, capsys):
    monkeypatch.setattr(action_email, "EMAIL_TO", "")
    monkeypatch.setattr(action_email, "_warned_unconfigured", False)
    sent = []
    monkeypatch.setattr(action_email, "_send_email", lambda *a, **k: sent.append(1))
    agent = action_email.EmailAction()
    res = agent.handle(_decision(), [], {"scene": "s", "frame": 1})
    agent.handle(_decision(), [], {"scene": "s", "frame": 1})
    assert res == {"sent": False, "skipped": "EMAIL_TO not set"}
    assert not sent
    assert capsys.readouterr().out.count("EMAIL_TO is not set") == 1  # logged once, clearly


def test_no_personal_address_in_source():
    for p in [*(ROOT / "src").rglob("*.py"), ROOT / ".env.example", ROOT / "README.md"]:
        assert "rekik11" not in p.read_text(encoding="utf-8"), p


# 6. scan_audio_clips: no hard-coded path, main guard -------------------------------------------
def test_scan_audio_clips_has_no_hardcoded_path_and_main_guard():
    src = (ROOT / "scripts" / "scan_audio_clips.py").read_text(encoding="utf-8")
    assert "/Users/" not in src
    assert '__name__ == "__main__"' in src
    # importing must not run the scan: heavy deps are stubbed so this works without the ML extra
    import sys
    stubs = {m: SimpleNamespace() for m in ("librosa", "panns_inference")}
    stubs["panns_inference"] = SimpleNamespace(AudioTagging=object, labels=[])
    stubs["src.pipeline.audio_score"] = SimpleNamespace(
        SAMPLE_RATE=32000, load_target_class_indices=lambda labels: ([], []), slice_windows=lambda *a: [])
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("scan_audio_clips", ROOT / "scripts" / "scan_audio_clips.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # would crash or scan if it ran at import
        with pytest.raises(SystemExit) as e:
            mod.main([])  # no folder, no env var -> usage error
        assert e.value.code == 2
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
