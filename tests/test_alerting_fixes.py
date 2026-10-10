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
def test_webhook_non_2xx_is_not_dispatched(monkeypatch, capsys):
    monkeypatch.setattr(action_webhook, "INCIDENT_WEBHOOK_URL", "https://hook.example/x")
    monkeypatch.setattr(action_webhook.requests, "post", lambda *a, **k: _Resp(500))
    res = action_webhook.WebhookAction().handle(_decision(), [], {"scene": "s", "frame": 1})
    assert res["dispatched"] is False
    assert "500" in res["reason"]
    assert "automation webhook failed" in capsys.readouterr().out


def test_webhook_redirect_is_not_dispatched(monkeypatch, capsys):
    monkeypatch.setattr(action_webhook, "INCIDENT_WEBHOOK_URL", "https://hook.example/x")
    def redirect(*_args, **kwargs):
        assert kwargs["allow_redirects"] is False
        return _Resp(302)

    monkeypatch.setattr(action_webhook.requests, "post", redirect)
    res = action_webhook.WebhookAction().handle(_decision(), [], {"scene": "s", "frame": 1})
    assert res["dispatched"] is False
    assert "302" in res["reason"]
    assert "automation webhook failed" in capsys.readouterr().out


def test_webhook_2xx_is_dispatched(monkeypatch):
    monkeypatch.setattr(action_webhook, "INCIDENT_WEBHOOK_URL", "https://hook.example/x")
    monkeypatch.setattr(action_webhook.requests, "post", lambda *a, **k: _Resp(200))
    res = action_webhook.WebhookAction().handle(_decision(), [], {"scene": "s", "frame": 1})
    assert res == {"dispatched": True}


def test_telegram_http_error_is_not_counted_as_sent(monkeypatch, tmp_path, capsys):
    subs = tmp_path / "subs.json"
    subs.write_text(json.dumps({"subscribers": [1, 2, 3]}), encoding="utf-8")
    img = tmp_path / "f.jpg"
    img.write_bytes(b"x")
    statuses = iter([200, 403, 302])  # rejected/redirected sends must not count
    monkeypatch.setattr(notifier.requests, "post", lambda *a, **k: _Resp(next(statuses)))
    n = notifier.TelegramNotifier(token="t", subscriber_file=str(subs))
    assert n.send_photo(str(img), "cap") == {"sent": 1}
    output = capsys.readouterr().out
    assert "HTTP 403" in output
    assert "HTTP 302" in output


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


def test_email_missing_attachment_is_logged_failure_without_cooldown(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(action_email, "SMTP_USER", "sender@example.com")
    monkeypatch.setattr(action_email, "SMTP_PASS", "test-password")
    monkeypatch.setattr(action_email, "EMAIL_CLIP", False)
    monkeypatch.setattr(action_email, "_frame_path", lambda *_a: str(tmp_path / "missing.jpg"))
    monkeypatch.setattr(action_email, "_last_email_ts", 0.0)
    result = action_email._send_and_record("scene", 1, _decision())
    assert result["sent"] is False
    assert "send failed" in capsys.readouterr().out
    assert action_email._last_email_ts == 0.0


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
def test_scan_audio_clips_has_no_hardcoded_path_and_main_guard(monkeypatch):
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

        # Both documented path sources must reach the scan, not merely pass
        # argument parsing. Stub the ML work so this is a fast unit test.
        import numpy as np

        clip_dir = ROOT / "tests" / ".tmp_scan_audio_clips"
        clip_dir.mkdir(exist_ok=True)
        clip = clip_dir / "V_1.mp4"
        clip.write_bytes(b"fixture")
        monkeypatch.setattr(mod, "librosa", SimpleNamespace(
            load=lambda *_a, **_k: (np.ones(16000, dtype=float), 32000)))
        monkeypatch.setattr(mod, "AudioTagging", lambda **_k: SimpleNamespace(
            inference=lambda _windows: (np.ones((1, 1)), None)))
        monkeypatch.setattr(mod, "load_target_class_indices", lambda _labels: ([0], ["violence"]))
        monkeypatch.setattr(mod, "slice_windows", lambda *_a: object())
        monkeypatch.setattr(mod, "audioset_labels", [])
        try:
            assert mod.main([str(clip_dir)]) == 0  # positional CLI path
            monkeypatch.setenv("RLVS_VIOLENCE_DIR", str(clip_dir))
            assert mod.main([]) == 0  # environment path
        finally:
            clip.unlink()
            clip_dir.rmdir()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
