from __future__ import annotations

from fastapi.testclient import TestClient

from src.agents import registry
from src.agents.plugins.action_telegram import TelegramAction
from src.agents.plugins.coordinator_llm import LLMCoordinator
from src.agents.plugins.sound import SoundAgent
from src.agents.plugins.speech import SpeechAgent
from src.agents.plugins.vision import VisionAgent
from src.alerts import notifier
from src.dashboard import api


def _install_safe_registry(monkeypatch):
    registry._CLASSES = {
        "perception": {
            "vision": VisionAgent,
            "sound": SoundAgent,
            "speech": SpeechAgent,
        },
        "coordinator": {"coordinator_llm": LLMCoordinator},
        "action": {"telegram": TelegramAction},
    }
    registry._discovered = True
    monkeypatch.setattr(registry, "discover", lambda force=False: None)



def test_scenes_and_scene_scores_endpoints():
    api._scene_frames.cache_clear()
    client = TestClient(api.app)

    scenes_res = client.get("/api/scenes")
    assert scenes_res.status_code == 200
    scenes = scenes_res.json()["scenes"]
    names = {row["scene"] for row in scenes}
    assert {"demo_incident", "demo_calm"}.issubset(names)

    series_res = client.get("/api/scene/demo_incident", params={"mode": "fused"})
    assert series_res.status_code == 200
    payload = series_res.json()
    assert payload["scene"] == "demo_incident"
    assert payload["mode"] == "fused"
    assert payload["n_frames"] > 0
    assert "fused_score" in payload["series"]



def test_frame_and_mode_validation_endpoints():
    api._scene_frames.cache_clear()
    client = TestClient(api.app)

    ok = client.get("/api/frame/demo_incident/0.jpg")
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("image/jpeg")

    oob = client.get("/api/frame/demo_incident/9999.jpg")
    assert oob.status_code == 404

    bad_mode = client.get("/api/scene/demo_incident", params={"mode": "unknown"})
    assert bad_mode.status_code == 400



def test_telegram_status_and_manual_alert_mock_send(monkeypatch):
    api._scene_frames.cache_clear()
    client = TestClient(api.app)

    notifier._mock.sent.clear()
    api._last_alert_ts = 0.0

    status = client.get("/api/telegram/status")
    assert status.status_code == 200
    s = status.json()
    assert s["mock"] is True
    assert s["token_set"] is True

    res = client.post(
        "/api/alert",
        json={"scene": "demo_incident", "frame": 0, "score": 1.23, "modality": "AUDIO"},
    )
    assert res.status_code == 200
    assert res.json().get("sent") == 1
    assert len(notifier._mock.sent) == 1



def test_decision_endpoint_falls_back_to_heuristic_when_ollama_fails(monkeypatch):
    _install_safe_registry(monkeypatch)
    api._scene_frames.cache_clear()
    client = TestClient(api.app)

    def boom(*_args, **_kwargs):
        raise RuntimeError("ollama offline")

    monkeypatch.setattr("src.agents.plugins.coordinator_llm.requests.post", boom)

    res = client.get("/api/decision/demo_incident")
    assert res.status_code == 200
    payload = res.json()
    assert payload["scene"] == "demo_incident"
    assert payload["decision"]["source"] == "heuristic"
