from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from src.agents import registry
from src.agents.base import Observation
from src.agents.context import load_scene_context, peak_frame
from src.agents.plugins.coordinator_llm import LLMCoordinator, _call_ollama, _heuristic
from src.agents.plugins.sound import SoundAgent
from src.agents.plugins.speech import SpeechAgent
from src.agents.plugins.vision import VisionAgent


def test_registry_discovery_and_catalog_with_safe_plugin_subset(monkeypatch):
    registry._discovered = False
    registry._CLASSES = {"perception": {}, "coordinator": {}, "action": {}}

    safe_names = [
        "vision",
        "sound",
        "speech",
        "coordinator_llm",
        "action_telegram",
        "action_email",
        "action_webhook",
    ]
    monkeypatch.setattr(
        registry.pkgutil,
        "iter_modules",
        lambda _path: [SimpleNamespace(name=name) for name in safe_names],
    )

    real_import_module = importlib.import_module

    def _import_or_reload(name: str):
        if name in sys.modules:
            return importlib.reload(sys.modules[name])
        return real_import_module(name)

    monkeypatch.setattr(registry.importlib, "import_module", _import_or_reload)

    registry.discover(force=True)
    names = {row["name"] for row in registry.catalog()}

    assert {"vision", "sound", "speech", "coordinator_llm", "telegram"}.issubset(names)


def test_agent_enabled_toggle(monkeypatch):
    monkeypatch.setenv("AGENT_VISION_ENABLED", "0")
    assert VisionAgent().enabled is False
    monkeypatch.setenv("AGENT_VISION_ENABLED", "1")
    assert VisionAgent().enabled is True


def test_demo_scene_observations_and_peak_frame_window():
    ctx = load_scene_context("demo_incident", "testing", threshold=1.0)
    frame = peak_frame(ctx)

    assert 55 <= frame <= 90

    vision = VisionAgent().observe(ctx, frame)
    sound = SoundAgent().observe(ctx, frame)
    speech = SpeechAgent().observe(ctx, frame)

    assert vision.triggered is True
    assert sound.triggered is True
    assert speech.triggered is True
    assert "idiot" in speech.detail


def test_demo_calm_no_high_severity_from_heuristic():
    ctx = load_scene_context("demo_calm", "testing", threshold=1.0)
    frame = peak_frame(ctx)
    obs = [
        VisionAgent().observe(ctx, frame),
        SoundAgent().observe(ctx, frame),
        SpeechAgent().observe(ctx, frame),
    ]

    decision = _heuristic(obs)
    assert decision.severity != "high"


@pytest.mark.parametrize(
    ("observations", "expected_severity", "expected_incident"),
    [
        (
            [
                Observation("vision", "Vision", 0.2, False),
                Observation("sound", "Sound", 0.1, False),
                Observation("speech", "Speech", 0.1, False, ""),
            ],
            "none",
            False,
        ),
        (
            [
                Observation("vision", "Vision", 1.2, True),
                Observation("sound", "Sound", 0.1, False),
                Observation("speech", "Speech", 0.1, False, ""),
            ],
            "medium",
            True,
        ),
        (
            [
                Observation("vision", "Vision", 1.2, True),
                Observation("sound", "Sound", 1.4, True),
                Observation("speech", "Speech", 0.1, False, ""),
            ],
            "high",
            True,
        ),
        (
            [
                Observation("vision", "Vision", 0.1, False),
                Observation("sound", "Sound", 0.1, False),
                Observation("speech", "Speech", 0.1, False, "threat"),
            ],
            "low",
            True,
        ),
    ],
)
def test_heuristic_cases(observations, expected_severity, expected_incident):
    d = _heuristic(observations)
    assert d.severity == expected_severity
    assert d.is_incident == expected_incident


class _DummyResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"message": {"content": self._content}}


def test_call_ollama_invalid_severity_gets_normalized(monkeypatch):
    def fake_post(*_args, **_kwargs):
        return _DummyResponse(
            '{"is_incident": true, "severity": "EXTREME", "modalities": ["x"], '
            '"reasoning": "r", "action": "a", "report": "rep"}'
        )

    monkeypatch.setattr("src.agents.plugins.coordinator_llm.requests.post", fake_post)

    d = _call_ollama([Observation("vision", "Vision", 0.0, False)], threshold=1.0)
    assert d.source == "llm"
    assert d.severity == "none"


def test_coordinator_falls_back_to_heuristic_when_ollama_errors(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr("src.agents.plugins.coordinator_llm.requests.post", boom)

    c = LLMCoordinator()
    d = c.decide([Observation("vision", "Vision", 1.2, True)], threshold=1.0)
    assert d.source == "heuristic"
    assert d.severity == "medium"
