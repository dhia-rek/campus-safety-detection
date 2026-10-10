from __future__ import annotations

import json

import requests

from src.alerts import notifier


def test_get_notifier_returns_mock_in_demo_mode():
    n = notifier.get_notifier()
    assert n.mock is True
    assert n.configured is True


def test_telegram_notifier_without_token_is_skipped(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fakejpg")

    n = notifier.TelegramNotifier(token="")
    result = n.send_photo(str(image), "caption")
    assert result == {"sent": 0, "skipped": "no TELEGRAM_TOKEN"}


def test_telegram_notifier_with_token_uses_subscribers_file(monkeypatch, tmp_path):
    subs_file = tmp_path / "subscribers.json"
    subs_file.write_text(json.dumps({"subscribers": [111, 222]}), encoding="utf-8")

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fakejpg")

    calls = []

    def fake_post(url, data=None, files=None, timeout=None):
        calls.append({"url": url, "chat_id": data["chat_id"], "timeout": timeout, "files": bool(files)})

        class _Resp:
            status_code = 200

        return _Resp()

    monkeypatch.setattr(notifier.requests, "post", fake_post)

    token = "123:SECRET_TOKEN"
    n = notifier.TelegramNotifier(token=token, subscriber_file=str(subs_file))
    result = n.send_photo(str(image), "caption")

    assert result == {"sent": 2}
    assert len(calls) == 2
    assert all(token in c["url"] for c in calls)
    assert {c["chat_id"] for c in calls} == {111, 222}


def test_telegram_notifier_error_log_does_not_leak_token(monkeypatch, tmp_path, capsys):
    subs_file = tmp_path / "subscribers.json"
    subs_file.write_text(json.dumps({"subscribers": [111]}), encoding="utf-8")

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fakejpg")

    token = "123:VERY_SECRET"

    def fake_post(*_args, **_kwargs):
        raise requests.RequestException("network down")

    monkeypatch.setattr(notifier.requests, "post", fake_post)

    n = notifier.TelegramNotifier(token=token, subscriber_file=str(subs_file))
    result = n.send_photo(str(image), "caption")

    out = capsys.readouterr().out
    assert result == {"sent": 0}
    assert "RequestException" in out
    assert token not in out
