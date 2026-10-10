"""Alert notifier seam: real Telegram, or a mock for demo mode and tests.

``get_notifier()`` returns the mock when DEMO_MODE=1 (src.paths) or when no
TELEGRAM_TOKEN is set and MOCK_NOTIFIER=1. The mock never touches the network;
it records messages in memory so the demo and tests can inspect them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from src.paths import DEMO_MODE


class MockNotifier:
    """Records alerts instead of sending them."""

    mock = True

    def __init__(self) -> None:
        self.sent: list[dict] = []

    @property
    def configured(self) -> bool:
        return True

    def subscribers(self) -> list[int]:
        return [0]  # one fake subscriber

    def send_photo(self, image_path: str, caption: str) -> dict:
        self.sent.append({"image": Path(image_path).name, "caption": caption})
        print(f"[mock-telegram] would send {Path(image_path).name}: {caption.splitlines()[0]}")
        return {"sent": 1, "mock": True}


class TelegramNotifier:
    """Sends photos to every subscriber in subscribers.json via the Bot API."""

    mock = False

    def __init__(self, token: str, subscriber_file: str = "subscribers.json") -> None:
        self.token = token
        self.subscriber_file = subscriber_file

    @property
    def configured(self) -> bool:
        return bool(self.token)

    def subscribers(self) -> list:
        try:
            with open(self.subscriber_file) as f:
                return json.load(f).get("subscribers", [])
        except (OSError, ValueError):
            return []

    def send_photo(self, image_path: str, caption: str) -> dict:
        if not self.token:
            return {"sent": 0, "skipped": "no TELEGRAM_TOKEN"}
        subs = self.subscribers()
        if not subs:
            return {"sent": 0, "skipped": "no subscribers"}
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        sent = 0
        for chat_id in subs:
            try:
                with open(image_path, "rb") as img:
                    requests.post(url, data={"chat_id": chat_id, "caption": caption}, files={"photo": img}, timeout=15)
                sent += 1
            except (OSError, requests.RequestException) as exc:
                print(f"[telegram] error sending to a subscriber: {type(exc).__name__}")  # never log the URL: it holds the token
        return {"sent": sent}


_mock = MockNotifier()


def get_notifier() -> MockNotifier | TelegramNotifier:
    if DEMO_MODE:
        return _mock
    return TelegramNotifier(os.getenv("TELEGRAM_TOKEN", ""))
