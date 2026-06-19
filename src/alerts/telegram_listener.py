"""
Telegram Bot subscription listener.

Runs as a background process alongside the React dashboard / FastAPI backend.
Handles three bot commands:
  /start  — subscribe the chat to bullying alerts
  /stop   — unsubscribe
  /status — report the number of active subscribers
  /help   — list available commands

Subscriber chat IDs are persisted in subscribers.json at the project root.
The bot token is read from the TELEGRAM_TOKEN environment variable or from
a .env file in the working directory.

Usage
-----
    export TELEGRAM_TOKEN="your-bot-token"
    python -m src.alerts.telegram_listener

    # Or using a .env file:
    python -m src.alerts.telegram_listener   (token read from .env automatically)
"""

# Make project root importable when run as a script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import os
import time

import requests


SUBSCRIBER_FILE  = "subscribers.json"   # relative to project root
POLL_INTERVAL_SEC = 2


def load_env_file(path: str = ".env") -> None:
    """Parse a simple KEY=VALUE .env file and populate os.environ (no-op if missing)."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def get_telegram_token() -> str:
    """Return the bot token from the environment, loading .env as a fallback."""
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if token:
        return token
    load_env_file(".env")
    return os.getenv("TELEGRAM_TOKEN", "").strip()


def load_subscribers() -> set[int]:
    """Read the subscriber list from disk. Returns an empty set if the file is absent."""
    if not os.path.isfile(SUBSCRIBER_FILE):
        return set()
    with open(SUBSCRIBER_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("subscribers", []))


def save_subscribers(subscribers: set[int]) -> None:
    """Persist *subscribers* to disk, sorted for deterministic output."""
    payload = {"subscribers": sorted(int(x) for x in subscribers)}
    with open(SUBSCRIBER_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def send_message(token: str, chat_id: int, text: str) -> None:
    """Send a plain-text message to a Telegram chat."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=20)


def main() -> None:
    token = get_telegram_token()
    if not token:
        raise RuntimeError(
            "TELEGRAM_TOKEN is not set. "
            "Export it as an environment variable or add TELEGRAM_TOKEN=... to a .env file."
        )

    subscribers = load_subscribers()
    print(f"Bot started. Active subscribers: {len(subscribers)}")

    offset: int | None = None

    while True:
        try:
            url    = f"https://api.telegram.org/bot{token}/getUpdates"
            params: dict = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset

            resp = requests.get(url, params=params, timeout=40)
            data = resp.json()

            if not data.get("ok"):
                time.sleep(POLL_INTERVAL_SEC)
                continue

            for upd in data.get("result", []):
                offset  = upd["update_id"] + 1
                msg     = upd.get("message", {})
                chat    = msg.get("chat", {})
                chat_id = chat.get("id")
                text    = (msg.get("text") or "").strip().lower()

                if chat_id is None:
                    continue

                if text.startswith("/start"):
                    subscribers.add(int(chat_id))
                    save_subscribers(subscribers)
                    send_message(token, chat_id, "Subscribed to bullying alerts.")
                    print(f"Subscribed: {chat_id}")

                elif text.startswith("/stop"):
                    subscribers.discard(int(chat_id))
                    save_subscribers(subscribers)
                    send_message(token, chat_id, "Unsubscribed from bullying alerts.")
                    print(f"Unsubscribed: {chat_id}")

                elif text.startswith("/status"):
                    send_message(token, chat_id, f"Active subscribers: {len(subscribers)}")

                elif text.startswith("/help"):
                    send_message(token, chat_id, "Commands:\n/start — subscribe\n/stop — unsubscribe\n/status — subscriber count")

        except KeyboardInterrupt:
            print("Listener stopped by user.")
            break
        except Exception as exc:
            print(f"Listener error: {exc}")
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
