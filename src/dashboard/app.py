"""
Streamlit real-time monitoring dashboard.

Replays a scored scene frame-by-frame, plots both visual and audio score streams
alongside the alert threshold, and sends Telegram photo alerts with a cooldown.
A Telegram status panel in the sidebar shows whether the bot token is set and
how many subscribers are currently registered.

Environment variables
---------------------
TELEGRAM_TOKEN : Telegram Bot API token (required for alerts)

Usage
-----
    export TELEGRAM_TOKEN="your-bot-token"
    streamlit run src/dashboard/app.py
"""

# Make project root importable when Streamlit launches this file directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import os
import time

import cv2
import pandas as pd
import requests
import streamlit as st

from src.paths import TEST_FRAMES_ROOT, SCORES_ROOT


# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bullying Detection", page_icon="🚨", layout="wide")

TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "")
SUBSCRIBER_FILE    = "subscribers.json"   # relative to project root (CWD)
ALERT_COOLDOWN_SEC = 10                   # minimum seconds between Telegram alerts
SPLIT              = "testing"

# Friendly score-mode label → (CSV kind, alert column, columns to chart).
SCORE_MODES = {
    "Fused (recommended)": {
        "kind":       "fused",
        "alert_col":  "fused_score",
        "chart_cols": ["visual_z", "audio_z"],
    },
    "Visual only": {
        "kind":       "frames",
        "alert_col":  "smoothed_score",
        "chart_cols": ["smoothed_score"],
    },
    "Audio only": {
        "kind":       "audio",
        "alert_col":  "audio_score",
        "chart_cols": ["audio_score"],
    },
}
# ─────────────────────────────────────────────────────────────────────────────


def send_telegram_alert(frame_path: str, score: float, frame_idx: int, modality: str) -> None:
    """Send an annotated frame photo to all registered Telegram subscribers."""
    try:
        with open(SUBSCRIBER_FILE) as f:
            subscribers = json.load(f).get("subscribers", [])
    except Exception:
        subscribers = []

    if not subscribers:
        print("No subscribers registered")
        return

    caption = (
        "🚨 CCTV BULLYING ALERT 🚨\n\n"
        f"Frame Index: {frame_idx}\n"
        f"Modality: {modality}\n"
        f"Score: {score:.3f}\n"
        "Status: Bullying activity detected"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    for chat_id in subscribers:
        try:
            with open(frame_path, "rb") as img:
                requests.post(
                    url,
                    data={"chat_id": chat_id, "caption": caption},
                    files={"photo": img},
                    timeout=15,
                )
            print(f"Alert sent to {chat_id}")
        except Exception as exc:
            print(f"Telegram error for {chat_id}: {exc}")


def modality_label(row, threshold: float) -> str:
    """Return VISUAL / AUDIO / BOTH / — based on which z-score(s) crossed the threshold."""
    v = float(row.get("visual_z", float("nan")))
    a = float(row.get("audio_z",  float("nan")))
    v_hot = v == v and v >= threshold   # NaN-safe
    a_hot = a == a and a >= threshold
    if v_hot and a_hot:
        return "BOTH"
    if v_hot:
        return "VISUAL"
    if a_hot:
        return "AUDIO"
    return "—"


def discover_scenes(frames_root: Path, split: str) -> list[str]:
    """List scenes that have a frames folder AND at least one matching score CSV."""
    if not frames_root.is_dir():
        return []
    out = []
    for d in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        for spec in SCORE_MODES.values():
            csv = SCORES_ROOT / split / spec["kind"] / f"{d.name}_{spec['kind']}.csv"
            if csv.is_file():
                out.append(d.name)
                break
    return out


def available_modes_for(scene: str, split: str) -> list[str]:
    """Return the human-readable mode labels whose CSV exists for this scene."""
    modes = []
    for label, spec in SCORE_MODES.items():
        csv = SCORES_ROOT / split / spec["kind"] / f"{scene}_{spec['kind']}.csv"
        if csv.is_file():
            modes.append(label)
    return modes


def telegram_status() -> tuple[bool, int]:
    """Return (token_set, n_subscribers)."""
    try:
        with open(SUBSCRIBER_FILE) as f:
            n = len(json.load(f).get("subscribers", []))
    except Exception:
        n = 0
    return bool(TELEGRAM_TOKEN), n


# ── Sidebar: Telegram status ──────────────────────────────────────────────────
token_set, n_subs = telegram_status()
st.sidebar.markdown("### 📡 Telegram")
st.sidebar.markdown(
    f"- Token: {'✅ set' if token_set else '❌ missing'}\n"
    f"- Subscribers: **{n_subs}**"
)
if not token_set:
    st.sidebar.caption("Set TELEGRAM_TOKEN in the shell before launching.")
elif n_subs == 0:
    st.sidebar.caption("Run `telegram_listener.py` and send /start from the bot.")
st.sidebar.divider()


# ── Sidebar: Scene + score mode ───────────────────────────────────────────────
st.sidebar.markdown("### ⚙ Configuration")

scenes = discover_scenes(TEST_FRAMES_ROOT, SPLIT)
if not scenes:
    st.warning(
        f"No scored scenes found under {SCORES_ROOT / SPLIT}. Run the pipeline first."
    )
    st.stop()

scene = st.sidebar.selectbox("🎬 Scene", scenes)

modes = available_modes_for(scene, SPLIT)
mode  = st.sidebar.selectbox("🎯 Score mode", modes)
spec  = SCORE_MODES[mode]
csv_path = SCORES_ROOT / SPLIT / spec["kind"] / f"{scene}_{spec['kind']}.csv"

threshold = st.sidebar.slider("🚨 Threshold", 0.0, 5.0, 1.0)
fps       = st.sidebar.slider("🎥 Playback FPS", 1, 24, 24)

start = st.sidebar.button("▶ Start Monitoring")


# ── Load data ────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(csv_path)
frames = sorted((TEST_FRAMES_ROOT / scene).glob("*.jpg"))

# Reindex onto a dense frame range so iloc[i] always corresponds to frame i.
# Frames missing from the CSV are filled with 0 (no anomaly signal).
df = (
    df_raw.set_index("frame")
          .reindex(range(len(frames)))
          .fillna(0.0)
          .reset_index()
)

has_modalities = {"visual_z", "audio_z"}.issubset(df.columns)
chart_cols     = [c for c in spec["chart_cols"] if c in df.columns]


# ── Layout ───────────────────────────────────────────────────────────────────
left, right = st.columns([1.4, 1])
video_box   = left.empty()
status_box  = left.empty()
right.markdown(f"**{mode}** — score streams vs threshold ({threshold:.2f})")

# Chart columns include the chosen modality streams + a constant Threshold line.
init = {c: [] for c in chart_cols}
init["Threshold"] = []
chart = right.line_chart(pd.DataFrame(init))


# ── Main playback loop ───────────────────────────────────────────────────────
if start:
    last_alert_ts = 0.0

    for i in range(min(len(frames), len(df))):
        row         = df.iloc[i]
        alert_score = float(row[spec["alert_col"]])
        img         = cv2.cvtColor(cv2.imread(str(frames[i])), cv2.COLOR_BGR2RGB)

        video_box.image(img, use_container_width=True)

        if alert_score >= threshold:
            mod = modality_label(row, threshold) if has_modalities else "—"
            status_box.error(
                f"BULLYING DETECTED  [{mod}]\n\nFrame {i}   •   Score {alert_score:.3f}",
                icon="🚨",
            )
            now = time.time()
            if now - last_alert_ts >= ALERT_COOLDOWN_SEC:
                send_telegram_alert(str(frames[i]), alert_score, i, mod)
                last_alert_ts = now
        else:
            status_box.success(
                f"NORMAL ACTIVITY\n\nFrame {i}   •   Score {alert_score:.3f}",
                icon="✅",
            )

        new_row = {c: [float(row[c])] for c in chart_cols}
        new_row["Threshold"] = [threshold]
        chart.add_rows(pd.DataFrame(new_row))
        time.sleep(1 / fps)
