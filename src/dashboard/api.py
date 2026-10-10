"""FastAPI backend for the React monitoring dashboard.

Exposes the scored scenes, per-frame score streams, frame images, the agent
catalog/decisions, and Telegram alerting as a small JSON/HTTP API that the
React front-end (web/) consumes.

Endpoints
---------
GET  /api/scenes                       → list scenes + available score modes
GET  /api/scene/{scene}?mode=fused     → per-frame score arrays for a scene
GET  /api/frame/{scene}/{idx}.jpg      → a single JPEG frame
GET  /api/telegram/status              → (token_set, n_subscribers)
POST /api/alert                        → push a Telegram photo alert

Environment variables
---------------------
TELEGRAM_TOKEN : Telegram Bot API token (required for alerts)
DEMO_MODE      : 1 = serve synthetic demo_data/ and mock the Telegram notifier

Run
---
    export TELEGRAM_TOKEN="your-bot-token"
    uvicorn src.dashboard.api:app --reload --port 8000
"""

import json
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.alerts.notifier import get_notifier
from src.paths import RUNTIME_ROOT, SCORES_ROOT, TEST_FRAMES_ROOT

# ── Config ────────────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SEC = 10
SPLIT              = "testing"

# Friendly score-mode label → (CSV kind, alert column, columns to chart).
SCORE_MODES = {
    # audio-first ordering: sound + verbal lead, vision is context.
    "fused":  {"alert_col": "fused_score",     "chart_cols": ["audio_z", "verbal_z", "visual_z"]},
    "frames": {"alert_col": "smoothed_score",  "chart_cols": ["smoothed_score"]},
    "audio":  {"alert_col": "audio_score",     "chart_cols": ["audio_score"]},
}
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Campus Safety Detection API")

# Vite dev server runs on 5173; allow it (and any localhost) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_last_alert_ts = 0.0


def _csv_path(scene: str, kind: str) -> Path:
    """Path of a scene's score CSV of the given kind (``fused``, ``frames`` or ``audio``)."""
    return SCORES_ROOT / SPLIT / kind / f"{scene}_{kind}.csv"


@lru_cache(maxsize=64)
def _scene_frames(scene: str) -> list[str]:
    """Sorted list of absolute frame-image paths for a scene (cached)."""
    return [str(p) for p in sorted((TEST_FRAMES_ROOT / scene).glob("*.jpg"))]


def _available_modes(scene: str) -> list[str]:
    """Score modes for which the scene has a CSV on disk."""
    return [kind for kind in SCORE_MODES if _csv_path(scene, kind).is_file()]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/scenes")
def list_scenes() -> dict:
    """Every scene that has frames AND a fused score CSV.

    Returns:
        ``{"scenes": [{scene, modes, n_frames, audio}, ...]}`` where ``audio``
        says whether an extracted .wav exists.
    """
    if not TEST_FRAMES_ROOT.is_dir():
        return {"scenes": []}
    scenes = []
    for d in sorted(p for p in TEST_FRAMES_ROOT.iterdir() if p.is_dir()):
        modes = _available_modes(d.name)
        # Only list fully-processed scenes — the dashboard needs the fused view,
        # so a half-scored scene (e.g. frames-only leftover) never shows up broken.
        if "fused" in modes:
            audio = (RUNTIME_ROOT / "clips" / f"{d.name}.wav").is_file()
            scenes.append({
                "scene": d.name, "modes": modes,
                "n_frames": len(_scene_frames(d.name)), "audio": audio,
            })
    return {"scenes": scenes}


@app.get("/api/scene/{scene}")
def scene_scores(scene: str, mode: str = "fused") -> dict:
    """Per-frame score streams for *scene*, densified onto frame 0..N-1.

    Frames missing from the CSV are filled with 0.0 so every series has exactly
    one value per frame image and the chart lines up with the video.

    Args:
        scene: Scene id.
        mode: One of ``SCORE_MODES`` (``fused``, ``frames``, ``audio``).

    Raises:
        HTTPException: 400 for an unknown mode; 404 if the CSV or frames are missing.
    """
    if mode not in SCORE_MODES:
        raise HTTPException(400, f"Unknown mode '{mode}'")
    csv = _csv_path(scene, mode)
    if not csv.is_file():
        raise HTTPException(404, f"No '{mode}' scores for scene '{scene}'")

    frames = _scene_frames(scene)
    if not frames:
        raise HTTPException(404, f"No frames for scene '{scene}'")

    df = (
        pd.read_csv(csv)
          .set_index("frame")
          .reindex(range(len(frames)))
          .fillna(0.0)
          .reset_index()
    )

    spec = SCORE_MODES[mode]
    cols = [c for c in [*spec["chart_cols"], spec["alert_col"]] if c in df.columns]
    series = {c: df[c].round(4).tolist() for c in cols}

    wav = RUNTIME_ROOT / "clips" / f"{scene}.wav"
    return {
        "scene":      scene,
        "mode":       mode,
        "n_frames":   len(frames),
        "alert_col":  spec["alert_col"],
        "chart_cols": [c for c in spec["chart_cols"] if c in df.columns],
        "has_audio":  "audio_z" in df.columns or "audio_score" in df.columns,
        "has_audio_clip": wav.is_file(),     # a playable .wav exists for this scene
        "series":     series,
    }


@app.get("/api/audio/{scene}.wav")
def scene_audio(scene: str) -> FileResponse:
    """Serve the scene's extracted audio track (if the clip had one).

    Raises:
        HTTPException: 404 if no .wav exists.
    """
    wav = RUNTIME_ROOT / "clips" / f"{scene}.wav"
    if not wav.is_file():
        raise HTTPException(404, "No audio for this scene")
    return FileResponse(str(wav), media_type="audio/wav")


@app.get("/api/frame/{scene}/{idx}.jpg")
def frame_image(scene: str, idx: int) -> FileResponse:
    """Serve the *idx*-th frame of a scene as JPEG.

    Raises:
        HTTPException: 404 if *idx* is out of range.
    """
    frames = _scene_frames(scene)
    if not (0 <= idx < len(frames)):
        raise HTTPException(404, "Frame index out of range")
    return FileResponse(frames[idx], media_type="image/jpeg")


@app.get("/api/detections/{scene}/{idx}")
def frame_detections(scene: str, idx: int) -> dict:
    """YOLO bounding boxes for one frame (for the live video overlay).

    Boxes are in the frame's native pixel coords: {label, conf, bbox:[x1,y1,x2,y2]}.
    Empty list if the scene hasn't been run through detect.py or *idx* is out of range.
    """
    from src.paths import DETECTIONS_ROOT
    frames = _scene_frames(scene)
    if not (0 <= idx < len(frames)):
        return {"boxes": []}
    stem = Path(frames[idx]).stem
    path = DETECTIONS_ROOT / SPLIT / scene / f"{stem}.json"
    if not path.is_file():
        return {"boxes": []}
    return {"boxes": json.loads(path.read_text())}


@app.get("/api/transcript/{scene}")
def scene_transcript(scene: str) -> dict:
    """Whisper transcript segments for a scene (empty if not transcribed)."""
    path = Path(str(RUNTIME_ROOT)) / "transcripts" / f"{scene}_transcript.json"
    if not path.is_file():
        return {"scene": scene, "segments": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"scene": scene, "segments": data.get("segments", [])}


@app.get("/api/progress")
def batch_progress() -> dict:
    """Report live batch-processing progress for the UI bar.

    Counts testing scenes that are detected (embedded) and fully ready (fused),
    out of the total.
    """
    from src.paths import FEATURES_ROOT
    total = sum(1 for p in TEST_FRAMES_ROOT.iterdir() if p.is_dir()) if TEST_FRAMES_ROOT.is_dir() else 0
    feat_dir = FEATURES_ROOT / SPLIT
    embedded = len(list(feat_dir.glob("*.npz"))) if feat_dir.is_dir() else 0
    fused_dir = SCORES_ROOT / SPLIT / "fused"
    ready = len(list(fused_dir.glob("*_fused.csv"))) if fused_dir.is_dir() else 0
    return {"total": total, "embedded": embedded, "ready": ready}


@app.get("/api/agents")
def agents_catalog() -> dict:
    """Return the installed agent catalog (perception / coordinator / action).

    Feeds the UI grid and admin console with each agent's role, tier and enabled state.
    """
    from src.agents import registry
    return {"agents": registry.catalog()}


def _decide(scene: str, frame: int | None, threshold: float, run_actions: bool) -> dict:
    """Load a scene and run the agent pipeline on one frame.

    Args:
        scene: Scene id.
        frame: Frame index; None means the scene's peak fused frame.
        threshold: Z-score alert threshold.
        run_actions: False for a dry run (no action agent is called).

    Raises:
        HTTPException: 404 if the scene has no fused scores.
    """
    from src.agents.context import load_scene_context, peak_frame
    from src.agents.runtime import analyze_frame

    try:
        ctx = load_scene_context(scene, SPLIT, threshold)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    f = peak_frame(ctx) if frame is None else frame
    return analyze_frame(ctx, f, run_actions=run_actions)


@app.get("/api/decision/{scene}")
def coordinator_decision(scene: str, frame: int | None = None, threshold: float = 1.0) -> dict:
    """Dry run: perception and coordinator only, with no side effects.

    Safe to call repeatedly (GET must not send alerts). ``actions`` is always
    empty; use ``POST /api/decision/{scene}/execute`` to also run the actions.
    Defaults to the scene's peak fused frame.

    Raises:
        HTTPException: 404 if the scene has no fused scores.
    """
    return _decide(scene, frame, threshold, run_actions=False)


@app.post("/api/decision/{scene}/execute")
def coordinator_decision_execute(scene: str, frame: int | None = None, threshold: float = 1.0) -> dict:
    """Run the full agent pipeline including the action agents (Telegram, e-mail, webhook...).

    Unlike the GET dry run this can send real alerts.

    Raises:
        HTTPException: 404 if the scene has no fused scores.
    """
    return _decide(scene, frame, threshold, run_actions=True)


@app.get("/api/telegram/status")
def telegram_status() -> dict:
    """Report whether alerting is configured, the subscriber count, and whether it is the mock."""
    notifier = get_notifier()
    return {"token_set": notifier.configured, "subscribers": len(notifier.subscribers()), "mock": notifier.mock}


class AlertRequest(BaseModel):
    """Body of ``POST /api/alert``: which frame/score the operator wants to flag."""

    scene: str
    frame: int
    score: float
    modality: str = "AUDIO"


@app.post("/api/alert")
def send_alert(req: AlertRequest) -> dict:
    """Manual alert button: push an annotated frame to subscribers (with cooldown).

    Reuses the Telegram action agent's send path so the manual and automatic
    alert routes are identical.

    Raises:
        HTTPException: 400 if no Telegram token is configured (mock counts as configured).

    Returns:
        The send result, or ``{"sent": 0, "skipped": "cooldown"}``. The cooldown
        is separate from the Telegram action agent's own cooldown.
    """
    global _last_alert_ts

    if not get_notifier().configured:
        raise HTTPException(400, "TELEGRAM_TOKEN not set")

    now = time.time()
    if now - _last_alert_ts < ALERT_COOLDOWN_SEC:
        return {"sent": 0, "skipped": "cooldown"}

    from src.agents.plugins.action_telegram import send_frame_photo

    caption = (
        "🚨 CCTV ANOMALY ALERT 🚨\n\n"
        f"Scene: {req.scene}\n"
        f"Frame Index: {req.frame}\n"
        f"Modality: {req.modality}\n"
        f"Score: {req.score:.3f}\n"
        "Status: Abnormal activity detected"
    )
    result = send_frame_photo(req.scene, req.frame, caption, SPLIT)
    if result.get("sent"):
        _last_alert_ts = now
    return result


# Serve the built React dashboard (web/dist) when present, e.g. in the Docker / Hugging Face image.
_WEB_DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"
if _WEB_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
