"""Generate the synthetic demo data in demo_data/ (deterministic, no real people).

Two made-up scenes drawn with PIL: stick figures on a pavement.
  demo_incident : two figures walk, then scuffle (frames 50-90), then part.
  demo_calm     : figures just walk past each other (a negative control).

The score CSVs are SYNTHETIC curves, not CLIP output: they exist so the dashboard,
agents and alert path can be shown without the dataset or any model. They go through
the real fusion code (src.pipeline.fuse.zscore + max). Also writes assets/demo.gif.

    uv run --with pillow python scripts/make_demo_data.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline.fuse import zscore  # noqa: E402

W, H, N, FPS = 320, 180, 120, 24
SPLIT_DIR = ROOT / "demo_data" / "dataset" / "testing" / "frames"
RUN = ROOT / "demo_data" / "runtime"
SCORES = RUN / "scores" / "testing"


def figure(d: ImageDraw.ImageDraw, x: float, y: float, color, swing: float = 0.0, arms_up: bool = False):
    d.ellipse([x - 7, y - 40, x + 7, y - 26], fill=color)
    d.line([x, y - 26, x, y - 8], fill=color, width=4)
    d.line([x, y - 8, x - 6 + swing, y + 8], fill=color, width=3)
    d.line([x, y - 8, x + 6 - swing, y + 8], fill=color, width=3)
    arm = -14 if arms_up else -4
    d.line([x, y - 22, x - 10 - swing, y + arm], fill=color, width=3)
    d.line([x, y - 22, x + 10 + swing, y + arm], fill=color, width=3)


def draw_frame(scene: str, i: int) -> Image.Image:
    img = Image.new("RGB", (W, H), (96, 120, 92))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 80, W, 150], fill=(120, 120, 126))
    d.line([0, 115, W, 115], fill=(220, 200, 80), width=2)
    rng = random.Random(i)
    walk = math.sin(i * 0.8) * 4
    if scene == "demo_incident":
        if i < 50:
            ax, bx = 40 + i * 2.0, 280 - i * 2.0
            figure(d, ax, 120, (200, 60, 60), walk)
            figure(d, bx, 120, (60, 90, 210), -walk)
        elif i < 90:
            ax, bx = 140 + rng.uniform(-6, 6), 176 + rng.uniform(-6, 6)
            figure(d, ax, 120 + rng.uniform(-3, 3), (200, 60, 60), rng.uniform(-9, 9), True)
            figure(d, bx, 120 + rng.uniform(-3, 3), (60, 90, 210), rng.uniform(-9, 9), True)
        else:
            k = i - 90
            figure(d, 140 - k * 3, 120, (200, 60, 60), walk)
            figure(d, 176 + k * 3, 120, (60, 90, 210), -walk)
    else:
        figure(d, 20 + i * 2.5, 112, (200, 60, 60), walk)
        figure(d, 300 - i * 2.5, 130, (60, 90, 210), -walk)
    d.text((6, 4), f"SYNTHETIC {scene}  f{i}", fill=(255, 255, 255))
    return img


def bump(center: float, width: float, height: float) -> np.ndarray:
    x = np.arange(N)
    return height * np.exp(-((x - center) ** 2) / (2 * width**2))


def make_scene(scene: str) -> list[Image.Image]:
    rng = np.random.default_rng(7 if scene == "demo_incident" else 11)
    fdir = SPLIT_DIR / scene
    fdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(N):
        im = draw_frame(scene, i)
        im.save(fdir / f"{i:05d}.jpg", quality=70)
        frames.append(im)

    incident = scene == "demo_incident"
    visual = rng.normal(0.0, 0.03, N) + (bump(70, 14, 1.6) if incident else bump(60, 25, 0.25))
    audio = np.abs(rng.normal(0.05, 0.03, N)) + bump(72, 10, 0.8) if incident else np.zeros(N)
    verbal = np.zeros(N)
    if incident:
        verbal += bump(76, 4, 1.0)

    df = pd.DataFrame({"frame": range(N), "smoothed_score": visual, "audio_score": audio, "verbal_score": verbal})
    df["visual_z"] = df["smoothed_score"]
    df["audio_z"] = zscore(df["audio_score"].values)
    df["verbal_z"] = zscore(df["verbal_score"].values) if incident else 0.0
    df["fused_score"] = np.maximum.reduce([df["visual_z"], df["audio_z"], df["verbal_z"]])
    out = SCORES / "fused"
    out.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(out / f"{scene}_fused.csv", index=False)
    (SCORES / "frames").mkdir(parents=True, exist_ok=True)
    df[["frame", "smoothed_score"]].round(5).to_csv(SCORES / "frames" / f"{scene}_frames.csv", index=False)
    (SCORES / "audio").mkdir(parents=True, exist_ok=True)
    df[["frame", "audio_score"]].round(5).to_csv(SCORES / "audio" / f"{scene}_audio.csv", index=False)

    tdir = RUN / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    segs = [{"start": 3.0, "end": 3.6, "text": "stop it, you idiot", "bad_words": ["idiot"]}] if incident else []
    (tdir / f"{scene}_transcript.json").write_text(json.dumps({"segments": segs}, indent=2))
    return frames


def make_gif(frames: list[Image.Image]) -> None:
    fused = pd.read_csv(SCORES / "fused" / "demo_incident_fused.csv")["fused_score"].values
    out = []
    for i in range(0, N, 2):
        canvas = Image.new("RGB", (W * 2, H), (18, 20, 26))
        canvas.paste(frames[i], (0, 0))
        d = ImageDraw.Draw(canvas)
        x0, y0, w, h = W + 10, 20, W - 20, H - 50
        pts = [(x0 + w * j / (N - 1), y0 + h - h * max(0, min(fused[j], 2.0)) / 2.0) for j in range(i + 1)]
        if len(pts) > 1:
            d.line(pts, fill=(255, 180, 60), width=2)
        thr_y = y0 + h - h * 1.0 / 2.0
        d.line([x0, thr_y, x0 + w, thr_y], fill=(120, 120, 120))
        d.text((x0, 4), "fused score (synthetic)  threshold=1.0", fill=(200, 200, 200))
        alert = fused[i] >= 1.0
        d.text((W + 10, H - 22), "ALERT -> mock notifier" if alert else "monitoring", fill=(255, 90, 90) if alert else (120, 200, 120))
        out.append(canvas)
    (ROOT / "assets").mkdir(exist_ok=True)
    out[0].save(ROOT / "assets" / "demo.gif", save_all=True, append_images=out[1:], duration=80, loop=0, optimize=True)


if __name__ == "__main__":
    f = make_scene("demo_incident")
    make_scene("demo_calm")
    make_gif(f)
    print("demo data written")
