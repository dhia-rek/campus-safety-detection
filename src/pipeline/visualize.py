"""
Debug visualiser — replay a scored scene with an overlaid live graph.

Opens two windows simultaneously:
  1. OpenCV frame window with NORMAL/ANOMALY label + score overlay.
  2. Matplotlib live bullying-score graph (rolling 200-frame window).

Press 'q' in the frame window to quit early.

Usage
-----
    python -m src.pipeline.visualize

    # Override scene
    SCENE=05_0019 python -m src.pipeline.visualize
"""

import os
from collections import deque

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from src.paths import TEST_FRAMES_ROOT, SCORES_ROOT


# ── Config ────────────────────────────────────────────────────────────────────
SCENE       = os.getenv("SCENE", "05_0019")
FRAME_DIR   = TEST_FRAMES_ROOT / SCENE
SCORE_CSV   = SCORES_ROOT / "testing" / "frames" / f"{SCENE}_frames.csv"
FPS         = 15
ANOMALY_TH  = 1.0   # Z-score threshold above which a frame is flagged
GRAPH_WINDOW = 200  # number of frames shown in the rolling graph
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    df = pd.read_csv(SCORE_CSV)
    frame_to_score = dict(zip(df["frame"], df["smoothed_score"]))

    frame_paths = sorted(FRAME_DIR.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {FRAME_DIR}")

    # Live matplotlib graph
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 4))
    x_buf: deque[int]   = deque(maxlen=GRAPH_WINDOW)
    y_buf: deque[float] = deque(maxlen=GRAPH_WINDOW)
    (line,) = ax.plot([], [], lw=2)
    ax.axhline(ANOMALY_TH, color="r", linestyle="--", label=f"threshold={ANOMALY_TH}")
    ax.set_ylim(-2, 6)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Z-score")
    ax.set_title("Live Bullying Score")
    ax.legend()

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        score = frame_to_score.get(i, 0.0)

        # Update rolling graph
        x_buf.append(i)
        y_buf.append(score)
        line.set_data(x_buf, y_buf)
        ax.set_xlim(max(0, i - GRAPH_WINDOW), i + 1)
        plt.pause(0.001)

        # Annotate frame
        if score > ANOMALY_TH:
            label, color = "ANOMALY", (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 6)
        else:
            label, color = "NORMAL", (0, 255, 0)

        cv2.putText(frame, label,              (40, 80),  cv2.FONT_HERSHEY_SIMPLEX, 2,   color,          4)
        cv2.putText(frame, f"Score: {score:.3f}", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.imshow("Live CCTV", frame)

        if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
