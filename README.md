# AI CCTV Bullying Detection System
### Zero-Shot Video Bullying Detection using CLIP + YOLO on ECE Engineering School Campus

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/CLIP-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-FF6600?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/React-Dashboard-61DAFB?style=for-the-badge&logo=react&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Telegram-Bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white"/>
</p>

<p align="center">
  <b>Detects anomalous events in ECE campus surveillance footage — without training on a single abnormal sample.</b>
</p>

---

## Live Demo

> Real-time bullying detection on ECE campus footage — React dashboard (FastAPI backend) + Telegram Bot alert

> Score spikes to **1.771** as the system detects an ongoing abnormal event across 90 consecutive frames — Telegram Bot simultaneously pushes the annotated frame screenshot to subscribers.

---

## Overview

Traditional CCTV systems require constant human monitoring — inefficient, expensive, and error-prone at scale.  
This project presents a **zero-shot bullying detection framework** that uses vision-language AI to automatically identify suspicious events in real-time surveillance footage.

Instead of training a supervised classifier on labeled abnormal videos, the system uses **natural language descriptions** of abnormal behaviours and compares them against live video frames using the **CLIP model** — no abnormal training data required.

---

## Key Features

| Feature | Detail |
|---|---|
| **Zero-shot detection** | No abnormal training samples needed |
| **Vision-Language AI** | CLIP compares frames against text descriptions of anomalies |
| **Object-level analysis** | YOLO crops individual objects before CLIP scoring |
| **Robust scoring** | Contrastive score → Z-score normalisation → Gaussian temporal smoothing |
| **Real-time dashboard** | React (Vite) live CCTV playback + 3-modality score graph, FastAPI backend |
| **Pluggable agents** | Perception · Coordinator · Action agents as drop-in modules (microkernel registry) |
| **Telegram alerts** | Auto-pushes annotated frame screenshots to subscribers |

---

## Detectable Anomalies

**Visual** (CLIP + text prompts in `prompts/abnormal.txt`):

| Category | Examples |
|---|---|
| **Violence** | Fighting, physical altercation, aggressive behaviour |
| **Dangerous Motion** | Running suddenly, people scattering, panic movement |
| **Unauthorized Vehicles** | Motorbike on campus, car on pedestrian walkway |
| **Object Throwing** | Person throwing a bag or object violently |
| **Chasing** | Person aggressively chasing another |
| **Falling** | Person collapsing suddenly |

**Audio** (PANNs + AudioSet classes in `prompts/violence_audio_classes.txt`):

| Category | AudioSet classes used |
|---|---|
| **Shouting / yelling** | `Shout`, `Yell`, `Children shouting` |
| **Screaming** | `Screaming` |
| **Crying** | `Crying, sobbing` |
| **Physical impact** | `Slap, smack`, `Smash, crash`, `Breaking` |
| **Glass** | `Glass` |
| **Firearms** | `Gunshot, gunfire` |

---

## System Architecture

```
Video Input                                            Audio Input (.wav)
    │                                                       │
    ▼                                                       ▼
Step 0 — Frame Extraction (extract_frames.py)       Step 5 — Audio Scoring (audio_score.py)
    │         Decode .avi → JPEG frames                     │   PANNs CNN14 → sum of
    ▼                                                       │   violence AudioSet probs
Step 1 — YOLO Detection (detect.py)                         │       → per-frame CSV
    │         YOLOv8 → bounding-box JSON sidecars           │
    ▼                                                       │
Step 2 — Object Cropping (crop.py)                          │
    │         Pad & resize crops → 224×224 JPEGs            │
    ▼                                                       │
Step 3 — CLIP Embedding (embed.py + text_features.py)       │
    │         Image crops + text prompts → 512-d            │
    ▼                                                       │
Step 4 — Visual Scoring (score.py)                          │
    │         Contrastive score → Z-score → Gaussian        │
    ▼         smoothing → per-frame CSV                     │
    └──────────────────────────────┬─────────────────────────┘
                                   ▼
                       Step 6 — Score Fusion (fuse.py)
                                   │   max(visual_z, audio_z) per frame
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
   React Dashboard        Telegram Listener   Step 7 — Evaluation
   (web/ + api.py)        (telegram_listener)  (evaluate.py)
   Live playback +        /start /stop         Frame-level ROC-AUC
   3-modality chart +     /status              vs ground-truth masks
   agent decision panel
   + Telegram alerts
```

---

## Repository Structure

```
Bullying_Detection-On-ECE-Campus/
│
├── src/                            # All source code
│   ├── paths.py                    # Centralised path constants (all env-var overridable)
│   │
│   ├── pipeline/                   # ML pipeline — run steps in order
│   │   ├── extract_frames.py       # Step 0: decode training videos → JPEG frames
│   │   ├── detect.py               # Step 1: YOLOv8 object detection → JSON sidecars
│   │   ├── crop.py                 # Step 2: crop + resize detected objects
│   │   ├── text_features.py        # Step 3a: encode text prompts with CLIP
│   │   ├── embed.py                # Step 3b: encode crop images with CLIP
│   │   ├── score.py                # Step 4: contrastive scoring + smoothing → CSV
│   │   ├── audio_score.py          # Step 5: PANNs audio scoring → per-frame CSV
│   │   ├── transcribe.py           # Speech → text (Whisper) + bad-word flagging
│   │   ├── fuse.py                 # Step 6: fuse visual + audio + verbal → fused CSV
│   │   ├── evaluate.py             # Step 7: frame-level ROC-AUC vs ground truth
│   │   └── visualize.py            # Debug: replay a scene with live score overlay
│   │
│   ├── agents/                     # Pluggable multi-agent layer (microkernel)
│   │   ├── base.py                 #   the contract: Observation, Decision, 3 base classes
│   │   ├── registry.py             #   @register + auto-discovery of plugins/
│   │   ├── runtime.py              #   core loop: perception → coordinator → action
│   │   ├── context.py              #   shared scene blackboard
│   │   └── plugins/                #   DROP-IN AGENTS (add a file = add a capability)
│   │       ├── vision.py · sound.py · speech.py       # perception
│   │       ├── coordinator_llm.py                     # coordinator (brain)
│   │       └── action_telegram.py · action_webhook.py # action
│   │
│   ├── dashboard/
│   │   └── api.py                  # FastAPI backend for the React dashboard
│   │
│   └── alerts/
│       └── telegram_listener.py    # Telegram Bot subscription handler
│
├── web/                            # React (Vite) dashboard — audio-first UI
│   └── src/                        #   components, api client, styles
│
├── prompts/
│   ├── normal.txt                       # CLIP text prompts for normal activities
│   ├── abnormal.txt                     # CLIP text prompts for abnormal activities
│   ├── bad_words.txt                    # verbal-abuse / hate-speech keywords
│   └── violence_audio_classes.txt       # AudioSet class names to count as violence
│
├── scripts/
│   └── run_pipeline.sh             # End-to-end pipeline runner (quick / full mode)
│
├── runtime_data/                   # Generated outputs (gitignored — re-creatable)
│   ├── detections/                 # YOLO JSON sidecars
│   ├── crops/                      # 224×224 object crop images
│   ├── features/                   # CLIP .npz embedding files
│   └── scores/                     # Per-frame score CSVs
│       ├── frames/                 #   visual smoothed scores
│       ├── audio/                  #   audio scores per frame
│       └── fused/                  #   fused visual+audio scores
│
├── Videos_bullying/                # ECE campus dataset (gitignored)
│
├── requirements.txt
├── .env.example                    # Template for secrets / path overrides
├── .gitignore
└── LICENSE
```

---

## Getting Started

### 1 — Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Set your Telegram Bot token

```bash
cp .env.example .env
# Edit .env and set TELEGRAM_TOKEN=your-bot-token
```

### 3 — Run the pipeline

**Quick mode** — single test scene (~2-3 min):
```bash
bash scripts/run_pipeline.sh quick
```

**Full mode** — entire dataset (hours):
```bash
bash scripts/run_pipeline.sh full
```

**Or run steps individually:**

```bash
# Step 0 — extract training frames (skip for testing-only runs)
python -m src.pipeline.extract_frames

# Step 1 — detect objects
DATASET_SPLIT=testing SUBDIR_FILTER=05_0019 python -m src.pipeline.detect

# Step 2 — crop objects
DATASET_SPLIT=testing SUBDIR_FILTER=05_0019 python -m src.pipeline.crop

# Step 3 — encode text prompts + image crops
python -m src.pipeline.text_features
DATASET_SPLIT=testing SUBDIR_FILTER=05_0019 python -m src.pipeline.embed

# Step 4 — score bullying timeline (visual)
python -m src.pipeline.score

# Step 5 — score audio (PANNs CNN14 on a .wav overlay)
WAV_PATH=demo.wav SCENE=05_0019 FPS=24 python -m src.pipeline.audio_score

# Step 6 — fuse visual + audio → fused per-frame CSV
SCENE=05_0019 python -m src.pipeline.fuse

# Step 7 — frame-level ROC-AUC evaluation (visual / fused / audio)
SCORE_KIND=fused python -m src.pipeline.evaluate
```

### 4 — Launch the dashboard

The dashboard is a React (Vite) front-end talking to a FastAPI backend.

```bash
# Terminal 1 — Telegram subscription listener
export TELEGRAM_TOKEN="your-bot-token"
python -m src.alerts.telegram_listener

# Terminal 2 — FastAPI backend (serves scores, frames, agent decisions, alerts)
export TELEGRAM_TOKEN="your-bot-token"
uvicorn src.dashboard.api:app --reload --port 8000

# Terminal 3 — React front-end (proxies /api to the backend)
cd web && npm install && npm run dev      # → http://localhost:5173
```

> Optional — the LLM coordinator runs locally via [Ollama](https://ollama.com)
> (`ollama run llama3.1:8b`). If it isn't running, the coordinator falls back to
> a transparent rule-based decision, so the dashboard still works.

### 5 — Add n8n automation

The project already includes an action agent that POSTs high-severity incidents to
`INCIDENT_WEBHOOK_URL`. To wire it into n8n:

1. Start n8n locally. Docker is the easiest reproducible option, but it is not required:

    ```bash
    cd automation/n8n
    docker compose up -d
    ```

    If you do not want Docker, use a local Node.js install instead:

    ```bash
    npx n8n start
    ```

2. Open `http://localhost:5678`, create a workflow, and add a **Webhook** trigger.
    Use the path `campus-incident` and the `POST` method.

3. Add the nodes you want after the trigger, for example:
    Telegram message, e-mail, Slack, Google Sheets, or an incident log.

4. Copy the webhook URL from n8n and set it in `.env`:

    ```bash
    INCIDENT_WEBHOOK_URL=http://localhost:5678/webhook/campus-incident
    ESCALATE_SEVERITIES=high
    ```

5. Restart the backend so the webhook action picks up the new environment.

If you expose n8n beyond localhost, keep it behind authentication and TLS.

The payload sent by the app contains the scene, frame index, the coordinator
decision, and the perception observations. That makes it easy to fan out to any
external workflow without changing the Python code.

---

## Dataset

**ECE Engineering School Campus Surveillance**

| Property | Details |
|---|---|
| Cameras | Multiple campus surveillance cameras |
| Environment | Engineering school campus (indoor + outdoor) |
| Training set | Normal activities only |
| Testing set | Normal + Abnormal activities |
| Ground truth | Frame-level binary masks (0 = Normal, 1 = Abnormal) |
| Evaluation metric | Frame-level ROC-AUC |

---

## Results

| Property | Value |
|---|---|
| Approach | Zero-shot multi-modal (no abnormal training samples) |
| Visual smoothing | Gaussian temporal smoothing (σ = 21 frames) |
| Audio scoring | PANNs CNN14 — 1 s window, 0.5 s hop, sum of violence-class probs |
| Fusion | `max(visual_z, audio_z)` per frame |
| Normalisation | Per-modality z-score (training calibration when available) |
| Evaluation | Frame-level ROC-AUC vs ECE campus ground-truth `test_frame_mask/*.npy` |

To reproduce the evaluation number:

```bash
SCORE_KIND=fused python -m src.pipeline.evaluate
```

The script prints per-scene and pooled ROC-AUC.

---

## Customising Detection

Edit the prompt files to tune what the system flags:

- **`prompts/normal.txt`** — one sentence per line describing normal campus activity
- **`prompts/abnormal.txt`** — one sentence per line describing the anomalies to detect

After editing, regenerate text features:
```bash
python -m src.pipeline.text_features
```

---

## Limitations

- Detection is **prompt-dependent** — behaviours not described in prompts will not be flagged
- No explicit **temporal modelling** — analysis is frame-by-frame, not sequential
- CLIP was not trained specifically for surveillance — performance varies with lighting/angle
- Depends on **YOLO detection quality** — missed detections lead to missed anomalies
- Audio scoring requires a **separate `.wav` overlay** — campus CCTV footage often has no audio track

---

## Future Work

- [ ] Optical flow integration for motion-based bullying detection
- [ ] Transformer-based temporal modelling across frame sequences
- [ ] Adaptive threshold selection per camera / scene
- [ ] LLM-assisted automatic prompt expansion
- [ ] Multi-camera simultaneous monitoring

---

## References

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al., OpenAI  
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211) — Kong et al.

---

## License

MIT License
