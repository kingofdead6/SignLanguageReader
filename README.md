# Real-Time Sign Language → Text

ASL alphabet recognition from a webcam, built on keypoints instead of pixels.
MediaPipe's HandLandmarker extracts 21 hand landmarks per frame; a
~50k-parameter MLP classifies them into 29 classes (A–Z + `space`, `del`,
`nothing`). Trains on CPU in under a minute. Streams predictions to a browser
UI over WebSockets.

**Python 3.13 compatible** — uses MediaPipe >= 0.10.35 with the new Tasks API
(the legacy `mp.solutions` API was removed in recent MediaPipe versions).
The required `hand_landmarker.task` model (~8 MB) downloads automatically on
first run.

## Architecture

```
Kaggle ASL Alphabet dataset (images, one-time)
        │
        ▼
extract_landmarks.py ── HandLandmarker (IMAGE mode) ──► data/X.npy (N×63), data/y.npy
        │
        ▼
train.py ── MLP (63→256→128→29) + mirror/noise augmentation ──► model/model.pt
        │
        ├──► realtime.py     OpenCV desktop demo (VIDEO mode, skeleton overlay, sentence builder)
        ├──► server.py       FastAPI WebSocket demo (server-side camera; localhost only)
        └──► export_web.py ── folds BatchNorm into Linear ──► web/model.json
                 │
                 ▼
             web/  (index.html + app.js)  fully client-side browser app:
             MediaPipe HandLandmarker WASM + JS MLP, uses the BROWSER's webcam,
             no Python server, deployable as a static site
```

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate    Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Run (in order)

```bash
# 1. Download dataset + extract keypoints (~1 GB download, cached; ~10 min processing)
python extract_landmarks.py            # or --per-class 800 for more data

# 2. Train (< 1 minute on CPU, expect ~97-99% val accuracy)
python train.py

# 3a. Desktop demo
python realtime.py                     # q = quit, c = clear sentence

# 3b. Web demo — server-side camera (localhost only)
python server.py                       # open http://localhost:8000

# 3c. Web demo — fully client-side (uses the browser's own webcam, deployable)
python export_web.py                   # weights -> web/model.json (BN folded in)
cd web && python -m http.server 8000   # open http://localhost:8000
```

### Two "web" versions — which one?

- **`server.py`** streams keypoints over WebSockets but runs the camera and the
  model **on the server** (`cv2.VideoCapture(0)`). The browser is only a display,
  so it only works where the camera is physically attached (i.e. localhost).
- **`web/`** is the real web app: MediaPipe's HandLandmarker WASM build runs in
  the browser using **your** webcam, and the classifier is the same trained MLP
  exported to `web/model.json` and evaluated in JavaScript (BatchNorm folded into
  the Linear layers, so it's `Linear→ReLU` only — verified to match PyTorch to
  ~1e-6). Nothing but static files; host it anywhere (GitHub Pages, Netlify, …).
  Note: browsers only grant camera access over `http://localhost` or `https://`,
  not `file://`, so serve it rather than double-clicking `index.html`.

## How committing a letter works (debounce)

A prediction is only appended to the sentence when, within a 15-frame window:
- the same class wins ≥ 12 frames, AND
- its mean confidence ≥ 0.80, AND
- it differs from the last committed letter OR 1.2 s cooldown has passed.

Show `space` to add a space, `del` to backspace, drop your hand (`nothing`)
between letters to reset the window. Tune `WINDOW`, `STABLE_VOTES`,
`CONF_THRESHOLD`, `COOLDOWN` at the top of `realtime.py` / `server.py`.

## MediaPipe API notes (new Tasks API)

- `create_landmarker()` in `utils.py` wraps `vision.HandLandmarker`:
  IMAGE mode for dataset extraction, VIDEO mode for the webcam
  (`detect_for_video()` requires a monotonically increasing timestamp in ms).
- Results live in `result.hand_landmarks` — a list per hand of 21
  NormalizedLandmark objects with `.x .y .z`.
- `mp.solutions.drawing_utils` no longer exists; `utils.draw_hand()` draws
  the skeleton manually with OpenCV using the standard 21-point topology.

## Known limitations (be upfront about these in your portfolio)

- **J and Z involve motion** — this static single-frame model approximates them
  from the final hand pose; accuracy on those two is lower by design.
- **M / N / S / T / E** are visually similar fists; expect occasional confusion.
- Dataset was recorded in fixed lighting with limited signers — because we use
  normalized keypoints (not pixels) it generalizes far better than a CNN would,
  but your own hand may still differ from the dataset's signing style. If a
  letter consistently misfires, compare your hand shape to dataset samples.
- This is fingerspelling recognition, not full ASL translation (no grammar,
  facial expressions, or continuous signing).

## Kaggle note

`kagglehub` downloads public datasets without credentials. If you hit an auth
error, create a free Kaggle account, generate an API token (Account → API →
Create New Token), and place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or
`C:\Users\<you>\.kaggle\` (Windows).
