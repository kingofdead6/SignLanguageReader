"""
Hosted prediction API — NO camera, NO OpenCV, NO MediaPipe on the server.

The client (browser) runs MediaPipe JS, extracts 21 hand landmarks per frame,
and sends them here. The server only normalizes + runs the MLP. This keeps
hosting cheap: payloads are ~1 KB/frame and inference is a 50k-param MLP.

Endpoints:
  GET  /health     -> {"status": "ok"}  (for uptime checks / Render health)
  POST /predict    -> stateless single prediction (for your own API integrations)
        body: {"landmarks": [[x, y, z] * 21]}
        resp: {"gesture": "A", "confidence": 0.94, "top3": [...]}
  WS   /ws         -> stateful real-time session (debounce + sentence builder,
                      one independent session per connection)
        client sends: {"type": "landmarks", "landmarks": [[x,y,z]*21]}
                      {"type": "empty"}            (no hand this frame)
                      {"type": "clear"}            (reset sentence)
        server replies per frame:
                      {"type": "frame", "gesture": "A", "confidence": 0.94,
                       "sentence": "HELLO", "committed": false}

Deploy notes:
  - Only needs: model/model.pt, model/classes.json, utils.py, train.py
  - Set ALLOWED_ORIGINS below to your frontend domain(s) before production.
  - Run: uvicorn server:app --host 0.0.0.0 --port 8000
    (or python server.py locally)

Usage:
  python server.py         # local test: open http://localhost:8000
"""
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from train import GestureMLP
from utils import normalize_landmarks

# ---------------- Tunables ----------------
WINDOW = 15            # frames in the voting window
STABLE_VOTES = 12      # frames that must agree before committing
CONF_THRESHOLD = 0.80  # mean softmax confidence required
COOLDOWN = 1.2         # seconds before the SAME letter can repeat

# Lock this down to your real frontend origin(s) when you deploy,
# e.g. ["https://yourportfolio.com"]
ALLOWED_ORIGINS = ["*"]
# -------------------------------------------

app = FastAPI(title="Sign Language Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Model (loaded once at startup) ----------------
MODEL_DIR = Path(__file__).resolve().parent / "model"
with open(MODEL_DIR / "classes.json") as f:
    CLASSES = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = GestureMLP(len(CLASSES)).to(DEVICE)
MODEL.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=DEVICE))
MODEL.eval()


def predict_vec(vec: np.ndarray):
    """(63,) normalized vector -> (probs ndarray over classes)."""
    with torch.no_grad():
        logits = MODEL(torch.from_numpy(vec).unsqueeze(0).to(DEVICE))
        return torch.softmax(logits, dim=1)[0].cpu().numpy()


def validate_landmarks(raw) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {arr.shape}")
    return arr


# ---------------- REST: stateless prediction ----------------
class LandmarksIn(BaseModel):
    landmarks: list  # [[x, y, z] * 21]


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(CLASSES)}


@app.post("/predict")
def predict(body: LandmarksIn):
    try:
        arr = validate_landmarks(body.landmarks)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    probs = predict_vec(normalize_landmarks(arr))
    order = np.argsort(probs)[::-1]
    return {
        "gesture": CLASSES[int(order[0])],
        "confidence": round(float(probs[order[0]]), 4),
        "top3": [
            {"gesture": CLASSES[int(i)],
             "confidence": round(float(probs[i]), 4)}
            for i in order[:3]
        ],
    }


# ---------------- WebSocket: stateful real-time session ----------------
class Session:
    """Per-connection debounce + sentence state (no globals — each
    connected client gets an independent session)."""

    def __init__(self):
        self.votes = deque(maxlen=WINDOW)  # (class_idx, confidence)
        self.sentence = ""
        self.last_commit = ("", 0.0)

    def step(self, idx: int, conf: float) -> bool:
        """Register one frame's vote; returns True if a letter committed."""
        self.votes.append((idx, conf))
        if len(self.votes) < WINDOW:
            return False
        top_idx, top_count = Counter(
            i for i, _ in self.votes).most_common(1)[0]
        if top_idx < 0 or top_count < STABLE_VOTES:
            return False
        mean_conf = float(np.mean(
            [c for i, c in self.votes if i == top_idx]))
        letter = CLASSES[top_idx]
        now = time.time()
        is_new = (letter != self.last_commit[0]
                  or now - self.last_commit[1] > COOLDOWN)
        if mean_conf < CONF_THRESHOLD or not is_new:
            return False

        if letter == "space":
            self.sentence += " "
        elif letter == "del":
            self.sentence = self.sentence[:-1]
        elif letter != "nothing":
            self.sentence += letter
        self.last_commit = (letter, now)
        self.votes.clear()  # require re-stabilization
        return True


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")

            if mtype == "clear":
                session.sentence = ""
                await ws.send_text(json.dumps({
                    "type": "frame", "gesture": "nothing",
                    "confidence": 0.0, "sentence": "", "committed": False,
                }))
                continue

            if mtype == "empty":
                committed = session.step(-1, 0.0)
                await ws.send_text(json.dumps({
                    "type": "frame", "gesture": "nothing", "confidence": 0.0,
                    "sentence": session.sentence, "committed": committed,
                }))
                continue

            if mtype == "landmarks":
                try:
                    arr = validate_landmarks(msg.get("landmarks"))
                except ValueError as e:
                    await ws.send_text(json.dumps(
                        {"type": "error", "detail": str(e)}))
                    continue
                probs = predict_vec(normalize_landmarks(arr))
                idx = int(probs.argmax())
                conf = float(probs[idx])
                committed = session.step(idx, conf)
                await ws.send_text(json.dumps({
                    "type": "frame",
                    "gesture": CLASSES[idx],
                    "confidence": round(conf, 3),
                    "sentence": session.sentence,
                    "committed": committed,
                }))
    except WebSocketDisconnect:
        pass


# ---------------- Local convenience: serve the demo page ----------------
@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "index.html")


def _check_websocket_support():
    """uvicorn only handles WebSockets if `websockets` or `wsproto` is
    installed (plain `pip install uvicorn` includes NEITHER — you need
    `pip install uvicorn[standard]`). Without one of them, uvicorn serves
    HTTP fine but silently rejects every WS upgrade, which looks like
    'WebSocket connection failed' in the browser with no obvious cause."""
    for lib in ("websockets", "wsproto"):
        try:
            __import__(lib)
            return lib
        except ImportError:
            continue
    sys.exit(
        "\nERROR: No WebSocket library installed — the /ws endpoint "
        "cannot work.\nFix with ONE of:\n"
        "    pip install websockets\n"
        "    pip install \"uvicorn[standard]\"\n"
        "then run this server again.\n"
    )


if __name__ == "__main__":
    import uvicorn
    ws_lib = _check_websocket_support()
    print(f"WebSocket support: OK (using '{ws_lib}')")
    print("Open the demo at:  http://127.0.0.1:8000")
    print("Health check:      http://127.0.0.1:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000, ws="auto", log_level="info")