"""
Hosted prediction API — NO camera, NO OpenCV, NO MediaPipe on the server.

The client (browser) runs MediaPipe JS, extracts 21 hand landmarks per frame,
and sends them here. The server only normalizes + runs the MLP. This keeps
hosting cheap: payloads are ~1 KB/frame and inference is a 50k-param MLP.

Endpoints:
  GET  /health     -> {"status": "ok", ...live stats}  (uptime checks +
                      remote traffic monitoring: ws_sessions_total,
                      ws_active_now, frames_processed)
  POST /predict    -> stateless single prediction (for your own API integrations)
        body: {"landmarks": [[x, y, z] * 21]}
        resp: {"gesture": "A", "confidence": 0.94, "top3": [...]}
  WS   /ws         -> stateful real-time session (debounce + sentence builder,
                      one independent session per connection)
        server sends FIRST, on connect:
                      {"type": "hello", "server": "sign-language-api",
                       "session": 1, "classes": 29}
        client sends: {"type": "landmarks", "landmarks": [[x,y,z]*21]}
                      {"type": "empty"}            (no hand this frame)
                      {"type": "clear"}            (reset sentence)
        server replies per frame:
                      {"type": "frame", "gesture": "A", "confidence": 0.94,
                       "sentence": "HELLO", "committed": false}

Deploy notes (Render / Railway / any container host):
  - Needs: model/model.pt, model/classes.json, model/poses.json,
    utils.py, train.py
  - requirements.txt MUST have "uvicorn[standard]" (plain uvicorn serves
    HTTP but silently rejects every WebSocket upgrade).
  - Start command:  uvicorn server:app --host 0.0.0.0 --port $PORT
  - Env vars:
        ALLOWED_ORIGINS=https://yourfrontend.com,https://www.yourfrontend.com
        PORT=<injected by the host; local default 8000>
  - Every WS connection prints an unmissable [CONNECT] banner in the host
    logs, and /health exposes live counters you can check from any browser.

Usage:
  python server.py         # local test: open http://localhost:8000
"""
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from train import GestureMLP
from utils import normalize_landmarks

# ---------------- Tunables ----------------
WINDOW = 15            # frames in the voting window
STABLE_VOTES = 12      # frames that must agree before committing
CONF_THRESHOLD = 0.80  # mean softmax confidence required
COOLDOWN = 1.2         # seconds before the SAME letter can repeat

# CORS — set on your host, e.g.
#   ALLOWED_ORIGINS=https://yourportfolio.com,https://www.yourportfolio.com
# Falls back to * so local dev needs zero setup.
ALLOWED_ORIGINS = [o.strip()
                   for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
                   if o.strip()]

# Live counters — printed on every connect, exposed on /health so you can
# check for traffic from any browser without opening the host's log tab.
STATS = {"started": time.time(), "ws_total": 0,
         "ws_active": 0, "frames_total": 0}
# -------------------------------------------

DEBUG = True   # set False to silence [debug] prints


def dbg(*args):
    if DEBUG:
        print("[debug]", *args, flush=True)


class ConnectionLogger:
    """Pure ASGI middleware: logs EVERY incoming http/websocket attempt,
    including ones that fail before reaching a route."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            client = scope.get("client") or ("?", "?")
            dbg(f"{scope['type'].upper():9s} {scope.get('path')} "
                f"from {client[0]}:{client[1]}")
        await self.app(scope, receive, send)


app = FastAPI(title="Sign Language Prediction API")
app.add_middleware(ConnectionLogger)


@app.on_event("shutdown")
async def _log_shutdown():
    # If you see this without wanting to stop the server: on Windows,
    # Ctrl+C (even to COPY terminal text) kills the process, and clicking
    # inside the console can pause it (press Esc to resume).
    print("[lifespan] SHUTDOWN signal received (Ctrl+C / SIGTERM / window "
          "closed) — the server is stopping NOW.", flush=True)


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
    return {
        "status": "ok",
        "classes": len(CLASSES),
        "uptime_s": round(time.time() - STATS["started"]),
        "ws_sessions_total": STATS["ws_total"],
        "ws_active_now": STATS["ws_active"],
        "frames_processed": STATS["frames_total"],
    }


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
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
    origin = ws.headers.get("origin", "-")
    await ws.accept()

    STATS["ws_total"] += 1
    STATS["ws_active"] += 1
    session_id = STATS["ws_total"]
    t0 = time.time()

    # ---- THE SIGN: unmissable in Render/host logs ----
    print("=" * 56, flush=True)
    print(f"[CONNECT] session #{session_id}", flush=True)
    print(f"[CONNECT] client : {client}", flush=True)
    print(f"[CONNECT] origin : {origin}", flush=True)
    print(f"[CONNECT] active : {STATS['ws_active']} session(s) now",
          flush=True)
    print("=" * 56, flush=True)

    # Mirror it to the client so the UI proves it reached the real API
    await ws.send_text(json.dumps({
        "type": "hello",
        "server": "sign-language-api",
        "session": session_id,
        "classes": len(CLASSES),
    }))

    session = Session()
    n_msgs = 0
    try:
        while True:
            raw = await ws.receive_text()
            n_msgs += 1
            if n_msgs == 1 or n_msgs % 100 == 0:
                dbg(f"WS {client}: {n_msgs} messages received "
                    f"(sentence='{session.sentence}')")
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
                STATS["frames_total"] += 1
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
    except WebSocketDisconnect as e:
        dbg(f"WS {client} disconnected cleanly (code={e.code})")
    except Exception as e:
        import traceback
        print(f"[debug] WS {client} CRASHED: {type(e).__name__}: {e}",
              flush=True)
        traceback.print_exc()
    finally:
        STATS["ws_active"] -= 1
        print(f"[DISCONNECT] session #{session_id}  client={client}  "
              f"lasted={round(time.time() - t0, 1)}s  msgs={n_msgs}  "
              f"active_now={STATS['ws_active']}", flush=True)


# ---------------- Text -> Sign: letter poses for the client animation ----
# Animation timing lives SERVER-side so every client renders consistently.
SIGN_TRANSITION_MS = 380   # morph between letters
SIGN_HOLD_MS = 680         # hold each letter

_POSES_CACHE: dict = {}


def load_poses() -> dict:
    """Load model/poses.json once and cache. 404 with instructions if absent."""
    if _POSES_CACHE:
        return _POSES_CACHE
    path = MODEL_DIR / "poses.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="poses.json not found — run `python export_poses.py` "
                   "after extract_landmarks.py, then restart the server.")
    with open(path) as f:
        _POSES_CACHE.update(json.load(f))
    return _POSES_CACHE


def _ease(t: float) -> float:
    """Cubic in-out — same curve the reference client uses."""
    return 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


@app.get("/poses")
def poses():
    return load_poses()


class SignIn(BaseModel):
    text: str
    fps: int | None = None   # if set, server bakes interpolated frames


@app.post("/sign")
def sign(body: SignIn):
    """
    Convert text into a signing animation.

    Default response: compact letter sequence + timing — the client
    interpolates between poses itself.

    With "fps" (e.g. 30): additionally returns fully-baked "frames" —
    every interpolated hand pose with timestamps. A renderer then needs
    ZERO animation logic: draw frame i, wait 1000/fps ms, repeat.
    """
    all_poses = load_poses()
    rest = all_poses.get("B") or next(iter(all_poses.values()))

    sequence, skipped = [], []
    for ch in body.text.upper():
        if ch == " " and "space" in all_poses:
            sequence.append({"ch": " ", "pose": all_poses["space"]})
        elif ch in all_poses:
            sequence.append({"ch": ch, "pose": all_poses[ch]})
        elif ch.strip():
            skipped.append(ch)

    if not sequence:
        raise HTTPException(status_code=422,
                            detail="No signable characters in text.")

    resp = {
        "rest_pose": rest,
        "transition_ms": SIGN_TRANSITION_MS,
        "hold_ms": SIGN_HOLD_MS,
        "sequence": sequence,
        "skipped": skipped,
    }

    if body.fps:
        fps = max(5, min(int(body.fps), 60))
        step = 1000.0 / fps
        frames, t = [], 0.0
        prev = rest
        steps_tr = max(1, round(SIGN_TRANSITION_MS / step))
        steps_hold = max(1, round(SIGN_HOLD_MS / step))
        chain = sequence + [{"ch": None, "pose": rest}]  # ease back to rest
        for item in chain:
            target = item["pose"]
            for i in range(1, steps_tr + 1):        # transition
                k = _ease(i / steps_tr)
                frames.append({
                    "t_ms": round(t), "ch": item["ch"],
                    "pose": [[round(a[0] + (b[0] - a[0]) * k, 4),
                              round(a[1] + (b[1] - a[1]) * k, 4)]
                             for a, b in zip(prev, target)],
                })
                t += step
            hold_pose = [[round(p[0], 4), round(p[1], 4)] for p in target]
            hold_steps = steps_hold if item["ch"] is not None else 1
            for _ in range(hold_steps):             # hold
                frames.append({"t_ms": round(t), "ch": item["ch"],
                               "pose": hold_pose})
                t += step
            prev = target
        resp["fps"] = fps
        resp["frames"] = frames

    return resp


# ---------------- API root ----------------
@app.get("/")
def index():
    return {
        "name": "Sign Language Prediction API",
        "endpoints": ["/health", "/poses", "/predict", "/sign", "/ws"],
        "frontend": "run the React app in ../frontend (npm run dev)",
    }


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
    import platform
    import uvicorn
    ws_lib = _check_websocket_support()
    ws_mod = __import__(ws_lib)
    port = int(os.environ.get("PORT", 8000))
    print("=" * 56)
    print(f"Python      : {platform.python_version()}  ({sys.executable})")
    print(f"uvicorn     : {uvicorn.__version__}")
    print(f"{ws_lib:<12}: {getattr(ws_mod, '__version__', '?')}")
    print(f"torch       : {torch.__version__}  (device: {DEVICE})")
    print(f"classes     : {len(CLASSES)}")
    print(f"CORS origins: {ALLOWED_ORIGINS}")
    print("=" * 56)
    print(f"WebSocket support: OK (using '{ws_lib}')")
    print(f"Open the demo at:  http://127.0.0.1:{port}")
    print(f"Health check:      http://127.0.0.1:{port}/health")
    uvicorn.run(app, host="0.0.0.0", port=port, ws="auto",
                log_level="info")