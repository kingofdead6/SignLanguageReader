"""
Step 3 — Real-time recognition from your webcam (desktop demo).

Updated for MediaPipe >= 0.10.35 (Tasks API / Python 3.13):
  - HandLandmarker in VIDEO mode with detect_for_video() + timestamps
  - Manual OpenCV skeleton drawing (drawing_utils no longer exists)

Pipeline per frame:
  webcam -> HandLandmarker -> normalize -> MLP -> softmax
  -> temporal debounce -> committed letter -> sentence

Controls:
  q      quit
  c      clear sentence
  Show 'space' sign  -> adds a space
  Show 'del' sign    -> deletes last character

Usage:
  python realtime.py
"""
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from train import GestureMLP
from utils import (create_landmarker, draw_hand, landmarks_to_array,
                   normalize_landmarks)

# ---------------- Tunables ----------------
WINDOW = 15            # frames in the voting window (~0.5 s at 30 fps)
STABLE_VOTES = 12      # frames that must agree before committing
CONF_THRESHOLD = 0.80  # mean softmax confidence required
COOLDOWN = 1.2         # seconds before the SAME letter can repeat
# -------------------------------------------


def main():
    model_dir = Path("model")
    with open(model_dir / "classes.json") as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureMLP(len(classes)).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt",
                                     map_location=device))
    model.eval()

    # VIDEO mode: tracks the hand between frames — faster and smoother
    landmarker = create_landmarker(mode="video", num_hands=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0).")

    votes = deque(maxlen=WINDOW)   # (class_idx, confidence) pairs
    sentence = ""
    last_commit = ("", 0.0)        # (letter, timestamp)
    t0 = time.monotonic()

    print("Running. 'q' quit, 'c' clear sentence.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # selfie mirror — feels natural
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        # detect_for_video needs a monotonically increasing timestamp in ms
        timestamp_ms = int((time.monotonic() - t0) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        label, conf = "nothing", 0.0

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]
            draw_hand(frame, hand_lms)
            vec = normalize_landmarks(landmarks_to_array(hand_lms))
            with torch.no_grad():
                logits = model(torch.from_numpy(vec).unsqueeze(0).to(device))
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            idx = int(probs.argmax())
            label, conf = classes[idx], float(probs[idx])
            votes.append((idx, conf))
        else:
            votes.append((-1, 0.0))  # -1 = no hand

        # ---- Debounce: commit only when the window agrees confidently ----
        if len(votes) == WINDOW:
            idx_counts = Counter(i for i, _ in votes)
            top_idx, top_count = idx_counts.most_common(1)[0]
            if top_idx >= 0 and top_count >= STABLE_VOTES:
                confs = [c for i, c in votes if i == top_idx]
                mean_conf = float(np.mean(confs))
                letter = classes[top_idx]
                now = time.time()
                is_new = (letter != last_commit[0]
                          or now - last_commit[1] > COOLDOWN)
                if mean_conf >= CONF_THRESHOLD and is_new:
                    if letter == "space":
                        sentence += " "
                    elif letter == "del":
                        sentence = sentence[:-1]
                    elif letter != "nothing":
                        sentence += letter
                    last_commit = (letter, now)
                    votes.clear()  # require re-stabilization

        # ---------------------- UI overlay ----------------------
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
        cv2.putText(frame, f"{label}  {conf*100:.0f}%", (12, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 160), 2)
        bar = int(conf * (w - 24))
        cv2.rectangle(frame, (12, 58), (12 + bar, 64), (0, 255, 160), -1)

        cv2.rectangle(frame, (0, h - 60), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, sentence[-40:] or "...", (12, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Sign Language -> Text", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            sentence = ""

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
