"""
Shared utilities — updated for MediaPipe >= 0.10.35 (Python 3.13 compatible).

The legacy `mp.solutions.hands` API was removed in recent MediaPipe versions;
everything now goes through the Tasks API (HandLandmarker), which needs a
.task model file. `ensure_model()` downloads it once (~8 MB) and caches it.

Normalization lives here and is used identically at training time and
inference time. If these ever differ, accuracy dies.
"""
import urllib.request
from pathlib import Path

import numpy as np

NUM_LANDMARKS = 21
FEATURES = NUM_LANDMARKS * 3  # 63

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

# MediaPipe 21-landmark hand topology (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def ensure_model() -> str:
    """Download the hand_landmarker.task model once; return its path."""
    if not MODEL_PATH.exists():
        print(f"Downloading hand landmarker model (~8 MB) to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
    return str(MODEL_PATH)


def create_landmarker(mode: str = "video", num_hands: int = 1,
                      min_detection_confidence: float = 0.5):
    """
    Build a HandLandmarker with the new Tasks API.

    mode: "image" for datasets (per-image detection),
          "video" for webcam streams (uses tracking between frames — faster).
    """
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    running_mode = (vision.RunningMode.IMAGE if mode == "image"
                    else vision.RunningMode.VIDEO)
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model()),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_array(landmark_list) -> np.ndarray:
    """
    Tasks-API landmark list (result.hand_landmarks[i], a plain list of
    NormalizedLandmark) -> (21, 3) numpy array.
    """
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list],
        dtype=np.float32,
    )


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize a (21, 3) landmark array so the model is invariant to hand
    position in frame and distance from camera.

    1. Translate so the wrist (landmark 0) is the origin.
    2. Scale by the wrist -> middle-finger-MCP distance (landmark 9),
       a stable anatomical reference.

    Returns a flat (63,) float32 vector.
    """
    pts = landmarks.astype(np.float32).copy()
    pts -= pts[0]  # wrist at origin

    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale

    return pts.flatten()


def mirror_landmarks(flat: np.ndarray) -> np.ndarray:
    """
    Horizontally mirror a flattened (63,) landmark vector (negate x coords).
    Used for augmentation so one model handles both left and right hands
    without relying on MediaPipe's handedness label.
    """
    out = flat.copy().reshape(NUM_LANDMARKS, 3)
    out[:, 0] *= -1.0
    return out.flatten()


def draw_hand(frame, landmark_list, color=(160, 255, 0),
              joint_color=(255, 255, 255)):
    """
    Draw the hand skeleton on a BGR frame with OpenCV.
    (mp.solutions.drawing_utils no longer exists in new MediaPipe versions.)
    """
    import cv2

    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmark_list]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for i, p in enumerate(pts):
        radius = 5 if i % 4 == 0 else 3
        cv2.circle(frame, p, radius, joint_color, -1)
