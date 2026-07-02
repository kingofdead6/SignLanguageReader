"""
Step 1 — Download the ASL Alphabet dataset from Kaggle and extract
MediaPipe hand keypoints from the images.

Updated for MediaPipe >= 0.10.35 (Tasks API / Python 3.13).

This is the one-time preprocessing step that converts the pixel dataset
into a tiny keypoint dataset. After this, you never touch pixels again.

Dataset: grassknoted/asl-alphabet
  - 29 classes: A-Z + 'space', 'del', 'nothing'
  - ~3000 images per class (200x200 RGB)
  - We subsample per class (default 400) because keypoints need far less data.

Output:
  data/X.npy        (N, 63)  normalized landmark vectors
  data/y.npy        (N,)     integer labels
  data/classes.json          index -> class name mapping

Usage:
  python extract_landmarks.py                 # 400 images per class
  python extract_landmarks.py --per-class 800 # more data, slower
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import kagglehub
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from utils import create_landmarker, landmarks_to_array, normalize_landmarks

random.seed(42)


def find_train_dir(dataset_root: Path) -> Path:
    """Locate the folder that contains the 29 class subfolders."""
    candidates = list(dataset_root.rglob("asl_alphabet_train"))
    for c in candidates:
        subdirs = [d for d in c.iterdir() if d.is_dir()]
        if len(subdirs) >= 25:  # the actual class folders live here
            return c
    # Fallback: any directory with >= 25 subdirectories
    for d in dataset_root.rglob("*"):
        if d.is_dir():
            subdirs = [s for s in d.iterdir() if s.is_dir()]
            if len(subdirs) >= 25:
                return d
    raise FileNotFoundError(
        f"Could not locate class folders under {dataset_root}. "
        "Inspect the download manually."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=400,
                        help="Max images to process per class (default 400)")
    parser.add_argument("--out", type=str, default="data",
                        help="Output directory (default ./data)")
    args = parser.parse_args()

    print("Downloading ASL Alphabet dataset via kagglehub "
          "(~1 GB, cached after first run)...")
    dataset_path = Path(kagglehub.dataset_download("grassknoted/asl-alphabet"))
    train_dir = find_train_dir(dataset_path)
    print(f"Found class folders at: {train_dir}")

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    classes = [d.name for d in class_dirs]
    print(f"{len(classes)} classes: {classes}")

    # IMAGE mode: independent per-image detection, right for datasets.
    # Lower detection threshold: dataset images are sometimes dim/blurry.
    landmarker = create_landmarker(mode="image", num_hands=1,
                                   min_detection_confidence=0.4)

    X, y = [], []
    skipped = 0

    for label_idx, class_dir in enumerate(class_dirs):
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)
        images = images[: args.per_class]

        detected = 0
        for img_path in tqdm(images, desc=f"{class_dir.name:>8}", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(rgb))
            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                # 'nothing' class has no hand by design -> encode as zeros
                if class_dir.name.lower() == "nothing":
                    X.append(np.zeros(63, dtype=np.float32))
                    y.append(label_idx)
                    detected += 1
                else:
                    skipped += 1
                continue

            arr = landmarks_to_array(result.hand_landmarks[0])
            X.append(normalize_landmarks(arr))
            y.append(label_idx)
            detected += 1

        print(f"{class_dir.name:>8}: kept {detected}/{len(images)}")

    landmarker.close()

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    with open(out / "classes.json", "w") as f:
        json.dump(classes, f)

    print(f"\nDone. {len(X)} samples saved to {out}/ "
          f"({skipped} images skipped, no hand detected).")
    print("Next: python train.py")


if __name__ == "__main__":
    main()
