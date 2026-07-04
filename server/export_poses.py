"""
Export one representative hand pose per class from the extracted keypoints,
for the Text -> Sign animation (the inverse direction).

For each class we pick the MEDOID: the real sample whose distance to the
class mean is smallest. Unlike the raw mean, a medoid is guaranteed to be an
anatomically valid hand pose (averaging can blend poses into mush).

Run AFTER extract_landmarks.py:
  python export_poses.py

Output:
  model/poses.json    {"A": [[x, y, z] * 21], "B": ..., ...}
                      (normalized coords: wrist at origin,
                       wrist->middle-MCP distance = 1)
"""
import json
from pathlib import Path

import numpy as np


def main():
    data = Path("data")
    X = np.load(data / "X.npy")          # (N, 63) normalized vectors
    y = np.load(data / "y.npy")
    with open(data / "classes.json") as f:
        classes = json.load(f)

    poses = {}
    for idx, name in enumerate(classes):
        if name.lower() == "nothing":
            continue  # zeros by construction — not a drawable pose
        vecs = X[y == idx]
        if len(vecs) == 0:
            print(f"WARNING: no samples for class '{name}', skipping")
            continue
        mean = vecs.mean(axis=0)
        medoid_i = int(np.linalg.norm(vecs - mean, axis=1).argmin())
        pose = vecs[medoid_i].reshape(21, 3)
        poses[name] = [[round(float(v), 4) for v in p] for p in pose]

    out = Path("model")
    out.mkdir(exist_ok=True)
    with open(out / "poses.json", "w") as f:
        json.dump(poses, f)

    print(f"Exported {len(poses)} poses to model/poses.json: "
          f"{sorted(poses.keys())}")
    print("The server's GET /poses endpoint will now serve them.")


if __name__ == "__main__":
    main()