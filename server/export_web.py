"""
Export the trained MLP to a browser-friendly model.json for the static web app.

BatchNorm is folded into the preceding Linear layer (exact for eval-mode
inference), so the browser only needs plain Linear -> ReLU -> Linear -> ReLU ->
Linear layers. The resulting JSON is loaded by web/app.js and runs entirely
client-side — no Python server required.

Folding math (per output unit i):
    bn(y) = gamma_i * (y_i - mean_i) / sqrt(var_i + eps) + beta_i
    with y = W x + b  =>  y'_i = (gamma_i/s_i) * W_i . x
                                 + gamma_i*(b_i - mean_i)/s_i + beta_i
    where s_i = sqrt(var_i + eps).

Usage:
  python export_web.py            # writes ../web/model.json
"""
import json
from pathlib import Path

import numpy as np
import torch

from train import GestureMLP

EPS = 1e-5  # torch BatchNorm1d default


def fold_linear_bn(lin_w, lin_b, bn_w, bn_b, bn_mean, bn_var):
    s = np.sqrt(bn_var + EPS)
    scale = bn_w / s                      # (out,)
    W = lin_w * scale[:, None]            # (out, in)
    b = bn_w * (lin_b - bn_mean) / s + bn_b
    return W, b


def main():
    model_dir = Path("model")
    with open(model_dir / "classes.json") as f:
        classes = json.load(f)

    model = GestureMLP(len(classes))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    sd = {k: v.numpy() for k, v in model.state_dict().items()}

    # Layer 0 (Linear 63->256) folded with BN layer 1
    W0, b0 = fold_linear_bn(
        sd["net.0.weight"], sd["net.0.bias"],
        sd["net.1.weight"], sd["net.1.bias"],
        sd["net.1.running_mean"], sd["net.1.running_var"],
    )
    # Layer 4 (Linear 256->128) folded with BN layer 5
    W1, b1 = fold_linear_bn(
        sd["net.4.weight"], sd["net.4.bias"],
        sd["net.5.weight"], sd["net.5.bias"],
        sd["net.5.running_mean"], sd["net.5.running_var"],
    )
    # Output layer 8 (Linear 128->29), no BN
    W2, b2 = sd["net.8.weight"], sd["net.8.bias"]

    out = {
        "classes": classes,
        # each layer: weight stored row-major (out x in), plus bias
        "layers": [
            {"W": W0.tolist(), "b": b0.tolist(), "act": "relu"},
            {"W": W1.tolist(), "b": b1.tolist(), "act": "relu"},
            {"W": W2.tolist(), "b": b2.tolist(), "act": "none"},
        ],
    }

    # sanity check: compare folded JS-style forward vs the torch model
    _verify(model, out)

    web_dir = Path("..") / "web"
    web_dir.mkdir(exist_ok=True)
    dest = web_dir / "model.json"
    with open(dest, "w") as f:
        json.dump(out, f)
    kb = dest.stat().st_size / 1024
    print(f"Wrote {dest} ({kb:.0f} KB), {len(classes)} classes.")


def _forward_np(spec, x):
    a = x
    for layer in spec["layers"]:
        W = np.array(layer["W"], dtype=np.float32)
        b = np.array(layer["b"], dtype=np.float32)
        a = W @ a + b
        if layer["act"] == "relu":
            a = np.maximum(a, 0.0)
    return a


def _verify(model, spec, n=200):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 63)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = np.stack([_forward_np(spec, x[i]) for i in range(n)])
    max_err = np.abs(ref - got).max()
    assert max_err < 1e-3, f"folding mismatch: {max_err}"
    print(f"Verify OK (max logit error {max_err:.2e}).")


if __name__ == "__main__":
    main()
