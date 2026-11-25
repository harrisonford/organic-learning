"""Exploratory (2026-11-25): what the cortex's first layer gets from pictures blown up from 32 to 64 pixels.
Window statistics (5x5x3, mean removed, unit length) for: CIFAR-100 at 64 (bilinear, as the benchmark and the lives
show CIFAR), CIFAR-100 native at 32, COCO val2017 photos brought down to 64. And what SAILnet's longest filter
answers to, by the window's place modulo 2 (the lattice of the blow-up)."""
import os, pickle, sys, glob
import numpy as np
from PIL import Image
sys.path.insert(0, "."); sys.path.insert(0, "experiments")
import softhebb_probe as P
from organic.cortex2 import _windows, _contrast
rng = np.random.default_rng(0)
un = P.load(f"{P.DATA}/cifar-100-python/train")
pick = rng.choice(len(un), 600, replace=False)
big = lambda x: np.asarray(Image.fromarray(x).resize((64, 64), Image.BILINEAR), np.float32) / 255
sets = {"CIFAR at 64 (blown up)": [big(un[i]) for i in pick], "CIFAR at 32 (native)": [un[i].astype(np.float32) / 255 for i in pick]}
files = sorted(glob.glob(os.path.expanduser("~/datasets/val2017/*.jpg")))[:600]
sets["COCO photos brought down to 64"] = [np.asarray(Image.open(f).convert("RGB").resize((64, 64), Image.BILINEAR), np.float32) / 255 for f in files]
base = pickle.load(open("out/bench/cortex_random_s0.pkl", "rb")); layer = base.layers[0]; Z = layer.Z[0]
def rows(ims, strong=True, n=40):
    out, where = [], []
    for x in ims:
        v, e = _contrast(_windows(x, 5)); h, w = e.shape
        v, e = v.reshape(-1, 75), e.ravel()
        idx = np.nonzero(e > 0.3 * e.max())[0] if strong else np.arange(len(e))
        idx = rng.choice(idx, min(n, len(idx)), replace=False)
        out.append(v[idx]); where.append(np.stack([idx // w, idx % w], 1))
    return np.concatenate(out), np.concatenate(where)
for name, ims in sets.items():
    V, _ = rows(ims)
    d = np.linalg.eigvalsh(np.cov(V.T))[::-1]; d = np.maximum(d, 0)
    pr = d.sum() ** 2 / (d * d).sum()
    print(f"{name:32}: windows' variance by direction: largest {d[0]:.3g}; directions holding 90% / 99% / 99.9% of it: {int(np.searchsorted(np.cumsum(d) / d.sum(), 0.9)) + 1} / {int(np.searchsorted(np.cumsum(d) / d.sum(), 0.99)) + 1} / {int(np.searchsorted(np.cumsum(d) / d.sum(), 0.999)) + 1}; "
          f"directions above a thousandth of the largest (the decorrelation's floor): {int((d > 1e-3 * d[0]).sum())} of 75; participation {pr:.1f}")
    W = V @ Z; W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    dw = np.maximum(np.linalg.eigvalsh(np.cov(W.T))[::-1], 0)
    print(f"{'':32}  after the cortex's decorrelation (settled on blown-up CIFAR): participation {dw.sum() ** 2 / (dw * dw).sum():.1f}; largest direction's share {dw[0] / dw.sum():.3f}")
Q = np.load("out/bench/sailc_weights_s0.npz")["W"]; r = np.linalg.norm(Q, axis=1); top = Q[np.argmax(r)] / r.max()
for name in ("CIFAR at 64 (blown up)", "COCO photos brought down to 64"):
    V, where = rows(sets[name]); W = V @ Z; W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    a = W @ top
    print(f"{name:32}: SAILnet's longest filter (length {r.max():.2f}; 107 cells alike): mean answer by the window's place modulo 2 (row, column): " + ", ".join(f"({i},{j}) {a[(where[:, 0] % 2 == i) & (where[:, 1] % 2 == j)].mean():+.3f}" for i in (0, 1) for j in (0, 1)) + f"; spread of the answer {a.std():.3f}; share of windows above 0.5: {float((a > 0.5).mean()):.3f}")
