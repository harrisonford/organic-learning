"""What did the cortex's development do? (docs/VISION.md, 2026-11-20; an observation, nothing is changed)

On the benchmark, filters developed by winner-take-all Hebbian learning name no better than filters never developed
(0.687 against 0.693, three seeds). Before any thought about why, look at what development did to the kinds:
    moved       how far each kind is from where it was born (cosine with its random start; 1 = it never moved)
    kurtosis    how heavy-tailed a kind's drive is over windows (a kind that has found a feature answers rarely and
                strongly; a random direction of decorrelated windows answers like noise: excess kurtosis near 0)
    use         over windows, how evenly the kinds share being the strongest (entropy, 1 = perfectly even)
    alike       how alike two kinds are (mean and top 1% of |cosine| between filters)
    active      of the 256 kinds, how many answer above the place's average, per window
    code        the 4,096-number summary over 5,000 benchmark pictures: how many directions carry its variance
                (participation ratio), and how much of it the ten strongest carry
    SEEDS=0,1,2 python experiments/cortex_dev_probe.py
"""
import os, pickle, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA = os.path.expanduser("~/datasets")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
d = pickle.load(open(f"{DATA}/cifar-100-python/test", "rb"), encoding="bytes")
pics = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)[:200]
big = [np.asarray(Image.fromarray(x).resize((64, 64), Image.BILINEAR), np.float32) / 255 for x in pics]

for seed in SEEDS:
    cortices = {v: pickle.load(open(f"out/bench/cortex_{v}_s{seed}.pkl", "rb")) for v in ("developed", "random")}
    born = cortices["random"].layers[0].W[0]
    for v, cortex in cortices.items():
        layer, rng = cortex.layers[0], np.random.default_rng(seed)
        W = layer.W[0]
        U = []
        for img in big:
            u, e, _ = layer._drive(img)
            u = u.reshape(-1, u.shape[2])
            strong = np.nonzero(e.ravel() > 0.3 * e.max())[0]  # (the windows development learns from)
            U.append(u[rng.choice(strong, min(300, len(strong)), replace=False)])
        U = np.concatenate(U)
        z = (U - U.mean(axis=0)) / (U.std(axis=0) + 1e-8)
        kurt = (z ** 4).mean(axis=0) - 3
        wins = np.bincount(np.argmax(U, axis=1), minlength=W.shape[0]) / len(U)
        use = float(-(wins[wins > 0] * np.log(wins[wins > 0])).sum() / np.log(len(wins)))
        cos = np.abs(W @ W.T)[np.triu_indices(len(W), 1)]
        active = (U > U.mean(axis=1, keepdims=True)).sum(axis=1)
        X = np.asarray(np.load(f"out/bench/cifar10_{v}_s{seed}_train.npy", mmap_mode="r")[:5000], np.float32)
        ev = np.linalg.eigvalsh(np.cov((X - X.mean(axis=0)).T.astype(np.float64)))[::-1]
        ev = np.maximum(ev, 0)
        moved = float(np.mean(np.sum(W * born, axis=1))) if v == "developed" and W.shape == born.shape else float("nan")
        print(f"seed {seed} {v:9} | moved {moved:.2f} | kurtosis {np.mean(kurt):.2f} (median {np.median(kurt):.2f}) | use {use:.3f} (never strongest: {int((wins == 0).sum())}) | "
              f"alike {cos.mean():.3f}, top 1% {np.percentile(cos, 99):.2f} | active {active.mean():.0f} of {W.shape[0]} | "
              f"code: {ev.sum() ** 2 / (ev ** 2).sum():.1f} directions, ten strongest carry {ev[:10].sum() / ev.sum():.2f}", flush=True)
