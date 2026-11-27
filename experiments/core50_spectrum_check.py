"""A diagnostic (docs/VISION.md 2026-11-27 08:40): do the word cells miss the middle fixation's gain because the
relay was settled on another world? The relay (organic/cortex2.py) evens out the code's directions so that a delta
rule seeing each moment once learns along the fine directions too (measured 2026-11-17); for CORe50 it was settled on
COCO thumbnails. Ridge, streaming LDA and the nearest class mean do not care how even the code is; a delta rule does.
For the whole frame's and the middle fixation's codes, on the first 10,000 frames of the benchmark's stream:
    how even the relayed code is on CORe50   participation ratio (sum of variances)^2 / sum of variances^2, out of
                                             4,096, and the share of the variance in the strongest 10 and 100 directions
    the delta reader (bench_stream.py's)     on the relayed code (should repeat its numbers), and on the relayed code
                                             evened once more by a relay settled on these 10,000 frames (no labels)
    ridge                                    on both (it should not change much)

    SEEDS=0,1,2 python experiments/core50_spectrum_check.py
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Relay  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
N = 10000
ytr, yte = np.load("out/bench/core50_train_labels.npy"), np.load("out/bench/core50_test_labels.npy")
C = int(ytr.max()) + 1
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
SHARE = 2.0 + len(set(ear.units("what is this?", learn=False)))
del ear


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def delta(X, y, Xt, yt):
    """bench_stream.py's Delta, the same arithmetic, frame by frame in the stream's order."""
    D = X.shape[1]
    W, base, use, mean = np.zeros((C, D), np.float32), np.zeros(C, np.float32), np.zeros(D, np.float32), np.zeros(D, np.float32)
    for n, (v, c) in enumerate(zip(X, y), 1):
        mean += (v - mean) / min(n, 1000)
        x = v - mean
        x /= np.linalg.norm(x) + 1e-8
        e = -(W @ x + base)
        e[c] += 1.0
        step = e / SHARE
        W += step[:, None] * (x / (1.0 + use / 5.0))[None, :]
        use += x * x
        base += step / (1.0 + n / 5.0)
    xt = Xt - mean
    xt /= np.linalg.norm(xt, axis=1, keepdims=True) + 1e-8
    return float(np.mean(np.argmax(xt @ W.T + base, axis=1) == yt))


def ridge(X, y, Xt, yt):
    m = X.mean(axis=0)
    A = ((X - m).T @ (X - m)).astype(np.float64)
    Y = np.eye(C)[y]
    B = (X - m).T @ (Y - Y.mean(axis=0))
    best = 0.0
    for lam in (0.01, 0.1, 1.0):
        W = np.linalg.solve(A + lam * (np.trace(A) / A.shape[0]) * np.eye(A.shape[0]), B)
        best = max(best, float(np.mean(np.argmax((Xt - m) @ W, axis=1) == yt)))
    return best


def evenness(X):
    d = np.clip(np.linalg.eigvalsh(np.cov((X - X.mean(axis=0)).T.astype(np.float64))), 0, None)[::-1]
    return d.sum() ** 2 / (d * d).sum(), d[:10].sum() / d.sum(), d[:100].sum() / d.sum()


for seed in SEEDS:
    order = np.random.default_rng(seed).permutation(len(ytr))[:N]
    y = ytr[order]
    for v in ("first", "centre"):
        relay = pickle.load(open(f"out/bench/cortex_c50{v}_s{seed}.pkl", "rb")).relay
        X = unit((np.asarray(np.load(f"out/bench/core50_c50{v}_s{seed}_train.npy", mmap_mode="r")[order], np.float32) - relay.mu) @ relay.Z)
        Xt = unit((np.asarray(np.load(f"out/bench/core50_c50{v}_s{seed}_test.npy"), np.float32) - relay.mu) @ relay.Z)
        pr, t10, t100 = evenness(X)
        own = Relay(relay.shrink).settle(X)
        X2, Xt2 = unit((X - own.mu) @ own.Z), unit((Xt - own.mu) @ own.Z)
        pr2, _, _ = evenness(X2)
        print(f"seed {seed} {v:6}: on CORe50 the relayed code's participation ratio {pr:.0f} of 4096 (strongest 10: {t10:.2f}, "
              f"100: {t100:.2f}; evened again: {pr2:.0f}) | delta {delta(X, y, Xt, yte):.3f}, evened again {delta(X2, y, Xt2, yte):.3f} | "
              f"ridge {ridge(X, y, Xt, yte):.3f}, evened again {ridge(X2, y, Xt2, yte):.3f}", flush=True)
