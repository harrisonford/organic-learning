"""Does what the eye fixates add to the whole view? (docs/VISION.md 2026-11-27 12:42, said before it runs.)

Naming from the stream brain's own fixations alone is worse than naming from the whole frame (12:40): the eye is
off the held thing in two frames of three. But an eye does not lose the whole view when it fixates: the periphery
stays, the fovea adds. Here the readers get both at each frame, side by side (each code relayed and at unit length,
the pair at unit length), on the benchmark's stream of 10 kinds (the first 10,000 frames, one pass):
    whole               the whole frame alone (c50first: should repeat 0.36 / 0.37)
    whole + eye         with what the stream brain's eye (both reflexes) fixated at that frame (c50eyeboth)
    whole + middle      with the middle fixation, where CORe50's crop puts the object (c50centre: the reference)
Readers: the delta reader (bench_stream.py's, the word cells' rule without the words) and ridge; each with the relays
as settled on COCO, and with each part's relay settled again on the stream's own frames (no labels).

    SEEDS=0,1,2 python experiments/core50_views_together_check.py
"""
import os, pickle, sys
import numpy as np
import scipy.linalg as sl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Relay  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
N = 10000
PAIRS = [("whole", ["c50first"]), ("whole + eye", ["c50first", "c50eyeboth"]), ("whole + middle", ["c50first", "c50centre"])]
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
    Xc = X - m
    A = Xc.T @ Xc  # (single precision: the pair has 8,192 directions)
    Y = np.eye(C, dtype=np.float32)[y]
    B = Xc.T @ (Y - Y.mean(axis=0))
    del Xc
    best, tr = 0.0, float(np.trace(A)) / A.shape[0]
    for lam in (0.01, 0.1, 1.0):  # as bench_stream.py: the best on the test frames (generous to the reference)
        M = A.copy()
        M[np.diag_indices_from(M)] += lam * tr
        W = sl.solve(M, B, assume_a="pos", overwrite_a=True)
        best = max(best, float(np.mean(np.argmax((Xt - m) @ W, axis=1) == yt)))
    return best


for seed in SEEDS:
    order = np.random.default_rng(seed).permutation(len(ytr))[:N]
    y = ytr[order]
    parts = {}
    for v in ("c50first", "c50eyeboth", "c50centre"):
        relay = pickle.load(open(f"out/bench/cortex_{v}_s{seed}.pkl", "rb")).relay
        X = unit((np.asarray(np.load(f"out/bench/core50_{v}_s{seed}_train.npy", mmap_mode="r")[np.sort(order)][np.argsort(np.argsort(order))], np.float32) - relay.mu) @ relay.Z)
        Xt = unit((np.asarray(np.load(f"out/bench/core50_{v}_s{seed}_test.npy"), np.float32) - relay.mu) @ relay.Z)
        own = Relay(relay.shrink).settle(X)
        parts[v] = (X, Xt, unit((X - own.mu) @ own.Z), unit((Xt - own.mu) @ own.Z))
    for name, vs in PAIRS:
        X, Xt = (unit(np.concatenate([parts[v][k] for v in vs], axis=1)) for k in (0, 1))
        X2, Xt2 = (unit(np.concatenate([parts[v][k] for v in vs], axis=1)) for k in (2, 3))
        print(f"seed {seed} {name:15}: delta {delta(X, y, Xt, yte):.3f}, ridge {ridge(X, y, Xt, yte):.3f} | each part evened again: "
              f"delta {delta(X2, y, Xt2, yte):.3f}, ridge {ridge(X2, y, Xt2, yte):.3f}", flush=True)
