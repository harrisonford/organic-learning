"""A diagnostic (docs/VISION.md 2026-11-27 08:10): is it the word cells' visual neurons that throw away the middle
fixation's gain? Every reader of bench_stream.py gets the relayed code at unit length; the word cells' visual
neurons (and the delta reader) then take each one's deviation from its running average and bring the pattern to unit
length once more. Here the reference reader, a ridge on the first 10,000 frames of the benchmark's stream, reads
    relayed     the relayed code as the benchmark gives it (ridge+relay: should repeat bench_stream's numbers)
    neurons     what the cells' visual neurons pass on: the deviation from the average, at unit length again
and, for each, how far a frame's length after centring varies (the second normalisation's reach), on the whole frame
and the middle fixation's codes.

    SEEDS=0,1,2 python experiments/core50_reader_norm_check.py
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # (the pickled cortex's classes)

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
N = 10000
ytr, yte = np.load("out/bench/core50_train_labels.npy"), np.load("out/bench/core50_test_labels.npy")
C = int(ytr.max()) + 1


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def ridge(X, y, Xt, yt):
    m = X.mean(axis=0)
    A = (X - m).T @ (X - m)
    Y = np.eye(C)[y]
    B = (X - m).T @ (Y - Y.mean(axis=0))
    best = 0.0
    for lam in (0.01, 0.1, 1.0):  # as bench_stream.py: the best on the test frames (generous to the reference)
        W = np.linalg.solve(A + lam * (np.trace(A) / A.shape[0]) * np.eye(A.shape[0]), B)
        best = max(best, float(np.mean(np.argmax((Xt - m) @ W, axis=1) == yt)))
    return best


for seed in SEEDS:
    order = np.random.default_rng(seed).permutation(len(ytr))[:N]
    for v in ("first", "centre"):
        relay = pickle.load(open(f"out/bench/cortex_c50{v}_s{seed}.pkl", "rb")).relay
        X = unit((np.asarray(np.load(f"out/bench/core50_c50{v}_s{seed}_train.npy", mmap_mode="r")[order], np.float64) - relay.mu) @ relay.Z)
        Xt = unit((np.asarray(np.load(f"out/bench/core50_c50{v}_s{seed}_test.npy"), np.float64) - relay.mu) @ relay.Z)
        y = ytr[order]
        m = X.mean(axis=0)
        L, Lt = np.linalg.norm(X - m, axis=1), np.linalg.norm(Xt - m, axis=1)
        a, b = ridge(X, y, Xt, yte), ridge(unit(X - m), y, unit(Xt - m), yte)
        print(f"seed {seed} {v:6}: ridge on the relayed code {a:.3f}, on the visual neurons' pattern {b:.3f} | length after "
              f"centring: mean {L.mean():.3f}, spread (sd / mean) {L.std() / L.mean():.3f} (test {Lt.std() / Lt.mean():.3f}); "
              f"the average's own length {np.linalg.norm(m):.3f}", flush=True)
