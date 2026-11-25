"""A faithful soft winner-take-all development of the filters (queue E 27; docs/VISION.md, 2026-11-25).

The paper says "developed filters do no better than filters never developed". Our development is a hard
winner-take-all with a conscience. Journe, Garcia Rodriguez, Guo & Moraitis (ICLR 2023, arXiv 2209.11883) report,
for one convolutional layer on CIFAR-10 under a trained linear classifier: hard winner-take-all 62.7, random
weights 63.9, their soft rule (SoftHebb) 71.1. Our own soft attempt of 2026-11-17 was "my version" and did worse.
Here: their rule and their layer, unchanged, every number theirs (read from the paper and from the authors' code,
hebbconv.py / layer.py / presets.json, preset 2SoftHebbCnnCIFAR):

    input            a picture at 32 x 32 in [0, 1]; each colour standardised (batch norm with no learned weights:
                     the batch's own mean and variance while developing, running averages after)
    drive            u = w . x on every 5 x 5 window (reflect padding), weights neither normalised nor biased
    competition      y = softmax(u) over the kinds at a place (inverse temperature 1)
    rule             dw_k = +y_k (x - u_k w_k) for the kind with the largest drive, -y_k (x - u_k w_k) for the others
                     ("soft anti-Hebbian"), summed over the places of a batch of 10 pictures
    step             the summed change divided by its largest absolute value in the layer (the code does this, the
                     paper does not say it), then times 0.08 (|norm of the neuron's weights - 1|)^0.5
    birth            weights normal, so that a neuron's weight vector has length about 25
    seen             every unlabelled picture once
    response         (u - mean of u over the kinds at the place)+ ^ 0.7 ("triangle"), averaged over 4 x 4 at stride 2
What is ours, and said: the unlabelled pictures are CIFAR-100's (the stream's own are never seen before the
stream; the authors develop on CIFAR-10's without labels); the kinds are 256 as in our cortex (and 96 as
published); and the measure. The layer's 16 x 16 map is summarised as every code of ours is (each kind averaged in
4 x 4 regions, unit length), the relay is settled on 6,000 unlabelled pictures, and the stream is read by
experiments/bench_stream.py (word cells and ridge, with and without the relay, 10,000 pictures):
    sh / shr          256 kinds, developed / as born (their "random weights" baseline)
    sh96 / shr96      96 kinds
A check that the rule is theirs and not again mine (REPLICATE=1, 96 kinds): their finding in their own measure, in
small. A ridge on the whole map (24,576 numbers a picture, as their classifier reads it) from 5,000 labelled
pictures, tested on 2,000: developed against as born. They report 71.1 against 63.9 with 50,000 pictures.
    SEEDS=0 KINDS=256,96 python experiments/softhebb_probe.py
"""
import os, pickle, sys, time, types
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Relay  # noqa: E402

DATA = os.path.expanduser("~/datasets")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
KINDS = [int(x) for x in os.environ.get("KINDS", "256,96").split(",")]
SIGHTS = int(os.environ.get("SIGHTS", 50000))  # unlabelled pictures seen, once each
REPLICATE = os.environ.get("REPLICATE", "1") == "1"
RATE, POWER, RADIUS, T_INV, TRIANGLE, BATCH = 0.08, 0.5, 25.0, 1.0, 0.7, 10  # the authors' (presets.json)
GRID, SHRINK, NRELAY = 4, 0.0005, 6000  # the benchmark's (experiments/cortex_norm_probe.py)


def load(path):
    d = pickle.load(open(path, "rb"), encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


class Standard:
    """Each colour standardised: torch's BatchNorm2d with affine off (momentum 0.1, unbiased running variance)."""

    def __init__(self):
        self.mean, self.var = np.zeros(3, np.float32), np.ones(3, np.float32)

    def developing(self, X):
        m, v, n = X.mean(axis=(0, 1, 2)), X.var(axis=(0, 1, 2)), X.size // 3
        self.mean, self.var = 0.9 * self.mean + 0.1 * m, 0.9 * self.var + 0.1 * v * n / (n - 1)
        return ((X - m) / np.sqrt(v + 1e-5)).astype(np.float32)

    def __call__(self, X):
        return ((X - self.mean) / np.sqrt(self.var + 1e-5)).astype(np.float32)


def windows(X):
    """(n, 32, 32, 3) -> every 5 x 5 window, reflect padded, as a row: (n, 32, 32, 75)."""
    P = np.pad(X, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="reflect")
    return sliding_window_view(P, (5, 5), axis=(1, 2)).reshape(len(X), 32, 32, 75)


def develop(kinds, pictures, rng):
    """The rule, one sight of each picture. Returns the weights as born, as developed, and the standardiser."""
    W = (RADIUS / np.sqrt(75) * rng.standard_normal((kinds, 75))).astype(np.float32)
    born, standard, t0 = W.copy(), Standard(), time.time()
    order = rng.permutation(len(pictures))[:SIGHTS]
    for b, a in enumerate(range(0, len(order), BATCH)):
        X = windows(standard.developing(pictures[order[a: a + BATCH]] / np.float32(255))).reshape(-1, 75)
        U = X @ W.T
        Y = np.exp(T_INV * (U - U.max(axis=1, keepdims=True)))
        Y /= Y.sum(axis=1, keepdims=True)
        rows, first = np.arange(len(U)), U.argmax(axis=1)
        S = -Y
        S[rows, first] = Y[rows, first]
        dW = S.T @ X - (S * U).sum(axis=0)[:, None] * W
        dW /= np.abs(dW).max() + 1e-30
        r = np.linalg.norm(W, axis=1)
        W += (RATE * (np.abs(r - 1) + 1e-10) ** POWER)[:, None] * dW
        if b in (0, 99, 999) or a + BATCH >= len(order):
            wins = np.bincount(first, minlength=kinds) / len(first)
            print(f"   {kinds} kinds after {a + len(X) // 1024} pictures: weight lengths {r.min():.2f} / {r.mean():.2f} / {r.max():.2f} (least, mean, most) | "
                  f"the strongest kind's share of a place's activity {Y.max(axis=1).mean():.2f} | a kind is the strongest at {wins.min():.4f} to {wins.max():.4f} of places ({time.time() - t0:.0f}s)", flush=True)
    return born, W, standard


def respond(W, standard, X):
    """Pictures (n, 32, 32, 3) in 0..255 -> the layer's map (n, 16, 16, kinds): triangle, then the 4 x 4 average."""
    U = windows(standard(X / np.float32(255))) @ W.T
    R = np.maximum(U - U.mean(axis=-1, keepdims=True), 0) ** TRIANGLE
    P = np.pad(R, ((0, 0), (1, 1), (1, 1), (0, 0)))
    return sliding_window_view(P, (4, 4), axis=(1, 2))[:, ::2, ::2].mean(axis=(-1, -2)).astype(np.float32)


def summary(M):
    """As every code of ours: each kind averaged in GRID x GRID regions, at unit length."""
    n, h, w, k = M.shape
    p = M.reshape(n, GRID, h // GRID, GRID, w // GRID, k).mean(axis=(2, 4)).reshape(n, -1)
    return p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)


def coded(W, standard, X, whole=False):
    out = []
    for a in range(0, len(X), 100):
        M = respond(W, standard, X[a: a + 100])
        out.append(M.reshape(len(M), -1) if whole else summary(M))
    return np.concatenate(out)


def read_code(label, X, y):
    """Two pictures alike and directions used, as experiments/expand_probe.py reads them."""
    Xc = X - X.mean(axis=0)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8
    L = (Xc @ Xc.T).astype(np.float64)
    off, same = ~np.eye(len(L), dtype=bool), y[:, None] == y[None, :]
    s = np.maximum(np.linalg.eigvalsh(L), 0)
    return f"{label}: two pictures alike {np.abs(L[off]).mean():.3f} (same class {L[off & same].mean():+.4f}, different {L[off & ~same].mean():+.4f}) | directions used {s.sum() ** 2 / (s * s).sum():.0f}"


def ridge_whole(Ftr, ytr, Fte, yte):
    """A ridge on the whole map, in its dual form; its one number chosen on 1,000 of the labelled pictures held out."""
    mu = Ftr.mean(axis=0)
    Ftr -= mu
    Fte -= mu
    K, Kte, T = (Ftr @ Ftr.T).astype(np.float64), (Fte @ Ftr.T).astype(np.float64), np.eye(10)[ytr] - 0.1
    n, fit, held, scale, best = len(K), np.arange(len(K) - 1000), np.arange(len(K) - 1000, len(K)), np.trace(K) / len(K), (-1, None)
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        A = np.linalg.solve(K[np.ix_(fit, fit)] + lam * scale * np.eye(len(fit)), T[fit])
        best = max(best, (float(np.mean((K[np.ix_(held, fit)] @ A).argmax(axis=1) == ytr[held])), lam))
    A = np.linalg.solve(K + best[1] * scale * np.eye(n), T)
    return float(np.mean((Kte @ A).argmax(axis=1) == yte)), best[1]


if __name__ == "__main__":
    unlabelled = load(f"{DATA}/cifar-100-python/train")
    Xtr, Xte = load(f"{DATA}/cifar-10-batches-py/data_batch_1"), load(f"{DATA}/cifar-10-batches-py/test_batch")[:2000]
    ytr, yte = np.load("out/bench/cifar10_train_labels.npy")[: len(Xtr)], np.load("out/bench/cifar10_test_labels.npy")[:2000]
    for seed in SEEDS:
        settle = np.random.default_rng(seed + 7).choice(len(unlabelled), NRELAY, replace=False)
        for kinds in KINDS:
            t0 = time.time()
            born, W, standard = develop(kinds, unlabelled, np.random.default_rng(3000 + seed))
            C = W / np.linalg.norm(W, axis=1, keepdims=True)
            alike = np.abs(C @ C.T)[~np.eye(kinds, dtype=bool)]
            print(f"seed {seed} {kinds} kinds developed ({time.time() - t0:.0f}s) | two kinds' weights alike {alike.mean():.3f} on average, pairs above 0.9: {int((alike > 0.9).sum() // 2)}", flush=True)
            for name, weights in (("sh", W), ("shr", born)):
                name = name + ("" if kinds == 256 else str(kinds))
                relay = Relay(SHRINK).settle(coded(weights, standard, unlabelled[settle]))
                for part, X in (("train", Xtr), ("test", Xte)):
                    S = coded(weights, standard, X)
                    np.save(f"out/bench/cifar10_{name}_s{seed}_{part}.npy", S.astype(np.float16))
                through = (S - relay.mu) @ relay.Z
                print(f"seed {seed} " + read_code(f"{name:6} summary", S[:1500], yte[:1500]) + " | " + read_code("through the relay", through[:1500], yte[:1500]), flush=True)
                pickle.dump(types.SimpleNamespace(relay=relay, W=weights, standard=(standard.mean, standard.var)), open(f"out/bench/cortex_{name}_s{seed}.pkl", "wb"))
                del relay, through, S
            if REPLICATE and kinds == 96:
                for name, weights in (("developed", W), ("as born", born)):
                    acc, lam = ridge_whole(coded(weights, standard, Xtr[:5000], whole=True), ytr[:5000], coded(weights, standard, Xte, whole=True), yte)
                    print(f"seed {seed} their measure in small, 96 kinds {name}: a ridge on the whole map from 5,000 pictures names {acc:.3f} of 2,000 new ones (its number: {lam})", flush=True)
