"""CIFAR-10 as a one-pass stream: backprop of the same shape, tuned for one pass (docs/PUBLISH.md, 2026-11-19).

The reference beside experiments/bench_stream.py. The same decorrelated windows in (the benchmark cortex's own
first step, settled on unlabelled CIFAR-100 pictures), 256 filters, the same response and regional average, a linear
readout to the ten classes; softmax, Adam. Every labelled picture comes once, in the same order as in
bench_stream.py for the seed. What is varied, because a reviewer will ask:
    FILTERS   random      born random, learned by backprop (end to end)
              developed   the cortex's developed filters as the start, then learned by backprop
              frozen      the cortex's developed filters kept; only the readout learned
    RATE, BATCH           the learning rate and how many pictures share a step (tuned on the first 10,000)
    REPLAY    for every new picture, this many earlier ones (drawn at random from all seen so far) are lived again:
              more updates from the same stream, what backprop is usually given in online learning
    RELAY     (2026-11-23: the like-for-like reference a referee asked for; without it this network has no relay and
              the word cells' lead over it is a whitening step it was never given)
              fixed     the cortex's relay step in front of the readout: the pooled code's correlations removed
                        (organic.cortex2.Relay, the same shrinkage), settled once on 6,000 unlabelled CIFAR-100
                        pictures seen through the filters as they are born, and kept; errors go back through it
              tracked   the same, settled again every REFIT pictures (default 5,000) on 2,000 unlabelled pictures
                        seen through the filters as they are then (the filters move; a fixed relay goes stale)
    SEEDS=0 FILTERS=random RATE=0.001 BATCH=8 REPLAY=0 N=50000 python experiments/bench_backprop.py
"""
import os, pickle, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA = os.path.expanduser("~/datasets")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
FILTERS, CORTEX_SEED = os.environ.get("FILTERS", "random"), int(os.environ.get("CORTEX_SEED", 0))
RATE, BATCH, REPLAY = float(os.environ.get("RATE", 1e-3)), int(os.environ.get("BATCH", 8)), int(os.environ.get("REPLAY", 0))
N, TESTS = int(os.environ.get("N", 50000)), int(os.environ.get("TESTS", 2000))
RELAY, REFIT, SHRINK = os.environ.get("RELAY", ""), int(os.environ.get("REFIT", 5000)), float(os.environ.get("SHRINK", 0.0005))
AT = [n for n in (500, 1000, 2000, 5000, 10000, 20000, 50000) if n <= N]
cortex = pickle.load(open(f"out/bench/cortex_developed_s{CORTEX_SEED}.pkl", "rb"))
layer = cortex.layers[0]
K, G, C = layer.kinds, 4, 10


def load(path):
    d = pickle.load(open(path, "rb"), encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b"labels"])


train = [load(f"{DATA}/cifar-10-batches-py/data_batch_{i}") for i in range(1, 6)]
Xtr, ytr = np.concatenate([a for a, _ in train]), np.concatenate([b for _, b in train])
Xte, yte = load(f"{DATA}/cifar-10-batches-py/test_batch")
if RELAY:
    from organic.cortex2 import Relay
    _u = pickle.load(open(f"{DATA}/cifar-100-python/train", "rb"), encoding="bytes")[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    UNLABELLED = _u[np.random.default_rng(7).choice(len(_u), 6000, replace=False)]


def windows(x):
    """A picture as the first layer gets it: enlarged as the eye has it, every window decorrelated, at unit length."""
    img = np.asarray(Image.fromarray(x).resize((64, 64), Image.BILINEAR), np.float32) / 255
    v = layer._drive(img)[2][0][0]
    return v.reshape(-1, v.shape[2]).astype(np.float32), v.shape[0]


_, side = windows(Xtr[0])
r_ = np.minimum(np.arange(side) * G // side, G - 1)
IDX = (r_[:, None] * G + r_[None, :]).ravel()
COUNT = np.bincount(IDX, minlength=G * G).astype(np.float32)


class Net:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.W = layer.W[0].copy() if FILTERS in ("developed", "frozen") else (lambda w: w / np.linalg.norm(w, axis=1, keepdims=True))(rng.normal(0, 1, layer.W[0].shape).astype(np.float32))
        self.b, self.R, self.c = np.zeros(K, np.float32), np.zeros((G * G * K, C), np.float32), np.zeros(C, np.float32)
        self.names = ["R", "c"] + ([] if FILTERS == "frozen" else ["W", "b"])
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.names}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.names}
        self.t = 0
        self.relay = None
        if RELAY:
            self.settle(6000)

    def summary(self, v):
        """The pooled code at unit length (what the cortex sends on), and what the way back needs."""
        u = v @ self.W.T + self.b
        a = u - u.mean(axis=1, keepdims=True)
        p = np.zeros((G * G, K), np.float32)
        np.add.at(p, IDX, np.maximum(a, 0))
        p /= COUNT[:, None]
        return a, p

    def settle(self, many):
        """The relay settled on unlabelled pictures seen through the filters as they are now."""
        S = []
        for x in UNLABELLED[:many]:
            raw = self.summary(windows(x)[0])[1].ravel()
            S.append(raw / (np.linalg.norm(raw) + 1e-8))
        self.relay = Relay(SHRINK).settle(np.array(S, np.float32))

    def forward(self, v):
        a, p = self.summary(v)
        f = p.ravel()
        if self.relay is not None:
            s_ = f / (np.linalg.norm(f) + 1e-8)
            w = (s_ - self.relay.mu) @ self.relay.Z
            return (w / (np.linalg.norm(w) + 1e-8) * 8.0) @ self.R + self.c, (v, a, p, w / (np.linalg.norm(w) + 1e-8) * 8.0, w)
        f = f / (np.linalg.norm(f) + 1e-8) * 8.0
        return f @ self.R + self.c, (v, a, p, f)

    def grads(self, logits, y, kept):
        v, a, p, f = kept[:4]
        z = np.exp(logits - logits.max())
        z /= z.sum()
        z[y] -= 1.0
        g = {"R": np.outer(f, z), "c": z}
        if "W" in self.names:
            df, raw = self.R @ z, p.ravel()
            norm = np.linalg.norm(raw) + 1e-8
            if self.relay is not None:  # back through the relay: f = 8 w / |w|, w = (raw / |raw| - mu) Z
                w = kept[4]
                wn = np.linalg.norm(w) + 1e-8
                ds = (8.0 * (df - (df @ w) * w / wn ** 2) / wn) @ self.relay.Z.T
                draw = (ds - (ds @ raw) * raw / norm ** 2) / norm
            else:
                draw = 8.0 * (df - (df @ raw) * raw / norm ** 2) / norm
            dr = (draw.reshape(G * G, K) / COUNT[:, None])[IDX]
            da = dr * (a > 0)
            du = da - da.mean(axis=1, keepdims=True)
            g["W"], g["b"] = du.T @ v, du.sum(axis=0)
        return g

    def step(self, g, n):
        self.t += 1
        for k in self.names:
            d = g[k] / n
            self.m[k] = 0.9 * self.m[k] + 0.1 * d
            self.v[k] = 0.999 * self.v[k] + 0.001 * d * d
            getattr(self, k).__isub__(RATE * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8))


for seed in SEEDS:
    order = np.random.default_rng(seed).permutation(len(Xtr))[:N]  # the stream of bench_stream.py
    tests = np.random.default_rng(99).permutation(len(Xte))[:TESTS]
    net, t0, again, row, acc, k = Net(seed), time.time(), np.random.default_rng(seed + 3), {}, None, 0
    for n, i in enumerate(order, 1):
        for j in [i] + ([int(order[again.integers(n)]) for _ in range(REPLAY)] if n > 1 else []):
            logits, kept = net.forward(windows(Xtr[j])[0])
            g = net.grads(logits, int(ytr[j]), kept)
            acc, k = (g if acc is None else {q: acc[q] + g[q] for q in g}), k + 1
            if k == BATCH:
                net.step(acc, BATCH)
                acc, k = None, 0
        if RELAY == "tracked" and n % REFIT == 0 and n < N:
            net.settle(2000)
        if n in AT:
            row[n] = float(np.mean([int(np.argmax(net.forward(windows(Xte[j])[0])[0])) == int(yte[j]) for j in tests]))
            print(f"   seed {seed} backprop {FILTERS} relay {RELAY or 'none'} rate {RATE} batch {BATCH} replay {REPLAY} after {n:>6}: {row[n]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"seed {seed} backprop filters {FILTERS:9} relay {RELAY or 'none':7} rate {RATE:<7} batch {BATCH:<3} replay {REPLAY}: " + " | ".join(f"{n} {a:.3f}" for n, a in row.items()), flush=True)
