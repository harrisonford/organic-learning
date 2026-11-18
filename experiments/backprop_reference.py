"""A backprop reference of the same shape, on the same pictures (docs/VISION.md, 2026-11-18).

Not part of the brain: a ruler. The survey of 2026-11-18 lists it first among what a reader will ask for: a network
of the same size trained by backpropagation on the same data and exposure. The brain's path from picture to name is
    windows of the picture, decorrelated  ->  256 kinds, each answering above the average of all kinds at its place
    ->  the average of each kind in each of 16 regions (4,096 numbers)  ->  [relay]  ->  word cells (delta rule)
developed without labels (the kinds: winner-take-all Hebbian) and taught online, each naming moment once.

Here the same shape is trained end to end by backprop (softmax, Adam, batches of 8): the same decorrelated windows
in (the retina's part is shared), 256 filters, the same response and regional average, a linear readout to the
names. The very moments a seed's word cells lived in experiments/words_on_cortex.py (N naming moments of world 3's
first words), and the same held-out pictures. Arms:
    backprop        filters and readout learned by backprop, from random filters
    readout only    the cortex's own developed filters kept; only the readout learned (by the same gradient)
    random filters  random filters kept; only the readout learned
Scores after 1 pass (the brain's exposure) and after more passes (which the brain does not get).
    SEEDS=0 ARMS="backprop,readout only,random filters" PASSES=5 python experiments/backprop_reference.py
"""
import os, pickle, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
ARMS = os.environ.get("ARMS", "backprop,readout only,random filters").split(",")
N, PASSES, BATCH, RATE = int(os.environ.get("N", 8000)), int(os.environ.get("PASSES", 5)), int(os.environ.get("BATCH", 8)), float(os.environ.get("RATE", 1e-3))
SHOW = [int(x) for x in os.environ.get("SHOW", "1,2,5").split(",")]
cur = Curriculum(seed=0, world=3)
SRC = ("cifar:name", "commons:name", "coco:name")
lived = {s: [it for it in cur.train["W1"][s] if cur.image(it[0]).max() > 0] for s in SRC}
held = {s: [it for it in cur.test["W1"][s] if cur.image(it[0]).max() > 0][:200] for s in SRC}
cortex = pickle.load(open(os.environ.get("CORTEX", "data/visual_cortex2.pkl"), "rb"))
layer = cortex.layers[0]
K, G = layer.kinds, 4


def windows(ref):
    """The picture as the first layer gets it: every window decorrelated, at unit length (positions x 75), and the side."""
    _, _, seen = layer._drive(cur.image(ref))
    v = seen[0][0]
    return v.reshape(-1, v.shape[2]).astype(np.float32), v.shape[0]


def region(side):
    """Which of the 16 regions each position is in, and how many positions each region has."""
    r = np.minimum(np.arange(side) * G // side, G - 1)
    idx = (r[:, None] * G + r[None, :]).ravel()
    return idx, np.bincount(idx, minlength=G * G).astype(np.float32)


class Net:
    def __init__(self, classes, arm, seed):
        rng = np.random.default_rng(seed)
        self.W = layer.W[0].copy() if arm == "readout only" else (lambda w: w / np.linalg.norm(w, axis=1, keepdims=True))(rng.normal(0, 1, layer.W[0].shape).astype(np.float32))
        self.b = np.zeros(K, np.float32)
        self.R = np.zeros((G * G * K, classes), np.float32)
        self.c = np.zeros(classes, np.float32)
        self.train_filters = arm == "backprop"
        self.names = ["R", "c"] + (["W", "b"] if self.train_filters else [])
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.names}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.names}
        self.t = 0

    def forward(self, v, idx, count):
        u = v @ self.W.T + self.b
        a = u - u.mean(axis=1, keepdims=True)
        r = np.maximum(a, 0)
        p = np.zeros((G * G, K), np.float32)
        np.add.at(p, idx, r)
        p /= count[:, None]
        f = p.ravel()
        f = f / (np.linalg.norm(f) + 1e-8) * 8.0  # (unit length as the brain's summary; x8 so that a zero readout starts with usable gradients)
        return f @ self.R + self.c, (v, a, p, f)

    def grads(self, logits, y, kept, idx, count):
        v, a, p, f = kept
        z = np.exp(logits - logits.max())
        z /= z.sum()
        z[y] -= 1.0
        g = {"R": np.outer(f, z), "c": z}
        if self.train_filters:
            df = self.R @ z
            raw = p.ravel()
            norm = np.linalg.norm(raw) + 1e-8
            draw = 8.0 * (df - (df @ raw) * raw / norm ** 2) / norm  # through the unit length
            dr = (draw.reshape(G * G, K) / count[:, None])[idx]
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
    rng = random.Random(seed)
    moments = [(rng.choice(lived[rng.choice(SRC)]), rng.choice(cur.frames)) for _ in range(N)]  # the moments the word cells lived (words_on_cortex.py)
    names = sorted({it[2][0] for s in SRC for it in lived[s]})
    idx = count = None
    for arm in ARMS:
        net, t0, order = Net(len(names), arm, seed), time.time(), random.Random(seed + 11)
        for p_ in range(1, PASSES + 1):
            seq = list(range(N))
            if p_ > 1:
                order.shuffle(seq)
            acc, right = None, 0
            for n, i in enumerate(seq):
                it = moments[i][0]
                v, side = windows(it[0])
                if idx is None:
                    idx, count = region(side)
                y = names.index(it[2][0])
                logits, kept = net.forward(v, idx, count)
                right += int(np.argmax(logits) == y)
                g = net.grads(logits, y, kept, idx, count)
                acc = g if acc is None else {k: acc[k] + g[k] for k in g}
                if n % BATCH == BATCH - 1:
                    net.step(acc, BATCH)
                    acc = None
            if p_ in SHOW or p_ == PASSES:
                row = {s: float(np.mean([names[int(np.argmax(net.forward(windows(it[0])[0], idx, count)[0]))] == it[2][0] for it in held[s]])) for s in SRC}
                print(f"seed {seed} {arm:15} after {p_} pass{'es' if p_ > 1 else ''} ({time.time() - t0:.0f}s; right while living this pass {right / N:.3f}): held out "
                      + " | ".join(f"{s.split(':')[0]} {x:.3f}" for s, x in row.items()) + f" | mean {np.mean(list(row.values())):.3f}", flush=True)
