"""Does a trace-rule IT layer make the visual code more object-like? (offline probe)

Biology: neurons in inferotemporal cortex respond to an object across positions
and sizes. Foldiak (1991) and Wallis & Rolls (VisNet, 1997): if a neuron's
learning uses a slowly decaying trace of its own activity, views that follow
each other in time (the eye moving over one object) get bound to the same
neurons, with no labels.

Here: unlabeled photos; each is seen as a short sequence of views (small shifts,
a flip, a slight zoom: fixational and small saccadic movements). An IT layer of
competitive units reads the visual cortex's layer-3 code (2x2 layout) and
learns with (a) the trace rule, (b) the same rule without trace (control).
Then a nearest-neighbour probe on picture-book naming compares codes.

    python experiments/trace_it_probe.py
"""

import collections
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import PICTURE_BOOK, Curriculum  # noqa: E402

rng = np.random.default_rng(0)
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
cur = Curriculum(seed=0)


def view(img, k):
    """A view of the same thing: shifted, maybe flipped, maybe zoomed."""
    if k == 0:
        return img
    dx, dy = rng.integers(-6, 7, 2)
    v = np.roll(np.roll(img, dx, 1), dy, 0)
    if rng.random() < 0.5:
        v = v[:, ::-1]
    if rng.random() < 0.3:  # zoom in a little
        c = v[6:58, 6:58]
        v = np.asarray(np.kron(c, np.ones((1, 1, 1))), np.float32)
        from PIL import Image
        v = np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize((64, 64)), np.float32) / 255
    return np.ascontiguousarray(v)


def l3(img):
    c = cortex.code(img, grid=2)
    v = np.concatenate([c[1], c[2]])
    return v / (np.linalg.norm(v) + 1e-8)


class IT:
    def __init__(self, dim, units=600, k=20, seed=0):
        r = np.random.default_rng(seed)
        self.W = r.normal(size=(units, dim)).astype(np.float32)
        self.W /= np.linalg.norm(self.W, axis=1, keepdims=True)
        self.k = k
        self.wins = np.zeros(units)

    def respond(self, x):
        a = self.W @ x
        th = np.partition(a, -self.k)[-self.k]
        return np.where(a >= th, np.maximum(a, 0), 0).astype(np.float32)

    def learn(self, xs, trace, rate):
        y_bar = None
        for x in xs:
            a = self.W @ x - 0.2 * (self.wins / (self.wins.mean() + 1e-8) - 1)
            y = np.zeros_like(a)
            y[np.argsort(a)[-self.k:]] = 1.0
            y_bar = y if y_bar is None or not trace else 0.6 * y_bar + 0.4 * y  # the trace: what was active a moment ago
            win = y_bar > 0.3
            self.wins += win
            self.W[win] += rate * (x - self.W[win])
            self.W[win] /= np.linalg.norm(self.W[win], axis=1, keepdims=True)


# unlabeled development: photos from all of CIFAR (the W1 pool and beyond) seen as short view sequences
pool = [it for src in cur.train for s in cur.train[src].values() for it in s if isinstance(it[0], tuple) and it[0][0] == "cifar"]
pool = random.Random(1).sample(pool, 6000)
dim = len(l3(cur.image(pool[0][0])))
its = {"trace": IT(dim, seed=1), "no trace": IT(dim, seed=1)}
for n, it in enumerate(pool):
    img = cur.image(it[0])
    seq = [l3(view(img, j)) for j in range(4)]
    for name, layer in its.items():
        layer.learn(seq, trace=(name == "trace"), rate=0.05 * (1 - n / len(pool)) + 0.005)
    if n % 2000 == 0:
        print("developing", n, flush=True)

# probe: picture-book naming (labels used only here)
tr = [x for x in cur.train["W1"]["cifar:name"]][:4000]
te = [x for x in cur.test["W1"]["cifar:name"]][:400]
def feats(items, f):
    X = np.array([f(cur.image(r)) for r, *_ in items]); return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
ytr, yte = [x[2][0] for x in tr], [x[2][0] for x in te]
for name, f in (("L2+L3 2x2 (now)", l3), ("IT trace", lambda im: its["trace"].respond(l3(im))), ("IT no trace", lambda im: its["no trace"].respond(l3(im)))):
    A, B = feats(tr, f), feats(te, f)
    nn = (B @ A.T).argmax(1)
    print(f"{name:18} 1-NN {np.mean([ytr[j] == yte[i] for i, j in enumerate(nn)]):.3f}  ({len(set(ytr))} classes, majority {collections.Counter(ytr).most_common(1)[0][1] / len(ytr):.2f})", flush=True)
