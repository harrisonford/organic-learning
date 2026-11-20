"""CIFAR-10 as a one-pass stream: who learns what from each picture seen once (docs/PUBLISH.md, 2026-11-19).

The codes come from experiments/bench_codes.py (a cortex developed without labels on CIFAR-100's pictures, or never
developed). Every labelled CIFAR-10 training picture is then seen once, in a random order; held-out test pictures
are named at fixed points of the stream. Learners on the same codes:
    cells      the brain's word cells (organic/words.py: delta rule, one moment at a time, full-size corrections),
               asked "what is this?" and told the class's name
    ncm        nearest class mean (running means; no gradient)
    slda       streaming linear discriminant analysis (running means and one shared covariance; Hayes & Kanan 2020)
    ridge      a ridge on everything seen so far (not a stream learner: what a linear readout can get; the reference)
"+relay": the code decorrelated on its way (the cortex's relay). "+store": the word cells with the episodic store.
Also read, between 5,000 and 5,300 pictures: is a picture seen once named right twenty pictures later (one-shot)?
    SEEDS=0 VARIANT=developed ARMS=cells+relay,cells,cells+relay+store,ncm,ncm+relay,slda,slda+relay,ridge,ridge+relay python experiments/bench_stream.py
"""
import collections, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
VARIANT, CORTEX_SEED = os.environ.get("VARIANT", "developed"), int(os.environ.get("CORTEX_SEED", 0))
ARMS = os.environ.get("ARMS", "cells+relay,cells,cells+relay+store,ncm,ncm+relay,slda,slda+relay,ridge,ridge+relay").split(",")
N, TESTS = int(os.environ.get("N", 50000)), int(os.environ.get("TESTS", 2000))
AT = [n for n in (500, 1000, 2000, 5000, 10000, 20000, 50000) if n <= N]
NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
Xtr = np.load(f"out/bench/cifar10_{VARIANT}_s{CORTEX_SEED}_train.npy", mmap_mode="r")
Xte = np.load(f"out/bench/cifar10_{VARIANT}_s{CORTEX_SEED}_test.npy", mmap_mode="r")
ytr, yte = np.load("out/bench/cifar10_train_labels.npy"), np.load("out/bench/cifar10_test_labels.npy")
relay = pickle.load(open(f"out/bench/cortex_{VARIANT}_s{CORTEX_SEED}.pkl", "rb")).relay
D, C = Xtr.shape[1], len(NAMES)


def seen(X, i, through):
    v = np.asarray(X[i], np.float32)
    return relay(v) if through else v


class Cells:
    def __init__(self, store):
        from organic.words import Words
        from organic.curriculum import tidy
        self.tidy = tidy
        ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
        self.words = Words(D, ear)
        self.words.plastic_by_error = False
        if store:
            from organic.episodes import Episodes
            self.words.episodes = Episodes(D, ear.width)
        self.ask = ear.units("what is this?", learn=False)
        self.said = [ear.units(w, learn=False) for w in NAMES]

    def learn(self, x, y):
        self.words.learn(self.ask, x, self.said[y])

    def name(self, x):
        w = self.tidy(self.words.recall(self.ask, x)[0] or "")
        return NAMES.index(w) if w in NAMES else -1


class NCM:
    def __init__(self):
        self.sum, self.n = np.zeros((C, D), np.float64), np.zeros(C)

    def learn(self, x, y):
        self.sum[y] += x
        self.n[y] += 1

    def ready(self):
        self.mu = (self.sum / np.maximum(self.n, 1)[:, None]).astype(np.float32)
        self.b = -0.5 * (self.mu * self.mu).sum(axis=1)

    def name(self, x):
        return int(np.argmax(self.mu @ x + self.b))


class Moments(NCM):
    """What a streaming linear learner keeps: every class's mean and the pictures' second moments (added up as they
    come, a block at a time). From these, at any point of the stream:
      slda    linear discriminant analysis with one covariance shared by the classes (streaming LDA, Hayes & Kanan
              2020, here from the exact sums rather than their running estimate), shrunk a little before inverting
      ridge   a ridge regression to the classes on everything seen so far
    Each tries a few strengths of shrinkage and keeps the best on the test pictures (generous to the reference)."""
    def __init__(self, kind):
        super().__init__()
        self.kind, self.A, self.block = kind, np.zeros((D, D), np.float64), []

    def learn(self, x, y):
        super().learn(x, y)
        self.block.append(x)
        if len(self.block) == 256:
            self.flush()

    def flush(self):
        if self.block:
            X = np.stack(self.block)
            self.A += (X.T @ X).astype(np.float64)
            self.block = []

    def ready(self):
        self.flush()
        n, mu = self.n.sum(), self.sum / np.maximum(self.n, 1)[:, None]
        self.readouts = []
        if self.kind == "slda":
            S = (self.A - (mu * self.n[:, None]).T @ mu) / n  # the scatter within the classes
            for eps in (1e-3, 1e-2, 1e-1):
                W = np.linalg.solve((1 - eps) * S + eps * (np.trace(S) / D) * np.eye(D), mu.T).T
                self.readouts.append((W.astype(np.float32), (-0.5 * (W * mu).sum(axis=1)).astype(np.float32), np.zeros(D, np.float32)))
        else:
            m = self.sum.sum(axis=0) / n
            A = self.A - n * np.outer(m, m)
            B = self.sum.T - np.outer(m, self.n)
            for lam in (0.01, 0.1, 1.0):
                W = np.linalg.solve(A + lam * (np.trace(A) / D) * np.eye(D), B)
                self.readouts.append((W.T.astype(np.float32), np.zeros(C, np.float32), m.astype(np.float32)))

    def names(self, X, y):
        """The best of the readouts on these pictures: (accuracy, which)."""
        accs = [float(np.mean(np.argmax((X - m) @ W.T + b, axis=1) == y)) for W, b, m in self.readouts]
        return max(accs), int(np.argmax(accs))


def make(arm):
    kind = arm.split("+")[0]
    return Cells("store" in arm) if kind == "cells" else NCM() if kind == "ncm" else Moments(kind)


for seed in SEEDS:
    order = np.random.default_rng(seed).permutation(len(Xtr))[:N]
    tests = np.random.default_rng(99).permutation(len(Xte))[:TESTS]
    for arm in ARMS:
        through, t0, learner = "relay" in arm, time.time(), make(arm)
        row, lag, shots = {}, collections.deque(), []
        for n, i in enumerate(order, 1):
            x, y = seen(Xtr, i, through), int(ytr[i])
            if 5000 < n <= 5320 and not isinstance(learner, Moments):
                if len(lag) == 20:  # a picture seen once, twenty pictures ago: is it named right now?
                    if isinstance(learner, NCM):
                        learner.ready()
                    xo, yo = lag.popleft()
                    shots.append(learner.name(xo) == yo)
                lag.append((x, y))
            learner.learn(x, y)
            if n in AT:
                if isinstance(learner, Moments):
                    learner.ready()
                    row[n], _ = learner.names(np.stack([seen(Xte, j, through) for j in tests]), yte[tests])
                else:
                    if isinstance(learner, NCM):
                        learner.ready()
                    row[n] = float(np.mean([learner.name(seen(Xte, j, through)) == int(yte[j]) for j in tests]))
                print(f"   seed {seed} {arm:18} after {n:>6}: {row[n]:.3f} ({time.time() - t0:.0f}s)", flush=True)
        print(f"seed {seed} {VARIANT} {arm:18} " + " | ".join(f"{n} {a:.3f}" for n, a in row.items())
              + (f" | seen once, 20 later: {np.mean(shots):.3f}" if shots else ""), flush=True)
