"""CIFAR-10 as a one-pass stream: a gradient readout on the very codes the word cells get (docs/VISION.md, 2026-11-20).

A control, not part of the brain. On the benchmark the word cells through the relay (0.687) are ahead of backprop of
the same shape in one pass (0.616). But that backprop learns its filters and has no relay. What a reader will ask:
is the word cells' lead their own, or is it the code's (fixed filters, decorrelated)? So here the code is kept exactly
as the word cells get it (experiments/bench_codes.py, with or without the relay) and only a readout to the ten
classes is learned by the gradient of a softmax, each picture once, in the stream's order:
    sgd     plain steps, one picture at a time (the nearest thing to the word cells' own delta rule)
    adam    Adam, 1 / 8 / 32 pictures a step
Every setting learns side by side on the same stream. Reported per family: the setting that was best after 10,000
pictures (how the end-to-end backprop was tuned), and the best at the end (generous to the reference).
    SEEDS=0,1,2 VARIANT=developed python experiments/bench_readout.py
"""
import os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
VARIANT = os.environ.get("VARIANT", "developed")
N, TESTS = int(os.environ.get("N", 50000)), int(os.environ.get("TESTS", 2000))
AT = [n for n in (500, 1000, 2000, 5000, 10000, 20000, 50000) if n <= N]
C = 10
SETTINGS = [("sgd", r, 1) for r in (0.003, 0.01, 0.03, 0.1, 0.3)] + [("adam", r, b) for r in (1e-4, 3e-4, 1e-3, 3e-3) for b in (1, 8, 32)]


class Readout:
    def __init__(self, D, kind, rate, batch):
        self.kind, self.rate, self.batch = kind, rate, batch
        self.W, self.c = np.zeros((D, C), np.float32), np.zeros(C, np.float32)
        self.gW, self.gc, self.k, self.t = np.zeros_like(self.W), np.zeros_like(self.c), 0, 0
        self.m, self.v = [np.zeros_like(self.W), np.zeros_like(self.c)], [np.zeros_like(self.W), np.zeros_like(self.c)]

    def learn(self, x, y):
        z = x @ self.W + self.c
        z = np.exp(z - z.max())
        z /= z.sum()
        z[y] -= 1.0
        if self.kind == "sgd":
            self.W -= self.rate * np.outer(x, z)
            self.c -= self.rate * z
            return
        self.gW += np.outer(x, z)
        self.gc += z
        self.k += 1
        if self.k == self.batch:
            self.t += 1
            for p, g, m, v in zip((self.W, self.c), (self.gW, self.gc), self.m, self.v):
                g /= self.batch
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                p -= self.rate * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)
                g[:] = 0
            self.k = 0

    def right(self, X, y):
        return float(np.mean(np.argmax(X @ self.W + self.c, axis=1) == y))


for seed in SEEDS:
    Xtr = np.load(f"out/bench/cifar10_{VARIANT}_s{seed}_train.npy", mmap_mode="r")
    Xte = np.load(f"out/bench/cifar10_{VARIANT}_s{seed}_test.npy", mmap_mode="r")
    ytr, yte = np.load("out/bench/cifar10_train_labels.npy"), np.load("out/bench/cifar10_test_labels.npy")
    relay = pickle.load(open(f"out/bench/cortex_{VARIANT}_s{seed}.pkl", "rb")).relay
    order = np.random.default_rng(seed).permutation(len(Xtr))[:N]  # the stream of bench_stream.py
    tests = np.random.default_rng(99).permutation(len(Xte))[:TESTS]

    def seen(X, i, through):
        v = np.asarray(X[i], np.float32)
        v = relay(v) if through else v / (np.linalg.norm(v) + 1e-8)
        return v * 8.0  # (unit length x8, as bench_backprop.py gives its readout)

    for through in (True, False):
        T = np.stack([seen(Xte, j, through) for j in tests])
        nets, rows, t0 = [Readout(Xtr.shape[1], *s) for s in SETTINGS], [{} for _ in SETTINGS], time.time()
        for n, i in enumerate(order, 1):
            x, y = seen(Xtr, i, through), int(ytr[i])
            for net in nets:
                net.learn(x, y)
            if n in AT:
                for net, row in zip(nets, rows):
                    row[n] = net.right(T, yte[tests])
                print(f"   seed {seed} {'relay' if through else 'plain'} after {n:>6}: best {max(r[n] for r in rows):.3f} ({time.time() - t0:.0f}s)", flush=True)
        for fam in ("sgd", "adam"):
            idx = [k for k, s in enumerate(SETTINGS) if s[0] == fam]
            for label, pick in (("tuned at 10,000", max(idx, key=lambda k: rows[k].get(10000, rows[k][AT[-1]]))), ("best at the end", max(idx, key=lambda k: rows[k][AT[-1]]))):
                _, r, b = SETTINGS[pick]
                print(f"seed {seed} {VARIANT} readout {fam:4} {'+relay' if through else 'plain '} {label:15} (rate {r}, {b} a step): " + " | ".join(f"{n} {a:.3f}" for n, a in rows[pick].items()), flush=True)
