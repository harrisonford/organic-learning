"""CIFAR-10 as a one-pass stream: backprop of the SOFT-RULE layer's shape (queue E 27; docs/VISION.md, 2026-11-25).

The reference beside the soft winner-take-all layer of experiments/softhebb_probe.py, as experiments/bench_backprop.py
is the reference beside our cortex. The same way in and the same shape: each colour standardised with the numbers
the soft layer kept (no labels in them), every 5 x 5 window with mirrored borders, 256 filters, the response
(what is above the average over the filters, to the power POWER), the average over 4 x 4 places at every second
place, the average in 4 x 4 regions, unit length, a linear readout to the ten classes; softmax, Adam. Every
labelled picture comes once, in the order of bench_stream.py for the seed. What is varied:
    FILTERS   random      born random at unit length, learned by backprop (end to end)
              developed   the soft rule's developed filters as the start, then learned by backprop
              frozen      the soft rule's developed filters kept; only the readout learned
    POWER     0.7 is the layer's own; 1 is the usual rectifier (backprop is given whichever serves it better). With
              0.7 the slope at zero is infinite, so the slope is taken at no less than EPS above zero.
    RATE, BATCH, REPLAY   as in bench_backprop.py (tuned on the first 10,000 pictures of seed 0)
    RELAY     fixed: the relay step in front of the readout (organic.cortex2.Relay, the same shrinkage), settled
              once on the 6,000 unlabelled CIFAR-100 pictures the soft layer's own relay had, seen through the
              filters as they are born, and kept; errors go back through it
    CHECK=1   only compares this file's way in, with the developed filters, against the codes kept by the probe
    SEEDS=0 FILTERS=random RATE=0.001 BATCH=32 N=50000 python experiments/bench_backprop_soft.py
"""
import os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import softhebb_probe as P  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
FILTERS, CORTEX_SEED = os.environ.get("FILTERS", "random"), int(os.environ.get("CORTEX_SEED", 0))
RATE, BATCH, REPLAY = float(os.environ.get("RATE", 1e-3)), int(os.environ.get("BATCH", 32)), int(os.environ.get("REPLAY", 0))
N, TESTS = int(os.environ.get("N", 50000)), int(os.environ.get("TESTS", 2000))
RELAY, SHRINK, POWER, EPS = os.environ.get("RELAY", ""), 0.0005, float(os.environ.get("POWER", 0.7)), 1e-3
AT = [n for n in (500, 1000, 2000, 5000, 10000, 20000, 50000) if n <= N]
G, C = 4, 10

# the average over 4 x 4 places at every second place (a border of nothing, counted), then over 4 x 4 regions
POOL = np.zeros((G * G, 32, 32), np.float32)
for i in range(16):
    for j in range(16):
        for r in range(max(2 * i - 1, 0), min(2 * i + 3, 32)):
            for c in range(max(2 * j - 1, 0), min(2 * j + 3, 32)):
                POOL[(i // 4) * G + j // 4, r, c] += 1.0 / 256
POOL = POOL.reshape(G * G, 1024)


class Net:
    def __init__(self, seed, kept, unlabelled):
        rng = np.random.default_rng(seed)
        self.standard = P.Standard()
        self.standard.mean, self.standard.var = kept.standard
        K = len(kept.W)
        self.W = kept.W.astype(np.float32).copy() if FILTERS in ("developed", "frozen") else (lambda w: w / np.linalg.norm(w, axis=1, keepdims=True))(rng.normal(0, 1, kept.W.shape).astype(np.float32))
        self.b, self.R, self.c = np.zeros(K, np.float32), np.zeros((G * G * K, C), np.float32), np.zeros(C, np.float32)
        self.names = ["R", "c"] + ([] if FILTERS == "frozen" else ["W", "b"])
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.names}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.names}
        self.t, self.relay = 0, None
        if RELAY:
            from organic.cortex2 import Relay
            S = []
            for x in unlabelled:
                raw = self.summary(self.windows(x))[1].ravel()
                S.append(raw / (np.linalg.norm(raw) + 1e-8))
            self.relay = Relay(SHRINK).settle(np.array(S, np.float32))

    def windows(self, x):
        return P.windows(self.standard(x[None] / np.float32(255))).reshape(1024, 75)

    def summary(self, v):
        u = v @ self.W.T + self.b
        a = np.maximum(u - u.mean(axis=1, keepdims=True), 0)
        return a, POOL @ (a if POWER == 1 else a ** POWER)

    def forward(self, v):
        a, p = self.summary(v)
        raw = p.ravel()
        s_ = raw / (np.linalg.norm(raw) + 1e-8)
        if self.relay is not None:
            w = (s_ - self.relay.mu) @ self.relay.Z
            f = w / (np.linalg.norm(w) + 1e-8) * 8.0
            return f @ self.R + self.c, (v, a, raw, f, w)
        return (s_ * 8.0) @ self.R + self.c, (v, a, raw, s_ * 8.0)

    def grads(self, logits, y, kept):
        v, a, raw, f = kept[:4]
        z = np.exp(logits - logits.max())
        z /= z.sum()
        z[y] -= 1.0
        g = {"R": np.outer(f, z), "c": z}
        if "W" in self.names:
            df, norm = self.R @ z, np.linalg.norm(raw) + 1e-8
            if self.relay is not None:  # back through the relay: f = 8 w / |w|, w = (raw / |raw| - mu) Z
                w = kept[4]
                wn = np.linalg.norm(w) + 1e-8
                ds = (8.0 * (df - (df @ w) * w / wn ** 2) / wn) @ self.relay.Z.T
                draw = (ds - (ds @ raw) * raw / norm ** 2) / norm
            else:
                draw = 8.0 * (df - (df @ raw) * raw / norm ** 2) / norm
            da = POOL.T @ draw.reshape(G * G, -1)
            da *= (a > 0) if POWER == 1 else (a > 0) * POWER * np.maximum(a, EPS) ** (POWER - 1)
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


if __name__ == "__main__":
    Xtr = np.concatenate([P.load(f"{P.DATA}/cifar-10-batches-py/data_batch_{i}") for i in range(1, 6)])
    Xte = P.load(f"{P.DATA}/cifar-10-batches-py/test_batch")
    ytr, yte = np.load("out/bench/cifar10_train_labels.npy"), np.load("out/bench/cifar10_test_labels.npy")
    kept = pickle.load(open(f"out/bench/cortex_sh_s{CORTEX_SEED}.pkl", "rb"))
    unlabelled = None
    if RELAY:
        unlabelled = P.load(f"{P.DATA}/cifar-100-python/train")
        unlabelled = unlabelled[np.random.default_rng(CORTEX_SEED + 7).choice(len(unlabelled), P.NRELAY, replace=False)]
    if os.environ.get("CHECK") == "1":
        assert FILTERS in ("developed", "frozen")
        net = Net(0, kept, unlabelled)
        mine = np.array([(lambda r: r / (np.linalg.norm(r) + 1e-8))(net.summary(net.windows(x))[1].ravel()) for x in Xtr[:200]])
        theirs = np.asarray(np.load(f"out/bench/cifar10_sh_s{CORTEX_SEED}_train.npy", mmap_mode="r")[:200], np.float32)
        print(f"power {POWER:g} (the probe's codes are at 0.7) | the way in against the probe's codes, 200 pictures: largest difference {float(np.abs(mine - theirs).max()):.2g} (a code's largest number {float(np.abs(theirs).max()):.2g})")
        if RELAY:
            print(f"the relay against the probe's: largest difference {float(np.abs(net.relay.Z - kept.relay.Z).max()):.2g} (its largest number {float(np.abs(kept.relay.Z).max()):.2g})")
        # the way back against small changes, on one picture, in double precision
        v, y = net.windows(Xtr[0]).astype(np.float64), int(ytr[0])
        net.W, net.b, net.R = net.W.astype(np.float64), net.b.astype(np.float64), np.random.default_rng(1).normal(0, 0.05, net.R.shape)
        net.names = ["R", "c", "W", "b"]
        logits, held = net.forward(v)
        g = net.grads(logits, y, held)
        loss = lambda: (lambda l: float(np.log(np.exp(l - l.max()).sum()) + l.max() - l[y]))(net.forward(v)[0].astype(np.float64))  # noqa: E731
        rng, worst = np.random.default_rng(2), 0.0
        for _ in range(6):
            k, i = int(rng.integers(len(net.W))), int(rng.integers(75))
            keep = net.W[k, i]
            net.W[k, i] = keep + 1e-5
            up = loss()
            net.W[k, i] = keep - 1e-5
            down = loss()
            net.W[k, i] = keep
            print(f"   filter {k} weight {i}: by the way back {g['W'][k, i]:+.6f}, by a small change {(up - down) / 2e-5:+.6f}")
        sys.exit(0)
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(Xtr))[:N]  # the stream of bench_stream.py
        tests = np.random.default_rng(99).permutation(len(Xte))[:TESTS]
        net, t0, again, row, acc, k = Net(seed, kept, unlabelled), time.time(), np.random.default_rng(seed + 3), {}, None, 0
        for n, i in enumerate(order, 1):
            for j in [i] + ([int(order[again.integers(n)]) for _ in range(REPLAY)] if n > 1 else []):
                logits, held = net.forward(net.windows(Xtr[j]))
                g = net.grads(logits, int(ytr[j]), held)
                acc, k = (g if acc is None else {q: acc[q] + g[q] for q in g}), k + 1
                if k == BATCH:
                    net.step(acc, BATCH)
                    acc, k = None, 0
            if n in AT:
                row[n] = float(np.mean([int(np.argmax(net.forward(net.windows(Xte[j]))[0])) == int(yte[j]) for j in tests]))
                print(f"   seed {seed} backprop soft shape {FILTERS} relay {RELAY or 'none'} power {POWER:g} rate {RATE} batch {BATCH} replay {REPLAY} after {n:>6}: {row[n]:.3f} ({time.time() - t0:.0f}s)", flush=True)
        print(f"seed {seed} backprop soft shape filters {FILTERS:9} relay {RELAY or 'none':5} power {POWER:<3g} rate {RATE:<7} batch {BATCH:<3} replay {REPLAY}: " + " | ".join(f"{n} {a:.3f}" for n, a in row.items()), flush=True)
