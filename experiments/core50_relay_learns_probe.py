"""Does a relay that keeps learning in the world it is in give the word cells' rule the middle fixation's gain?
(docs/VISION.md 2026-11-27 08:55, said before it runs.)

core50_spectrum_check.py found that a relay settled again on CORe50's own frames (no labels) lifts the delta reader on
the middle fixation from 0.34 to 0.40-0.41 on three seeds (ridge's level), and not on the whole frame: the batch relay
was settled on COCO, and a delta rule needs the code's directions evened out in the world it reads. A brain has no
second batch: its relay is a circuit of interneurons that learns for as long as it is used (organic/cortex2.py
RelayCircuit; Lipshutz, Pehlevan & Chklovskii 2022; on CIFAR it matched the batch relay, VISION.md 2026-11-25). Here
the circuit is born random on top of the relay as it was settled on COCO, and learns along the benchmark's stream
(10,000 frames, one pass, no labels, its published rule and our step 1e-4, a change every 50 views), while the delta
reader (bench_stream.py's, the word cells' rule without the words) learns from what it sends at each moment. The
test frames are then read with the circuit as it ended (held still). Beside it, the same reader on the relay alone.

    SEEDS=0,1,2 python experiments/core50_relay_learns_probe.py
"""
import os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import RelayCircuit  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
N = int(os.environ.get("N", 10000))  # (a smoke test: fewer)
ytr, yte = np.load("out/bench/core50_train_labels.npy"), np.load("out/bench/core50_test_labels.npy")
C = int(ytr.max()) + 1
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
SHARE = 2.0 + len(set(ear.units("what is this?", learn=False)))
del ear


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


class Delta:
    """bench_stream.py's Delta, the same arithmetic."""
    def __init__(self, D):
        self.W, self.base = np.zeros((C, D), np.float32), np.zeros(C, np.float32)
        self.use, self.mean, self.seen = np.zeros(D, np.float32), np.zeros(D, np.float32), 0

    def _x(self, v, learn):
        if learn:
            self.seen += 1
            self.mean += (v - self.mean) / min(self.seen, 1000)
        x = v - self.mean
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    def learn(self, v, y):
        x = self._x(v, True)
        e = -(self.W @ x + self.base)
        e[y] += 1.0
        step = e / SHARE
        self.W += step[:, None] * (x / (1.0 + self.use / 5.0))[None, :]
        self.use += x * x
        self.base += step / (1.0 + self.seen / 5.0)

    def names(self, V, y):
        return float(np.mean(np.argmax(self._x(V, False) @ self.W.T + self.base, axis=1) == y))


for seed in SEEDS:
    order = np.random.default_rng(seed).permutation(len(ytr))[:N]
    y = ytr[order]
    for v in ("first", "centre"):
        t0 = time.time()
        relay = pickle.load(open(f"out/bench/cortex_c50{v}_s{seed}.pkl", "rb")).relay
        X = unit((np.asarray(np.load(f"out/bench/core50_c50{v}_s{seed}_train.npy", mmap_mode="r")[order], np.float32) - relay.mu) @ relay.Z)
        Xt = unit((np.asarray(np.load(f"out/bench/core50_c50{v}_s{seed}_test.npy"), np.float32) - relay.mu) @ relay.Z)
        alone, learns, circuit = Delta(X.shape[1]), Delta(X.shape[1]), RelayCircuit(X.shape[1], rate=1e-4, block=50, seed=seed)
        at = {}
        for n, (x, c) in enumerate(zip(X, y), 1):
            alone.learn(x, c)
            learns.learn(circuit(x), c)
            if n in (1000, 5000, N):
                circuit.plastic = False
                at[n] = (alone.names(Xt, yte), learns.names(unit(circuit.code(Xt)), yte))
                circuit.plastic = True
        circuit.plastic = False
        d = np.clip(np.linalg.eigvalsh(np.cov(unit(circuit.code(X)).T.astype(np.float64))), 0, None)
        print(f"seed {seed} {v:6}: delta on the relay alone / with the circuit learning on: " + " | ".join(f"{n} {a:.3f} / {b:.3f}" for n, (a, b) in at.items())
              + f" | the circuit's code on the stream: participation ratio {d.sum() ** 2 / (d * d).sum():.0f} ({time.time() - t0:.0f}s)", flush=True)
