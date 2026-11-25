"""The relay by a local rule, learned as the pictures come (queue E 26 and 28; docs/VISION.md, 2026-11-24 night).

The relay between the cortex and the word cells (organic/cortex2.py `Relay`) removes the summary's own
correlations. It is computed at once from 6,000 summaries, with a constant of mine (`shrink`) that keeps the weak
directions from being blown up. In a brain this is the work of inhibitory interneurons. Here: a published circuit,
its rules unchanged (Lipshutz, Pehlevan & Chklovskii 2022, arXiv 2209.10634, Algorithm 2; the leak of the principal
neurons as in Duong, Simoncelli, Chklovskii & Lipshutz 2023, arXiv 2308.13633, equation 6):
    interneurons     z = W^T y
    principal cells  y settles where  x - W z - y = 0, that is  y = (I + W W^T)^-1 x
    synapses         W += rate (y z^T - W)          (Hebbian, between the two cells at its ends)
The inhibition W W^T can only take away, and the cell's own leak is 1: a direction whose variance is under 1 cannot
be raised, so it is passed as it is, and every direction above is brought down to 1. With the input scaled so that
a cell's variance is 1 on average (each cell's own running average; no constant), the floor falls where the
batch relay's hand-set `shrink` put it within a factor of two (measured: the mean variance is 0.00094 of the
largest, shrink is 0.0005).
Where this probe departs from the papers, and says so: the pictures of a block (BLOCK of them) settle with the
synapses as they were when the block began, and the settled state is computed in closed form (the papers allow
it); the average and the scale are running averages over everything seen.

Read first on the code (how alike two pictures look, how many directions it uses, what share of the neurons'
pairs is still correlated), then on the stream with the word cells. Arms:
    relay       the batch relay (the reference)
    settled     where the circuit's rule comes to rest, computed at once (its fixed point)
    developed   the circuit after one sight of unlabelled pictures, then held still
    developed20 the same after 20,000 unlabelled pictures (does more sight close what is left?)
    plastic     the circuit learning from nothing along the stream, while the word cells learn from what it sends
    lifelong    developed as above, then learning on along the stream (what a brain's relay would do)
The unlabelled pictures (experiments/bench_dev_codes.py; out/bench/cifar100_norm0_s{seed}_dev.npy): the first 6,000
are the very pictures the batch relay was settled on, so `settled`, `developed` and `lifelong` have the relay's
own data and none of the stream's. If that file is missing the stream's own pictures are used, without their
names, and the label says so (the first run of seed 0, 2026-11-24, was such a run).
    SEEDS=0 VARIANT=norm0 ARMS=relay,settled,developed,developed20,plastic,lifelong python experiments/relay_local_probe.py
"""
import collections, os, pickle, sys, time
import numpy as np
import scipy.linalg as sl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
VARIANT = os.environ.get("VARIANT", "norm0")
ARMS = os.environ.get("ARMS", "relay,settled,developed,developed20,plastic,lifelong").split(",")
RATE, BLOCK = float(os.environ.get("RATE", 1e-3)), int(os.environ.get("BLOCK", 50))  # the rate: the papers' (1e-3 a picture)
N = int(os.environ.get("N", 10000))
AT = [n for n in (500, 1000, 2000, 5000, 10000) if n <= N]
NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
ytr, yte = np.load("out/bench/cifar10_train_labels.npy"), np.load("out/bench/cifar10_test_labels.npy")


class Circuit:
    """Principal cells and as many interneurons; see the top of the file."""

    def __init__(self, D, rng):
        self.W = (rng.standard_normal((D, D)) / np.sqrt(D)).astype(np.float32)  # born random, W W^T about 1
        self.mu, self.var, self.n = np.zeros(D, np.float32), 1.0, 0
        self.A = None

    def _settle(self):
        M = self.W @ self.W.T
        M[np.diag_indices_from(M)] += 1.0
        self.A = sl.cho_factor(M, lower=True, overwrite_a=True)  # kept factored: the settled state of any picture is two triangular solves

    def code(self, V):
        """What the principal cells send for each row of V (with the synapses as they are)."""
        if self.A is None:
            self._settle()
        X = (np.asarray(V, np.float32) - self.mu) / np.float32(np.sqrt(self.var))
        return sl.cho_solve(self.A, X.T).T

    def learn(self, V):
        """A block of pictures: every cell's running average and scale, then every synapse by its own two cells."""
        V = np.asarray(V, np.float32)
        Y = self.code(V)
        Z = Y @ self.W
        self.W += RATE * (Y.T @ Z - len(V) * self.W)
        self.A = None
        n = self.n + len(V)
        self.mu += (V - self.mu).sum(axis=0) / n
        self.var += (float(((V - self.mu) ** 2).mean(axis=1).sum()) - len(V) * self.var) / n  # (born at 1: the first block sets it)
        self.n = n


def fixed_point(V):
    """Where the circuit's rule comes to rest for these pictures: variances above 1 (in the circuit's units)
    brought to 1, the others left."""
    mu = V.mean(axis=0)
    C = np.cov((V - mu).T).astype(np.float64)
    d, E = np.linalg.eigh(C)
    d = np.maximum(d, 0) / (np.trace(C) / len(C))
    Z = ((E / np.maximum(np.sqrt(d), 1.0)) @ E.T).astype(np.float32)
    return lambda X: (np.asarray(X, np.float32) - mu) @ Z, int((d > 1).sum())


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def read_code(label, X, y, seed):
    """As experiments/expand_probe.py, and the share of pairs of cells whose answers go together by more than 0.1."""
    Xc = X - X.mean(axis=0)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8
    L = (Xc @ Xc.T).astype(np.float64)
    off, same = ~np.eye(len(L), dtype=bool), y[:, None] == y[None, :]
    s = np.maximum(np.linalg.eigvalsh(L), 0)
    cells = np.random.default_rng(3).permutation(X.shape[1])[:512]
    R = np.corrcoef((X[:, cells] + 1e-9 * np.random.default_rng(4).standard_normal((len(X), 512))).T)
    print(f"seed {seed} {label:11} two pictures alike {np.abs(L[off]).mean():.3f} (same class {L[off & same].mean():+.4f}, different {L[off & ~same].mean():+.4f}) | "
          f"directions used {s.sum() ** 2 / (s * s).sum():.0f} | pairs of cells that go together {float((np.abs(R[~np.eye(512, dtype=bool)]) > 0.1).mean()):.3f}", flush=True)


class Cells:
    def __init__(self, D):
        from organic.words import Words
        from organic.curriculum import tidy
        self.tidy = tidy
        ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
        self.words = Words(D, ear)
        self.words.plastic_by_error = False
        self.ask = ear.units("what is this?", learn=False)
        self.said = [ear.units(w, learn=False) for w in NAMES]

    def learn(self, x, y):
        self.words.learn(self.ask, x, self.said[y])

    def name(self, x):
        w = self.tidy(self.words.recall(self.ask, x)[0] or "")
        return NAMES.index(w) if w in NAMES else -1


def stream(code, learn, Vtr, Vte, order, tests, seed, label):
    """The word cells on the stream (as experiments/bench_stream.py). `code(V)`: what the cells get for the
    summaries V now; `learn(V)`: the relay's own learning from a block that has just been lived (None: it is held
    still). A picture seen once is shown again twenty pictures later (coded as the relay is then)."""
    t0, cells, row, lag, shots = time.time(), Cells(Vtr.shape[1]), {}, collections.deque(), []
    for a in range(0, len(order), BLOCK):
        block = order[a: a + BLOCK]
        X = unit(code(Vtr[block]))
        for n, (i, x) in enumerate(zip(block, X), a + 1):
            if 5000 < n <= 5320:
                if len(lag) == 20:
                    io = lag.popleft()
                    shots.append(cells.name(unit(code(Vtr[[io]]))[0]) == int(ytr[io]))
                lag.append(i)
            cells.learn(x, int(ytr[i]))
            if n in AT:
                te = unit(code(Vte[tests]))
                row[n] = float(np.mean([cells.name(v) == int(yte[j]) for v, j in zip(te, tests)]))
        if learn is not None:
            learn(Vtr[block])
    print(f"seed {seed} {label:11} cells " + " | ".join(f"{n} {a:.3f}" for n, a in row.items())
          + (f" | seen once, 20 later: {np.mean(shots):.3f}" if shots else "") + f" ({time.time() - t0:.0f}s)", flush=True)


def main():
    for seed in SEEDS:
        relay = pickle.load(open(f"out/bench/cortex_{VARIANT}_s{seed}.pkl", "rb")).relay
        Vtr = np.asarray(np.load(f"out/bench/cifar10_{VARIANT}_s{seed}_train.npy")[:N], np.float32)
        Vte = np.asarray(np.load(f"out/bench/cifar10_{VARIANT}_s{seed}_test.npy"), np.float32)
        order = np.random.default_rng(seed).permutation(len(Vtr))[:N]
        tests = np.random.default_rng(99).permutation(len(Vte))[:2000]
        sample = tests[:1500]
        dev = f"out/bench/cifar100_{VARIANT}_s{seed}_dev.npy"
        own = os.path.exists(dev)  # the unlabelled pictures the batch relay had; else the stream's own
        Vdev = np.load(dev, mmap_mode="r") if own else Vtr
        print(f"seed {seed} development on " + ("the unlabelled pictures the batch relay was settled on" if own else "THE STREAM'S OWN PICTURES (no names)"), flush=True)
        for arm in ARMS:
            learn = None
            if arm == "relay":
                code = lambda V: (np.asarray(V, np.float32) - relay.mu) @ relay.Z  # noqa: E731
            elif arm == "settled":
                code, above = fixed_point(np.asarray(Vdev[:6000] if own else Vdev, np.float32))
                print(f"seed {seed} settled    directions above the floor {above} of {Vtr.shape[1]}", flush=True)
            else:
                t0, c = time.time(), Circuit(Vtr.shape[1], np.random.default_rng(2000 + seed))
                code = c.code
                if arm != "plastic":
                    sights = len(Vdev) if arm == "developed20" or not own else 6000
                    first = np.arange(sights) if own else np.random.default_rng(500 + seed).permutation(len(Vdev))
                    for a in range(0, len(first), BLOCK):
                        c.learn(np.asarray(Vdev[first[a: a + BLOCK]], np.float32))
                    print(f"seed {seed} {arm:10} developed on {len(first)} pictures ({time.time() - t0:.0f}s)", flush=True)
                if arm in ("plastic", "lifelong"):
                    learn = c.learn
            if learn is None:
                read_code(arm, code(Vte[sample]), yte[sample], seed)
            stream(code, learn, Vtr, Vte, order, tests, seed, arm)
            if learn is not None:
                read_code(arm, code(Vte[sample]), yte[sample], seed)


if __name__ == "__main__":
    main()
