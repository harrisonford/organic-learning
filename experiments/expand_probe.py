"""A sparse expansion before the cells that associate (queue E 29f; docs/VISION.md, 2026-11-24 evening).

Three brain structures put a wide, sparsely active layer in front of the cells that learn associations: the
mushroom body (about 50 glomeruli onto 2,000 Kenyon cells, each with about 7 claws on glomeruli picked at random,
5-10% of the cells active, the level held by one inhibitory neuron that all of them drive), the cerebellum's
granule layer (4 dendrites each) and the dentate gyrus in front of CA3 (a few times more cells than its input,
2-5% active). Nothing is learned in the wiring: it is random, and the inhibition is one loop.

Here: the cortex's summary, through the relay, onto K cells; each sums FAN inputs picked at random; the ACTIVE
most driven stay on, each by how far it is above the inhibition's level (the least driven of those that stay).
Read first on the code itself (how alike two different pictures look; how alike two of one class; how many
directions the code uses), then on the stream with the word cells and with a ridge on everything seen (in its
dual form: the code is too wide for the other), and on a picture seen once and asked about twenty later.
Controls: as many cells as inputs (ratio 1: sparseness without expansion) and the same wiring with no
inhibition (a linear map: expansion without sparseness).
    SEEDS=0 VARIANT=norm0 RATIOS=1,4,10 python experiments/expand_probe.py
"""
import collections, os, pickle, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
VARIANT = os.environ.get("VARIANT", "norm0")
RATIOS = [float(x) for x in os.environ.get("RATIOS", "1,4,10").split(",")]
FAN, ACTIVE = int(os.environ.get("FAN", 7)), float(os.environ.get("ACTIVE", 0.05))
N = int(os.environ.get("N", 10000))
AT = [n for n in (500, 1000, 2000, 5000, 10000) if n <= N]
NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
ytr, yte = np.load("out/bench/cifar10_train_labels.npy"), np.load("out/bench/cifar10_test_labels.npy")


def wiring(K, D, rng):
    """Each of K cells takes FAN of the D inputs, picked at random, all at the same strength."""
    cols = np.stack([rng.choice(D, FAN, replace=False) for _ in range(K)]).ravel()
    return sp.csr_matrix((np.ones(K * FAN, np.float32), (np.repeat(np.arange(K), FAN), cols)), shape=(K, D))


def expand(P, X, active):
    """What the wide layer passes on for each row of X. active None: no inhibition (the cells' sums as they are)."""
    H = np.asarray((P @ X.T).T, np.float32)
    if active is None:
        return H
    k = max(1, int(round(active * H.shape[1])))
    level = np.partition(H, -k - 1, axis=1)[:, -k - 1][:, None]  # the inhibition: the most driven cell that stays off
    return np.maximum(H - level, 0)


def read_code(X, y):
    """How alike two different pictures look, two of one class, two of different classes (mean cosine of the
    codes as the word cells get them: each number's deviation from its average); and how many directions the code
    uses (the participation ratio of its variances)."""
    Xc = X - X.mean(axis=0)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8
    L = (Xc @ Xc.T).astype(np.float64)  # every pair's likeness
    off, same = ~np.eye(len(L), dtype=bool), y[:, None] == y[None, :]
    s = np.maximum(np.linalg.eigvalsh(L), 0)  # the variances along the code's directions
    return float(np.abs(L[off]).mean()), float(L[off & same].mean()), float(L[off & ~same].mean()), float(s.sum() ** 2 / (s * s).sum())


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


def stream(code, D, order, tests, seed, label):
    """The word cells on the stream (as experiments/bench_stream.py: every picture once, tests at fixed points; a
    picture seen once and asked about twenty later, between 5,000 and 5,320)."""
    t0, cells, row, lag, shots = time.time(), Cells(D), {}, collections.deque(), []
    te = code("te", tests)
    for n, i in enumerate(order, 1):
        x, y = code("tr", [i])[0], int(ytr[i])
        if 5000 < n <= 5320:
            if len(lag) == 20:
                xo, yo = lag.popleft()
                shots.append(cells.name(xo) == yo)
            lag.append((x, y))
        cells.learn(x, y)
        if n in AT:
            row[n] = float(np.mean([cells.name(v) == int(yte[j]) for v, j in zip(te, tests)]))
    print(f"seed {seed} {label:22} cells " + " | ".join(f"{n} {a:.3f}" for n, a in row.items())
          + (f" | seen once, 20 later: {np.mean(shots):.3f}" if shots else "") + f" ({time.time() - t0:.0f}s)", flush=True)


def ridge(G, Kt, K, y, yt, seed, label):
    """A ridge to the classes on everything seen, in its dual form (from the pictures' likenesses to each other: G
    among the seen ones, Kt of the test pictures to them; the code's average taken out here); three strengths, the
    best on the test pictures kept (generous to the reference, as in bench_stream)."""
    g, c = G.mean(axis=1), G.mean()
    Kt = Kt - Kt.mean(axis=1, keepdims=True) - g[None, :] + c
    G -= g[:, None]
    G -= g[None, :]
    G += c
    Y, accs, ridge_unit = np.eye(len(NAMES), dtype=np.float32)[y], [], float(np.trace(G)) / K
    for lam in (0.01, 0.1, 1.0):
        G[np.diag_indices_from(G)] += lam * ridge_unit
        alpha = sl.cho_solve(sl.cho_factor(G, lower=True), Y)
        G[np.diag_indices_from(G)] -= lam * ridge_unit
        accs.append(float(np.mean(np.argmax(Kt @ alpha, axis=1) == yt)))
    print(f"seed {seed} {label:22} ridge on {len(y)}: {max(accs):.3f}", flush=True)


for seed in SEEDS:
    relay = pickle.load(open(f"out/bench/cortex_{VARIANT}_s{seed}.pkl", "rb")).relay
    through = lambda X: (lambda W: W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8))((np.asarray(X, np.float32) - relay.mu) @ relay.Z)
    Rtr = through(np.load(f"out/bench/cifar10_{VARIANT}_s{seed}_train.npy")[:N])
    Rte = through(np.load(f"out/bench/cifar10_{VARIANT}_s{seed}_test.npy"))
    D = Rtr.shape[1]
    order = np.random.default_rng(seed).permutation(len(Rtr))[:N]
    tests = np.random.default_rng(99).permutation(len(Rte))[:2000]
    y, yt = ytr[: len(Rtr)], yte[tests]
    sample = np.random.default_rng(5).permutation(len(Rtr))[:2000]

    def report(label, X):
        a, same, other, dim = read_code(X, y[sample])
        print(f"seed {seed} {label:22} two pictures alike {a:.3f} (same class {same:+.4f}, different {other:+.4f}) | "
              f"directions used {dim:.0f} of {X.shape[1]} | share of cells on {float((X != 0).mean()):.3f}", flush=True)

    report("relay", Rtr[sample])
    stream(lambda part, idx: (Rtr if part == "tr" else Rte)[idx], D, order, tests, seed, "relay")
    ridge(Rtr[order] @ Rtr[order].T, Rte[tests] @ Rtr[order].T, D, y[order], yt, seed, "relay")
    arms = [(f"x{r:g} sparse", int(round(r * D)), ACTIVE) for r in RATIOS] + [("x4 no inhibition", 4 * D, None)]
    for label, K, active in arms:
        P = wiring(K, D, np.random.default_rng(1000 + seed))
        code = lambda part, idx, P=P, active=active: expand(P, (Rtr if part == "tr" else Rte)[idx], active)
        report(label, code("tr", sample))
        stream(code, K, order, tests, seed, label)
        if active is None:  # a linear map of the relay's code: the likenesses through the wiring's own overlaps
            M = np.asarray((P.T @ P).todense(), np.float32)
            RM = Rtr[order] @ M
            G, Kt = RM @ Rtr[order].T, Rte[tests] @ RM.T
        else:  # too wide to hold whole: kept sparse, the likenesses a block of pictures at a time
            E = sp.vstack([sp.csr_matrix(code("tr", order[a: a + 500])) for a in range(0, len(order), 500)]).tocsr()
            G = np.concatenate([np.asarray(E @ code("tr", order[a: a + 500]).T, np.float32).T for a in range(0, len(order), 500)])
            Kt = np.concatenate([np.asarray(E @ code("te", tests[a: a + 500]).T, np.float32).T for a in range(0, len(tests), 500)])
            del E
        ridge(G, Kt, K, y[order], yt, seed, label)
        del G, Kt
