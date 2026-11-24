"""The sparse layer as the episodic store's key (queue E 29f, second step; docs/VISION.md, 2026-11-24 evening).

The store (organic/episodes.py) compares what is seen now with every stored moment and lets a moment answer by
exp(steep x (likeness - 1)), with steep = 12 set by hand: the steepness is what makes it a memory of moments and
not of kinds. The dentate gyrus is said to do that work by its wiring: many cells, few on, so that two inputs
that are alike share few active cells. If so, the store keyed by the sparse layer's code needs no exponential.

Read here, with no learner: 1,000 test pictures are stored. Then asked with
    the same pictures blurred by a pixel   (a stored moment met again, not quite the same: it should answer)
    1,000 pictures never stored            (nothing should answer)
For each key: how often the blurred picture's strongest answer is its own stored moment; how strongly that
moment answers; how strongly the strongest stored moment answers a picture never stored; and how well the two
are told apart by the answer's strength alone (the share of pairs, one of each, that it orders rightly).
    SEEDS=0,1,2 python experiments/expand_store_probe.py
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scipy.sparse as sp

FAN = 7


def wiring(K, D, rng):  # as experiments/expand_probe.py (not imported: that file runs when it is loaded)
    cols = np.stack([rng.choice(D, FAN, replace=False) for _ in range(K)]).ravel()
    return sp.csr_matrix((np.ones(K * FAN, np.float32), (np.repeat(np.arange(K), FAN), cols)), shape=(K, D))


def expand(P, X, active):
    H = np.asarray((P @ X.T).T, np.float32)
    k = max(1, int(round(active * H.shape[1])))
    return np.maximum(H - np.partition(H, -k - 1, axis=1)[:, -k - 1][:, None], 0)

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
STEEP = 12.0  # the store's, as the lives have it


def unit(X):
    X = X - X.mean(axis=0)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def told_apart(a, b):
    """The share of pairs (one of a, one of b) in which a's is the larger."""
    order = np.argsort(np.concatenate([a, b]), kind="stable")
    rank = np.empty(len(order))
    rank[order] = np.arange(1, len(order) + 1)
    return float((rank[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def read(label, stored, again, new, answer, seed):
    A, B = answer(again @ stored.T), answer(new @ stored.T)
    own = A[np.arange(len(A)), np.arange(len(A))]
    print(f"seed {seed} {label:28} blurred picture's strongest answer is its own moment {np.mean(A.argmax(axis=1) == np.arange(len(A))):.3f} | "
          f"its own moment answers {own.mean():.3f} | strongest answer to a picture never stored {B.max(axis=1).mean():.3f} | "
          f"told apart by strength {told_apart(own, B.max(axis=1)):.3f}", flush=True)


for seed in SEEDS:
    relay = pickle.load(open(f"out/bench/cortex_norm0_s{seed}.pkl", "rb")).relay
    through = lambda X: (lambda W: W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8))((np.asarray(X, np.float32) - relay.mu) @ relay.Z)
    sharp = through(np.load(f"out/bench/cifar10_norm0_s{seed}_test.npy"))
    blurred = through(np.load(f"out/bench/cifar10_norm0b_s{seed}_test.npy"))
    pick = np.random.default_rng(7).permutation(len(sharp))
    kept, never = pick[:1000], pick[1000:2000]
    both = unit(np.concatenate([sharp, blurred]))  # each number's deviation from its average, as the word cells get it
    s, b = both[: len(sharp)], both[len(sharp):]
    print(f"seed {seed} a picture and itself blurred alike {float((s[kept] * b[kept]).sum(axis=1).mean()):.3f} on the relay's code", flush=True)
    steep, plain = (lambda L: np.exp(STEEP * (np.minimum(L, 1.0) - 1.0))), (lambda L: np.maximum(L, 0.0))
    read("relay, exp(12 (like - 1))", s[kept], b[kept], s[never], steep, seed)
    read("relay, the likeness itself", s[kept], b[kept], s[never], plain, seed)
    for ratio in [int(r) for r in os.environ.get("RATIOS", "4").split(",")]:
        P = wiring(ratio * sharp.shape[1], sharp.shape[1], np.random.default_rng(1000 + seed))
        E = unit(np.concatenate([expand(P, X[i: i + 500], 0.05) for X in (sharp, blurred) for i in range(0, len(X), 500)]))
        es, eb = E[: len(sharp)], E[len(sharp):]
        print(f"seed {seed} a picture and itself blurred alike {float((es[kept] * eb[kept]).sum(axis=1).mean()):.3f} on {ratio}x sparse cells", flush=True)
        read(f"x{ratio} sparse, likeness itself", es[kept], eb[kept], es[never], plain, seed)
        read(f"x{ratio} sparse, exp(12 (like - 1))", es[kept], eb[kept], es[never], steep, seed)
        del E, es, eb
