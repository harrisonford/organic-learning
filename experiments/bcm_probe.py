"""A layer of BCM cells in the soft layer's place (queue E 27; docs/VISION.md, 2026-11-25 07:30).

The published soft rule (experiments/softhebb_probe.py) develops filters that serve the learner, but one step of
it is not local (every change divided by the largest change in the layer) and neither stand-in for that step
worked. The biological rule with its own stability is BCM (Bienenstock, Cooper & Munro 1982; the quadratic form
of Intrator & Cooper 1992). Here: that rule, with every number from its authors' simulator (B. Blais's Plasticity,
plasticity/train.pyx and plasticity/utils.py, read 2026-11-25), nothing chosen or tuned by us:

    activity     y = sigma(w . x); sigma(u) = tanh(u) below zero, 50 tanh(u / 50) above (bottom -1, top 50)
    rule         dw = eta y (y - theta) x                       eta = 8e-6   (no derivative of sigma: their default)
    threshold    theta <- theta + (y^2 - theta) / tau           tau = 1000   (the change first, then the threshold)
    birth        w uniform in [-0.05, 0.05]; theta uniform in [0.01, 0.02]
    seen         250,000 presentations (500 x 500), one window each
    between cells   nothing (their default: no lateral connections). Their two lateral forms have no numbers we
                 could take: a matrix to be supplied, and a uniform strength that learns but whose rule, at these
                 defaults, moves it by about eta^2.
What is ours, and said: the input. Theirs is a 13 x 13 round patch of a natural image after a centre-surround
filter, zero mean and unit variance. Ours is the soft layer's: a 5 x 5 window (mirrored borders) of an unlabelled
CIFAR-100 picture, each colour at zero mean and unit variance over the unlabelled pictures. Every cell is shown
the same window at a presentation, as the cells at one place of a layer are. The layer's response, averages,
summary, relay and stream reading are the soft layer's, unchanged (softhebb_probe.respond / coded; bench_stream).
    bcm / bcmr     256 cells, developed / as born

    STAGE=develop SEEDS=0 python experiments/bcm_probe.py   -> out/bench/bcm_weights_s{seed}.npz and the readings
    STAGE=code    SEEDS=0 python experiments/bcm_probe.py   -> out/bench/cifar10_{bcm,bcmr}_s{seed}_{train,test}.npy,
                                                              cortex_{bcm,bcmr}_s{seed}.pkl
    ITERS=2500000 ... as a reading of whether 250,000 presentations were enough (said when used)
"""
import collections, os, pickle, sys, time, types
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import softhebb_probe as P  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
STAGE, ITERS, KINDS = os.environ.get("STAGE", "develop"), int(os.environ.get("ITERS", 250000)), 256
ETA, TAU, BOTTOM, TOP = 8e-6, 1000.0, -1.0, 50.0  # the simulator's defaults (plasticity/utils.py)
TAG = "" if ITERS == 250000 else f"_{ITERS}"


def colours(pictures):
    """Each colour's mean and variance over the unlabelled pictures (what the soft layer's running averages estimate)."""
    s, q, n = np.zeros(3), np.zeros(3), 0
    for a in range(0, len(pictures), 5000):
        X = pictures[a: a + 5000] / np.float64(255)
        s, q, n = s + X.sum(axis=(0, 1, 2)), q + (X * X).sum(axis=(0, 1, 2)), n + X.size // 3
    standard = P.Standard()
    standard.mean, standard.var = (s / n).astype(np.float32), (q / n - (s / n) ** 2).astype(np.float32)
    return standard


def sigma(u):
    return np.where(u < 0, BOTTOM * np.tanh(u / BOTTOM), TOP * np.tanh(u / TOP))


def near_copies(W):
    C = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-30)
    A = np.abs(C @ C.T)
    alike = A[~np.eye(len(W), dtype=bool)]
    left, groups = list(range(len(W))), 0
    while left:  # how many different filters: a filter and all still ungrouped ones alike above 0.9 make one group
        k = left[0]
        left = [j for j in left if A[k, j] <= 0.9]
        groups += 1
    return float(alike.mean()), int((alike > 0.9).sum() // 2), groups


def develop(pictures, standard, rng):
    W, th = rng.uniform(-0.05, 0.05, (KINDS, 75)), rng.uniform(0.01, 0.02, KINDS)
    born, t0, before, kept = W.copy(), time.time(), W.copy(), {}
    marks = sorted({n for n in (1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000) if n <= ITERS} | {ITERS})
    ys, n = collections.deque(maxlen=5000), 0
    while n < ITERS:
        m = min(1000, ITERS - n)
        pick, r, c = rng.integers(len(pictures), size=m), rng.integers(32, size=m), rng.integers(32, size=m)
        Xp = np.pad(standard(pictures[pick] / np.float32(255)), ((0, 0), (2, 2), (2, 2), (0, 0)), mode="reflect")
        V = np.stack([Xp[i, r[i]: r[i] + 5, c[i]: c[i] + 5].transpose(2, 0, 1).ravel() for i in range(m)]).astype(np.float64)
        for x in V:
            y = sigma(W @ x)
            W += (ETA * y * (y - th))[:, None] * x[None, :]
            th += (y * y - th) / TAU
            ys.append(y)
        n += m
        if n in marks:
            Y, r_ = np.array(ys), np.linalg.norm(W, axis=1)
            turn = 1 - np.sum(W * before, axis=1) / (np.linalg.norm(before, axis=1) * r_ + 1e-30)
            alike, pairs, groups = near_copies(W)
            print(f"   after {n:>7} presentations: weight lengths {r_.min():.3f} / {r_.mean():.3f} / {r_.max():.3f} (least, mean, most) | thresholds {th.min():.3g} / {th.mean():.3g} / {th.max():.3g} | "
                  f"activity: mean {Y.mean():.3f}, largest {Y.max():.2f}, a cell above its threshold at {float((Y > th).mean()):.3f} of presentations | "
                  f"turned since the last reading {turn.mean():.4f} (most {turn.max():.4f}) | two cells' weights alike {alike:.3f}, pairs above 0.9: {pairs}, different filters {groups} ({time.time() - t0:.0f}s)", flush=True)
            before, kept[n] = W.copy(), W.astype(np.float32)
    return born.astype(np.float32), W.astype(np.float32), th, kept


if __name__ == "__main__":
    unlabelled = P.load(f"{P.DATA}/cifar-100-python/train")
    standard = colours(unlabelled)
    if os.environ.get("CHECK") == "1":  # the single window against the soft layer's own windows
        X, rng = unlabelled[:3], np.random.default_rng(0)
        Xp = np.pad(standard(X / np.float32(255)), ((0, 0), (2, 2), (2, 2), (0, 0)), mode="reflect")
        theirs = P.windows(standard(X / np.float32(255)))
        worst = max(float(np.abs(Xp[i, r: r + 5, c: c + 5].transpose(2, 0, 1).ravel() - theirs[i, r, c]).max()) for i, r, c in rng.integers(0, (3, 32, 32), size=(200, 3)))
        print(f"a window taken singly against the soft layer's: largest difference {worst:.2g} | colours: mean {standard.mean}, variance {standard.var}")
        sys.exit(0)
    for seed in SEEDS:
        if STAGE == "develop":
            born, W, th, kept = develop(unlabelled, standard, np.random.default_rng(5000 + seed))
            np.savez(f"out/bench/bcm_weights{TAG}_s{seed}.npz", born=born, W=W, th=th, mean=standard.mean, var=standard.var, **{f"at{n}": w for n, w in kept.items()})
            alike, pairs, groups = near_copies(W)
            b = near_copies(born)
            print(f"seed {seed} bcm, {ITERS} presentations: two cells' weights alike {alike:.3f}, pairs above 0.9: {pairs}, different filters {groups} of {KINDS} (as born: {b[0]:.3f}, {b[1]}, {b[2]})", flush=True)
            continue
        Xtr, Xte = P.load(f"{P.DATA}/cifar-10-batches-py/data_batch_1"), P.load(f"{P.DATA}/cifar-10-batches-py/test_batch")[:2000]
        yte = np.load("out/bench/cifar10_test_labels.npy")[:2000]
        d = np.load(f"out/bench/bcm_weights{TAG}_s{seed}.npz")
        settle = np.random.default_rng(seed + 7).choice(len(unlabelled), P.NRELAY, replace=False)
        for name, weights in ((f"bcm{TAG}", d["W"]), ("bcmr", d["born"])):
            t0 = time.time()
            relay = P.Relay(P.SHRINK).settle(P.coded(weights, standard, unlabelled[settle]))
            for part, X in (("train", Xtr), ("test", Xte)):
                S = P.coded(weights, standard, X)
                np.save(f"out/bench/cifar10_{name}_s{seed}_{part}.npy", S.astype(np.float16))
            through = (S - relay.mu) @ relay.Z
            print(f"seed {seed} " + P.read_code(f"{name:6} summary", S[:1500], yte[:1500]) + " | " + P.read_code("through the relay", through[:1500], yte[:1500]) + f" ({time.time() - t0:.0f}s)", flush=True)
            pickle.dump(types.SimpleNamespace(relay=relay, W=weights, standard=(standard.mean, standard.var)), open(f"out/bench/cortex_{name}_s{seed}.pkl", "wb"))
            del relay, through, S
