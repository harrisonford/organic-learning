"""A published layer whose inhibition learns its own strength, in the soft layer's place (queue E 27; docs/VISION.md,
2026-11-25 08:40).

Zylberberg, Murphy & DeWeese (2011), PLoS Computational Biology 7(10): e1002250 ("SAILnet"): leaky integrate-and-fire
cells, Földiák's three local rules. Here the layer of the authors' published code (init.m, SAILnet.m, activities.m,
read 2026-11-25), unchanged, every number theirs:

    cells        for 50 steps  u <- 0.9 u + 0.1 (Q x - W a),  a = the cells that fired on the step before;
                 a cell fires when u reaches its threshold theta and starts again from 0; its answer is its count n
    between cells   dW_im = alpha (n_i n_m - p^2), no self-connection, never below zero (inhibition only)
    into a cell     dQ_ik = beta n_i (x_k - n_i Q_ik)                         (Oja's form of Hebb's rule)
    threshold       dtheta_i = gamma (n_i - p)
    birth        Q Gaussian, each cell's weights at unit length; W = 0; theta = 2
    numbers      p = 0.05 spikes an image; batches of 100 (the batch's average change);
                 25,000 batches at (alpha, beta, gamma) = (1, 0.01, 0.1), the code's;
                 then 10,000 batches at (0.1, 0.001, 0.01), the paper's last rates
What is ours, and said: the input is the soft layer's (a 5 x 5 colour window of an unlabelled CIFAR-100 picture,
each colour at zero mean and unit variance; theirs is a 16 x 16 patch of a whitened image at zero mean and unit
deviation), and the reading: the learned feed-forward weights Q go through the soft layer's own response, averages,
summary, relay and stream (softhebb_probe.respond / coded; bench_stream), as every arm of the dissection did. The
layer's own answer (spike counts under inhibition) is not read here.
    sail / sailr     256 cells, developed / as born

    STAGE=develop SEEDS=0 python experiments/sailnet_probe.py   -> out/bench/sail_weights_s{seed}.npz
    STAGE=code    SEEDS=0 python experiments/sailnet_probe.py   -> out/bench/cifar10_{sail,sailr}_s{seed}_{train,test}.npy,
                                                                  cortex_{sail,sailr}_s{seed}.pkl
"""
import os, pickle, sys, time, types
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import softhebb_probe as P  # noqa: E402
from bcm_probe import colours, near_copies  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
STAGE, KINDS, BATCH, TARGET, STEPS = os.environ.get("STAGE", "develop"), 256, 100, 0.05, 50
PHASES = [(int(os.environ.get("FIRST", 25000)), 1.0, 0.01, 0.1), (int(os.environ.get("LAST", 10000)), 0.1, 0.001, 0.01)]


def activities(X, Q, W, theta):
    """Spike counts of every cell for a batch of inputs (activities.m)."""
    B = X @ Q.T
    u, a, n = np.zeros_like(B), np.zeros_like(B), np.zeros_like(B)
    for _ in range(STEPS):
        u = 0.9 * u + 0.1 * (B - a @ W.T)
        a = (u >= theta).astype(B.dtype)
        n += a
        u[u > theta] = 0.0
    return n


def develop(pictures, standard, rng):
    Q = rng.standard_normal((KINDS, 75))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    W, theta, born, t0, b = np.zeros((KINDS, KINDS)), 2.0 * np.ones(KINDS), Q.copy(), time.time(), 0
    total = sum(p[0] for p in PHASES)
    marks, rate, both = {100, 1000, 5000, 10000, PHASES[0][0], total}, [], []
    for batches, alpha, beta, gamma in PHASES:
        for _ in range(batches):
            pick, r, c = rng.integers(len(pictures), size=BATCH), rng.integers(32, size=BATCH), rng.integers(32, size=BATCH)
            Xp = np.pad(standard(pictures[pick] / np.float32(255)), ((0, 0), (2, 2), (2, 2), (0, 0)), mode="reflect")
            X = np.stack([Xp[i, r[i]: r[i] + 5, c[i]: c[i] + 5].transpose(2, 0, 1).ravel() for i in range(BATCH)]).astype(np.float64)
            n = activities(X, Q, W, theta)
            C = n.T @ n / BATCH
            W += alpha * (C - TARGET ** 2)
            np.fill_diagonal(W, 0.0)
            np.maximum(W, 0.0, out=W)
            Q += beta * (n.T @ X) / BATCH - beta * (n * n).sum(axis=0)[:, None] * Q / BATCH
            theta += gamma * (n.mean(axis=0) - TARGET)
            rate.append(n.mean(axis=0))
            both.append((C - np.diag(np.diag(C)))[~np.eye(KINDS, dtype=bool)].mean())
            b += 1
            if b in marks:
                R, lengths = np.array(rate[-200:]).mean(axis=0), np.linalg.norm(Q, axis=1)
                alike, pairs, groups = near_copies(Q)
                print(f"   after {b:>6} batches (rates {alpha:g}, {beta:g}, {gamma:g}): spikes an image, per cell {R.min():.3f} / {R.mean():.3f} / {R.max():.3f} (least, mean, most; the target {TARGET}) | "
                      f"two cells firing to one image {float(np.mean(both[-200:])):.5f} (the target {TARGET ** 2:.5f}) | thresholds {theta.min():.2f} / {theta.mean():.2f} / {theta.max():.2f} | "
                      f"inhibition {W.mean():.3f} on average, most {W.max():.2f}, pairs with any {float((W > 0).mean()):.2f} | weight lengths {lengths.min():.2f} / {lengths.mean():.2f} / {lengths.max():.2f} | "
                      f"two cells' weights alike {alike:.3f}, pairs above 0.9: {pairs}, different filters {groups} ({time.time() - t0:.0f}s)", flush=True)
    return born.astype(np.float32), Q.astype(np.float32), W.astype(np.float32), theta


if __name__ == "__main__":
    unlabelled = P.load(f"{P.DATA}/cifar-100-python/train")
    standard = colours(unlabelled)
    for seed in SEEDS:
        if STAGE == "develop":
            born, Q, W, theta = develop(unlabelled, standard, np.random.default_rng(6000 + seed))
            np.savez(f"out/bench/sail_weights_s{seed}.npz", born=born, W=Q, lateral=W, theta=theta, mean=standard.mean, var=standard.var)
            alike, pairs, groups = near_copies(Q)
            print(f"seed {seed} sail: two cells' weights alike {alike:.3f}, pairs above 0.9: {pairs}, different filters {groups} of {KINDS}", flush=True)
            continue
        Xtr, Xte = P.load(f"{P.DATA}/cifar-10-batches-py/data_batch_1"), P.load(f"{P.DATA}/cifar-10-batches-py/test_batch")[:2000]
        yte = np.load("out/bench/cifar10_test_labels.npy")[:2000]
        d = np.load(f"out/bench/sail_weights_s{seed}.npz")
        settle = np.random.default_rng(seed + 7).choice(len(unlabelled), P.NRELAY, replace=False)
        for name, weights in (("sail", d["W"]), ("sailr", d["born"])):
            t0 = time.time()
            relay = P.Relay(P.SHRINK).settle(P.coded(weights, standard, unlabelled[settle]))
            for part, X in (("train", Xtr), ("test", Xte)):
                S = P.coded(weights, standard, X)
                np.save(f"out/bench/cifar10_{name}_s{seed}_{part}.npy", S.astype(np.float16))
            through = (S - relay.mu) @ relay.Z
            print(f"seed {seed} " + P.read_code(f"{name:6} summary", S[:1500], yte[:1500]) + " | " + P.read_code("through the relay", through[:1500], yte[:1500]) + f" ({time.time() - t0:.0f}s)", flush=True)
            pickle.dump(types.SimpleNamespace(relay=relay, W=weights, standard=(standard.mean, standard.var)), open(f"out/bench/cortex_{name}_s{seed}.pkl", "wb"))
            del relay, through, S
