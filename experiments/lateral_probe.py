"""Kinds that keep each other apart: lateral inhibition that is itself learned (docs/VISION.md, 2026-11-20; a probe).

`cortex_dev_probe.py` found that winner-take-all Hebbian development with a conscience makes the kinds selective and
makes about 330 pairs of them near copies: nothing keeps two kinds from settling on the same feature. In cortex,
neurons of like tuning inhibit each other and that inhibition is learned (anti-Hebbian lateral connections, Földiák
1990; with local rules it gives V1-like fields, Zylberberg, Murphy & DeWeese 2011). Here the kinds of one layer
develop on the same decorrelated windows with three local rules and no winner chosen from outside:
    feedforward  a kind that is active moves towards the window, in proportion to its activity (Hebbian; its
                 synapses keep unit length)
    lateral      two kinds whose activities go together come to inhibit each other; inhibition between kinds whose
                 activities do not go together fades (anti-Hebbian)
    threshold    a kind that is active too often raises its threshold, one too seldom lowers it (homeostasis)
A window's activity settles under the inhibition before anything is learned. Set by hand: how often a kind is to be
active (ACTIVE), and the three rates.

Read first, before any accuracy: are the copies gone, is the selectivity kept (the table of cortex_dev_probe.py)?
Then, with CODES=1, the kinds are put in the benchmark cortex (seen with the layer's usual response) and CIFAR-10 is
coded as out/bench/cifar10_lateral_s<seed>_*.npy for experiments/bench_stream.py (VARIANT=lateral).
    SEEDS=0 STEPS=16000 ACTIVE=0.1 CODES=0 python experiments/lateral_probe.py
"""
import os, pickle, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA = os.path.expanduser("~/datasets")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
STEPS, ACTIVE, CODES = int(os.environ.get("STEPS", 16000)), float(os.environ.get("ACTIVE", 0.1)), os.environ.get("CODES") == "1"
HEBB, ANTI, THRESH = float(os.environ.get("HEBB", 0.02)), float(os.environ.get("ANTI", 0.01)), float(os.environ.get("THRESH", 0.01))
GRID, SHRINK = 4, 0.0005


def load(path, key=b"labels"):
    d = pickle.load(open(path, "rb"), encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[key])


big = lambda x: np.asarray(Image.fromarray(x).resize((64, 64), Image.BILINEAR), np.float32) / 255  # noqa: E731
unlabelled, _ = load(f"{DATA}/cifar-100-python/train", b"fine_labels")
held, _ = load(f"{DATA}/cifar-100-python/test", b"fine_labels")


def settle(W, L, t, V, steps=8):
    """The kinds' activity for these windows, settled under their lateral inhibition."""
    d = V @ W.T - t
    y = np.maximum(d, 0)
    for _ in range(steps):
        y = 0.5 * y + 0.5 * np.maximum(d - y @ L, 0)
    return y


def windows(layer, img, rng, n):
    """n strong windows of a picture as the layer gets them (decorrelated, unit length): what development learns from."""
    _, e, seen = layer._drive(img)
    v, e = seen[0][0].reshape(-1, seen[0][0].shape[2]), e.ravel()
    strong = np.nonzero(e > 0.3 * e.max())[0]
    return v[rng.choice(strong, min(n, len(strong)), replace=False)]


def table(label, layer, W, seed, L=None, t=None):
    rng = np.random.default_rng(seed)
    V = np.concatenate([windows(layer, big(x), rng, 300) for x in held[:200]])
    U = V @ W.T
    z = (U - U.mean(axis=0)) / (U.std(axis=0) + 1e-8)
    wins = np.bincount(np.argmax(U, axis=1), minlength=len(W)) / len(U)
    use = float(-(wins[wins > 0] * np.log(wins[wins > 0])).sum() / np.log(len(wins)))
    pairs = np.triu_indices(len(W), 1)
    cos = np.abs(W @ W.T)[pairs]
    R = np.maximum(U - U.mean(axis=1, keepdims=True), 0)  # the response the layer sees with
    rc = np.corrcoef(R.T)[pairs]
    own = ""
    if L is not None:
        Y = settle(W, L, t, V)
        yc = np.corrcoef((Y + 1e-9 * rng.random(Y.shape)).T)[pairs]
        own = f" | settled: active {np.mean(Y > 0):.3f} of the time, never {int((Y.max(axis=0) == 0).sum())}, activities alike top 1% {np.percentile(yc, 99):.2f}; inhibition mean {L.mean():.3f}, max {L.max():.2f}"
    print(f"seed {seed} {label:10} | kurtosis {np.mean((z ** 4).mean(axis=0) - 3):.2f} | use {use:.3f} | filters alike {cos.mean():.3f}, top 1% {np.percentile(cos, 99):.2f}, pairs above 0.8: {int((cos > 0.8).sum())} | "
          f"responses alike top 1% {np.percentile(rc, 99):.2f}{own}", flush=True)


for seed in SEEDS:
    t0, rng = time.time(), np.random.default_rng(seed + 100)
    cortex = pickle.load(open(f"out/bench/cortex_random_s{seed}.pkl", "rb"))  # the windows' decorrelation settled, the kinds as born
    layer = cortex.layers[0]
    table("born", layer, layer.W[0], seed)
    table("developed", layer, pickle.load(open(f"out/bench/cortex_developed_s{seed}.pkl", "rb")).layers[0].W[0], seed)
    W = layer.W[0].copy()
    K = len(W)
    L, t = np.zeros((K, K), np.float32), np.zeros(K, np.float32)
    mean, scale = np.zeros(K, np.float32), np.full(K, 0.1, np.float32)  # each kind's own running activity and its size
    for n in range(STEPS):
        V = windows(layer, big(unlabelled[rng.integers(len(unlabelled))]), rng, 32)
        y = settle(W, L, t, V)
        scale += 0.01 * (np.sqrt((y * y).mean(axis=0)) - scale)
        a = np.minimum(y / (scale + 1e-3), 4.0)  # a kind's activity in units of its own usual activity
        mean += 0.01 * (a.mean(axis=0) - mean)
        rate = HEBB * (1 - n / STEPS) + HEBB / 10
        W += rate * ((a.T @ V) / len(V) - a.mean(axis=0)[:, None] * W)  # towards the windows it was active for
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        L += ANTI * ((a.T @ a) / len(V) - np.outer(mean, mean))  # together more than by chance: more inhibition
        np.maximum(L, 0, out=L)
        np.fill_diagonal(L, 0)
        t += THRESH * ((y > 0).mean(axis=0) - ACTIVE)
        if (n + 1) % 4000 == 0:
            print(f"   seed {seed} after {n + 1} pictures ({time.time() - t0:.0f}s): active {np.mean(y > 0):.3f}, inhibition mean {L.mean():.3f} max {L.max():.2f}, thresholds {t.min():.2f}..{t.max():.2f}", flush=True)
    table("lateral", layer, W, seed, L, t)
    if CODES:
        layer.W[0] = W.astype(np.float32)
        layer.lateral, layer.threshold = L, t  # kept for a later look; the layer's response does not use them
        pics = [big(unlabelled[i]) for i in rng.choice(len(unlabelled), 6000, replace=False)]
        cortex.relay = None
        cortex.settle_relay(pics, GRID, SHRINK)
        relay, cortex.relay = cortex.relay, None
        train = [load(f"{DATA}/cifar-10-batches-py/data_batch_{i}") for i in range(1, 6)]
        for name, X in (("train", np.concatenate([a for a, _ in train])), ("test", load(f"{DATA}/cifar-10-batches-py/test_batch")[0])):
            out = np.zeros((len(X), 16 * K), np.float16)
            for i, x in enumerate(X):
                out[i] = cortex.whole(big(x), GRID)
            np.save(f"out/bench/cifar10_lateral_s{seed}_{name}.npy", out)
        cortex.relay = relay
        pickle.dump(cortex, open(f"out/bench/cortex_lateral_s{seed}.pkl", "wb"))
        print(f"seed {seed} lateral: cortex and codes kept ({time.time() - t0:.0f}s)", flush=True)
