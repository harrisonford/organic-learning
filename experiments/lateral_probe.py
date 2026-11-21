"""Kinds that keep each other apart: lateral inhibition that is itself learned (docs/VISION.md, 2026-11-20/21; a probe).

`cortex_dev_probe.py` found that winner-take-all Hebbian development with a conscience makes the kinds selective and
makes 658 pairs of them near copies: nothing keeps two kinds from settling on the same feature. In cortex, neurons of
like tuning inhibit each other and that inhibition is learned (anti-Hebbian lateral connections, Földiák 1990).

A first form of my own (2026-11-20; in git history) failed: its inhibition grew without end. This is a published
network instead, unchanged in its rules: the Hebbian/anti-Hebbian network that Pehlevan & Chklovskii derive from
online non-negative similarity matching (arXiv:1503.00680, their equations 7 and 8, read in the paper):
    activity     y_i = max(W_i x - sum_{j != i} M_ij y_j, 0), each kind in turn until nothing changes
    feedforward  W_ij += y_i (x_j - W_ij y_i) / Y_i          (Hebbian, of Oja's form)
    lateral      M_ij += y_i (y_j - M_ij y_i) / Y_i          (anti-Hebbian: it enters the activity with a minus)
    how plastic  Y_i += y_i^2: a kind that has been active much changes little (no rate set by hand)
Each change depends only on the two neurons at the ends of the synapse. Where this probe departs from the paper,
and says so: the kinds are all there from the start, born random as the layer's are (the paper recruits them one by
one under a regulariser); the windows of one picture settle together before their changes are made one by one; and
Y starts at Y0 (the paper starts its own image run at 1,000 for inputs of another size).

Read first, before any accuracy: are the copies gone, is the selectivity kept (the table of cortex_dev_probe.py)?
The developed network is kept (out/bench/lateral_s<seed>.npz). Then, with CODES=1, the code made of the *settled
activities* is kept for a short stream as out/bench/cifar10_settled_s<seed>_*.npy (experiments/bench_stream.py with
VARIANT=settled N=10000).
    SEEDS=0 STEPS=16000 Y0=10 CODES=0 python experiments/lateral_probe.py
"""
import os, pickle, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA = os.path.expanduser("~/datasets")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
STEPS, Y0, CODES = int(os.environ.get("STEPS", 16000)), float(os.environ.get("Y0", 10)), os.environ.get("CODES") == "1"
GRID, SHRINK = 4, 0.0005


def load(path, key=b"labels"):
    d = pickle.load(open(path, "rb"), encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[key])


big = lambda x: np.asarray(Image.fromarray(x).resize((64, 64), Image.BILINEAR), np.float32) / 255  # noqa: E731
unlabelled, _ = load(f"{DATA}/cifar-100-python/train", b"fine_labels")
held, _ = load(f"{DATA}/cifar-100-python/test", b"fine_labels")


def settle(W, M, V, sweeps=30):
    """The kinds' activity for these windows: each kind in turn, its drive less what the others' activity takes
    from it, until nothing changes (the paper's coordinate descent)."""
    d = np.ascontiguousarray((V @ W.T).T)  # (kinds, windows)
    y = np.maximum(d, 0)
    for _ in range(sweeps):
        moved = 0.0
        for i in range(len(W)):
            new = np.maximum(d[i] - M[i] @ y, 0)
            moved = max(moved, float(np.abs(new - y[i]).max()))
            y[i] = new
        if moved < 1e-3:
            break
    return np.ascontiguousarray(y.T)


def windows(layer, img, rng, n):
    """n strong windows of a picture as the layer gets them (decorrelated, unit length): what development learns from."""
    _, e, seen = layer._drive(img)
    v, e = seen[0][0].reshape(-1, seen[0][0].shape[2]), e.ravel()
    strong = np.nonzero(e > 0.3 * e.max())[0]
    return v[rng.choice(strong, min(n, len(strong)), replace=False)]


def table(label, layer, W, seed, M=None):
    rng = np.random.default_rng(seed)
    V = np.concatenate([windows(layer, big(x), rng, 300) for x in held[:200]])
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    U = V @ Wn.T
    z = (U - U.mean(axis=0)) / (U.std(axis=0) + 1e-8)
    wins = np.bincount(np.argmax(U, axis=1), minlength=len(W)) / len(U)
    use = float(-(wins[wins > 0] * np.log(wins[wins > 0])).sum() / np.log(len(wins)))
    pairs = np.triu_indices(len(W), 1)
    cos = np.abs(Wn @ Wn.T)[pairs]
    R = np.maximum(U - U.mean(axis=1, keepdims=True), 0)  # the response the layer sees with
    rc = np.corrcoef(R.T)[pairs]
    own = ""
    if M is not None:
        Y = settle(W, M, V[:3000])
        yc = np.corrcoef((Y + 1e-9 * rng.random(Y.shape)).T)[pairs]
        own = (f" | settled: a kind is active {np.mean(Y > 0):.3f} of the time (never: {int((Y.max(axis=0) == 0).sum())} kinds), activities alike top 1% {np.percentile(yc, 99):.2f}; "
               f"inhibition mean {M.mean():.3f}, max {M.max():.2f}; length of a kind's synapses {np.linalg.norm(W, axis=1).mean():.2f}")
    print(f"seed {seed} {label:10} | kurtosis {np.mean((z ** 4).mean(axis=0) - 3):.2f} | use {use:.3f} | filters alike {cos.mean():.3f}, top 1% {np.percentile(cos, 99):.2f}, pairs above 0.8: {int((cos > 0.8).sum())} | "
          f"responses alike top 1% {np.percentile(rc, 99):.2f}{own}", flush=True)


for seed in SEEDS:
    t0, rng = time.time(), np.random.default_rng(seed + 100)
    cortex = pickle.load(open(f"out/bench/cortex_random_s{seed}.pkl", "rb"))  # the windows' decorrelation settled, the kinds as born
    layer = cortex.layers[0]
    table("born", layer, layer.W[0], seed)
    table("developed", layer, pickle.load(open(f"out/bench/cortex_developed_s{seed}.pkl", "rb")).layers[0].W[0], seed)
    kept = f"out/bench/lateral_s{seed}.npz"
    if os.path.exists(kept) and os.environ.get("AGAIN") != "1":  # the network as it developed last time
        W, M = (np.load(kept)[k] for k in ("W", "M"))
    else:
        W = layer.W[0].astype(np.float64).copy()
        K = len(W)
        M, Y = np.zeros((K, K)), np.full(K, Y0)
        for n in range(STEPS):
            V = windows(layer, big(unlabelled[rng.integers(len(unlabelled))]), rng, 32).astype(np.float64)
            ys = settle(W, M, V)
            for x, y in zip(V, ys):
                Y += y * y
                g = y / Y
                W += g[:, None] * (x[None, :] - W * y[:, None])
                M += g[:, None] * (y[None, :] - M * y[:, None])
                np.fill_diagonal(M, 0)
            if (n + 1) % 2000 == 0:
                print(f"   seed {seed} after {n + 1} pictures ({time.time() - t0:.0f}s): active {np.mean(ys > 0):.3f}, inhibition mean {M.mean():.3f} max {M.max():.2f}, synapses' length {np.linalg.norm(W, axis=1).mean():.2f}, Y {Y.min():.0f}..{Y.max():.0f}", flush=True)
        np.savez(kept, W=W, M=M)
    K = len(W)
    table("lateral", layer, W, seed, M)
    if CODES:
        # The code made of the settled activities: what the circuit says, not what its filters would say alone.
        # Every window of the view settles under the learned inhibition; then, as the cortex does, the average
        # activity of each kind in each of 16 regions, at unit length. Only the pictures a short stream needs:
        # the first N_CODE of the seed's stream (experiments/bench_stream.py) and its 2,000 test pictures; a relay
        # settled on RELAY_N unlabelled pictures is kept beside it, to see what is left for a relay to do.
        W32, M32 = W.astype(np.float32), M.astype(np.float32)

        def summary(img):
            v = layer._drive(img)[2][0][0]
            h, w = v.shape[:2]
            m = settle(W32, M32, v.reshape(-1, v.shape[2]), sweeps=12).reshape(h, w, K)
            gh, gw = h // GRID, w // GRID
            q = m[: gh * GRID, : gw * GRID].reshape(GRID, gh, GRID, gw, K).mean(axis=(1, 3)).ravel()
            return (q / (np.linalg.norm(q) + 1e-8)).astype(np.float32)

        from organic.cortex2 import Relay
        n_code, relay_n = int(os.environ.get("N_CODE", 10000)), int(os.environ.get("RELAY_N", 3000))
        relay = Relay(SHRINK).settle([summary(big(unlabelled[i])) for i in rng.choice(len(unlabelled), relay_n, replace=False)])
        print(f"   seed {seed}: relay settled on {relay_n} pictures ({time.time() - t0:.0f}s)", flush=True)
        train = [load(f"{DATA}/cifar-10-batches-py/data_batch_{i}") for i in range(1, 6)]
        Xtr, Xte = np.concatenate([a for a, _ in train]), load(f"{DATA}/cifar-10-batches-py/test_batch")[0]
        for name, X, which in (("train", Xtr, np.random.default_rng(seed).permutation(len(Xtr))[:n_code]), ("test", Xte, np.random.default_rng(99).permutation(len(Xte))[:2000])):
            out = np.zeros((len(X), 16 * K), np.float16)
            for k, i in enumerate(which):
                out[i] = summary(big(X[i]))
                if (k + 1) % 2000 == 0:
                    print(f"   seed {seed}: {name} {k + 1} coded ({time.time() - t0:.0f}s)", flush=True)
            np.save(f"out/bench/cifar10_settled_s{seed}_{name}.npy", out)
        cortex.relay = relay
        pickle.dump(cortex, open(f"out/bench/cortex_settled_s{seed}.pkl", "wb"))
        print(f"seed {seed} settled: codes of the first {n_code} stream pictures and the test pictures kept ({time.time() - t0:.0f}s)", flush=True)
