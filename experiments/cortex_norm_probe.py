"""The cortex's response by divisive normalisation (queue E 24, item 1; docs/RESEARCH_CORTEX.md; VISION.md 2026-11-23).

Today a kind's response at a place is its drive above the average drive of all kinds there, rectified: subtractive.
The form found throughout cortex is divisive (Carandini & Heeger 2012): a cell's rectified drive, raised to a power
of about two, divided by the pooled activity of the cells around it plus a constant,
    R_k = D_k^2 / (sigma^2 + mean over the pool of D_j^2),
which gives a response that saturates with contrast, suppression by other orientations at the same place and by the
surround, and a sparse code. The same developed filters and the same decorrelated windows; only the response changes:
    now        the present response (the control, coded again here on the same pictures)
    divisive   D_k = the kind's rectified drive times the window's contrast (a simple cell); the pool is all kinds at
               the place; sigma^2 is the median pooled energy over the windows of unlabelled pictures (the gain set
               by the prevailing contrast: Ohzawa, Sclar & Freeman 1985). No constant is searched.
    surround   the same, the pool also over the places around (a square twice the window across: Cavanaugh et al. 2002)
Written for each: the codes of 10,000 CIFAR-10 training pictures and 2,000 test pictures with a relay settled on
6,000 unlabelled pictures (out/bench/cifar10_<name>_s<seed>_*.npy, cortex_<name>_s<seed>.pkl), for
experiments/bench_stream.py (VARIANT=<name> N=10000). Printed: what the response does to the code itself, at single
places before any averaging: how sparse (Treves-Rolls, over the kinds at a place and over places for a kind), the
share of kinds above half the strongest, and how the response grows with contrast.
    SEEDS=0 python experiments/cortex_norm_probe.py
"""
import copy, os, pickle, sys, time, types
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA = os.path.expanduser("~/datasets")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
VARIANTS = os.environ.get("VARIANTS", "now,divisive,surround").split(",")
NTRAIN, NTEST, NRELAY = int(os.environ.get("NTRAIN", 10000)), int(os.environ.get("NTEST", 2000)), int(os.environ.get("NRELAY", 6000))
GRID, SHRINK = 4, 0.0005
NAME = {"now": "norm0", "divisive": "divnorm", "surround": "divnorm_s"}


def load(path):
    d = pickle.load(open(path, "rb"), encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


big = lambda x: np.asarray(Image.fromarray(x).resize((64, 64), Image.BILINEAR), np.float32) / 255  # noqa: E731
unlabelled = load(f"{DATA}/cifar-100-python/train")
Xtr, Xte = load(f"{DATA}/cifar-10-batches-py/data_batch_1")[:NTRAIN], load(f"{DATA}/cifar-10-batches-py/test_batch")[:NTEST]
assert NTRAIN <= 10000  # the first batch only: the stream's labels are read by position


def box(E, reach):
    """The mean of a map over a square of 2 * reach + 1 places around each place (at the edge: over what there is)."""
    P = np.pad(E, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = E.shape
    r0, r1 = np.maximum(np.arange(h) - reach, 0), np.minimum(np.arange(h) + reach + 1, h)
    c0, c1 = np.maximum(np.arange(w) - reach, 0), np.minimum(np.arange(w) + reach + 1, w)
    total = P[r1][:, c1] - P[r0][:, c1] - P[r1][:, c0] + P[r0][:, c0]
    return total / ((r1 - r0)[:, None] * (c1 - c0)[None, :])


def pooled_energy(layer, x, surround):
    u, e, _ = layer._drive(x)
    D = np.maximum(u, 0) * e[..., None]
    E = (D * D).mean(axis=2)
    return D, box(E, layer.window) if surround else E


def divisive(surround, sigma2):
    def respond(self, x, pooled=True):
        D, E = pooled_energy(self, x, surround)
        r = D * D / (sigma2 + E[..., None])
        if not pooled or self.pool == 1:
            return r.astype(np.float32)
        p = self.pool
        h, w = (r.shape[0] // p) * p, (r.shape[1] // p) * p
        return r[:h, :w].reshape(h // p, p, w // p, p, -1).mean(axis=(1, 3)).astype(np.float32)
    return respond


def sparse(a, axis):
    """Treves-Rolls sparseness along an axis, as 1 - (mean)^2 / mean of squares: 0 all alike, towards 1 a few active."""
    return 1 - a.mean(axis=axis) ** 2 / ((a * a).mean(axis=axis) + 1e-12)


for seed in SEEDS:
    base = pickle.load(open(f"out/bench/cortex_developed_s{seed}.pkl", "rb"))
    rng = np.random.default_rng(seed + 7)
    settle = [big(unlabelled[i]) for i in rng.choice(len(unlabelled), NRELAY, replace=False)]
    for variant in VARIANTS:
        t0 = time.time()
        cortex = copy.deepcopy(base)
        cortex.relay = None
        layer, note = cortex.layers[0], ""
        if variant != "now":
            E = np.concatenate([pooled_energy(layer, im, variant == "surround")[1].ravel() for im in settle[:300]])
            sigma2 = float(np.median(E))
            layer.respond = types.MethodType(divisive(variant == "surround", sigma2), layer)
            note = f" (sigma^2 {sigma2:.4g}: the median pooled energy)"
        # the code itself, at single places of the map, on 200 unlabelled pictures
        maps = np.stack([layer.respond(im, pooled=False)[::6, ::6].reshape(-1, layer.kinds) for im in settle[:200]])  # (pictures, places, kinds)
        flat = maps.reshape(-1, layer.kinds)
        live = flat.max(axis=1) > 0
        top = flat[live].max(axis=1, keepdims=True)
        im = settle[0]
        grey = im.mean()
        grow = [float(layer.respond(grey + c * (im - grey), pooled=False).mean()) for c in (0.25, 0.5, 1.0)]
        print(f"seed {seed} {variant:9}{note}: sparse over kinds at a place {sparse(flat[live], 1).mean():.3f} | kinds above half the strongest {np.mean(flat[live] > 0.5 * top):.3f}, "
              f"above a tenth {np.mean(flat[live] > 0.1 * top):.3f} | sparse over places for a kind {sparse(flat, 0).mean():.3f} | "
              f"response at a quarter and half the contrast, of that at full: {grow[0] / grow[2]:.2f}, {grow[1] / grow[2]:.2f}", flush=True)
        cortex.settle_relay(settle, GRID, SHRINK)
        relay, cortex.relay = cortex.relay, None
        for name, X in (("train", Xtr), ("test", Xte)):
            out = np.zeros((len(X), GRID * GRID * layer.kinds), np.float16)
            for i, x in enumerate(X):
                out[i] = cortex.whole(big(x), GRID)
            np.save(f"out/bench/cifar10_{NAME[variant]}_s{seed}_{name}.npy", out)
        keep = copy.deepcopy(base)  # the stream reads only the relay from this file
        keep.relay = relay
        pickle.dump(keep, open(f"out/bench/cortex_{NAME[variant]}_s{seed}.pkl", "wb"))
        print(f"seed {seed} {variant}: codes and relay kept as {NAME[variant]} ({time.time() - t0:.0f}s)", flush=True)
