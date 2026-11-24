"""A second stage of cortex that learns from what stays the same while the eye drifts (queue E 29a; docs/RESEARCH_ARCH.md).

A cell that goes on answering while the picture slides a little under the eye is a complex cell. The established
account of how such cells come about is temporal: within a fixation the eye drifts (the image moves across about a
receptive field: Rucci & Poletti 2015), what is seen a moment later is almost always the same thing a little
shifted, and a cell whose synapses still carry a trace of having just fired learns the shifted view too (Földiák
1991; Wallis & Rolls 1997; the "swap" experiments in monkey IT: Li & DiCarlo 2008; a modern local rule with the
same term: Halvagal & Zenke 2023). Our learner saw stills, and its only tolerance to position is a fixed average.
Here the first stage stays as it is (developed filters, pooled 2 x 2). A second stage (256 kinds in 16 groups, each
listening to 32 kinds of the first over 3 x 3 places: the second layer of 2026-11-17, organic/cortex2.py) develops
on drifting views, by the first stage's own rule (winner-take-all Hebbian with a conscience), in two arms that see
exactly the same views:
    plain   only the winner of each view learns it (no memory of the view before)
    trace   a cell that won a view before at this place still carries a trace (0.7 a view, the persistence the
            eye's own memory of what it saw has in the learner) and learns the present view in proportion to it
Printed: how alike a cell's answers are to a picture and to the same picture two pixels aside (first stage, and each
arm's second stage), how sparse. Kept for experiments/bench_stream.py (VARIANT=l2plain / l2trace, N=10000): the
second stage's summary of 10,000 CIFAR-10 pictures and 2,000 test pictures, with a relay settled on 6,000 others.
    SEEDS=0 python experiments/cortex_trace_probe.py
"""
import copy, os, pickle, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Layer2, Relay  # noqa: E402

DATA = os.path.expanduser("~/datasets")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
ARMS = os.environ.get("ARMS", "plain,trace").split(",")
PICTURES, VIEWS, KEEP = int(os.environ.get("PICTURES", 4000)), int(os.environ.get("VIEWS", 4)), 0.7
NTRAIN, NTEST, NRELAY = int(os.environ.get("NTRAIN", 10000)), int(os.environ.get("NTEST", 2000)), int(os.environ.get("NRELAY", 6000))
GRID, SHRINK = 4, 0.0005


def load(path):
    d = pickle.load(open(path, "rb"), encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


big = lambda x: np.asarray(Image.fromarray(x).resize((64, 64), Image.BILINEAR), np.float32) / 255  # noqa: E731
unlabelled = load(f"{DATA}/cifar-100-python/train")
Xtr, Xte = load(f"{DATA}/cifar-10-batches-py/data_batch_1")[:NTRAIN], load(f"{DATA}/cifar-10-batches-py/test_batch")[:NTEST]


def shifted(img, dy, dx):
    """The view with the eye moved by (dy, dx) pixels: the picture slides, its edge repeated."""
    p = np.pad(img, ((4, 4), (4, 4), (0, 0)), mode="edge")
    return p[4 + dy : 4 + dy + img.shape[0], 4 + dx : 4 + dx + img.shape[1]]


def drift(img, rng, views):
    """A fixation: the eye drifts a pixel at a time (a random walk, at most three pixels from where it began)."""
    out, dy, dx = [img], 0, 0
    for _ in range(views - 1):
        dy, dx = int(np.clip(dy + rng.integers(-1, 2), -3, 3)), int(np.clip(dx + rng.integers(-1, 2), -3, 3))
        out.append(shifted(img, dy, dx))
    return out


def learn(layer, xs, rng, trace, samples=32, rate=0.02):
    """The second stage learns from successive views (xs: the first stage's maps of them). At places sampled once
    for the fixation, the winner of each view moves towards it (as organic/cortex2.py's rule); with `trace`, cells
    that won an earlier view at the place move towards the present one too, as far as their trace has not faded."""
    seen = [layer._drive(x)[2] for x in xs]
    for g in range(layer.groups):
        e = seen[0][g][1].ravel()
        strong = np.nonzero(e > 0.3 * e.max())[0] if e.max() > 1e-6 else []
        if not len(strong):
            continue
        W, wins = layer.W[g], layer.wins[g]
        V = [s[g][0].reshape(-1, s[g][0].shape[2]) for s in seen]
        for i in rng.choice(strong, min(samples, len(strong)), replace=False):
            tr = np.zeros(len(W), np.float32)
            for v in V:
                c = int(np.argmax(W @ v[i] - 0.3 * (wins / (wins.mean() + 1e-8) - 1)))
                wins[c] += 1
                tr = KEEP * tr if trace else 0 * tr
                tr[c] = 1.0
                for k in np.nonzero(tr > 0.1)[0]:
                    w = W[k] + rate * tr[k] * (v[i] - W[k])
                    W[k] = w / np.linalg.norm(w)


def alike(a, b):
    """The mean cosine between two maps' answers at the same places (places where something answers)."""
    a, b = a.reshape(-1, a.shape[2]), b.reshape(-1, b.shape[2])
    n = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(((a * b).sum(axis=1)[n > 1e-9] / n[n > 1e-9]).mean())


def sparse(a):
    a = a.reshape(-1, a.shape[-1])
    a = a[a.max(axis=1) > 0]
    return float((1 - a.mean(axis=1) ** 2 / ((a * a).mean(axis=1) + 1e-12)).mean())


for seed in SEEDS:
    base = pickle.load(open(f"out/bench/cortex_developed_s{seed}.pkl", "rb"))
    base.relay = None
    first = base.layers[0]
    rng0 = np.random.default_rng(seed + 7)
    settle = [big(unlabelled[i]) for i in rng0.choice(len(unlabelled), NRELAY, replace=False)]
    probe = settle[:200]
    print(f"seed {seed} first stage: answers to a picture and to it two pixels aside alike {np.mean([alike(first.respond(im), first.respond(shifted(im, 0, 2))) for im in probe]):.3f} "
          f"(unpooled {np.mean([alike(first.respond(im, pooled=False), first.respond(shifted(im, 0, 2), pooled=False)) for im in probe]):.3f}) | sparse {np.mean([sparse(first.respond(im)) for im in probe]):.3f}", flush=True)
    for arm in ARMS:
        t0, rng = time.time(), np.random.default_rng(seed + 11)  # the same pictures and drifts in both arms
        second = Layer2(first.kinds, 256, 3, 1, groups=16, fan_in=32, seed=seed + 1, scaled=False)
        second.settle((first.respond(unlabelled_big) for unlabelled_big in settle[:800]), rng)
        for n in range(PICTURES):
            views = drift(big(unlabelled[rng.integers(len(unlabelled))]), rng, VIEWS)
            learn(second, [first.respond(v) for v in views], rng, arm == "trace", rate=0.05 * (1 - n / PICTURES) + 0.005)
        cortex = copy.deepcopy(base)
        cortex.layers = [first, second]
        code = lambda im: cortex.code(im, GRID)[1]  # noqa: E731  (the second stage's summary, unit length)
        maps = [(second.respond(first.respond(im)), second.respond(first.respond(shifted(im, 0, 2)))) for im in probe]
        print(f"seed {seed} {arm:5}: developed on {PICTURES} fixations of {VIEWS} views ({time.time() - t0:.0f}s; wins min/max {int(min(w.min() for w in second.wins))}/{int(max(w.max() for w in second.wins))}) | "
              f"answers to a picture and to it two pixels aside alike {np.mean([alike(a, b) for a, b in maps]):.3f} | sparse {np.mean([sparse(a) for a, _ in maps]):.3f}", flush=True)
        relay = Relay(SHRINK).settle([code(im) for im in settle])
        for name, X in (("train", Xtr), ("test", Xte)):
            out = np.zeros((len(X), GRID * GRID * second.kinds), np.float16)
            for i, x in enumerate(X):
                out[i] = code(big(x))
            np.save(f"out/bench/cifar10_l2{arm}_s{seed}_{name}.npy", out)
        keep = copy.deepcopy(base)  # the stream reads only the relay from this file
        keep.relay = relay
        pickle.dump(keep, open(f"out/bench/cortex_l2{arm}_s{seed}.pkl", "wb"))
        pickle.dump(second, open(f"out/bench/second_{arm}_s{seed}.pkl", "wb"))
        print(f"seed {seed} {arm}: codes and relay kept as l2{arm} ({time.time() - t0:.0f}s)", flush=True)
