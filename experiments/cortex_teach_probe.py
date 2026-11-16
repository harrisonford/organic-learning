"""Can the visual cortex learn from the words? More to see, and words that reach back (offline; docs/VISION.md).

The cortex developed by itself on 4,000 pictures per layer and has not changed since; no readout names more than
0.55 of held-out pictures from its code. Two questions, on the picture book cut from COCO (95,000 pictures of 32
names, experiments/world/coco_crops.py):

  ARM=as_is      the cortex as every life has had it
  ARM=more       the same cortex developed afresh, by its own rule, on DEVELOP pictures per layer (more to see)
  ARM=taught     the cortex as it is, then TAUGHT pictures are named for it: every visual neuron that talks to the
                 word cells hears back from them. A neuron whose activity spoke for the word that was then heard
                 moves its synapses toward what it was looking at; one that spoke for a word that was not heard
                 moves away. (Feedback-gated plasticity: Roelfsema & van Ooyen 2005; Roelfsema & Holtmaat 2018;
                 three-factor rules: Fremaux & Gerstner 2016.) The error travels one synapse, from a word cell
                 to the visual neurons it listens to, in every layer; nothing is passed on from layer to layer.
  ARM=taught_top the same, only the top layer listens.

Measured with a reference readout (ridge, fitted in a batch) on the code of FIT pictures: held-out COCO pictures
(other photographs), and the first-words test pictures of CIFAR and Commons that no arm trained on.
    ARM=taught TAUGHT=20000 python experiments/cortex_teach_probe.py
"""
import json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.world import DATA  # noqa: E402
from organic.visual_cortex import VisualCortex  # noqa: E402
from organic.curriculum import Curriculum  # noqa: E402

ARM = os.environ.get("ARM", "as_is")
SEED = int(os.environ.get("SEED", 0))
TAUGHT, DEVELOP, FIT, RATE = int(os.environ.get("TAUGHT", 20000)), int(os.environ.get("DEVELOP", 40000)), int(os.environ.get("FIT", 6000)), float(os.environ.get("RATE", 0.02))
GRID = 4
pics = np.load(os.path.join(DATA, "cococrops", "pictures_64.npy"), mmap_mode="r")
meta = [json.loads(l) for l in open(os.path.join(DATA, "cococrops", "pictures.jsonl"))]
names = sorted({m["word"] for m in meta})
label = np.array([names.index(m["word"]) for m in meta])
held = np.array([m["image"] % 20 == 0 for m in meta])  # one photograph in twenty is never trained on
rng = np.random.default_rng(SEED)
train_rows, test_rows = rng.permutation(np.nonzero(~held)[0]), rng.permutation(np.nonzero(held)[0])[:1500]
img = lambda r: np.asarray(pics[r], np.float32) / 255  # noqa: E731
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
log = lambda *a: print(f"[{ARM} {time.strftime('%H:%M:%S')}]", *a, flush=True)  # noqa: E731


def code(c, image):
    v = np.concatenate(c.code(image, grid=GRID))
    return v / (np.linalg.norm(v) + 1e-8)


def forward(c, image):
    """The code, and for every layer what teaching needs: for each (grid cell, kind) the window it responded to most."""
    x, parts, keep = np.asarray(image, np.float32), [], []
    for layer in c.layers:
        v, e = layer.windows(x)
        r = np.maximum(v @ layer.W.T, 0) * e[..., None]
        if layer.active < layer.kinds:
            kth = np.partition(r, -layer.active, axis=2)[..., -layer.active][..., None]
            r = np.where(r >= kth, r, 0)
        p = layer.pool
        h, w = (r.shape[0] // p) * p, (r.shape[1] // p) * p
        x = r[:h, :w].reshape(h // p, p, w // p, p, -1).max(axis=(1, 3)).astype(np.float32)
        gh, gw = (h // p) // GRID * p, (w // p) // GRID * p  # a grid cell, in positions before pooling
        R = r[: gh * GRID, : gw * GRID].reshape(GRID, gh, GRID, gw, -1).transpose(0, 2, 1, 3, 4).reshape(GRID * GRID, gh * gw, -1)
        Vw = v[: gh * GRID, : gw * GRID].reshape(GRID, gh, GRID, gw, -1).transpose(0, 2, 1, 3, 4).reshape(GRID * GRID, gh * gw, -1)
        where = R.argmax(axis=1)  # (cells, kinds)
        top = np.take_along_axis(R, where[:, None, :], axis=1)[:, 0, :]
        norm = np.linalg.norm(top) + 1e-8
        parts.append((top / norm).ravel())
        keep.append((Vw, where, top > 0, norm))
    v = np.concatenate(parts)
    return v / (np.linalg.norm(v) + 1e-8), keep


def measure(c, tag):
    t0 = time.time()
    fit = train_rows[-FIT:]  # pictures the taught arms did not learn from
    A = np.stack([code(c, img(r)) for r in fit]); B = np.stack([code(c, img(r)) for r in test_rows])
    mean = A.mean(0); A, B = A - mean, B - mean
    Y = np.eye(len(names), dtype=np.float32)[label[fit]]
    G = A.T @ A
    best = max(float(((B @ np.linalg.solve(G + lam * np.eye(G.shape[0], dtype=np.float32), A.T @ Y)).argmax(1) == label[test_rows]).mean()) for lam in (0.03, 0.1, 0.3))
    cur = Curriculum(seed=0)
    tr = [it for src in ("cifar:name", "commons:name") for it in cur.train["W1"][src]][::3][:3000]
    te = [it for src in ("cifar:name", "commons:name") for it in cur.test["W1"][src]]
    tr, te = [it for it in tr if cur.image(it[0]).max() > 0], [it for it in te if cur.image(it[0]).max() > 0]
    old = sorted({it[2][0] for it in tr})
    A2 = np.stack([code(c, cur.image(it[0])) for it in tr]); B2 = np.stack([code(c, cur.image(it[0])) for it in te])
    m2 = A2.mean(0); A2, B2 = A2 - m2, B2 - m2
    Y2 = np.eye(len(old), dtype=np.float32)[[old.index(it[2][0]) for it in tr]]
    y2 = np.array([old.index(it[2][0]) if it[2][0] in old else -1 for it in te])
    G2 = A2.T @ A2
    best2 = max(float(((B2 @ np.linalg.solve(G2 + lam * np.eye(G2.shape[0], dtype=np.float32), A2.T @ Y2)).argmax(1) == y2).mean()) for lam in (0.03, 0.1, 0.3))
    per = [float(((B[:, a:b] @ np.linalg.solve(A[:, a:b].T @ A[:, a:b] + 0.1 * np.eye(b - a, dtype=np.float32), A[:, a:b].T @ Y)).argmax(1) == label[test_rows]).mean())
           for a, b in zip(np.cumsum([0] + [l.kinds * 16 for l in c.layers])[:-1], np.cumsum([l.kinds * 16 for l in c.layers]))]
    log(f"{tag}: held-out COCO pictures named {best:.3f} ({len(names)} names, {FIT} to fit) | layers alone {' / '.join(f'{p:.3f}' for p in per)} | CIFAR+Commons test pictures {best2:.3f} ({len(old)} names, {len(tr)} to fit) | {time.time() - t0:.0f}s")
    return best, best2


log(f"{len(pics)} pictures, {len(names)} names, {int(held.sum())} held out; commonest name {np.bincount(label).max() / len(label):.3f}")
if ARM == "as_is":
    measure(cortex, "as it is")
elif ARM == "more":
    fresh = VisualCortex(seed=SEED, kinds=tuple(l.kinds for l in cortex.layers))
    pool = [img(r) for r in train_rows[:8000]]
    fresh.develop(pool, np.random.default_rng(SEED), per_layer=DEVELOP, log=log)
    measure(fresh, f"developed afresh on {DEVELOP} per layer")
    pickle.dump(fresh, open(f"out/tmp/cortex_more_{DEVELOP}.pkl", "wb"))
else:
    layers = [len(cortex.layers) - 1] if ARM == "taught_top" else list(range(len(cortex.layers)))
    dims = [l.kinds * GRID * GRID for l in cortex.layers]
    offs = np.cumsum([0] + dims)
    Wout = np.zeros((len(names), offs[-1]), np.float32)
    use = np.zeros(offs[-1], np.float32)  # each visual neuron's own use: its synapses onto the word cells consolidate with it
    mean = np.zeros(offs[-1], np.float32)
    right = []
    for n, r in enumerate(train_rows[:TAUGHT], 1):
        x, keep = forward(cortex, img(r))
        mean += (x - mean) / min(n, 1000)
        xc = x - mean
        y = Wout @ xc
        e = -y
        e[label[r]] += 1.0
        right.append(int(np.argmax(y)) == label[r])
        if n > 500:  # the cortex listens once the word cells have something to say
            t = Wout.T @ e  # what each visual neuron hears back: it spoke for the heard word (+) or for another (-)
            for li in layers:
                layer = cortex.layers[li]
                Vw, where, on, norm = keep[li]
                tl = t[offs[li] : offs[li + 1]].reshape(GRID * GRID, layer.kinds) * on / norm
                cells, kinds = np.nonzero(tl)
                if len(cells):
                    np.add.at(layer.W, kinds, RATE * tl[cells, kinds][:, None] * Vw[cells, where[cells, kinds]])
                    layer.W /= np.linalg.norm(layer.W, axis=1, keepdims=True) + 1e-8
        Wout += np.outer(e, xc / (1.0 + use / 5.0)) * 0.5
        use += xc * xc
        if n % 2000 == 0:
            log(f"{n} named: the word cells were right on {np.mean(right[-2000:]):.3f} of the last 2000 (before hearing the name)")
    measure(cortex, f"taught on {TAUGHT} ({'top layer' if ARM == 'taught_top' else 'all layers'}, rate {RATE})")
    pickle.dump(cortex, open(f"out/tmp/cortex_{ARM}_{TAUGHT}.pkl", "wb"))
