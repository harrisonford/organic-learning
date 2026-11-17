"""Which of the fixes that recur in the literature help our cortex? (docs/RESEARCH_VISION.md, docs/VISION.md)

One layer at a time, read out by the same linear reference as before (a ridge on the code; no brain involved):
held-out COCO things (35 names) and the first-words pictures of CIFAR + Commons (22 names).

A variant is a list of words in VARIANT (comma-separated variants, each "+"-joined):

  old            the living cortex's first layer as it is (64 kinds, relu x contrast, the strongest 8 kept, max over
                 a 4x4 grid): the reference
  tri            the same filters; the response is each kind's drive above the average of all kinds at that place
                 (lateral inhibition, the "triangle" of Coates et al. 2011), nothing cut off
  random         filters never developed (random directions)
  white          the windows are decorrelated first, as the retina does (whitening from the windows' own statistics)
  k256, k1024    more kinds (developed afresh by the cortex's own winner-take-all rule, on DEVELOP pictures)
  soft           developed with graded competition and a push away for the runners-up (SoftHebb, Journé et al. 2023)
  avg            the grid summary is the average over the region instead of the strongest
  g2 / g6        a 2x2 or 6x6 grid instead of 4x4
  p2             the response squared before pooling (a steeper nonlinearity)
  dog            no whitening matrix: the picture first passes a retina of centre-surround cells (each point minus the
                 average of its surround, per colour channel), which is what whitening natural scenes amounts to
  noe            the response is not scaled by the window's contrast

    VARIANT=old,tri,random+tri,white+tri,white+tri+k256 python experiments/cortex2_probe.py
"""
import json, os, pickle, sys, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.world import DATA  # noqa: E402
from organic.curriculum import Curriculum  # noqa: E402

VARIANTS = os.environ.get("VARIANT", "old,tri,random+tri,white+tri,white+tri+k256").split(",")
SEED, FIT, DEVELOP = int(os.environ.get("SEED", 0)), int(os.environ.get("FIT", 6000)), int(os.environ.get("DEVELOP", 4000))
pics = np.load(os.path.join(DATA, "cococrops", "pictures_64.npy"), mmap_mode="r")
meta = [json.loads(l) for l in open(os.path.join(DATA, "cococrops", "pictures.jsonl"))]
names = sorted({m["word"] for m in meta})
label = np.array([names.index(m["word"]) for m in meta])
held = np.array([m["image"] % 20 == 0 for m in meta])
rng = np.random.default_rng(SEED)
train_rows, test_rows = rng.permutation(np.nonzero(~held)[0]), rng.permutation(np.nonzero(held)[0])[:1500]
img = lambda r: np.asarray(pics[r], np.float32) / 255  # noqa: E731
old = pickle.load(open("data/visual_cortex.pkl", "rb")).layers[0]
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)  # noqa: E731
cur = Curriculum(seed=0)
tr2 = [it for src in ("cifar:name", "commons:name") for it in cur.train["W1"][src]][::3][:3000]
te2 = [it for src in ("cifar:name", "commons:name") for it in cur.test["W1"][src]]
tr2, te2 = [it for it in tr2 if cur.image(it[0]).max() > 0], [it for it in te2 if cur.image(it[0]).max() > 0]
old2 = sorted({it[2][0] for it in tr2})
WIN = 5


def windows(x):
    """(H, W, C) -> every 5x5 window as a row, its mean removed, at unit length; and each window's contrast."""
    v = sliding_window_view(x, (WIN, WIN), axis=(0, 1))
    v = v.transpose(0, 1, 3, 4, 2).reshape(v.shape[0], v.shape[1], -1)
    v = v - v.mean(axis=2, keepdims=True)
    e = np.linalg.norm(v, axis=2)
    return v / (e[..., None] + 1e-6), e


def ridge(A, Y, B, lam):
    if A.shape[1] > A.shape[0]:
        K = A @ A.T
        K[np.diag_indices_from(K)] += lam
        return (B @ A.T) @ np.linalg.solve(K, Y)
    G = A.T @ A
    G[np.diag_indices_from(G)] += lam
    return B @ np.linalg.solve(G, A.T @ Y)


def blur(x, n=2):
    """The average of each point's surround (a box of side 2n+1, edges repeated), per channel."""
    p = np.pad(x, ((n, n), (n, n), (0, 0)), mode="edge")
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    k = 2 * n + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def build(variant):
    """The layer of a variant: (whitening matrix or None, filters, how to respond)."""
    opts = set(variant.split("+"))
    kinds = 1024 if "k1024" in opts else 256 if "k256" in opts else 64
    dev = np.random.default_rng(SEED + 5)
    retina = (lambda x: x - blur(x)) if "dog" in opts else (lambda x: x)
    pool = [retina(img(r)) for r in train_rows[-DEVELOP:]]
    Z = None
    if "white" in opts:
        # the windows' own correlations, removed: what a retina's centre-surround cells do to natural scenes
        sample = np.concatenate([(lambda v, e: v.reshape(-1, v.shape[2])[dev.choice(v.shape[0] * v.shape[1], 40, replace=False)])(*windows(p)) for p in pool[:2000]])
        C = np.cov(sample.T)
        d, E = np.linalg.eigh(C)
        Z = (E / np.sqrt(d + 1e-3 * d.max())) @ E.T
        Z = Z.astype(np.float32)
    if "old" in opts or ("tri" in opts and not opts & {"white", "random", "k256", "k1024", "soft", "dog"}):
        W = old.W.copy()
    else:
        W = dev.normal(0, 1, (kinds, WIN * WIN * 3)).astype(np.float32)
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        if "random" not in opts:
            wins = np.zeros(kinds)
            total = DEVELOP * (4 if kinds > 64 else 1)
            for n in range(total):
                v, e = windows(pool[dev.integers(len(pool))])
                v, e = v.reshape(-1, v.shape[2]), e.ravel()
                strong = np.nonzero(e > 0.3 * e.max())[0] if e.max() > 1e-6 else []
                if not len(strong):
                    continue
                take = v[dev.choice(strong, min(32, len(strong)), replace=False)]
                if Z is not None:
                    take = take @ Z
                    take /= np.linalg.norm(take, axis=1, keepdims=True) + 1e-8
                rate = 0.05 * (1 - n / total) + 0.005
                for x in take:
                    u = W @ x
                    if "soft" in opts:  # graded competition: the winner moves towards the window, the others a little away
                        y = np.exp((u - u.max()) / 0.1)
                        y /= y.sum()
                        c = int(np.argmax(u))
                        y = -y
                        y[c] = -y[c]
                        W += rate * y[:, None] * (x[None, :] - u[:, None] * W)
                        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
                    else:  # the cortex's own rule: the winner moves towards the window; a conscience evens out the wins
                        c = int(np.argmax(u - 0.3 * (wins / (wins.mean() + 1e-8) - 1)))
                        wins[c] += 1
                        w = W[c] + rate * (x - W[c])
                        W[c] = w / np.linalg.norm(w)
    grid = 2 if "g2" in opts else 6 if "g6" in opts else 4

    def code(x):
        v, e = windows(retina(x))
        if "noe" in opts:
            e = np.ones_like(e)
        if Z is not None:
            v = v @ Z
            v = v / (np.linalg.norm(v, axis=2, keepdims=True) + 1e-8)
        u = v @ W.T
        if "tri" in opts:
            r = np.maximum(u - u.mean(axis=2, keepdims=True), 0) * e[..., None]
        else:
            r = np.maximum(u, 0) * e[..., None]
            k = max(8, kinds // 8)
            kth = np.partition(r, -k, axis=2)[..., -k][..., None]
            r = np.where(r >= kth, r, 0)
        if "p2" in opts:
            r = r * r
        h, w = (r.shape[0] // grid) * grid, (r.shape[1] // grid) * grid
        r = r[:h, :w].reshape(grid, h // grid, grid, w // grid, -1)
        p = (r.mean(axis=(1, 3)) if "avg" in opts else r.max(axis=(1, 3))).ravel()
        return (p / (np.linalg.norm(p) + 1e-8)).astype(np.float32)

    return code


for variant in VARIANTS:
    t0 = time.time()
    code = build(variant)
    fit = train_rows[:FIT]
    A = np.stack([code(img(r)) for r in fit]); B = np.stack([code(img(r)) for r in test_rows])
    mean = A.mean(0); A, B = A - mean, B - mean
    Y = np.eye(len(names), dtype=np.float32)[label[fit]]
    coco = max(float((ridge(A, Y, B, lam).argmax(1) == label[test_rows]).mean()) for lam in (0.01, 0.03, 0.1, 0.3, 1.0))
    A2 = np.stack([code(cur.image(it[0])) for it in tr2]); B2 = np.stack([code(cur.image(it[0])) for it in te2])
    m2 = A2.mean(0); A2, B2 = A2 - m2, B2 - m2
    Y2 = np.eye(len(old2), dtype=np.float32)[[old2.index(it[2][0]) for it in tr2]]
    y2 = np.array([old2.index(it[2][0]) if it[2][0] in old2 else -1 for it in te2])
    first = max(float((ridge(A2, Y2, B2, lam).argmax(1) == y2).mean()) for lam in (0.01, 0.03, 0.1, 0.3, 1.0))
    log(f"{variant:28} code of {A.shape[1]:>6} numbers | held-out COCO things {coco:.3f} | CIFAR+Commons {first:.3f} | {time.time() - t0:.0f}s")
    del A, B, A2, B2
