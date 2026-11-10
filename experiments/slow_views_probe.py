"""Second attempt at the two-part rule on the eye's own views: one layer above the visual cortex (offline; docs/VISION.md).

The first attempt changed the cortex's filters themselves and made naming worse (out/lpl_*.log): its
formulation let overall contrast dominate, and filters need far more views than a life here has.
This one keeps the cortex as it is and adds one layer of neurons on top of its code. Each neuron,
seeing successive views of one picture (whole, then closer around a gaze point, closer still, another
point), changes its synapses
    predict       to answer the new view as it answered the last one (what stays the same is the thing),
    hebb          to move away from its own mean, the more the less it varies (or prediction collapses),
    decorrelate   away from what the neurons it co-varies with are doing,
all in units of its own variability (Halvagar & Zenke 2023). The layer starts as random synapses.

Measured on whole held-out pictures, first-words naming: nearest remembered picture (1-NN) and a
linear readout (ridge), for the cortex code itself, the layer before learning, and after each pass.
    python experiments/slow_views_probe.py            ARM=both|hebb|predict  K=512  RATE=0.02  PASSES=6
"""
import os, pickle, random, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum  # noqa: E402

K = int(os.environ.get("K", 512))
RATE = float(os.environ.get("RATE", 0.02))
PASSES = int(os.environ.get("PASSES", 6))
ARM = os.environ.get("ARM", "both")
A_PRED, A_HEBB, A_DEC = {"both": (1, 1, 1), "hebb": (0, 1, 1), "predict": (1, 0, 0)}[ARM]
cur = Curriculum(seed=0)
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
TARGETS = [(x, y) for y in np.linspace(4, 60, 5) for x in np.linspace(4, 60, 5)]
rng = random.Random(0)


def crop(img, cx, cy, side):
    x0 = int(min(max(cx - side / 2, 0), 64 - side))
    y0 = int(min(max(cy - side / 2, 0), 64 - side))
    part = (img[y0 : y0 + side, x0 : x0 + side] * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(part).resize((64, 64), Image.BILINEAR), np.float32) / 255


def code(img):
    return np.concatenate(cortex.code(img, grid=4)).astype(np.float32)


path = "out/tmp/view_codes.npz"
if not os.path.exists(path):
    tr = [it for src in ("cifar:name", "commons:name") for it in cur.train["W1"][src]]
    te = rng.sample(cur.test["W1"]["cifar:name"], 400) + cur.test["W1"]["commons:name"][:380]
    t0 = time.time()
    views = np.zeros((len(tr), 5, 7168), np.float32)
    for i, it in enumerate(tr):
        img = cur.image(it[0])
        vs = [img]
        for _ in range(2):
            cx, cy = TARGETS[rng.randrange(25)]
            vs += [crop(img, cx, cy, 40), crop(img, cx, cy, 24)]
        views[i] = [code(v) for v in vs]
    B = np.array([code(cur.image(it[0])) for it in te])
    np.savez(path, views=views, B=B, ytr=np.array([it[2][0] for it in tr]), yte=np.array([it[2][0] for it in te]))
    print(f"view codes computed in {time.time() - t0:.0f}s", flush=True)
d = np.load(path)
views, B, ytr, yte = d["views"], d["B"], d["ytr"], d["yte"]
mu_c = views.reshape(-1, views.shape[2]).mean(0)  # each cortex neuron passes on its deviation from its own average
norm = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)  # noqa: E731
Xv, Xb = norm(views - mu_c), norm(B - mu_c)
classes = sorted(set(ytr))
Ytr = np.array([[y == c for c in classes] for y in ytr], np.float32)
sub = np.random.default_rng(0).permutation(len(ytr))[:4000]  # remembered pictures for the readouts


def measure(f, label):
    A_, B_ = norm(f(Xv[sub, 0])), norm(f(Xb))
    nn = float(np.mean(ytr[sub][(B_ @ A_.T).argmax(1)] == yte))
    Kmat = A_ @ A_.T
    alpha = np.linalg.solve(Kmat + 1.0 * np.eye(len(Kmat)), Ytr[sub])
    ridge = float(np.mean(np.array(classes)[(B_ @ A_.T @ alpha).argmax(1)] == yte))
    # how alike are two views of one picture, compared with views of different pictures
    Y = norm(f(Xv[:2000].reshape(-1, Xv.shape[2]))).reshape(2000, 5, -1)
    same = float(np.mean([(Y[:, a] * Y[:, b]).sum(1).mean() for a in range(5) for b in range(a)]))
    other = float((Y[:1000, 0] @ Y[1000:2000, 3].T).mean())
    print(f"{label:28} nearest picture {nn:.3f} | linear readout {ridge:.3f} | two views of one picture alike {same:.2f}, of different pictures {other:.2f}", flush=True)


measure(lambda X: X, "the cortex code itself")
g = np.random.default_rng(1)
W = g.normal(0, 1, (K, Xv.shape[2])).astype(np.float32)
W /= np.linalg.norm(W, axis=1, keepdims=True)
measure(lambda X: X @ W.T, f"{K} random neurons on top")
mu, var, R = np.zeros(K, np.float32), np.ones(K, np.float32), np.zeros((K, K), np.float32)
first = Xv[:512].reshape(-1, Xv.shape[2]) @ W.T
mu, var = first.mean(0), first.var(0) + 1e-8
t0 = time.time()
for p in range(PASSES):
    order = np.random.default_rng(p).permutation(len(Xv))
    for s in range(0, len(order), 64):
        X = Xv[order[s : s + 64]]  # (pictures, views, inputs)
        Y = X @ W.T
        flat = Y.reshape(-1, K)
        mu += 0.05 * (flat.mean(0) - mu)
        var += 0.05 * (((flat - mu) ** 2).mean(0) - var)
        sd = np.sqrt(var) + 1e-8
        Z = (flat - mu) / sd
        R += 0.05 * (Z.T @ Z / len(Z) - R)
        off = R - np.diag(np.diag(R))
        # error signal of each neuron at each view t >= 1 (the view before it is t - 1)
        dY = Y[:, 1:] - Y[:, :-1]
        cur_ = Y[:, 1:]
        delta = (A_PRED * dY - A_HEBB * (cur_ - mu)) / sd ** 2 + A_DEC * (((cur_ - mu) / sd) @ off) / sd
        grad = np.einsum("nvk,nvd->kd", delta, X[:, 1:]) / (delta.shape[0] * delta.shape[1])
        W -= RATE * sd[:, None] ** 2 * grad  # a step in units of each neuron's own variability
        W /= np.linalg.norm(W, axis=1, keepdims=True)
    measure(lambda X: X @ W.T, f"{ARM}: after pass {p + 1} ({time.time() - t0:.0f}s)")
    print(f"      mean |correlation| between neurons {np.abs(off).mean():.3f}", flush=True)
