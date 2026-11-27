"""How even is the lives' relayed code on each kind of picture the lives name? (docs/VISION.md 2026-11-27 10:50,
said before it runs.)

On CORe50 the word cells' rule missed a better code because the relay had been settled on other pictures
(VISION.md 08:28, 10:11). The lives' cortex (data/visual_cortex2.pkl) has one relay, settled once, and the lives name
pictures from three sources at first words: CIFAR (32 x 32 pictures enlarged), Commons photographs and COCO crops.
For each source, with the lives' own world (world 4, read only) and cortex:
    evenness      the relayed code's participation ratio on 3,000 of that source's pictures (of 4,096), and the
                  share of the variance in the strongest 10 directions
    readers       names of held-out pictures of the source after one pass over its lived pictures in a random order:
                  the delta reader (bench_stream.py's: the word cells' rule without the words) and a ridge on the
                  same pictures (the reference: what a linear reader can get from them); the gap between the two is
                  what the delta rule loses to the code's unevenness
    evened again  the same two readers with a relay settled once more on the source's own lived pictures (no
                  labels), only where there are more pictures than the code has directions

    python experiments/relay_by_source_probe.py
"""
import os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Relay  # noqa: E402
from organic.curriculum import Curriculum  # noqa: E402

SEED, CAP = int(os.environ.get("SEED", 0)), int(os.environ.get("CAP", 10000))
cur = Curriculum(seed=0, world=4)
cortex = pickle.load(open("data/visual_cortex2.pkl", "rb"))
relay, cortex.relay = cortex.relay, None
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
SHARE = 2.0 + len(set(ear.units("what is this?", learn=False)))
del ear
rng = np.random.default_rng(SEED)
t0 = time.time()


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def coarse(img):
    f = img.shape[0] // 64
    return img if f == 1 else img.reshape(64, f, 64, f, 3).mean(axis=(1, 3))


def pictures(items, cap):
    """(the raw summaries, the names) of a source's pictures, each picture once."""
    seen, X, y = set(), [], []
    for i in rng.permutation(len(items)):
        ref, name = items[i][0], items[i][2][0]
        if ref in seen:
            continue
        seen.add(ref)
        img = cur.image(ref)
        if img.max() > 0:
            X.append(cortex.whole(coarse(img), 4).astype(np.float32))
            y.append(name)
        if len(X) >= cap:
            break
    return np.stack(X), y


def delta(X, y, Xt, yt, C):
    D = X.shape[1]
    W, base, use, mean = np.zeros((C, D), np.float32), np.zeros(C, np.float32), np.zeros(D, np.float32), np.zeros(D, np.float32)
    for n, (v, c) in enumerate(zip(X, y), 1):
        mean += (v - mean) / min(n, 1000)
        x = v - mean
        x /= np.linalg.norm(x) + 1e-8
        e = -(W @ x + base)
        e[c] += 1.0
        step = e / SHARE
        W += step[:, None] * (x / (1.0 + use / 5.0))[None, :]
        use += x * x
        base += step / (1.0 + n / 5.0)
    xt = Xt - mean
    xt /= np.linalg.norm(xt, axis=1, keepdims=True) + 1e-8
    return float(np.mean(np.argmax(xt @ W.T + base, axis=1) == yt))


def ridge(X, y, Xt, yt, C):
    m = X.mean(axis=0)
    A = ((X - m).T @ (X - m)).astype(np.float64)
    Y = np.eye(C)[y]
    B = (X - m).T @ (Y - Y.mean(axis=0))
    best = 0.0
    for lam in (0.01, 0.1, 1.0):  # as bench_stream.py: the best on the held-out pictures (generous to the reference)
        W = np.linalg.solve(A + lam * (np.trace(A) / A.shape[0]) * np.eye(A.shape[0]), B)
        best = max(best, float(np.mean(np.argmax((Xt - m) @ W, axis=1) == yt)))
    return best


def evenness(X):
    d = np.clip(np.linalg.eigvalsh(np.cov((X - X.mean(axis=0)).T.astype(np.float64))), 0, None)[::-1]
    return d.sum() ** 2 / (d * d).sum(), d[:10].sum() / d.sum()


print(f"the lives' relay: {type(relay).__name__}, shrink {getattr(relay, 'shrink', None)}", flush=True)
for src in ("coco:name", "cifar:name", "commons:name"):
    R, names = pictures(cur.train["W1"][src], CAP)
    Rt, names_t = pictures(cur.test["W1"][src], 2000)
    words = sorted(set(names))
    keep = [i for i, w in enumerate(names_t) if w in words]
    y, yt = np.array([words.index(w) for w in names]), np.array([words.index(names_t[i]) for i in keep])
    X, Xt = unit((R - relay.mu) @ relay.Z), unit((Rt[keep] - relay.mu) @ relay.Z)
    C = len(words)
    pr_raw, _ = evenness(R[:3000])  # on the same number of pictures for every source (the ratio grows with their number)
    pr, t10 = evenness(X[:3000])
    line = (f"{src:13}: {len(X)} lived pictures, {len(Xt)} held out, {C} names | participation ratio on 3,000: raw {pr_raw:.0f}, relayed {pr:.0f} of 4096 "
            f"(strongest 10: {t10:.2f}) | delta {delta(X, y, Xt, yt, C):.3f}, ridge {ridge(X, y, Xt, yt, C):.3f}")
    if len(X) > 1.5 * X.shape[1]:
        own = Relay(relay.shrink).settle(X)
        X2, Xt2 = unit((X - own.mu) @ own.Z), unit((Xt - own.mu) @ own.Z)
        line += f" | evened again on its own lived pictures: delta {delta(X2, y, Xt2, yt, C):.3f}, ridge {ridge(X2, y, Xt2, yt, C):.3f}"
    print(line + f" ({time.time() - t0:.0f}s)", flush=True)
