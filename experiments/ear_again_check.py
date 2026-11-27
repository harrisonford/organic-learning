"""Is a stretch of speech found again in another voice's caption of the same photograph? (docs/VISION.md
2026-11-27 14:49, said before it runs.)

The ear's lexicon holds the common words only (VISION.md 14:36). What brings a rare word form, by the literature
(14:49): it is said again in close succession. SpokenCOCO has that: five readers describe each photograph, and a
rare word of one caption is in another of the same photograph 26-58% of the time (a count on the written captions).
Before a fast store is built on it, the premise on the sound side, with nothing learned above the hearing cortex:
the ear's first level as experiments/grounded_speech_probe.py builds it (the eye's layer on the cochlea, developed
on DEV clips, its relay; stretches cut at the loudness troughs and silences; a stretch's code its two halves' mean
codes), and windows of one, two and three stretches in a row. For PHOTOS photographs, two captions of each by two
readers (a, b). For every window of a: its best match (the codes' dot product) among the windows of b, and among
the windows of each of K captions of other photographs (other readers). Measured: how often b holds the best match
of the K + 1 captions (chance 1 / (K + 1)); the same by how many written content words a and b share (a measure
only: words of four letters or more, outside the 40 commonest).

    PHOTOS=600 SEED=0 python experiments/ear_again_check.py
    MODE=voice PHOTOS=300 K=5 ...       the same reader against another reader (VISION.md 14:58)
    RELAY=0 | BANDS=4 | HALVES=1 ...   one thing of the stretch code changed (VISION.md 14:57)
"""
import collections, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Layer2, Relay  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
PHOTOS, K, SEED, DEV = int(os.environ.get("PHOTOS", 400)), int(os.environ.get("K", 9)), int(os.environ.get("SEED", 0)), int(os.environ.get("DEV", 16000))
KINDS, WINDOW, HALF, SMOOTH = 128, 8, 5, 5
# what the stretch code is made of (defaults: the ear as it is)
MODE = os.environ.get("MODE", "photo")
BANDS, RELAY, HALVES = int(os.environ.get("BANDS", 8)), os.environ.get("RELAY", "1") == "1", int(os.environ.get("HALVES", 2))
t0 = time.time()


def say(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


rng = np.random.default_rng(SEED)
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
by_image = collections.defaultdict(list)
for u in utts:
    by_image[u["image"]].append(u)
images = sorted(by_image)
images = [images[i] for i in rng.permutation(len(images))]
common = {w for w, _ in collections.Counter(w for u in utts[::20] for w in set(u["text"].split())).most_common(40)}
pairs = []
for i in images:
    us = by_image[i]
    order = rng.permutation(len(us))
    a = us[order[0]]
    b = next((us[j] for j in order[1:] if us[j]["speaker"] != a["speaker"]), None)
    if b is not None:
        pairs.append((a, b))
    if len(pairs) >= PHOTOS:
        break
ear_pool = [u for i in images[-5000:] for u in by_image[i]]  # (photographs far from the tested ones in the shuffle)
del utts


def raw(u):
    return np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32)


def gram(u):
    g = raw(u)
    return (g - g.mean(0)) / (g.std(0) + 1e-3)  # adapted to the voice


def clip(u):
    g = gram(u)
    s = int(rng.integers(0, max(1, len(g) - 100)))
    return g[s: s + 100][..., None]


layer = Layer2(1, KINDS, WINDOW, 1, seed=SEED)
layer.settle((clip(ear_pool[i]) for i in rng.choice(len(ear_pool), 800, replace=False)), rng)
for n in range(DEV):
    layer.learn(clip(ear_pool[rng.integers(len(ear_pool))]), rng, rate=0.05 * (1 - n / DEV) + 0.005)


def frames(u):
    r = layer.respond(gram(u)[..., None], pooled=False)
    b = r.shape[1] // BANDS
    codes = r[:, : BANDS * b].reshape(r.shape[0], BANDS, b, KINDS).mean(axis=2).reshape(r.shape[0], -1)
    return codes, np.convolve(raw(u).sum(axis=1), np.ones(WINDOW) / WINDOW, "valid")[: len(codes)]


relay = Relay(0.0005).settle(np.concatenate([unit(frames(ear_pool[i])[0])[::5] for i in rng.choice(len(ear_pool), 400, replace=False)]))
say(f"BANDS {BANDS}, relay {RELAY}, halves {HALVES}: hearing cortex and its relay; {len(pairs)} photographs, two captions each by two readers")


def stretches(u):  # as grounded_speech_probe.py
    c, loud = frames(u)
    C = unit((unit(c) - relay.mu) @ relay.Z) if RELAY else unit(c)
    env = np.convolve(loud, np.ones(SMOOTH) / SMOOTH, "same")
    quiet = env < 0.1 * env.max()
    troughs = [t for t in range(1, len(env) - 1) if env[t] == env[max(0, t - HALF): t + HALF + 1].min()]
    cuts = sorted(set(troughs) | {t for t in range(1, len(env)) if quiet[t] != quiet[t - 1]} | {0, len(env)})
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 5 or quiet[a:b].mean() > 0.5:
            if out:
                out[-1][1] = True
            continue
        h = (a + b) // 2
        out.append([np.concatenate([unit(C[a:h].mean(0)), unit(C[h:b].mean(0))]) / np.sqrt(2) if HALVES == 2 else unit(C[a:b].mean(0)), False])
    return out


def windows(u):
    """The caption's windows of 1, 2 and 3 stretches in a row (none across a silence), each at unit length."""
    ss = stretches(u)
    v, gap = [s for s, _ in ss], [g for _, g in ss]
    D = len(v[0]) if v else 2 * KINDS * BANDS
    out = {}
    for n in (1, 2, 3):
        w = [np.concatenate(v[i: i + n]) / np.sqrt(n) for i in range(len(v) - n + 1) if not any(gap[i: i + n - 1])]
        out[n] = np.stack(w).astype(np.float32) if w else np.zeros((0, n * D), np.float32)
    return out


def content(u):
    return {w for w in u["text"].split() if len(w) >= 4 and w not in common}


if MODE == "photo":
    Wa, Wb = [], []
    for k, (a, b) in enumerate(pairs):
        Wa.append(windows(a))
        Wb.append(windows(b))
        if (k + 1) % 100 == 0:
            say(f"{k + 1} photographs heard")
    shared = np.array([len(content(a) & content(b)) for a, b in pairs])
    say(f"content words shared by a photograph's two captions: none {np.mean(shared == 0):.2f}, one {np.mean(shared == 1):.2f}, two or more {np.mean(shared >= 2):.2f}")
    for n in (1, 2, 3):
        first, m_same, m_other, group = [], [], [], []
        for p, (a, b) in enumerate(pairs):
            A, B = Wa[p][n], Wb[p][n]
            if not len(A) or not len(B):
                continue
            others = [q for q in rng.permutation(len(pairs)) if q != p and pairs[q][1]["speaker"] not in (a["speaker"], b["speaker"]) and len(Wb[q][n])][:K]
            same = (A @ B.T).max(axis=1)
            ctrl = np.stack([(A @ Wb[q][n].T).max(axis=1) for q in others])  # K x windows of a
            first.append(same > ctrl.max(axis=0))
            m_same.append(same)
            m_other.append(ctrl[0])
            group.append(np.full(len(A), min(shared[p], 2)))
        first, m_same, m_other, group = map(np.concatenate, (first, m_same, m_other, group))
        say(f"windows of {n} stretch{'es' if n > 1 else ''}: {len(first)} windows; the same photograph's other caption holds the best match of {K + 1} captions: {first.mean():.3f} (chance {1 / (K + 1):.3f}); "
            f"by content words shared: none {first[group == 0].mean():.3f}, one {first[group == 1].mean():.3f}, two or more {first[group == 2].mean():.3f} | "
            f"best match, same photograph / one other: median {np.median(m_same):.3f} / {np.median(m_other):.3f}, top tenth {np.quantile(m_same, 0.9):.3f} / {np.quantile(m_other, 0.9):.3f}")


# MODE=voice (VISION.md 2026-11-27 14:58): is it the voice? Captions chosen by their written words (a measure only):
# for an anchor caption, one by the SAME reader about another photograph that shares two or more content words with
# it, and one by ANOTHER reader that shares as many; each against K captions that share none (by the same reader,
# by K other readers). How often the sharing caption holds a window's best match of the K + 1 (chance 1 / (K + 1)).
if MODE == "voice":
    pool = [u for i in images[:-5000] for u in by_image[i]]
    by_reader = collections.defaultdict(list)
    for u in pool:
        by_reader[u["speaker"]].append(u)
    sample = [pool[i] for i in rng.choice(len(pool), 30000, replace=False)]
    sample_c = [content(u) for u in sample]
    anchors = []
    for i in rng.permutation(len(pool)):
        a = pool[i]
        mine = [u for u in by_reader[a["speaker"]] if u["image"] != a["image"]]
        if len(mine) < 30:
            continue
        ca = content(a)
        sh = np.array([len(ca & content(u)) for u in mine])
        n, zero = int(sh.max()), np.nonzero(sh == 0)[0]
        if n < 2 or len(zero) < K:
            continue
        start = int(rng.integers(len(sample)))
        b_other, z_other, readers = None, [], {a["speaker"]}
        for j in list(range(start, len(sample))) + list(range(start)):
            u, k = sample[j], len(ca & sample_c[j])
            if u["speaker"] in readers or u["image"] == a["image"]:
                continue
            if k == n and b_other is None:
                b_other = u
                readers.add(u["speaker"])
            elif k == 0 and len(z_other) < K:
                z_other.append(u)
                readers.add(u["speaker"])
            if b_other is not None and len(z_other) == K:
                break
        if b_other is None or len(z_other) < K:
            continue
        anchors.append((a, mine[int(np.argmax(sh))], [mine[j] for j in rng.choice(zero, K, replace=False)], b_other, z_other, n))
        if len(anchors) >= PHOTOS:
            break
    say(f"{len(anchors)} anchor captions; content words shared with the sharing captions: mean {np.mean([x[5] for x in anchors]):.1f}")
    res = {(n, c): [] for n in (1, 2, 3) for c in ("same reader", "another reader")}
    for k, (a, bs, zs, bo, zo, _) in enumerate(anchors):
        A = windows(a)
        for c, b, zz in (("same reader", bs, zs), ("another reader", bo, zo)):
            B, Z = windows(b), [windows(u) for u in zz]
            for n in (1, 2, 3):
                if len(A[n]) and len(B[n]) and all(len(z[n]) for z in Z):
                    res[(n, c)].append((A[n] @ B[n].T).max(axis=1) > np.max([(A[n] @ z[n].T).max(axis=1) for z in Z], axis=0))
        if (k + 1) % 100 == 0:
            say(f"{k + 1} anchors heard")
    for n in (1, 2, 3):
        say(f"windows of {n} stretch{'es' if n > 1 else ''}: the caption that shares content words holds the best match of {K + 1} (chance {1 / (K + 1):.3f}): "
            + "; ".join(f"{c} {np.concatenate(res[(n, c)]).mean():.3f} ({len(np.concatenate(res[(n, c)]))} windows)" for c in ("same reader", "another reader")))
