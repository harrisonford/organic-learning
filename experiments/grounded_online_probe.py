"""The one-voice ear in one pass: the kinds and the link to sight learning together, caption by caption
(docs/VISION.md 2026-11-27 18:40, said before it runs).

experiments/grounded_speech_probe.py with READER=top grows the ear in two stages: the kinds on one set of captions,
then, with the kinds held still, the Hebb link on another set heard with their photographs. A child does not hear
in two stages, and a live ear (docs/ASYNC.md, the live page) cannot. Here one reader's captions come once, each with
its photograph, in one order: the stretches of the caption go to the kinds as they are at that moment (and move
them: the same winner-take-all rule with a conscience, the first stretches founding the kinds), and the bag so
made is tied to the photograph's code by Hebb's rule at once (the bag centred by the running mean of the bags so
far, at unit length). Only the hearing cortex below the kinds (the eye's layer on the cochlea, and its relay) is
developed first, on the first FIRST captions' sound alone, as the eye's cortex is before a life.
Read at checkpoints, nothing learning: 1,000 photographs and captions of the same reader never heard or seen, the
caption among 1,000 for its photograph and back (R@1, 5, 10, the median rank), beside the two-stage ear's numbers
at as many pairs (VISION.md 17:36). At the end: the same with the bags tied to other photographs (a second link
kept beside the first: chance), and, as a measure only, the caption's written words from its bag.

    READER=top SEED=0 python experiments/grounded_online_probe.py
"""
import json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Layer2, Relay  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
READER, SEED, TEST, FIRST = os.environ.get("READER", "top"), int(os.environ.get("SEED", 0)), int(os.environ.get("TEST", 1000)), int(os.environ.get("FIRST", 1000))
MAXU, DEV = int(os.environ.get("MAXU", 0)), int(os.environ.get("DEV", 16000))
KINDS, WINDOW, BANDS, HALF, SMOOTH, K23 = 128, 8, 8, 5, 5, 1024
CHECKS = [int(x) for x in os.environ.get("CHECKS", "1000,2500,5000,7500").split(",")]
t0 = time.time()


def say(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


rng = np.random.default_rng(SEED)
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "train" and u["rows"] >= 50]
count = {}
for u in utts:
    count[u["speaker"]] = count.get(u["speaker"], 0) + 1
who = sorted(count, key=count.get, reverse=True)[int(READER[3:] or 1) - 1] if READER.startswith("top") else READER
utts = [u for u in utts if u["speaker"] == who]
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
row_of = {i: r for r, i in enumerate(json.load(open(os.path.expanduser("~/datasets/coco_thumbs/train2017_ids.json"))))}
thumbs = np.load(os.path.expanduser("~/datasets/coco_thumbs/train2017_64.npy"), mmap_mode="r")
by_image = {}
for u in utts:
    by_image.setdefault(u["image"], []).append(u)
images = sorted(by_image)
images = [images[i] for i in rng.permutation(len(images))]
test_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[:TEST]]
life_u = [by_image[i][int(rng.integers(len(by_image[i])))] for i in images[TEST:]][: MAXU or None]  # one caption a photograph, in this order
say(f"reader {READER}: {len(utts)} captions of {len(images)} photographs; the life: {len(life_u)} captions, each with its photograph, once; test {len(test_u)} never heard or seen")


def raw(u):
    return np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32)


def gram(u):
    g = raw(u)
    return (g - g.mean(0)) / (g.std(0) + 1e-3)  # adapted to the voice


def clip(u):
    g = gram(u)
    s = int(rng.integers(0, max(1, len(g) - 100)))
    return g[s: s + 100][..., None]


# the hearing cortex below the kinds, from the first captions' sound alone (as grounded_speech_probe.py builds it)
first = life_u[:FIRST]
layer = Layer2(1, KINDS, WINDOW, 1, seed=SEED)
layer.settle((clip(first[i]) for i in rng.choice(len(first), min(800, len(first)), replace=False)), rng)
for n in range(DEV):
    layer.learn(clip(first[rng.integers(len(first))]), rng, rate=0.05 * (1 - n / DEV) + 0.005)


def frames(u):
    r = layer.respond(gram(u)[..., None], pooled=False)
    b = r.shape[1] // BANDS
    codes = r[:, : BANDS * b].reshape(r.shape[0], BANDS, b, KINDS).mean(axis=2).reshape(r.shape[0], -1)
    return codes, np.convolve(raw(u).sum(axis=1), np.ones(WINDOW) / WINDOW, "valid")[: len(codes)]


relay = Relay(0.0005).settle(np.concatenate([unit(frames(first[i])[0])[::5] for i in rng.choice(len(first), min(400, len(first)), replace=False)]))
say(f"hearing cortex and its relay, from the first {len(first)} captions' sound")


def stretches(u):
    c, loud = frames(u)
    C = unit((unit(c) - relay.mu) @ relay.Z)
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
        out.append([np.concatenate([unit(C[a:h].mean(0)), unit(C[h:b].mean(0))]) / np.sqrt(2), False])
    return out


class Kinds:
    def __init__(self, n, dim):
        self.U, self.wins, self.founded = np.zeros((n, dim), np.float32), np.zeros(n), 0

    def __call__(self, s, learning):
        if learning and self.founded < len(self.U):
            self.U[self.founded], self.wins[self.founded] = s, 1
            self.founded += 1
            return self.founded - 1
        f = self.founded
        c = int(np.argmax(self.U[:f] @ s - (0.3 * (self.wins[:f] / (self.wins[:f].mean() + 1e-8) - 1) if learning else 0)))
        if learning:
            self.wins[c] += 1
            w = self.U[c] + (1.0 / (1 + self.wins[c] / 5)) * (s - self.U[c])
            self.U[c] = w / (np.linalg.norm(w) + 1e-8)
        return c

    def many(self, S):
        return np.argmax(self.U[: self.founded] @ S.T, axis=0)


D = 2 * KINDS * BANDS
pairs, triples = Kinds(K23, 2 * D), Kinds(K23, 3 * D)


def bag(u, learning):
    ss = stretches(u)
    v, gap = [s for s, _ in ss], [g for _, g in ss]
    out = np.zeros(2 * K23, np.float32)
    P = [np.concatenate(v[i: i + 2]) / np.sqrt(2) for i in range(len(v) - 1) if not gap[i]]
    T = [np.concatenate(v[i: i + 3]) / np.sqrt(3) for i in range(len(v) - 2) if not (gap[i] or gap[i + 1])]
    if learning:
        for s in P:
            out[pairs(s, True)] += 1
        for s in T:
            out[K23 + triples(s, True)] += 1
    else:
        if P and pairs.founded:
            np.add.at(out, pairs.many(np.stack(P)), 1)
        if T and triples.founded:
            np.add.at(out, K23 + triples.many(np.stack(T)), 1)
    return out


cortex = pickle.load(open(f"out/bench/cortex_ph_d_s{SEED}.pkl", "rb"))
vrelay, cortex.relay = cortex.relay, None


def see(ids):
    out = np.empty((len(ids), vrelay.Z.shape[1]), np.float32)
    for a in range(0, len(ids), 500):
        X = np.stack([cortex.whole(np.asarray(thumbs[row_of[i]], np.float32) / 255, 4) for i in ids[a: a + 500]])
        out[a: a + 500] = unit((X - vrelay.mu) @ vrelay.Z)
    return out


V_life, V_test = see([u["image"] for u in life_u]), see([u["image"] for u in test_u])
del cortex
say("the photographs' codes")
rec = lambda r: " ".join(f"R@{k} {np.mean(r < k):.3f}" for k in (1, 5, 10))  # noqa: E731


def read(M, mean, name):
    Bt = np.stack([bag(u, False) for u in test_u])
    S = unit(unit(Bt - mean) @ M) @ V_test.T
    r_cap, r_img = (S > np.diag(S)[:, None]).sum(1), (S > np.diag(S)[None, :]).sum(0)
    say(f"{name}: photograph -> its caption among {len(test_u)}: {rec(r_img)}; caption -> its photograph: {rec(r_cap)}; median rank {np.median(r_img) + 1:.0f} / {np.median(r_cap) + 1:.0f}")
    return Bt


M, M_other = np.zeros((2 * K23, V_life.shape[1]), np.float32), np.zeros((2 * K23, V_life.shape[1]), np.float32)
mean, heard = np.zeros(2 * K23, np.float32), np.zeros((len(life_u), 2 * K23), np.float32)
other = rng.permutation(len(life_u))  # for the control link: each bag tied to another caption's photograph
for n, u in enumerate(life_u, 1):
    b = bag(u, True)
    heard[n - 1] = b
    mean += (b - mean) / n
    x = unit(b - mean)
    M += x[:, None] * V_life[n - 1][None, :]
    M_other += x[:, None] * V_life[other[n - 1]][None, :]
    if n in CHECKS or n == len(life_u):
        read(M, mean, f"after {n} captions (kinds founded {pairs.founded} + {triples.founded})")
Bt = read(M_other, mean, "the bags tied to other photographs (the control)")
pickle.dump({"layer": layer, "relay": relay, "pairs": (pairs.U, pairs.wins, pairs.founded), "triples": (triples.U, triples.wins, triples.founded), "M": M, "mean": mean},
            open(f"out/bench/grounded_online_ear_s{SEED}_r{READER}.pkl", "wb"))

# a measure only: the caption's written words from its bag (ridge from the bags as they were heard, to the words)
vocab = {}
for u in life_u:
    for w in set(u["text"].split()):
        vocab[w] = vocab.get(w, 0) + 1
vocab = {w: k for k, w in enumerate(sorted(w for w, c in vocab.items() if c >= 5))}


def words(us):
    W = np.zeros((len(us), len(vocab)), np.float32)
    for i, u in enumerate(us):
        for w in u["text"].split():
            if w in vocab:
                W[i, vocab[w]] += 1
    return W


Wa, Wt = words(life_u), words(test_u)
mw, Bc = Wa.mean(0), unit(heard - mean)
G = (Bc.T @ Bc).astype(np.float64)
P = unit(Bt - mean) @ np.linalg.solve(G + 1.0 * (np.trace(G) / len(G)) * np.eye(len(G)), Bc.T @ (Wa - mw)).astype(np.float32)
Sw = unit(P) @ unit(Wt - mw).T
r_w = (Sw > np.diag(Sw)[:, None]).sum(1)
nt = (Wt > 0).sum(0)
cor = lambda lo, hi: np.median([np.corrcoef(P[:, k], Wt[:, k] > 0)[0, 1] for k in np.nonzero((nt >= lo) & (nt < hi))[0]])  # noqa: E731
say(f"the caption's own written words from its bag (bags as heard along the life, kinds as they ended): {rec(r_w)}, median rank {np.median(r_w) + 1:.0f}; "
    f"words heard (correlation median), said in 30+ / 15-29 / 8-14 / 4-7 of the test captions: {cor(30, 10**6):.2f} / {cor(15, 30):.2f} / {cor(8, 15):.2f} / {cor(4, 8):.2f}")
