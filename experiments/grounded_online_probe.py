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
# (2026-11-27 19:42) LIVE (seconds; 0: off, each caption known whole as before): nothing is known of a caption before it
# is heard. Each band is set against a running mean and spread of what was heard so far, and "quiet" is under a
# tenth of the loudest moment so far, all fading over LIVE seconds; the life's captions are one stream in the order
# heard, and a test caption is heard by an ear that has just heard the first FIRST captions (VISION.md 19:42).
LIVE = float(os.environ.get("LIVE", 0))
# (2026-11-27 21:04) RELAY=circuit: the hearing relay is the circuit of interneurons with its local rule
# (organic/cortex2.py RelayCircuit, the eye's rate; its synapses change once a caption, not once every 50 views) in
# place of the matrix settled in one batch:
# born random, it learns from the first FIRST captions' frames before the life and goes on learning from every
# caption of the life; held still when the test captions are heard (VISION.md 21:05).
RELAY = os.environ.get("RELAY", "batch")
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


started = {}  # LIVE: what the ear holds of the sound so far when a caption begins (mean and spread a band, the loudest moment)


def hear(g, state):
    """The bands of g against what was heard before each frame and the frame itself; the state after."""
    from scipy.signal import lfilter
    k = 0.01 / LIVE
    mu0, var0 = (g[0], np.ones(g.shape[1], np.float32)) if state is None else state[:2]
    mu = lfilter([k], [1, k - 1], g, axis=0, zi=((1 - k) * mu0)[None])[0]
    var = lfilter([k], [1, k - 1], (g - mu) ** 2, axis=0, zi=((1 - k) * var0)[None])[0]
    return ((g - mu) / (np.sqrt(var) + 1e-3)).astype(np.float32), (mu[-1].astype(np.float32), var[-1].astype(np.float32))


def loudest(env, top):
    """The loudest moment so far at each frame of env, fading; the last."""
    k, ref = 0.01 / LIVE, np.empty_like(env)
    for t, e in enumerate(env):
        top = max(float(e), top * (1 - k))
        ref[t] = top
    return ref, top


def envelope(u):
    loud = np.convolve(raw(u).sum(axis=1), np.ones(WINDOW) / WINDOW, "valid")
    return np.convolve(loud, np.ones(SMOOTH) / SMOOTH, "same")


if LIVE:
    state, states = None, []
    for u in life_u:
        states.append(state)
        started[u["uttid"]] = state
        _, (mu, var) = hear(raw(u), state)
        state = (mu, var, loudest(envelope(u), 0.0 if state is None else state[2])[1])
    states.append(state)
    for u in test_u:
        started[u["uttid"]] = states[min(FIRST, len(life_u))]
    say(f"live ear: every band against the sound so far, fading over {LIVE:g} s; nothing of a caption known before it is heard")


def gram(u):
    g = raw(u)
    if LIVE:
        return hear(g, started[u["uttid"]])[0]
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


if RELAY == "circuit":
    from organic.cortex2 import RelayCircuit  # noqa: E402
    relay = RelayCircuit(KINDS * BANDS, rate=1e-4, block=50, seed=SEED)

    def through(c, learning):
        X = unit(c)
        C = unit(relay.code(X))  # the caption's frames with the synapses as they were when it began
        if learning:
            relay.learn(X)  # one block a caption (about 330 frames; blocks of 50 took 2 s a caption, the settled state worked out anew each block)
        return C

    for u in first:
        through(frames(u)[0], True)
else:
    relay = Relay(0.0005).settle(np.concatenate([unit(frames(first[i])[0])[::5] for i in rng.choice(len(first), min(400, len(first)), replace=False)]))

    def through(c, learning):
        return unit((unit(c) - relay.mu) @ relay.Z)
say(f"hearing cortex and its relay, from the first {len(first)} captions' sound")


def stretches(u, learning=False):
    c, loud = frames(u)
    C = through(c, learning)
    env = np.convolve(loud, np.ones(SMOOTH) / SMOOTH, "same")
    quiet = env < 0.1 * (loudest(env, 0.0 if started[u["uttid"]] is None else started[u["uttid"]][2])[0] if LIVE else env.max())
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
        s = np.asarray(s, np.float32)  # (2026-11-27 23:41) at double precision every match copied the kinds: 65 ms a kind, most of a run
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
        return np.argmax(self.U[: self.founded] @ np.asarray(S, np.float32).T, axis=0)


D = 2 * KINDS * BANDS
pairs, triples = Kinds(K23, 2 * D), Kinds(K23, 3 * D)


def bag(u, learning):
    ss = stretches(u, learning)
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
        if RELAY == "circuit":  # (2026-11-28) is the circuit still settling? its evenness on held-out sound along the life
            Ce = np.concatenate([through(frames(u)[0], False)[::5] for u in test_u[:200]])
            de = np.clip(np.linalg.eigvalsh(np.cov(Ce.T.astype(np.float64))), 0, None)
            say(f"the circuit's code on held-out sound after {n} captions: participation ratio {de.sum() ** 2 / (de * de).sum():.0f} of {Ce.shape[1]}")
Bt = read(M_other, mean, "the bags tied to other photographs (the control)")
pickle.dump({"layer": layer, "relay": relay, "pairs": (pairs.U, pairs.wins, pairs.founded), "triples": (triples.U, triples.wins, triples.founded), "M": M, "mean": mean},
            open(f"out/bench/grounded_online_ear_s{SEED}_r{READER}{f'_live{LIVE:g}' if LIVE else ''}{'_circuit' if RELAY == 'circuit' else ''}.pkl", "wb"))

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
# how evenly the relay's code uses its directions on sound never heard (participation ratio; the batch relays gave 570-643 of 1,024, VISION.md 18:34)
Ct = np.concatenate([through(frames(u)[0], False)[::5] for u in test_u[:300]])
d = np.clip(np.linalg.eigvalsh(np.cov(Ct.T.astype(np.float64))), 0, None)
say(f"the hearing relay ({RELAY}) on held-out sound: participation ratio {d.sum() ** 2 / (d * d).sum():.0f} of {Ct.shape[1]}")
