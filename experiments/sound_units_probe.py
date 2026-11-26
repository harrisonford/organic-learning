"""An ear that hears sound and finds its own units (queue E 32; docs/VISION.md 2026-11-26 12:50 and 14:45).

Kris, 2026-11-26: no transcription inside the brain; the ear develops its own cortex. Material: people reading COCO
captions aloud (SpokenCOCO, CC BY-SA), through organic/cochlea.py (experiments/world/spokencoco_cochlea.py). The
transcripts are read only at the end, to measure.

Three levels, each the rule the brain already uses one level down:
  1 hearing cortex  the eye's layer (organic/cortex2.Layer2) on the cochlea's output read as a picture: windows of 8
                    rows (80 ms) x 8 bands, decorrelated, 128 kinds, winner-take-all Hebbian with a conscience,
                    developed on DEV one-second clips with the eye's schedule. A frame's code: each kind's answer
                    averaged over BANDS stretches of the bands (8: 1,024 numbers; 4 in the first reading), through its own relay, at unit length.
  2 syllables       auditory cortex follows speech's slow loudness rhythm (4-8 a second: the syllables; Giraud &
                    Poeppel 2012): a stretch ends at a trough of the loudness (smoothed over 50 ms; the lowest within
                    50 ms either side) and at silence (under a tenth of the utterance's loudest). A stretch's code is
                    its two halves' mean codes (order kept). The stretches go to KINDS2 syllable kinds by the eye's
                    own rule: the kind that matches best, handicapped by a conscience, moves towards the stretch,
                    its step 1 / (1 + wins / 5) (the consolidation law of the orchestrator's clusters). The first
                    KINDS2 stretches found the kinds.
  3 words           the byte ear's rule one level up (organic/ear.py; Saffran, Aslin & Newport 1996): counts of which
                    kind follows the last one and two kinds (back-off to one when two are new); a word starts where
                    the uncertainty (entropy) of the next kind jumps up, and at every silence. The ear's words are the
                    kind sequences so found.
Measured on utterances never heard while learning (nothing learns then), by their transcripts:
    stretches per syllable (syllables counted as vowel groups in the written words) and words found per word
    word-like syllable kinds and word-like words: heard in 20+ held-out utterances (10+ for words), whose commonest
    written word comes with them in half their utterances or more, where that word is in a fifth of the utterances
    or fewer; the same count with the units shuffled among the utterances is the chance level

    PART=val DEV=16000 python experiments/sound_units_probe.py
    PART=val MAXU=600 DEV=200 python experiments/sound_units_probe.py    (a smoke test)
"""
import json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Layer2, Relay  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
PART, DEV, SEED = os.environ.get("PART", "val"), int(os.environ.get("DEV", 16000)), int(os.environ.get("SEED", 0))
HELD, KINDS2 = 0.2, int(os.environ.get("KINDS2", 512))
KINDS, WINDOW, HALF, SMOOTH = 128, 8, 5, 5
BANDS4 = int(os.environ.get("BANDS", 8))  # stretches of the bands a frame's code keeps apart (4 in the first reading: it blurs the formants)
t0 = time.time()


def say(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == PART and u["rows"] >= 50]
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
rng = np.random.default_rng(SEED)
utts = [utts[i] for i in rng.permutation(len(utts))][: int(os.environ.get("MAXU", 0)) or None]
cut = int(len(utts) * (1 - HELD))
learn_u, test_u = utts[:cut], utts[cut:]
say(f"{len(utts)} utterances ({len(learn_u)} to learn from, {len(test_u)} held out), {len({u['speaker'] for u in utts})} speakers")
ADAPT = os.environ.get("ADAPT", "1") == "1"


def raw(u):
    return np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32)


def gram(u):
    """The utterance's cochleagram; with ADAPT each band taken relative to its own level and spread over the utterance
    (the ear adapts to the voice: Ladefoged & Broadbent 1957; Dean, Harper & McAlpine 2005)."""
    g = raw(u)
    if ADAPT:
        g = (g - g.mean(0)) / (g.std(0) + 1e-3)
    return g



# 1 hearing cortex
layer = Layer2(1, KINDS, WINDOW, 1, seed=SEED)


def clip(u):
    g = gram(u)
    s = int(rng.integers(0, max(1, len(g) - 100)))
    return g[s: s + 100][..., None]


layer.settle((clip(learn_u[i]) for i in rng.choice(len(learn_u), min(800, len(learn_u)), replace=False)), rng)
for n in range(DEV):
    layer.learn(clip(learn_u[rng.integers(len(learn_u))]), rng, rate=0.05 * (1 - n / DEV) + 0.005)
say(f"hearing cortex developed on {DEV} one-second clips; wins min/max {int(layer.wins[0].min())}/{int(layer.wins[0].max())}")


def frames(u):
    """The utterance's frame codes (before the relay) and each frame's loudness (over the same 80 ms)."""
    g = gram(u)
    r = layer.respond(g[..., None], pooled=False)  # (time, bands, kinds)
    b = r.shape[1] // BANDS4
    codes = r[:, : BANDS4 * b].reshape(r.shape[0], BANDS4, b, KINDS).mean(axis=2).reshape(r.shape[0], -1)
    loud = np.convolve(raw(u).sum(axis=1), np.ones(WINDOW) / WINDOW, "valid")[: len(codes)]  # loudness from the cochlea itself
    return codes, loud


relay = Relay(0.0005).settle(np.concatenate([unit(frames(learn_u[i])[0])[::5] for i in rng.choice(len(learn_u), min(400, len(learn_u)), replace=False)]))
say("relay settled")


def stretches(u):
    """Level 2: the utterance cut at loudness troughs and silences; each stretch's code and whether a silence follows."""
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
                out[-1][1] = True  # a silence follows the last stretch
            continue
        h = (a + b) // 2
        out.append([np.concatenate([unit(C[a:h].mean(0)), unit(C[h:b].mean(0))]) / np.sqrt(2), False])
    return out


U, wins, founded = np.zeros((KINDS2, 2 * KINDS * BANDS4), np.float32), np.zeros(KINDS2), 0


class Kinds:
    """Kinds by the eye's rule (winner-take-all with a conscience, step 1 / (1 + wins / 5)); the first n inputs found them."""

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


WIN = int(os.environ.get("KINDS3", 1024))
pairs, triples = Kinds(WIN, 2 * 2 * KINDS * BANDS4), Kinds(WIN, 3 * 2 * KINDS * BANDS4)


def windows(ss, learning):
    """Level 3 by likeness: kinds of two- and three-stretch windows (neighbouring syllables side by side), never across a silence."""
    v = [s for s, _ in ss]
    gap = [g for _, g in ss]
    p2 = [pairs(np.concatenate(v[i: i + 2]) / np.sqrt(2), learning) for i in range(len(v) - 1) if not gap[i]]
    p3 = [triples(np.concatenate(v[i: i + 3]) / np.sqrt(3), learning) for i in range(len(v) - 2) if not (gap[i] or gap[i + 1])]
    return p2, p3
counts1, counts2 = {}, {}  # level 3: next kind after one kind, after two


def kind(s, learning):
    global founded
    if learning and founded < KINDS2:
        U[founded], wins[founded] = s, 1
        founded += 1
        return founded - 1
    c = int(np.argmax(U[:founded] @ s - (0.3 * (wins[:founded] / (wins[:founded].mean() + 1e-8) - 1) if learning else 0)))
    if learning:
        wins[c] += 1
        w = U[c] + (1.0 / (1 + wins[c] / 5)) * (s - U[c])
        U[c] = w / (np.linalg.norm(w) + 1e-8)
    return c


def entropy(d):
    if not d:
        return np.log(KINDS2)
    p = np.array(list(d.values()), float)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def words(ks, gaps, learning):
    """Level 3: the kind sequence cut where the next kind's uncertainty jumps up, and at silences."""
    out, cur, prev_h = [], [], None
    for i, k in enumerate(ks):
        a = ks[i - 1] if i else None
        b = (ks[i - 2], ks[i - 1]) if i > 1 else None
        d = counts2.get(b) if b in counts2 else counts1.get(a, {})
        h = entropy(d)
        if cur and (prev_h is not None and h > prev_h or gaps[i - 1]):
            out.append(tuple(cur))
            cur = []
        cur.append(k)
        prev_h = h
        if learning:
            if a is not None:
                counts1.setdefault(a, {})[k] = counts1.get(a, {}).get(k, 0) + 1
            if b is not None:
                counts2.setdefault(b, {})[k] = counts2.get(b, {}).get(k, 0) + 1
    if cur:
        out.append(tuple(cur))
    return out


for i, u in enumerate(learn_u):
    ss = stretches(u)
    ks = [kind(s, True) for s, _ in ss]
    words(ks, [g for _, g in ss], True)
    windows(ss, True)
    if (i + 1) % 5000 == 0:
        say(f"{i + 1} utterances heard; {founded} syllable kinds, wins min/max {int(wins[:founded].min())}/{int(wins[:founded].max())}")

# measured on the held-out utterances, nothing learning
syll = lambda w: max(1, len(re.findall(r"[aeiouy]+", w)))  # noqa: E731
W, K, WD, K2, K3 = [], [], [], [], []
n_str = n_wd = n_words = n_syl = 0
for u in test_u:
    ss = stretches(u)
    ks = [kind(s, False) for s, _ in ss]
    wd = words(ks, [g for _, g in ss], False)
    p2, p3 = windows(ss, False)
    K2.append(p2)
    K3.append(p3)
    written = re.findall(r"[a-z']+", u["text"].lower())
    W.append(set(written))
    K.append(ks)
    WD.append(wd)
    n_str, n_wd, n_words, n_syl = n_str + len(ss), n_wd + len(wd), n_words + len(written), n_syl + sum(syll(w) for w in written)
say(f"held out: {n_str / n_syl:.2f} stretches a syllable ({n_str / len(test_u):.1f} an utterance); {n_wd / n_words:.2f} words found a word; "
    f"words found {len({w for wd in WD for w in wd})} different, mean length {np.mean([len(w) for wd in WD for w in wd]):.2f} stretches")
base = {}
for ws in W:
    for w in ws:
        base[w] = base.get(w, 0) + 1


def wordlike(per_utt, least):
    occ = {}
    for j, us in enumerate(per_utt):
        for k in set(us):
            occ.setdefault(k, []).append(j)
    found = []
    for k, js in occ.items():
        if len(js) < least:
            continue
        c = {}
        for j in js:
            for w in W[j]:
                c[w] = c.get(w, 0) + 1
        w = max(c, key=lambda x: c[x] / len(js) - base[x] / len(W))
        if c[w] / len(js) >= 0.5 and base[w] / len(W) <= 0.2:
            found.append((w, c[w] / len(js), base[w] / len(W), len(js)))
    return sorted(found, key=lambda f: -f[1])


def shuffled(per_utt):
    flat = [x for us in per_utt for x in us]
    order = rng.permutation(len(flat))
    out, i = [], 0
    for us in per_utt:
        out.append([flat[j] for j in order[i: i + len(us)]])
        i += len(us)
    return out


def lifted(per_utt, least=20):
    """For each kind heard in `least`+ utterances: the written word it lifts most (its share in the kind's utterances
    over its share in all), counted when that word comes with the kind 5+ times. Returns the kinds' top lifts."""
    occ = {}
    for j, us in enumerate(per_utt):
        for k in set(us):
            occ.setdefault(k, []).append(j)
    out = []
    for k, js in occ.items():
        if len(js) < least:
            continue
        c = {}
        for j in js:
            for w in W[j]:
                c[w] = c.get(w, 0) + 1
        best = max((w for w in c if c[w] >= 5), key=lambda w: (c[w] / len(js)) / (base[w] / len(W)), default=None)
        if best:
            out.append(((c[best] / len(js)) / (base[best] / len(W)), best, c[best] / len(js), len(js)))
    return sorted(out, reverse=True)


for name, per_utt, least in (("syllable kinds", K, 20), ("words found (byte-ear rule)", WD, 10), ("two-syllable kinds", K2, 20), ("three-syllable kinds", K3, 20)):
    found, chance = wordlike(per_utt, least), wordlike(shuffled(per_utt), least)
    say(f"{name}: word-like {len(found)} (shuffled: {len(chance)}): " + "; ".join(f"{w} {p:.2f} (base {b:.2f}, {n}x)" for w, p, b, n in found[:30]))
    lf, lc = lifted(per_utt), lifted(shuffled(per_utt))
    if lf:
        say(f"   {name}: {len(lf)} kinds heard in 20+ utterances; lift of their best word: median {np.median([x[0] for x in lf]):.2f} (shuffled {np.median([x[0] for x in lc]) if lc else 0:.2f}); "
            f"lift 3 or more: {sum(x[0] >= 3 for x in lf)} (shuffled {sum(x[0] >= 3 for x in lc)}); top: " + "; ".join(f"{w} x{l:.1f} ({p:.2f}, {n})" for l, w, p, n in lf[:20]))
