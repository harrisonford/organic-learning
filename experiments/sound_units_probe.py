"""An ear that hears sound and finds its own units (queue E 32; docs/VISION.md 2026-11-26 12:50 and 14:40).

Kris, 2026-11-26: no transcription inside the brain; the ear develops its own cortex. Material: people reading COCO
captions aloud (SpokenCOCO, CC BY-SA), through organic/cochlea.py (experiments/world/spokencoco_cochlea.py). The
transcripts are read only at the end, to measure.

  1 hearing cortex  the eye's layer (organic/cortex2.Layer2) on the cochlea's output read as a picture: windows of 8
                    rows (80 ms) x 8 bands, decorrelated, 128 kinds, winner-take-all Hebbian with a conscience,
                    developed on DEV one-second clips with the eye's schedule. A frame's code: each kind's answer
                    averaged over 4 stretches of the bands (512 numbers), through its own relay, at unit length.
  2 prediction      a linear predictor of the next frame's code from the last three, learned online by the delta
                    rule (its change averaged over an utterance's frames; the step 1 / (1 + n / 5) with n the
                    utterances heard, the word cells' law); its error is one minus the cosine.
  3 units           silence (a frame under a tenth of the utterance's loudest) ends a unit, and so does a peak of the
                    prediction's error (the largest within 50 ms either side: a syllable lasts 100 ms or more). A
                    stretch's code is its two halves' mean codes (order kept). It goes to the unit it matches best
                    if the match passes the vigilance, which then moves towards it (step 1 / (1 + uses / 5)); else it
                    founds a new unit (at the cap, the least used unit makes room). Three vigilances side by side:
                    the number is ours, so its effect is shown.
Measured on utterances never heard while learning, by their transcripts:
    stretches per word and per syllable (syllables counted as vowel groups in the written words)
    word-like units: a unit heard 20+ times whose commonest word comes with it in half its utterances or more,
                     where that word is in a fifth of the utterances or fewer; the same count with the units'
                     utterances shuffled is the chance level

    PART=val DEV=16000 python experiments/sound_units_probe.py      -> printed; out/bench/sound_units_*.log by the queue
"""
import json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cortex2 import Layer2, Relay  # noqa: E402

DIR = os.path.expanduser("~/datasets/spokencoco")
PART, DEV, SEED = os.environ.get("PART", "val"), int(os.environ.get("DEV", 16000)), int(os.environ.get("SEED", 0))
HELD, CAP, VIG = 0.2, 2000, [float(v) for v in os.environ.get("VIGILANCE", "0.65,0.75,0.85").split(",")]
KINDS, WINDOW, BANDS4, CONTEXT, HALF = 128, 8, 4, 3, 5
t0 = time.time()


def say(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == PART and u["rows"] >= 50]
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
rng = np.random.default_rng(SEED)
utts = [utts[i] for i in rng.permutation(len(utts))][: int(os.environ.get("MAXU", 0)) or None]  # MAXU: a smoke test
cut = int(len(utts) * (1 - HELD))
learn_u, test_u = utts[:cut], utts[cut:]
say(f"{len(utts)} utterances ({len(learn_u)} to learn from, {len(test_u)} held out), {len({u['speaker'] for u in utts})} speakers")
gram = lambda u: np.asarray(grams[u["start"]: u["start"] + u["rows"]], np.float32)  # noqa: E731

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
    """The utterance's frame codes (before the relay) and each frame's loudness."""
    g = gram(u)
    r = layer.respond(g[..., None], pooled=False)  # (time, bands, kinds)
    b = r.shape[1] // BANDS4
    codes = r[:, : BANDS4 * b].reshape(r.shape[0], BANDS4, b, KINDS).mean(axis=2).reshape(r.shape[0], -1)
    loud = g.sum(axis=1)[: len(codes)]
    return codes, loud


relay = Relay(0.0005).settle(np.concatenate([unit(frames(learn_u[i])[0])[::5] for i in rng.choice(len(learn_u), min(400, len(learn_u)), replace=False)]))
code = lambda u: (lambda c, l: (unit((unit(c) - relay.mu) @ relay.Z), l))(*frames(u))  # noqa: E731
say("relay settled")

# 2 prediction and 3 units, learned online over the utterances in order
D = KINDS * BANDS4
P = np.zeros((D, CONTEXT * D), np.float32)
seen = 0
lexicons = {v: {"U": np.zeros((CAP, 2 * D), np.float32), "uses": np.zeros(CAP), "n": 0} for v in VIG}


def stretches(C, loud):
    """Prediction errors, then the stretches between silences and error peaks."""
    global P, seen
    err = np.zeros(len(C))
    if len(C) > CONTEXT:
        X = np.stack([C[t - CONTEXT: t].ravel() for t in range(CONTEXT, len(C))])
        Y, Q = C[CONTEXT:], X @ P.T
        err[CONTEXT:] = 1 - (Q * Y).sum(1) / (np.linalg.norm(Q, axis=1) + 1e-8)
        if learning:  # the delta rule, its change averaged over the utterance's frames; the step falls with utterances heard
            seen += 1
            P += (1.0 / (1 + seen / 5)) * ((Y - Q).T @ X) / (len(X) * CONTEXT)
    quiet = loud < 0.1 * loud.max()
    peaks = [t for t in range(CONTEXT + 1, len(C) - 1) if err[t] == err[max(0, t - HALF): t + HALF + 1].max() and err[t] > err[t - 1]]
    cuts = sorted(set(peaks) | {t for t in range(1, len(C)) if quiet[t] != quiet[t - 1]} | {0, len(C)})
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= 3 and not quiet[a:b].mean() > 0.5:
            h = (a + b) // 2
            out.append(np.concatenate([unit(C[a:h].mean(0)), unit(C[h:b].mean(0))]) / np.sqrt(2))
    return out


def assign(lex, s, vig, grow):
    n = lex["n"]
    if n:
        m = lex["U"][:n] @ s
        k = int(np.argmax(m))
        if m[k] >= vig or not grow:
            if grow:
                lex["uses"][k] += 1
                w = lex["U"][k] + (1.0 / (1 + lex["uses"][k] / 5)) * (s - lex["U"][k])
                lex["U"][k] = w / (np.linalg.norm(w) + 1e-8)
            return k
    if n < CAP:
        k = n
        lex["n"] += 1
    else:
        k = int(np.argmin(lex["uses"]))
    lex["U"][k], lex["uses"][k] = s, 1
    return k


learning = True
for i, u in enumerate(learn_u):
    for s in stretches(*code(u)):
        for v, lex in lexicons.items():
            assign(lex, s, v, True)
    if (i + 1) % 5000 == 0:
        say(f"{i + 1} utterances heard; units: " + ", ".join(f"vigilance {v}: {lex['n']}" for v, lex in lexicons.items()))

# measured on the held-out utterances
learning = False
words_of = lambda u: set(re.findall(r"[a-z']+", u["text"].lower()))  # noqa: E731
syll = lambda w: max(1, len(re.findall(r"[aeiouy]+", w)))  # noqa: E731
heard = {v: [] for v in VIG}
n_str = n_words = n_syl = 0
for u in test_u:
    ss = stretches(*code(u))
    n_str += len(ss)
    ws = re.findall(r"[a-z']+", u["text"].lower())
    n_words += len(ws)
    n_syl += sum(syll(w) for w in ws)
    for v, lex in lexicons.items():
        heard[v].append([assign(lex, s, v, False) for s in ss])
say(f"held out: {n_str / len(test_u):.1f} stretches an utterance; {n_str / n_words:.2f} a word, {n_str / n_syl:.2f} a syllable")
W = [words_of(u) for u in test_u]
base = {}
for ws in W:
    for w in ws:
        base[w] = base.get(w, 0) + 1


def wordlike(units_per_utt):
    occ = {}
    for j, us in enumerate(units_per_utt):
        for k in set(us):
            occ.setdefault(k, []).append(j)
    found = []
    for k, js in occ.items():
        if len(js) < 20:
            continue
        c = {}
        for j in js:
            for w in W[j]:
                c[w] = c.get(w, 0) + 1
        w = max(c, key=lambda x: c[x] / len(js) - base[x] / len(W))
        if c[w] / len(js) >= 0.5 and base[w] / len(W) <= 0.2:
            found.append((w, c[w] / len(js), base[w] / len(W), len(js)))
    return found


for v in VIG:
    found = wordlike(heard[v])
    shuf = [list(rng.permutation(np.concatenate([np.array(us, int) for us in heard[v] if us]))) for _ in range(1)][0]
    sizes, it, sh = [len(us) for us in heard[v]], 0, []
    for s_ in sizes:
        sh.append(shuf[it: it + s_])
        it += s_
    chance = wordlike(sh)
    found.sort(key=lambda f: -f[1])
    say(f"vigilance {v}: {lexicons[v]['n']} units; word-like {len(found)} (shuffled: {len(chance)}): "
        + "; ".join(f"{w} {p:.2f} (base {b:.2f}, {n}x)" for w, p, b, n in found[:25]))
