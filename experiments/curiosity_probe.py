"""Gates as measurements, curiosity as the driver: does an order of learning emerge by itself? (offline)

docs/REVIEW_2026-11-10.md 4.6. In every life so far the curriculum holds the brain at a stage until it
passes a test. Infants are not held: they attend to what they are making progress on (Oudeyer's learning
progress; Twomey & Westermann 2018; Ten et al. 2021). Here the world is not staged at all: at every
moment it offers a few moments of any kind, first words to kindergarten, and the word cells
(organic/words.py) attend to one of them.

How the brain feels a moment, from the inside: its surprise at the reply (how far its content dendrites
were from expecting the units that were then heard). Two traces predict that surprise from the echo of
the utterance just heard, one fast and one slow (the same delta rule, two time scales). Where the fast
one is below the slow one, surprise is falling for this kind of utterance: that is learning progress,
and it is what attention follows. Nothing tells the brain which kind a moment is.

  curious    attends to the offered moment with the most learning progress (one time in five, to any: attention wanders)
  uniform    attends to any offered moment (no preference)
  staged     the curriculum as a schedule: first words, then two words, then preschool, then kindergarten
             (with review of earlier ones), the same number of moments
  novelty    attends to the moment it expects to be most surprised by (hardest first)

Every BLOCK moments: held-out scores per stage (the yardstick, as the curriculum grades them), and
where attention went. If an order of stages shows up in the curious arm, nobody put it there.
    ARM=curious python experiments/curiosity_probe.py [moments, default 60000]
"""
import json, os, pickle, random, sys, time, zlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade, tidy  # noqa: E402
from organic.words import Words, SILENCE  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
ARM = os.environ.get("ARM", "curious")
SEED = int(os.environ.get("SEED", 0))
BLOCK = int(os.environ.get("BLOCK", 5000))
OFFERED = 4
STAGES = ["W1", "W2", "P", "K"]
FAST, SLOW = 0.1, 0.01  # the two time scales of the surprise traces
cur = Curriculum(seed=0)
ear = pickle.load(open(os.environ.get("EAR", "out/kid_s3.ear.pkl"), "rb"))
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
VC = "out/tmp/vis_cache.pkl"
V = pickle.load(open(VC, "rb")) if os.path.exists(VC) else {}


def vis(ref):
    if ref not in V:
        img = cur.image(ref)
        if img.max() == 0:
            V[ref] = None
        else:
            v = np.concatenate(cortex.code(img, grid=4))
            V[ref] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
    return V[ref]


hear = lambda q: ear.units(q, learn=False)  # noqa: E731
rng = random.Random(SEED)
pools = {s: cur.train[s] for s in STAGES}
tests = {s: [it for src in sorted(cur.test[s]) for it in cur.test[s][src][: max(1, 200 // len(cur.test[s]))]] for s in STAGES}
words = Words(7168, ear, max_units=3000, max_cues=8192)
w_fast, w_slow = np.zeros(ear.width, np.float32), np.zeros(ear.width, np.float32)


def offer(stage=None):
    s = stage or rng.choice(STAGES)
    src = rng.choice(sorted(pools[s]))
    it = rng.choice(pools[s][src])
    return (it[0], cur.wording(it, rng), it[2][0], s, src)


def surprise(heard, v, reply):
    """How far the content dendrites were from expecting the units that were then heard (0 = fully expected)."""
    cues = [c for c in (words._cue(u, False) for u in set(heard)) if c is not None]
    o = words._content(words._visual(v), cues) + words.base[: len(words.units)]
    got = [float(np.clip(o[words.index[u]], 0, 1)) if u in words.index else 0.0 for u in reply]
    return 1.0 - float(np.mean(got)) if got else 1.0


def score():
    out = {}
    for s in STAGES:
        g = [grade(s, words.recall(hear(it[1]), vis(it[0]))[0] or "", it[2]) for it in tests[s]]
        out[s] = float(np.mean(g))
    return out


print(f"arm {ARM}, seed {SEED}; held out per stage: " + ", ".join(f"{s} {len(tests[s])}" for s in STAGES), flush=True)
attended, t0, log = {s: 0 for s in STAGES}, time.time(), []
for n in range(1, N + 1):
    if ARM == "staged":  # a schedule: a quarter of the life per stage, 30% review of earlier stages
        k = min(len(STAGES) - 1, (n - 1) * len(STAGES) // N)
        if k and rng.random() < 0.3:
            k = rng.randrange(k)
        m = offer(STAGES[k])
    else:
        cands = [offer() for _ in range(OFFERED)]
        if ARM == "uniform" or rng.random() < 0.2:
            m = cands[0]
        else:
            echoes = [ear.question(hear(c[1]) + [SILENCE], adapt=False) for c in cands]
            if ARM == "curious":
                interest = [float(w_slow @ e - w_fast @ e) for e in echoes]
            else:  # novelty
                interest = [float(w_fast @ e) for e in echoes]
            m = cands[int(np.argmax(interest))]
    ref, q, a, stage, src = m
    heard, v, reply = hear(q), vis(ref), hear(a)
    s = surprise(heard, v, reply)
    e = ear.question(heard + [SILENCE], adapt=False)
    w_fast += FAST * (s - w_fast @ e) * e
    w_slow += SLOW * (s - w_slow @ e) * e
    words.learn(heard, v, reply)
    attended[stage] += 1
    if n % BLOCK == 0:
        sc = score()
        share = {s_: attended[s_] / BLOCK for s_ in STAGES}
        log.append({"moments": n, "score": sc, "attention": share, "cells": len(words.units)})
        print(f"{n:>6} ({time.time() - t0:.0f}s) | held out " + " ".join(f"{s_} {sc[s_]:.2f}" for s_ in STAGES)
              + " | attention " + " ".join(f"{s_} {share[s_]:.2f}" for s_ in STAGES) + f" | {len(words.units)} unit cells", flush=True)
        attended = {s_: 0 for s_ in STAGES}
json.dump(log, open(f"out/curiosity_{ARM}_s{SEED}.json", "w"))
if len(V) > 30000:
    pass
