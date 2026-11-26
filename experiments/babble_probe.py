"""Babbling teaches the mouth what it sounds like; then a heard vowel draws its own gesture (docs/ASYNC.md; E 32).

The stream brain with a vocal tract (organic/vocal.py): 32 gestures, each a place in the vowel space (F1, F2) drawn
at random, the child's own pitch (280 Hz). Phase 1, babbling in silence: the mouth's race fires gestures (about two a
second), their sound reaches the ear with the world's (here: none), and Hebb's rule keeps for each gesture the mean
hearing code while it sounded (the auditory-motor map; DIVA: Guenther, Ghosh & Tourville 2006). Phase 2: vowels in
another voice (140 Hz, an adult's), at random places of the vowel space, each followed by silence; the first gesture
made after each onset is scored by how near it is to the heard vowel in the vowel space (its rank among the 32
gestures; 1 is the nearest; chance 16.5), against the same brain with imitation switched off (habit alone).
The hearing cortex is the eye's rule developed on speech (SpokenCOCO's validation recordings), 8 band stretches.

    BABBLE=300 TRIALS=64 python experiments/babble_probe.py
"""
import json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cochlea import Cochlea  # noqa: E402
from organic.cortex2 import Layer2  # noqa: E402
from organic.stream import StreamBrain  # noqa: E402
from organic.vocal import VocalTract  # noqa: E402

BABBLE, TRIALS, SEED = float(os.environ.get("BABBLE", 300)), int(os.environ.get("TRIALS", 64)), int(os.environ.get("SEED", 0))
rng = np.random.default_rng(SEED)
DIR = os.path.expanduser("~/datasets/spokencoco")
utts = [u for u in map(json.loads, open(os.path.join(DIR, "cochlea.jsonl"))) if u["split"] == "val" and u["rows"] >= 120]
grams = np.memmap(os.path.join(DIR, "cochlea_64.f16"), np.float16, "r").reshape(-1, 64)
hearing = Layer2(1, 128, 8, 1, seed=SEED)
clip = lambda u: (lambda s: np.asarray(grams[s: s + 100], np.float32)[..., None])(u["start"] + int(rng.integers(0, u["rows"] - 100)))  # noqa: E731
hearing.settle((clip(utts[rng.integers(len(utts))]) for _ in range(800)), rng)
for n in range(2000):
    hearing.learn(clip(utts[rng.integers(len(utts))]), rng, rate=0.05 * (1 - n / 2000) + 0.005)
cortex = pickle.load(open(f"out/bench/cortex_ph_d_s{SEED}.pkl", "rb"))
VOICES = [float(v) for v in os.environ.get("VOICES", "280,140").split(",")]
ear = Cochlea()
silence = np.zeros(64, np.float32)
print(f"hearing developed on 2,000 seconds of speech", flush=True)


def vowels(n):
    f1 = rng.uniform(250, 850, n)
    return np.stack([f1, np.maximum(rng.uniform(700, 2500, n), f1 + 300)], 1)


heard = vowels(TRIALS)
babbled = {}
for voice in VOICES:
  adult = VocalTract(f0=voice, seed=SEED + 1)
  for imitate in (True, False):
    t0 = time.process_time()
    brain = StreamBrain(cortex, hearing, seed=SEED, tract=VocalTract(f0=280.0, seed=SEED), bands=8, habit_step=0.0)
    made = 0
    for _ in range(int(BABBLE * 100)):
        made += len(brain.step(None, silence))
    coverage = int((brain.amap_n > 0).sum())
    brain.imitate = imitate
    if imitate:  # the map itself: a vowel said at each gesture's own place, in the heard voice: which gesture's learned sound is nearest?
        def code_of(rows):
            cs = []
            for i in range(len(rows) - 7):
                r = hearing.respond(np.stack(rows[i: i + 8])[..., None], pooled=False)
                b = r.shape[1] // 8
                v = r[0, : 8 * b].reshape(8, b, -1).mean(axis=1).ravel()
                cs.append(v / (np.linalg.norm(v) + 1e-8))
            v = np.mean(cs, 0)
            return v / (np.linalg.norm(v) + 1e-8)
        known = brain.amap_n > 0
        A = brain.amap / (np.linalg.norm(brain.amap, axis=1, keepdims=True) + 1e-8)
        mr, own = [], 0
        for u, (f1, f2) in enumerate(brain.gestures):
            if not known[u]:
                continue
            c = A @ code_of(list(ear.hear(adult.say(f1, f2, 200))))
            c[~known] = -np.inf
            k = int(np.argmax(c))
            own += k == u
            d = np.hypot(*(np.log(brain.gestures) - np.log([f1, f2])).T)
            mr.append(int((d < d[k]).sum()) + 1)
        print(f"voice {voice:.0f} Hz: the map picks the gesture itself for {own} of {known.sum()} (chance 1 each); rank of its pick by nearness: mean {np.mean(mr):.1f} (chance 16.5)", flush=True)
    ranks = []
    for f1, f2 in heard:  # the heard voice
        rows = list(ear.hear(adult.say(f1, f2, 200))) + [silence] * 80
        first = None
        for r in rows:
            for e in brain.step(None, r):
                if e[0] == "sound" and first is None:
                    first = e[1]
        if first is not None:
            d = np.hypot(*(np.log(brain.gestures) - np.log([f1, f2])).T)
            ranks.append(int((d < d[first]).sum()) + 1)
    ranks = np.array(ranks)
    print(f"voice {voice:.0f} Hz, imitation {'on ' if imitate else 'off'}: babbled {made} gestures in {BABBLE:.0f} s ({coverage} of 32 heard at least once); "
          f"{len(ranks)} of {TRIALS} heard vowels answered within a second; rank of the gesture made among 32 by nearness to the vowel heard: "
          f"mean {ranks.mean():.1f} (chance 16.5), nearest three {np.mean(ranks <= 3):.2f} (chance 0.09) | CPU {time.process_time() - t0:.0f}s", flush=True)
