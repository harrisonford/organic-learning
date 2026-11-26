"""Does babbling drift toward the vowels of the language heard? (docs/ASYNC.md; queue E 32)

Infants' babbling comes to resemble the sounds of the language around them before they speak (de Boysson-Bardies
and colleagues, 1980s; to verify). The stream brain (organic/stream.py) babbles with its vocal tract while an adult
(another voice, 140 Hz) says vowels of one small language every 1.5 s or so; its habits grow from what is heard
(`heard_habit`: a heard sound strengthens the habits of the gestures whose learned sound matches it; nothing grows
from its own repetition). Two languages of three vowels each, two brains raised alike but on one each, and a brain in
silence. Measured in the last third of the time: how near the babbled gestures are (in the vowel space, log
formants) to the nearest vowel of each language. If babbling drifts, each brain babbles nearer its own language.

    SECONDS=600 python experiments/babble_drift_probe.py
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.cochlea import Cochlea  # noqa: E402
from organic.cortex2 import Layer2  # noqa: E402
from organic.stream import StreamBrain  # noqa: E402
from organic.vocal import VocalTract  # noqa: E402

SECONDS, SEED, HEARD = float(os.environ.get("SECONDS", 600)), int(os.environ.get("SEED", 0)), float(os.environ.get("HEARD_HABIT", 0.01))
LANGS = {"i a u": [(300, 2300), (750, 1250), (350, 800)], "e o y": [(500, 1900), (550, 950), (300, 1700)]}
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
ear, adult, silence = Cochlea(), VocalTract(f0=140.0, seed=SEED + 1), np.zeros(64, np.float32)
said = {name: [list(ear.hear(adult.say(f1, f2, 200))) for f1, f2 in vs] for name, vs in LANGS.items()}


def nearness(gestures, made, vowels):
    """Mean distance (log formants) from each gesture made to the nearest vowel of a language."""
    g = np.log(gestures[made])
    v = np.log(np.array(vowels, float))
    return float(np.mean(np.min(np.hypot(g[:, None, 0] - v[None, :, 0], g[:, None, 1] - v[None, :, 1]), axis=1)))


for world in ["silence"] + list(LANGS):
    brain = StreamBrain(cortex, hearing, seed=SEED, tract=VocalTract(f0=280.0, seed=SEED), bands=8, habit_step=0.0, heard_habit=HEARD)
    queue, made = [], []
    for s in range(int(SECONDS * 100)):
        if world != "silence" and not queue and rng.random() < 0.01 * 100 / 150:  # a vowel every 1.5 s or so
            queue = list(said[world][rng.integers(3)])
        row = queue.pop(0) if queue else silence
        for e in brain.step(None, row):
            if e[0] == "sound" and s >= SECONDS * 100 * 2 / 3:
                made.append(e[1])
    made = np.array(made, int)
    near = {name: nearness(brain.gestures, made, vs) for name, vs in LANGS.items()}
    every = {name: nearness(brain.gestures, np.arange(len(brain.gestures)), vs) for name, vs in LANGS.items()}  # babbling evenly
    print(f"raised on {world:8}: {len(made)} gestures in the last third; distance to the nearest vowel of 'i a u' {near['i a u']:.3f} "
          f"(even babbling {every['i a u']:.3f}), of 'e o y' {near['e o y']:.3f} (even {every['e o y']:.3f}); habits' spread: top gesture "
          f"{brain.habit.max() / brain.habit.sum():.3f} of all", flush=True)
