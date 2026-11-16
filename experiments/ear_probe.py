"""Ears compared on a life's diet (docs/EAR_LEXICON.md): how they cut speech, and whether a word is the same unit everywhere.

A fresh ear of each kind hears what a small life hears up to two words: questions and the adults' answers (world 2),
one unasked comment and one overheard sentence per moment. Then, on held-out utterances:
  the standard segmentation scores (experiments/segmentation_probe.py), and
  the same word in six settings ("dog", "is there a dog?", "look, a dog!", "say dog", ...): in how many is one of the
  units exactly the word.
The grown ears are saved to out/tmp/ear_<kind>.pkl for the word cells' probe (EAR=... experiments/imagery_probe.py).
    KINDS=first,lexicon python experiments/ear_probe.py
"""
import os, pickle, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from organic.curriculum import Curriculum, PICTURE_BOOK
from organic.ear import Ear
from organic.ear_lexicon import LexiconEar
from organic.ear_carving import CarvingEar, EdgeEar
from experiments.segmentation_probe import score, show

KINDS = os.environ.get("KINDS", "first,lexicon").split(",")
DIET = [("BB", int(os.environ.get("BB", 5000))), ("B", int(os.environ.get("B", 8000))), ("W1", int(os.environ.get("W1", 12000))), ("W2", int(os.environ.get("W2", 10000)))]
SETTINGS = ["{w}", "is there a {w}?", "look, a {w}!", "what does the {w} say?", "say {w}", "can you say {w} for me"]
make = {"first": lambda: Ear(seed=7), "lexicon": lambda: LexiconEar(seed=7), "carving": lambda: CarvingEar(seed=7), "edge": lambda: EdgeEar(seed=7)}


def diet(seed=0):
    cur = Curriculum(seed=0, world=2)
    cur.rng.seed(seed + 5)
    wiki = open(os.path.expanduser("~/datasets/simplewiki/sentences.txt"))
    for stage, n in DIET:
        for _ in range(n):
            if stage != "BB":
                m = cur.moment(stage)
                yield m["question"]
                yield m["answer"]
            yield cur.observation(stage if stage != "BB" else "W1")["says"]
            yield (wiki.readline() or "the end").strip()


test = Curriculum(seed=0, world=2)
test.rng.seed(4242)
held = [test.moment(s)["question"] for _ in range(300) for s in ("B", "W1", "W2", "P")]
names = sorted(PICTURE_BOOK)[:60]
for kind in KINDS:
    ear, t0 = make[kind](), time.time()
    marks = {5000 * 2: "after babbling"}
    for n, text in enumerate(diet()):
        ear.units(text)
    r = score(ear, held)
    show(f"{kind} ({ear.utterances} heard, {time.time() - t0:.0f}s)", r)
    print("   most fused:", [(u.decode(errors="replace"), n) for u, n in r["fused"]])
    found = np.mean([[w.encode() in ear.units(s.format(w=w), learn=False) for s in SETTINGS] for w in names], axis=0)
    print("   the word is one of the units: " + " | ".join(f"{s.format(w='W')!r} {f:.2f}" for s, f in zip(SETTINGS, found)) + f" | all six {np.mean([all(w.encode() in ear.units(s.format(w=w), learn=False) for s in SETTINGS) for w in names]):.2f}")
    for t in ["is there a dog?", "what does the cow say?", "can you say tulip for me", "please say alcohol again", "how are you?", "do you have a cat?", "what color is the bus?", "look, a rocket!", "the dog is big"]:
        print("     ", " | ".join(u.decode(errors="replace") for u in ear.units(t, learn=False)))
    print(f"   units known {len(ear.heard)}, the 12 most heard: {[(u.decode(errors='replace'), n) for u, n in sorted(ear.heard.items(), key=lambda kv: -kv[1])[:12]]}", flush=True)
    if hasattr(ear, "edge"):
        p = lambda k, b: ear.edge[k, b] / max(1.0, ear.edge[k, b] + ear.inside[k, b])  # noqa: E731
        print("   an edge follows / comes before: " + ", ".join(f"{chr(b)!r} {p(0, b):.2f}/{p(1, b):.2f}" for b in b" ?!.,'aestdn"), flush=True)
    pickle.dump(ear, open(f"out/tmp/ear_{kind}.pkl", "wb"))
