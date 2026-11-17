"""Forgetting what is no longer taught, and dendritic branches against it (docs/RESEARCH_MEMORY.md, 2026-11-17).

The first world 3 lives lost their naming of COCO things within 10,000 moments of those lessons stopping, while the
same words went on being taught on other pictures; the loss was in the word cells' synapses (forget_diag.py).
Here the word cells alone live that schedule:

  phase 1   N1 naming moments, the three sources taking turns (CIFAR, Commons, COCO things)
  phase 2   N2 moments without COCO things (the other two go on)

and name held-out pictures of each source after each phase. Arms: the number of dendritic branches a cell has for
what it sees (1: the cells as they are).
    BRANCHES=1,4 SEEDS=0,1,2 CORTEX=data/visual_cortex.pkl python experiments/branch_probe.py
"""
import os, pickle, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, tidy  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
ARMS = [int(x) for x in os.environ.get("BRANCHES", "1,4").split(",")]
N1, N2, GRID = int(os.environ.get("N1", 6000)), int(os.environ.get("N2", 6000)), int(os.environ.get("GRID", 4))
cur = Curriculum(seed=0, world=3)
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
cortex = pickle.load(open(os.environ.get("CORTEX", "data/visual_cortex.pkl"), "rb"))
dim = sum(l.kinds for l in cortex.layers) * GRID * GRID
hear = lambda q: ear.units(q, learn=False)  # noqa: E731
SRC = ("cifar:name", "commons:name", "coco:name")
lived = {s: [it for it in cur.train["W1"][s] if cur.image(it[0]).max() > 0] for s in SRC}
held = {s: [it for it in cur.test["W1"][s] if cur.image(it[0]).max() > 0][:200] for s in SRC}
shared = sorted({it[2][0] for it in lived["coco:name"]} & ({it[2][0] for it in lived["cifar:name"]} | {it[2][0] for it in lived["commons:name"]}))
cache = {}


def code(ref):
    if ref not in cache:
        v = np.concatenate(cortex.code(cur.image(ref), grid=GRID))
        cache[ref] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float16)
    return cache[ref].astype(np.float32)


def named(words):
    row = {s: float(np.mean([tidy(words.recall(hear("what is this?"), code(it[0]))[0] or "") == it[2][0] for it in held[s]])) for s in SRC}
    both = [it for it in held["coco:name"] if it[2][0] in shared]
    row["coco, names also taught elsewhere"] = float(np.mean([tidy(words.recall(hear("what is this?"), code(it[0]))[0] or "") == it[2][0] for it in both]))
    return row


print(f"{len(shared)} of the COCO names are also names of CIFAR or Commons pictures: {' '.join(shared)}", flush=True)
for seed in SEEDS:
    for B in ARMS:
        rng, t0 = random.Random(seed), time.time()
        words = Words(dim, ear, branches=B)
        for n in range(N1):
            it = rng.choice(lived[rng.choice(SRC)])
            words.learn(hear(rng.choice(cur.frames)), code(it[0]), hear(it[2][0]))
        first = named(words)
        for n in range(N2):
            it = rng.choice(lived[rng.choice(SRC[:2])])
            words.learn(hear(rng.choice(cur.frames)), code(it[0]), hear(it[2][0]))
        second = named(words)
        used = float((words.branch_used[:, : len(words.units)] > 0).sum(0)[1:].mean()) if B > 1 else 1.0
        print(f"seed {seed} branches {B} ({time.time() - t0:.0f}s; branches in use per cell {used:.1f}): "
              + " | ".join(f"{k} {first[k]:.3f} → {second[k]:.3f}" for k in first), flush=True)
