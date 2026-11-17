"""Do the word cells name more with another cortex? (docs/VISION.md, 2026-11-17)

A linear reference says what a code holds; the brain's word cells learn by their own local rule, online, each
picture seen once. Here the real word cells (organic/words.py) live N naming moments of world 3's first words
(CIFAR, Commons and COCO things, the sources taking turns, the question worded afresh each time) with each cortex,
and then name held-out pictures of each source.
    CORTEX=data/visual_cortex.pkl,out/tmp/cortex2_a.pkl SEEDS=0,1,2 N=8000 python experiments/words_on_cortex.py
"""
import os, pickle, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, tidy  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
N, GRID = int(os.environ.get("N", 8000)), int(os.environ.get("GRID", 4))
PATHS = os.environ.get("CORTEX", "data/visual_cortex.pkl").split(",")
cur = Curriculum(seed=0, world=3)
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
hear = lambda q: ear.units(q, learn=False)  # noqa: E731
SRC = ("cifar:name", "commons:name", "coco:name")
lived = {s: [it for it in cur.train["W1"][s] if cur.image(it[0]).max() > 0] for s in SRC}
held = {s: [it for it in cur.test["W1"][s] if cur.image(it[0]).max() > 0][:200] for s in SRC}

for path in PATHS:
    cortex = pickle.load(open(path, "rb"))
    dim = sum(l.kinds for l in cortex.layers) * GRID * GRID
    cache = {}

    def code(ref):
        if ref not in cache:
            v = np.concatenate(cortex.code(cur.image(ref), grid=GRID))
            cache[ref] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float16)
        return cache[ref].astype(np.float32)

    for seed in SEEDS:
        rng, t0 = random.Random(seed), time.time()
        words = Words(dim, ear)
        right = []
        for n in range(N):
            it = rng.choice(lived[rng.choice(SRC)])
            q, v = hear(rng.choice(cur.frames)), code(it[0])
            if n >= N - 1500:  # while living: was the name expected before it was heard?
                right.append(tidy(words.recall(q, v)[0] or "") == it[2][0])
            words.learn(q, v, hear(it[2][0]))
        row = {s: float(np.mean([tidy(words.recall(hear("what is this?"), code(it[0]))[0] or "") == it[2][0] for it in held[s]])) for s in SRC}
        print(f"{os.path.basename(path):24} seed {seed} ({time.time() - t0:.0f}s, code {dim}): lived, last 1,500: {np.mean(right):.3f} | held out: "
              + " | ".join(f"{s} {v:.3f}" for s, v in row.items()) + f" | mean {np.mean(list(row.values())):.3f}", flush=True)
