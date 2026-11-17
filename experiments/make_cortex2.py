"""Finish a second cortex for a life (docs/VISION.md, 2026-11-17): give a developed VisualCortex2 its relay (the
summary decorrelated on its way to the word cells) from pictures of the lived side, and save it.
    FROM=out/tmp/cortex2_l1n.pkl TO=data/visual_cortex2.pkl GRID=4 SHRINK=0.0005 N=6000 python experiments/make_cortex2.py
"""
import os, pickle, random, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum  # noqa: E402

cur = Curriculum(seed=0, world=3)
rng = random.Random(5)
SRC = ("cifar:name", "commons:name", "coco:name")
lived = {s: [it for it in cur.train["W1"][s] if cur.image(it[0]).max() > 0] for s in SRC}
cortex = pickle.load(open(os.environ.get("FROM", "out/tmp/cortex2_l1n.pkl"), "rb"))
N, GRID, SHRINK = int(os.environ.get("N", 6000)), int(os.environ.get("GRID", 4)), float(os.environ.get("SHRINK", 0.0005))
cortex.settle_relay([cur.image(rng.choice(lived[rng.choice(SRC)])[0]) for _ in range(N)], GRID, SHRINK)
cortex.grid = GRID  # the relay belongs to this grid
pickle.dump(cortex, open(os.environ.get("TO", "data/visual_cortex2.pkl"), "wb"))
print(f"relay settled on {N} pictures (grid {GRID}, shrink {SHRINK}); summary of {len(cortex.relay.mu)} numbers; saved", flush=True)
