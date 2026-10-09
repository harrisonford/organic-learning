"""Inside a timed brain: confidence per tick vs the patience gating it.

    python experiments/timed_analysis.py out/timed_deadline_s0.pkl
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.evaluate import same_answer  # noqa: E402
from organic.scenes import LOGIC_INTENTS, RELATION_INTENTS, SINGLE_INTENTS, episode, time_class  # noqa: E402

brain = pickle.load(open(sys.argv[1], "rb"))
rng = np.random.default_rng(5)
rows = {}
for _ in range(300):
    e = episode(rng, "test", SINGLE_INTENTS + LOGIC_INTENTS + RELATION_INTENTS)
    scene = brain.look(e["image"], False)
    per_tick = []
    for t in range(7):
        out, _ = brain.live(e["image"], e["question"], learn=False, tick=t, scene=scene)
        inv = brain.last_involved
        tot = sum(inv.values()) or 1
        pat = sum(brain.speech.patience[c] * g for c, g in inv.items()) / tot
        ok = same_answer(out, e["answer"], e["intent"], len(e["objects"]))
        per_tick.append((brain.last_confidence, pat, ok))
    rows.setdefault(time_class(e["intent"]), []).append(per_tick)
for cls, eps in rows.items():
    a = np.array(eps)  # (n, ticks, 3)
    print(f"{cls:<8} n={len(eps)}")
    print("   tick:        " + " ".join(f"{t:>5}" for t in range(7)))
    print("   confidence:  " + " ".join(f"{v:5.2f}" for v in a[:, :, 0].mean(0)))
    print("   patience:    " + " ".join(f"{v:5.2f}" for v in a[:, :, 1].mean(0)))
    print("   correct if   " + " ".join(f"{v:5.2f}" for v in a[:, :, 2].mean(0)) + "   (answer were given at this tick)")
p = brain.speech.patience[: brain.speech.C]
print("patience over clusters: quantiles", np.round(np.quantile(p, [0.05, 0.25, 0.5, 0.75, 0.95]), 2))
