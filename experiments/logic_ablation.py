"""Is logic answered from what is seen, or from word statistics alone?

Loads the brain raised by curriculum.py and answers logic questions
(1) normally, (2) with match/mismatch neurons silenced, (3) with eyes closed.
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.evaluate import same_answer  # noqa: E402
from organic.scenes import LOGIC_INTENTS, episode  # noqa: E402

brain = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "out/curriculum.pkl", "rb"))
real_comparator = brain.comparator


def score(mode, n=400):
    rng = np.random.default_rng(3)
    per = {}
    for _ in range(n):
        e = episode(rng, "test", LOGIC_INTENTS)
        image = None if mode == "eyes closed" else e["image"]
        brain.comparator = (lambda q, l: np.zeros(brain.co_dim, np.float32)) if mode == "no comparator" else real_comparator
        out, _ = brain.live(image, e["question"], learn=False)
        per.setdefault(e["intent"], []).append(same_answer(out, e["answer"], e["intent"], 1))
    brain.comparator = real_comparator
    print(f"{mode:<15} " + " ".join(f"{k}={np.mean(v):.2f}" for k, v in sorted(per.items())) + f"   all={np.mean([x for v in per.values() for x in v]):.2f}")


for mode in ("normal", "no comparator", "eyes closed"):
    score(mode)
