"""Raise a brain (or load a saved one) and print explain() traces for failures by intent."""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.brain import Brain  # noqa: E402
from organic.evaluate import explain, same_answer  # noqa: E402
from organic.scenes import episode  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
split = sys.argv[2] if len(sys.argv) > 2 else "test"
intents = sys.argv[3].split(",") if len(sys.argv) > 3 else None
cache = f"/tmp/claude-1000/brain_{N}.pkl"
if os.path.exists(cache) and not os.environ.get("FRESH"):
    b = pickle.load(open(cache, "rb"))
else:
    rng = np.random.default_rng(0)
    b = Brain()
    for i in range(N):
        e = episode(rng)
        b.live(e["image"], e["question"], e["answer"])
    pickle.dump(b, open(cache, "wb"))
r = np.random.default_rng(5)
shown = 0
for _ in range(400):
    e = episode(r, split)
    if intents and e["intent"] not in intents:
        continue
    out, _ = b.live(e["image"], e["question"], learn=False)
    if not same_answer(out, e["answer"], e["intent"], len(e["objects"])):
        print(f"\n[{e['intent']}] objects={[(o['color'], o['shape'], o['size'], o['cell']) for o in e['objects']]}")
        print(f"(person: {e['answer']})\n" + explain(b, e["image"], e["question"]))
        shown += 1
        if shown >= 5:
            break
