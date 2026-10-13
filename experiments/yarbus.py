"""The Yarbus test: does the question decide where the eye looks?

Scenes: two objects, one on the left and one on the right (same row).
Questions: "what color/shape is the left/right one?". Color is visible in the
coarse glance (its layout already shows which side is which colour); shape
needs a sharp look at the right side, which only the goal loop can know.

    python experiments/yarbus.py [moments] [out.jsonl]
    LOOPS=both|reflex|goal  SEED=0
"""

import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.loop import LoopBrain  # noqa: E402
from organic.loop2 import LoopBrain2  # noqa: E402
from organic.peye import Parts  # noqa: E402
from organic.scenes import COLORS, SHAPES, render  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
out_path = sys.argv[2] if len(sys.argv) > 2 else "out/yarbus.jsonl"
LOOPS = os.environ.get("LOOPS", "both")
SEED = int(os.environ.get("SEED", 0))
EVERY = int(os.environ.get("EVERY", 1000))


def moment(rng):
    row = int(rng.integers(3))
    cols = list(COLORS)
    c1, c2 = rng.choice(len(cols), 2, replace=False)
    s1, s2 = rng.choice(len(SHAPES), 2, replace=False)
    left = {"shape": SHAPES[s1], "color": cols[c1], "cell": (0, row), "size": "small"}
    right = {"shape": SHAPES[s2], "color": cols[c2], "cell": (2, row), "size": "small"}
    side = ["left", "right"][rng.integers(2)]
    kind = ["color", "shape"][rng.integers(2)]
    obj = left if side == "left" else right
    return {"image": render([left, right], rng), "question": f"what {kind} is the {side} one?", "answer": obj[kind], "side": side, "kind": kind}


parts = Parts()
parts.W = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "parts.npy"))
parts.frozen = True
DESIGN = int(os.environ.get("DESIGN", 1))
Brain = LoopBrain2 if DESIGN == 2 else LoopBrain
brain = Brain(parts, seed=SEED, goal_loop=LOOPS in ("both", "goal"), reflex_loop=LOOPS in ("both", "reflex"))
rng = np.random.default_rng(SEED)
test = [moment(np.random.default_rng(1000 + i)) for i in range(240)]
log = open(out_path, "w")
t0 = time.time()


def evaluate():
    per = {}
    for e in test:
        said, tick, _, path, *extra = brain.live(e["image"], e["question"], learn=False, explore=False)
        stays = extra[0] if extra else 0
        key = f"{e['kind']}-{e['side']}"
        # where did saccades land? columns 0-1 = left half, 3-4 = right half
        cols = [t % 5 for t in path[1:]]
        toward = np.mean([(c < 2) if e["side"] == "left" else (c > 2) for c in cols]) if cols else np.nan
        per.setdefault(key, []).append((said == e["answer"], tick, toward, stays))
    return {k: {"acc": float(np.mean([a for a, _, _, _ in v])), "tick": float(np.mean([t for _, t, _, _ in v])),
                "stays": float(np.mean([st for _, _, _, st in v])),
                "gaze_to_asked_side": float(np.nanmean([g for _, _, g, _ in v])) if any(not np.isnan(g) for _, _, g, _ in v) else None} for k, v in sorted(per.items())}


for age in range(1, N + 1):
    e = moment(rng)
    brain.live(e["image"], e["question"], e["answer"])
    if age % EVERY == 0:
        m = evaluate()
        rec = {"age": age, "loops": LOOPS, "seed": SEED, "design": DESIGN, "measure": m, "clusters": int(brain.orch.C), "seen_assemblies": int(getattr(brain, "sense", None).C) if hasattr(brain, "sense") else None,
               "patience": float(brain.orch.patience[: brain.orch.C].mean()), "seconds": round(time.time() - t0)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"age {age:>5} " + " | ".join(f"{k}: acc {v['acc']:.2f} tick {v['tick']:.1f} stays {v['stays']:.1f} gaze->asked {v['gaze_to_asked_side'] if v['gaze_to_asked_side'] is None else round(v['gaze_to_asked_side'], 2)}" for k, v in m.items())
              + f" | clusters {brain.orch.C} | {time.time() - t0:.0f}s", flush=True)
pickle.dump(brain, open(out_path.replace(".jsonl", ".pkl"), "wb"))
