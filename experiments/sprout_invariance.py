"""Does sprouting evolve structure that generalizes across positions?

Life: "what shape is it?" / "what color is it?" about one object, which only
ever appears in the left and middle columns. Tests: new objects in those
columns, and in the right column (never seen there). At the end, for every
area (including sprouted ones), how position-invariant are its assemblies?

    python experiments/sprout_invariance.py [episodes] [out.jsonl]
    SPROUT=1 ...   # let areas sprout
"""

import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.raw_brain import RawBrain  # noqa: E402
from organic.scenes import ASK, COLORS, SHAPES, answer, random_object, render  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
out_path = sys.argv[2] if len(sys.argv) > 2 else "out/sprout_inv.jsonl"
SEED = int(os.environ.get("SEED", 0))
EVERY = int(os.environ.get("EVERY", 500))
TRAIN_COLS, TEST_COLS = (0, 1), (2,)


def moment(rng, cols):
    o = random_object(rng, False)
    o["cell"] = (int(rng.choice(cols)), int(rng.integers(3)))
    intent = ["shape", "color"][rng.integers(2)]
    phr = ASK[intent]
    q = phr[rng.integers(len(phr) - 1)]
    return {"image": render([o], rng), "objects": [o], "intent": intent, "question": q, "answer": answer(intent, [o], None, rng)}


brain = RawBrain(seed=SEED, sprouting=bool(os.environ.get("SPROUT")), trace=float(os.environ.get("TRACE", 0)))
rng = np.random.default_rng(SEED)
trng = np.random.default_rng(99)
tests = {"trained positions": [moment(trng, TRAIN_COLS) for _ in range(150)], "new positions": [moment(trng, TEST_COLS) for _ in range(150)]}
log = open(out_path, "w")
t0 = time.time()


def score(eps):
    per = {}
    for e in eps:
        out, _, _ = brain.think(e["image"], e["question"], learn=False)
        per.setdefault(e["intent"], []).append(" ".join(out.split()) == " ".join(e["answer"].split()))
    return {k: float(np.mean(v)) for k, v in per.items()}


for age in range(1, N + 1):
    e = moment(rng, TRAIN_COLS)
    brain.think(e["image"], e["question"], e["answer"])
    if age % EVERY == 0:
        res = {name: score(eps) for name, eps in tests.items()}
        rec = {"age": age, "areas": {k: a.C for k, a in brain.areas.items()}, "events": brain.events, "scores": res, "seconds": round(time.time() - t0)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"age {age:>5} " + " | ".join(f"{n}: " + " ".join(f"{k}={v:.2f}" for k, v in sc.items()) for n, sc in res.items()) + f" | {brain.report()}", flush=True)

# invariance: does one assembly stand for one shape wherever it appears?
probe_rng = np.random.default_rng(7)
print("\ninvariance per area (probe: each shape at all 9 positions, 3 colors each)")
print(f"{'area':<10}{'shape purity':>14}{'assemblies per shape':>22}{'same assembly L vs R':>22}")
for name in brain.areas:
    winners = {}
    for shape in SHAPES:
        for col in range(3):
            for row in range(3):
                for color in list(COLORS)[:3]:
                    o = {"shape": shape, "color": color, "cell": (col, row), "size": "small"}
                    hist = brain.perceive(render([o], probe_rng), learn=False)
                    st = hist[-1][name]
                    if len(st) and st.max() > 0:
                        winners.setdefault(shape, []).append((col, int(np.argmax(st))))
    if not winners:
        continue
    by_assembly = {}
    for shape, ws in winners.items():
        for _, a in ws:
            by_assembly.setdefault(a, []).append(shape)
    purity = sum(max(v.count(x) for x in set(v)) for v in by_assembly.values()) / sum(len(v) for v in by_assembly.values())
    per_shape = np.mean([len({a for _, a in ws}) for ws in winners.values()])
    same_lr = np.mean([len({a for c, a in ws if c == 0} & {a for c, a in ws if c == 2}) > 0 for ws in winners.values()])
    print(f"{name:<10}{purity:>14.2f}{per_shape:>22.1f}{same_lr:>22.2f}")
pickle.dump(brain, open(out_path.replace(".jsonl", ".pkl"), "wb"))
