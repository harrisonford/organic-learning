"""A life through the curriculum (docs/CURRICULUM.md): stage by stage, gated by mastery.

    python experiments/curriculum_life.py out/cur_s0.jsonl
    SEED=0  CORTEX=path  EVERY=5000  STAGE_BUDGET=300000  TOTAL=2000000  RESUME=path.pkl
    MAX_CLUSTERS=24000  MAX_OUTPUTS=4096  MAX_GB=11  EAR_HABITUATION=0

Every EVERY moments the brain is tested on held-out items of every stage up to
the current one (and the next, to see what is coming). It is promoted when the
current stage's score reaches its pass mark and every earlier stage keeps at
least 80% of the score it had when passed. If a stage is not passed within
STAGE_BUDGET moments, the life ends with a verdict: this structure, with this
much experience, stops at that stage.
"""

import json
import os
import pickle
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import NAMES, PASS, RETAIN, STAGES, Curriculum, grade  # noqa: E402
from organic.loop2 import LoopBrain2  # noqa: E402
from organic.peye import Parts  # noqa: E402

out_path = sys.argv[1] if len(sys.argv) > 1 else "out/cur_s0.jsonl"
SEED = int(os.environ.get("SEED", 0))
EVERY = int(os.environ.get("EVERY", 5000))
STAGE_BUDGET = int(os.environ.get("STAGE_BUDGET", 300000))
TOTAL = int(os.environ.get("TOTAL", 2000000))

if os.environ.get("RESUME"):
    brain = pickle.load(open(os.environ["RESUME"], "rb"))
else:
    parts = Parts()
    parts.W = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "parts.npy"))
    parts.frozen = True
    cfg = {"reflex": "contrast", "adapt": True, "sleep_cleanup": True}
    if os.environ.get("CORTEX"):
        cfg["cortex"] = pickle.load(open(os.environ["CORTEX"], "rb"))
    brain = LoopBrain2(parts, seed=SEED, size=64, **cfg)
    brain.curriculum = {"stage": "B", "since": 0, "passed": {}, "promotions": []}
if os.environ.get("EAR_HABITUATION") == "0":
    brain.ear.habituation = False
if os.environ.get("MAX_CLUSTERS") or os.environ.get("MAX_OUTPUTS"):
    brain.orch.set_budget(int(os.environ.get("MAX_CLUSTERS", 0)) or None, int(os.environ.get("MAX_OUTPUTS", 0)) or None)

world = Curriculum(seed=0)
world.rng.seed(1000003 * SEED + brain.age)  # a resumed life draws moments it has not drawn before
print(world.summary(), flush=True)
tests = {s: world.test_set(s) for s in STAGES}
state = brain.curriculum
log = open(out_path, "a")
t0 = time.time()


def evaluate(stages):
    res = {}
    for s in stages:
        scores, ticks, exact = [], [], []
        for m in tests[s]:
            said, tick, *_ = brain.live(m["image"], m["question"], learn=False, explore=False)
            scores.append(grade(s, said, m["answers"]))
            exact.append(said in m["answers"])
            ticks.append(tick)
        res[s] = {"score": float(np.mean(scores)), "exact": float(np.mean(exact)), "tick": float(np.mean(ticks)), "n": len(scores)}
    return res


def save():
    pkl = out_path.replace(".jsonl", ".pkl")
    with open(pkl + ".tmp", "wb") as f:
        pickle.dump(brain, f)
    os.replace(pkl + ".tmp", pkl)


verdict = None
rewards = []
while brain.age < TOTAL:
    stage = state["stage"]
    m = world.moment(stage)
    _, tick, r, *_ = brain.live(m["image"], m["question"], m["answer"], grade=lambda said, m=m: grade(m["stage"], said, m["answers"]))
    rewards.append(r)
    if brain.age % EVERY:
        continue
    k = STAGES.index(stage)
    res = evaluate(STAGES[: k + 2])
    cur = res[stage]["score"]
    held = all(res[s]["score"] >= RETAIN * state["passed"][s] for s in STAGES[:k])
    promoted = cur >= PASS[stage] and held
    rec = {"age": brain.age, "stage": stage, "at_stage": brain.age - state["since"], "measure": res, "lived_reward": float(np.mean(rewards[-EVERY:])),
           "clusters": int(brain.orch.C), "outputs": len(brain.orch.outputs), "promoted": promoted, "held_earlier": held,
           "memory_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "seconds": round(time.time() - t0)}
    print(f"age {brain.age:>8} stage {stage:<2} ({brain.age - state['since']:>7} there) | "
          + " | ".join(f"{s} {v['score']:.2f}/{v['exact']:.2f}" for s, v in res.items()) + f" | clusters {brain.orch.C} | {'PROMOTED' if promoted else ''}", flush=True)
    if promoted:
        state["passed"][stage] = cur
        state["promotions"].append((stage, brain.age))
        if k + 1 == len(STAGES):
            verdict = "passed every stage"
        else:
            state["stage"], state["since"] = STAGES[k + 1], brain.age
    elif brain.age - state["since"] >= STAGE_BUDGET:
        verdict = f"stops at {stage} ({NAMES[stage]}): score {cur:.2f} < {PASS[stage]} after {STAGE_BUDGET:,} moments" if cur < PASS[stage] else \
                  f"stops at {stage} ({NAMES[stage]}): passes it but cannot hold earlier stages"
    rec["verdict"] = verdict
    log.write(json.dumps(rec) + "\n")
    log.flush()
    save()
    if verdict or rec["memory_gb"] > float(os.environ.get("MAX_GB", 1e9)):
        print("VERDICT:", verdict or f"stopped for memory at {rec['memory_gb']} GB", flush=True)
        break
