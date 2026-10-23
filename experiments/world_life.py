"""A long life in the photo world, with hidden deadlines.

    python experiments/world_life.py [moments] [out.jsonl]
    SEED=0  BRAIN='{...}'  EVERY=10000  RESUME=path.pkl  FREEZE=1  PER_KIND=0
    MAX_GB=11  MAX_CLUSTERS=24000  MAX_OUTPUTS=4096

The brain (design 2 + contrast reflex + adaptation + sleep cleanup by default)
looks at 64x64 photos, hears a question, answers when sure enough, and learns
from what an adult would have said plus a reward that halves for every tick
past the question's hidden deadline. Every EVERY moments it is tested on photos
it has never seen, per kind of question, and the world is refreshed so new
questions written by the adults join in (unless FREEZE=1: the same world all
life long, for controlled comparisons). PER_KIND>0 tests on that many items of
every kind instead of 600 random moments.
"""

import collections
import json
import os
import pickle
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.loop2 import LoopBrain2  # noqa: E402
from organic.peye import Parts  # noqa: E402
from organic.world import World  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
out_path = sys.argv[2] if len(sys.argv) > 2 else "out/world.jsonl"
SEED = int(os.environ.get("SEED", 0))
EVERY = int(os.environ.get("EVERY", 10000))
cfg = {"reflex": "contrast", "adapt": True, "sleep_cleanup": True, **json.loads(os.environ.get("BRAIN", "{}"))}

parts = Parts()
parts.W = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "parts.npy"))
parts.frozen = True
if os.environ.get("CORTEX"):  # a visual cortex developed beforehand (experiments/develop_cortex.py)
    cfg["cortex"] = pickle.load(open(os.environ["CORTEX"], "rb"))
if os.environ.get("RESUME"):
    brain = pickle.load(open(os.environ["RESUME"], "rb"))
else:
    brain = LoopBrain2(parts, seed=SEED, size=64, **cfg)
if os.environ.get("MAX_CLUSTERS") or os.environ.get("MAX_OUTPUTS"):  # memory budget (docs/PRUNING.md)
    brain.orch.set_budget(int(os.environ.get("MAX_CLUSTERS", 0)) or None, int(os.environ.get("MAX_OUTPUTS", 0)) or None)
world = World(seed=0)
if brain.age:  # a resumed life continues with moments it has not drawn before
    import random
    world.rng = random.Random(1000003 * SEED + brain.age)
print(world.summary(), flush=True)
test_rng = np.random.default_rng(99)
PER_KIND = int(os.environ.get("PER_KIND", 0))
if PER_KIND:
    by = {}
    for it in world.test:
        by.setdefault(it[3], []).append(it)
    test = []
    for kind in sorted(by):
        for i in test_rng.choice(len(by[kind]), min(PER_KIND, len(by[kind])), replace=False):
            ref, q, a, _ = by[kind][i]
            test.append({"image": world.image(ref), "question": q, "answer": a, "kind": kind})
else:
    test = [world.moment("test") for _ in range(600)]
log = open(out_path, "a")
t0 = time.time()


PROBES = ["what color is the ball?", "thank you very much!", "mary went to the kitchen. where is mary?", "is there a dog in the picture?"]


def evaluate():
    """Accuracy and timing per kind, plus signals about how answers are produced:
    modal = share of the most common reply, copied = reply is a word heard in the
    question, producible = the right answer exists as an output neuron now."""
    per, said_by = {}, {}
    for m in test:
        said, tick, _, path, stays, _ = brain.live(m["image"], m["question"], learn=False, explore=False)
        per.setdefault(m["kind"], []).append((said == m["answer"], tick, stays, said is not None and f" {said} " in f" {m['question']} ".replace("?", " ").replace(".", " ").replace(",", " "),
                                              f"word:{m['answer']}" in brain.orch.out_index))
        said_by.setdefault(m["kind"], []).append(said)
    res = {}
    for k, v in sorted(per.items()):
        c = collections.Counter(said_by[k])
        res[k] = {"n": len(v), "acc": float(np.mean([x[0] for x in v])), "tick": float(np.mean([x[1] for x in v])), "stays": float(np.mean([x[2] for x in v])),
                  "modal": c.most_common(1)[0][1] / len(v), "modal_word": c.most_common(1)[0][0], "distinct": len(c),
                  "copied": float(np.mean([x[3] for x in v])), "producible": float(np.mean([x[4] for x in v]))}
    return res


def ear_state():
    ear = brain.ear
    tot = sum(ear.heard.values()) or 1
    return {"units": len(ear.heard), "contexts": len(ear.counts), "utterances": ear.utterances,
            "mean_unit_bytes": round(sum(len(u) * n for u, n in ear.heard.items()) / tot, 2),
            "probes": [[u.decode("utf-8", "replace") for u in ear.units(t, learn=False)] for t in PROBES]}


def save_ear():
    """The ear alone (small), so it can be studied without loading the whole brain."""
    path = out_path.replace(".jsonl", "_ear.pkl")
    with open(path + ".tmp", "wb") as f:
        pickle.dump(brain.ear, f)
    os.replace(path + ".tmp", path)


save_ear()
start = brain.age
rewards = []
for age in range(start + 1, start + N + 1):
    m = world.moment()
    _, tick, r, *_ = brain.live(m["image"], m["question"], m["answer"], deadline=m["deadline"])
    rewards.append(r)
    if age % EVERY == 0:
        res = evaluate()
        rec = {"age": age, "seed": SEED, "measure": res, "lived_reward": float(np.mean(rewards[-EVERY:])), "clusters": int(brain.orch.C),
               "familiar": float(getattr(brain.orch, "familiar_fraction", float("nan"))), "sleep": brain.sleep_log[-3:], "world": world.summary(), "seconds": round(time.time() - t0),
               "memory_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
               "outputs": len(brain.orch.outputs), "pruned": (brain.orch.prune_log or [])[-4:], "ear": ear_state()}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"age {age:>7} reward {rec['lived_reward']:.2f} clusters {rec['clusters']} familiar {rec['familiar']:.3f} | "
              + " | ".join(f"{k}: {v['acc']:.2f} t{v['tick']:.1f}" for k, v in res.items()) + f" | {rec['memory_gb']} GB | {time.time() - t0:.0f}s", flush=True)
        pkl = out_path.replace(".jsonl", ".pkl")
        with open(pkl + ".tmp", "wb") as f:  # atomic: a crash or restart mid-save keeps the previous save
            pickle.dump(brain, f)
        os.replace(pkl + ".tmp", pkl)
        save_ear()
        if rec["memory_gb"] > float(os.environ.get("MAX_GB", 1e9)):  # the machine has 14 GB; stop before it runs out
            print(f"stopping: memory {rec['memory_gb']} GB", flush=True)
            break
        if not os.environ.get("FREEZE"):
            rng = world.rng  # keep drawing moments where we were (a fresh rng would replay the same moments)
            world = World(seed=0)
            world.rng = rng  # new adult questions join the world (same photo split)
