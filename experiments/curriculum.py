"""Does the structure grow to meet a new kind of problem?

A brain lives with the base tasks (describe, color, shape, where, size, funny,
yes/no about one property, count). Midway through life, logic questions start
to appear too (am i ..., and, either/or, not). Every checkpoint logs the
structure (speech clusters, word-forms, births) and accuracy per question type,
so we can see whether, and where, new neurons appear when the new task arrives.

    python experiments/curriculum.py [base_episodes] [mixed_episodes] [out.jsonl]
    python experiments/curriculum.py base:3000,logic:3000,relations:3000 out/x.jsonl

Each phase adds its tasks to everything learned before. Brain settings can be
passed as JSON in the BRAIN environment variable, e.g. BRAIN='{"sprouting": true}'.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.brain import Brain  # noqa: E402
from organic.evaluate import evaluate  # noqa: E402
from organic.scenes import LOGIC_INTENTS, RELATION_INTENTS, SINGLE_INTENTS, episode  # noqa: E402

TASKS = {"base": SINGLE_INTENTS, "logic": LOGIC_INTENTS, "relations": RELATION_INTENTS}
if ":" in sys.argv[1] if len(sys.argv) > 1 else False:
    phases = [(name, int(n)) for name, n in (p.split(":") for p in sys.argv[1].split(","))]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "out/curriculum.jsonl"
else:
    base_n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    mixed_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    out_path = sys.argv[3] if len(sys.argv) > 3 else "out/curriculum.jsonl"
    phases = [("base", base_n), ("logic", mixed_n)]
schedule = []
for k, (name, n) in enumerate(phases):
    intents = [i for p, _ in phases[: k + 1] for i in TASKS[p]]
    label = "+".join(p for p, _ in phases[: k + 1])
    schedule += [(label, intents)] * n
every = int(os.environ.get("EVERY", 250))
os.makedirs(os.path.dirname(out_path), exist_ok=True)

rng = np.random.default_rng(0)
brain = Brain(**json.loads(os.environ.get("BRAIN", "{}")))  # e.g. BRAIN={"hear_categories": true}
log = open(out_path, "w")
t0 = time.time()
births_before = 0
for age in range(1, len(schedule) + 1):
    phase, intents = schedule[age - 1]
    e = episode(rng, "train", intents)
    brain.live(e["image"], e["question"], e["answer"])
    if age % 1000 == 0:
        brain.sleep(replay=200)
    if age % every == 0:
        _, base = evaluate(brain, "test", n=150, intents=SINGLE_INTENTS)
        _, logic = evaluate(brain, "test", n=120, seed=7, intents=LOGIC_INTENTS)
        _, rel = evaluate(brain, "test", n=120, seed=8, intents=RELATION_INTENTS)
        rec = {
            "age": age,
            "phase": phase,
            "speech_clusters": int(brain.speech.C),
            "speech_births": int(brain.speech.births),
            "new_clusters": int(brain.speech.births - births_before),
            "word_forms": int(brain.V),
            "sensory": {k: int(a.C) for k, a in brain.areas.items()},
            "sprouted": {a.name: int(a.area.C) for a, _ in brain.sprouted},
            "events": [ev for ev in brain.events if ev[0] > age - every],
            "base_acc": float(np.mean(list(base.values()))),
            "logic_acc": float(np.mean(list(logic.values()))),
            "relation_acc": float(np.mean(list(rel.values()))),
            "per_intent": {**base, **logic, **rel},
            "seconds": round(time.time() - t0),
        }
        births_before = brain.speech.births
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(
            f"age {age:>5} [{phase:>10}] clusters={rec['speech_clusters']:>5} (+{rec['new_clusters']:>3}) "
            f"word-forms={rec['word_forms']:>4} base={rec['base_acc']:.2f} logic={rec['logic_acc']:.2f} "
            f"relations={rec['relation_acc']:.2f} "
            + " ".join(f"{k}={v:.2f}" for k, v in {**logic, **rel}.items())
            + "".join(f" {k}={v}" for k, v in rec["sprouted"].items())
            + "".join(f"  <{ev[1]} at {ev[0]}>" for ev in rec["events"]),
            flush=True,
        )
import pickle  # noqa: E402

pickle.dump(brain, open(out_path.replace(".jsonl", ".pkl"), "wb"))
