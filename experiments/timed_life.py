"""System 1 and system 2 in one brain: does time pressure shape the structure?

The brain lives with every kind of question from birth. After each answer it
gets a reward that depends on correctness and on how many ticks it took.
Some questions secretly have tight deadlines, some have none. The brain is not
told which. We watch accuracy, answer time and reward per time class evolve.

    python experiments/timed_life.py [episodes] [out.jsonl]      # with deadlines
    NO_DEADLINES=1 python experiments/timed_life.py ...          # control: only the small tick cost
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.brain import Brain  # noqa: E402
from organic.evaluate import same_answer  # noqa: E402
from organic.scenes import LOGIC_INTENTS, RELATION_INTENTS, SINGLE_INTENTS, episode, reward, time_class  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
out_path = sys.argv[2] if len(sys.argv) > 2 else "out/timed.jsonl"
deadlines = not os.environ.get("NO_DEADLINES")
SEED = int(os.environ.get("SEED", 0))
EVERY = int(os.environ.get("EVERY", 500))
ALL = SINGLE_INTENTS + LOGIC_INTENTS + RELATION_INTENTS

rng = np.random.default_rng(SEED)
brain = Brain(seed=SEED, **json.loads(os.environ.get("BRAIN", "{}")))
test_rng = np.random.default_rng(99)
test = [episode(test_rng, "test", ALL) for _ in range(360)]
log = open(out_path, "w")
t0 = time.time()


def measure():
    per = {}
    for e in test:
        out, tick, _ = brain.think(e["image"], e["question"], learn=False)
        ok = same_answer(out, e["answer"], e["intent"], len(e["objects"]))
        r = reward(e["intent"], tick, ok, deadlines)
        for key in (e["intent"], time_class(e["intent"])):
            per.setdefault(key, []).append((ok, tick, r))
    return {k: {"acc": float(np.mean([a for a, _, _ in v])), "tick": float(np.mean([t for _, t, _ in v])), "reward": float(np.mean([r for _, _, r in v]))} for k, v in per.items()}


lived_reward = []
for age in range(1, N + 1):
    e = episode(rng, "train", ALL)
    _, tick, got = brain.think(e["image"], e["question"], e["answer"], reward=lambda t, ok, i=e["intent"]: reward(i, t, ok, deadlines))
    lived_reward.append(got)
    if age % 2000 == 0:
        brain.sleep(replay=0)
    if age % EVERY == 0:
        m = measure()
        p = brain.speech.patience[: brain.speech.C]
        rec = {"age": age, "deadlines": deadlines, "seconds": round(time.time() - t0), "lived_reward": float(np.mean(lived_reward[-EVERY:])),
               "speech_clusters": int(brain.speech.C), "patience_mean": float(p.mean()), "patience_low": float((p < 0.3).mean()),
               "measure": m}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"age {age:>5} clusters={brain.speech.C:>5} patience={p.mean():.2f} (low {rec['patience_low']:.2f}) lived-reward={rec['lived_reward']:.2f} | "
              + "  ".join(f"{c}: acc {m[c]['acc']:.2f} tick {m[c]['tick']:.1f} rew {m[c]['reward']:.2f}" for c in ("fast", "normal", "precise") if c in m), flush=True)
import pickle  # noqa: E402

pickle.dump(brain, open(out_path.replace(".jsonl", ".pkl"), "wb"))
