"""Does the speaking brain learn to say words? Stage B (echo first words) and W1.

    python experiments/speech_test.py [moments] [stage]

Babbling first (BABBLE spontaneous strings), then moments of the stage. Every
EVERY moments: exact echo rate and lenient score on held-out items, what it
says, how many motor chunks it has, the babble's sound similarity to heard
speech (drift), and nonword repetition ("say blick!", never heard).
"""

import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade, sound_sim  # noqa: E402
from organic.peye import Parts  # noqa: E402
from organic.speech import SpeakingBrain  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
STAGE = sys.argv[2] if len(sys.argv) > 2 else "B"
EVERY = int(os.environ.get("EVERY", 1000))
BABBLE = int(os.environ.get("BABBLE", 5000))
NONWORDS = ["blick", "tam", "pug", "sorn", "kib", "melp"]

parts = Parts()
parts.W = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "parts.npy"))
parts.frozen = True
brain = SpeakingBrain(parts, seed=int(os.environ.get("SEED", 0)), size=64, reflex="contrast", adapt=True, sleep_cleanup=True)
brain.ear.habituation = False
cur = Curriculum(seed=0)
test = cur.test_set(STAGE, n=60)
t0 = time.time()
# babbling before words: hearing others (the stage's utterances) and itself
for i in range(BABBLE):
    if i % 4 == 0:
        m = cur.moment(STAGE)
        brain.heard_recent = 0.9 * brain.heard_recent + 0.1 * brain.ear.question(brain.ear.units(m["question"]))
    brain.babble()
heard = [cur.moment(STAGE)["question"] for _ in range(200)]
babbles = [brain.babble(explore=True) for _ in range(200)]
print(f"after {BABBLE} babbles: sound map for {int((brain.mouth.amap_n > 0).sum())} gestures; babble sounds like speech {np.mean([max(sound_sim(b, h) for h in heard[:50]) for b in babbles]):.2f}; "
      f"e.g. {babbles[:6]}", flush=True)

for age in range(1, N + 1):
    m = cur.moment(STAGE)
    brain.live(m["image"], m["question"], m["answer"], grade=lambda said, m=m: grade(m["stage"], said, m["answers"]))
    if age % EVERY == 0:
        said, copies = [], 0
        for t in test:
            said.append(brain.live(t["image"], t["question"], learn=False, explore=False)[0])
            copies += getattr(brain, "copied", False)
        ex = np.mean([s in t["answers"] for s, t in zip(said, test)])
        sc = np.mean([grade(STAGE, s, t["answers"]) for s, t in zip(said, test)])
        nw = [brain.live(cur.image(("dark",)), f"say {w}!", learn=False, explore=False)[0] for w in NONWORDS]
        chunks = [u.decode("utf-8", "replace") for u in brain.mouth.units if len(u) > 1]
        print(f"age {age:>6} exact {ex:.2f} score {sc:.2f} copy {copies / len(test):.2f} | clusters {brain.orch.C} | chunks {len(chunks)} (e.g. {chunks[-8:]}) | "
              f"says {[(t['question'][:14], s) for s, t in list(zip(said, test))[:4]]} | nonwords {list(zip(NONWORDS, nw))[:4]} "
              f"(sound {np.mean([sound_sim(n or '', w) for n, w in zip(nw, NONWORDS)]):.2f}) | {time.time() - t0:.0f}s", flush=True)
pickle.dump(brain, open(os.environ.get("OUT", "/tmp/speech_brain.pkl"), "wb"))
