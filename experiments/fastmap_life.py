"""Fast mapping in a whole living brain (docs/RESEARCH_MEMORY.md, 2026-11-17).

fastmap_probe.py asks the word cells alone and finds almost nothing. A living brain also has the orchestrator's
clusters, which learn by imitation what to say in a situation: do they hold one moment? For a life's pickle: for
each kind of thing never named in first words, one lesson ("what is this?" - "fox") on a picture, then asked at once
on the same picture and on another of the kind, and again after 20 ordinary first-words moments. Which way of
speaking answered (copy, own habits, recall) is counted too.
    python experiments/fastmap_life.py out/world3b_s0.pkl
"""
import collections, os, pickle, random, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade, tidy  # noqa: E402

KINDS = int(os.environ.get("KINDS", 40))
cur = Curriculum(seed=0, world=3)
new = {}
for stage in ("P", "G1"):
    for it in cur.train[stage]["cifar:name"]:
        if " " not in it[2][0]:
            new.setdefault(it[2][0], []).append(it[0])
for path in sys.argv[1:]:
    brain = pickle.load(open(path, "rb"))
    rng = random.Random(0)
    rows, routes = [], collections.Counter()

    def ask(ref, w):
        said = brain.live(cur.image(ref), "what is this?", learn=False, explore=False)[0] or ""
        routes["copy" if brain.copied else "recall" if brain.recalled is not None else "own habits"] += 1
        return tidy(said) == w

    for w in rng.sample(sorted(new), KINDS):
        a, b = rng.sample(new[w], 2)
        brain.live(cur.image(a), rng.choice(cur.frames), w, grade=lambda said, w=w: grade("W1", said, [w]))
        now = [ask(a, w), ask(b, w)]
        for _ in range(20):
            m = cur.moment("W1")
            brain.live(m["image"], m["question"], m["answer"], grade=lambda said, m=m: grade(m["stage"], said, m["answers"]))
        rows.append(now + [ask(a, w), ask(b, w)])
    r = np.mean(rows, axis=0)
    print(f"{path} (age {brain.age}), {KINDS} new kinds, one lesson each: at once: the same picture {r[0]:.2f}, another of the kind {r[1]:.2f} | "
          f"after 20 moments: {r[2]:.2f}, {r[3]:.2f} | spoke by {dict(routes)}", flush=True)
    del brain
