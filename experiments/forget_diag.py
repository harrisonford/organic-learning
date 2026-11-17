"""Is what a life stops being taught erased, or only no longer retrieved? (docs/RESEARCH_MEMORY.md, "check first")

The first world 3 lives were taught COCO things until 50,000 moments and then, through a fault, not any more; their
naming of held-out COCO things fell from 0.30-0.34 to 0.19-0.25. For a brain (a life's pickle or a level): on the
first-words test pictures, by source, how often the living brain names the thing, which way of speaking it chose
(copy, own habits, recall), and how often the word cells alone would have named it.
    python experiments/forget_diag.py out/levels/world3_s0_W1.pkl out/world3_s0.pkl
"""
import collections, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, tidy  # noqa: E402

cur = Curriculum(seed=0, world=3)
tests = [m for m in cur.test_set("W1", n=400) if m["source"] in ("cifar:name", "commons:name", "coco:name")]
for path in sys.argv[1:]:
    brain = pickle.load(open(path, "rb"))
    brain.learning = False
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in tests:
        img = np.asarray(m["image"], np.float32)
        said = brain.live(img, m["question"], learn=False, explore=False)[0] or ""
        route = "copy" if brain.copied else "recall" if brain.recalled is not None else "own"
        cells = brain.words.recall(brain.ear.units(m["question"], False), brain._vis(img))[0] or ""
        r = rows[m["source"]]
        r["brain"].append(tidy(said) in m["answers"])
        r["cells"].append(tidy(cells) in m["answers"])
        r[route].append(1)
    print(f"{path} (age {brain.age})")
    for src, r in sorted(rows.items()):
        n = len(r["brain"])
        print(f"  {src:14} n {n:>3} | the brain names it {np.mean(r['brain']):.2f} | the word cells alone {np.mean(r['cells']):.2f} | spoke by recall {len(r['recall']) / n:.2f}, copy {len(r['copy']) / n:.2f}, own habits {len(r['own']) / n:.2f}", flush=True)
    del brain
