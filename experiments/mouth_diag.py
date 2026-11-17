"""Does the mouth lose what the word cells recalled? (2026-11-17, docs/CONVERSATION.md)

A life's answers at two words are often blends ("yodadance", "noback"). For a brain (a life's pickle) and held-out
test items of each stage and source: the score of what the word cells recalled (their text, before any speaking)
beside the score of what the brain then said, which way of speaking it chose, and how often what was said is
exactly what was recalled. Examples of the two when they differ.
    python experiments/mouth_diag.py out/world3_s0.pkl
"""
import collections, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade, tidy  # noqa: E402

cur = Curriculum(seed=0, world=int(os.environ.get("WORLD", 3)))
for path in sys.argv[1:]:
    brain = pickle.load(open(path, "rb"))
    brain.learning = False
    print(f"{path} (age {brain.age})")
    for stage in ("W1", "W2"):
        rows, shown = collections.defaultdict(lambda: collections.defaultdict(list)), collections.Counter()
        for m in cur.test_set(stage, n=300):
            img = np.asarray(m["image"], np.float32)
            said = brain.live(img, m["question"], learn=False, explore=False)[0] or ""
            route = "copy" if brain.copied else "recall" if brain.recalled is not None else "own"
            cells = brain.words.recall(brain.ear.units(m["question"], False), brain._vis(img))[0] or ""
            r = rows[m["source"]]
            r["said"].append(grade(stage, said, m["answers"]))
            r["cells"].append(grade(stage, cells, m["answers"]))
            r["same"].append(tidy(said) == tidy(cells))
            r[route].append(1)
            if tidy(said) != tidy(cells) and shown[m["source"]] < 4 and route == "recall":
                shown[m["source"]] += 1
                r["examples"].append(f"{m['question']!r} -> recalled {cells!r}, said {said!r} (wanted {m['answers'][0]!r})")
        for src, r in sorted(rows.items()):
            n = len(r["said"])
            print(f"  {stage} {src:14} n {n:>3} | said {np.mean(r['said']):.2f} | the word cells' own text {np.mean(r['cells']):.2f} | said = recalled {np.mean(r['same']):.2f} | by recall {len(r['recall']) / n:.2f}, copy {len(r['copy']) / n:.2f}, own {len(r['own']) / n:.2f}", flush=True)
            for e in r["examples"]:
                print("      " + e)
    del brain
