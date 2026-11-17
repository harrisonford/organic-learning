"""When the word cells have nothing to say, does the brain do better by not letting them speak? (2026-11-17)

In first words the echo stage dips (0.85 to 0.5) and holds a life back: asked to say a word it has never heard
("say datum now") the brain speaks by recall four times in five, and the word cells' own text is a known word or a
run-on. For a brain (a life's pickle), on held-out echo and first-words items, the score
  as it is | with the word cells silent below their own firing level (Words.setpoint) | with the word cells never speaking
and how often the cells stayed silent under their own level.
    python experiments/silent_route_diag.py out/tmp/eyes2_s2_30k.pkl
"""
import collections, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade  # noqa: E402

cur = Curriculum(seed=0, world=3)
for path in sys.argv[1:]:
    brain = pickle.load(open(path, "rb"))
    brain.learning = False
    recall = brain.words.recall
    print(f"{path} (age {brain.age})")
    for stage in ("B", "W1"):
        rows = collections.defaultdict(lambda: collections.defaultdict(list))
        for m in cur.test_set(stage, n=300):
            img = np.asarray(m["image"], np.float32)
            r = rows[m["source"]]
            brain.words.setpoint = False
            r["as it is"].append(grade(stage, brain.live(img, m["question"], learn=False, explore=False)[0] or "", m["answers"]))
            brain.words.setpoint = True
            r["own level"].append(grade(stage, brain.live(img, m["question"], learn=False, explore=False)[0] or "", m["answers"]))
            r["silent"].append(brain.words.recall(brain.ear.units(m["question"], False), brain._vis(img))[0] is None)
            brain.words.setpoint = False
            brain.novel_drive = True
            r["novel"].append(grade(stage, brain.live(img, m["question"], learn=False, explore=False)[0] or "", m["answers"]))
            r["copied"].append(bool(brain.copied))
            brain.novel_drive = False
            heard = brain.ear.units(m["question"], False)
            r["novelty"].append(max(1.0 / (1.0 + brain.ear.heard.get(u, 0) / brain.orch.maturation) for u in heard))
            r["sure"].append(recall(heard, brain._vis(img))[1])
            brain.words.recall = lambda *a, **k: (None, 0.0)
            r["never"].append(grade(stage, brain.live(img, m["question"], learn=False, explore=False)[0] or "", m["answers"]))
            del brain.words.recall
        for src, r in sorted(rows.items()):
            print(f"  {stage} {src:14} n {len(r['silent']):>3} | as it is {np.mean(r['as it is']):.2f} | silent below their own level {np.mean(r['own level']):.2f} (silent on {np.mean(r['silent']):.2f}) | the word cells never speaking {np.mean(r['never']):.2f} | copy driven by the ear's novelty {np.mean(r['novel']):.2f} (copied {np.mean(r['copied']):.2f}; novelty {np.mean(r['novelty']):.2f}, the cells' sureness {np.mean(r['sure']):.2f})", flush=True)
