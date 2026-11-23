"""What a saved learner does, item by item, for the paper's "what it looks like" section (nothing learns).

For the held-out questions of first words and of pages: the question, the wanted answers, what the learner said,
whether that is right by the letter-level grade (no judge here), and how long it took: the ticks it looked before it
began to speak (it speaks when its own confidence passes a patience it has learned), the ticks it spoke (one motor
unit a tick), how often it moved its eye. Also the learner's size, organ by organ (numbers held in arrays).
Writes out/tmp/showcase_<name>.json and, for the figure, the pictures of the first items of each source
(out/tmp/showcase_<name>.npz).
    python experiments/showcase_probe.py out/cleanw_s0.pkl
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade, tidy  # noqa: E402

N, SHOW = int(os.environ.get("N", 300)), int(os.environ.get("SHOW", 16))
cur = Curriculum(seed=0, world=4)


def size(obj, seen=None, depth=0):
    """How many numbers an object holds in arrays (each array counted once)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 6:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.size)
    if isinstance(obj, dict):
        return sum(size(v, seen, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(size(v, seen, depth + 1) for v in obj)
    return sum(size(v, seen, depth + 1) for v in vars(obj).values()) if hasattr(obj, "__dict__") else 0


for path in sys.argv[1:]:
    name = os.path.basename(path)[:-4]
    brain = pickle.load(open(path, "rb"))
    brain.learning = False
    spoke = []
    speak = brain._speak
    brain._speak = lambda *a, **k: (lambda out: (spoke.append(len(out[0])), out)[1])(speak(*a, **k))  # how many units were spoken
    seen = set()
    sizes = {k: size(v, seen) for k, v in vars(brain).items()}
    sizes = {k: v for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]) if v > 1000}
    words = brain.words
    facts = {"age": brain.age, "numbers by organ": sizes, "numbers in all": sum(sizes.values()),
             "word cells": len(words.units) if words is not None else None,
             "moments in the store": int(words.episodes.n) if getattr(words, "episodes", None) is not None else None,
             "units of the ear": len(getattr(brain.ear, "units_list", getattr(brain.ear, "vocab", []))) or None}
    rows, pictures = [], {}
    for stage in ("W1", "W2"):
        shown = {}
        for m in cur.test_set(stage, n=N):
            spoke.clear()
            said, tick, _, pth, stays, _ = brain.live(np.asarray(m["image"], np.float32), m["question"], learn=False, explore=False)
            said = tidy(said or "")
            talk = spoke[-1] if spoke else 0
            row = {"stage": stage, "source": m["source"], "question": m["question"], "answers": m["answers"], "said": said,
                   "right": bool(grade(stage, said, m["answers"]) >= 0.5), "ticks": int(tick), "looking": int(tick - talk), "speaking": int(talk),
                   "eye moves": len(pth) - 1, "stays": int(stays)}
            if m["source"] in ("cifar:name", "page:name", "page:there") and shown.get(m["source"], 0) < SHOW:
                row["picture"] = f"{m['source']}_{shown.get(m['source'], 0)}"
                pictures[row["picture"]] = (np.clip(np.asarray(m["image"]), 0, 1) * 255).astype(np.uint8)
                shown[m["source"]] = shown.get(m["source"], 0) + 1
            rows.append(row)
    json.dump({"brain": name, "facts": facts, "rows": rows}, open(f"out/tmp/showcase_{name}.json", "w"), indent=1)
    np.savez_compressed(f"out/tmp/showcase_{name}.npz", **pictures)
    print(f"{path}: age {brain.age}; numbers held in arrays {facts['numbers in all']:,}: " + ", ".join(f"{k} {v:,}" for k, v in list(sizes.items())[:8]))
    by = {}
    for r in rows:
        by.setdefault((r["stage"], r["source"]), []).append(r)
    for (stage, src), v in sorted(by.items()):
        a = lambda key, rs=v: float(np.mean([r[key] for r in rs])) if rs else float("nan")  # noqa: E731
        ok, bad = [r for r in v if r["right"]], [r for r in v if not r["right"]]
        print(f"   {stage} {src:14} n {len(v):3} | right {a('right'):.2f} | ticks looking {a('looking'):.1f} (when right {a('looking', ok):.1f}, when wrong {a('looking', bad):.1f}) | "
              f"speaking {a('speaking'):.1f} | eye moves {a('eye moves'):.1f}", flush=True)
    del brain
