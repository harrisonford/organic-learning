"""Word cells that can have nothing to say (offline; docs/CONVERSATION.md, proposal 2).

A word cell always had something to say: the cell with the most content drive spoke, however little that was, so a
word never heard ("please say alcohol again") got an answer ("fish") and the copy route could not be left to do
its work. Here each cell keeps its own two levels of content drive, measured as it lives: when its unit was then
heard, and when it was the most driven cell but its unit was not heard. A cell driven less than halfway between
them stays silent. No number is set by hand.

The cells live echo games first, then first and two words with echo still mixed in, as a life does. Held out:
  echo of new words:  the cells should stay silent (the mouth repeats by sound)
  pictures, text:     the cells should still speak, and be right as often as before
    SEEDS=0,1,2 EAR=out/tmp/ear_edge.pkl python experiments/silence_probe.py
"""
import json, os, pickle, random, sys, time, zlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, grade, tidy, words as split  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
cur = Curriculum(seed=0, world=2)
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_edge.pkl"), "rb"))
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
V = pickle.load(open("out/tmp/vis_cache.pkl", "rb")) if os.path.exists("out/tmp/vis_cache.pkl") else {}


def vis(ref):
    if ref not in V:
        img = cur.image(ref)
        V[ref] = None
        if img.max() > 0:
            v = np.concatenate(cortex.code(img, grid=4))
            V[ref] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
    return V[ref]


hear = lambda q: ear.units(q, learn=False)  # noqa: E731
pool = lambda s: [it for items in cur.train[s].values() for it in items]  # noqa: E731
echo, w1, w2 = pool("B"), pool("W1"), pool("W2")
tests = {"echo, new words": ("B", cur.test["B"]["echo:wide"][:200]),
         "pictures": ("W1", [it for src in ("cifar:name", "commons:name") for it in cur.test["W1"][src][:150]]),
         "first-words text": ("W1", cur.test["W1"]["curriculum"][:150]),
         "social": ("W1", cur.test["W1"].get("social", [])),
         "two words": ("W2", [it for items in cur.test["W2"].values() for it in items[:60]])}
results = {}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    pick = lambda items, n: [items[i] for i in rng.choice(len(items), min(n, len(items)), replace=False)]  # noqa: E731
    later = pick(w1, 9000) + pick(w2, 9000) + pick(echo, 3000)
    order = pick(echo, 5000) + [later[i] for i in rng.permutation(len(later))]
    for arm in ("always speaks", "own firing level"):
        words, wr, t0 = Words(7168, ear, setpoint=arm == "own firing level"), random.Random(seed), time.time()
        for it in order:
            words.learn(hear(cur.wording(it, wr)), vis(it[0]), hear(it[2][0]))
        row = {}
        for name, (stage, items) in tests.items():
            said = [words.recall(hear(it[1]), vis(it[0]))[0] for it in items]
            g = [grade(stage, s or "", it[2]) for s, it in zip(said, items)]
            spoke = [s is not None for s in said]
            row[name] = (float(np.mean(spoke)), float(np.mean(g)), float(np.mean([x for x, sp in zip(g, spoke) if sp] or [0])))
        results.setdefault(arm, []).append(row)
        n = len(words.units)
        print(f"seed {seed} {arm:16} ({time.time() - t0:.0f}s, levels on/off {words.m_on[1:n].mean():.2f}/{words.m_off[1:n].mean():.2f}): "
              + " | ".join(f"{k}: speaks {a:.2f}, right {b:.2f}, right when it speaks {c:.2f}" for k, (a, b, c) in row.items()), flush=True)
print(f"\nmean over seeds {SEEDS}:")
for arm, rows in results.items():
    print(f"  {arm:16} " + " | ".join(f"{k}: speaks {np.mean([r[k][0] for r in rows]):.2f}, right {np.mean([r[k][1] for r in rows]):.3f}±{np.std([r[k][1] for r in rows]):.3f}, right when it speaks {np.mean([r[k][2] for r in rows]):.2f}" for k in rows[0]))
json.dump(results, open("out/silence_probe.json", "w"))
