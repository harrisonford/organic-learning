"""Words made of the ear's own units vs the hub (offline; docs/AREAS.md).

The same moments, in the same order, are lived by
  hub, additive        organic/hub.py as first built (count tables + a Cortex visual area + answer strings)
  hub, kinds           the same hub with answer kinds regrouped in sleep (as in the last life)
  words                organic/words.py: the ear's own units as cells; a content dendrite (picture, heard cues)
                       and a context dendrite (the echo of the exchange), each learning to predict the cell's firing
  words, xN            the same after N passes over the moments (a life sees its pictures several times)

Measured on held-out items:
  W1 pictures, each asked three ways: right, the same answer all three times, right all three times
  W1 adult items (the informative word is in the question)
  W2 items (colour, yes/no, counting, two-word phrases), graded as the curriculum grades two words
TRAIN=w1 (first words only) or w1w2 (a life at two words: both). SEEDS=0,1,2 (the order of moments).
    TRAIN=w1w2 python experiments/areas_probe.py
"""
import json, os, pickle, random, sys, time, zlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, grade, tidy  # noqa: E402
from organic.hub import Hub  # noqa: E402
from organic.world import WORLD_DIR  # noqa: E402

TRAIN = os.environ.get("TRAIN", "w1w2")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
N = int(os.environ.get("N", 9000))
CELLS = int(os.environ.get("CELLS", 0)) or None  # the hub's visual cells (a life gives it 1,500; None: one per picture if it wants)
PASSES = int(os.environ.get("PASSES", 3))  # a life sees its pictures several times: the words are also measured after PASSES passes
cur = Curriculum(seed=0)
ear = pickle.load(open(os.environ.get("EAR", "out/kid_s3.ear.pkl"), "rb"))
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
FRAMES = json.load(open(os.path.join(WORLD_DIR, "naming_frames.json")))["frames"]
V = {}


def vis(ref):
    if ref not in V:
        img = cur.image(ref)
        if img.max() == 0:
            V[ref] = None
        else:
            v = np.concatenate(cortex.code(img, grid=4))
            V[ref] = v / (np.linalg.norm(v) + 1e-8)
    return V[ref]


def hear(q):
    return ear.units(q, learn=False)


w1 = [it for items in cur.train["W1"].values() for it in items]
w2 = [it for items in cur.train["W2"].values() for it in items]
test_pics = [it for src in ("cifar:name", "commons:name") for it in cur.test["W1"][src][:150]]
test_text = cur.test["W1"].get("curriculum", [])[:150]
test_w2 = [it for items in cur.test["W2"].values() for it in items[:60]]
dim = next(vis(it[0]) for it in w1 if vis(it[0]) is not None).size
results = {}

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    order = [w1[i] for i in rng.choice(len(w1), min(N, len(w1)), replace=False)]
    if TRAIN == "w1w2":
        order += [w2[i] for i in rng.choice(len(w2), min(N, len(w2)), replace=False)]
    order = [order[i] for i in rng.permutation(len(order))]
    hub = Hub(dim, seed=seed, max_cells=CELLS)
    words = Words(dim, ear)
    t0 = time.time()
    wr = random.Random(seed)
    asked = [cur.wording(it, wr) for it in order]  # a naming question is worded afresh each time, as in a life
    for q, (ref, _, ans, st, src) in zip(asked, order):
        hub.learn(hear(q), vis(ref), ans[0])
    t1 = time.time()
    for p in range(PASSES):
        for it in order:
            words.learn(hear(cur.wording(it, wr)), vis(it[0]), hear(it[2][0]))
        if p == 0:
            once = pickle.loads(pickle.dumps(words))
    t2 = time.time()
    print(f"seed {seed}: {len(order)} moments | hub {hub.area.C} visual cells, {len(hub.lexicon)} answer strings, {t1 - t0:.0f}s | "
          f"words {words.cells()}, {t2 - t1:.0f}s for {PASSES} passes", flush=True)

    def hub_add(q, v):
        votes = hub.votes(hear(q), v)
        return hub.lexicon[int(np.argmax(votes))] if votes.size and votes.sum() > 0 else ""

    methods = {"hub, additive": hub_add,
               "hub, kinds": lambda q, v: hub.recall(hear(q), v)[0] or "",
               "words": lambda q, v: once.recall(hear(q), v)[0] or "",
               f"words, x{PASSES}": lambda q, v: words.recall(hear(q), v)[0] or ""}
    for name, f in methods.items():
        acc, same, every = [], 0, 0
        for it in test_pics:
            asks = [FRAMES[(zlib.crc32(repr(it[0]).encode()) + 17 * j + 5) % len(FRAMES)] for j in range(3)]
            said = [f(q, vis(it[0])) for q in asks]
            ok = [tidy(s) in [tidy(a) for a in it[2]] for s in said]
            acc += ok
            same += len(set(said)) == 1
            every += all(ok)
        txt = float(np.mean([grade("W1", f(it[1], None), it[2]) for it in test_text]))
        w2s = float(np.mean([grade("W2", f(it[1], vis(it[0])), it[2]) for it in test_w2]))
        row = [float(np.mean(acc)), same / len(test_pics), every / len(test_pics), txt, w2s]
        results.setdefault(name, []).append(row)
        print(f"  {name:16} W1 pictures: right {row[0]:.3f} | same x3 {row[1]:.3f} | right x3 {row[2]:.3f} || W1 adult items {row[3]:.3f} || W2 {row[4]:.3f}", flush=True)
    if seed == SEEDS[0]:  # what the tissue says, and what its echo area expects after a naming frame and a colour question
        for q, ref in [(test_pics[0][1], test_pics[0][0]), (test_pics[200][1], test_pics[200][0])] + [(it[1], it[0]) for it in test_w2[::37][:8]]:
            act, drive, ctx = words.expect(hear(q), vis(ref))
            top = [words.units[k].decode(errors="replace") or "(silence)" for k in np.argsort(-ctx)[:6]]
            con = [words.units[k].decode(errors="replace") for k in np.argsort(-drive)[:4]]
            print(f"    {q[:44]!r:46} -> {words.recall(hear(q), vis(ref))[0]!r:14} context expects {top} | content drives {con}", flush=True)

print(f"\nTRAIN={TRAIN}, mean over seeds {SEEDS} (and the spread between seeds):")
for name, rows in results.items():
    m, s = np.mean(rows, axis=0), np.std(rows, axis=0)
    print(f"  {name:16} W1 pictures: right {m[0]:.3f}±{s[0]:.3f} | same x3 {m[1]:.3f}±{s[1]:.3f} | right x3 {m[2]:.3f}±{s[2]:.3f} || W1 adult {m[3]:.3f}±{s[3]:.3f} || W2 {m[4]:.3f}±{s[4]:.3f}")
json.dump(results, open(f"out/areas_probe_{TRAIN}_{CELLS or 'all'}.json", "w"))
