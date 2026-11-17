"""Fast mapping: does a word heard once with a new kind of thing stay? (docs/RESEARCH_MEMORY.md, 2026-11-17)

The memory question in the form it has at this age. A two-year-old hears "look, a koba!" once and picks the koba out
a moment later (Carey & Bartlett 1978), and has mostly lost it five minutes later unless it was named again (Horst
& Samuelson 2008). The word cells have slow synapses only; with corrections at full size one hearing moves them a
whole step. Is that a memory of one moment?

The word cells first live N naming moments of first words. Then, for each of about 65 kinds of things they were
never told the name of (CIFAR kinds of later stages), with pictures a and b of the kind:

    the name is heard E times, each time with another picture of the kind (as an unasked comment "look, a fox!",
    or as a lesson "what is this?" - "fox"); the last picture is a
    asked at once:      the same picture a | another picture b of the kind
    then 20 ordinary naming moments
    asked again:        a | b

Also: held-out naming of the first words before and after all of it (does fast mapping cost the old names?).
    SEEDS=0,1,2 CORTEX=data/visual_cortex2.pkl python experiments/fastmap_probe.py
"""
import os, pickle, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, tidy  # noqa: E402
from organic.episodes import Episodes  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
N, GRID = int(os.environ.get("N", 6000)), 4
cur = Curriculum(seed=0, world=3)
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
cortex = pickle.load(open(os.environ.get("CORTEX", "data/visual_cortex2.pkl"), "rb"))
dim = sum(l.kinds for l in cortex.layers) * GRID * GRID
hear = lambda q: ear.units(q, learn=False)  # noqa: E731
SRC = ("cifar:name", "commons:name", "coco:name")
lived = {s: [it for it in cur.train["W1"][s] if cur.image(it[0]).max() > 0] for s in SRC}
held = [it for s in SRC for it in cur.test["W1"][s][:100] if cur.image(it[0]).max() > 0]
new = {}
for stage in ("P", "G1"):
    for it in cur.train[stage]["cifar:name"]:
        if " " not in it[2][0]:
            new.setdefault(it[2][0], []).append(it[0])
cache = {}


def code(ref):
    if ref not in cache:
        if getattr(cortex, "relay", None) is not None:
            cache[ref] = cortex.whole(cur.image(ref), GRID).astype(np.float32)
        else:
            v = np.concatenate(cortex.code(cur.image(ref), grid=GRID))
            cache[ref] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
    return cache[ref]


ASK = "what is this?"
says = lambda words, ref: tidy(words.recall(hear(ASK), code(ref))[0] or "")  # noqa: E731
ARMS = [(a.startswith("full"), a.endswith("+store")) for a in os.environ.get("ARMS", "own,full").split(",")]
print(f"{len(new)} kinds never named in first words; {len(held)} held-out first-words pictures", flush=True)
# what one picture of a kind is worth to the eyes: with one stored picture of each new kind, is another picture of a
# kind closest to its own kind's stored picture? (chance: one in the number of kinds)
r0 = random.Random(7)
pairs = {w: r0.sample(refs, 2) for w, refs in new.items()}
K = np.stack([code(pairs[w][0]) for w in sorted(new)])
near = np.mean([int(np.argmax(K @ code(pairs[w][1]))) == i for i, w in enumerate(sorted(new))])
print(f"one picture of each of {len(new)} kinds stored: another picture of a kind is nearest to its own kind's {near:.2f} of the time", flush=True)
for seed in SEEDS:
    for full, store in ARMS:
        for how in ("comment", "lesson"):
            for E in (1, 3):
                rng, t0 = random.Random(seed), time.time()
                words = Words(dim, ear)
                words.plastic_by_error = not full
                if store:
                    words.episodes = Episodes(dim, ear.width, capacity=int(os.environ.get("CAPACITY", 2000)), steep=float(os.environ.get("STEEP", 12)))
                ordinary = lambda: (lambda it: words.learn(hear(rng.choice(cur.frames)), code(it[0]), hear(it[2][0])))(rng.choice(lived[rng.choice(SRC)]))  # noqa: E731
                for _ in range(N):
                    ordinary()
                before = float(np.mean([says(words, it[0]) == it[2][0] for it in held]))
                rows = []
                for w in rng.sample(sorted(new), len(new)):
                    pics = rng.sample(new[w], E + 1)
                    for p in pics[:E]:
                        if how == "comment":
                            frame = rng.choice(cur.comments)
                            words.learn([], code(p), hear("look, " + frame.replace("a THING", ("an " if w[0] in "aeiou" else "a ") + w).replace("THING", w)))
                        else:
                            words.learn(hear(rng.choice(cur.frames)), code(p), hear(w))
                    a, b = pics[E - 1], pics[E]
                    now = [says(words, a) == w, says(words, b) == w]
                    for _ in range(20):
                        ordinary()
                    rows.append(now + [says(words, a) == w, says(words, b) == w])
                after = float(np.mean([says(words, it[0]) == it[2][0] for it in held]))
                r = np.mean(rows, axis=0)
                print(f"seed {seed} {'full-size' if full else 'by own error':12} {how + (' + store' if store else ''):15} heard {E}x ({time.time() - t0:.0f}s): at once: the same picture {r[0]:.2f}, another of the kind {r[1]:.2f} | "
                      f"after 20 moments: {r[2]:.2f}, {r[3]:.2f} | first words held out {before:.3f} → {after:.3f}", flush=True)
