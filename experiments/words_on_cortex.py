"""Do the word cells name more with another cortex? (docs/VISION.md, 2026-11-17)

A linear reference says what a code holds; the brain's word cells learn by their own local rule, online, each
picture seen once. Here the real word cells (organic/words.py) live N naming moments of world 3's first words
(CIFAR, Commons and COCO things, the sources taking turns, the question worded afresh each time) with each cortex,
and then name held-out pictures of each source.
    CORTEX=data/visual_cortex.pkl,out/tmp/cortex2_a.pkl SEEDS=0,1,2 N=8000 python experiments/words_on_cortex.py

Where the gap to the linear reference is (2026-11-17), ARMS (comma-separated):
  base       as above
  ridge      no word cells: a ridge fitted on the very pictures the word cells lived (what a linear readout can get from them)
  passes3    the same lived moments three times over (is it only that one look at each picture is not enough?)
  white      the code decorrelated before it reaches the word cells (from the statistics of 3,000 lived pictures)
  rate       every visual synapse learns as if it were the only input (the correction is not shared with the cues)
Arms can be joined with "+", and take a number: white0.01 (how much of the largest variance is added to every
direction before decorrelating: less = stronger decorrelation), gain3 (every correction three times as large).
The codes of a cortex are kept in out/tmp/woc_<cortex>.pkl so that further arms take seconds.
"""
import os, pickle, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, tidy  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
N, GRID = int(os.environ.get("N", 8000)), int(os.environ.get("GRID", 4))
ARMS = os.environ.get("ARMS", "base").split(",")
PATHS = os.environ.get("CORTEX", "data/visual_cortex.pkl").split(",")
cur = Curriculum(seed=0, world=3)
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
hear = lambda q: ear.units(q, learn=False)  # noqa: E731
SRC = ("cifar:name", "commons:name", "coco:name")
lived = {s: [it for it in cur.train["W1"][s] if cur.image(it[0]).max() > 0] for s in SRC}
held = {s: [it for it in cur.test["W1"][s] if cur.image(it[0]).max() > 0][:200] for s in SRC}

for path in PATHS:
    cortex = pickle.load(open(path, "rb"))
    dim = sum(l.kinds for l in cortex.layers) * GRID * GRID
    cpath = f"out/tmp/woc_{os.path.basename(path)}"
    cache = pickle.load(open(cpath, "rb")) if os.path.exists(cpath) else {}
    had = len(cache)

    def code(ref):
        if ref not in cache:
            if getattr(cortex, "relay", None) is not None:  # a cortex with its own relay: the whole view as the brain gets it
                cache[ref] = cortex.whole(cur.image(ref), GRID).astype(np.float32)
                return cache[ref]
            v = np.concatenate(cortex.code(cur.image(ref), grid=GRID))
            cache[ref] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float16)
        return cache[ref].astype(np.float32)

    import re
    num = lambda arm, key, default: next((float(m.group(1) or default) for part in arm.split("+") for m in [re.fullmatch(key + r"([0-9.]*)", part)] if m), None)  # noqa: E731
    ZS = {}
    if any("white" in a for a in ARMS):
        r0 = random.Random(123)
        S = np.stack([code(r0.choice(lived[r0.choice(SRC)])[0]) for _ in range(3000)])
        mu = S.mean(0)
        d, E = np.linalg.eigh(np.cov((S - mu).T))
        for shrink in {num(a, "white", 0.05) for a in ARMS if "white" in a}:
            ZS[shrink] = ((E / np.sqrt(np.maximum(d, 0) + shrink * d.max())) @ E.T).astype(np.float32)
    wcache = {}

    def seen(ref, arm):
        shrink = num(arm, "white", 0.05)
        if shrink is None:
            return code(ref)
        if (ref, shrink) not in wcache:
            v = (code(ref) - mu) @ ZS[shrink]
            wcache[(ref, shrink)] = v / (np.linalg.norm(v) + 1e-8)
        return wcache[(ref, shrink)]

    for seed in SEEDS:
        for arm in ARMS:
            rng, t0 = random.Random(seed), time.time()
            moments = [(rng.choice(lived[rng.choice(SRC)]), rng.choice(cur.frames)) for _ in range(N)]
            if arm == "ridge":
                names = sorted({it[2][0] for it, _ in moments})
                A = np.stack([code(it[0]) for it, _ in moments]); mean = A.mean(0); A = A - mean
                Y = np.eye(len(names), dtype=np.float32)[[names.index(it[2][0]) for it, _ in moments]]
                K = A @ A.T
                best = None
                for lam in (0.03, 0.1, 0.3, 1.0):
                    Kl = K.copy(); Kl[np.diag_indices_from(Kl)] += lam
                    Wd = A.T @ np.linalg.solve(Kl, Y)
                    row = {s_: float(np.mean([names[int(np.argmax((code(it[0]) - mean) @ Wd))] == it[2][0] for it in held[s_]])) for s_ in SRC}
                    if best is None or np.mean(list(row.values())) > np.mean(list(best.values())):
                        best = row
                row, right = best, [np.nan]
            else:
                words = Words(dim, ear)
                words.gain = num(arm, "gain", 1.0) or 1.0
                words.plastic_by_error = "full" not in arm.split("+")
                right = []
                for p in range(int(num(arm, "passes", 3) or 1)):
                    for n, (it, frame) in enumerate(moments):
                        q, v = hear(frame), seen(it[0], arm)
                        if p == 0 and n >= N - 1500:  # while living: was the name expected before it was heard?
                            right.append(tidy(words.recall(q, v)[0] or "") == it[2][0])
                        if "rate" in arm.split("+"):
                            words.learn([], v, hear(it[2][0]))  # nothing heard before the name: the picture is the only input besides the baseline
                        else:
                            words.learn(q, v, hear(it[2][0]))
                ask = [] if "rate" in arm.split("+") else hear("what is this?")
                row = {s_: float(np.mean([tidy(words.recall(ask, seen(it[0], arm))[0] or "") == it[2][0] for it in held[s_]])) for s_ in SRC}
            print(f"{os.path.basename(path):22} {arm:22} seed {seed} ({time.time() - t0:.0f}s, code {dim}): lived, last 1,500: {np.mean(right):.3f} | held out: "
                  + " | ".join(f"{s_} {v:.3f}" for s_, v in row.items()) + f" | mean {np.mean(list(row.values())):.3f}", flush=True)
    if len(cache) > had:
        pickle.dump(cache, open(cpath, "wb"))
