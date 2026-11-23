"""A name heard once for a thing on a page: what the fast store keeps it under (docs/VISION.md, 2026-11-23).

`page_views_probe.py` found that the episodic store, written and read under the whole view, gives back another
page's name on a page (coarse pages look alike). `Episodes.key = "looked"` keeps a moment under what the eye was on
instead. On held-out pages that takes the harm away. This probe asks for the store's own job there: a child shown a
page, "look, a fox!", once.

The word cells first live N naming moments: single pictures and pages (a thing on a page, a third of the time with
another beside it), looking as the brain does (the whole view coarsely, and what the gaze fixates). Then, for each of
about 65 kinds of things never named before, with pictures a and b of the kind:
    picture a is on a page (alone, or a known thing beside it) and its name is heard once, when asked
    asked at once, and again after 20 ordinary moments:
        the very page | picture a on another page, in another place | picture a alone, filling the view |
        picture b of the kind on a page
Arms: the word cells alone; with the store keyed by the whole view; with the store keyed by what the eye is on.
Also: held-out first-words naming, single pictures and pages, before and after (what does it cost the old names?).
    SEEDS=0,1,2 N=6000 python experiments/fastmap_page_probe.py
"""
import os, pickle, random, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, tidy  # noqa: E402
from organic.episodes import Episodes  # noqa: E402
from organic.scenes import page  # noqa: E402
from organic.gaze import fixations  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
ARMS = os.environ.get("ARMS", "cells,store:whole,store:looked").split(",")
N, GRID, TILE, CELL, APART = int(os.environ.get("N", 6000)), 4, 52, 64, 0.5
cur = Curriculum(seed=0, world=3)
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
cortex = pickle.load(open(os.environ.get("CORTEX", "data/visual_cortex2.pkl"), "rb"))
dim = sum(l.kinds for l in cortex.layers) * GRID * GRID
hear = lambda q: ear.units(q, learn=False)  # noqa: E731
ASK = "what is this?"


def by_name(side):
    d = {}
    for src in ("cifar:name", "commons:name"):
        for it in side["W1"][src]:
            if cur.image(it[0]).max() > 0:
                d.setdefault(it[2][0], []).append(it[0])
    return d


lived, held = by_name(cur.train), by_name(cur.test)
names = sorted(lived)
new = {}
for stage in ("P", "G1"):
    for it in cur.train[stage]["cifar:name"]:
        if " " not in it[2][0] and it[2][0] not in lived:
            new.setdefault(it[2][0], []).append(it[0])
CODES = {}


def views(key):
    """(the whole view, what the gaze fixated) of a single picture ("one", ref) or a page ("page", tiles, shade)."""
    if key not in CODES:
        img = cur.image(key[1]) if key[0] == "one" else page([cur.image(t) if t is not None else None for t in key[1]], key[2], TILE, CELL)
        f = img.shape[0] // 64
        whole = img if f == 1 else img.reshape(64, f, 64, f, 3).mean(axis=(1, 3))
        side = img.shape[0] // 2
        zoom = lambda r, c: (lambda w: w if side == 64 else np.asarray(Image.fromarray((np.clip(w, 0, 1) * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR), np.float32) / 255)(img[r : r + side, c : c + side])  # noqa: E731
        CODES[key] = (cortex.whole(whole, GRID).astype(np.float16), [cortex.whole(zoom(r, c), GRID).astype(np.float16) for r, c in fixations(img, apart=APART)])
    whole, fix = CODES[key]
    return whole.astype(np.float32), [x.astype(np.float32) for x in fix]


def a_page(ref, rng, beside=None):
    """A page with this picture in a random quarter (and a known thing beside it, or nothing)."""
    tiles = [ref] + ([rng.choice(lived[rng.choice(names)])] if beside else []) + [None] * (3 - bool(beside))
    rng.shuffle(tiles)
    return ("page", tuple(tiles), round(rng.uniform(0.75, 0.95), 3))


print(f"{len(new)} kinds never named in first words; {len(names)} first-words names", flush=True)
for seed in SEEDS:
    for arm in ARMS:
        rng, wr, t0 = random.Random(seed), random.Random(seed + 1), time.time()
        words = Words(dim, ear)
        words.plastic_by_error = False
        if arm.startswith("store"):
            words.episodes = Episodes(dim, ear.width)
            words.episodes.key = arm.split(":")[1]

        def see(key):
            whole, fix = views(key)
            words.look(fix)
            return whole

        says = lambda key: tidy(words.recall(hear(ASK), see(key))[0] or "")  # noqa: E731

        def ordinary():
            w = rng.choice(names)
            key = ("one", rng.choice(lived[w])) if rng.random() < 0.5 else a_page(rng.choice(lived[w]), rng, rng.random() < 1 / 3)
            words.learn(hear(wr.choice(cur.frames)), see(key), hear(w))

        for _ in range(N):
            ordinary()
        tests = [(w, r) for w in sorted(held) if w in lived for r in held[w][:4]]
        hr = random.Random(99)
        test_pages = [(w, a_page(r, hr)) for w, r in tests]
        old = lambda: (float(np.mean([says(("one", r)) == w for w, r in tests])), float(np.mean([says(k) == w for w, k in test_pages])))  # noqa: E731
        before, rows = old(), []
        for w in rng.sample(sorted(new), len(new)):
            a, b = rng.sample(new[w], 2)
            first = a_page(a, rng, rng.random() < 1 / 3)
            words.learn(hear(wr.choice(cur.frames)), see(first), hear(w))  # the name, heard once
            asks = [first, a_page(a, rng), ("one", a), a_page(b, rng)]
            now = [says(k) == w for k in asks]
            for _ in range(20):
                ordinary()
            rows.append(now + [says(k) == w for k in asks])
        after, r = old(), np.mean(rows, axis=0)
        print(f"seed {seed} {arm:13} ({time.time() - t0:.0f}s) at once: the very page {r[0]:.2f} | the picture on another page {r[1]:.2f} | the picture alone {r[2]:.2f} | another of the kind on a page {r[3]:.2f} || "
              f"after 20 moments: {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.2f} || first words held out, single pictures {before[0]:.3f} → {after[0]:.3f}, on a page {before[1]:.3f} → {after[1]:.3f}", flush=True)
