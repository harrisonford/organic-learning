"""Looking: the word cells see what the eye fixates (offline; docs/VISION.md, option 1).

The word cells have only ever had the whole view. An eye fixates: the thing looked at fills the view. Here a moment
has several views (the whole page, then what the eye fixated), each word cell is driven by the view that drives it
most, and when its word is heard it learns from that view (a word said over a page of several things is about one
of them: cross-situational learning, Yu & Smith 2007).

The world: single pictures named as in first words, and pages (organic/scenes.py) with a thing in one quarter, named
when asked. The eye here simply fixates the four quarters (in a brain the gaze has to find them).

  whole     the whole view only, as every life so far
  looks     the whole view and the four fixations, on pages and on single pictures alike
Held out (pictures never seen): single pictures named; a thing on a page named; a thing named when two others are on the page.
    SEEDS=0,1,2 python experiments/look_probe.py
"""
import collections, os, pickle, random, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, tidy  # noqa: E402
from organic.scenes import page  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
N = int(os.environ.get("N", 6000))
cur = Curriculum(seed=0, world=2)
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
cortex = pickle.load(open(os.environ.get("CORTEX", "data/visual_cortex.pkl"), "rb"))
hear = lambda q: ear.units(q, learn=False)  # noqa: E731
CODES = {}


def code(img):
    v = np.concatenate(cortex.code(img, grid=4))
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def views(key, make):
    """(whole view, the four fixations) of a picture or page, coded once."""
    if key not in CODES:
        img = make()
        quarters = [np.asarray(Image.fromarray((img[r : r + 32, c : c + 32] * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR), np.float32) / 255 for r in (0, 32) for c in (0, 32)]
        CODES[key] = (code(img), [code(q) for q in quarters])
    return CODES[key]


def by_name(side):
    d = collections.defaultdict(list)
    for src in ("cifar:name", "commons:name"):
        for it in side["W1"][src]:
            if cur.image(it[0]).max() > 0:
                d[it[2][0]].append(it[0])
    return d


lived, held = by_name(cur.train), by_name(cur.test)
names = sorted(lived)


def a_page(pool, rng, others):
    """A page with one named thing in a random quarter and `others` other things: (key, maker, name)."""
    w = rng.choice(names)
    tiles = [rng.choice(pool[w])] + [rng.choice(pool[rng.choice([x for x in names if x != w])]) for _ in range(others)]
    tiles += [None] * (4 - len(tiles))
    order = list(range(4))
    rng.shuffle(order)
    tiles = [tiles[i] for i in order]
    shade = rng.uniform(0.75, 0.95)
    return ("page", tuple(tiles), round(shade, 3)), (lambda: page([cur.image(t) if t is not None else None for t in tiles], shade)), w


def run(arm, seed):
    rng, wr = random.Random(seed), random.Random(seed + 1)
    words = Words(7168, ear)
    singles = [(r, w) for w in names for r in lived[w]]
    t0 = time.time()
    for n in range(N):
        if rng.random() < 0.5:  # a single picture, named when asked
            r, w = rng.choice(singles)
            key, make = ("one", r), (lambda r=r: cur.image(r))
        else:  # a page with the thing in one quarter (half of the time with another thing beside it)
            key, make, w = a_page(lived, rng, rng.choice([0, 0, 1]))
        whole, looks = views(key, make)
        words.look(looks if arm == "looks" else None).learn(hear(wr.choice(cur.frames)), whole, hear(w))
    row = {}
    te = random.Random(99)
    tests = {"single pictures": [(("one", r), (lambda r=r: cur.image(r)), w) for w in names for r in held[w]][:300],
             "a thing on a page": [a_page(held, te, 0) for _ in range(300)],
             "a thing on a page, another beside it": [a_page(held, te, 1) for _ in range(300)]}
    for name, items in tests.items():
        ok = []
        for key, make, w in items:
            whole, looks = views(key, make)
            said = words.look(looks if arm == "looks" else None).recall(hear("what is this?"), whole)[0] or ""
            ok.append(tidy(said) == w)
        row[name] = float(np.mean(ok))
    print(f"seed {seed} {arm:6} ({time.time() - t0:.0f}s, {len(words.units)} cells): " + " | ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)
    return row


for seed in SEEDS:
    for arm in ("whole", "looks"):
        run(arm, seed)
