"""A new word, a thing it can name and a thing it cannot: which does the word go to? (docs/VISION.md, 2026-11-18)

Children from about 17 months take a new word for the thing that has no name yet ("mutual exclusivity", Markman &
Wachtel 1988; Halberda 2003). Text-only models trained by gradient on child-directed speech show the opposite
(survey of 2026-11-18); a robot that only associates words and objects showed the bias (Twomey et al. 2016). It is
the behaviour test the survey rated highest for this brain. Nothing in the brain was built for it.

The word cells (new cortex and relay, wide pages, the gaze, match neurons, full-size corrections: as the lives of
world 4) live N moments: single pictures named, things on pages named, "is there a ...?". HOLD kinds of thing are
never shown or named. Then, each trial from the same grown cells:
  a page with a thing of a known kind and a thing of a held-out kind, both pictures never seen; a made-up word is
  said of it once, as a name is ("what is this?" - "dax").
What is read after that one moment:
  drawn to    the new word's cell is driven by sight of the unnamed thing more than of the named one (each seen
              alone, filling the view). Chance 0.5. This is the looking measure of the infant studies.
  ... another the same with other pictures of the two kinds (does it hold for the kind, not the picture?)
  says it     asked "what is this?" of the unnamed thing alone, it says the new word; and of the named thing alone
              (it should not); and whether the named thing keeps its own name.
Control: two things of held-out kinds on the page (the word has no reason to go to either: the first is "its" one).
    SEEDS=0,1,2 N=6000 python experiments/newword_probe.py
"""
import collections, copy, os, pickle, random, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, tidy  # noqa: E402
from organic.scenes import page, Scenes  # noqa: E402
from organic.gaze import fixations  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
N, HOLD, TRIALS = int(os.environ.get("N", 6000)), int(os.environ.get("HOLD", 4)), int(os.environ.get("TRIALS", 60))
LESSONS = int(os.environ.get("LESSONS", 1))  # how many times the new word is said of the page
TILE, CELL, APART = 52, 64, 0.5
NEW = "dax wug blick toma fep zav gorp modi kiv pim sib tulver nop juf tiz bem cado lorp vash keet".split()
cur = Curriculum(seed=0, world=2)
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
cortex = pickle.load(open(os.environ.get("CORTEX", "data/visual_cortex2.pkl"), "rb"))
hear = lambda q: ear.units(q, learn=False)  # noqa: E731
CODES = {}


def code(img):
    return cortex.whole(img, 4).astype(np.float32)


def coarse(img):
    f = img.shape[0] // 64
    return img if f == 1 else img.reshape(64, f, 64, f, 3).mean(axis=(1, 3))


def zoom(img, r, c):
    side = img.shape[0] // 2
    w = img[r : r + side, c : c + side]
    return w if side == 64 else np.asarray(Image.fromarray((np.clip(w, 0, 1) * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR), np.float32) / 255


def views(key, make):
    """(the whole view, what the gaze fixated) of a picture or page, coded once."""
    if key not in CODES:
        img = make()
        CODES[key] = (code(coarse(img)).astype(np.float16), [code(zoom(img, r, c)).astype(np.float16) for r, c in fixations(img, apart=APART)])
    whole, fix = CODES[key]
    return whole.astype(np.float32), [f.astype(np.float32) for f in fix]


def by_name(side):
    d = collections.defaultdict(list)
    for src in ("cifar:name", "commons:name"):
        for it in side["W1"][src]:
            if cur.image(it[0]).max() > 0:
                d[it[2][0]].append(it[0])
    return d


lived_all, held_all = by_name(cur.train), by_name(cur.test)


def a_page(tiles, shade):
    return ("page", tuple(tiles), round(shade, 3)), (lambda: page([cur.image(t) if t is not None else None for t in tiles], shade, TILE, CELL))


for seed in SEEDS:
    CODES.clear()
    rng, wr, t0 = random.Random(seed), random.Random(seed + 1), time.time()
    kinds = sorted(lived_all)
    unknown = sorted(random.Random(seed + 5).sample(kinds, HOLD))  # kinds of thing never shown or named
    names = [w for w in kinds if w not in unknown]
    lived = {w: lived_all[w] for w in names}
    words = Words(cortex.relay.Z.shape[0], ear, match=True)
    words.plastic_by_error = False
    singles = [(r, w) for w in names for r in lived[w]]
    scenes = Scenes(lived, cur.image, seed, TILE, CELL)

    def see(key, make, w_=None):
        whole, fix = views(key, make)
        (w_ or words).look(fix)
        return whole

    for n in range(N):
        kind = rng.choice(["one", "one", "one", "page", "page", "there"])
        if kind == "one":
            r, w = rng.choice(singles)
            words.learn(hear(wr.choice(cur.frames)), see(("one", r), lambda r=r: cur.image(r)), hear(w))
        elif kind == "page":  # a thing on a page (a third of the time another beside it), named when asked
            w = rng.choice(names)
            on = [w] + [rng.choice([x for x in names if x != w]) for _ in range(rng.choice([0, 0, 1]))]
            tiles = [rng.choice(lived[x]) for x in on] + [None] * (4 - len(on))
            rng.shuffle(tiles)
            key, make = a_page(tiles, rng.uniform(0.75, 0.95))
            words.learn(hear(wr.choice(cur.frames)), see(key, make), hear(w))
        else:
            tiles, shade, q, a = scenes.make("there", rng.randrange(10 ** 9))
            key, make = a_page(tiles, shade)
            words.learn(hear(q), see(key, make), hear(a))
    ask = hear("what is this?")
    known = float(np.mean([tidy(words.recall(ask, see(("one", r), lambda r=r: cur.image(r)))[0] or "") == w for w in names for r in held_all[w][:12]]))
    grown, te = words, random.Random(seed + 99)
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for trial in range(TRIALS):
        for cond in ("named + unnamed", "two unnamed"):
            w = copy.deepcopy(grown, {id(ear): ear})
            new = NEW[trial % len(NEW)]
            u, k = te.choice(unknown), te.choice(names)
            if cond == "two unnamed":
                k = te.choice([x for x in unknown if x != u])
            pool = held_all if cond == "named + unnamed" else {**held_all}
            pu, pk = te.sample(held_all[u], 2), te.sample(pool[k], 2)  # the pictures on the page, and another of each kind
            tiles = [pu[0], pk[0], None, None]
            te.shuffle(tiles)
            key, make = a_page(tiles, te.uniform(0.75, 0.95))
            for _ in range(LESSONS):
                w.learn(hear(te.choice(cur.frames)), see(key, make, w), hear(new))
            cell = w.index.get(hear(new)[0])
            alone = lambda r: views(("one", r), lambda r=r: cur.image(r))[0]  # noqa: E731  (the thing alone, filling the view)
            drive = lambda r: float(w.Wv[cell] @ alone(r))  # noqa: E731
            says = lambda r: tidy(w.recall(ask, see(("one", r), lambda r=r: cur.image(r), w))[0] or "")  # noqa: E731
            r_ = rows[cond]
            r_["drawn to the unnamed thing"].append(float(drive(pu[0]) > drive(pk[0])))
            r_["... another of each kind"].append(float(drive(pu[1]) > drive(pk[1])))
            r_["drive: unnamed"].append(drive(pu[0]))
            r_["drive: the other"].append(drive(pk[0]))
            r_["says the new word of the unnamed thing"].append(float(says(pu[0]) == new))
            r_["says the new word of the other"].append(float(says(pk[0]) == new))
            if cond == "named + unnamed":
                r_["the named thing keeps its name"].append(float(says(pk[0]) == k))
                before = tidy(grown.recall(ask, see(("one", pk[0]), lambda r=pk[0]: cur.image(r), grown))[0] or "")
                r_["(it had its name before)"].append(float(before == k))
    print(f"seed {seed} ({time.time() - t0:.0f}s; {len(names)} kinds named, never shown: {', '.join(unknown)}; held-out naming of the known kinds {known:.3f}; {TRIALS} trials, the word said {LESSONS}x)", flush=True)
    for cond, r_ in rows.items():
        print(f"  {cond:16} " + " | ".join(f"{k_} {np.mean(v):.3f}" for k_, v in r_.items()), flush=True)
