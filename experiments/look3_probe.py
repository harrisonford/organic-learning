"""Looking, third probe: a serial gaze, and whether the heard word can lead it (docs/VISION.md, 2026-11-23 21:45).

Today the word cells get the whole view and everything the eye fixated at once (look2_probe.py, the lives). Here the
word cells are raised exactly so, and then, with nothing learning, the looks are taken one at a time and we ask in
what order. Two orders of the same candidate places (organic/gaze.py finds them):

  reflex   the strongest contrast first: the newborn's gaze, what the learner has now
  word     the place whose coarse view (the periphery: the whole page at the view's coarseness, that place enlarged)
           most drives the cells of the words just heard, first. No new learning: it is the word cells' own visual
           synapses, read before the eye moves (the template account: Bichot et al. 2015; a heard word draws the
           eye to what it names: Tanenhaus et al. 1995, Bergelson & Swingley 2012)
  random   the same places in a shuffled order (the floor)

Measured on held-out pages (pictures never seen), "is there a dog?" and "how many dogs are there?":
  on which look the named thing is in the fovea whole, on pages where one candidate place holds it and another
  does not (where the order of looks can matter);
  the answer when the word cells are given only the first k looks (and the whole view), k = 1, 2, all.
Pages are twice the view (128 x 128, a thing 52 x 52 in each quarter), as in world 4.
    SEEDS=0,1,2 N=6000 python experiments/look3_probe.py
"""
import collections, os, pickle, random, re, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, tidy  # noqa: E402
from organic.scenes import page, Scenes, plural  # noqa: E402
from organic.gaze import fixations  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
N = int(os.environ.get("N", 6000))
TESTS = int(os.environ.get("TESTS", 300))
TILE, CELL, APART = 52, 64, 0.5
cur = Curriculum(seed=0, world=2)
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
cortex = pickle.load(open(os.environ.get("CORTEX", "data/visual_cortex2.pkl"), "rb"))
hear = lambda q: ear.units(q, learn=False)  # noqa: E731
code = lambda img: cortex.whole(img, 4).astype(np.float32)  # noqa: E731  (the summary through the relay, unit length)
DIM = sum(l.kinds for l in cortex.layers) * 16


def coarse(img):
    f = img.shape[0] // 64
    return img if f == 1 else img.reshape(64, f, 64, f, 3).mean(axis=(1, 3))


def zoom(img, r, c, side):
    w = img[r : r + side, c : c + side]
    return w if side == 64 else np.asarray(Image.fromarray((np.clip(w, 0, 1) * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR), np.float32) / 255


def by_name(side):
    d = collections.defaultdict(list)
    for src in ("cifar:name", "commons:name"):
        for it in side["W1"][src]:
            if cur.image(it[0]).max() > 0:
                d[it[2][0]].append(it[0])
    return d


lived, held = by_name(cur.train), by_name(cur.test)
names = sorted(lived)
name_of = {r: w for w, refs in held.items() for r in refs}


def make_page(tiles, shade):
    return page([cur.image(t) if t is not None else None for t in tiles], shade, TILE, CELL)


def looks_of(img):
    """The candidate places (reflex order), each as the fovea has it and as the periphery has it."""
    fix = fixations(img, apart=APART)
    side = img.shape[0] // 2
    far = coarse(img)
    k = img.shape[0] // far.shape[0]
    return fix, [code(zoom(img, r, c, side)) for r, c in fix], [code(zoom(far, r // k, c // k, side // k)) for r, c in fix]


def holds(fix, tile, size):
    """Which candidate places have the thing in quarter `tile` whole in the fovea (at least 0.9 of it)."""
    m = (CELL - TILE) // 2
    tr, tc = (tile // 2) * CELL + m, (tile % 2) * CELL + m
    out = []
    for r, c in fix:
        h = max(0, min(r + size, tr + TILE) - max(r, tr)) * max(0, min(c + size, tc + TILE) - max(c, tc))
        out.append(h >= 0.9 * TILE * TILE)
    return out


def run(seed):
    rng, wr = random.Random(seed), random.Random(seed + 1)
    words = Words(DIM, ear, match=True)
    singles = [(r, w) for w in names for r in lived[w]]
    lived_scenes, held_scenes = Scenes(lived, cur.image, seed, TILE, CELL), Scenes(held, cur.image, 99, TILE, CELL)
    t0 = time.time()

    def see(img):  # as the learner sees now: the whole view, and all the eye fixated at once
        _, sharp, _ = looks_of(img)
        words.look(sharp)
        return code(coarse(img))

    for n in range(N):
        kind = rng.choice(["one", "one", "one", "page", "there", "count"])
        if kind == "one":
            r, w = rng.choice(singles)
            words.learn(hear(wr.choice(cur.frames)), see(cur.image(r)), hear(w))
        elif kind == "page":
            w = rng.choice(names)
            on = [w] + [rng.choice([x for x in names if x != w]) for _ in range(rng.choice([0, 0, 1]))]
            tiles = [rng.choice(lived[x]) for x in on] + [None] * (4 - len(on))
            rng.shuffle(tiles)
            words.learn(hear(wr.choice(cur.frames)), see(make_page(tiles, rng.uniform(0.75, 0.95))), hear(w))
        else:
            tiles, shade, q, a = lived_scenes.make(kind, rng.randrange(10 ** 9))
            words.learn(hear(q), see(make_page(tiles, shade)), hear(a))
    print(f"seed {seed}: word cells raised on {N} moments ({time.time() - t0:.0f}s, {len(words.units)} cells)", flush=True)
    te = random.Random(1000 + seed)
    for kind in ("there", "count"):
        when = collections.defaultdict(list)  # order -> on which look the named thing was in the fovea
        right = collections.defaultdict(list)  # (order, how many looks) -> answers right
        places, pulled = [], []
        for n in range(TESTS):
            tiles, shade, q, a = held_scenes.make(kind, n)
            img = make_page(tiles, shade)
            fix, sharp, far = looks_of(img)
            heard, whole = hear(q), code(coarse(img))
            places.append(len(fix))
            asked = [w for w in names if re.search(rf"\b({re.escape(w)}|{re.escape(plural(w))})\b", q)]
            asked = max(asked, key=len) if asked else None
            cells = len(words.units)
            h = words._hearing(heard, cells)
            pull = np.zeros(len(fix), np.float32)
            if h is not None and fix:
                pull = h @ np.maximum(words.Wv[:cells] @ np.stack([words._visual(v) for v in far]).T, 0)
            orders = {"reflex": list(range(len(fix))), "word": [int(i) for i in np.argsort(-pull, kind="stable")], "random": te.sample(range(len(fix)), len(fix))}
            there = [k for k, t in enumerate(tiles) if t is not None and name_of.get(t) == asked]
            if there and fix:
                inside = np.any([holds(fix, k, img.shape[0] // 2) for k in there], axis=0)
                if inside.any() and (~inside).any():  # only pages where the order of looks can matter
                    pulled.append(float(pull[inside].max() > pull[~inside].max()))
                    for name, order in orders.items():
                        when[name].append(1 + next(i for i, j in enumerate(order) if inside[j]))
            for name, order in orders.items():
                for k in sorted({min(k, len(fix)) for k in (1, 2, len(fix))}):
                    words.look([sharp[j] for j in order[:k]])
                    right[(name, "all" if k == len(fix) else k)].append(tidy(words.recall(heard, whole)[0] or "") == a)
        base = collections.Counter(held_scenes.make(kind, n)[3] for n in range(TESTS)).most_common(1)[0][1] / TESTS
        print(f"seed {seed} {kind:5}: places a page {np.mean(places):.1f} | the named thing in one place and something else in another on {len(when['reflex'])} pages; "
              f"its place is the one the word pulls most on {np.mean(pulled):.2f} of them", flush=True)
        for name in ("reflex", "word", "random"):
            w = np.array(when[name])
            print(f"   {name:6}: the named thing in the fovea on look {w.mean():.2f} (on the first look {np.mean(w == 1):.2f}, by the second {np.mean(w <= 2):.2f}) | "
                  f"right with 1 look {np.mean(right[(name, 1)]):.3f} (n {len(right[(name, 1)])}), 2 looks {np.mean(right[(name, 2)]):.3f} (n {len(right[(name, 2)])}), "
                  f"all {np.mean(right[(name, 'all')]):.3f} | commonest answer {base:.2f}", flush=True)


for seed in SEEDS:
    run(seed)
