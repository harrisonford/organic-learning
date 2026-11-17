"""Looking, second probe: where the eye looks, and answering about what it looked at (docs/VISION.md, 2026-11-17).

The first probe (look_probe.py) gave the word cells the four quarters of the view by rule and asked only for names.
Here:

  where the eye looks
    whole     the whole view only (every life so far)
    quarters  the whole view and its four quarters, by rule (the reference: a gaze that always finds the things)
    gaze      the whole view and what stands out: a window of half the view is fixated where its inside differs
              from the scene's average colour more than its surround does (centre-surround), the strongest first,
              never the same place twice, until nothing stands out from its surround any more

  what is asked (lived half and half: single pictures named; pages)
    a thing on a page named, alone or with another beside it
    "is there a dog?"          yes / no          (organic/scenes.py)
    "how many dogs are there?" one / two / three

  +match    the word cells have match neurons (organic/words.py): the heard word's own cell, driven by a view too

Held out: pictures never seen, pages made of them.
    SEEDS=0,1,2 ARMS=whole,whole+match,quarters,quarters+match,gaze+match N=8000 python experiments/look2_probe.py
"""
import collections, os, pickle, random, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.words import Words  # noqa: E402
from organic.curriculum import Curriculum, tidy  # noqa: E402
from organic.scenes import page, Scenes  # noqa: E402
from organic.gaze import fixations  # noqa: E402

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
ARMS = os.environ.get("ARMS", "whole,whole+match,quarters,quarters+match,gaze,gaze+match").split(",")
N = int(os.environ.get("N", 8000))
TILE = int(os.environ.get("TILE", 26))  # the side of a picture on a page (30: the pages of the first probe)
CELL = int(os.environ.get("CELL", 32))  # the side of a quarter of the page (64: a page twice the view; the eye sees it whole only coarsely)
APART = float(os.environ.get("APART", 0))  # the gaze only goes to what stands apart (organic/gaze.py)
cur = Curriculum(seed=0, world=2)
ear = pickle.load(open(os.environ.get("EAR", "out/tmp/ear_first.pkl"), "rb"))
cortex = pickle.load(open(os.environ.get("CORTEX", "data/visual_cortex.pkl"), "rb"))
hear = lambda q: ear.units(q, learn=False)  # noqa: E731
CODES, LOOKED = {}, collections.Counter()


DIM = sum(l.kinds for l in cortex.layers) * 16
FULL = os.environ.get("FULL") == "1"  # every correction of the word cells at full size (docs/VISION.md, 2026-11-17)


def code(img):
    if getattr(cortex, "relay", None) is not None:  # the second cortex: its summary through its relay
        return cortex.whole(img, 4).astype(np.float32)
    v = np.concatenate(cortex.code(img, grid=4))
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def coarse(img):
    """The whole view as the eye has it: a scene larger than the view is seen whole only coarsely."""
    f = img.shape[0] // 64
    return img if f == 1 else img.reshape(64, f, 64, f, 3).mean(axis=(1, 3))


def zoom(img, r, c, side=None):
    """What the eye fixates fills the view: a window of half the scene, in the scene's full detail."""
    side = side or img.shape[0] // 2
    w = img[r : r + side, c : c + side]
    return w if side == 64 else np.asarray(Image.fromarray((np.clip(w, 0, 1) * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR), np.float32) / 255


def views(key, make):
    """(whole view, the four quarters, what the gaze fixated) of a picture or page, coded once."""
    if key not in CODES:
        img = make()
        fix = fixations(img, apart=APART)
        h = img.shape[0] // 2
        LOOKED[(key[0], len(fix))] += 1
        half = lambda v: v.astype(np.float16)  # noqa: E731  (the codes of one seed's pictures: 0.8 GB at half precision)
        CODES[key] = (half(code(coarse(img))), [half(code(zoom(img, r, c))) for r in (0, h) for c in (0, h)], [half(code(zoom(img, r, c))) for r, c in fix])
    whole, quarters, fix = CODES[key]
    return whole.astype(np.float32), [q.astype(np.float32) for q in quarters], [f.astype(np.float32) for f in fix]


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
    """A page with one named thing in a random quarter and `others` other things: (key, maker, name, the names on it)."""
    w = rng.choice(names)
    on = [w] + [rng.choice([x for x in names if x != w]) for _ in range(others)]
    tiles = [rng.choice(pool[x]) for x in on] + [None] * (4 - len(on))
    rng.shuffle(tiles)
    shade = rng.uniform(0.75, 0.95)
    return ("page", tuple(tiles), round(shade, 3)), (lambda: page([cur.image(t) if t is not None else None for t in tiles], shade, TILE, CELL)), w, on


def a_scene(scenes, kind, n):
    tiles, shade, q, a = scenes.make(kind, n)
    return ("page", tuple(tiles), round(shade, 3)), (lambda: page([cur.image(t) if t is not None else None for t in tiles], shade, TILE, CELL)), q, a


def run(arm, seed):
    where, match = arm.split("+")[0], "+match" in arm
    rng, wr = random.Random(seed), random.Random(seed + 1)
    words = Words(DIM, ear, match=match)
    words.match_bursts = "bursts" in arm
    words.plastic_by_error = not FULL
    singles = [(r, w) for w in names for r in lived[w]]
    lived_scenes, held_scenes = Scenes(lived, cur.image, seed), Scenes(held, cur.image, 99)
    t0 = time.time()

    def see(key, make):
        whole, quarters, fix = views(key, make)
        words.look(None if where == "whole" and not match else [] if where == "whole" else quarters if where == "quarters" else fix)
        return whole

    for n in range(N):
        kind = rng.choice(["one", "one", "one", "page", "there", "count"])
        if kind == "one":  # a single picture, named when asked
            r, w = rng.choice(singles)
            words.learn(hear(wr.choice(cur.frames)), see(("one", r), lambda r=r: cur.image(r)), hear(w))
        elif kind == "page":  # a thing on a page (a third of the time another beside it), named when asked
            key, make, w, _ = a_page(lived, rng, rng.choice([0, 0, 1]))
            words.learn(hear(wr.choice(cur.frames)), see(key, make), hear(w))
        else:
            key, make, q, a = a_scene(lived_scenes, kind, rng.randrange(10 ** 9))
            words.learn(hear(q), see(key, make), hear(a))
    te = random.Random(99)
    row = {}
    ok = [tidy(words.recall(hear("what is this?"), see(("one", r), lambda r=r: cur.image(r)))[0] or "") == w for w in names for r in held[w]][:300]
    row["single"] = float(np.mean(ok))
    for label, others in (("on a page", 0), ("with another", 1)):
        ok, either = [], []
        for _ in range(300):
            key, make, w, on = a_page(held, te, others)
            said = tidy(words.recall(hear("what is this?"), see(key, make))[0] or "")
            ok.append(said == w)
            either.append(said in on)
        row[label] = float(np.mean(ok))
        if others:
            row["either of the two"] = float(np.mean(either))
    for kind in ("there", "count"):
        got, ans = [], []
        for n in range(300):
            key, make, q, a = a_scene(held_scenes, kind, n)
            got.append(tidy(words.recall(hear(q), see(key, make))[0] or "") == a)
            ans.append(a)
        row[kind] = float(np.mean(got))
        row[kind + " base"] = max(collections.Counter(ans).values()) / len(ans)
    print(f"seed {seed} {arm:15} ({time.time() - t0:.0f}s, {len(words.units)} cells): " + " | ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)


for seed in SEEDS:
    CODES.clear()  # each seed sees its own pictures
    kept = f"out/tmp/look2_{os.path.basename(os.environ.get('CORTEX', 'data/visual_cortex.pkl'))}_{CELL}_{TILE}_{APART}_{N}_s{seed}.pkl"
    if os.path.exists(kept):  # the codes of this seed's pictures, kept so that further arms take seconds
        CODES.update(pickle.load(open(kept, "rb")))
    had = len(CODES)
    for arm in ARMS:
        run(arm, seed)
    if len(CODES) > had:
        pickle.dump(CODES, open(kept, "wb"))
print("how many places the gaze fixated:", {k: v for k, v in sorted(LOOKED.items())}, flush=True)
