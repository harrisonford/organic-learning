"""Can the hub do conjunctions (question kind AND picture)? Additive vs multiplicative organ evidence (offline).

In a life, the hub learned first words (same answer across wordings 0.77-0.97) and then
lost picture naming at two words: W2 asks about the same pictures with other answers
("yes", "two", "brown"), and additive votes cannot say "a naming question AND this picture
-> its name; a colour question AND this picture -> its colour". Here the hub is trained on
W1 + W2 moments (as a life at W2 lives them: both stages) and tested on
  - W1 pictures, each asked three ways (naming): right, same answer x3, right x3
  - W2 test items (colours, yes/no, counting about pictures)
with two ways of combining the organs' evidence:
  additive        votes = sound + vision (as built)
  multiplicative  each organ's evidence scales the others' (coincidence detection): log-evidence adds,
                  each organ's distribution taken with its baseline (one spontaneous co-occurrence spread
                  over the lexicon), so an organ that has never seen a word with this input can veto it
    python experiments/hub_conjunction_probe.py
"""
import json, os, pickle, sys, zlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum, grade, tidy  # noqa: E402
from organic.hub import Hub  # noqa: E402
from organic.world import WORLD_DIR  # noqa: E402

cur = Curriculum(seed=0)
ear = pickle.load(open("out/kid_s3.ear.pkl", "rb"))
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
FRAMES = json.load(open(os.path.join(WORLD_DIR, "naming_frames.json")))["frames"]
rng = np.random.default_rng(0)


def vis(ref):
    img = cur.image(ref)
    if img.max() == 0:
        return None
    v = np.concatenate(cortex.code(img, grid=4))
    return v / (np.linalg.norm(v) + 1e-8)


# a life at W2: W2 moments, with W1 reviewed (as the curriculum's 70/30 does), in a mixed order
w1 = [it for items in cur.train["W1"].values() for it in items]
w2 = [it for items in cur.train["W2"].values() for it in items]
order = [w1[i] for i in rng.choice(len(w1), 9000, replace=False)] + [w2[i] for i in rng.choice(len(w2), 9000, replace=False)]
order = [order[i] for i in rng.permutation(len(order))]
dim = next(vis(it[0]) for it in order if vis(it[0]) is not None).size
hub = Hub(dim, seed=0)
cache = {}
for n, (ref, q, ans, st, src) in enumerate(order):
    if ref not in cache:
        cache[ref] = vis(ref)
    hub.learn(ear.units(q, learn=False), cache[ref], ans[0])
print(f"hub: {len(order)} moments (W1 + W2), {hub.area.C} visual cells, {len(hub.lexicon)} words", flush=True)


def evidence(units, v):
    """Each organ's distribution over the lexicon (sound: its units' links averaged; vision: the area's output)."""
    K = len(hub.lexicon)
    snd = np.zeros(K)
    known = [u for u in set(units) if u in hub.sound]
    for u in known:
        for k, c in hub.sound[u].items():
            snd[k] += c / hub.heard[u] / len(known)
    see = np.zeros(K)
    if v is not None:
        r = hub.area.forward(v, np.array(hub.out))
        if r is not None and r["probs"] is not None:
            see = r["probs"]
    return snd if known else None, see if v is not None else None


def answer(q, v, how):
    snd, see = evidence(ear.units(q, learn=False), v)
    K = len(hub.lexicon)
    organs = [e for e in (snd, see) if e is not None]
    if how == "additive":
        votes = sum(organs)
    else:  # each organ's evidence with its baseline: one spontaneous co-occurrence spread over the lexicon
        votes = sum(np.log(e + 1.0 / K) for e in organs)
    return hub.lexicon[int(np.argmax(votes))]


test_pics = [it for src in ("cifar:name", "commons:name") for it in cur.test["W1"][src][:150]]
test_w2 = [it for items in cur.test["W2"].values() for it in items[:60]]
for how in ("additive", "multiplicative"):
    acc, same, every = [], 0, 0
    for it in test_pics:
        v = cache.get(it[0]) if it[0] in cache else vis(it[0])
        asks = [FRAMES[(zlib.crc32(repr(it[0]).encode()) + 17 * j + 5) % len(FRAMES)] for j in range(3)]
        said = [answer(q, v, how) for q in asks]
        ok = [tidy(s) in [tidy(a) for a in it[2]] for s in said]
        acc += ok
        same += len(set(said)) == 1
        every += all(ok)
    w2s = np.mean([grade("W2", answer(it[1], vis(it[0]), how), it[2]) for it in test_w2])
    print(f"{how:15} W1 pictures: right {np.mean(acc):.3f} | same answer x3 {same / len(test_pics):.3f} | right x3 {every / len(test_pics):.3f}"
          f" || W2 (colour, yes/no, counting): {w2s:.3f}", flush=True)
