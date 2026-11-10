"""Vision with both halves of the rule, on the eye's own views (offline; docs/VISION.md).

Two earlier negatives: competitive Hebbian features name first-words objects no better than random
filters (0.42 both), and a trace rule alone was worse (0.17). Halvagar & Zenke (2023) show why: Hebbian
learning alone gives no invariance and prediction alone collapses; both together, layer-local, learn
invariant objects from sequences of views. Here every layer of a random visual cortex learns at once
(Layer.predictive) from the views the eye makes of each picture a life sees: the whole picture, then
closer around one gaze point, closer still, another gaze point...

Arms (ARMS=...):
  random     random filters, never learning (the baseline to beat: 0.42)
  developed  the competitive-Hebbian cortex of every life so far, frozen
  both       predict + own variance + decorrelation (the two-part rule)
  hebb       own variance + decorrelation only
  predict    prediction only
  both_dev   the two-part rule starting from the developed cortex
Every BLOCK pictures: how well each code tells first-words objects apart (1-NN naming, CIFAR and Commons,
2,000 remembered pictures, 780 held out; the all-layer 4x4 code the brain uses), on whole pictures.
    python experiments/lpl_vision_probe.py [pictures, default 40000]     ARMS=both,hebb BLOCK=5000 RATE=0.005
"""
import copy, os, pickle, random, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum  # noqa: E402
from organic.visual_cortex import VisualCortex  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
BLOCK = int(os.environ.get("BLOCK", 5000))
RATE = float(os.environ.get("RATE", 0.005))
GRID = 4
RULES = {"both": dict(predict=1.0, hebb=1.0, decorrelate=1.0), "hebb": dict(predict=0.0, hebb=1.0, decorrelate=1.0),
         "predict": dict(predict=1.0, hebb=0.0, decorrelate=0.0), "both_dev": dict(predict=1.0, hebb=1.0, decorrelate=1.0)}
cur = Curriculum(seed=0)
rng = random.Random(0)
dev = pickle.load(open("data/visual_cortex.pkl", "rb"))
arms = {"random": VisualCortex(seed=3), "developed": dev, "both": VisualCortex(seed=3), "hebb": VisualCortex(seed=3),
        "predict": VisualCortex(seed=3), "both_dev": copy.deepcopy(dev)}
arms = {k: v for k, v in arms.items() if k in os.environ.get("ARMS", "random,developed,both,hebb,predict").split(",")}
TARGETS = [(x, y) for y in np.linspace(4, 60, 5) for x in np.linspace(4, 60, 5)]  # the eye's 25 gaze points


def crop(img, cx, cy, side):
    """What the eye sees around a gaze point at a zoom: a window kept inside the picture, at the cortex's resolution."""
    x0 = int(min(max(cx - side / 2, 0), 64 - side))
    y0 = int(min(max(cy - side / 2, 0), 64 - side))
    part = (img[y0 : y0 + side, x0 : x0 + side] * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(part).resize((64, 64), Image.BILINEAR), np.float32) / 255


def views(img, r):
    """As the eye looks in a life: the whole picture, then medium and fine around a gaze point, then another point."""
    out = [img]
    for _ in range(2):
        cx, cy = TARGETS[r.randrange(25)]
        out += [crop(img, cx, cy, 40), crop(img, cx, cy, 24)]
    return out


naming_tr = rng.sample(cur.train["W1"]["cifar:name"], 1000) + rng.sample(cur.train["W1"]["commons:name"], 1000)
naming_te = rng.sample(cur.test["W1"]["cifar:name"], 400) + cur.test["W1"]["commons:name"][:380]
ytr, yte = [it[2][0] for it in naming_tr], [it[2][0] for it in naming_te]
TR = [cur.image(it[0]) for it in naming_tr]
TE = [cur.image(it[0]) for it in naming_te]


def code(cx, img):
    v = np.concatenate(cx.code(img, grid=GRID))
    return v / (np.linalg.norm(v) + 1e-8)


def naming(cx):
    A = np.array([code(cx, im) for im in TR])
    B = np.array([code(cx, im) for im in TE])
    nn = (B @ A.T).argmax(1)
    return float(np.mean([ytr[j] == yte[i] for i, j in enumerate(nn)]))


def health(cx):
    """Per layer: the share of kinds that are alive (vary at all) and the mean |correlation| between kinds."""
    out = []
    for layer in cx.layers:
        if not hasattr(layer, "R"):
            return ""
        off = np.abs(layer.R - np.diag(np.diag(layer.R)))
        cv = np.sqrt(layer.var) / (layer.mu + 1e-8)
        out.append(f"{np.mean(cv > 0.05):.2f}/{off.mean():.2f}")
    return " alive/corr " + " ".join(out)


print(f"naming: {len(TR)} remembered, {len(TE)} held out (sampling error about {np.sqrt(0.42 * 0.58 / len(TE)):.3f})", flush=True)
print("start   " + " | ".join(f"{k}: naming {naming(cx):.3f}" for k, cx in arms.items()), flush=True)
t0, seen = time.time(), 0
vr = {k: random.Random(1) for k in arms}
while seen < N:
    for _ in range(BLOCK):
        m = cur.moment(rng.choice(["W1", "W2", "P", "K"]))
        if m["image"].max() == 0:  # eyes closed (text items)
            continue
        for k in arms:
            if k in RULES:
                arms[k].watch(views(m["image"], vr[k]), rate=RATE, **RULES[k])
        seen += 1
    print(f"{seen:>6} pictures ({time.time() - t0:.0f}s) | " + " | ".join(f"{k}: naming {naming(cx):.3f}{health(cx)}" for k, cx in arms.items() if k in RULES), flush=True)
    for k in arms:
        if k in RULES:
            pickle.dump(arms[k], open(f"out/tmp/lpl_{k}.pkl", "wb"))
