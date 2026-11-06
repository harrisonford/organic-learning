"""Does a visual cortex that keeps learning during life settle on its own, and stay useful? (offline)

docs/PLASTICITY.md step 2. Three cortices see the same stream of pictures a life sees
(W1-K moments: CIFAR, Commons, COCO):
  frozen    the developed cortex, read only (as in every life so far)
  plastic   the developed cortex, learning while living (VisualCortex.live: each kind learns
            in proportion to its own recent mismatch, no clock)
  scratch   a random cortex, learning only while living (every layer at once from the start)
Every BLOCK pictures: drift of each cortex's code on a fixed probe set since the last
measure (settling = drift falling by itself), mean mismatch per layer, and how well the
code tells first-words objects apart (1-NN naming on W1 pictures, CIFAR and Commons,
with the 4x4 all-layer code kid_s3 uses).
    python experiments/plastic_vision_probe.py [pictures, default 24000]     ARMS=scratch BLOCK=10000 (one arm, longer)
"""
import copy, os, pickle, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum  # noqa: E402
from organic.visual_cortex import VisualCortex  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
BLOCK = int(os.environ.get("BLOCK", 4000))
GRID = 4
cur = Curriculum(seed=0)
rng = random.Random(0)
dev = pickle.load(open("data/visual_cortex.pkl", "rb"))
arms = {"frozen": dev, "plastic": copy.deepcopy(dev), "scratch": VisualCortex(seed=3)}
arms = {k: v for k, v in arms.items() if k in os.environ.get("ARMS", "frozen,plastic,scratch").split(",")}
nrng = {k: np.random.default_rng(i) for i, k in enumerate(arms)}

probe = [m["image"] for m in (cur.moment(s) for s in ("W1", "W2", "P", "K") for _ in range(60)) if m["image"].max() > 0][:100]  # real pictures only (text items are eyes closed)
naming_tr = rng.sample(cur.train["W1"]["cifar:name"], 1000) + rng.sample(cur.train["W1"]["commons:name"], 1000)
naming_te = rng.sample(cur.test["W1"]["cifar:name"], 150) + rng.sample(cur.test["W1"]["commons:name"], 150)


def code(cx, img):
    v = np.concatenate(cx.code(img, grid=GRID))
    return v / (np.linalg.norm(v) + 1e-8)


def naming(cx):
    A = np.array([code(cx, cur.image(it[0])) for it in naming_tr])
    B = np.array([code(cx, cur.image(it[0])) for it in naming_te])
    ytr, yte = [it[2][0] for it in naming_tr], [it[2][0] for it in naming_te]
    nn = (B @ A.T).argmax(1)
    return float(np.mean([ytr[j] == yte[i] for i, j in enumerate(nn)]))


prev = {k: np.array([code(cx, im) for im in probe]) for k, cx in arms.items()}
print("start   " + " | ".join(f"{k}: naming {naming(cx):.3f}" for k, cx in arms.items()), flush=True)
t0, seen = time.time(), 0
while seen < N:
    for _ in range(BLOCK):
        m = cur.moment(rng.choice(["W1", "W2", "P", "K"]))
        if m["image"].max() == 0:  # eyes closed (text items)
            continue
        for k in [a for a in ("plastic", "scratch") if a in arms]:
            arms[k].live(m["image"], nrng[k])
        seen += 1
    row = []
    for k, cx in arms.items():
        now = np.array([code(cx, im) for im in probe])
        d = 1 - float((now * prev[k]).sum(1).mean())
        prev[k] = now
        mm = cx.plasticity()
        mms = "/".join("-" if x is None else f"{x:.3f}" for x in mm)
        row.append(f"{k}: drift {d:.4f} mismatch {mms} naming {naming(cx):.3f}")
    print(f"{seen:>6} pictures ({time.time() - t0:.0f}s) | " + " | ".join(row), flush=True)
