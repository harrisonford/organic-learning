"""The second cortex as an organ (organic/cortex2.py): developed, then read out layer by layer (docs/VISION.md).

The same linear reference as before (a ridge on the code): held-out COCO things (35 names), and the first-words
pictures of CIFAR + Commons (22 names). The cortex is saved to out/tmp/cortex2_<TAG>.pkl.
    KINDS=256,256 GROUPS=16 FAN_IN=32 POOL=avg PER_LAYER=16000 TAG=a python experiments/cortex3_probe.py
"""
import json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.world import DATA  # noqa: E402
from organic.curriculum import Curriculum  # noqa: E402
from organic.cortex2 import VisualCortex2  # noqa: E402

KINDS = tuple(int(k) for k in os.environ.get("KINDS", "256,256").split(","))
GROUPS, FAN_IN = int(os.environ.get("GROUPS", 16)), int(os.environ.get("FAN_IN", 32))
PER_LAYER, FIT, SEED, GRID = int(os.environ.get("PER_LAYER", 16000)), int(os.environ.get("FIT", 6000)), int(os.environ.get("SEED", 0)), int(os.environ.get("GRID", 4))
TAG = os.environ.get("TAG", "a")
pics = np.load(os.path.join(DATA, "cococrops", "pictures_64.npy"), mmap_mode="r")
meta = [json.loads(l) for l in open(os.path.join(DATA, "cococrops", "pictures.jsonl"))]
names = sorted({m["word"] for m in meta})
label = np.array([names.index(m["word"]) for m in meta])
held = np.array([m["image"] % 20 == 0 for m in meta])
rng = np.random.default_rng(SEED)
train_rows, test_rows = rng.permutation(np.nonzero(~held)[0]), rng.permutation(np.nonzero(held)[0])[:1500]
img = lambda r: np.asarray(pics[r], np.float32) / 255  # noqa: E731
log = lambda *a: print(f"[{TAG} {time.strftime('%H:%M:%S')}]", *a, flush=True)  # noqa: E731
cur = Curriculum(seed=0)
tr2 = [it for src in ("cifar:name", "commons:name") for it in cur.train["W1"][src]][::3][:3000]
te2 = [it for src in ("cifar:name", "commons:name") for it in cur.test["W1"][src]]
tr2, te2 = [it for it in tr2 if cur.image(it[0]).max() > 0], [it for it in te2 if cur.image(it[0]).max() > 0]
old2 = sorted({it[2][0] for it in tr2})


def ridge(A, Y, B, lam):
    if A.shape[1] > A.shape[0]:
        K = A @ A.T
        K[np.diag_indices_from(K)] += lam
        return (B @ A.T) @ np.linalg.solve(K, Y)
    G = A.T @ A
    G[np.diag_indices_from(G)] += lam
    return B @ np.linalg.solve(G, A.T @ Y)


def read(A, Y, B, y):
    mean = A.mean(0)
    A, B = A - mean, B - mean
    return max(float((ridge(A, Y, B, lam).argmax(1) == y).mean()) for lam in (0.01, 0.03, 0.1, 0.3, 1.0))


path = os.environ.get("CORTEX")
if path:
    cortex = pickle.load(open(path, "rb"))
else:
    cortex = VisualCortex2(seed=SEED, kinds=KINDS, groups=GROUPS, fan_in=FAN_IN, pool_kind=os.environ.get("POOL", "avg"))
    # developed on pictures of both kinds the brain will see (COCO things and the first-words pictures), none of them test pictures
    pool = [img(r) for r in train_rows[-3000:]] + [cur.image(it[0]) for it in tr2[:1500]]
    cortex.develop(pool, np.random.default_rng(SEED + 5), per_layer=PER_LAYER, log=log)
    pickle.dump(cortex, open(f"out/tmp/cortex2_{TAG}.pkl", "wb"))
t0 = time.time()
codes = lambda image: cortex.code(image, grid=GRID)  # noqa: E731
fit = train_rows[:FIT]
A = [codes(img(r)) for r in fit]
per = (time.time() - t0) / len(fit)
B = [codes(img(r)) for r in test_rows]
A2 = [codes(cur.image(it[0])) for it in tr2]; B2 = [codes(cur.image(it[0])) for it in te2]
Y = np.eye(len(names), dtype=np.float32)[label[fit]]
Y2 = np.eye(len(old2), dtype=np.float32)[[old2.index(it[2][0]) for it in tr2]]
y2 = np.array([old2.index(it[2][0]) if it[2][0] in old2 else -1 for it in te2])
unit = lambda v: v / (np.linalg.norm(v) + 1e-8)  # noqa: E731
pick = lambda rows, ls: np.stack([unit(np.concatenate([r[l] for l in ls])) for r in rows])  # noqa: E731
n = len(cortex.layers)
for ls in [[l] for l in range(n)] + ([list(range(n))] if n > 1 else []):
    coco = read(pick(A, ls), Y, pick(B, ls), label[test_rows])
    first = read(pick(A2, ls), Y2, pick(B2, ls), y2)
    log(f"layers {'+'.join(str(l + 1) for l in ls):6} code of {pick(A[:1], ls).shape[1]:>6} | held-out COCO things {coco:.3f} | CIFAR+Commons {first:.3f}")
log(f"{per * 1000:.0f} ms to code a picture; kinds {KINDS}, groups {GROUPS}, fan-in {FAN_IN}, grid {GRID}")
