"""Let the visual cortex develop on unlabeled training photos, then probe it.

    python experiments/develop_cortex.py [samples_per_layer]

Saves data/visual_cortex.pkl. The probe (nearest neighbours, labels only used
to read the result) compares the eye's own output with each cortex layer on
held-out photos: person present, biggest thing, where the person is, reading,
and CIFAR-100 identity.
"""

import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.visual_cortex import VisualCortex  # noqa: E402
from organic.world import World  # noqa: E402

per_layer = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
KINDS = tuple(int(k) for k in (sys.argv[2] if len(sys.argv) > 2 else "64,128,256").split(","))
NAME = sys.argv[3] if len(sys.argv) > 3 else "visual_cortex"
rng = np.random.default_rng(0)
world = World(seed=0)
train_refs = sorted({it[0] for it in world.train}, key=str)
train_refs = [r for r in train_refs if not (isinstance(r, tuple) and r[0] in ("dark", "read"))]
print(len(train_refs), "training photos for development", flush=True)
sample = [train_refs[i] for i in rng.choice(len(train_refs), min(8000, len(train_refs)), replace=False)]
images = [world.image(r) for r in sample]
t0 = time.time()
vc = VisualCortex(kinds=KINDS).develop(images, rng, per_layer=per_layer, log=lambda m: print(f"{m} ({time.time() - t0:.0f}s)", flush=True))
pickle.dump(vc, open(os.path.join(os.path.dirname(__file__), "..", "data", f"{NAME}.pkl"), "wb"))


def probe_items(kind, q=None, n_train=3000, n_test=600):
    tr = [it for it in world.train if it[3] == kind and (q is None or it[1] == q)]
    te = [it for it in world.test if it[3] == kind and (q is None or it[1] == q)]
    return tr[:n_train], te[:n_test]


def knn(Ftr, ytr, Fte, yte, k=5):
    Ftr = Ftr / (np.linalg.norm(Ftr, axis=1, keepdims=True) + 1e-8)
    Fte = Fte / (np.linalg.norm(Fte, axis=1, keepdims=True) + 1e-8)
    nn = np.argsort(-(Fte @ Ftr.T), axis=1)[:, :k]
    ytr = np.array(ytr)
    return float(np.mean([max(set(ytr[r]), key=list(ytr[r]).count) == y for r, y in zip(nn, yte)]))


probes = [("is there a person?", probe_items("presence", "is there a person?")), ("biggest thing", probe_items("identity", "what is the biggest thing here?")),
          ("where is the person", probe_items("where")), ("reading", probe_items("text")), ("cifar-100 identity", probe_items("identity", "what is this?"))]
for name, (tr, te) in probes:
    if not tr or not te:
        continue
    ytr = [it[2] for it in tr]
    yte = [it[2] for it in te]
    maj = max(set(ytr), key=ytr.count)
    base = np.mean([y == maj for y in yte])
    Ctr = [vc.code(world.image(it[0])) for it in tr]
    Cte = [vc.code(world.image(it[0])) for it in te]
    res = [knn(np.array([c[i] for c in Ctr]), ytr, np.array([c[i] for c in Cte]), yte) for i in range(3)]
    allc = knn(np.array([np.concatenate(c) for c in Ctr]), ytr, np.array([np.concatenate(c) for c in Cte]), yte)
    print(f"{name:<20} n={len(tr)}/{len(te)} baseline {base:.2f} | layer1 {res[0]:.2f} layer2 {res[1]:.2f} layer3 {res[2]:.2f} all {allc:.2f}", flush=True)
