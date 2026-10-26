"""Can what the eye and visual cortex send on tell toddler objects apart?

Nearest-neighbour probe on the W1 photo items (CIFAR-100 and COCO 'biggest thing'
restricted to toddler nouns): features are the eye's coarse whole-view code (as at
the first tick) and the visual cortex's layer 2/3 codes. Clusters in the
orchestrator are prototype memories, so this is roughly their ceiling.
"""
import collections, os, pickle, sys, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.curriculum import Curriculum
from organic.loop2 import NoisyEye
from organic.peye import Parts

parts = Parts(); parts.W = np.load("data/parts.npy"); parts.frozen = True
eye = NoisyEye(parts, size=64, noise=0.0)
cortex = pickle.load(open("data/visual_cortex.pkl", "rb"))
cur = Curriculum(seed=0)
for src in ("cifar:name", "coco:biggest"):
    tr, te = cur.train["W1"][src], cur.test["W1"][src]
    rng = random.Random(0)
    tr, te = rng.sample(tr, min(3000, len(tr))), rng.sample(te, min(400, len(te)))
    def feats(items):
        E, C, y = [], [], []
        for ref, q, ans, st, s in items:
            img = cur.image(ref)
            E.append(eye.view(img, 12, 0)); c = cortex.code(img, grid=1)[1:]; C.append(np.concatenate(c)); y.append(ans[0])
        E, C = np.array(E), np.array(C)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8), C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8), y
    Etr, Ctr, ytr = feats(tr); Ete, Cte, yte = feats(te)
    maj = collections.Counter(ytr).most_common(1)[0]
    for name, A, B in (("eye coarse", Etr, Ete), ("cortex L2+L3", Ctr, Cte), ("both", np.hstack([Etr, Ctr]), np.hstack([Ete, Cte]))):
        nn = (B @ A.T).argmax(1)
        acc = np.mean([ytr[j] == yte[i] for i, j in enumerate(nn)])
        print(f"{src:13} {name:13} 1-NN {acc:.2f} | classes {len(set(ytr))}, majority '{maj[0]}' {maj[1]/len(ytr):.2f}", flush=True)
