"""All saved photo-world brains on one fixed, held-out test set (200 per kind)."""

import collections
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.world import World  # noqa: E402

world = World(seed=0)
by = collections.defaultdict(list)
for it in world.test:
    by[it[3]].append(it)
rng = np.random.default_rng(7)
test = []
for kind in sorted(by):
    items = by[kind]
    for i in rng.choice(len(items), min(200, len(items)), replace=False):
        test.append(items[i])
counts = collections.Counter(a for *_, a, k in world.train for _ in [0])
maj = {k: collections.Counter(a for _, _, a, kk in world.train if kk == k).most_common(1)[0][0] for k in by}
print("kind      n    baseline", end="")
brains = sys.argv[1:]
for b in brains:
    print(f"  {os.path.basename(b)[:-4]:>12}", end="")
print()
results = {}
for path in brains:
    brain = pickle.load(open(path, "rb"))
    per = collections.defaultdict(list)
    for ref, q, a, kind in test:
        said, tick, *_ = brain.live(world.image(ref), q, learn=False, explore=False)
        per[kind].append((said == a, tick))
    results[path] = per
for kind in sorted(by):
    n = len([t for t in test if t[3] == kind])
    base = np.mean([a == maj[kind] for _, _, a, k in test if k == kind])
    row = f"{kind:<9} {n:>4}   {base:6.2f}  "
    for path in brains:
        v = results[path][kind]
        row += f"  {np.mean([c for c, _ in v]):6.2f} t{np.mean([t for _, t in v]):4.1f}"
    print(row, flush=True)
