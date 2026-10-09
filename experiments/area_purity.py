"""Do unsupervised sensory areas discover the categories people name?

For each vigilance level, grow an area on 1000 objects, then check how pure
its assemblies are on 500 new ones (fraction belonging to the assembly's
majority category). No labels are used for growing.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.area import SensoryArea  # noqa: E402
from organic.eye import Eye  # noqa: E402
from organic.scenes import random_object, render  # noqa: E402

rng = np.random.default_rng(0)
eye = Eye()
data = []
for _ in range(1500):
    o = random_object(rng, False)
    data.append((o, eye.look(render([o], rng))[0]))


def purity(feat, key, vig):
    a = SensoryArea("x", len(feat(data[0][1])), vigilance=vig)
    for _, f in data[:1000]:
        a.perceive(feat(f))
    tab = {}
    for o, f in data[1000:]:
        idx, act = a.perceive(feat(f), learn=False)
        tab.setdefault(idx[np.argmax(act)], []).append(key(o))
    pure = sum(max(v.count(x) for x in set(v)) for v in tab.values()) / 500
    return f"{a.C:>3} assemblies, purity {pure:.2f}"


streams = {
    "shape": (lambda f: f.form, lambda o: o["shape"]),
    "color": (lambda f: f.color, lambda o: o["color"]),
    "where": (lambda f: f.where, lambda o: (o["cell"], o["size"])),
}
for vig in [0.8, 0.86, 0.9, 0.93, 0.95, 0.97]:
    print(f"vigilance {vig}: " + " | ".join(f"{k}: {purity(*v, vig)}" for k, v in streams.items()), flush=True)
