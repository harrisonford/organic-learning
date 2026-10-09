"""Has a division of labor emerged between the raw areas?

For each area, how strongly its best assemblies predict each family of words
(the lift a word gets from the area's most specific assembly, averaged over
the family). Families are only used to read the result; the brain never sees them.

    python experiments/raw_specialization.py out/raw_deadline_s0.pkl
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.scenes import COLORS, SHAPES  # noqa: E402

FAMILIES = {
    "color": list(COLORS),
    "shape": SHAPES,
    "place": ["left", "right", "top", "bottom", "middle"],
    "size": ["big", "small"],
    "yes/no": ["yes", "no"],
}

brain = pickle.load(open(sys.argv[1], "rb"))
V = brain.V
k = 0.02
ov = brain.overlap()
print(f"{'area':<8}" + "".join(f"{f:>9}" for f in FAMILIES) + "   assemblies")
for name, area in brain.areas.items():
    area._ensure(V=V)
    seasoned = np.nonzero(area.named[: area.C] >= 10)[0]  # assemblies with real experience
    if len(seasoned) == 0:
        print(f"{name:<8} (no assembly with enough experience yet)")
        continue
    lift = ((area.Wname[seasoned, :V] + k) / (brain.baseline[:V] + k)).max(axis=0)
    row = []
    for fam, words in FAMILIES.items():
        vals = []
        for w in words:
            # any word-form that sounds like the word (e.g. "red", "red?", "it is red")
            idx = [i for i in range(V) if w.encode() in brain.name(i)]
            if idx:
                vals.append(max(lift[i] for i in idx))
        row.append(np.mean(vals) if vals else float("nan"))
    print(f"{name:<8}" + "".join(f"{v:9.2f}" for v in row) + f"   {area.C} ({len(seasoned)} seasoned)")
