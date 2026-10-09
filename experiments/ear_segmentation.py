"""Watch word-like units emerge from raw bytes as the ear hears more speech."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from organic.ear import Ear  # noqa: E402
from organic.scenes import episode  # noqa: E402

rng = np.random.default_rng(0)
ear = Ear()
probe = ["what color is it?", "no, it is a green triangle", "is it red and big?", "haha yes, it is a funny face"]
for n in [0, 10, 50, 200, 1000, 3000]:
    while ear.utterances < n:
        e = episode(rng)
        ear.units(e["question"])
        ear.units(e["answer"])
    print(f"after {n} utterances:")
    for p in probe:
        print("   " + " | ".join(u.decode() for u in ear.units(p, learn=False)))
