"""An organism: one shared cortex fed by several sensory organs."""

import numpy as np

from .cortex import Cortex
from .organs import LanguageOrgan, VisionOrgan


class Organism:
    def __init__(self, seed=0, **cortex_kwargs):
        self.cortex = Cortex(dim=1024, seed=seed, **cortex_kwargs)
        self.vision = VisionOrgan(self, start=0, width=512)
        self.language = LanguageOrgan(self, start=512, width=512, seed=seed + 1)
        self.organs = [self.vision, self.language]

    def territories(self):
        """Which organ's region each cluster's identity lives in (emergent specialization)."""
        cx = self.cortex
        counts = {o.prefix: 0 for o in self.organs}
        counts["mixed"] = 0
        Z = cx.Z[: cx.C]
        for z in Z:
            energy = {o.prefix: float((z[o.start : o.start + o.width] ** 2).sum()) for o in self.organs}
            best = max(energy, key=energy.get)
            counts[best if energy[best] > 0.8 * sum(energy.values()) else "mixed"] += 1
        return {k: v * cx.n for k, v in counts.items()}

    def report(self):
        cx = self.cortex
        t = self.territories()
        terr = ", ".join(f"{k}={v}" for k, v in t.items())
        links = sum(len(r) for r in cx.assoc.values())
        return (
            f"neurons={cx.neuron_count()} clusters={cx.C} births={cx.births} deaths={cx.deaths} "
            f"associations={links} outputs={len(cx.outputs)} | territories: {terr}"
        )
