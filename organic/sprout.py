"""Sprouted areas: new cortex that grows when the brain stays surprised.

When speech keeps failing in a way that more clusters inside the speech area
don't fix, the brain grows a whole new area, with no assigned function.

Its neurons receive a few random axons each from everything that already
exists (the question echo, the match/mismatch neurons, the output of every
sensory area) and fire sparsely (k-winners-take-all). This is the random
sparse expansion of cerebellar granule cells and insect mushroom-body Kenyon
cells, which makes combinations of inputs separable. On top of that
expansion the area organizes itself like any sensory area (novelty-driven
assemblies), and its assemblies project to the speech area through a new,
unmyelinated pathway. Whether it becomes useful, and for what, is up to
experience.
"""

import numpy as np

from .area import SensoryArea


class SproutedArea:
    def __init__(self, name, blocks, width=512, fan_in=8, active=0.05, vigilance=0.75, seed=0, stable=False):
        """blocks: list of (name, size) of the source populations, in order."""
        self.name = name
        self.blocks = blocks
        rng = np.random.default_rng(seed)
        offsets = np.cumsum([0] + [n for _, n in blocks])
        # each neuron picks fan_in axons, each from a random source population
        which = rng.integers(len(blocks), size=(width, fan_in))
        self.idx = offsets[which] + (rng.random((width, fan_in)) * np.array([n for _, n in blocks])[which]).astype(int)
        self.w = rng.choice([-1.0, 1.0], size=(width, fan_in)).astype(np.float32)
        self.k = max(1, int(active * width))
        # stable: the sparse expansion itself is the output (no assemblies keep
        # being born on top of it, so what downstream sees does not drift)
        self.stable = stable
        self.area = None if stable else SensoryArea(name, width, vigilance=vigilance, seed=seed)
        self.out_dim = width if stable else self.area.axon_dim

    def size(self):
        return len(self.w) if self.stable else self.area.C

    def respond(self, sources, learn):
        """sources: list of vectors, one per block (each normalized here)."""
        x = np.concatenate([v / (np.linalg.norm(v) + 1e-8) for v in sources]).astype(np.float32)
        h = (self.w * x[self.idx]).sum(axis=1)
        thresh = np.partition(h, -self.k)[-self.k]
        code = np.where(h >= thresh, np.maximum(h, 0), 0).astype(np.float32)
        if self.stable:
            return code
        idx, act = self.area.perceive(code, learn)
        return self.area.output(idx, act)
