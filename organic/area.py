"""Sensory areas: cortex that organizes itself from experience alone.

A sensory area has no teacher. It receives one stream from an organ (form,
color, place, number...) and grows assemblies of neurons by novelty:

- Every assembly has a preferred stimulus (its identity / receptive field).
- A stimulus that no assembly responds to strongly enough (below the area's
  *vigilance*) triggers neurogenesis: a new assembly is born tuned to it.
- Otherwise the best-matching assemblies respond (soft winner-take-all through
  lateral inhibition) and their tuning drifts toward the stimulus. The drift
  rate falls with experience (1/n), so an assembly's identity becomes the
  running average of everything it has stood for, and it gets harder to change
  as it matures.

Cross-modal binding: while the area is active, the organism hears words. Each
assembly keeps a Hebbian trace of which words co-occur with its firing
(P(word | assembly) as a running average). Word neurons in turn adapt to their
own average activity (homeostatic intrinsic plasticity), so what reaches the
lexicon is the *lift*: how much more a word is expected than usual, given what
this area sees. That is priming: seeing red makes "red" ready to be said.
"""

import numpy as np


class SensoryArea:
    def __init__(self, name, dim, vigilance=0.9, temp=0.03, neurons_per_assembly=6, min_rate=0.01, axon_dim=64, seed=0):
        self.name = name
        self.dim = dim
        self.vigilance = vigilance
        self.temp = temp
        self.n = neurons_per_assembly
        self.min_rate = min_rate
        self.C = 0
        self.cap = 32
        self.Z = np.zeros((self.cap, dim), np.float32)
        self.usage = np.zeros(self.cap, np.int64)
        self.vcap = 64
        self.Wname = np.zeros((self.cap, self.vcap), np.float32)  # word co-occurrence traces
        self.named = np.zeros(self.cap, np.float32)  # how much naming experience
        self.births = 0
        # each assembly's axons reach downstream neurons in its own fixed random
        # pattern, so the area's output can be read as one vector
        self.axon_dim = axon_dim
        self.axons = np.zeros((self.cap, axon_dim), np.float32)
        self.rng = np.random.default_rng(seed + sum(map(ord, name)))

    # ---------------------------------------------------------------- growth

    def _ensure(self, C=None, V=None):
        if C is not None and C >= self.cap:
            self.cap *= 2
            for k in ("Z", "usage", "Wname", "named", "axons"):
                a = getattr(self, k)
                b = np.zeros((self.cap,) + a.shape[1:], a.dtype)
                b[: a.shape[0]] = a
                setattr(self, k, b)
        if V is not None and V > self.vcap:
            while self.vcap < V:
                self.vcap *= 2
            b = np.zeros((self.cap, self.vcap), np.float32)
            b[:, : self.Wname.shape[1]] = self.Wname
            self.Wname = b

    def _born(self, x):
        self._ensure(C=self.C)
        c = self.C
        self.Z[c] = x
        self.usage[c] = 1
        self.Wname[c] = 0
        self.named[c] = 0
        a = np.zeros(self.axon_dim, np.float32)
        a[self.rng.choice(self.axon_dim, 8, replace=False)] = self.rng.choice([-1.0, 1.0], 8)
        self.axons[c] = a / np.sqrt(8)
        self.C += 1
        self.births += 1
        return c

    # ------------------------------------------------------------- perceive

    def perceive(self, x, learn=True, bias=None):
        """Respond to a stimulus. Returns (assembly indices, activity).

        bias: optional per-assembly boost from lingering activity (trace rule):
        assemblies that were just active are favoured, so successive inputs
        tend to join them instead of giving birth to new ones.
        """
        n = np.linalg.norm(x)
        if n == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        x = (x / n).astype(np.float32)
        if self.C == 0:
            if not learn:
                return np.zeros(0, np.int64), np.zeros(0, np.float32)
            return np.array([self._born(x)]), np.ones(1, np.float32)
        aff = self.Z[: self.C] @ x
        if bias is not None and len(bias):
            aff = aff.copy()
            aff[: len(bias)] += bias[: self.C]
        best = int(np.argmax(aff))
        if aff[best] < self.vigilance and learn:
            c = self._born(x)
            return np.array([c]), np.ones(1, np.float32)
        k = min(4, self.C)
        idx = np.argpartition(-aff, k - 1)[:k]
        act = np.exp((aff[idx] - aff[best]) / self.temp)
        act /= act.sum()
        keep = act > 0.05
        idx, act = idx[keep], act[keep] / act[keep].sum()
        if learn:
            for c, a in zip(idx, act):
                self.usage[c] += 1
                rate = a / self.usage[c]
                z = self.Z[c] + rate * (x - self.Z[c])
                self.Z[c] = z / np.linalg.norm(z)
        return idx, act

    # -------------------------------------------------------------- binding

    def bind(self, idx, act, heard):
        """Hebbian trace between the active assemblies and the words heard."""
        self._ensure(V=len(heard))
        for c, a in zip(idx, act):
            self.named[c] += a
            rate = max(self.min_rate, a / self.named[c])
            self.Wname[c, : len(heard)] += rate * (heard - self.Wname[c, : len(heard)])

    def expectation(self, idx, act, V):
        """P(word | what this area sees now), from the active assemblies' traces."""
        if len(idx) == 0:
            return None
        self._ensure(V=V)
        return (act[:, None] * self.Wname[idx, :V]).sum(axis=0)

    def output(self, idx, act):
        """What downstream neurons receive from the active assemblies."""
        if len(idx) == 0:
            return np.zeros(self.axon_dim, np.float32)
        return (act[:, None] * self.axons[idx]).sum(axis=0)

    def neuron_count(self):
        return self.C * self.n
