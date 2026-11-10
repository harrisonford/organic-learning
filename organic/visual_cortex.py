"""A visual cortex that develops by itself on natural photos.

Between the eye and the orchestrator, a stack of retinotopic layers, each made
of a few kinds of neurons that tile the visual field (like V1 -> V2 -> V4/IT):

  layer 1  5x5 windows of the photo (RGB)              -> 64 kinds, pooled 2x2
  layer 2  3x3 windows of layer-1 maps                 -> 128 kinds, pooled 2x2
  layer 3  3x3 windows of layer-2 maps                 -> 256 kinds, pooled to 2x2

Every layer learns with the same rule as the parts bank and V1: competitive
Hebbian learning on contrast-normalized windows (the window's mean is removed
and it is scaled to unit length, as retina/LGN/cortex normalize contrast);
the best-matching kind moves toward the input, kinds that win too often are
handicapped so every kind finds a niche. Responses are rectified, scaled by the
window's contrast, kept sparse (the strongest few kinds per position) and
max-pooled over neighbouring positions (complex-cell-like tolerance).

Development is layer by layer, as in infants (V1 matures before higher areas):
each layer self-organizes on unlabeled photos while the layers below are
already stable. Nothing here sees a label or a question.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Layer:
    def __init__(self, in_ch, kinds, window, pool, active, seed):
        self.in_ch, self.kinds, self.window, self.pool, self.active = in_ch, kinds, window, pool, active
        rng = np.random.default_rng(seed)
        w = rng.normal(0, 1, (kinds, window * window * in_ch)).astype(np.float32)
        self.W = w / np.linalg.norm(w, axis=1, keepdims=True)
        self.wins = np.zeros(kinds)

    def windows(self, x):
        """(H, W, C) -> normalized windows (h, w, d) and their contrast energy (h, w)."""
        v = sliding_window_view(x, (self.window, self.window), axis=(0, 1))  # h, w, C, k, k
        v = v.transpose(0, 1, 3, 4, 2).reshape(v.shape[0], v.shape[1], -1)
        v = v - v.mean(axis=2, keepdims=True)
        e = np.linalg.norm(v, axis=2)
        return v / (e[..., None] + 1e-6), e

    def respond(self, x):
        v, e = self.windows(x)
        r = np.maximum(v @ self.W.T, 0) * e[..., None]
        if self.active < self.kinds:
            kth = np.partition(r, -self.active, axis=2)[..., -self.active][..., None]
            r = np.where(r >= kth, r, 0)
        h, w = (r.shape[0] // self.pool) * self.pool, (r.shape[1] // self.pool) * self.pool
        r = r[:h, :w].reshape(h // self.pool, self.pool, w // self.pool, self.pool, -1).max(axis=(1, 3))
        return r.astype(np.float32)

    def learn(self, x, rng, samples=32, rate=0.02):
        v, e = self.windows(x)
        v, e = v.reshape(-1, v.shape[2]), e.ravel()
        active = np.nonzero(e > 0.3 * e.max())[0] if e.max() > 1e-6 else []
        if not len(active):
            return
        for i in rng.choice(active, min(samples, len(active)), replace=False):
            s = self.W @ v[i] - 0.3 * (self.wins / (self.wins.mean() + 1e-8) - 1)
            c = int(np.argmax(s))
            self.wins[c] += 1
            w = self.W[c] + rate * (v[i] - self.W[c])
            self.W[c] = w / np.linalg.norm(w)


    def live(self, x, rng, samples=8, base=0.05, memory=0.05):
        """Learning while living (docs/PLASTICITY.md): the same competitive Hebbian step, but each kind
        learns in proportion to its own recent mismatch (1 - how well it fits the windows it wins).
        No clock: a kind whose inputs stay familiar stops changing; new kinds of input make it plastic again."""
        if not hasattr(self, "mismatch"):
            self.mismatch = np.full(self.kinds, np.nan, np.float32)  # unknown until the kind wins once
        v, e = self.windows(x)
        v, e = v.reshape(-1, v.shape[2]), e.ravel()
        active = np.nonzero(e > 0.3 * e.max())[0] if e.max() > 1e-6 else []
        if not len(active):
            return
        for i in rng.choice(active, min(samples, len(active)), replace=False):
            s = self.W @ v[i] - 0.3 * (self.wins / (self.wins.mean() + 1e-8) - 1)
            c = int(np.argmax(s))
            self.wins[c] += 1
            err = 1.0 - float(self.W[c] @ v[i])
            m = self.mismatch[c]
            self.mismatch[c] = err if np.isnan(m) else m + memory * (err - m)
            w = self.W[c] + base * self.mismatch[c] * (v[i] - self.W[c])
            self.W[c] = w / np.linalg.norm(w)


    def predictive(self, x, z_prev, rate=0.005, predict=1.0, hebb=1.0, decorrelate=1.0, memory=0.002, warmup=200):
        """The two-part rule on a sequence of views of one thing (docs/VISION.md; Halvagar & Zenke 2023,
        latent predictive learning). Each kind keeps its own slow statistics: its mean response, its variance,
        and how it co-varies with the other kinds of its layer. Seeing view x after a view where the kinds
        answered z_prev, each kind changes its synapses on the inputs that drive it now by
            predict:      toward answering as it did a moment ago (what stays the same across views is the thing)
            hebb:         away from its own mean, the more so the less it varies (so that it does not go silent
                          or constant: prediction alone collapses)
            decorrelate:  away from what the kinds it co-varies with are doing (lateral inhibition)
        all in units of its own variability. Returns the kinds' responses to x (z, to pass as z_prev next)."""
        v, e = self.windows(x)
        P = v.shape[0] * v.shape[1]
        v, e = v.reshape(P, -1), e.reshape(P)
        r = np.maximum(v @ self.W.T, 0) * e[:, None]
        z = r.mean(0)
        if not hasattr(self, "mu"):
            self.mu, self.var, self.R, self.views = z.copy(), (0.5 * z) ** 2 + 1e-8, np.zeros((self.kinds, self.kinds), np.float32), 0
        dz = z - self.mu
        sd = np.sqrt(self.var) + 1e-8
        zs = dz / sd
        self.mu += memory * dz
        self.var += memory * (dz ** 2 - self.var)
        self.R += memory * (np.outer(zs, zs) - self.R)
        self.views += 1
        if z_prev is not None and self.views > warmup:
            delta = (predict * (z - z_prev) - hebb * dz) / sd ** 2
            if decorrelate:
                off = self.R - np.diag(np.diag(self.R))
                delta += decorrelate * (off @ zs) / sd
            G = ((r > 0) * e[:, None]).T @ v / P  # what drives each kind now
            delta = np.clip(delta * np.linalg.norm(G, axis=1), -1.0, 1.0)  # a synapse changes by a bounded amount at a time
            G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
            w = self.W - rate * delta[:, None] * G
            self.W = (w / np.linalg.norm(w, axis=1, keepdims=True)).astype(np.float32)
        return z


class VisualCortex:
    def __init__(self, seed=0, kinds=(64, 128, 256)):
        k1, k2, k3 = kinds
        self.layers = [
            Layer(3, k1, 5, 2, max(8, k1 // 8), seed),
            Layer(k1, k2, 3, 2, max(12, k2 // 10), seed + 1),
            Layer(k2, k3, 3, 1, max(16, k3 // 16), seed + 2),
        ]

    def maps(self, img):
        """Activity maps of every layer for a photo (H, W, 3) in [0, 1]."""
        out, x = [], np.asarray(img, np.float32)
        for layer in self.layers:
            x = layer.respond(x)
            out.append(x)
        return out

    def code(self, img, grid=2):
        """What each layer sends on: its map max-pooled to a grid x grid summary,
        normalized per layer. Returns a list (layer 1, 2, 3)."""
        codes = []
        for m in self.maps(img):
            h, w, c = m.shape
            gh, gw = h // grid, w // grid
            p = m[: gh * grid, : gw * grid].reshape(grid, gh, grid, gw, c).max(axis=(1, 3)).ravel()
            codes.append(p / (np.linalg.norm(p) + 1e-8))
        return codes

    def live(self, img, rng, samples=8):
        """Every layer learns at once from what the eye sees (Layer.live): plasticity follows each kind's mismatch."""
        x = np.asarray(img, np.float32)
        for layer in self.layers:
            layer.live(x, rng, samples=samples)
            x = layer.respond(x)

    def watch(self, views, **kw):
        """Successive views of one thing (the eye's own fixations and zooms on a picture): every layer learns
        at once with the two-part rule (Layer.predictive)."""
        prev = [None] * len(self.layers)
        for img in views:
            x = np.asarray(img, np.float32)
            for i, layer in enumerate(self.layers):
                prev[i] = layer.predictive(x, prev[i], **kw)
                x = layer.respond(x)

    def plasticity(self):
        """Mean recent mismatch per layer (what drives learning now)."""
        return [float(np.nanmean(layer.mismatch)) if hasattr(layer, "mismatch") else None for layer in self.layers]

    def develop(self, images, rng, per_layer=4000, log=None):
        """Layer-by-layer self-organization on unlabeled photos."""
        for li, layer in enumerate(self.layers):
            for n in range(per_layer):
                img = images[rng.integers(len(images))]
                x = np.asarray(img, np.float32)
                for below in self.layers[:li]:
                    x = below.respond(x)
                layer.learn(x, rng, rate=0.05 * (1 - n / per_layer) + 0.005)
            if log:
                log(f"layer {li + 1} developed ({layer.kinds} kinds, min/max wins {int(layer.wins.min())}/{int(layer.wins.max())})")
        return self
