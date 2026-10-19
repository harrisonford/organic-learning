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
