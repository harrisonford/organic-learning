"""A second visual cortex (docs/RESEARCH_VISION.md, docs/VISION.md 2026-11-17).

The first one (organic/visual_cortex.py) is a stack of hard winner-take-all layers whose neurons answer with their
raw drive, the strongest few kept. Measured on 2026-11-17: its developed first layer names things worse than
filters that were never developed, and its deeper layers add nothing. The literature on learning without backprop
names the reasons and the same few remedies; each is something cortex and retina are known to do:

  decorrelation    the windows of the picture reach the cortex decorrelated, as the retina's centre-surround cells
                   decorrelate natural scenes (Atick & Redlich 1992): whitening from the windows' own statistics
  lateral inhibition
                   a neuron's response is its drive above the average drive of all kinds at the same place, so the
                   code is graded and dense instead of a few winners (the "triangle" of Coates, Lee & Ng 2011)
  more kinds       V1 is highly overcomplete; 64 kinds of 5x5 windows are few
  local fan-in     above the first layer no neuron listens to every kind below: each group of neurons has its own
                   random subset of the kinds below (dendrites reach a neighbourhood, not everything)
  averaged summary what a layer sends on is the average activity of each kind in each region of the view

Learning stays what it was: winner-take-all Hebbian learning with a conscience, one layer after the other, no
labels and no backprop; only the response used to see with is graded.

Same interface as VisualCortex: cortex.layers[i].kinds, cortex.code(img, grid) -> one summary per layer.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _windows(x, k):
    """(H, W, C) -> every k x k window as a row (h, w, k*k*C)."""
    v = sliding_window_view(x, (k, k), axis=(0, 1))
    return v.transpose(0, 1, 3, 4, 2).reshape(v.shape[0], v.shape[1], -1)


def _contrast(v):
    """Each window with its mean removed and at unit length, and its contrast (the length it had)."""
    v = v - v.mean(axis=-1, keepdims=True)
    e = np.linalg.norm(v, axis=-1)
    return v / (e[..., None] + 1e-6), e


class Layer2:
    def __init__(self, in_ch, kinds, window, pool, groups=1, fan_in=None, seed=0, pool_kind="avg"):
        self.in_ch, self.kinds, self.window, self.pool, self.pool_kind = in_ch, kinds, window, pool, pool_kind
        rng = np.random.default_rng(seed)
        self.groups = groups
        per = kinds // groups
        # which kinds of the layer below each group of neurons listens to (all of them for a single group)
        self.reach = [np.sort(rng.choice(in_ch, fan_in, replace=False)) if fan_in and fan_in < in_ch else np.arange(in_ch) for _ in range(groups)]
        self.W = [(lambda w: w / np.linalg.norm(w, axis=1, keepdims=True))(rng.normal(0, 1, (per, window * window * len(r))).astype(np.float32)) for r in self.reach]
        self.Z = [None] * groups  # each group's decorrelation of its own windows
        self.wins = [np.zeros(per) for _ in range(groups)]

    def _drive(self, x):
        """(drive of every kind at every place (h, w, kinds), the windows' contrast (h, w)) and the groups' windows."""
        out, energy, seen = [], 0.0, []
        for g, reach in enumerate(self.reach):
            v, e = _contrast(_windows(x[..., reach] if len(reach) < x.shape[2] else x, self.window))
            if self.Z[g] is not None:
                v = v @ self.Z[g]
                v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
            out.append(v @ self.W[g].T)
            energy = energy + e * e
            seen.append((v, e))
        return np.concatenate(out, axis=2), np.sqrt(energy / self.groups), seen

    def respond(self, x, pooled=True):
        """The layer's activity map: each kind's drive above the average of all kinds at that place, scaled by the
        contrast of what is there; pooled for the layer above."""
        u, e, _ = self._drive(x)
        r = np.maximum(u - u.mean(axis=2, keepdims=True), 0) * e[..., None]
        if not pooled or self.pool == 1:
            return r.astype(np.float32)
        p = self.pool
        h, w = (r.shape[0] // p) * p, (r.shape[1] // p) * p
        r = r[:h, :w].reshape(h // p, p, w // p, p, -1)
        return (r.mean(axis=(1, 3)) if self.pool_kind == "avg" else r.max(axis=(1, 3))).astype(np.float32)

    def settle(self, pictures, rng, per_picture=40):
        """Decorrelation: from windows of pictures (already passed through the layers below), each group learns to
        remove its own windows' correlations. Computed here from the statistics at once; a retina does it with
        centre-surround cells, cortex with lateral anti-Hebbian inhibition (Földiák 1990)."""
        for g, reach in enumerate(self.reach):
            rows = []
            for x in pictures:
                v, e = _contrast(_windows(x[..., reach] if len(reach) < x.shape[2] else x, self.window))
                v = v.reshape(-1, v.shape[2])
                rows.append(v[rng.choice(len(v), min(per_picture, len(v)), replace=False)])
            C = np.cov(np.concatenate(rows).T)
            d, E = np.linalg.eigh(C)
            self.Z[g] = ((E / np.sqrt(np.maximum(d, 0) + 1e-3 * d.max())) @ E.T).astype(np.float32)

    def learn(self, x, rng, samples=32, rate=0.02):
        """Winner-take-all Hebbian learning with a conscience, in every group: the kind that answers a window most
        moves towards it; kinds that win too often are handicapped so that all come into use."""
        _, _, seen = self._drive(x)
        for g, (v, e) in enumerate(seen):
            v, e = v.reshape(-1, v.shape[2]), e.ravel()
            strong = np.nonzero(e > 0.3 * e.max())[0] if e.max() > 1e-6 else []
            if not len(strong):
                continue
            W, wins = self.W[g], self.wins[g]
            for i in rng.choice(strong, min(samples, len(strong)), replace=False):
                c = int(np.argmax(W @ v[i] - 0.3 * (wins / (wins.mean() + 1e-8) - 1)))
                wins[c] += 1
                w = W[c] + rate * (v[i] - W[c])
                W[c] = w / np.linalg.norm(w)


class VisualCortex2:
    def __init__(self, seed=0, kinds=(256, 256), groups=16, fan_in=32, pool_kind="avg"):
        self.layers = [Layer2(3, kinds[0], 5, 2, seed=seed, pool_kind=pool_kind)]
        for i, k in enumerate(kinds[1:], 1):
            self.layers.append(Layer2(kinds[i - 1], k, 3, 2 if i + 1 < len(kinds) else 1, groups=groups, fan_in=fan_in, seed=seed + i, pool_kind=pool_kind))

    def maps(self, img):
        """Every layer's activity map (before its pooling) for a picture (H, W, 3) in [0, 1]."""
        out, x = [], np.asarray(img, np.float32)
        for layer in self.layers:
            r = layer.respond(x, pooled=False)
            out.append(r)
            if layer.pool > 1:
                p = layer.pool
                h, w = (r.shape[0] // p) * p, (r.shape[1] // p) * p
                r = r[:h, :w].reshape(h // p, p, w // p, p, -1)
                r = r.mean(axis=(1, 3)) if layer.pool_kind == "avg" else r.max(axis=(1, 3))
            x = r
        return out

    def code(self, img, grid=2):
        """What each layer sends on: the average activity of each kind in each of grid x grid regions, at unit
        length per layer."""
        codes = []
        for m in self.maps(img):
            h, w, c = m.shape
            gh, gw = h // grid, w // grid
            p = m[: gh * grid, : gw * grid].reshape(grid, gh, grid, gw, c).mean(axis=(1, 3)).ravel()
            codes.append((p / (np.linalg.norm(p) + 1e-8)).astype(np.float32))
        return codes

    def develop(self, images, rng, per_layer=16000, log=None):
        """Layer by layer on unlabeled pictures: first the decorrelation of what the layer gets, then its kinds."""
        for li, layer in enumerate(self.layers):
            def below(img):
                x = np.asarray(img, np.float32)
                for b in self.layers[:li]:
                    x = b.respond(x)
                return x
            layer.settle([below(images[i]) for i in rng.choice(len(images), min(1500, len(images)), replace=False)], rng)
            for n in range(per_layer):
                layer.learn(below(images[rng.integers(len(images))]), rng, rate=0.05 * (1 - n / per_layer) + 0.005)
            if log:
                log(f"layer {li + 1} developed ({layer.kinds} kinds in {layer.groups} group(s); wins min/max {int(min(w.min() for w in layer.wins))}/{int(max(w.max() for w in layer.wins))})")
        return self
