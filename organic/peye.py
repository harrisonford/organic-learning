"""A progressive eye with a learning motor organ.

Nothing here knows about objects, categories or tasks. The organ turns an
image into a stream of activity over time, and learns, by itself, both how
to represent what it sees and where to look.

Sensor (fixed, raw)
    At each fixation three views arrive, one per tick, coarse to fine:
      tick 1  the whole image at low resolution (8x8),
      tick 2  a region around the gaze point at medium resolution (16x16 -> 8x8),
      tick 3  a small region at full resolution (8x8).
    The only processing: each view is shifted to zero mean and scaled to unit
    length (so only its pattern matters, not its overall brightness).

Clump (learning)
    Neighbouring groups of neurons, one per level (G1 coarse, G2 medium, G3
    fine), and a top group (G4) that integrates across the fixations of one
    image. Every group grows assemblies by novelty and keeps a leaky state.
    Coding is distributed: the 16 best-matching assemblies respond, graded,
    so similar inputs share active assemblies. Each group has a fixed budget
    of assemblies; when it is full, novelty recycles the least-used one.
    The groups are neighbours: G4 reads the others' states through their
    assemblies' fixed axon patterns, together with where the eye points.

    Prediction: each coarser group learns what the next, finer view tends to
    look like when its assemblies are active (G1 predicts G2's view given where
    the eye is about to land, G2 predicts G3's view). Prediction error scales
    how fast the receiving group adapts: what was expected changes little,
    what was surprising changes more.

    Trace rule: assemblies that were just active are favoured, so the views
    of one image tend to gather in the same top-level assemblies.

Motor organ (learning)
    A map of possible gaze targets (a 5x5 grid over the image). Its drive comes
    from the coarse group and the top group (what is seen so far). The target
    is chosen by competition with some exploration noise. After the fixation, a
    dopamine-like signal compares how the moment went with what is usual, and
    strengthens or weakens the synapses that chose that target. Two kinds of
    signal can be tried:
      "settle":  reward = how much more recognizable the image became
                 (the top group's best match rose),
      "curious": reward = how surprising the new fine view was
                 (prediction error at G3).
"""

import numpy as np

from .area import SensoryArea


def unit(v):
    v = np.asarray(v, np.float32).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def block_mean(img, f):
    H, W, C = img.shape
    return img[: H // f * f, : W // f * f].reshape(H // f, f, W // f, f, C).mean(axis=(1, 3))


class Sensor:
    """Raw multi-resolution views around a gaze point."""

    def __init__(self, size=32, grid=5):
        self.size = size
        self.targets = [(x, y) for y in np.linspace(4, size - 4, grid) for x in np.linspace(4, size - 4, grid)]

    @staticmethod
    def _crop(img, cx, cy, side):
        H, W, C = img.shape
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        out = np.zeros((side, side, C), np.float32)
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(W, x0 + side), min(H, y0 + side)
        if xs1 > xs0 and ys1 > ys0:
            out[ys0 - y0 : ys1 - y0, xs0 - x0 : xs1 - x0] = img[ys0:ys1, xs0:xs1]
        return out

    def views(self, img, target):
        cx, cy = self.targets[target]
        return [
            unit(block_mean(img, img.shape[0] // 8)),  # whole image, coarse
            unit(block_mean(self._crop(img, cx, cy, 16), 2)),  # around the gaze, medium
            unit(self._crop(img, cx, cy, 8)),  # at the gaze, fine
        ]

    def gaze(self, target):
        v = np.zeros(len(self.targets), np.float32)
        v[target] = 1.0
        return v


class Predictor:
    """What the next level tends to look like when an assembly is active (outstar)."""

    def __init__(self, dim, contexts=1):
        self.dim, self.contexts = dim, contexts
        self.P = np.zeros((0, contexts, dim), np.float32)
        self.n = np.zeros((0, contexts), np.float32)

    def _grow(self, C):
        if C > len(self.P):
            extra = C - len(self.P)
            self.P = np.concatenate([self.P, np.zeros((extra, self.contexts, self.dim), np.float32)])
            self.n = np.concatenate([self.n, np.zeros((extra, self.contexts), np.float32)])

    def predict(self, state, context=0):
        self._grow(len(state))
        w = state * (self.n[: len(state), context] > 0)
        if w.sum() <= 0:
            return None
        return (w[:, None] * self.P[: len(state), context]).sum(axis=0) / w.sum()

    def learn(self, state, x, context=0):
        self._grow(len(state))
        idx = np.nonzero(state > 0.05 * state.max())[0] if state.max() > 0 else []
        for c in idx:
            self.n[c, context] += state[c]
            rate = state[c] / self.n[c, context]
            self.P[c, context] += rate * (x - self.P[c, context])


class Motor:
    """A gaze-target map whose synapses are tuned by a dopamine-like signal."""

    def __init__(self, inputs, targets, temperature=0.1, rate=0.05, seed=0):
        self.W = np.zeros((inputs, targets), np.float32)
        self.temperature = temperature
        self.rate = rate
        self.rng = np.random.default_rng(seed)
        self.baseline = 0.0

    def choose(self, drive, explore=True):
        u = drive @ self.W
        if not explore:
            return int(np.argmax(u + 1e-6 * self.rng.standard_normal(len(u))))
        p = np.exp((u - u.max()) / self.temperature)
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p))

    def reinforce(self, drive, target, reward):
        delta = reward - self.baseline
        self.baseline += 0.01 * (reward - self.baseline)
        self.W[:, target] += self.rate * delta * drive


class ProgressiveEye:
    def __init__(self, fixations=4, vigilance=0.8, decay=0.5, trace=0.5, motor="settle", capacity=4096, seed=0):
        self.sensor = Sensor()
        T = len(self.sensor.targets)
        self.fixations = fixations
        self.decay = decay
        self.trace = trace
        self.motor_signal = motor  # "settle", "curious", "random" or "center"
        kw = dict(vigilance=vigilance, axon_dim=256, population=16, temp=0.1, capacity=capacity)
        self.groups = [SensoryArea(f"G{i + 1}", 192, seed=seed + i, **kw) for i in range(3)]
        self.top = SensoryArea("G4", 3 * 256 + T, seed=seed + 3, **kw)
        self.predict_mid = Predictor(192, contexts=T)  # G1 -> next medium view, given gaze target
        self.predict_fine = Predictor(192)  # G2 -> next fine view
        self.motor = Motor(2 * 256, T, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.out_dim = 4 * 256

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _decayed(state, area, decay):
        if len(state) < area.C:
            state = np.concatenate([state, np.zeros(area.C - len(state), np.float32)])
        return decay * state

    @staticmethod
    def _out(area, state):
        if not len(state) or state.max() <= 0:
            return np.zeros(area.axon_dim, np.float32)
        return state @ area.axons[: len(state)]

    def _perceive(self, area, state, x, learn, plasticity=1.0):
        state = self._decayed(state, area, self.decay)
        bias = self.trace * state / state.max() if self.trace and len(state) and state.max() > 0 else None
        idx, act = area.perceive(x, learn, bias, plasticity)
        state = self._decayed(state, area, 1.0)
        state[idx] += act
        return state

    def output(self, states):
        parts = [self._out(a, s) for a, s in zip(self.groups + [self.top], states)]
        return np.concatenate([p / (np.linalg.norm(p) + 1e-8) for p in parts])

    # -------------------------------------------------------------------- look

    def look(self, img, learn=True, explore=True):
        """View one image; returns the organ's output vector after every tick."""
        img = np.asarray(img, np.float32)
        states = [np.zeros(0, np.float32) for _ in range(4)]
        outputs, path, errors = [], [], []
        target = 12  # the eye starts at the center of the 5x5 grid
        for f in range(self.fixations):
            if f > 0:
                drive = np.concatenate([self._out(self.groups[0], states[0]), self._out(self.top, states[3])])
                drive /= np.linalg.norm(drive) + 1e-8
                if self.motor_signal == "random":
                    target = int(self.rng.integers(len(self.sensor.targets)))
                elif self.motor_signal == "center":
                    target = 12
                else:
                    target = self.motor.choose(drive, explore)
            path.append(target)
            settle_before = self._recognition(states)
            views = self.sensor.views(img, target)
            surprise = []
            for level, x in enumerate(views):
                scale = 1.0
                if level == 1:
                    pred = self.predict_mid.predict(states[0], target)
                    err = 1.0 - float(pred @ x / (np.linalg.norm(pred) + 1e-8)) if pred is not None else 1.0
                    surprise.append(err)
                    scale = 0.25 + err
                    if learn:
                        self.predict_mid.learn(states[0], x, target)
                elif level == 2:
                    pred = self.predict_fine.predict(states[1])
                    err = 1.0 - float(pred @ x / (np.linalg.norm(pred) + 1e-8)) if pred is not None else 1.0
                    surprise.append(err)
                    scale = 0.25 + err
                    if learn:
                        self.predict_fine.learn(states[1], x)
                states[level] = self._perceive(self.groups[level], states[level], x, learn, scale)
                states[3] = self._perceive(self.top, states[3], self._top_input(states, target), learn)
                outputs.append(self.output(states))
            errors.append(surprise)
            if learn and f > 0 and self.motor_signal in ("settle", "curious"):
                reward = (self._recognition(states) - settle_before) if self.motor_signal == "settle" else surprise[-1]
                self.motor.reinforce(drive, target, reward)
        return outputs, path, errors

    def _top_input(self, states, target):
        parts = [self._out(a, s) for a, s in zip(self.groups, states[:3])]
        parts = [p / (np.linalg.norm(p) + 1e-8) for p in parts]
        return np.concatenate(parts + [self.sensor.gaze(target)])

    def _recognition(self, states):
        """How well the top group's current state matches what it knows (0-1)."""
        s = states[3]
        if not len(s) or s.max() <= 0:
            return 0.0
        return float(s.max() / (s.sum() + 1e-8))

    def report(self):
        sizes = " ".join(f"{a.name}={a.C}" for a in self.groups + [self.top])
        return f"assemblies {sizes}"
