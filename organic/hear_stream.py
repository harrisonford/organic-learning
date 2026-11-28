"""The text-free ear on the clock: sound in 10 ms at a time, sound kinds out as they are heard (docs/ASYNC.md, "The
ear's way onto the clock"; docs/VISION.md 2026-11-27 22:26).

The ear that finds photographs in one voice (experiments/grounded_online_probe.py) hears a caption whole: its
frames coded at once, cut into stretches, the stretches matched to kinds. This is the same ear, the same numbers,
with only the past: every 10 ms one row of the cochlea comes in and

    bands      each band against a running mean and spread of the sound so far, fading over `tau` seconds
    cortex     the hearing layer's answer to the last 80 ms (one frame a step)
    relay      as the probe's (a settled matrix, or the circuit)
    loudness   the mean of the last 80 ms, smoothed over 50 ms; "quiet" is under a tenth of the loudest moment so far
    cuts       at the troughs of loudness (known 50 ms after: a trough is the lowest of 50 ms either side) and where
               quiet begins or ends; a stretch is what lies between two cuts (two halves' mean codes); stretches
               under 50 ms or mostly quiet are gaps
    kinds      every two and every three stretches in a row without a gap go to the kinds (winner-take-all with a
               conscience, the first stretches founding them), which they move if `learning`

`step(row)` returns the kinds heard at that step, as ("pair" | "triple", kind). A kind is known about 70 ms after
its last stretch ended. `pause()` says a silence was there (an utterance ended): what is pending is closed as the
probe closes a caption, so that with a pause after every caption this ear gives the probe's bags exactly
(experiments/hear_stream_check.py). (organic/ear.py is the lives' byte ear; this is another organ.)
"""
import numpy as np


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


class Kinds:
    def __init__(self, n, dim):
        self.U, self.wins, self.founded = np.zeros((n, dim), np.float32), np.zeros(n), 0

    def __call__(self, s, learning):
        s = np.asarray(s, np.float32)  # (a stretch code at double precision made every match copy the kinds: 65 ms a kind)
        if learning and self.founded < len(self.U):
            self.U[self.founded], self.wins[self.founded] = s, 1
            self.founded += 1
            return self.founded - 1
        f = self.founded
        if f == 0:
            return None
        c = int(np.argmax(self.U[:f] @ s - (0.3 * (self.wins[:f] / (self.wins[:f].mean() + 1e-8) - 1) if learning else 0)))
        if learning:
            self.wins[c] += 1
            w = self.U[c] + (1.0 / (1 + self.wins[c] / 5)) * (s - self.U[c])
            self.U[c] = w / (np.linalg.norm(w) + 1e-8)
        return c


class StreamEar:
    def __init__(self, layer, relay, kinds=128, bands=8, k23=1024, tau=2.0, window=8, half=5, smooth=5, learning=True):
        self.layer, self.relay, self.n_kinds, self.bands, self.k23 = layer, relay, kinds, bands, k23
        self.k, self.window, self.half, self.smooth, self.learning = 0.01 / tau, window, half, smooth, learning
        D = 2 * kinds * bands
        self.pairs, self.triples = Kinds(k23, 2 * D), Kinds(k23, 3 * D)
        self.mu, self.var, self.top = None, None, 0.0
        self._fresh()

    def _fresh(self):
        self.z, self.sums = [], []  # the last rows of this utterance: its bands, its loudness
        self.C, self.loud, self.env, self.quiet = [], [], [], []  # a code frame each, since the utterance began
        self.cuts, self.done, self.chain = [0], 0, []  # cuts found; the last one closed; stretches in a row without a gap

    def _through(self, c):
        r = self.relay
        if hasattr(r, "Z"):
            return unit((unit(c) - r.mu) @ r.Z)
        return unit(r.code(unit(c)[None])[0])

    def _env(self, e, end=False):
        """Loudness at frame e smoothed over `smooth` frames (zeros beyond the utterance's ends, as the probe has it)."""
        h = self.smooth // 2
        return float(sum(self.loud[max(0, e - h): e + h + 1])) / self.smooth

    def _add_env(self, e, out):
        v = self._env(e)
        self.top = max(v, self.top * (1 - self.k))
        self.env.append(v)
        self.quiet.append(v < 0.1 * self.top)
        if e >= 1 and self.quiet[e] != self.quiet[e - 1]:
            self.cuts.append(e)

    def _trough(self, t):
        if t >= 1 and self.env[t] == min(self.env[max(0, t - self.half): t + self.half + 1]):
            self.cuts.append(t)

    def _close(self, upto, out):
        """Every stretch whose end is settled (no cut can still appear before `upto`) goes to the kinds."""
        cuts = sorted(set(self.cuts))
        i = cuts.index(self.done)
        while i + 1 < len(cuts) and cuts[i + 1] <= upto:
            a, b = cuts[i], cuts[i + 1]
            if b - a < 5 or float(np.mean(self.quiet[a:b])) > 0.5:
                self.chain = []
            else:
                h = (a + b) // 2
                C = np.asarray(self.C[a:b])
                self.chain = (self.chain + [np.concatenate([unit(C[: h - a].mean(0)), unit(C[h - a:].mean(0))]) / np.sqrt(2)])[-3:]
                if len(self.chain) >= 2:
                    k = self.pairs(np.concatenate(self.chain[-2:]) / np.sqrt(2), self.learning)
                    if k is not None:
                        out.append(("pair", k))
                if len(self.chain) >= 3:
                    k = self.triples(np.concatenate(self.chain) / np.sqrt(3), self.learning)
                    if k is not None:
                        out.append(("triple", k))
            self.done = b
            i += 1
        self.cuts = [c for c in cuts if c >= self.done]

    def step(self, row):
        row, out = np.asarray(row, np.float32), []
        if self.mu is None:
            self.mu, self.var = row.copy(), np.ones_like(row)
        self.mu = self.mu + self.k * (row - self.mu)
        self.var = self.var + self.k * ((row - self.mu) ** 2 - self.var)
        self.z.append((row - self.mu) / (np.sqrt(self.var) + 1e-3))
        self.sums.append(float(row.sum()))
        if len(self.z) < self.window:
            return out
        self.z, self.sums = self.z[-self.window:], self.sums[-self.window:]
        r = self.layer.respond(np.stack(self.z)[..., None], pooled=False)
        b = r.shape[1] // self.bands
        c = r[0, : self.bands * b].reshape(self.bands, b, self.n_kinds).mean(axis=1).ravel()
        self.C.append(self._through(c))
        self.loud.append(float(np.mean(self.sums)))
        j, h = len(self.C) - 1, self.smooth // 2
        if j - h >= 0:
            e = j - h  # the smoothed loudness of frame e is known now
            self._add_env(e, out)
            if e - self.half >= 0:
                self._trough(e - self.half)
                self._close(e - self.half, out)
        if len(self.C) > 4000 and self.done > 2000:  # an endless utterance: what is long closed is let go
            d = self.done - 100
            self.C, self.loud, self.env, self.quiet = self.C[d:], self.loud[d:], self.env[d:], self.quiet[d:]
            self.cuts, self.done = [c - d for c in self.cuts], self.done - d
        return out

    def pause(self):
        """A silence was there: the utterance is closed as the probe closes a caption, and the next begins afresh."""
        out, n = [], len(self.C)
        for e in range(len(self.env), n):
            self._add_env(e, out)
        for t in range(max(0, n - self.smooth // 2 - self.half), n - 1):
            self._trough(t)
        if n:
            self.cuts.append(n)
            self._close(n, out)
        self._fresh()
        return out
