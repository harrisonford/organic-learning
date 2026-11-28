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
import copy, threading
import numpy as np
from .cortex2 import Layer2, RelayCircuit, _contrast, _windows


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


class HearingLayer(Layer2):
    """A hearing cortex born with the brain (2026-11-28 07:40; docs/VISION.md 06:39, measured there as LAYER=born of
    experiments/grounded_online_probe.py). The same layer as the kept ears' (organic/cortex2.py Layer2: kinds of
    window x window patches of the cochlea), with nothing fitted beforehand: the correlations of its windows are
    removed by a circuit of interneurons with the relay's local rule (RelayCircuit on a window's values, at the
    published rate for tens of cells), and its kinds learn by their own rule (winner-take-all with a conscience),
    each kind's rate falling with its own wins as the sound kinds' do. One window a frame, at a place chosen by
    chance: `pick` after each answer, `learn_from` with the picks of an utterance when it has ended (the ear
    decides which; the kinds take only windows with contrast)."""

    def __init__(self, kinds=128, window=8, seed=0, rate=1e-3):
        super().__init__(1, kinds, window, 1, seed=seed)
        self.circuit = RelayCircuit(window * window, rate=rate, block=50, seed=seed)
        self.at, self.top, self.rng, self.last = None, 0.0, np.random.default_rng(seed), None

    def _drive(self, x):
        raw, e = _contrast(_windows(x, self.window))
        if self.at is None:  # the circuit's settled state as one matrix, until its synapses change
            self.at = StreamEar._settled(self.circuit)
        inv, mu, scale = self.at
        v = unit(((raw - mu) / scale) @ inv)
        self.last = (raw, v, e)
        return v @ self.W[0].T, e, [(v, e)]

    def pick(self, k):
        """One window of the last frame answered: as the circuit gets it, as the kinds get it, and whether its
        contrast is over 0.3 of the strongest lately (fading at k a frame)."""
        raw, v, e = self.last
        at = int(self.rng.integers(raw.shape[1]))
        self.top = max(float(e[-1].max()), self.top * (1 - k))
        return raw[-1, at], v[-1, at], bool(e[-1, at] > 0.3 * self.top)

    def learn_from(self, picks):
        W, wins = self.W[0], self.wins[0]
        for _, s, strong in picks:
            if strong:
                c = int(np.argmax(W @ s - 0.3 * (wins / (wins.mean() + 1e-8) - 1)))
                wins[c] += 1
                w = W[c] + (1.0 / (1 + wins[c] / 5)) * (s - W[c])
                W[c] = w / np.linalg.norm(w)
        R = np.stack([r for r, _, _ in picks])
        if float(R.std(axis=0).max()) > 1e-6:  # windows all alike (a dead microphone) teach the circuit nothing
            self.circuit.learn(R)
            self.at = None


class StreamEar:
    def __init__(self, layer, relay, kinds=128, bands=8, k23=1024, tau=2.0, window=8, half=5, smooth=5, learning=True):
        self.layer, self.relay, self.n_kinds, self.bands, self.k23 = layer, relay, kinds, bands, k23
        self.k, self.window, self.half, self.smooth, self.learning = 0.01 / tau, window, half, smooth, learning
        D = 2 * kinds * bands
        self.pairs, self.triples = Kinds(k23, 2 * D), Kinds(k23, 3 * D)
        self.mu, self.var, self.top = None, None, 0.0
        # (2026-11-28) a relay that is a circuit (organic/cortex2.py RelayCircuit) goes on learning from what is heard,
        # as in the one-pass probe: the frames of an utterance are coded with the synapses as they were when it began,
        # and at its end (or after `relay_block` frames of an endless one) every synapse changes by them. Off unless asked.
        # True: by every frame, as the probe; "sound": only by the frames this ear does not hold to be quiet (a
        # microphone is never silent, and a circuit that heard the room for minutes would be the room's).
        # `relay_later`: that change is worked out beside the clock (about a tenth of a second for 1,024 cells) and the
        # ear goes on with the synapses as they were until it is done.
        self.relay_learns, self.relay_block, self.X = False, 1000, []
        self.relay_later, self._relay_busy, self._relay_at = False, False, None
        # (2026-11-28 07:40) `layer_learns` with a HearingLayer: the hearing cortex too learns from what is heard, by
        # the picks of an utterance at its end (or after `relay_block` frames of an endless one). True: every frame;
        # "sound": only frames this ear does not hold to be quiet. Off unless asked.
        self.layer_learns, self.P = False, []
        self._fresh()

    def __setstate__(self, d):  # an ear kept before the relay could learn, or while a change was being worked out
        self.__dict__.update(d)
        for k, v in (("relay_learns", False), ("relay_block", 1000), ("relay_later", False), ("_relay_at", None), ("layer_learns", False)):
            self.__dict__.setdefault(k, v)
        for k in ("X", "F", "P", "L"):
            self.__dict__.setdefault(k, [])
        self._relay_busy = False

    def _fresh(self):
        self.z, self.sums = [], []  # the last rows of this utterance: its bands, its loudness
        self.C, self.loud, self.env, self.quiet = [], [], [], []  # a code frame each, since the utterance began
        self.cuts, self.done, self.chain = [0], 0, []  # cuts found; the last one closed; stretches in a row without a gap
        self.F = []  # a frame each as the relay receives it (kept only when `relay_learns` is "sound")
        self.L = []  # a frame each: the hearing cortex's pick (kept only when `layer_learns`)
        self.told = []  # a code frame each: whether the listener said this moment holds no sound of the world (see step)

    def _through(self, c):
        r = self.relay
        if hasattr(r, "Z"):
            return unit((unit(c) - r.mu) @ r.Z)
        x = unit(c)
        if self.relay_learns == "sound":  # only frames this ear holds to be sound (not quiet) reach the circuit's learning
            self.F.append(x)
        elif self.relay_learns and self.learning:
            self.X.append(x)
            if len(self.X) >= self.relay_block:
                self._relay_learn()
        if self._relay_at is None:
            self._relay_at = self._settled(r)
        inv, mu, scale = self._relay_at  # the circuit's settled state for any frame, with the synapses as they are
        return unit(((x - mu) / scale) @ inv)

    @staticmethod
    def _settled(r):
        import scipy.linalg as sl
        D = len(r.mu)
        r.code(np.zeros((1, D), np.float32))
        return np.ascontiguousarray(sl.cho_solve(r._A, np.eye(D, dtype=np.float32))), r.mu.copy(), np.float32(np.sqrt(r.var))

    def _relay_learn(self):
        if not self.X or self._relay_busy:
            return
        X, self.X = np.asarray(self.X, np.float32), []

        def work():
            r = copy.copy(self.relay)
            r.W, r.mu = r.W.copy(), r.mu.copy()
            r.learn(X)
            at = self._settled(r)
            self.relay, self._relay_at, self._relay_busy = r, at, False

        if self.relay_later:
            self._relay_busy = True
            threading.Thread(target=work, daemon=True).start()
        else:
            work()

    def _layer_learn(self):
        if self.P:
            P, self.P = self.P, []
            self.layer.learn_from(P)

    def _env(self, e, end=False):
        """Loudness at frame e smoothed over `smooth` frames (zeros beyond the utterance's ends, as the probe has it)."""
        h = self.smooth // 2
        return float(sum(self.loud[max(0, e - h): e + h + 1])) / self.smooth

    def _add_env(self, e, out):
        v = self._env(e)
        self.top = max(v, self.top * (1 - self.k))
        self.env.append(v)
        told = self.told[e]  # False, True, or a level of loudness under which the listener holds there is no sound
        self.quiet.append(v < 0.1 * self.top or told is True or (told is not False and v < told))
        if e >= 1 and self.quiet[e] != self.quiet[e - 1]:
            self.cuts.append(e)
        if self.relay_learns == "sound" and self.learning and not self.quiet[e]:
            self.X.append(self.F[e])
            if len(self.X) >= self.relay_block:
                self._relay_learn()
        if self.layer_learns and self.learning and (self.layer_learns is True or not self.quiet[e]):
            self.P.append(self.L[e])
            if len(self.P) >= self.relay_block:
                self._layer_learn()

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

    def step(self, row, quiet=False):
        """`quiet`: the listener's own word that this row is not a sound of the world (a room's hum under its level, or
        its own voice): it counts as quiet whatever its loudness. Or a number: the loudness (as this ear smooths it)
        under which there is no sound, the room's level as the listener keeps it."""
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
        if self.layer_learns:
            self.L.append(self.layer.pick(self.k))
        self.loud.append(float(np.mean(self.sums)))
        self.told.append(quiet if isinstance(quiet, bool) else float(quiet))
        j, h = len(self.C) - 1, self.smooth // 2
        if j - h >= 0:
            e = j - h  # the smoothed loudness of frame e is known now
            self._add_env(e, out)
            if e - self.half >= 0:
                self._trough(e - self.half)
                self._close(e - self.half, out)
        if len(self.C) > 4000 and self.done > 2000:  # an endless utterance: what is long closed is let go
            d = self.done - 100
            self.C, self.loud, self.env, self.quiet, self.told = self.C[d:], self.loud[d:], self.env[d:], self.quiet[d:], self.told[d:]
            self.F, self.L = self.F[d:], self.L[d:]
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
        if self.relay_learns and self.learning:
            self._relay_learn()
        if self.layer_learns and self.learning:
            self._layer_learn()
        return out
