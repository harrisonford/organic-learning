"""The ear: hears raw bytes and discovers its own units of sound.

There is no tokenizer and no string processing. An utterance arrives as the
raw UTF-8 bytes of the text, one byte at a time, followed by silence.

Inspired by Meta's Byte Latent Transformer (Pagnoni et al. 2024), which groups
bytes into patches wherever a small byte model finds the next byte hard to
predict, and by infant statistical learning (Saffran, Aslin & Newport 1996):
babies find word boundaries where transitional probabilities between
syllables drop.

1. Statistical learning: Hebbian counts of which byte follows the last 1-4
   bytes heard (with back-off to shorter contexts when a context is new).
2. Segmentation: at each byte the ear measures its surprise (entropy of the
   prediction for that byte). A new unit starts where the entropy jumps up
   (BLT's approximate-monotonic rule), or where the ear has no idea at all.
   Early in life nothing is predictable, so every byte is its own unit;
   word-like units only emerge from experience.
3. Each byte excites a fixed random set of neurons, and so does each short
   byte sequence (combination-sensitive neurons, BLT's hash n-gram
   embeddings). A unit's sound is its bytes and n-grams placed by position,
   plus its n-grams regardless of position, so units that share pieces of
   sound overlap ("red", "red?", "it red and").
4. Habituation (stimulus-specific adaptation): units heard all the time
   ("is", "it", "?") excite less than rare, informative ones. Off by default
   since 2026-10-25: the orchestrator's input adaptation does this downstream.

The silence after an utterance is heard too: it is the empty unit b"".
"""

import zlib

import numpy as np

SILENCE = b""


class Ear:
    START = b"\x02"  # marker for "I am about to speak" in the self-echo

    def __init__(self, width=512, sparsity=16, max_context=4, rise=1.2, unknown=0.85, seed=7):
        self.width = width
        self.sparsity = sparsity
        self.max_context = max_context
        self.rise = rise  # entropy jump (bits) that starts a new unit
        self.h_max = np.log2(257)
        self.unknown = unknown * self.h_max  # "no idea at all" also starts a unit
        self.seed = seed
        self.counts = {}  # context bytes -> counts of the next byte (256 = silence)
        self.codes = {}
        self.heard = {}
        self.utterances = 0

    # ------------------------------------------------ statistical learning

    def _predict(self, context):
        for n in range(min(self.max_context, len(context)), -1, -1):
            c = self.counts.get(bytes(context[len(context) - n :]))
            if c is not None and c.sum() >= 2:
                return c
        return None

    def surprise(self, context):
        """Entropy (bits) of the ear's guess about the byte that comes next."""
        c = self._predict(context)
        if c is None:
            return self.h_max
        p = (c + 0.01) / (c.sum() + 0.01 * 257)
        return float(-(p * np.log2(p)).sum())

    def _learn(self, data):
        for i in range(len(data) + 1):
            nxt = data[i] if i < len(data) else 256
            for n in range(0, min(self.max_context, i) + 1):
                key = bytes(data[i - n : i])
                c = self.counts.get(key)
                if c is None:
                    c = self.counts[key] = np.zeros(257, np.float32)
                c[nxt] += 1

    def units(self, text, learn=True):
        """Hear an utterance; return the units the ear segments it into."""
        data = text.encode("utf-8")
        out, start, prev_h = [], 0, None
        for i in range(len(data)):
            h = self.surprise(data[:i])
            if i > 0 and (h - prev_h > self.rise or h >= self.unknown):
                out.append(data[start:i])
                start = i
            prev_h = h
        if data:
            out.append(data[start:])
        if learn:
            self._learn(data)
            self.utterances += 1
            for u in set(out):
                self.heard[u] = self.heard.get(u, 0) + 1
        return out

    # --------------------------------------------------------------- sound

    def _neurons(self, key):
        """A fixed random sparse pattern for a byte or byte sequence."""
        rng = np.random.default_rng(zlib.crc32(key) ^ self.seed)
        v = np.zeros(self.width, np.float32)
        idx = rng.choice(self.width, self.sparsity, replace=False)
        v[idx] = rng.choice([-1.0, 1.0], self.sparsity)
        return v

    def sound(self, unit):
        """How a unit excites auditory cortex.

        Two populations: byte/n-gram neurons tied to where they occur in the
        unit (order), and combination-sensitive neurons that respond to a short
        byte sequence wherever it occurs (so "red" and "it red and" overlap).
        """
        if unit not in self.codes:
            placed = np.zeros(self.width, np.float32)
            anywhere = np.zeros(self.width, np.float32)
            if unit == SILENCE:
                placed = self._neurons(b"silence")
            for k in range(len(unit)):
                f = self._neurons(unit[k : k + 1])
                for n in (2, 3):
                    if k + 1 >= n:
                        g = self._neurons(b"ng" + unit[k + 1 - n : k + 1])
                        f = f + 0.7 * g
                        anywhere += g
                placed += np.roll(f, 5 * k)
            v = placed / (np.linalg.norm(placed) + 1e-8)
            if anywhere.any():
                v = v + anywhere / np.linalg.norm(anywhere)
            self.codes[unit] = v / (np.linalg.norm(v) + 1e-8)
        return self.codes[unit]

    def combinations(self, data):
        """The combination-sensitive population's response to a stretch of speech, wherever the ear cut it into
        units: every short byte sequence (2 or 3 bytes) excites its own neurons, summed and scaled to unit size.
        "dog", "dog?" and "ooh a dog" share most of it (compare Wickelfeatures, Rumelhart & McClelland 1986)."""
        v = np.zeros(self.width, np.float32)
        for n in (2, 3):
            for k in range(len(data) - n + 1):
                key = b"ng" + data[k : k + n]
                if key not in self.combo:
                    self.combo[key] = self._neurons(key)
                v += self.combo[key]
        return v / (np.linalg.norm(v) + 1e-8)

    combo = {}  # the n-gram neurons' fixed patterns (a cache shared by all ears with the same seed and width)

    # Off by default since 2026-10-25: a controlled pair (100k moments each) showed no difference with the
    # fixed rule on or off; the orchestrator's own input adaptation does the job (docs/LOG.md)
    habituation = False

    def salience(self, unit):
        """Habituated response: rare units stand out, ubiquitous ones fade."""
        if self.utterances == 0 or not self.habituation:
            return 1.0
        freq = (self.heard.get(unit, 0) + 1) / (self.utterances + 1)
        return float(np.clip(1.0 / np.sqrt(freq), 1.0, 8.0))

    # --------------------------------------------------------------- echoes

    def echo(self, units, depth, decay, voice=None, habituate=False):
        voice = voice or self.sound
        v = np.zeros(self.width, np.float32)
        for k, u in enumerate(reversed(units[-depth:])):
            weight = self.salience(u) if habituate else 1.0
            v += weight * decay**k * np.roll(voice(u), 11 * (k + 1))
        return v

    # synaptic fatigue (stimulus-specific adaptation), off for brains made before it
    fatigue = False
    fuel = None

    def enable_fatigue(self, seed=None):
        """Every auditory neuron gets its own fatigue: how much it tires when it fires
        (some barely at all) and how fast it recovers. Neurons that fire often stay
        tired, so sounds heard all the time come through weaker than rare ones. No
        knowledge of words: a property of synapses, different in every neuron."""
        rng = np.random.default_rng(self.seed + 13 if seed is None else seed)
        self.drain = rng.uniform(0.0, 0.5, self.width).astype(np.float32)
        self.recover = np.exp(rng.uniform(np.log(0.002), np.log(0.05), self.width)).astype(np.float32)
        self.fuel = np.ones(self.width, np.float32)
        self.fatigue = True

    def _heard(self, units, adapt):
        """Each unit as the auditory neurons actually respond to it, in order, and the
        fatigue it leaves (only while living, not in a frozen brain)."""
        fuel = self.fuel if adapt else self.fuel.copy()
        out = {}
        for u in units:
            s = self.sound(u)
            out[u] = s * fuel
            a = np.abs(s)
            fuel -= self.drain * a / (a.max() + 1e-8) * fuel
            fuel += self.recover * (1.0 - fuel)
        return out

    # prediction-error hearing (predictive coding), off for brains made before it
    predictive = False

    def enable_prediction(self, rate=0.1, trace=0.0):
        """The auditory cortex learns to expect: from what it has just heard (the last
        unit, or a fading trace of the last few when trace > 0), it predicts the sound
        of the next unit, and what it passes on is only the error, what was heard minus
        what was expected. Each neuron adjusts its own prediction weights by its own
        error (a local delta rule), scaled down when its input fired strongly (divisive
        normalization, so learning neither stalls nor oscillates). Nothing says which
        words matter: what is cancelled is whatever experience has made predictable."""
        self.P = np.zeros((self.width, self.width), np.float32)
        self.p_rate, self.p_trace = rate, trace
        self.predictive = True

    def _surprise_codes(self, units, learn):
        out = {}
        ctx = self.sound(SILENCE).copy()
        for u in list(units) + [SILENCE]:  # the silence after the utterance is expected too
            s = self.sound(u)
            err = s - self.P @ ctx
            if learn:
                self.P += (self.p_rate / (float(ctx @ ctx) + 1e-6)) * np.outer(err, ctx)
            if u != SILENCE:
                out[u] = err
            ctx = self.p_trace * ctx + (1 - self.p_trace) * s
        return out

    def question(self, units, voice=None, adapt=True):
        """A fading, order-aware echo of what was asked, plus its gist."""
        if voice is None and self.predictive:
            voice = self._surprise_codes(units, adapt).__getitem__
        if voice is None and self.fatigue:
            heard = self._heard(units, adapt)
            voice = heard.__getitem__
        voice = voice or self.sound
        seq = self.echo(units, depth=8, decay=0.75, voice=voice, habituate=True)
        gist = sum((self.salience(u) * voice(u) for u in units), np.zeros(self.width, np.float32))
        v = seq / (np.linalg.norm(seq) + 1e-8) + gist / (np.linalg.norm(gist) + 1e-8)
        return v / (np.linalg.norm(v) + 1e-8)

    def said(self, units, voice=None):
        """Efference copy of my own speech: the last few units I produced."""
        v = self.echo([self.START] + list(units), depth=3, decay=0.5, voice=voice)
        return v / (np.linalg.norm(v) + 1e-8)
