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
   ("is", "it", "?") excite less than rare, informative ones.

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

    def salience(self, unit):
        """Habituated response: rare units stand out, ubiquitous ones fade."""
        if self.utterances == 0:
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

    def question(self, units, voice=None):
        """A fading, order-aware echo of what was asked, plus its gist."""
        voice = voice or self.sound
        seq = self.echo(units, depth=8, decay=0.75, voice=voice, habituate=True)
        gist = sum((self.salience(u) * voice(u) for u in units), np.zeros(self.width, np.float32))
        v = seq / (np.linalg.norm(seq) + 1e-8) + gist / (np.linalg.norm(gist) + 1e-8)
        return v / (np.linalg.norm(v) + 1e-8)

    def said(self, units, voice=None):
        """Efference copy of my own speech: the last few units I produced."""
        v = self.echo([self.START] + list(units), depth=3, decay=0.5, voice=voice)
        return v / (np.linalg.norm(v) + 1e-8)
