"""The ear: hears words and keeps a fading echo of them.

Hard-wired, like the cochlea and early auditory cortex. Every word gets a fixed
random sparse "sound" the first time it is heard. Two echoes reach cortex:

- what was asked: the question's words, each rotated by how long ago it was
  heard (so order matters) and fading with time, plus an even, order-free
  "gist" of all of it;
- what I have said so far (efference copy of the organism's own speech): a
  short fading echo of the last few words, plus a marker for "I'm about to
  start answering".

Habituation (stimulus-specific adaptation): auditory neurons respond less to
sounds they hear all the time. Each word's echo is scaled by 1/sqrt of how
often it has been heard, so rare, informative words ("color", "where")
stand out over ubiquitous ones ("is", "it", "?").
"""

import numpy as np


class Ear:
    START = "<start>"

    def __init__(self, width=512, sparsity=24, seed=7):
        self.width = width
        self.sparsity = sparsity
        self.rng = np.random.default_rng(seed)
        self.sounds = {}
        self.heard = {}  # word -> number of utterances it was heard in
        self.utterances = 0

    def sound(self, word):
        if word not in self.sounds:
            v = np.zeros(self.width, np.float32)
            idx = self.rng.choice(self.width, self.sparsity, replace=False)
            v[idx] = self.rng.choice([-1.0, 1.0], self.sparsity)
            self.sounds[word] = v / np.sqrt(self.sparsity)
        return self.sounds[word]

    @staticmethod
    def words(text):
        for p in "?.,!":
            text = text.replace(p, f" {p} ")
        return text.lower().split()

    def listen(self, words):
        """Hearing an utterance habituates the neurons tuned to its words."""
        self.utterances += 1
        for w in set(words):
            self.heard[w] = self.heard.get(w, 0) + 1

    def salience(self, w):
        if self.utterances == 0:
            return 1.0
        freq = (self.heard.get(w, 0) + 1) / (self.utterances + 1)
        return float(np.clip(1.0 / np.sqrt(freq), 1.0, 8.0))

    def echo(self, words, depth, decay, voice=None, habituate=False):
        voice = voice or self.sound
        v = np.zeros(self.width, np.float32)
        for k, w in enumerate(reversed(words[-depth:])):
            weight = self.salience(w) if habituate else 1.0
            v += weight * decay**k * np.roll(voice(w), 11 * (k + 1))
        return v

    def question(self, words):
        seq = self.echo(words, depth=6, decay=0.75, habituate=True)
        gist = sum((self.salience(w) * self.sound(w) for w in words), np.zeros(self.width, np.float32))
        v = seq / (np.linalg.norm(seq) + 1e-8) + gist / (np.linalg.norm(gist) + 1e-8)
        return v / (np.linalg.norm(v) + 1e-8)

    def said(self, words, voice=None):
        v = self.echo([self.START] + words, depth=3, decay=0.5, voice=voice)
        return v / (np.linalg.norm(v) + 1e-8)
