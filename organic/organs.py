"""Sensory organs: they turn raw input into a sensory vector the cortex can feel.

Organs are not trained. Like a retina or a cochlea they apply a fixed,
hard-wired transformation, and each one writes into its own region of the
shared sensory sheet. The cortex never learns which region belongs to which
organ; clusters specialize on their own because their identities drift toward
whatever stimuli they keep responding to.

Each organ also owns a group of output neurons in the cortex ("vision:*",
"language:*") and grows a new one the first time it meets a new label or word.
"""

import numpy as np


class Organ:
    prefix = "organ"

    def __init__(self, organism, start, width):
        self.organism = organism
        self.cortex = organism.cortex
        self.start, self.width = start, width

    def place(self, local):
        """Write a local feature vector into this organ's region of the sheet."""
        s = np.zeros(self.cortex.dim, np.float32)
        s[self.start : self.start + self.width] = local
        n = np.linalg.norm(s)
        return s / n if n > 0 else s

    def output(self, name):
        return self.cortex.output_neuron(f"{self.prefix}:{name}")

    def group(self):
        return self.cortex.output_group(f"{self.prefix}:")

    def decode(self, group, probs):
        return self.cortex.outputs[group[int(np.argmax(probs))]].split(":", 1)[1]


class VisionOrgan(Organ):
    """A tiny retina for 16x16 grayscale images.

    Features: a blurred copy of the image (where light is), plus four oriented
    edge maps pooled over an 8x8 grid (what kind of contours are where).
    """

    prefix = "vision"
    size = 16

    def __init__(self, organism, start, width=512):
        super().__init__(organism, start, width)
        assert width >= 512

    def encode(self, img):
        img = np.asarray(img, np.float32)
        img = img - img.mean()
        blur = (img + np.roll(img, 1, 0) + np.roll(img, -1, 0) + np.roll(img, 1, 1) + np.roll(img, -1, 1)) / 5
        gx = np.roll(blur, -1, 1) - np.roll(blur, 1, 1)
        gy = np.roll(blur, -1, 0) - np.roll(blur, 1, 0)
        edges = [np.abs(gx), np.abs(gy), np.abs(gx + gy), np.abs(gx - gy)]
        pooled = [e.reshape(8, 2, 8, 2).mean(axis=(1, 3)).ravel() for e in edges]
        local = np.zeros(self.width, np.float32)
        local[:256] = blur.ravel()
        local[256:512] = np.concatenate(pooled) * 2.0
        return self.place(local)

    def learn(self, img, label):
        target = self.output(label)
        return self.cortex.learn(self.encode(img), target, self.group())

    def recognize(self, img):
        group = self.group()
        r = self.cortex.forward(self.encode(img), group)
        if r is None or r["probs"] is None:
            return "?", 0.0
        return self.decode(group, r["probs"]), float(r["probs"].max())


class LanguageOrgan(Organ):
    """Hears text one word at a time and speaks one word at a time.

    Each word gets a fixed random "sound" vector the first time it is heard.
    The sensory context is a fading echo of the last few words, where each
    word's vector is rotated by how long ago it was heard (so order matters),
    plus a quieter echo of the whole prompt (so the topic is remembered).
    """

    prefix = "language"
    SEP, END = "<sep>", "<end>"

    def __init__(self, organism, start, width=512, echo=4, decay=0.6, seed=1):
        super().__init__(organism, start, width)
        self.echo, self.decay = echo, decay
        self.rng = np.random.default_rng(seed)
        self.sounds = {}

    def sound(self, word):
        if word not in self.sounds:
            v = np.zeros(self.width, np.float32)
            idx = self.rng.choice(self.width, 24, replace=False)
            v[idx] = self.rng.choice([-1.0, 1.0], 24)
            self.sounds[word] = v
        return self.sounds[word]

    @staticmethod
    def words(text):
        return text.lower().replace("?", " ?").replace(".", " .").replace(",", " ,").split()

    def encode(self, prompt_words, heard):
        local = np.zeros(self.width, np.float32)
        for k, w in enumerate(reversed(heard[-self.echo :])):
            local += self.decay**k * np.roll(self.sound(w), 7 * k)
        topic = sum((self.sound(w) for w in prompt_words), np.zeros(self.width, np.float32))
        if prompt_words:
            local += 0.5 * np.roll(topic / np.sqrt(len(prompt_words)), 97)
        return self.place(local)

    def learn_dialogue(self, prompt, response):
        p = self.words(prompt)
        heard = p + [self.SEP]
        stats = []
        for w in self.words(response) + [self.END]:
            target = self.output(w)
            stats.append(self.cortex.learn(self.encode(p, heard), target, self.group()))
            heard.append(w)
        return stats

    def respond(self, prompt, max_words=16, temperature=0.0):
        p = self.words(prompt)
        heard = p + [self.SEP]
        group = self.group()
        out = []
        for _ in range(max_words):
            r = self.cortex.forward(self.encode(p, heard), group)
            if r is None or r["probs"] is None:
                out.append("...")
                break
            probs = r["probs"]
            if temperature > 0:
                q = probs ** (1.0 / temperature)
                word = self.cortex.outputs[group[self.rng.choice(len(q), p=q / q.sum())]].split(":", 1)[1]
            else:
                word = self.decode(group, probs)
            if word == self.END:
                break
            out.append(word)
            heard.append(word)
        return " ".join(out)
