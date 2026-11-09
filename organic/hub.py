"""The association hub: word forms linked to organ cells across situations (docs/PLASTICITY.md).

A cluster of the orchestrator stores a whole situation (the question's sound + the picture
+ the gaze ...), so the same picture asked in other words is a different situation, and the
name learned for it is not recalled (kid_s3: the same answer across three wordings 0-7%).
Children link words to things across situations (Smith & Yu 2008), in an association hub
fed by separate sensory areas (anterior temporal lobe, hippocampus).

- Visual area: a Cortex (the orchestrator's own neurons: growth by affinity, maturation,
  outstar links, pruning) that sees only the picture. Its cells link to lexical neurons.
- Sound cells: each unit the ear hears links to the lexical neurons it co-occurs with,
  normalised by how often the unit is heard (its links become P(word | unit)).
- Lexical neurons: one per word form the adults have said.

A unit heard with every word ("what", "look", "is") has flat links and favours nothing; a
picture's cells favour the words said when such things were seen. Nothing marks which
inputs matter.

Answer kinds (2026-11-08): additive links cannot do a conjunction ("a naming question AND this
picture -> its name; a colour question AND this picture -> its colour"), so at two words the
picture's links drifted to "yes", "two", "brown". Children group words by the contexts they
are heard in (Mintz 2003). Each lexical neuron's context is the distribution of question
units it answered; during sleep, kind cells regroup lexical neurons by familiarity (the
clusters' own rule). The question's units then select a kind, and the organs select the word
within it. Kinds found with no labels: names, yes/no, numbers, colours, family words.

The hub only proposes: speaking it is a learned choice of the speaking brain (recall, next
to copy and own), credited by outcome.
"""

import zlib

import numpy as np

from .cortex import Cortex


class Hub:
    def __init__(self, visual_dim, seed=0, max_cells=None):
        self.area = Cortex(dim=visual_dim, neurons_per_cluster=8, theta_familiar=0.95, associate=False, maturation=5.0, seed=seed)
        if max_cells:
            self.area.set_budget(max_cells, None)
        self.lexicon, self.out = [], []  # word forms and their output neurons in the visual area
        self.index = {}
        self.sound = {}  # heard unit -> {word index: co-occurrences}
        self.heard = {}  # heard unit -> moments it was heard with a word

    familiar = 0.95 ** 4  # kind familiarity: the orchestrator's per-neuron familiarity over a context (to iterate on)
    sleep_every = 2000

    context_width = 512  # a word's context as neural activity (sum of its heard units' fixed sparse patterns)

    def _unit_code(self, u):
        """A fixed random sparse pattern for a heard unit (like the ear's byte neurons)."""
        if not hasattr(self, "_codes"):
            self._codes = {}
        if u not in self._codes:
            rng = np.random.default_rng(zlib.crc32(u) ^ 1234)
            v = np.zeros(self.context_width, np.float32)
            v[rng.choice(self.context_width, 16, replace=False)] = rng.choice([-1.0, 1.0], 16)
            self._codes[u] = v
        return self._codes[u]

    def kinds_now(self):
        """Regroup lexical neurons into kinds by their contexts (sleep). The most heard words found kinds first."""
        K = len(self.lexicon)
        ctx = np.zeros((K, self.context_width), np.float32)
        for u, links in self.sound.items():
            code = self._unit_code(u)
            for k, c in links.items():
                ctx[k] += c * code
        heard = np.zeros(K)
        for links in self.sound.values():
            for k, c in links.items():
                heard[k] += c
        norm = ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-8)
        cells, kind = np.zeros((0, self.context_width), np.float32), np.zeros(K, np.int64)
        for k in np.argsort(-heard):
            sims = cells @ norm[k] / (np.linalg.norm(cells, axis=1) + 1e-8) if len(cells) else np.zeros(0)
            j = int(np.argmax(sims)) if len(sims) else -1
            if j < 0 or sims[j] < self.familiar:
                cells = np.vstack([cells, ctx[k][None]])
                j = len(cells) - 1
            else:
                cells[j] += ctx[k]
            kind[k] = j
        self.kind, self.n_kinds = kind, len(cells)

    def word(self, w):
        if w not in self.index:
            self.index[w] = len(self.lexicon)
            self.lexicon.append(w)
            self.out.append(self.area.output_neuron("word:" + w))
        return self.index[w]

    def votes(self, units, v):
        """Each organ casts one vote over the lexicon: the heard units (their link distributions,
        averaged) and the picture (the visual area's output). v is None when the eyes see nothing."""
        n = len(self.lexicon)
        votes = np.zeros(n)
        if not n:
            return votes
        known = [u for u in set(units) if u in self.sound]
        if known:
            for u in known:
                for k, c in self.sound[u].items():
                    votes[k] += c / self.heard[u] / len(known)
        if v is not None and self.area.C:
            r = self.area.forward(v, np.array(self.out))
            if r is not None and r["probs"] is not None:
                votes += r["probs"]
        return votes

    def recall(self, units, v):
        """The word form the hub proposes, and how strongly (its share of the votes), or (None, 0).
        Once kinds exist (after the first sleep), the question's units select a kind and the votes
        select the word within it."""
        votes = self.votes(units, v)
        if not votes.size or votes.sum() <= 0:
            return None, 0.0
        total = votes.sum()  # confidence is a share of all the votes, not of the chosen kind's (a kind of one is not certainty)
        kind = getattr(self, "kind", None)
        if kind is not None and len(kind) < len(votes):  # words learned since the last sleep: a kind of their own until then
            extra = len(votes) - len(kind)
            self.kind = kind = np.concatenate([kind, self.n_kinds + np.arange(extra)])
            self.n_kinds += extra
        if kind is not None:
            known = [u for u in set(units) if u in self.sound]
            if known:
                pk = np.zeros(self.n_kinds)
                for u in known:
                    for k, c in self.sound[u].items():
                        pk[kind[k]] += c / self.heard[u]
                votes = np.where(kind == int(np.argmax(pk)), votes, 0.0)
                if votes.sum() <= 0:
                    return None, 0.0
        k = int(np.argmax(votes))
        return self.lexicon[k], float(votes[k] / total)

    def learn(self, units, v, answer):
        k = self.word(answer)
        for u in set(units):
            self.sound.setdefault(u, {})
            self.sound[u][k] = self.sound[u].get(k, 0) + 1
            self.heard[u] = self.heard.get(u, 0) + 1
        if v is not None:
            self.area.learn(v, self.out[k], np.array(self.out))
            self.area.maybe_prune()
        self.moments = getattr(self, "moments", 0) + 1
        if self.moments % self.sleep_every == 0:
            self.kinds_now()  # sleep: lexical neurons regroup into kinds (a word learned since has a kind of its own until then)
