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
inputs matter. The hub only proposes: speaking it is a learned choice of the speaking brain
(recall, next to copy and own), credited by outcome.
"""

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
        """The word form the hub proposes, and how strongly (its share of the votes), or (None, 0)."""
        votes = self.votes(units, v)
        if not votes.size or votes.sum() <= 0:
            return None, 0.0
        k = int(np.argmax(votes))
        return self.lexicon[k], float(votes[k] / votes.sum())

    def learn(self, units, v, answer):
        k = self.word(answer)
        for u in set(units):
            self.sound.setdefault(u, {})
            self.sound[u][k] = self.sound[u].get(k, 0) + 1
            self.heard[u] = self.heard.get(u, 0) + 1
        if v is not None:
            self.area.learn(v, self.out[k], np.array(self.out))
            self.area.maybe_prune()
