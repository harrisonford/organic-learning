"""An ear whose known units carve up what it hears (docs/EAR_SEGMENTATION.md step 4; docs/EAR_LEXICON.md).

The first ear cuts where its byte-by-byte uncertainty jumps past two hand-set thresholds. It fuses frequent
phrases into one unit, breaks rare words into pieces, and hears "cat", "cat?" and "a cat" as unrelated units.
Infants do something else as well: a word they already know ("mommy", their own name) is heard as itself inside
a longer utterance, and what is left beside it becomes a candidate word (Bortfeld et al. 2005; Brent & Cartwright
1996; Brent & Siskind 2001 on words heard alone). Here:

  - The units the ear has heard are its lexicon, each as strong as the number of times it was heard.
  - An utterance is heard as the sequence of units that explains it most cheaply, in bits. A known unit costs the
    surprise of meeting it, -log2(times heard / all hearings). A stretch that is no known unit costs the surprise
    of meeting a new unit at all, -log2(kinds of unit / all hearings), plus the ear's own byte-by-byte surprise
    at its sounds, starting afresh at its first byte (the statistical learner the first ear already has).
  - Known units compete with each other and with "this is something new" through that one sum. Nothing else
    decides: no threshold on the entropy, no size limit, no list of words. A word heard alone (an adult's
    one-word answer) becomes a unit at once; later it carves itself out of longer speech and leaves its
    neighbours as new units. Before anything is known, a whole utterance is one unit (a formula, as in Peters 1983).
  - Use it or lose it: when the lexicon outgrows its room, the units heard least and longest ago go.
"""
import numpy as np

from .ear import Ear


class LexiconEar(Ear):
    max_lexicon = 60000  # room for units (a skull of fixed size); reaching it forgets a tenth

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.tokens = 0  # all hearings of all units
        self.last = {}  # when each unit was last heard (in utterances)

    def _info(self, context, byte):
        c = self._predict(context)
        return self.h_max if c is None else float(-np.log2((c[byte] + 0.01) / (c.sum() + 0.01 * 257)))

    def units(self, text, learn=True):
        data = text.encode("utf-8")
        L = len(data)
        if L == 0:
            return []
        types = len(self.heard)
        total = self.tokens + types
        new = -np.log2(types / total) if types else 0.0  # the surprise of meeting a unit never heard
        m = self.max_context
        # the ear's surprise at each byte given the bytes before it, and at the first bytes of a stretch that starts afresh
        full = np.array([self._info(data[max(0, j - m) : j], data[j]) for j in range(L)])
        cum = np.concatenate([[0.0], np.cumsum(full)])
        fresh = [[self._info(data[i : i + k], data[i + k]) for k in range(min(m, L - i))] for i in range(L)]
        best = np.full(L + 1, np.inf)
        back = np.zeros(L + 1, np.int64)
        best[0] = 0.0
        heard = self.heard
        for i in range(L):
            if not np.isfinite(best[i]):
                continue
            head, base = 0.0, best[i] + new
            for j in range(i + 1, L + 1):
                k = j - i - 1
                if k < m:
                    head += fresh[i][k]
                    cost = base + head
                else:
                    cost = base + head + cum[j] - cum[i + m]
                n = heard.get(data[i:j])
                if n:
                    known = best[i] - np.log2(n / total)
                    if known < cost:
                        cost = known
                if cost < best[j]:
                    best[j], back[j] = cost, i
        out, j = [], L
        while j > 0:
            out.append(data[back[j] : j])
            j = int(back[j])
        out.reverse()
        if learn:
            self._learn(data)
            self.utterances += 1
            for u in set(out):
                heard[u] = heard.get(u, 0) + 1
                self.last[u] = self.utterances
                self.tokens += 1
            if len(heard) > self.max_lexicon:
                self._forget()
        return out

    def _forget(self, fraction=0.1):
        """The least heard, longest unheard units go, and their sound codes with them."""
        value = sorted(heard_n / (1.0 + (self.utterances - self.last.get(u, 0)) / 30000.0) for u, heard_n in self.heard.items())
        cut = value[int(fraction * len(value))]
        for u in [u for u, n in self.heard.items() if n / (1.0 + (self.utterances - self.last.get(u, 0)) / 30000.0) <= cut]:
            self.tokens -= self.heard.pop(u)
            self.last.pop(u, None)
            self.codes.pop(u, None)
