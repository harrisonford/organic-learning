"""An ear in which words heard alone carve up longer speech (docs/EAR_SEGMENTATION.md step 4; docs/EAR_CARVING.md).

The first ear cuts where its byte-by-byte uncertainty jumps. It breaks rare words into pieces ("ro|ck|et"), fuses
frequent neighbours ("a cat"), and hears "cat", "cat?" and "a cat" as unrelated units. This ear keeps those cuts
as a first hearing and adds what infants add (Brent & Siskind 2001; Bortfeld et al. 2005; Peters 1983):

  - **A stretch of speech heard alone, between two silences, is a candidate word** (Bloomfield's free form). An
    adult's one-word answer "dog", or "tulip" in an echo game, is heard alone.
  - **A known free form pops out when it is heard again inside longer speech**, joining pieces or splitting a
    fused unit, the longest first...
  - ...**if what it leaves over is acceptable** (the possible-word constraint, Norris et al. 1997): the remainder
    of each first-hearing unit it cuts through must itself be a unit the ear knows at least as well as the unit
    being cut, or both must be new (parts more familiar than the whole are a phrase; a whole more familiar than
    its parts is a word: Hay 2001). "cat" pops out of "a cat" ("a " is far more familiar than "a cat"); "no" does
    not pop out of "now " ("w " is not). A free form in the middle of one unit, leaving a remainder on both
    sides ("ouch" in "touch "), is a word inside a word and stays there. A frame whose slot keeps changing ("there a dog?", "there a cat?") comes
    apart by the same rule, because the frame is heard more often than any one filling of it.
  - **Only minimal free forms pop out**: a free form that itself contains a smaller free form in this way ("is
    there a dog?", heard alone a hundred times) is a phrase, and is heard through its parts.

No thresholds: familiarity is the ear's own count of each unit. Use it or lose it when the room is full.
"""
from .ear import Ear


class CarvingEar(Ear):
    max_alone = 60000  # room for stretches heard alone; reaching it forgets the tenth heard least and longest ago

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.first = {}  # unit of the first hearing -> times it was cut that way (whether or not it was then carved)
        self.alone = {}  # stretch -> times heard alone (between two silences)
        self.alone_last = {}
        self._minimal = {}  # remembered until the next sleep of this cache (cleared as experience grows)
        self.longest = 1

    def _first(self, data):
        """The first ear's cuts, as (start, end) spans."""
        spans, start, prev_h = [], 0, None
        for i in range(len(data)):
            h = self.surprise(data[:i])
            if i > 0 and (h - prev_h > self.rise or h >= self.unknown):
                spans.append((start, i))
                start = i
            prev_h = h
        spans.append((start, len(data)))
        return spans

    def _acceptable(self, data, spans, i, j):
        """May the stretch data[i:j] be heard as one thing here? What it leaves of the units it cuts must be familiar."""
        for a, b in spans:
            if a < i and j < b:  # it would leave a remainder on both sides of one unit: a word inside a word ("ouch" in "touch ")
                return False
            for rest in ((data[a:i],) if a < i < b else ()) + ((data[j:b],) if a < j < b else ()):
                # both as the first hearing cuts them: evidence that does not come from the carving itself
                part, whole = self.first.get(rest, 0), self.first.get(data[a:b], 0)
                # more familiar than the unit it comes out of; or both new (what is left beside a known word is a new candidate)
                if not (part > whole or part == whole == 0):
                    return False
        return True

    def _found(self, data, spans, whole=True):
        """Free forms in data that may pop out, as (start, end), the longest first; `whole=False` leaves out data itself."""
        L, out = len(data), []
        top = min(L, self.longest)
        for i in range(L):
            for j in range(min(L, i + top), i, -1):
                if (j - i < L or whole) and data[i:j] in self.alone and self._acceptable(data, spans, i, j) and self.minimal(data[i:j]):
                    out.append((i, j))
        out.sort(key=lambda s: (s[0] - s[1], -self.alone[data[s[0] : s[1]]]))
        return out

    def minimal(self, form):
        """A free form is minimal (a word, not a phrase) if no smaller free form pops out of it."""
        if form not in self._minimal:
            self._minimal[form] = False  # while asking (a form cannot be analysed through itself)
            self._minimal[form] = not self._found(form, self._first(form), whole=False)
        return self._minimal[form]

    def units(self, text, learn=True):
        data = text.encode("utf-8")
        if not data:
            return []
        spans = self._first(data)
        cuts = {b for _, b in spans}
        taken = []
        for i, j in self._found(data, spans):
            if all(j <= a or i >= b for a, b in taken):
                taken.append((i, j))
                cuts -= set(range(i + 1, j))
                cuts |= {i, j}
        cuts.discard(0)
        edges = [0] + sorted(cuts)
        out = [data[a:b] for a, b in zip(edges[:-1], edges[1:])]
        if learn:
            self._learn(data)
            self.utterances += 1
            for u in set(out):
                self.heard[u] = self.heard.get(u, 0) + 1
            for u in {data[a:b] for a, b in spans}:
                self.first[u] = self.first.get(u, 0) + 1
            self.alone[data] = self.alone.get(data, 0) + 1
            self.alone_last[data] = self.utterances
            self.longest = max(self.longest, len(data))
            if self.utterances % 500 == 0:
                self._minimal = {}  # what is a word and what a phrase is judged again as experience grows
            if len(self.alone) > self.max_alone:
                self._forget()
        return out

    def _forget(self, fraction=0.1):
        value = {u: n / (1.0 + (self.utterances - self.alone_last.get(u, 0)) / 30000.0) for u, n in self.alone.items()}
        cut = sorted(value.values())[int(fraction * len(value))]
        for u in [u for u, v in value.items() if v <= cut]:
            del self.alone[u]
            self.alone_last.pop(u, None)
        self.longest = max((len(u) for u in self.alone), default=1)


class EdgeEar(CarvingEar):
    """The ear learns which sounds mark the edges of the words it knows, and uses them on speech it does not know.

    Infants first cut speech by its statistics and by the words they know; by nine months they have learned from
    those words what the edges of words sound like in their language, and cut new speech by that (Jusczyk, Houston &
    Newsome 1999; Mattys & Jusczyk 2001; Thiessen & Saffran 2003). Here the teacher is the carving above: each
    time a known free form pops out of longer speech, the positions at its two ends are word edges and the
    positions inside it are not. For every byte the ear counts how often it stood just left, and just right, of
    an edge and of an inside position. Hearing: a position is a cut when, from the bytes on its two sides, an edge
    is more likely than not (the two pieces of evidence multiplied, against how common edges are); where neither
    byte has ever been seen beside a known word, the first ear's cut stands. A unit made only of bytes that are
    more often at the edge of known words than inside them (what is left between two words) is a pause or a mark,
    not a word: it stays with the unit before it.
    In this text world the ear ends up treating the space and the marks as it would a pause; nobody told it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import numpy as np
        self.edge = np.zeros((2, 256))  # [left of the position, right of it][byte]: times at a word edge
        self.inside = np.zeros((2, 256))  # ... times inside a known word

    marks_attach = True

    def _pops(self, data, spans):
        taken = []
        for i, j in self._found(data, spans):
            if all(j <= a or i >= b for a, b in taken):
                taken.append((i, j))
        return taken

    def units(self, text, learn=True):
        data = text.encode("utf-8")
        if not data:
            return []
        L = len(data)
        spans = self._first(data)
        first_cuts = {b for _, b in spans}
        E, I = self.edge, self.inside
        prior = (E.sum() + 1.0) / (I.sum() + 1.0)
        cuts = []
        for p in range(1, L):
            l, r = data[p - 1], data[p]
            seen_l, seen_r = E[0, l] + I[0, l], E[1, r] + I[1, r]
            if seen_l + seen_r == 0:
                if p in first_cuts:
                    cuts.append(p)
                continue
            odds = 1.0 / prior  # two pieces of evidence, each an odds, share one prior
            if seen_l:
                odds *= (E[0, l] + 0.5) / (I[0, l] + 0.5)
            if seen_r:
                odds *= (E[1, r] + 0.5) / (I[1, r] + 0.5)
            if not (seen_l and seen_r):
                odds *= prior  # only one side known: its odds alone
            if odds > 1.0:
                cuts.append(p)
        if self.marks_attach:  # a byte heard more often at the edge of known words than inside them is a pause or a mark
            pause = lambda u: all(E[0, b] + E[1, b] > I[0, b] + I[1, b] for b in u)  # noqa: E731
        else:  # only bytes that are an edge on both sides (a "?" then stays a unit of its own)
            pause = lambda u: all(E[0, b] > I[0, b] and E[1, b] > I[1, b] for b in u)  # noqa: E731
        out = []
        for a, b in zip([0] + cuts, cuts + [L]):
            u = data[a:b]
            if out and pause(u):
                out[-1] += u
            else:
                out.append(u)
        if len(out) > 1 and pause(out[0]):
            out[1] = out[0] + out[1]
            out = out[1:]
        if learn:
            for i, j in self._pops(data, spans):  # the teacher: a known word heard inside longer speech
                if j - i == L:
                    continue
                for p, row in [(i, E), (j, E)] + [(p, I) for p in range(i + 1, j)]:
                    if 0 < p < L:
                        row[0, data[p - 1]] += 1
                        row[1, data[p]] += 1
            self._learn(data)
            self.utterances += 1
            for u in set(out):
                self.heard[u] = self.heard.get(u, 0) + 1
            for u in {data[a:b] for a, b in spans}:
                self.first[u] = self.first.get(u, 0) + 1
            self.alone[data] = self.alone.get(data, 0) + 1
            self.alone_last[data] = self.utterances
            self.longest = max(self.longest, len(data))
            if self.utterances % 500 == 0:
                self._minimal = {}
            if len(self.alone) > self.max_alone:
                self._forget()
        return out
