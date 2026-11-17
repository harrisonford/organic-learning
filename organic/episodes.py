"""Episodes: a fast store of single moments (docs/RESEARCH_MEMORY.md, 2026-11-17).

The word cells are slow learners on purpose: each moment moves their synapses a little, so what they hold is what
many moments have in common. Measured on 2026-11-17: a name heard once with a new kind of thing is not there a
moment later, even on the very same picture. A brain has a second system for that (complementary learning systems,
McClelland, McNaughton & O'Reilly 1995): the hippocampus writes a moment at once, under what was seen and heard,
and gives it back when the same situation returns.

  write   a moment the word cells did not expect is written whole, once: what was seen (the cortex's summary as the
          word cells get it) or, eyes closed, what was heard (the ear's echo of the utterance), and the units of the
          reply that then came. How strongly: as far as the reply was unexpected (novelty gates the hippocampus).
          An expected moment is not written; the slow system has it already.
  read    what is seen (or heard) now is compared with every stored moment; a stored moment answers the more the
          closer it is, steeply (a moment, not a kind: an exponential of the likeness, as in dense associative
          memory, Krotov & Hopfield 2016). Its reply's units are passed to the word cells as content drive.
  forget  the store is small. When it is full the oldest moment goes (a child's fast-mapped word is gone in minutes
          unless it is heard again, Horst & Samuelson 2008; heard again, it is written again).

Nothing here says what to remember or what a moment means. Two numbers are set by hand: how many moments it holds,
and how steep the comparison is.
"""
import numpy as np


class Episodes:
    def __init__(self, visual_dim, heard_dim, capacity=2000, steep=12.0):
        self.capacity, self.steep = capacity, steep
        self.seen = np.zeros((capacity, visual_dim), np.float16)
        self.heard = np.zeros((capacity, heard_dim), np.float32)
        self.eyes = np.zeros(capacity, bool)  # was something seen in that moment?
        self.strength = np.zeros(capacity, np.float32)
        self.reply = [()] * capacity
        self.n = self.at = 0
        self.writes = self.reads = 0

    def write(self, seen, heard, reply_units, unexpected):
        """One moment, at once. `unexpected` in [0, 1]: how far the word cells were from expecting the reply."""
        if unexpected <= 0 or not reply_units:
            return
        k = self.at
        self.eyes[k] = seen is not None
        self.seen[k] = seen if seen is not None else 0
        self.heard[k] = heard if heard is not None else 0
        self.strength[k], self.reply[k] = unexpected, tuple(reply_units)
        self.at = (self.at + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)
        self.writes += 1

    def read(self, seen, heard):
        """{unit: drive} from the stored moments like this one (seen now, or heard now with the eyes closed)."""
        if not self.n:
            return {}
        if seen is not None:
            like = np.where(self.eyes[: self.n], self.seen[: self.n].astype(np.float32) @ seen, -1.0)
        elif heard is not None:
            like = np.where(self.eyes[: self.n], -1.0, self.heard[: self.n] @ heard)
        else:
            return {}
        answer = self.strength[: self.n] * np.exp(self.steep * (np.minimum(like, 1.0) - 1.0))
        out = {}
        for k in np.nonzero(answer > 0.01)[0]:
            for u in self.reply[k]:
                if answer[k] > out.get(u, 0.0):
                    out[u] = float(answer[k])
        self.reads += bool(out)
        return out
