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

  replay  (2026-11-18, docs/RESEARCH_MEMORY.md; off unless `fade` is set) at rest, stored moments come back by
          themselves, the strong ones more often, and the word cells live them again (hippocampal replay teaching
          the cortex: McClelland et al. 1995). A moment that came back is held as strongly as the word cells were
          surprised by it this time: what the slow synapses hold by now fades from the store, what they have lost
          again stays. Between two returns a moment's strength fades with time, and when the store is full the
          weakest moment goes, not the oldest. So what is kept is what was not learned yet, however old.

Nothing here says what to remember or what a moment means. Set by hand: how many moments it holds, how steep the
comparison is, and (with replay) how long an unvisited moment lasts.
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
        self.asked = [None] * capacity  # the units heard before the reply (to live the moment again)
        self.when = np.zeros(capacity, np.int64)  # the store's clock when the moment's strength was last set
        self.clock = 0

    fade = None  # how many moments an unvisited memory lasts (its strength falls by e); None: no fading, the oldest goes
    last = None  # where the last moment was written (None: it was not)
    asked = when = views = mark = None  # a store pickled before 2026-11-18 has none of these
    clock = 0
    mark_now = -1  # for the experimenter only: a label for the moments written now (the stage they are from)
    returns = None  # for the experimenter only: {label: [how many came back, their summed surprise]}
    tag = None  # None: any stored moment may come back at rest (every brain before 2026-11-24). An array: the reward
    # that came at each stored moment while awake; only such moments come back, the more reward the more often
    # (awake ripples at reward mark what sleep replays: docs/VISION.md, 2026-11-24 night)

    def _new(self):
        if self.asked is None:
            self.asked, self.when = [None] * self.capacity, np.zeros(self.capacity, np.int64)  # (None: written before 2026-11-18, cannot come back)
        if self.views is None:
            self.views, self.mark = [None] * self.capacity, np.full(self.capacity, -1, np.int8)

    def grow(self, capacity):
        """A larger store: what it holds stays."""
        if capacity <= self.capacity:
            return
        self._new()
        more = capacity - self.capacity
        self.seen = np.concatenate([np.asarray(self.seen, np.float32), np.zeros((more, self.seen.shape[1]), np.float32)])
        self.heard = np.concatenate([self.heard, np.zeros((more, self.heard.shape[1]), np.float32)])
        self.eyes = np.concatenate([self.eyes, np.zeros(more, bool)])
        self.strength = np.concatenate([self.strength, np.zeros(more, np.float32)])
        self.when = np.concatenate([self.when, np.zeros(more, np.int64)])
        self.mark = np.concatenate([self.mark, np.full(more, -1, np.int8)])
        if self.tag is not None:
            self.tag = np.concatenate([self.tag, np.zeros(more, np.float32)])
        self.reply, self.asked, self.views = self.reply + [()] * more, self.asked + [None] * more, self.views + [None] * more
        self.at, self.capacity = self.n % capacity, capacity

    def held(self):
        """How strongly each stored moment is held now."""
        s = self.strength[: self.n]
        return s if self.fade is None else s * np.exp(-(self.clock - self.when[: self.n]) / self.fade)

    def write(self, seen, heard, reply_units, unexpected, asked=None, views=None):
        """One moment, at once. `unexpected` in [0, 1]: how far the word cells were from expecting the reply.
        `asked`: the units heard before the reply; `views`: what the eye fixated (both to live the moment again;
        a moment written without `asked` cannot come back)."""
        self.clock += 1
        self.last = None
        if unexpected <= 0 or not reply_units:
            return
        self._new()
        if self.fade is not None and self.n == self.capacity:  # full: the weakest goes, if the new moment is stronger
            k = int(np.argmin(self.held()))
            if self.held()[k] >= unexpected:
                return
        else:
            k = self.at
            self.at = (self.at + 1) % self.capacity
        self.eyes[k] = seen is not None
        self.seen[k] = seen if seen is not None else 0
        self.heard[k] = heard if heard is not None else 0
        self.strength[k], self.reply[k], self.when[k] = unexpected, tuple(reply_units), self.clock
        self.asked[k] = None if asked is None else tuple(asked)
        self.views[k] = None if views is None or not len(views) else np.asarray(views, np.float16)
        self.mark[k] = self.mark_now
        if self.tag is not None:
            self.tag[k] = 0.0  # no reward has come at this moment yet
        self.n = min(self.n + 1, self.capacity)
        self.writes += 1
        self.last = k

    def come_back(self, rng, count):
        """The stored moments that return at rest: the more strongly held, the more often."""
        if not self.n or self.asked is None:
            return []
        p = self.held().astype(np.float64) * np.array([a is not None for a in self.asked[: self.n]])
        if self.tag is not None:
            p = p * self.tag[: self.n]
        if p.sum() <= 0:
            return []
        return [int(k) for k in rng.choice(self.n, size=min(count, int((p > 0).sum())), replace=False, p=p / p.sum())]

    def rewarded(self, reward):
        """Reward came at the moment just written (it is marked by it; a moment that was not kept is not)."""
        if self.tag is not None and self.last is not None:
            self.tag[self.last] = max(float(reward), 0.0)

    def lived_again(self, k, unexpected):
        """A moment that came back is held as far as the word cells were surprised by it this time."""
        self.strength[k], self.when[k] = unexpected, self.clock
        if self.returns is not None:
            r = self.returns.setdefault(int(self.mark[k]), [0, 0.0])
            r[0], r[1] = r[0] + 1, r[1] + unexpected

    key = "whole"  # what a moment is kept under: "whole" (the whole view) or "looked" (what the eye was on; 2026-11-23)

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_kc", None)  # (the keys laid side by side: made again when needed)
        return d

    def _looked(self):
        """Every stored moment's keys, side by side, and whose each is: the windows the eye fixated in that moment,
        or the whole view when it fixated nothing."""
        stamp = (self.n, self.writes)
        if getattr(self, "_kc", None) is None or self._kc[0] != stamp:
            keys, owner = [], []
            for k in range(self.n):
                if not self.eyes[k]:
                    continue
                v = self.views[k] if self.views is not None and self.views[k] is not None else self.seen[k][None]
                keys.append(np.asarray(v, np.float32))
                owner += [k] * len(v)
            self._kc = (stamp, np.concatenate(keys) if keys else np.zeros((0, self.seen.shape[1]), np.float32), np.array(owner, int))
        return self._kc[1], self._kc[2]

    def read(self, seen, heard, looks=None):
        """{unit: drive} from the stored moments like this one (seen now, or heard now with the eyes closed).
        `looks`: what the eye fixates now. With key "looked" a moment is like this one as far as one of the windows
        the eye was on then is like one it is on now (the whole view standing in where nothing was fixated)."""
        if not self.n:
            return {}
        if seen is not None and self.key == "looked":
            now = np.asarray(looks, np.float32) if looks is not None and len(looks) else np.asarray(seen, np.float32)[None]
            keys, owner = self._looked()
            like = np.full(self.n, -1.0, np.float32)
            if len(keys):
                np.maximum.at(like, owner, (keys @ now.T).max(axis=1))
        elif seen is not None:
            like = np.where(self.eyes[: self.n], np.asarray(self.seen[: self.n], np.float32) @ seen, -1.0)
        elif heard is not None:
            like = np.where(self.eyes[: self.n], -1.0, self.heard[: self.n] @ heard)
        else:
            return {}
        answer = self.held() * np.exp(self.steep * (np.minimum(like, 1.0) - 1.0))
        out = {}
        for k in np.nonzero(answer > 0.01)[0]:
            for u in self.reply[k]:
                if answer[k] > out.get(u, 0.0):
                    out[u] = float(answer[k])
        self.reads += bool(out)
        return out
