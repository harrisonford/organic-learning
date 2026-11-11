"""Words: unit cells that learn to predict when they will be heard (docs/AREAS.md).

A word form here is one of the ear's own units (the chunks the ear discovered), not an answer
string. Each unit that has been heard in a reply or a comment has a cell. The cell fires when
its unit is heard; its dendrites learn to predict that firing from what was there just before,
each with the same local rule (dendritic prediction of somatic firing, Urbanczik & Senn 2014;
for cues it is the Rescorla-Wagner rule of animal learning):

    change of a synapse = the cell's plasticity x (fired - what this dendrite expected) x input activity

  content dendrite   inputs: the visual cortex's neurons (each passing on its deviation from its own
                     average), the ear's unit cells heard in the utterance (cues, wherever they were),
                     and the cell's own baseline. It predicts whether the unit will be in the reply.
                     Inputs compete to explain: a unit heard before every reply ("what", "is") ends
                     up explaining nothing, the picture or the informative cue explains the word.
  context dendrite   input: the ear's order-aware echo of the exchange so far (what was heard, the
                     silence that ended it, what has been said since). It predicts the unit that
                     comes next: which kind of reply this kind of utterance gets, and in what order.

Plasticity has two local factors and no schedule or clock:
  the cell's own recent error    a new cell is fully plastic, one whose predictions come true stops
                                 changing, one that starts failing becomes plastic again;
  the input's consolidation      an input neuron's synapses stiffen with its own use, 1 / (1 + use / 5),
                                 the law of the orchestrator's clusters. An input that is there in
                                 thousands of moments ("what", the baseline) changes slowly and so carries
                                 long-run averages; a new cue, or a visual neuron (each a small part of a
                                 picture), stays plastic. A new word on a new cue is learned at once (fast
                                 mapping); an uninformative input cannot chase the last reply.

To recall, a cell is active as far as its content dendrite is driven above its baseline (relative to
the most driven cell) AND its context dendrite expects it now (coincidence of the two compartments,
as in pyramidal cells). The most active unit is "heard" inwardly, joins the echo, and the next is
predicted, until the silence cell (which has only a context dendrite: silence needs no content) wins.

This replaces the hub (count tables, answer strings, kinds clustered in sleep): cue competition does
what the tables did, the context dendrite what kinds did, the echo of what was just said gives order.
An observation ("look, a dog!", nobody asked) is the same thing with nothing heard before it.
"""

import numpy as np

SILENCE = b""


class Words:
    max_reply = 8  # only stops a runaway loop
    maturation = 5.0  # an input's synapses consolidate with its use (the orchestrator's value)
    memory = 0.05  # how fast a cell's sense of its own recent error follows its errors (0.02-0.1 give the same result)

    imagery = False  # heard units bring back what was seen with them (docs/CONVERSATION.md, B); off in brains made before it

    sound = False  # the content dendrite also hears the sound of what was said, not only which units it was cut into

    def __init__(self, visual_dim, ear, max_units=None, max_cues=None, imagery=False, sound=False):
        self.ear, self.visual_dim = ear, visual_dim
        self.imagery, self.sound = imagery, sound
        # content dendrite: synapses from the ear's combination-sensitive neurons (the sound of everything just heard)
        self.Ws = np.zeros((64, ear.width), np.float32) if sound else None
        self.use_s = np.zeros(ear.width, np.float32)
        # with sound, the feedback onto the visual neurons comes from those neurons too (what a sound predicts will be seen)
        self.Gs = np.zeros((ear.width, visual_dim), np.float32) if sound and imagery else None
        self.use_gs = np.zeros(ear.width, np.float32)
        # feedback synapses from the ear's unit cells onto the visual neurons: what each heard unit predicts will be seen
        self.G = np.zeros((64, visual_dim), np.float32)
        self.use_g = np.zeros(64, np.float32)
        # content dendrite: synapses that read the visual neurons' bursts (driven from below and from above at once)
        self.Wb = np.zeros((64, visual_dim), np.float32) if imagery else None
        self.use_b = np.zeros(visual_dim, np.float32)
        self.max_units, self.max_cues = max_units, max_cues
        self.units, self.index = [SILENCE], {SILENCE: 0}
        self.cues, self.cue_index = [], {}
        self.Wv = np.zeros((64, visual_dim), np.float32)  # content dendrite: visual synapses
        self.Wc = np.zeros((64, 64), np.float32)  # content dendrite: synapses from the ear's unit cells (cues)
        self.base = np.zeros(64, np.float32)  # the cell's own baseline
        self.Wa = np.zeros((64, ear.width), np.float32)  # context dendrite: synapses from the echo
        self.L = np.zeros((64, 64), np.float32)  # lateral links: what the context was expecting when this cell's unit was heard
        self.err_c = np.ones(64, np.float32)  # each compartment's own recent error = its plasticity
        self.err_a = np.ones(64, np.float32)
        self.heard = np.zeros(64, np.int64)
        self.last = np.zeros(64, np.int64)
        self.cue_last = np.zeros(64, np.int64)
        self.mean_v, self.seen = np.zeros(visual_dim, np.float32), 0  # each visual neuron's own average
        self.use_v = np.zeros(visual_dim, np.float32)  # each input's accumulated activity (squared): its consolidation
        self.use_c = np.zeros(64, np.float32)
        self.use_a = np.zeros(ear.width, np.float32)
        self.born = np.zeros(64, np.int64)
        self.moments = 0

    # ------------------------------------------------------------------ cells

    def _grow(self, n_units, n_cues):
        if n_units > len(self.base):
            cap = max(n_units, 2 * len(self.base))
            pad = lambda a: np.concatenate([a, np.zeros((cap - a.shape[0],) + a.shape[1:], a.dtype)])  # noqa: E731
            self.Wv, self.Wc, self.Wa, self.base = pad(self.Wv), pad(self.Wc), pad(self.Wa), pad(self.base)
            if getattr(self, "Wb", None) is not None:
                self.Wb = pad(self.Wb)
            if getattr(self, "Ws", None) is not None:
                self.Ws = pad(self.Ws)
            self.L = np.concatenate([pad(self.L), np.zeros((cap, cap - self.L.shape[1]), np.float32)], axis=1)
            self.heard, self.last, self.born = pad(self.heard), pad(self.last), pad(self.born)
            for name in ("err_c", "err_a"):
                a = getattr(self, name)
                setattr(self, name, np.concatenate([a, np.ones(cap - len(a), np.float32)]))
        if n_cues > self.Wc.shape[1]:
            cap = max(n_cues, 2 * self.Wc.shape[1])
            self.Wc = np.concatenate([self.Wc, np.zeros((self.Wc.shape[0], cap - self.Wc.shape[1]), np.float32)], axis=1)
            self.cue_last = np.concatenate([self.cue_last, np.zeros(cap - len(self.cue_last), np.int64)])
            self.use_c = np.concatenate([self.use_c, np.zeros(cap - len(self.use_c), np.float32)])
            if self.imagery and self.Gs is None:
                self.G = np.concatenate([self.G, np.zeros((cap - self.G.shape[0], self.visual_dim), np.float32)])
                self.use_g = np.concatenate([self.use_g, np.zeros(cap - len(self.use_g), np.float32)])

    def cell(self, u):
        if u not in self.index:
            self._grow(len(self.units) + 1, 0)
            self.index[u] = len(self.units)
            self.born[len(self.units)] = self.moments
            self.units.append(u)
        return self.index[u]

    def _cue(self, u, grow):
        if u not in self.cue_index:
            if not grow:
                return None
            self._grow(0, len(self.cues) + 1)
            self.cue_index[u] = len(self.cues)
            self.cues.append(u)
        return self.cue_index[u]

    # ------------------------------------------------------------------ inputs

    def _visual(self, v, learn=False):
        """What the visual neurons pass on: each one's deviation from its own average activity."""
        if v is None:
            return None
        if learn:
            self.seen += 1
            self.mean_v += (v - self.mean_v) / min(self.seen, 1000)
        x = v - self.mean_v
        return x / (np.linalg.norm(x) + 1e-8)

    def _heard_sound(self, units):
        """What the ear's combination-sensitive neurons pass on about a stretch of speech (None: nothing heard)."""
        if not self.sound or not len(units):
            return None
        return self.ear.combinations(b"".join(units))

    def _sight(self, v, cues, learn=False, snd=None):
        """(x, b): what the visual neurons pass on, and their bursts. The heard units reach the visual neurons from
        above through their feedback synapses (what each has come to predict will be seen). Eyes open, a neuron
        driven from below and from above at once bursts (coincidence of its two compartments, as in the word
        cells): b says where what is seen is what the words were about. Eyes closed, the feedback alone is a faint
        picture in the mind's eye, as strong as the words' predictions are: that is passed on instead of sight."""
        x = self._visual(v, learn)
        if not self.imagery or (snd is None and not len(cues)):
            return x, None
        above = snd @ self.Gs if getattr(self, "Gs", None) is not None else self.G[cues].sum(axis=0)
        if x is None:
            return (above if above.any() else None), None
        return x, np.sqrt(len(x)) * np.maximum(x, 0) * np.maximum(above, 0)

    def _echo(self, heard, said):
        return self.ear.question(list(heard) + [SILENCE] + list(said), adapt=False)

    def _content(self, x, cues, b=None, snd=None):
        n = len(self.units)
        o = np.zeros(n, np.float32)
        if snd is not None:
            o += self.Ws[:n] @ snd
        if x is not None:
            o += self.Wv[:n] @ x
        if b is not None:
            o += self.Wb[:n] @ b
        if len(cues):
            o += self.Wc[:n][:, cues].sum(axis=1)
        return o

    # ------------------------------------------------------------------ recall

    def expect(self, heard, v, said=()):
        """(activity, content drive above baseline, context expectation) of every unit cell, for the next unit."""
        n = len(self.units)
        cues = [c for c in (self._cue(u, False) for u in set(heard)) if c is not None]
        snd = self._heard_sound(heard)
        x, b = self._sight(v, cues, snd=snd)
        drive = np.maximum(self._content(x, cues, b, snd), 0)
        ctx = np.maximum(self.Wa[:n] @ self._echo(heard, said), 0)
        if self.lateral:
            ctx = self._spread(ctx)
        top = drive[1:].max() if n > 1 else 0.0
        rel = drive / top if top > 0 else np.zeros(n, np.float32)
        rel[0] = 1.0  # the silence cell has only a context dendrite: silence needs no content
        act = rel * ctx
        for u in said:  # a cell that has just fired is refractory while the reply lasts
            act[self.index[u]] = 0.0
        return act, drive, ctx

    lateral = True

    def _spread(self, ctx):
        """Lateral links between unit cells: a cell is expected as far as the units now expected are the ones that
        were expected whenever it was heard (its company; each cell's links scaled to its strongest). The context
        cannot know which name will come, only that a name will: units that replace each other share expectation."""
        n = len(ctx)
        L = self.L[:n, :n]
        top = L.max(axis=1)
        return np.where(top > 0, (L @ ctx) / (top + 1e-8), ctx)

    def recall(self, heard, v):
        """The reply the cells expect, as text, and how sure (the first unit: content drive x context expectation)."""
        said, conf = [], 0.0
        for i in range(self.max_reply):
            act, drive, ctx = self.expect(heard, v, said)
            if act.max() <= 0:
                break
            k = int(np.argmax(act))
            if i == 0:
                conf = float(min(drive[k], 1.0) * min(ctx[k], 1.0)) if k else 0.0
            if k == 0:
                break
            said.append(self.units[k])
        text = b"".join(said).decode("utf-8", "replace").strip()
        return (text or None), conf

    # ------------------------------------------------------------------ learning

    def learn(self, heard, v, reply_units):
        """An utterance (reply_units) was heard after `heard` (a question, or nothing: an unasked comment), while
        seeing v (or nothing). Every cell's dendrites compare what they expected with whether the cell fired."""
        self.moments += 1
        targets = [self.cell(u) for u in reply_units]
        cues = [self._cue(u, True) for u in set(heard)]
        n = len(self.units)
        self.cue_last[cues] = self.moments
        self.heard[targets] += 1
        self.last[targets] = self.moments
        # content dendrite: will the unit be in the reply?
        snd = self._heard_sound(heard)
        x, b = self._sight(v, cues, learn=True, snd=snd)
        fired = np.zeros(n, np.float32)
        fired[targets] = 1.0
        e = fired - (self._content(x, cues, b, snd) + self.base[:n])
        e[0] = 0.0  # silence has no content dendrite
        step = self.err_c[:n] * e / (1.0 + (x is not None) + (b is not None) + (snd is not None) + len(cues))  # the inputs share the correction
        if snd is not None:
            self.Ws[:n] += step[:, None] * (snd / (1.0 + self.use_s / self.maturation))[None, :]
            self.use_s += snd * snd
        self._own_error(self.err_c, fired, e)
        self.base[:n] += step / (1.0 + (self.moments - self.born[:n]) / self.maturation)  # the baseline: an input that is always on
        if x is not None:
            self.Wv[:n] += step[:, None] * (x / (1.0 + self.use_v / self.maturation))[None, :]
            self.use_v += x * x
        if b is not None:
            self.Wb[:n] += step[:, None] * (b / (1.0 + self.use_b / self.maturation))[None, :]
            self.use_b += b * b
        if cues:
            self.Wc[:n][:, cues] += step[:, None] / (1.0 + self.use_c[cues] / self.maturation)[None, :]
            self.use_c[cues] += 1.0
        if self.imagery and v is not None and self.Gs is not None:
            # with sound: the sound of everything heard while something is seen (asked and answered) predicts the sight
            both = self._heard_sound(list(heard) + list(reply_units))
            self.Gs += np.outer(both / (1.0 + self.use_gs / self.maturation), x - both @ self.Gs)
            self.use_gs += both * both
        elif self.imagery and v is not None:
            # feedback synapses: every unit heard while something is seen (asked, answered or said unasked) learns to
            # predict the sight; the units share the error, and each consolidates with its own use (the same law)
            told = cues + [self._cue(u, True) for u in set(reply_units) - set(heard)]
            self.cue_last[told] = self.moments
            miss = x - self.G[told].sum(axis=0)
            self.G[told] += (miss[None, :] / len(told)) / (1.0 + self.use_g[told] / self.maturation)[:, None]
            self.use_g[told] += 1.0
        # context dendrite: which unit comes next, step by step, and the silence after the last
        for i, t in enumerate(targets + [0]):
            echo = self._echo(heard, reply_units[:i])
            fired = np.zeros(n, np.float32)
            fired[t] = 1.0
            o = self.Wa[:n] @ echo
            g = np.maximum(o, 0)
            if g.sum() > 0:  # Hebbian: the cell that fired x the cells that were expected in its place
                self.L[t, :n] += g / g.sum()
            e = fired - o
            self.Wa[:n] += (self.err_a[:n] * e)[:, None] * (echo / (1.0 + self.use_a / self.maturation))[None, :]
            self.use_a += echo * echo
            self._own_error(self.err_a, fired, e)
        if self.max_units and n > self.max_units:
            self._forget_units()
        if self.max_cues and len(self.cues) > self.max_cues:
            self._forget_cues()

    def _own_error(self, err, fired, e):
        """A compartment's sense of its own recent error moves only when the cell took part (it fired, or the
        dendrite expected it to): a rarely heard word stays as plastic as its last hearing left it."""
        n = len(fired)
        took_part = np.clip(np.maximum(fired, np.abs(fired - e)), 0.0, 1.0)
        err[:n] += self.memory * took_part * (np.abs(e) - err[:n])

    # ------------------------------------------------------------------ a skull of fixed size

    def _forget_units(self, fraction=0.1):
        """Use it or lose it: the least heard, longest unheard unit cells go (never silence)."""
        n = len(self.units)
        value = self.heard[:n] / (1.0 + (self.moments - self.last[:n]) / 30000.0)
        value[0] = np.inf
        drop = set(np.argsort(value)[: max(1, int(fraction * n))].tolist())
        keep = [i for i in range(n) if i not in drop]
        for name in ("Wv", "Wc", "Wa", "base", "err_c", "err_a", "heard", "last", "born", "L") + (("Wb",) if getattr(self, "Wb", None) is not None else ()) + (("Ws",) if getattr(self, "Ws", None) is not None else ()):
            a = getattr(self, name)
            a[: len(keep)] = a[keep]
            a[len(keep) : n] = 1.0 if name.startswith("err") else 0
        self.L[:, : len(keep)] = self.L[:, keep]
        self.L[:, len(keep) : n] = 0
        self.units = [self.units[i] for i in keep]
        self.index = {u: i for i, u in enumerate(self.units)}

    def _forget_cues(self, fraction=0.1):
        n = len(self.cues)
        drop = set(np.argsort(self.cue_last[:n])[: max(1, int(fraction * n))].tolist())
        keep = [i for i in range(n) if i not in drop]
        self.Wc[:, : len(keep)] = self.Wc[:, keep]
        self.Wc[:, len(keep) : n] = 0
        for a in (self.cue_last, self.use_c) + ((self.G, self.use_g) if self.imagery and getattr(self, "Gs", None) is None else ()):
            a[: len(keep)] = a[keep]
            a[len(keep) : n] = 0
        self.cues = [self.cues[i] for i in keep]
        self.cue_index = {u: i for i, u in enumerate(self.cues)}

    def cells(self):
        return {"unit cells": len(self.units), "cues": len(self.cues), "own error (content, context)":
                (round(float(self.err_c[: len(self.units)].mean()), 3), round(float(self.err_a[: len(self.units)].mean()), 3))}
