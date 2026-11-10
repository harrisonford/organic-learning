"""One tissue, many areas (docs/REVIEW_2026-11-10.md 4.1, 4.3; docs/AREAS.md).

The hub (organic/hub.py) worked, but it is an algorithm: count tables, a batch clustering of
"kinds", and lexical neurons that are the adults' whole answer strings. Here the same job is
done by areas of one tissue (organic.cortex.Cortex: growth by affinity, instar/outstar,
maturation, myelin), wired by organ, with ONE rule for all of them:

    cells that are active now learn to predict the unit that is heard next.

  visual area     cells grown over what the eyes send (the picture)
  cue area        cells grown over the sound of single heard units (wherever they were in the utterance)
  echo area       cells grown over the echo of everything heard so far in the exchange (order-aware,
                  the ear's own fading echo), including what has already been said of the reply

What they predict are the ear's own units (the chunks the ear discovered), one per step, and the
silence that ends a turn (the ear's empty unit). Nothing is told which units are words: a unit
is a word form because some cells predict it. There is no lexicon of answer strings.

To recall, the areas' predictions meet on the unit cells: the visual and cue areas drive them
(what goes with this picture and with these heard units, summed, each cell's vote weighted by
its myelin, i.e. by how often its predictions came true), and the echo area's prediction is
their gain (what can come next after this kind of utterance: the Cortex's own `gain`, priming
from another area). The strongest unit is "heard" inwardly, joins the echo, and the next is
predicted, until silence is. A reply is a sequence of the brain's own units.

What this replaces: the hub's count tables (cue area), its lexical strings (unit cells), answer
kinds and their sleep-time clustering (the echo area's expectation is the kind: a naming frame
expects names, a colour question colours), and the order of words in a reply (the echo of what
was just said predicts what follows).
"""

import numpy as np

from .cortex import Cortex

SILENCE = b""


class Tissue:
    max_units = 8  # a reply longer than this is cut (only stops a runaway loop)

    def __init__(self, visual_dim, ear, seed=0, max_cells=None, gain=True, lateral=True, cue=True, baseline=0.0, min_rate=0.0, silence="echo"):
        kw = dict(neurons_per_cluster=8, theta_familiar=0.95, associate=False, maturation=5.0)
        self.ear = ear
        self.areas = {"visual": Cortex(dim=visual_dim, seed=seed, **kw),
                      "cue": Cortex(dim=ear.width, seed=seed + 1, **kw),
                      "echo": Cortex(dim=ear.width, seed=seed + 2, **kw)}
        for a in self.areas.values():
            a.min_rate = min_rate  # a cell's expectations are its whole history, not its last hundred moments
        if max_cells:
            self.areas["visual"].set_budget(max_cells, None)
        self.silence = silence  # which areas learn to predict the silence after a reply: "echo" (the sequence) or "all"
        self.units, self.index = [], {}  # the unit cells every area's axons reach (the ear's units heard in replies)
        self.gain, self.lateral, self.cue, self.baseline = gain, lateral, cue, baseline
        self.company = {}  # unit cell -> what the echo area was expecting when it was heard (summed): the company it keeps
        self.moments = 0

    def cell(self, u):
        if u not in self.index:
            self.index[u] = len(self.units)
            self.units.append(u)
            for a in self.areas.values():
                a.output_neuron("hear:" + u.hex())
        return self.index[u]

    def _group(self):
        return np.arange(len(self.units))

    def _echo(self, heard, said):
        """The echo of the exchange so far: what was heard, the silence that ended it, what has been said since."""
        return self.ear.question(list(heard) + [SILENCE] + list(said), adapt=False)

    def _raw(self, area, s):
        r = self.areas[area].forward(s, self._group())
        return None if r is None or r["probs"] is None else r["raw"]

    def expect(self, heard, v, said=()):
        """Activity of the unit cells: what the brain expects to hear next. Returns (activity, drive, gain)."""
        n = len(self.units)
        drive = np.zeros(n)
        if v is not None:
            r = self._raw("visual", v)
            if r is not None:
                drive += r
        for u in set(heard) if self.cue else ():
            r = self._raw("cue", self.ear.sound(u))
            if r is not None:
                drive += r
        g = self._raw("echo", self._echo(heard, said))
        if g is None or g.sum() <= 0:
            gain = np.ones(n)
        else:
            g = g / g.sum()
            if self.lateral:  # expectation spreads to the units that keep the same company
                g = self._spread(g)
            gain = g + self.baseline / n  # a spontaneous rate spread over all cells
        if not self.gain:
            return drive + (g if g is not None else 0), drive, gain
        return drive * gain, drive, gain

    def _spread(self, g):
        """Lateral links between unit cells: a cell is excited by an expectation as far as it resembles the
        expectations that were there when the cell's own unit was heard (each cell's links scaled to sum to one)."""
        out = np.zeros(len(g))
        for k, row in self.company.items():
            n = min(len(row), len(g))
            tot = row.sum()
            if tot > 0:
                out[k] = float(row[:n] @ g[:n]) / tot
        return out / out.sum() if out.sum() > 0 else g

    def recall(self, heard, v):
        """The reply the areas expect, as text, and how sure (the first unit's share of the activity)."""
        if not self.units:
            return None, 0.0
        said, conf = [], 0.0
        for i in range(self.max_units):
            act, _, _ = self.expect(heard, v, said)
            if act.sum() <= 0:
                break
            k = int(np.argmax(act))
            if i == 0:
                conf = float(act[k] / act.sum())
            if self.units[k] == SILENCE:
                break
            said.append(self.units[k])
        text = b"".join(said).decode("utf-8", "replace").strip()
        return (text or None), conf

    def learn(self, heard, v, reply_units):
        """An exchange was heard (and seen): every area's active cells learn to predict each unit of the
        reply as it comes, and the silence after it."""
        group_before = len(self.units)
        targets = [self.cell(u) for u in list(reply_units) + [SILENCE]]
        group = self._group()
        cues = [self.ear.sound(u) for u in set(heard)]
        for i, t in enumerate(targets):
            content = i < len(reply_units) or self.silence == "all"  # silence says nothing about what is seen or was heard
            if v is not None and content:
                self.areas["visual"].learn(v, t, group)
            for s in cues if content else ():
                self.areas["cue"].learn(s, t, group)
            r = self.areas["echo"].learn(self._echo(heard, reply_units[:i]), t, group)["forward"]
            if r is not None and r["probs"] is not None and r["raw"].sum() > 0:
                # Hebbian, between the cell that fired (the unit heard) and the cells that were expected: its company
                row = self.company.get(t)
                if row is None or len(row) < len(group):
                    grown = np.zeros(max(len(group), 2 * (len(row) if row is not None else 32)))
                    if row is not None:
                        grown[: len(row)] = row
                    row = self.company[t] = grown
                row[: len(group)] += r["raw"] / r["raw"].sum()
        for a in self.areas.values():
            a.maybe_prune()
        self.moments += 1
        return len(self.units) - group_before

    def cells(self):
        return {k: int(a.C) for k, a in self.areas.items()}
