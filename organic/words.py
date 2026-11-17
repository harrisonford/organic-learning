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

    setpoint = False  # a cell stays silent below its own firing level (docs/CONVERSATION.md, proposal 2)
    sound = False  # the content dendrite also hears the sound of what was said, not only which units it was cut into

    match_bursts = False  # the match neurons count all-or-none bursts instead of adding up graded drive (2026-11-17)
    match = False  # the heard word's own cell, when it is also driven by what is seen, tells the others (docs/VISION.md)
    match_cells = 16

    # the content dendrite's plasticity follows the cell's own recent error (False: every correction at full size; the
    # error itself already makes a cell that is right stop changing; tried 2026-11-17, experiments/words_on_cortex.py)
    plastic_by_error = True
    episodes = None  # a fast store of single moments beside the slow synapses (organic/episodes.py); None: the cells alone
    gain = 1.0  # a probe's knob on the size of every correction (experiments/words_on_cortex.py); 1 in every brain

    branches = 1  # dendritic branches for what is seen (docs/RESEARCH_MEMORY.md, 2026-11-17); 1 in brains made before

    def __init__(self, visual_dim, ear, max_units=None, max_cues=None, imagery=False, sound=False, setpoint=False, match=False, branches=1):
        self.ear, self.visual_dim = ear, visual_dim
        # A cell's visual synapses sit on several dendritic branches. The cell is driven by the branch that what is
        # seen drives most, and only that branch learns; a look of its word that drives no branch as the cell is
        # usually driven when its word is heard takes a branch not yet used. So a new look of a known word (a bus in a
        # photograph after buses in drawings) need not overwrite the old one (branch-specific plasticity, Cichon &
        # Gan 2015; dendritic subunits, Poirazi & Mel 2001).
        self.branches = branches
        self.Wx = np.zeros((branches - 1, 64, visual_dim), np.float32) if branches > 1 else None
        self.branch_used = np.zeros((branches, 64), np.int32)
        self.setpoint = setpoint
        self.match = False
        # each cell's own two levels of content drive: when its unit was then heard, and when it was the most driven
        # cell but its unit was not heard. Between them lies the level below which it has nothing to say.
        self.m_on, self.m_off = np.zeros(64, np.float32), np.zeros(64, np.float32)
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
        if match:
            self.enable_match()

    def enable_match(self):
        """Match neurons: a small population that hears how strongly, and in how many of the eye's views, the cells of
        the words just heard are driven by what is seen. Every neuron mixes the two (the strongest view, all views
        together) in its own way and has its own threshold; what their firing means is learned by the word cells'
        content dendrites like any other input. Can be given to a brain that has lived without them."""
        rng = np.random.default_rng(getattr(self.ear, "seed", 0) + 31)
        self.m_mix = rng.normal(0, 1, (self.match_cells, 2)).astype(np.float32)
        self.m_thr = rng.uniform(-1, 1, self.match_cells).astype(np.float32)
        self.m_mean, self.m_var, self.m_seen = np.zeros(2, np.float32), np.ones(2, np.float32), 0
        self.Wm = np.zeros((len(self.base), self.match_cells), np.float32)
        self.use_m = np.zeros(self.match_cells, np.float32)
        self.v_on = np.zeros(len(self.base), np.float32)  # each cell's usual drive from what is seen when its word is heard
        self.match = True

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
            if self.match:
                self.Wm = pad(self.Wm)
                if getattr(self, "v_on", None) is not None:
                    self.v_on = pad(self.v_on)
            if self.branches > 1:
                grown = cap - self.Wx.shape[1]
                self.Wx = np.concatenate([self.Wx, np.zeros((self.branches - 1, grown, self.visual_dim), np.float32)], axis=1)
                self.branch_used = np.concatenate([self.branch_used, np.zeros((self.branches, grown), np.int32)], axis=1)
            self.L = np.concatenate([pad(self.L), np.zeros((cap, cap - self.L.shape[1]), np.float32)], axis=1)
            self.heard, self.last, self.born = pad(self.heard), pad(self.last), pad(self.born)
            if getattr(self, "m_on", None) is not None:
                self.m_on, self.m_off = pad(self.m_on), pad(self.m_off)
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

    use_core = False  # brains made before 2026-11-16 key their cells by the whole unit ("dog?" and "dog" two cells)

    def _core(self, u):
        core = getattr(self.ear, "core", None)
        return core(u) if core is not None and self.use_core and u else u

    def cell(self, u):
        u = self._core(u)
        if u not in self.index:
            self._grow(len(self.units) + 1, 0)
            self.index[u] = len(self.units)
            self.born[len(self.units)] = self.moments
            self.units.append(u)
        return self.index[u]

    def _cue(self, u, grow):
        u = self._core(u)
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

    feedback_dims, fb = 0, None  # set by the brain when its cortex listens to the word cells

    _looks = None  # the views of this moment besides the whole one: what the eye fixated, each filling the view

    def _views(self, v, learn=False):
        """Looking (docs/VISION.md): with the eye's fixations, what the visual neurons pass on is one pattern per view
        (the whole view first). A word cell is driven by the view that drives it most, and learns from that view."""
        views = ([] if v is None else [self._visual(v, learn)]) + [self._visual(x) for x in (self._looks or [])]
        return np.stack(views) if views else None

    def _sight(self, v, cues, learn=False, snd=None):
        if self._looks is not None:
            return self._views(v, learn), None
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

    echo_relay = None  # the ear's echo decorrelated on its way to the context dendrites (as the cortex's summary is; 2026-11-17)

    def _echo(self, heard, said):
        v = self.ear.question(list(heard) + [SILENCE] + list(said), adapt=False)
        return self.echo_relay(v) if self.echo_relay is not None else v

    def _hearing(self, heard, n):
        """How far each cell's own unit is being heard: a cell fires when its unit is heard, and partly when a unit
        that sounds like it is ("dogs" for "dog"; the ear's sound of a unit, squared so that faint likeness is nothing)."""
        said = [u for u in {self._core(u) for u in heard} if u]
        if not said or n < 2:
            return None
        cs = getattr(self, "_cs", None)
        if cs is None or self._cs_units is not self.units or len(cs) < n:
            cs = np.zeros((max(n, 2 * (0 if cs is None else len(cs))), self.ear.width), np.float32)
            for i, u in enumerate(self.units[1:n], 1):
                cs[i] = self.ear.sound(u)
            self._cs, self._cs_units, self._cs_n = cs, self.units, n
        elif self._cs_n < n:
            for i in range(self._cs_n, n):
                cs[i] = self.ear.sound(self.units[i])
            self._cs_n = n
        return np.maximum(cs[:n] @ np.stack([self.ear.sound(u) for u in said]).T, 0).max(axis=1) ** 2

    def _matched(self, d, heard, learn):
        """The match neurons' firing. In every view, the cells of the words being heard burst as far as that view
        drives them too (heard and seen at once, in the same cell); the bursts of all cells are pooled per view.
        The neurons get the strongest view and the sum over the views (the eye's fixations when it made any), each
        relative to its own running average and spread."""
        h = self._hearing(heard, d.shape[0])
        if h is None:
            pooled = np.zeros(2, np.float32)
        else:
            looks = np.maximum(d[:, 1:] if d.shape[1] > 1 else d, 0)
            if self.match_bursts:
                # all or none: in a view a heard word's cell bursts when the view drives it at least half as much as
                # what is seen usually drives it when its word is heard (its own level, v_on). Then the sum over the
                # views is a count of the views that hold the thing, whatever its drive in each.
                looks = (looks >= 0.5 * np.maximum(self.v_on[: d.shape[0], None], 1e-6)).astype(np.float32) * (self.v_on[: d.shape[0], None] > 0)
            per_view = h @ looks
            pooled = np.array([per_view.max(), per_view.sum()], np.float32)
        if learn:
            self.m_seen += 1
            rate = 1.0 / min(self.m_seen, 1000)
            self.m_mean += rate * (pooled - self.m_mean)
            self.m_var += rate * ((pooled - self.m_mean) ** 2 - self.m_var)
        z = np.clip((pooled - self.m_mean) / (np.sqrt(self.m_var) + 1e-6), -3, 3)
        # each neuron's firing saturates, and the population is scaled so that it is an input of at most unit size like
        # the others (unscaled, the first corrections overshot and the weights ran away: a smoke test on 2026-11-17)
        return np.tanh(np.maximum(self.m_mix @ z - self.m_thr, 0)) / np.sqrt(self.match_cells)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("_cs", "_cs_units", "_cs_n")}  # a cache

    _m = None

    def _content(self, x, cues, b=None, snd=None, heard=None, learn=False):
        n = len(self.units)
        o = np.zeros(n, np.float32)
        self._m = None
        if snd is not None:
            o += self.Ws[:n] @ snd
        if x is not None and x.ndim == 2:  # several views: each cell takes the one that drives it most
            d = self.Wv[:n] @ x.T
            self._best = d.argmax(axis=1)
            o += d.max(axis=1)
            self._seen_drive = d.max(axis=1)
            if self.match and heard is not None:
                self._m = self._matched(d, heard, learn)
                o += self.Wm[:n] @ self._m
        elif x is not None and self.branches > 1:  # the branch that this look drives most
            d = np.concatenate([(self.Wv[:n] @ x)[None], self.Wx[:, :n] @ x])
            self._branch = d.argmax(axis=0)
            o += d.max(axis=0)
        elif x is not None:
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
        drive = np.maximum(self._content(x, cues, b, snd, heard), 0)
        if self.episodes is not None:  # moments like this one, stored whole: their replies' units are driven too
            seen_now = None if x is None else x[0] if x.ndim == 2 else x
            for u, d in self.episodes.read(seen_now, self._echo(heard, ()) if len(heard) else None).items():
                i = self.index.get(u)
                if i:
                    drive[i] += d
        ctx = np.maximum(self.Wa[:n] @ self._echo(heard, said), 0)
        if self.lateral:
            ctx = self._spread(ctx)
        top = drive[1:].max() if n > 1 else 0.0
        rel = drive / top if top > 0 else np.zeros(n, np.float32)
        rel[0] = 1.0  # the silence cell has only a context dendrite: silence needs no content
        act = rel * ctx
        for u in said:  # a cell that has just fired is refractory while the reply lasts
            act[self.index[self._core(u)]] = 0.0
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

    def look(self, looks):
        """Set what the eye fixated in this moment (a list of view codes, or None): used by the next expect/recall/learn."""
        self._looks = list(looks) if looks is not None else None
        return self

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
                # nothing to say: the cell that would speak is driven less than halfway between its own two levels
                if self.setpoint and k and drive[k] < 0.5 * (self.m_on[k] + self.m_off[k]):
                    return None, 0.0
            if k == 0:
                break
            said.append(self.units[k])
        pause = getattr(self.ear, "pause", b"") if self.use_core and hasattr(self.ear, "core") else b""  # between two words a speaker leaves a pause
        text = pause.join(said).decode("utf-8", "replace").strip()
        return (text or None), conf

    # ------------------------------------------------------------------ learning

    def surprise(self, heard, v, reply_units):
        """How far the content dendrites were from expecting the units that were then heard (0: fully expected, 1: not at all)."""
        cues = [c for c in (self._cue(u, False) for u in set(heard)) if c is not None]
        snd = self._heard_sound(heard)
        x, b = self._sight(v, cues, snd=snd)
        o = self._content(x, cues, b, snd, heard) + self.base[: len(self.units)]
        got = [float(np.clip(o[self.index[self._core(u)]], 0, 1)) if self._core(u) in self.index else 0.0 for u in reply_units]
        return 1.0 - float(np.mean(got)) if got else 1.0

    def learn(self, heard, v, reply_units):
        """An utterance (reply_units) was heard after `heard` (a question, or nothing: an unasked comment), while
        seeing v (or nothing). Every cell's dendrites compare what they expected with whether the cell fired."""
        self.moments += 1
        self.fb = None
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
        o = self._content(x, cues, b, snd, heard, learn=True)
        m = self._m
        if self.episodes is not None and targets:  # the moment is written whole, as far as the slow synapses did not expect its reply
            unexpected = 1.0 - float(np.mean(np.clip(o[targets] + self.base[targets], 0, 1)))
            seen_now = None if x is None else x[0] if x.ndim == 2 else x
            self.episodes.write(seen_now, self._echo(heard, ()) if len(heard) else None, [self.units[t] for t in targets], unexpected)
        if getattr(self, "m_on", None) is not None and n > 1:  # each cell's own levels (measured, before this moment changes it)
            d = np.maximum(o, 0)
            if targets:
                self.m_on[targets] += self.memory * (d[targets] - self.m_on[targets])
            d[targets] = -1.0
            d[0] = -1.0
            k = int(np.argmax(d))
            if d[k] >= 0:
                self.m_off[k] += self.memory * (d[k] - self.m_off[k])
        if self.match and targets and x is not None and x.ndim == 2 and getattr(self, "v_on", None) is not None:
            self.v_on[targets] += self.memory * (np.maximum(self._seen_drive[targets], 0) - self.v_on[targets])
        e = fired - (o + self.base[:n])
        e[0] = 0.0  # silence has no content dendrite
        step = (self.err_c[:n] if self.plastic_by_error else 1.0) * e / (1.0 + (x is not None) + (b is not None) + (snd is not None) + (m is not None) + len(cues))  # the inputs share the correction
        if self.gain != 1.0:
            step = step * self.gain
        if m is not None:
            self.Wm[:n] += step[:, None] * (m / (1.0 + self.use_m / self.maturation))[None, :]
            self.use_m += m * m
        if snd is not None:
            self.Ws[:n] += step[:, None] * (snd / (1.0 + self.use_s / self.maturation))[None, :]
            self.use_s += snd * snd
        self._own_error(self.err_c, fired, e)
        if self.feedback_dims and x is not None:
            # what the visual neurons hear back (the first feedback_dims of them: the cortex's first layer): each cell's
            # error returns along its own synapses, to the view that drove it (docs/VISION.md, the cortex taught by the words)
            W = self.Wv[:n, : self.feedback_dims]
            self.fb = [W[self._best == k].T @ e[self._best == k] for k in range(len(x))] if x.ndim == 2 else [W.T @ e]
        self.base[:n] += step / (1.0 + (self.moments - self.born[:n]) / self.maturation)  # the baseline: an input that is always on
        if x is not None and x.ndim == 2:
            # each cell learns from the view it took the word to be about: the one that drove it most; a cell that fired
            # with no view driving it yet (a new word) takes all the views together (it could be any of them)
            chosen = x[self._best]
            undriven = (fired > 0) & ((self.Wv[:n] @ x.T).max(axis=1) <= 0)
            if undriven.any():
                chosen[undriven] = x.mean(axis=0)
            self.Wv[:n] += step[:, None] * chosen / (1.0 + self.use_v / self.maturation)[None, :]
            self.use_v += (x * x).mean(axis=0)
        elif x is not None and self.branches > 1:
            took = self._branch.copy()
            for t in targets:  # its word was heard: is this a look the cell knows?
                if t and o[t] < 0.5 * self.m_on[t]:
                    free = np.nonzero(self.branch_used[:, t] == 0)[0]
                    if len(free) and self.branch_used[:, t].any():
                        took[t] = free[0]
            xs = x / (1.0 + self.use_v / self.maturation)
            for k in range(self.branches):
                mine = np.nonzero(took == k)[0]
                if len(mine):
                    W = self.Wv if k == 0 else self.Wx[k - 1]
                    W[mine] += step[mine, None] * xs[None, :]
            self.branch_used[took[targets], targets] += 1
            self.use_v += x * x
        elif x is not None:
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
        for name in ("Wv", "Wc", "Wa", "base", "err_c", "err_a", "heard", "last", "born", "L") + (("m_on", "m_off") if getattr(self, "m_on", None) is not None else ()) + (("Wb",) if getattr(self, "Wb", None) is not None else ()) + (("Ws",) if getattr(self, "Ws", None) is not None else ()) + (("Wm",) if self.match else ()) + (("v_on",) if self.match and getattr(self, "v_on", None) is not None else ()):
            a = getattr(self, name)
            a[: len(keep)] = a[keep]
            a[len(keep) : n] = 1.0 if name.startswith("err") else 0
        self.L[:, : len(keep)] = self.L[:, keep]
        self.L[:, len(keep) : n] = 0
        if self.branches > 1:
            for a in (self.Wx, self.branch_used):
                a[:, : len(keep)] = a[:, keep]
                a[:, len(keep) : n] = 0
        self.units = [self.units[i] for i in keep]
        self.index = {u: i for i, u in enumerate(self.units)}
        self._cs = None

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
