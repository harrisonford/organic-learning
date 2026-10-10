"""A barebones brain on raw organs, with activity that unfolds in time.

Only generic rules, and round, untuned numbers:

Organs (fixed)
    retina.Retina with a V1 that developed once (data/v1.npy) and is frozen;
    ear.Ear, hearing raw bytes. The eye moves itself; nothing tells it
    where to look.

Areas
    One area per organ output stream: "v1" (the simple-cell map), "magno"
    (the transient map) and "gaze" (where the eye points). Each grows
    assemblies by novelty and lets the winners drift toward what they
    respond to. There are no color, shape or place areas; whether anything
    like them appears is up to experience.

Activity in time
    Each area keeps a leaky state over its assemblies: every tick the old
    state decays by half and the current response adds to it. A percept
    builds up over the ticks and fixations of a moment, and a decision reads
    the state, not one input.

Binding and priming
    Assemblies keep Hebbian traces of the word-forms heard about the scene
    (spreading to forms that share auditory neurons). Word-forms are
    homeostatic, so the current state of each area raises (never vetoes) the
    word-forms it expects more than usual.

Speech
    The affinity / neurogenesis cortex (cortex.Cortex), fed only by the echo
    of the question and the echo of what it has said so far. Word-forms that
    just fired are fatigued.

Thinking against time
    At each tick it silently plans an answer and speaks when its confidence
    passes the patience of the clusters doing the planning. Patience is tuned
    by a scalar reward after speaking.

Sprouting (optional, the brain decides)
    Every area watches its own novelty: the fraction of its inputs that gave
    birth to a new assembly. If, long after infancy, that stays high
    ("everything still looks new"), the area sprouts a child area fed by the
    recent history (leaky state) of its own activity, read through its
    assemblies' fixed axon patterns. Successive glimpses in a moment are
    usually of the same thing, so the child can come to group views
    (temporal contiguity). Children follow the same rule.

What is deliberately absent: match/mismatch neurons, an "objects not yet
named" signal, attention rules, word-category spread, per-attribute areas.
"""

import os

import numpy as np

from .area import SensoryArea
from .cortex import Cortex
from .ear import SILENCE, Ear
from .retina import V1, Retina

END = SILENCE
V1_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "v1.npy")


def norm_text(text):
    return " ".join(text.replace(",", " , ").replace("?", " ? ").split())


class RawBrain:
    def __init__(self, seed=0, vigilance=0.8, decay=0.5, ticks=9, sprouting=False):
        self.retina = Retina(seed=seed)
        v1 = V1(kinds=16, window=3, seed=seed)
        v1.W = np.load(V1_FILE).astype(np.float32)
        v1.frozen = True
        self.retina.v1 = v1
        self.ear = Ear(seed=seed + 7)
        dims = {"v1": self.retina.rings * self.retina.angles * v1.kinds, "magno": self.retina.rings * self.retina.angles, "gaze": 14}
        self.vigilance = vigilance
        self.seed = seed
        self.areas = {k: SensoryArea(k, d, vigilance=vigilance, seed=seed, axon_dim=256) for k, d in dims.items()}
        self.parent = {}  # sprouted area -> the area whose history feeds it
        self.sprouting = sprouting
        self.novelty = {k: [] for k in self.areas}  # 1 if an input gave birth, else 0
        self.events = []
        self.speech = Cortex(dim=2 * self.ear.width, neurons_per_cluster=8, theta_familiar=0.95, associate=False, maturation=5.0, seed=seed)
        self.decay = decay
        self.ticks = ticks
        self.vcap = 64
        self.baseline = np.zeros(self.vcap, np.float32)
        self.visual_moments = 0
        self.age = 0
        self.time_cost = 0.02
        self._overlap = None

    # -------------------------------------------------------------- lexicon

    def unit(self, u):
        i = self.speech.output_neuron(u.hex())
        if i >= self.vcap:
            old = self.vcap
            while self.vcap <= i:
                self.vcap *= 2
            b = np.zeros(self.vcap, np.float32)
            b[:old] = self.baseline
            self.baseline = b
        return i

    def known(self, u):
        return self.speech.out_index.get(u.hex())

    def name(self, i):
        return bytes.fromhex(self.speech.outputs[i])

    @property
    def V(self):
        return len(self.speech.outputs)

    def overlap(self):
        """Word-forms that share auditory neurons (cached)."""
        V = self.V
        if self._overlap is None or self._overlap.shape[0] != V:
            S = np.stack([self.ear.sound(self.name(i)) for i in range(V)])
            W = np.clip(S @ S.T, 0, 1) ** 2
            W[W < 0.15] = 0
            np.fill_diagonal(W, 1.0)
            self._overlap = W.astype(np.float32)
        return self._overlap

    # ----------------------------------------------------------- perceiving

    def _organ_signals(self, tick):
        out = {"gaze": tick.eye_position}
        if tick.magno.any():
            out["magno"] = tick.magno.ravel()
        if tick.v1 is not None and tick.v1.any():
            out["v1"] = tick.v1.ravel()
        return out

    def perceive(self, image, learn):
        """Let the eye look for a while; return each area's state after every tick."""
        states = {k: np.zeros(0, np.float32) for k in self.areas}
        history = []
        if image is None:
            return [dict(states) for _ in range(self.ticks)]
        for tick in self.retina.view(image, ticks=self.ticks):
            signals = self._organ_signals(tick)
            new = {}
            for k, area in self.areas.items():  # parents come before their children
                if k in self.parent:
                    src = new[self.parent[k]]
                    if src.sum() > 0:
                        signals[k] = src @ self.areas[self.parent[k]].axons[: len(src)]
                s = states[k]
                if len(s) < area.C:
                    s = np.concatenate([s, np.zeros(area.C - len(s), np.float32)])
                s = self.decay * s
                if k in signals:
                    before = area.C
                    idx, act = area.perceive(signals[k], learn)
                    if learn:
                        self.novelty[k].append(1.0 if area.C > before else 0.0)
                    if len(s) < area.C:
                        s = np.concatenate([s, np.zeros(area.C - len(s), np.float32)])
                    s[idx] += act
                new[k] = s
            states = new
            history.append({k: v.copy() for k, v in states.items()})
        return history

    def priming(self, state):
        """Facilitation of word-forms by what each area's state expects."""
        V = self.V
        g = np.ones(V, np.float32)
        k = 0.02
        for name, s in state.items():
            if not len(s) or s.sum() <= 0:
                continue
            area = self.areas[name]
            area._ensure(V=V)
            e = (s[:, None] * area.Wname[: len(s), :V]).sum(axis=0) / s.sum()
            lift = (e + k) / (self.baseline[:V] + k)
            lift = (self.overlap() * lift[None, :]).max(axis=1)
            g *= np.maximum(lift, 1.0)
        return g

    # ---------------------------------------------------------------- speech

    def speak(self, q_units, state, target=None, max_units=16):
        """Plan (or, with a target, imitate) an answer given the current state."""
        q_code = self.ear.question(q_units)
        group = np.arange(self.V)
        prime = self.priming(state)
        fatigue = np.zeros(self.V, np.float32)
        said, conf, involved, correct = [], 1.0, {}, 0
        steps = len(target) if target else max_units
        for t in range(steps):
            s = np.concatenate([q_code, self.ear.said(said)])
            s /= np.linalg.norm(s)
            gain = prime * (1.0 - 0.9 * fatigue)
            if target:
                w = target[t]
                r = self.speech.learn(s, self.known(w), group, gain)
                correct += r["correct"]
            else:
                r = self.speech.forward(s, group, gain)
                if r is None or r["probs"] is None:
                    break
                p = r["probs"]
                w = self.name(int(np.argmax(p)))
                conf = min(conf, float(p[self.known(w)]))
                for c, g in zip(r["active"], r["g"]):
                    involved[int(c)] = involved.get(int(c), 0.0) + float(g)
            if w == END:
                break
            fatigue *= 0.5
            fatigue[self.known(w)] = 1.0
            said.append(w)
        return b"".join(said).decode("utf-8", "replace"), conf, involved, correct

    # --------------------------------------------------------------- a moment

    def think(self, image, question, answer=None, reward=None, learn=True):
        """Look, listen, plan at every tick, speak when sure enough, then learn.

        Returns (answer, tick, reward received).
        """
        q_units = self.ear.units(question, learn)
        for u in q_units:
            self.unit(u)
        target = None
        if answer is not None:
            target = self.ear.units(answer, learn) + [END]
            for u in target:
                self.unit(u)
        history = self.perceive(image, learn)
        for t, state in enumerate(history):
            out, conf, involved, _ = self.speak(q_units, state)
            total = sum(involved.values())
            patience = sum(self.speech.patience[c] * g for c, g in involved.items()) / total if total else 1.0
            if conf >= patience or t == len(history) - 1:
                break
        got = None
        if answer is not None and learn:
            self.age += 1
            correct = norm_text(out) == norm_text(answer)
            got = reward(t, correct) if reward else None
            self.speak(q_units, history[t], target=target)  # imitate, from the state I decided in
            self._bind(history, target)
            if got is not None:
                self._tune_patience(involved, t, correct, got, len(history) - 1)
            if self.age % 2000 == 0:
                self.speech.sleep(min_age=2000, merge_above=0.995)
            if self.sprouting:
                self._consider_sprouting()
        return out, t, got

    def _bind(self, history, target):
        """Hebbian traces between everything seen during the moment and the answer."""
        if not any(len(s) for s in history[-1].values()):
            return
        V = self.V
        heard = [self.known(u) for u in target if u != END]
        if not heard:
            return
        h = self.overlap()[:, heard].max(axis=1)
        self.visual_moments += 1
        rate = max(0.003, 1.0 / self.visual_moments)
        self.baseline[:V] += rate * (h - self.baseline[:V])
        seen = {k: sum(np.pad(st[k], (0, len(history[-1][k]) - len(st[k]))) for st in history) for k in self.areas}
        for k, s in seen.items():
            if len(s) and s.sum() > 0:
                idx = np.nonzero(s > 0.05 * s.max())[0]
                self.areas[k].bind(idx, s[idx] / s[idx].max(), h)

    def _consider_sprouting(self, window=2000, still_new=0.1, infancy=1000, max_areas=8):
        """An area that keeps finding everything new grows a child on its history."""
        if self.age < infancy:
            return
        for k in list(self.areas):
            hist = self.novelty[k]
            if len(hist) < window:
                continue
            rate = float(np.mean(hist[-window:]))
            self.novelty[k] = hist[-window:]
            has_child = k in self.parent.values()
            if rate > still_new and not has_child and len(self.areas) < max_areas:
                name = f"{k}+"
                src = self.areas[k]
                self.areas[name] = SensoryArea(name, src.axon_dim, vigilance=self.vigilance, seed=self.seed + len(self.areas), axon_dim=256)
                self.parent[name] = k
                self.novelty[name] = []
                self.events.append((self.age, f"{k} sprouted {name} (novelty {rate:.2f})"))

    def _tune_patience(self, involved, t, correct, got, last, eta=0.05):
        total = sum(involved.values()) or 1.0
        if correct and got < 1.0 - self.time_cost * t - 1e-6:
            step = -eta
        elif not correct and t < last:
            step = +eta
        elif correct:
            step = -0.2 * eta
        else:
            return
        for c, g in involved.items():
            if c < self.speech.C:
                self.speech.patience[c] = float(np.clip(self.speech.patience[c] + step * g / total, 0.02, 0.99))

    def report(self):
        areas = " ".join(f"{k}={a.C}" for k, a in self.areas.items())
        if self.events:
            areas += " | " + "; ".join(f"{e} at {a}" for a, e in self.events)
        return f"assemblies: {areas} | speech clusters={self.speech.C} word-forms={self.V}"
